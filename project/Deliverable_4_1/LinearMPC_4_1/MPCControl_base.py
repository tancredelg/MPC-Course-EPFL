import cvxpy as cp
import numpy as np
from scipy.signal import cont2discrete
from mpt4py import Polyhedron
from control import dlqr


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem
    x_param: cp.Parameter
    x_var: cp.Variable
    u_var: cp.Variable

    """Constraints & Weights (to be defined in subclasses)"""
    Q: np.ndarray
    R: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray

    # Terminal components
    Qf: np.ndarray  # Terminal cost matrix aka the P matrix
    X_f: Polyhedron

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition (Extract subsystem)
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        # Discretize
        self.A, self.B = self._discretize(A_red, B_red, Ts)

        # Store trim values for this subsystem
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        # Define constraints and weights (implemented in subclasses)
        self.set_tuning_parameters()
        self.set_constraints()

        # Compute Terminal Components (LQR + Invariant Set)
        self._compute_terminal_components()

        self._setup_controller()

    def set_tuning_parameters(self):
        """Override in subclass"""
        raise NotImplementedError

    def set_constraints(self):
        """Override in subclass"""
        raise NotImplementedError

    def _compute_terminal_components(self):
        """Compute LQR Qf matrix and Terminal Invariant Set X_f"""

        # 1. LQR
        K, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)

        # 2. Terminal Set Calculation
        # The set must satisfy: x_min <= x <= x_max  AND  u_min <= -Kx <= u_max
        # System closed loop: x+ = (A - BK)x

        A_cl = self.A - self.B @ K

        # Define polytope matrices Hx <= h
        # State constraints
        H_x = np.vstack([np.eye(self.nx), -np.eye(self.nx)])
        h_x = np.hstack([self.x_max, -self.x_min])

        X_poly = Polyhedron.from_Hrep(H_x, h_x)

        # Input constraints mapped to state: u_min <= -Kx <= u_max
        # -Kx <= u_max  ->  -Kx <= u_max
        # -Kx >= u_min  ->   Kx <= -u_min
        H_u = np.vstack([-K, K])
        h_u = np.hstack([self.u_max, -self.u_min])

        U_poly = Polyhedron.from_Hrep(H_u, h_u)

        # KU_poly = Polyhedron.from_Hrep(U_poly.A @ K, U_poly.b)

        # Compute invariant set (O_inf)
        # We need a robust invariant set calculation. Use code from Exercise 4.
        O_inf = X_poly.intersect(U_poly)
        itr = 1
        converged = False
        while itr < 50:
            Oprev = O_inf
            F, f = O_inf.A, O_inf.b
            # Compute the pre-set
            O_inf = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            O_inf.minHrep(True)
            _ = O_inf.Vrep  # TODO: this is a tempary fix since the contains() method is not robust enough when both inner and outer polyhera only has H-rep.
            if O_inf == Oprev:
                converged = True
                break
            # print("Iteration {0}... not yet converged\n".format(itr))
            itr += 1

        if converged:
            print(f"Maximum invariant set successfully computed after {itr} iterations:")
            print(f"  Dimension: {O_inf.dim}")
        else:
            print("Warning: Maximum invariant set computation did not converge.")

        self.X_f = O_inf

    # hard constraints
    # def _setup_controller(self) -> None:
    #     # CVXPY Variables
    #     self.x_var = cp.Variable((self.nx, self.N + 1))
    #     self.u_var = cp.Variable((self.nu, self.N))
    #     self.x_param = cp.Parameter(self.nx)

    #     s_var = cp.Variable((self.nx, self.N + 1), nonneg=True)
    #     rho_slack = 1e4 # need to see how this behaves, might need to increase

    #     cost = 0
    #     constraints = []

    #     constraints.append(self.x_var[:, 0] == self.x_param)

    #     for k in range(self.N):
    #         # Cost
    #         cost += cp.quad_form(self.x_var[:, k], self.Q) + cp.quad_form(self.u_var[:, k], self.R)

    #         # Dynamics
    #         constraints.append(
    #             self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
    #         )

    #         # State Constraints - now becomes soft constraints
    #         constraints.append(self.x_var[:, k] <= self.x_max)
    #         constraints.append(self.x_var[:, k] >= self.x_min)

    #         # constraints.append(self.x_var[:, k] <= self.x_max + s_var[:, k])
    #         # constraints.append(self.x_var[:, k] >= self.x_min - s_var[:, k])

    #         # Input Constraints
    #         constraints.append(self.u_var[:, k] <= self.u_max)
    #         constraints.append(self.u_var[:, k] >= self.u_min)

    #     # Terminal Cost
    #     cost += cp.quad_form(self.x_var[:, self.N], self.Qf)

    #     # L1 for penalty of the soft constraints
    #     # cost += rho_slack * cp.sum(s_var[:, k])   # L1 penalty

    #     # Terminal Constraint (Invariant Set)
    #     # A_f * x_N <= b_f
    #     # Extract A and b from the mpt4py polyhedron
    #     A_f = self.X_f.A
    #     b_f = self.X_f.b
    #     constraints.append(A_f @ self.x_var[:, self.N] <= b_f)

    #     self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def _setup_controller(self) -> None:
        # CVXPY Variables
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x_param = cp.Parameter(self.nx)

        s_var = cp.Variable((self.nx, self.N + 1), nonneg=True)
        rho_slack = 1e9  # need to see how this behaves, might need to increase

        cost = 0
        constraints = []

        constraints.append(self.x_var[:, 0] == self.x_param)

        for k in range(self.N):
            # Cost
            cost += cp.quad_form(self.x_var[:, k], self.Q) + cp.quad_form(self.u_var[:, k], self.R)

            # L1 for penalty of the soft constraints
            cost += rho_slack * cp.sum(s_var[:, k])  # L1 penalty

            # Dynamics
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
            )

            # State Constraints - now becomes soft constraints
            constraints.append(self.x_var[:, k] <= self.x_max + s_var[:, k])
            constraints.append(self.x_var[:, k] >= self.x_min - s_var[:, k])

            # Input Constraints
            constraints.append(self.u_var[:, k] <= self.u_max)
            constraints.append(self.u_var[:, k] >= self.u_min)

        # Terminal Cost
        cost += cp.quad_form(self.x_var[:, self.N], self.Qf)

        # Terminal Constraint (Invariant Set)
        # A_f * x_N <= b_f
        # Extract A and b from the mpt4py polyhedron
        A_f = self.X_f.A
        b_f = self.X_f.b
        constraints.append(A_f @ self.x_var[:, self.N] <= b_f)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Calculate Delta x0
        if x_target is None:
            x_target = np.zeros(self.nx)  # Target is 0 deviation (steady state)

        dx0 = x0 - x_target

        self.x_param.value = dx0

        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            # Use CLARABEL or OSQP. ECOS sometimes struggles with feasibility.
        except Exception as e:
            print(f"Solver failed: {e}")

        if self.u_var.value is None:
            print(f"Optimization failed for {self.__class__.__name__}. Status: {self.ocp.status}")
            # Fallback (return zero delta input)
            u_opt = np.zeros(self.nu)
            x_traj = np.zeros((self.nx, self.N + 1))
            u_traj = np.zeros((self.nu, self.N))
        else:
            u_opt = self.u_var.value[:, 0]
            x_traj = self.x_var.value
            u_traj = self.u_var.value

        # The result is Delta u. We define the output as Delta u + u_target (usually us)
        # But wait, the standard usually returns the *actual* input to apply.
        # The template asks for u0, x_traj, u_traj.
        # Usually, MPC returns the computed u_opt (delta) + u_trim.

        # For x_traj and u_traj, we usually plot the absolute values.

        # If u_target is None, we assume regulation to Trim
        if u_target is None:
            u_ref = self.us
        else:
            u_ref = u_target

        # Return absolute values
        return u_opt + u_ref, x_traj + x_target.reshape(-1, 1), u_traj + u_ref.reshape(-1, 1)
