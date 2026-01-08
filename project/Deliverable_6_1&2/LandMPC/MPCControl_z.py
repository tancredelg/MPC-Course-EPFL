import cvxpy as cp
import numpy as np
from scipy.signal import cont2discrete
from mpt4py import Polyhedron
from control import dlqr
from .MPCControl_base import MPCControl_base
import matplotlib.pyplot as plt


class MPCControl_z(MPCControl_base):
    """Complete states indices"""

    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    def set_tuning_parameters(self):
        # State: [vz, z]
        self.Q = np.diag([0.0001, 16000.0])
        # Input: [Pavg]
        self.R = np.diag([1.0])

    def set_constraints(self):
        # State: [vz, z] - No hard limit, but safe to bound for solver stability
        inf = 1e9

        # Absolute state limits: [vz, z]
        x_abs_min = np.array([-40, 0.0])  # z >= 0
        x_abs_max = np.array([40, 100.0])

        self.x_min = x_abs_min - self.xs
        self.x_max = x_abs_max - self.xs

        # Input: Pavg [40, 80]
        # Trim is likely around 60.
        safety_margin = 0.2
        u_abs_min = np.array([40.0 + safety_margin])
        u_abs_max = np.array([80.0 - safety_margin])

        # Delta constraints
        self.u_min = u_abs_min - self.us
        self.u_max = u_abs_max - self.us

    def _compute_terminal_components(self):
        """Compute LQR Qf matrix and Terminal Invariant Set X_f"""

        # 1. LQR
        K, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K = -K

        # 2. Terminal Set Calculation
        # The set must satisfy: x_min <= x <= x_max  AND  u_min <= -Kx <= u_max
        # System closed loop: x+ = (A - BK)x
        self.A_cl = self.A + self.B @ self.K

        # Define polytope matrices Hx <= h
        # Noise constraints
        H_w = np.vstack([np.eye(self.nu), -np.eye(self.nu)])
        w_min = -15.0
        w_max = 5.0
        h_w = np.hstack([w_max * np.ones(self.nu), -w_min * np.ones(self.nu)])  # -> [5, 15]
        W_poly = Polyhedron.from_Hrep(H_w, h_w)
        W = self.B @ W_poly

        # Define polytope matrices Hx <= h
        # State constraints
        H_x = np.vstack([np.eye(self.nx), -np.eye(self.nx)])
        h_x = np.hstack([self.x_max, -self.x_min])

        self.X_poly = Polyhedron.from_Hrep(H_x, h_x)

        # Input constraints mapped to state: u_min <= u <= u_max
        H_u = np.vstack([np.eye(self.nu), -np.eye(self.nu)])
        h_u = np.hstack([self.u_max, -self.u_min])

        self.U_poly = Polyhedron.from_Hrep(H_u, h_u)

        ## done -- task 0
        # task 1
        self.E = self.min_robust_invariant_set(self.A_cl, W, 120)

        self.task1()
        # task 1  -- done

    def task1(self):
        # tightened state constraints
        X_tilde = self.X_poly - self.E

        # tightened input constraints
        # we provide two equivalent methods to compute the tightened input constraints. You can verify that they yield the same result.
        KE = self.E.affine_map(self.K)  # optionally: K @ E

        # option 1: direct Pontryagin difference
        U_tilde_1 = self.U_poly - KE

        # option 2: manual tightening the right-hand side via support function
        # tilde_U = { u | A u <= b - max_{e in E} A K e }
        U_tilde_b = self.U_poly.b.copy()
        for i in range(U_tilde_b.shape[0]):
            U_tilde_b[i] -= KE.support(self.U_poly.A[i, :])
        U_tilde_2 = Polyhedron.from_Hrep(A=self.U_poly.A, b=U_tilde_b)

        print("Are the two tightened sets equal?", U_tilde_1 == U_tilde_2)
        U_tilde = U_tilde_1  # or U_tilde_2

        # Compute the terminal set for nominal mpc
        X_and_KU = self.X_poly.intersect(Polyhedron.from_Hrep(self.U_poly.A @ self.K, self.U_poly.b))
        Xf = self.max_invariant_set(self.A_cl, X_and_KU)

        # Compute the terminal set for tube mpc
        X_tilde_and_KU_tilde = X_tilde.intersect(Polyhedron.from_Hrep(U_tilde.A @ self.K, U_tilde.b))
        Xf_tilde = self.max_invariant_set(self.A_cl, X_tilde_and_KU_tilde)

        self.X_tilde = X_tilde
        self.U_tilde = U_tilde
        self.Xf_tilde = Xf_tilde
        self.X = self.X_poly
        self.Xf = Xf

        print("finished _compute_terminal_components")

    # Compute minimal robust invariant set
    @staticmethod
    def min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 30) -> Polyhedron:
        nx = A_cl.shape[0]
        Omega = W
        itr = 1
        A_cl_ith_power = np.eye(nx)
        while itr <= max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
            Omega_next = Omega + A_cl_ith_power @ W  # Minkowski sum of Omega and A^i * W
            Omega_next.minHrep()  # optionally: Omega_next.minVrep()
            Omega_next.minVrep()

            if itr % 10 == 0:
                print(f"iter {itr}")

            if np.linalg.matrix_norm(A_cl_ith_power, ord=2) < 3e-1:
                print(
                    "Minimal robust invariant set computation converged after {0} iterations.".format(itr)
                )
                break

            if itr == max_iter - 1:
                print(
                    "Minimal robust invariant set computation did NOT converge after {0} iterations.".format(
                        itr
                    )
                )

            Omega = Omega_next
            itr += 1
        return Omega_next

    @staticmethod
    def max_invariant_set(A_cl, X: Polyhedron, max_iter=30) -> Polyhedron:
        """
        Compute invariant set for an autonomous linear time invariant system x^+ = A_cl x
        """
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            # Compute the pre-set
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            O.minHrep()
            if O == Oprev:
                converged = True
                break
            print("Iteration {0}... not yet converged".format(itr))
            itr += 1

        if converged:
            print("Maximum invariant set successfully computed after {0} iterations.\n".format(itr))
        return O

    ### Continue here Task 3
    def _setup_controller(self) -> None:
        # CVXPY Variables
        self.z_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x0_var = cp.Parameter(self.nx)

        cost = 0
        constraints = []

        # initial state x0 in z0 + E
        constraints.append(self.E.A @ (self.x0_var - self.z_var[:, 0]) <= self.E.b)

        for k in range(self.N):
            # Cost
            cost += cp.quad_form(self.z_var[:, k], self.Q) + cp.quad_form(self.u_var[:, k], self.R)

            # Dynamics
            constraints.append(
                self.z_var[:, k + 1] == self.A @ self.z_var[:, k] + self.B @ self.u_var[:, k]
            )

            # State Constraints
            constraints.append(self.X_tilde.A @ self.z_var[:, k] <= self.X_tilde.b)

            # Input Constraints
            constraints.append(self.U_tilde.A @ self.u_var[:, k] <= self.U_tilde.b)

        # Terminal Cost
        cost += cp.quad_form(self.z_var[:, self.N], self.Qf)

        # Terminal Constraint (Invariant Set)
        # Extract A and b from the mpt4py polyhedron
        A_f = self.Xf_tilde.A
        b_f = self.Xf_tilde.b
        constraints.append(A_f @ self.z_var[:, self.N] <= b_f)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1. Coordinate Transform (Absolute -> Delta)
        # x0 is the absolute state from the simulator [vz, z]
        if x_target is None:
            x_target = np.zeros(self.nx)

        # We solve for deviations from the TRIM point (xs)
        dx0_real = x0 - self.xs
        self.x0_var.value = dx0_real

        # 2. Solve Nominal MPC
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            print(f"Solver failed: {e}")

        # Extract Nominal Plan (The "Ghost" Rocket)
        if self.u_var.value is None:
            # Emergency Fallback
            u_nom_delta = np.zeros(self.nu)
            z_curr_delta = dx0_real
        else:
            # u_nom_delta: The optimal input for the perfect system
            u_nom_delta = self.u_var.value[:, 0]
            # z_curr_delta: Where the solver THINKS we are (optimized initial state)
            z_curr_delta = self.z_var.value[:, 0]

        # 3. Calculate Ancillary Feedback (The Tube Correction)
        # Compare Real State (dx0_real) vs Nominal State (z_curr_delta)
        dist_error = dx0_real - z_curr_delta

        # self.K comes from dlqr (usually u = -Kx).
        # But in Tube MPC we often write u = u_nom + K(x-z).
        # Since we stored self.K = -dlqr_K, use ADDITION.
        # u_fb = (-K_dlqr) * error
        u_feedback = self.K @ dist_error

        # 4. Combine
        u_total_delta = u_nom_delta + u_feedback

        # 5. Return Absolute Input
        return (
            u_total_delta + self.us,
            self.z_var.value + self.xs.reshape(-1, 1),
            self.u_var.value + self.us.reshape(-1, 1),
        )

        # # Calculate Delta x0
        # if x_target is None:
        #     x_target = np.zeros(self.nx)  # Target is 0 deviation (steady state)

        # dx0 = x0 - x_target
        # self.x0_var.value = dx0

        # try:
        #     self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        #     # Use CLARABEL or OSQP. ECOS sometimes struggles with feasibility.
        # except Exception as e:
        #     print(f"Solver failed: {e}")

        # if self.u_var.value is None:
        #     print(f"Optimization failed for {self.__class__.__name__}. Status: {self.ocp.status}")
        #     # Fallback (return zero delta input)
        #     u_nom = np.zeros(self.nu)
        #     z_curr = dx0  # Best guess
        # else:
        #     u_nom = self.u_var.value[:, 0]
        #     z_curr = self.z_var.value[:, 0]  # Extract optimal nominal state z0

        #     x_traj = self.z_var.value
        #     u_traj = self.u_var.value

        # # Add Ancillary Feedback
        # # u = u_nom + K * (x_real - z_nom)
        # # Note: All variables here are in Delta coordinates (deviations)

        # error = dx0 - z_curr
        # u_feedback = self.K @ error

        # u_delta_applied = u_nom + u_feedback

        # # If u_target is None, we assume regulation to Trim
        # if u_target is None:
        #     u_ref = self.us
        # else:
        #     u_ref = u_target

        # # Return total applied input (Trim + Delta_Nominal + Delta_Feedback)
        # return u_delta_applied + u_ref, x_traj + x_target.reshape(-1, 1), u_traj + u_ref.reshape(-1, 1)
