import numpy as np
import casadi as ca
from control import dlqr
from typing import Tuple


class NmpcCtrl:
    """
    Nonlinear MPC controller using CasADi.
    """

    def __init__(self, rocket, H: float, xs: np.ndarray, us: np.ndarray):
        self.rocket = rocket
        self.Ts = rocket.Ts
        self.N = int(H / self.Ts)

        # 1. Linearize around TARGET (xs, us) to get Terminal Cost P
        # This acts as the "anchor" at the end of the horizon
        sys = rocket.linearize_sys(xs, us)
        A, B = sys.A, sys.B

        # Tuning: Q and R for the cost function
        # State: [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
        Q_diag = np.array(
            [
                10,
                10,
                10,  # Rates
                100,
                100,
                100,  # Angles (Keep upright!)
                10,
                10,
                10,  # Velocities
                50,
                50,
                500,  # Positions (High priority on Z)
            ]
        )
        R_diag = np.array([10, 10, 1, 1])  # Inputs: d1, d2, Pavg, Pdiff

        Q = np.diag(Q_diag)
        R = np.diag(R_diag)

        # Compute Terminal Cost P using LQR
        _, P, _ = dlqr(A, B, Q, R)

        # 2. Setup CasADi Optimization Problem
        self.opti = ca.Opti()

        # Decision Variables
        # X: States [12 x N+1], U: Inputs [4 x N]
        self.X = self.opti.variable(12, self.N + 1)
        self.U = self.opti.variable(4, self.N)

        # Parameters
        self.x_init = self.opti.parameter(12)  # Initial State (Current measurement)
        self.x_ref = self.opti.parameter(12)  # Target State
        self.u_ref = self.opti.parameter(4)  # Target Input (Trim)

        # Set parameter values that don't change often
        self.opti.set_value(self.x_ref, xs)
        self.opti.set_value(self.u_ref, us)

        # --- Objective Function ---
        obj = 0
        for k in range(self.N):
            # State error
            e_x = self.X[:, k] - self.x_ref
            # Input error (deviation from hover trim)
            e_u = self.U[:, k] - self.u_ref

            obj += ca.mtimes([e_x.T, Q, e_x]) + ca.mtimes([e_u.T, R, e_u])

        # Terminal Cost
        e_xN = self.X[:, self.N] - self.x_ref
        obj += ca.mtimes([e_xN.T, P, e_xN])

        self.opti.minimize(obj)

        # --- Constraints ---

        # 1. Initial Condition
        self.opti.subject_to(self.X[:, 0] == self.x_init)

        # 2. Dynamics (Runge-Kutta 4)
        for k in range(self.N):
            x_next = self.rk4_step(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # 3. State Constraints
        # z >= 0 (Avoid ground collision)
        self.opti.subject_to(self.X[11, :] >= 0)

        # Singularity Avoidance: |beta| < 80 deg (approx 1.4 rad)
        self.opti.subject_to(self.opti.bounded(-1.4, self.X[4, :], 1.4))

        # 4. Input Constraints
        # d1, d2: +/- 15 deg (0.26 rad)
        self.opti.subject_to(self.opti.bounded(-0.26, self.U[0, :], 0.26))
        self.opti.subject_to(self.opti.bounded(-0.26, self.U[1, :], 0.26))

        # Pavg: [40, 80]
        self.opti.subject_to(self.opti.bounded(40.0, self.U[2, :], 80.0))

        # Pdiff: [-20, 20]
        self.opti.subject_to(self.opti.bounded(-20.0, self.U[3, :], 20.0))

        # --- Solver Settings ---
        p_opts = {"expand": True}  # Expand graph for speed
        s_opts = {"max_iter": 100, "print_level": 0, "tol": 1e-3, "acceptable_tol": 1e-2}
        self.opti.solver("ipopt", p_opts, s_opts)

        # Storage for Warm Start
        self.u_prev = np.tile(us.reshape(-1, 1), (1, self.N))
        self.x_prev = np.tile(xs.reshape(-1, 1), (1, self.N + 1))

    def rk4_step(self, x, u):
        """
        Symbolic RK4 integrator.
        Calls rocket.f_symbolic(x, u) which returns (x_dot, y).
        """
        dt = self.Ts

        # k1
        k1, _ = self.rocket.f_symbolic(x, u)

        # k2
        k2, _ = self.rocket.f_symbolic(x + 0.5 * dt * k1, u)

        # k3
        k3, _ = self.rocket.f_symbolic(x + 0.5 * dt * k2, u)

        # k4
        k4, _ = self.rocket.f_symbolic(x + dt * k3, u)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1. Set Parameter (Current State)
        self.opti.set_value(self.x_init, x0)

        # 2. Warm Start (Important for NMPC speed/convergence)
        # Initialize variables with the solution from the previous step
        self.opti.set_initial(self.X, self.x_prev)
        self.opti.set_initial(self.U, self.u_prev)

        # 3. Solve
        try:
            sol = self.opti.solve()

            # Extract optimal values
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)

            # Store for next warm start (Shift Strategy)
            # Shift X: drop x0, append xN (or duplicate last)
            self.x_prev = np.hstack([x_opt[:, 1:], x_opt[:, -1:]])
            # Shift U: drop u0, append uN-1
            self.u_prev = np.hstack([u_opt[:, 1:], u_opt[:, -1:]])

            # Prepare outputs
            u0 = u_opt[:, 0]
            x_ol = x_opt
            u_ol = u_opt
            t_ol = t0 + np.arange(self.N + 1) * self.Ts

            return u0, x_ol, u_ol, t_ol

        except RuntimeError as e:
            print(f"NMPC Solver Failed at t={t0:.2f}")
            # Fallback: Return hover inputs or previous guess
            # This prevents the simulation from crashing immediately
            u_safe = self.opti.value(self.u_ref)
            return u_safe, self.x_prev, self.u_prev, t0 + np.arange(self.N + 1) * self.Ts
