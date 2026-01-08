import numpy as np
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    def set_tuning_parameters(self):
        # State penalty: [wy, beta, vx, x]
        self.Q = np.diag([1.0, 15.0, 2.0, 10.0])

        # Input penalty: [d2]
        self.R = np.diag([5.0])

    def set_constraints(self):
        # Limits from PDF
        # States: [wy, beta, vx]
        # beta limit: 10 deg (~0.1745 rad)
        inf = 1e9
        beta_limit = np.deg2rad(10)
        ang_v_limit = 50.0
        lin_v_limit = 100.0

        # Delta States Constraints relative to trim (trim is 0 for these)
        self.x_min = np.array([-ang_v_limit, -beta_limit, -lin_v_limit, -inf])
        self.x_max = np.array([ang_v_limit, beta_limit, lin_v_limit, inf])

        # Input Limits: Servo angle
        # d2 limit: 15 deg (~0.26 rad)
        d_limit = np.deg2rad(15)

        # Calculate Delta u constraints
        # u_model = u_abs - u_trim
        # For X-system, trim d2 is 0.
        self.u_min = np.array([-d_limit]) - self.us
        self.u_max = np.array([d_limit]) - self.us

    def _compute_terminal_components(self):
        """
        Override for Part 6.2 Nominal Controllers.
        We ONLY need the Terminal Cost (Qf), NOT the Terminal Set (Xf).
        """
        # 1. Calculate LQR Terminal Cost P (stored as self.Qf)
        # We assume u = -Kx
        K, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)

        # 2. Skip Set Calculation
        # We don't need X_f because we won't enforce x_N in X_f
        self.X_f = None
