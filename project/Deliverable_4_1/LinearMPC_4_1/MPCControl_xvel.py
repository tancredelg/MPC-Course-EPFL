import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def set_tuning_parameters(self):
        # State penalty: [wy, beta, vx]
        # High penalty on velocity (tracking), medium on angle (stability), low on rate
        self.Q = np.diag([100.0, 30.0, 2.0])

        # Input penalty: [d2]
        self.R = np.diag([0.5])

    def set_constraints(self):
        # Limits from PDF
        # States: [wy, beta, vx]
        # beta limit: 10 deg (~0.1745 rad)
        inf = 1e9
        beta_limit = np.deg2rad(10)

        # Delta States Constraints relative to trim (trim is 0 for these)
        self.x_min = np.array([-inf, -beta_limit, -inf])
        self.x_max = np.array([inf, beta_limit, inf])

        # Input Limits: Servo angle
        # d2 limit: 15 deg (~0.26 rad)
        d_limit = np.deg2rad(15)

        # Calculate Delta u constraints
        # u_model = u_abs - u_trim
        # For X-system, trim d2 is 0.
        self.u_min = np.array([-d_limit]) - self.us
        self.u_max = np.array([d_limit]) - self.us
