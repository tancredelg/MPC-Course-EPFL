import numpy as np
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])
    termininal_set = False

    def set_tuning_parameters(self):
        # State penalty: [wy, beta, vx, x]
        self.Q = np.diag([1.0, 100.0, 2.0, 50.0])
        # Input penalty: [d2]
        self.R = np.diag([10.0])

    def set_constraints(self):
        # Physical Limits
        beta_limit = np.deg2rad(10)  # Tilt limit
        d_limit = np.deg2rad(15)  # Servo limit
        v_limit = 3.0  # Max horizontal speed
        w_limit = 0.5  # Max rotation speed

        # State Constraints: [wy, beta, vx, x]
        pos_limit = 10
        self.x_min = np.array([-w_limit, -beta_limit, -v_limit, -pos_limit])
        self.x_max = np.array([w_limit, beta_limit, v_limit, pos_limit])

        # Input Limits (relative to trim)
        # Since us for d1/d2 is typically 0:
        self.u_min = np.array([-d_limit]) - self.us
        self.u_max = np.array([d_limit]) - self.us
