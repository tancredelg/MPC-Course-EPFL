import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def set_tuning_parameters(self):
        # State: [vz]
        self.Q = np.diag([60.0])
        # Input: [Pavg]
        self.R = np.diag([1.0])

    def set_constraints(self):
        # State: [vz] - No hard limit, but safe to bound for solver stability
        inf = 1e9
        self.x_min = np.array([-inf])
        self.x_max = np.array([inf])

        # Input: Pavg [40, 80]
        # Trim is likely around 60.
        safety_margin = 0.2
        u_abs_min = np.array([40.0 + safety_margin])
        u_abs_max = np.array([80.0 - safety_margin])

        # Delta constraints
        self.u_min = u_abs_min - self.us
        self.u_max = u_abs_max - self.us
