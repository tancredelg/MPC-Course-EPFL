import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def set_tuning_parameters(self):
        # State: [vz]
        self.Q = np.diag([20.0])
        # Input: [Pavg]
        self.R = np.diag([1.0])

    def set_constraints(self):
        # State: [vz] - No hard limit, but safe to bound for solver stability
        inf = 1e9
        self.x_min = np.array([-inf])
        self.x_max = np.array([inf])

        # Input: Pavg [40, 80]
        # Trim is likely around 60.
        u_abs_min = np.array([40.0])
        u_abs_max = np.array([80.0])

        # Delta constraints
        self.u_min = u_abs_min - self.us
        self.u_max = u_abs_max - self.us

    # def _setup_controller(self) -> None:
    #     #################################################
    #     # YOUR CODE HERE

    #     self.ocp = ...

    #     # YOUR CODE HERE
    #     #################################################

    # def get_u(
    #     self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     #################################################
    #     # YOUR CODE HERE

    #     u0 = ...
    #     x_traj = ...
    #     u_traj = ...

    #     # YOUR CODE HERE
    #     #################################################

    #     return u0, x_traj, u_traj
