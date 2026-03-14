import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def set_tuning_parameters(self):
        # State penalty: [wy, alpha, vx]
        # High penalty on velocity (tracking), medium on angle (stability), low on rate
        self.Q = np.diag([1.0, 10.0, 20.0])

        # Input penalty: [d2]
        self.R = np.diag([5.0])

    def set_constraints(self):
        # States: [wy, alpha, vx]
        beta_limit = np.deg2rad(8)
        d_limit = np.deg2rad(15)
        v_limit = 6.0
        w_limit = 0.7 
        self.x_min = np.array([-w_limit, -beta_limit, -v_limit])
        self.x_max = np.array([w_limit, beta_limit, v_limit])

        # Input Limits: Servo angle
        d_limit = np.deg2rad(15) - 1e-3

        # Calculate Delta u constraints
        # u_model = u_abs - u_trim
        # For X-system, trim d2 is 0.
        self.u_min = np.array([-d_limit]) - self.us
        self.u_max = np.array([d_limit]) - self.us

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
