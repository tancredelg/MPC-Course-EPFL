import numpy as np
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def set_tuning_parameters(self):
        """
        Define the Cost Matrices Q and R.
        These are used by the Base Class to calculate LQR (P) and setup the MPC cost.
        """
        # State vector for this subsystem is: [wz, gamma]
        # We want to stabilize gamma (angle) aggressively.
        # We put a smaller cost on wz (rate) to dampen oscillations.
        self.Q = np.diag([1.0, 20.0])

        # Input vector is: [Pdiff]
        # We penalize input usage. If this is too low, the controller might
        # oscillate (Bang-Bang). If too high, it reacts too slowly.
        self.R = np.diag([1.0])

    def set_constraints(self):
        """
        Define the constraints for States (x) and Inputs (u).
        """
        # Input Constraints
        limit_pdiff = 20.0  # Max differential thrust (N)
        self.u_min = np.array([-limit_pdiff]) - self.us
        self.u_max = np.array([limit_pdiff]) - self.us

        # State Constraints
        # States: [wz, gamma]
        inf = 1e9
        self.x_min = np.array([-inf, -inf])
        self.x_max = np.array([inf, inf])

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
