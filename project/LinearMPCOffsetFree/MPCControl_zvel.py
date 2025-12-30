import numpy as np
import cvxpy as cp
from control import dlqr
from scipy.signal import place_poles
from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):

    x_ids = np.array([8])  # vz
    u_ids = np.array([2])  # Pavg

    # estimator memory
    x_hat = None
    d_hat = 0.0
    u_prev_delta = None  # previous delta input (u_abs - u_trim)

    # optional safety
    vz_ref_clip = 6.0  # clamp PI-provided vz_ref to avoid impossible SS requests

    def set_tuning_parameters(self):
        self.Q = np.diag([20.0])
        self.R = np.diag([1.0])

    def set_constraints(self):
        inf = 1e9
        self.x_min_abs = np.array([-inf])
        self.x_max_abs = np.array([ inf])

        # absolute Pavg bounds
        self.u_min_abs = np.array([40.0 + 1e-2])
        self.u_max_abs = np.array([80.0])

        # base class expects these, but we will use *delta* bounds in this controller
        self.x_min = self.x_min_abs.copy()
        self.x_max = self.x_max_abs.copy()
        self.u_min = self.u_min_abs - self.us
        self.u_max = self.u_max_abs - self.us

    def _compute_terminal_components(self):
        # terminal cost only, no terminal set
        K, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K_lqr = np.array(K)
        self.X_f = None

        # ----- INPUT-BIAS disturbance model: Bd = B -----
        A = float(self.A[0, 0])
        B = float(self.B[0, 0])

        # augmented observer for z = [x; d]:
        # [x^+] = [A  B][x] + [B] u_delta
        # [d^+]   [0  1][d]   [0]
        self.A_hat = np.array([[A, B],
                               [0.0, 1.0]])
        self.B_hat = np.array([[B],
                               [0.0]])

        C_hat = np.array([[1.0, 0.0]])  # y = x
        poles = np.array([0.8, 0.9])    # tune if needed
        L = place_poles(self.A_hat.T, C_hat.T, poles).gain_matrix.T  # (2,1)

        # IMPORTANT: match your sign convention: + L (y_hat - y)
        self.L = -L

    def _setup_controller(self):
        """
        Build:
          (1) SS QP: xs, us_delta (delta around trim self.us)
          (2) Tracking MPC QP: e, du (delta around SS), no terminal set
        """
        N = self.N

        # ---------------- (1) Steady-state target selector ----------------
        # parameters
        self.vz_ref_par = cp.Parameter(1, name="vz_ref")
        self.d_hat_par_ss = cp.Parameter(1, name="d_hat_ss")

        # vars
        xs = cp.Variable(1, name="xs")           # steady-state vz
        us_d = cp.Variable(1, name="us_delta")   # steady-state delta input (Pavg - trim)
        e_ref = cp.Variable(1, name="e_ref")     # slack on reference (prevents infeasible SS)

        # ss equation: xs = A xs + B (us_d + d_hat)
        # ref equation (with slack): xs = vz_ref + e_ref
        # input bounds on delta: us_d in [u_abs_min - u_trim, u_abs_max - u_trim]
        A = self.A
        B = self.B

        ss_cons = [
            xs == A @ xs + B @ (us_d + self.d_hat_par_ss),  # Bd = B baked here
            # xs == self.vz_ref_par + e_ref,
            xs == self.vz_ref_par,
            us_d >= (self.u_min_abs - self.us),
            us_d <= (self.u_max_abs - self.us),
        ]

        # objective: keep delta input small + heavily penalize ref slack
        rho_ref = 1e4
        # ss_obj = cp.sum_squares(us_d) + rho_ref * cp.sum_squares(e_ref)
        ss_obj = cp.sum_squares(us_d)

        self.xs_var, self.usd_var = xs, us_d
        self.ss_prob = cp.Problem(cp.Minimize(ss_obj), ss_cons)

        # ---------------- (2) Tracking MPC on deviations ----------------
        # deviation vars:
        # e_k = x_k - xs, du_k = u_delta_k - us_delta
        e = cp.Variable((1, N + 1), name="e")
        du = cp.Variable((1, N), name="du")

        # parameters
        self.e0_par = cp.Parameter(1, name="e0")
        self.usd_par = cp.Parameter(1, name="us_delta")  # needed for moving input bounds

        # delta-u bounds depend on current us_delta:
        # u_abs = u_trim + (us_delta + du)  in [u_min_abs, u_max_abs]
        # => du in [u_min_abs - u_trim - us_delta, u_max_abs - u_trim - us_delta]
        du_lb = self.u_min_abs - self.us - self.usd_par
        du_ub = self.u_max_abs - self.us - self.usd_par

        cons = [e[:, 0] == self.e0_par]
        obj = 0

        for k in range(N):
            cons += [
                e[:, k + 1] == self.A @ e[:, k] + self.B @ du[:, k],
                du[:, k] >= du_lb,
                du[:, k] <= du_ub,
                # (vz bounds are huge; you can omit e-bounds)
            ]
            obj += cp.quad_form(e[:, k], self.Q) + cp.quad_form(du[:, k], self.R)

        obj += cp.quad_form(e[:, N], self.Qf)

        self.e_var, self.du_var = e, du
        self.ocp = cp.Problem(cp.Minimize(obj), cons)
        print("_setup_controller for z_vel")

    def compute_steady_state(self, vz_ref: float):
        self.vz_ref_par.value = np.array([vz_ref], dtype=float)
        self.d_hat_par_ss.value = np.array([self.d_hat], dtype=float)

        self.ss_prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if self.ss_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"SS QP failed: {self.ss_prob.status}")

        xs = float(self.xs_var.value)
        us_delta = float(self.usd_var.value)
        return xs, us_delta
    
    def _clip_d_hat_to_feasible(self):
        # delta input bounds around trim
        umin_d = float((self.u_min_abs - self.us)[0])  # e.g. 40 - u_trim
        umax_d = float((self.u_max_abs - self.us)[0])  # e.g. 80 - u_trim

        # need us_delta = -d_hat to be feasible -> d_hat in [-umax_d, -umin_d]
        dmin = -umax_d
        dmax = -umin_d
        self.d_hat = float(np.clip(self.d_hat, dmin, dmax))


    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):
        # measurement
        vz_meas = float(np.array(x0).reshape((-1,))[0])
        vz_ref = 0.0 if x_target is None else float(np.array(x_target).reshape((-1,))[0])
        vz_ref = float(np.clip(vz_ref, -self.vz_ref_clip, self.vz_ref_clip))

        # init estimator
        if self.x_hat is None:
            self.x_hat = vz_meas
            self.d_hat = 0.0
            self.u_prev_delta = 0.0  # start with delta=0 => u_abs = trim

        # -------- augmented observer update (INPUT-BIAS model) --------
        z = np.array([self.x_hat, self.d_hat])  # [x_hat; d_hat]
        y_hat = z[0]                            # y = x
        # note: observer uses u_delta (NOT absolute u!)
        z_next = (self.A_hat @ z) + (self.B_hat.flatten() * self.u_prev_delta) + (self.L.flatten() * (y_hat - vz_meas))
        self.x_hat = float(z_next[0])
        self.d_hat = float(z_next[1])
        self._clip_d_hat_to_feasible()

        # -------- steady-state target --------
        xs, us_delta = self.compute_steady_state(vz_ref)

        # -------- MPC on deviations --------
        e0 = float(vz_meas - xs)
        self.e0_par.value = np.array([e0], dtype=float)
        self.usd_par.value = np.array([us_delta], dtype=float)

        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if self.ocp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MPC failed: {self.ocp.status}")

        du0 = float(self.du_var.value[0, 0])

        # compose delta input then absolute input
        u_delta = us_delta + du0
        u_abs = float(self.us[0] + u_delta)
        u_abs = float(np.clip(u_abs, self.u_min_abs[0], self.u_max_abs[0]))

        # update prev delta input for observer
        self.u_prev_delta = float(u_abs - self.us[0])

        # provide predicted absolute trajectories for plots (optional)
        e_traj = np.array(self.e_var.value)          # (1, N+1)
        du_traj = np.array(self.du_var.value)        # (1, N)
        x_traj = e_traj + xs                          # x = e + xs
        u_traj = du_traj + (self.us[0] + us_delta)    # u_abs = trim + us_delta + du

        return np.array([u_abs]), x_traj, u_traj
