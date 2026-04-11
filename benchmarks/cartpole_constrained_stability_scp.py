import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

cmap = plt.cm.Set1
colors = cmap.colors

FONT_SIZE = 33
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE
})
plt.rc("xtick", labelsize=FONT_SIZE, labelcolor="black")
plt.rc("ytick", labelsize=FONT_SIZE, labelcolor="black")


def run(cfg):
    K = cfg.K
    x_mins = cfg.x_mins
    x_maxes = cfg.x_maxes

    T_vals_list = list(cfg.T_vals)
    r = cfg.r
    dt = cfg.dt
    u_bound = cfg.u_bound

    x_min = x_mins[0]
    x_max = x_maxes[0]
    rho_max = cfg.rho_max
    mip_gap = getattr(cfg, 'mip_gap', 0.01)

    n_T = len(T_vals_list)
    times    = np.zeros((n_T, K))
    opt_vals = np.zeros((n_T, K))

    for ti, T in enumerate(T_vals_list):
        # --- Phase 1: upper bound on V_0 = J_scp*(x_0) for this horizon T ---
        phase1 = CartpoleSCPPhase1(
            T=T, r=r, dt=dt, mass=1, length=1, g=9.8,
            x_lo=x_min, x_hi=x_max, u_bound=u_bound,
            verbose=False, time_limit=cfg.time_limit, mip_gap=mip_gap,
        )
        phase1.solve()
        V_0_max = phase1.V0_max()
        print(f"T={T}: Phase 1 V_0_max = {V_0_max}")

        # --- Phase 2: bisection over rho using V_0_max from Phase 1 ---
        for k in range(K):
            rho_lo = 0.0
            rho_hi = rho_max
            best_rho = None
            best_sol = None
            total_time = 0.0

            while rho_hi - rho_lo > cfg.tol:
                rho_mid = (rho_lo + rho_hi) / 2.0

                ver = CartpoleSCPVerify(
                    K=k + 1, T=T, r=r, dt=dt,
                    mass=1, length=1, g=9.8, rho=rho_mid,
                    x_lo=x_min, x_hi=x_max, u_bound=u_bound,
                    verbose=False, time_limit=cfg.time_limit,
                    V_0_max=V_0_max, mip_gap=mip_gap,
                )

                status, solve_time = ver.solve()
                total_time += solve_time

                print(f"T={T}, k={k+1}, rho={rho_mid:.6f}, status: {status}")
                sol = ver.solution_dict()
                print("Objective:", sol["obj"])

                if status == GRB.TIME_LIMIT:
                    print(f"T={T}, k={k+1}, rho={rho_mid:.6f}: time limit, cannot certify")
                if status == GRB.OPTIMAL:
                    rho_lo = rho_mid
                else:
                    rho_hi = rho_mid
                    best_rho = rho_mid
                    best_sol = sol

            if best_sol is not None:
                print(f"Best verified rho for T={T}, k={k+1}: {best_rho:.6f}")
                opt_vals[ti, k] = rho_hi
            else:
                print(f"Could not verify any rho <= {rho_max} for T={T}, k={k+1}")
                opt_vals[ti, k] = np.inf
            times[ti, k] = total_time

    k_axis = np.arange(K) + 1
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    fig_rate, ax_rate = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        ax_rate.plot(k_axis, opt_vals[ti],
                     marker=markers[ti % len(markers)], linewidth=2, color=colors[ti])
    ax_rate.set_xlabel('iterations $k$')
    ax_rate.set_ylabel('rate $\\rho$')
    ax_rate.grid(True)
    fig_rate.tight_layout()
    fig_rate.savefig('rates_scp.pdf', bbox_inches='tight')
    plt.close(fig_rate)

    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        ax_time.plot(k_axis, times[ti],
                     marker=markers[ti % len(markers)], linewidth=2, color=colors[ti])
    ax_time.set_xlabel('iterations $k$')
    ax_time.set_ylabel('total solve time (sec)')
    ax_time.set_yscale('log')
    ax_time.grid(True)
    fig_time.tight_layout()
    fig_time.savefig('times_scp.pdf', bbox_inches='tight')
    plt.close(fig_time)


# ---------------------------------------------------------------------------
# Helper: add one SCP-linearized T-horizon QP and embed its full KKT
# ---------------------------------------------------------------------------

def _add_scp_mpc(M, tag, T, r_cost, dt, mass, length, g, u_bound, x_init):
    """
    Add one T-horizon QP linearized at x_init (a dict of Gurobi vars keyed by dim)
    and embed its full KKT (necessary AND sufficient for this convex QP).

    The dynamics are linearized around x_init at u=0:
        A_k = [[1, dt], [dt*g/L*cos_k, 1]]      cos_k = cos(x_init[0])
        B_k = [[0], [dt/(m*L^2)]]
        c_k = [0, dt*g/L*(sin_k - cos_k*x_init[0])]   (affine first-order offset)

    QP:
        min  sum_t [ x_t^T Q x_t + u_t^T R u_t ]
        s.t. x_{t+1} = A_k x_t + B_k u_t + c_k,   t = 0..T-1
             x_0     = x_init
             u_t     in [-u_bound, u_bound]

    Returns:
        x_m   : {t: vars(n_x)}   primal states  t=0..T
        u_m   : {t: vars(n_u)}   primal controls t=0..T-1
        lam   : {t: vars(n_x)}   costates       t=0..T
        mu_up : {t: var}         upper dual      t=0..T-1
        mu_lo : {t: var}         lower dual      t=0..T-1
        V_expr : gurobi LinExpr  value = sum x^TQx + u^TRu  (quadratic in vars)
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)
    R = np.eye(n_u) * r_cost
    B0_1 = dt / (mass * length ** 2)
    gdtL  = g * dt / length

    # sin/cos at linearization point x_init[0]
    sin_k = M.addVar(lb=-1.0, ub=1.0, name=f"sin_{tag}")
    cos_k = M.addVar(lb=-1.0, ub=1.0, name=f"cos_{tag}")
    M.addGenConstrSin(x_init[0], sin_k, name=f"sinc_{tag}")
    M.addGenConstrCos(x_init[0], cos_k, name=f"cosc_{tag}")

    x_m = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xm_{tag}_{t}")
           for t in range(T + 1)}
    u_m = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"um_{tag}_{t}")
           for t in range(T)}
    lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"lam_{tag}_{t}")
             for t in range(T + 1)}
    mu_up = {t: M.addVar(lb=0.0, name=f"mup_{tag}_{t}") for t in range(T)}
    mu_lo = {t: M.addVar(lb=0.0, name=f"mlo_{tag}_{t}") for t in range(T)}

    # Initial condition
    for i in range(n_x):
        M.addConstr(x_m[0][i] == x_init[i], name=f"ic_{tag}_{i}")

    # Linearized dynamics
    #   x_m[t+1][0] = x_m[t][0] + dt * x_m[t][1]
    #   x_m[t+1][1] = gdtL*cos_k*x_m[t][0] + x_m[t][1] + B0_1*u_m[t][0]
    #                + gdtL*sin_k - gdtL*cos_k*x_init[0]
    # Bilinear terms (cos_k * x_m, cos_k * x_init[0]) handled by NonConvex=2.
    for t in range(T):
        M.addConstr(
            x_m[t + 1][0] == x_m[t][0] + dt * x_m[t][1],
            name=f"dyn_{tag}_{t}_0")
        M.addConstr(
            x_m[t + 1][1] == gdtL * cos_k * x_m[t][0]
                              + x_m[t][1]
                              + B0_1 * u_m[t][0]
                              + gdtL * sin_k
                              - gdtL * cos_k * x_init[0],
            name=f"dyn_{tag}_{t}_1")

    # ---------------------------------------------------------------
    # KKT (necessary AND sufficient for the convex linearized QP)
    # ---------------------------------------------------------------
    # Terminal costate: lam[T] = Q x_m[T]
    for i in range(n_x):
        M.addConstr(lam[T][i] == Q[i, i] * x_m[T][i], name=f"tc_{tag}_{i}")

    # Backward costate: lam[t] = Q x_m[t] + A_k^T lam[t+1]
    #   A_k^T = [[1, gdtL*cos_k], [dt, 1]]
    for t in range(T - 1, -1, -1):
        M.addConstr(
            lam[t][0] == Q[0, 0] * x_m[t][0]
                         + lam[t + 1][0]
                         + gdtL * cos_k * lam[t + 1][1],
            name=f"cs0_{tag}_{t}")
        M.addConstr(
            lam[t][1] == Q[1, 1] * x_m[t][1]
                         + dt * lam[t + 1][0]
                         + lam[t + 1][1],
            name=f"cs1_{tag}_{t}")

    # Stationarity w.r.t. u_t: R u_t + B_k^T lam[t+1] + mu_up - mu_lo = 0
    for t in range(T):
        M.addConstr(
            R[0, 0] * u_m[t][0] + B0_1 * lam[t + 1][1] + mu_up[t] - mu_lo[t] == 0,
            name=f"stat_{tag}_{t}")

    # Complementarity: control bounds
    for t in range(T):
        M.addConstr(mu_up[t] * (u_m[t][0] - u_bound) == 0,  name=f"cup_{tag}_{t}")
        M.addConstr(mu_lo[t] * (-u_bound - u_m[t][0]) == 0, name=f"clo_{tag}_{t}")

    # Value function expression: V = sum_t x_t^T Q x_t + sum_t u_t^T R u_t
    V_expr = (
        gp.quicksum(Q[i, i] * x_m[t][i] * x_m[t][i]
                    for t in range(T + 1) for i in range(n_x))
        + gp.quicksum(R[0, 0] * u_m[t][0] * u_m[t][0]
                      for t in range(T))
    )

    return x_m, u_m, lam, mu_up, mu_lo, V_expr


# ---------------------------------------------------------------------------
# Phase 1: max V_scp(x_0) over x_0 ∈ X_0
# ---------------------------------------------------------------------------

class CartpoleSCPPhase1:
    """
    Compute V_0_max = max_{x_0 ∈ X_0} J_scp*(x_0).

    J_scp*(x_0) is the value of the SCP-linearized QP at x_0:
        min  sum_t [ x_t^T Q x_t + u_t^T R u_t ]
        s.t. x_{t+1} = A(x_0) x_t + B u_t + c(x_0)   (linearized at x_0)
             x_0     = x_0
             u_t     ∈ [-u_bound, u_bound]

    The KKT of this convex QP is embedded (necessary AND sufficient).
    Bilinear terms from the linearization at x_0 are handled with NonConvex=2.
    """

    def __init__(self, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x_lo=0.0, x_hi=4.0, u_bound=10.0, verbose=True,
                 time_limit=None, mip_gap=0.01):
        M = gp.Model("cartpole_scp_phase1")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        M.Params.MIPGap = mip_gap
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        n_x = 2
        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")

        _, _, _, _, _, V_expr = _add_scp_mpc(
            M, tag="p1", T=T, r_cost=r, dt=dt, mass=mass, length=length,
            g=g, u_bound=u_bound, x_init=x0,
        )

        self._V_expr = V_expr
        M.setObjective(V_expr, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def V0_max(self):
        if self.model.SolCount == 0:
            return None
        return self.model.ObjVal


# ---------------------------------------------------------------------------
# Verify: find x_0 ∈ X_0 s.t. V_scp(x_K) >= rho * V_scp(x_0)
# ---------------------------------------------------------------------------

class CartpoleSCPVerify:
    """
    Searches for x_0 ∈ X_0 such that V_scp(x_K) >= rho * V_scp(x_0),
    i.e., the SCP Lyapunov function does NOT decrease by factor rho in K steps.

    At each step k = 0..K-1:
      - Solve SCP QP at x[k] (linearized at x[k], KKT embedded).
      - Apply u[k] = u_scp*(x[k]) to the TRUE nonlinear dynamics → x[k+1].

    V_curr = J_scp*(x[0]),  V_next = J_scp*(x[K]).
    Constraint: V_next - rho * V_curr >= eps  (looking for a counterexample).

    If INFEASIBLE → no counterexample → rho-stability certified.
    """

    def __init__(self, K=3, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 rho=0.9, x_lo=0.0, x_hi=4.0, u_bound=10.0,
                 verbose=True, time_limit=None, V_0_max=None, mip_gap=0.01):
        self.K = K
        self.T = T

        n_x, n_u = 2, 1

        M = gp.Model("cartpole_scp_verify")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        M.Params.MIPGap = mip_gap
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # Closed-loop state trajectory: x[0] ∈ X_0, x[1..K] free
        self.x = {}
        self.x[0] = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x_0")
        for k in range(1, K + 1):
            self.x[k] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_{k}")

        # Applied controls
        self.u = {}

        # SCP QP results at each step (for solution inspection)
        self.x_mpc = {}
        self.u_mpc = {}
        self.lam_mpc = {}

        V_curr_expr = None
        V_next_expr = None

        for k in range(K + 1):
            x_m, u_m, lam, mu_up, mu_lo, V_expr = _add_scp_mpc(
                M, tag=f"k{k}", T=T, r_cost=r, dt=dt, mass=mass, length=length,
                g=g, u_bound=u_bound, x_init=self.x[k],
            )
            self.x_mpc[k] = x_m
            self.u_mpc[k] = u_m
            self.lam_mpc[k] = lam

            if k == 0:
                V_curr_expr = V_expr
                # Optional bound: V_curr <= V_0_max (helps solver)
                if V_0_max is not None:
                    M.addConstr(V_expr <= V_0_max, name="V_curr_bound")

            if k == K:
                V_next_expr = V_expr

            if k < K:
                # Link: applied control = first SCP control
                self.u[k] = u_m[0]
                M.addConstr(self.u[k][0] == u_m[0][0], name=f"ulink_{k}")

                # True nonlinear dynamics: x[k+1] = f(x[k], u[k])
                sin_k = M.addVar(lb=-1.0, ub=1.0, name=f"sin_cl_{k}")
                cos_k = M.addVar(lb=-1.0, ub=1.0, name=f"cos_cl_{k}")
                tdd_k = M.addVar(lb=-GRB.INFINITY, name=f"tdd_cl_{k}")
                M.addGenConstrSin(self.x[k][0], sin_k, name=f"sinc_cl_{k}")
                M.addGenConstrCos(self.x[k][0], cos_k, name=f"cosc_cl_{k}")
                M.addConstr(tdd_k == g / length * sin_k + self.u[k][0] / (mass * length ** 2),
                            name=f"tdd_cl_{k}")
                M.addConstr(self.x[k + 1][0] == self.x[k][0] + dt * self.x[k][1],
                            name=f"cl_theta_{k}")
                M.addConstr(self.x[k + 1][1] == self.x[k][1] + dt * tdd_k,
                            name=f"cl_thetadot_{k}")

        # Lyapunov counterexample: V_next >= rho * V_curr
        eps = 1e-6
        M.addConstr(V_next_expr - rho * V_curr_expr >= eps, name="lyap_decrease")

        self.V_curr_expr = V_curr_expr
        self.V_next_expr = V_next_expr
        M.setObjective(0, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {"obj": None}
        return {
            "obj": self.model.ObjVal,
            "V_curr": self.V_curr_expr.getValue(),
            "V_next": self.V_next_expr.getValue(),
            "x": {k: {i: self.x[k][i].X for i in range(2)} for k in range(self.K + 1)},
            "u": {k: self.u[k][0].X for k in range(self.K)},
            "x_mpc": {k: {t: {i: self.x_mpc[k][t][i].X for i in range(2)}
                          for t in range(self.T + 1)}
                      for k in range(self.K + 1)},
        }
