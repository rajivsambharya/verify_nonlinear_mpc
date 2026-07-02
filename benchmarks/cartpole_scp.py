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
    x_mins = cfg.x_mins
    x_maxes = cfg.x_maxes

    T_vals_list      = list(cfg.T_vals)
    n_scp_iters_list = list(cfg.n_scp_iters)
    trust_region     = cfg.trust_region_size
    r                = cfg.r
    dt               = cfg.dt
    mass             = cfg.mass
    length           = cfg.length

    x_min   = x_mins[0]
    x_max   = x_maxes[0]
    rho_max = cfg.rho_max

    n_T          = len(T_vals_list)
    n_iters_axis = len(n_scp_iters_list)
    times        = np.zeros((n_T, n_iters_axis))
    opt_vals     = np.zeros((n_T, n_iters_axis))
    kkt_rhos     = np.zeros(n_T)
    kkt_times    = np.zeros(n_T)

    for ti, T in enumerate(T_vals_list):
        # --- Phase 1 (KKT): upper bound on J*(x_0) ---
        print(f"\n=== Phase 1 KKT: T={T} ===")
        p1_kkt = CartpolePhase1(
            T=T, r=r, dt=dt, mass=mass, length=length, g=9.8,
            x_lo=x_min, x_hi=x_max, verbose=False, time_limit=cfg.time_limit,
        )
        p1_kkt.solve()
        V_0_max_kkt = p1_kkt.V0_max()
        print(f"  V_0_max (KKT) = {V_0_max_kkt}")

        # --- KKT bisection (true nonlinear MPC, J* Lyapunov) ---
        print(f"\n=== KKT verification: T={T} ===")
        rho_lo, rho_hi = 0.0, rho_max
        total_kkt_time = 0.0
        while rho_hi - rho_lo > cfg.tol:
            rho_mid = (rho_lo + rho_hi) / 2.0
            ver_kkt = CartpoleKKTVerify(
                K=1, T=T, r=r, dt=dt,
                mass=mass, length=length, g=9.8, rho=rho_mid,
                x_lo=x_min, x_hi=x_max, verbose=False,
                time_limit=cfg.time_limit, V_0_max=V_0_max_kkt,
            )
            status_kkt, t_kkt = ver_kkt.solve()
            total_kkt_time += t_kkt
            print(f"  KKT T={T}, rho={rho_mid:.6f}, status={status_kkt}")
            if status_kkt == GRB.OPTIMAL:
                sol_kkt = ver_kkt.solution_dict()
                x0_wc = sol_kkt["x"][0]
                x1_wc = sol_kkt["x"][1]
                u0_wc = sol_kkt["u"][0]
                print(f"    worst-case x0: theta={x0_wc[0]:.4f}, dtheta={x0_wc[1]:.4f}"
                      f"  u0={u0_wc[0]:.4f}"
                      f"  V_curr={sol_kkt['V_curr']:.4f}, V_next={sol_kkt['V_next']:.4f}")
                rho_lo = rho_mid
            else:
                rho_hi = rho_mid
        kkt_rhos[ti]  = rho_hi
        kkt_times[ti] = total_kkt_time
        print(f"  KKT certified rho={rho_hi:.6f}, time={total_kkt_time:.1f}s")

        # --- SCP bisection for each n_iters ---
        for ii, n_iters in enumerate(n_scp_iters_list):
            print(f"\n=== Phase 1 SCP: T={T}, n_iters={n_iters} ===")
            p1_scp = CartpoleSCPPolicyPhase1(
                T=T, r=r, dt=dt, mass=mass, length=length, g=9.8,
                x_lo=x_min, x_hi=x_max, verbose=False,
                time_limit=cfg.time_limit, n_iters=n_iters,
                trust_region=trust_region,
            )
            p1_scp.solve()
            V_0_max_scp = p1_scp.V0_max()
            print(f"  V_0_max (SCP n={n_iters}) = {V_0_max_scp}")

            rho_lo     = 0.0
            rho_hi     = rho_max
            best_sol   = None
            total_time = 0.0

            while rho_hi - rho_lo > cfg.tol:
                rho_mid = (rho_lo + rho_hi) / 2.0

                ver = CartpoleSCPPolicyVerify(
                    T=T, r=r, dt=dt,
                    mass=mass, length=length, g=9.8, rho=rho_mid,
                    x_lo=x_min, x_hi=x_max, verbose=False,
                    time_limit=cfg.time_limit,
                    n_iters=n_iters, trust_region=trust_region,
                    V_0_max=V_0_max_scp,
                )

                status, solve_time = ver.solve()
                total_time += solve_time

                print(f"  T={T}, n={n_iters}, rho={rho_mid:.6f}, status={status}")
                sol = ver.solution_dict()
                print(f"  Objective: {sol['obj']}")

                if status == GRB.OPTIMAL:
                    x0_wc = sol["x0"]
                    x1_wc = sol["x1"]
                    print(f"    worst-case x0: theta={x0_wc[0]:.4f}, dtheta={x0_wc[1]:.4f}"
                          f"  u0={sol['u0']:.4f}"
                          f"  V_curr={sol['V_curr']:.4f}, V_next={sol['V_next']:.4f}")
                    rho_lo = rho_mid
                else:
                    rho_hi = rho_mid
                    best_sol = sol

            if best_sol is not None:
                print(f"  Best verified rho: T={T}, n={n_iters}: {rho_hi:.6f}")
                opt_vals[ti, ii] = rho_hi
            else:
                print(f"  Could not verify any rho <= {rho_max} for T={T}, n={n_iters}")
                opt_vals[ti, ii] = np.inf
            times[ti, ii] = total_time

    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    fig_rate, ax_rate = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        color = colors[ti]
        ax_rate.plot(n_scp_iters_list, opt_vals[ti],
                     marker=markers[ti % len(markers)], linewidth=2, color=color,
                     label=f'SCP T={T_vals_list[ti]}')
        ax_rate.axhline(kkt_rhos[ti], color=color, linewidth=1.5, linestyle='--',
                        label=f'KKT T={T_vals_list[ti]}')
    ax_rate.set_xlabel('SCP iterations')
    ax_rate.set_ylabel('rate $\\rho$')
    ax_rate.legend()
    ax_rate.grid(True)
    fig_rate.tight_layout()
    fig_rate.savefig('rates_scp.pdf', bbox_inches='tight')
    plt.close(fig_rate)

    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        color = colors[ti]
        ax_time.plot(n_scp_iters_list, times[ti],
                     marker=markers[ti % len(markers)], linewidth=2, color=color,
                     label=f'SCP T={T_vals_list[ti]}')
        ax_time.axhline(kkt_times[ti], color=color, linewidth=1.5, linestyle='--',
                        label=f'KKT T={T_vals_list[ti]}')
    ax_time.set_xlabel('SCP iterations')
    ax_time.set_ylabel('total solve time (sec)')
    ax_time.set_yscale('log')
    ax_time.legend()
    ax_time.grid(True)
    fig_time.tight_layout()
    fig_time.savefig('times_scp.pdf', bbox_inches='tight')
    plt.close(fig_time)


# ---------------------------------------------------------------------------
# SCP iterations: forward pass + constrained LQR KKT (with trust region)
# ---------------------------------------------------------------------------

def _add_scp_iters(M, tag, x_0, n_iters, T, r, dt, mass, length, g, u_max,
                   trust_region=None):
    """
    Add n_iters of SCP at state x_0 (Gurobi vars).

    Each iteration:
      1. Forward pass: x_bar[t+1] = f(x_bar[t], u_bar[t])  (true nonlinear)
         u_bar = 0 for iter 0; u_bar = u_lin from previous iter otherwise.
      2. Linearize at (x_bar[t], u_bar[t]):
           A[t] = [[1, dt], [dt*g/L*cos_bar[t], 1]]   B = [[0], [b_u]]
           c[t] = [0, dt*g/L*(sin_bar[t] - cos_bar[t]*x_bar[t][0])]
      3. Solve constrained QP on linearized + offset system from x_0:
           min  sum_t ||x||^2 + r*u^2
           s.t. x[t+1] = A[t]x[t] + Bu[t] + c[t],   x[0] = x_0
                |u[t]| <= u_max                         (box — KKT: mu_up/mu_lo)
                |u[t] - u_bar[t]| <= trust_region       (KKT: tr_up/tr_lo)

    Bilinear products (cos_bar*x_lin, cos_bar*lam, cos_bar*x_bar, tr*u_bar)
    handled by NonConvex=2.

    Returns (u_lin, x_lin) from last iteration.
    """
    b_u   = dt / (mass * length**2)
    n_x, n_u = 2, 1

    u_lin_prev = None  # None → u_bar = 0 for first iter

    for it in range(n_iters):
        it_tag = f"{tag}_it{it}"

        # ------------------------------------------------------------------
        # 1. Forward pass
        # ------------------------------------------------------------------
        x_bar   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xbar_{it_tag}_{t}")
                   for t in range(T + 1)}
        sin_bar = {t: M.addVar(lb=-1, ub=1, name=f"sinbar_{it_tag}_{t}") for t in range(T)}
        cos_bar = {t: M.addVar(lb=-1, ub=1, name=f"cosbar_{it_tag}_{t}") for t in range(T)}

        for i in range(n_x):
            M.addConstr(x_bar[0][i] == x_0[i], name=f"fp_init_{it_tag}_{i}")

        for t in range(T):
            M.addGenConstrSin(x_bar[t][0], sin_bar[t], name=f"fp_sin_{it_tag}_{t}")
            M.addGenConstrCos(x_bar[t][0], cos_bar[t], name=f"fp_cos_{it_tag}_{t}")
            M.addConstr(x_bar[t+1][0] == x_bar[t][0] + dt * x_bar[t][1],
                        name=f"fp_dyn0_{it_tag}_{t}")
            if it == 0:
                M.addConstr(
                    x_bar[t+1][1] == x_bar[t][1] + dt*g/length * sin_bar[t],
                    name=f"fp_dyn1_{it_tag}_{t}")
            else:
                M.addConstr(
                    x_bar[t+1][1] == x_bar[t][1] + dt*g/length * sin_bar[t]
                    + b_u * u_lin_prev[t][0],
                    name=f"fp_dyn1_{it_tag}_{t}")

        # ------------------------------------------------------------------
        # 2/3. Constrained LQR via KKT on linearized + offset system
        # ------------------------------------------------------------------
        x_lin = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xlin_{it_tag}_{t}")
                 for t in range(T + 1)}
        u_lin = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"ulin_{it_tag}_{t}")
                 for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"laml_{it_tag}_{t}")
                 for t in range(T + 1)}
        mu_up = {t: M.addVar(lb=0.0, name=f"mu_up_{it_tag}_{t}") for t in range(T)}
        mu_lo = {t: M.addVar(lb=0.0, name=f"mu_lo_{it_tag}_{t}") for t in range(T)}

        if trust_region is not None:
            tr_up = {t: M.addVar(lb=0.0, name=f"tr_up_{it_tag}_{t}") for t in range(T)}
            tr_lo = {t: M.addVar(lb=0.0, name=f"tr_lo_{it_tag}_{t}") for t in range(T)}

        for i in range(n_x):
            M.addConstr(x_lin[0][i] == x_0[i], name=f"lqr_init_{it_tag}_{i}")

        for t in range(T):
            u_bar_t = 0.0 if it == 0 else u_lin_prev[t][0]

            # theta: linear, no affine offset
            M.addConstr(
                x_lin[t+1][0] == x_lin[t][0] + dt * x_lin[t][1],
                name=f"lqr_dyn0_{it_tag}_{t}")
            # theta_dot: linearized + affine offset c[t][1] = dt*g/L*(sin_bar - cos_bar*x_bar[0])
            M.addConstr(
                x_lin[t+1][1] == dt*g/length * cos_bar[t] * x_lin[t][0]
                + x_lin[t][1] + b_u * u_lin[t][0]
                + dt*g/length * sin_bar[t]
                - dt*g/length * cos_bar[t] * x_bar[t][0],
                name=f"lqr_dyn1_{it_tag}_{t}")

            # Trust region primal: |u[t] - u_bar[t]| <= trust_region
            if trust_region is not None:
                M.addConstr(u_lin[t][0] - u_bar_t <= trust_region,
                            name=f"tr_p_up_{it_tag}_{t}")
                M.addConstr(u_bar_t - u_lin[t][0] <= trust_region,
                            name=f"tr_p_lo_{it_tag}_{t}")

        # Terminal costate (Q = I)
        for i in range(n_x):
            M.addConstr(lam[T][i] == x_lin[T][i], name=f"lqr_term_{it_tag}_{i}")

        # Backward costate: A[t]^T = [[1, dt*g/L*cos_bar[t]], [dt, 1]]
        for t in range(T - 1, -1, -1):
            M.addConstr(
                lam[t][0] == x_lin[t][0] + lam[t+1][0]
                + dt*g/length * cos_bar[t] * lam[t+1][1],
                name=f"lqr_back0_{it_tag}_{t}")
            M.addConstr(
                lam[t][1] == x_lin[t][1] + dt * lam[t+1][0] + lam[t+1][1],
                name=f"lqr_back1_{it_tag}_{t}")

        # Stationarity: r*u + b_u*lam[t+1][1] + mu_up - mu_lo [+ tr_up - tr_lo] = 0
        for t in range(T):
            stat = r * u_lin[t][0] + b_u * lam[t+1][1] + mu_up[t] - mu_lo[t]
            if trust_region is not None:
                stat = stat + tr_up[t] - tr_lo[t]
            M.addConstr(stat == 0, name=f"lqr_stat_{it_tag}_{t}")

        # Box complementarity
        for t in range(T):
            M.addConstr(mu_up[t] * (u_lin[t][0] - u_max)  == 0, name=f"box_up_{it_tag}_{t}")
            M.addConstr(mu_lo[t] * (-u_max - u_lin[t][0]) == 0, name=f"box_lo_{it_tag}_{t}")

        # Trust region complementarity
        if trust_region is not None:
            for t in range(T):
                u_bar_t = 0.0 if it == 0 else u_lin_prev[t][0]
                M.addConstr(
                    tr_up[t] * (u_lin[t][0] - u_bar_t - trust_region) == 0,
                    name=f"tr_c_up_{it_tag}_{t}")
                M.addConstr(
                    tr_lo[t] * (-trust_region - u_lin[t][0] + u_bar_t) == 0,
                    name=f"tr_c_lo_{it_tag}_{t}")

        u_lin_prev = u_lin

    return u_lin_prev, x_lin  # last iteration


# ---------------------------------------------------------------------------
# KKT: true nonlinear MPC verification (unchanged from cartpole_ilqr.py)
# ---------------------------------------------------------------------------

class CartpoleSCPPolicyVerify:
    """
    Lyapunov verification for the SCP policy (K=1 step).

    Policy at x[0]: n_iters SCP, apply u[0] from last iteration.
    Lyapunov V(x) = cost of last SCP LQR solution from x.

    Checks feasibility of V(x[1]) - rho*V(x[0]) >= 0 with V(x[0]) >= eps.
    Infeasible => contraction rate rho certified.
    """

    def __init__(self, T=5, r=0.0, dt=0.01, mass=1, length=1, g=9.8,
                 rho=0.5, x_lo=-2.0, x_hi=2.0, verbose=False,
                 time_limit=None, n_iters=1, trust_region=None, V_0_max=None):
        n_x, n_u = 2, 1
        u_max = 1000

        M = gp.Model("cartpole_scp_verify")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # x[0] in X_0
        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")
        # x[1] free (no state constraints)
        x1 = M.addVars(n_x, lb=-GRB.INFINITY, name="x1")

        # SCP at x[0]: compute applied control + Lyapunov
        u0_scp, x0_lqr = _add_scp_iters(
            M, "scp0", x0, n_iters, T, r, dt, mass, length, g, u_max,
            trust_region=trust_region)

        u0 = M.addVars(n_u, lb=-u_max, ub=u_max, name="u0")
        M.addConstr(u0[0] == u0_scp[0][0], name="apply_u")

        # True nonlinear dynamics: x[1] = f(x[0], u0)
        sin0 = M.addVar(lb=-1, ub=1,      name="sin0")
        tdd0 = M.addVar(lb=-GRB.INFINITY, name="tdd0")
        M.addGenConstrSin(x0[0], sin0,                                           name="sin_dyn")
        M.addConstr(x1[0] == x0[0] + dt * x0[1],                                name="dyn_theta")
        M.addConstr(tdd0 == g/length * sin0 + u0[0] / (mass * length**2),       name="tdd_dyn")
        M.addConstr(x1[1] == x0[1] + dt * tdd0,                                 name="dyn_thetadot")

        # SCP at x[1]
        u1_scp, x1_lqr = _add_scp_iters(
            M, "scp1", x1, n_iters, T, r, dt, mass, length, g, u_max,
            trust_region=trust_region)

        # Lyapunov values = cost of last SCP LQR trajectory
        V_curr = (
            gp.quicksum(x0_lqr[t][i] * x0_lqr[t][i] for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(r * u0_scp[t][0] * u0_scp[t][0] for t in range(T))
        )
        V_next = (
            gp.quicksum(x1_lqr[t][i] * x1_lqr[t][i] for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(r * u1_scp[t][0] * u1_scp[t][0] for t in range(T))
        )

        if V_0_max is not None:
            M.addConstr(V_curr <= V_0_max, name="V0_bound")

        eps = 1 - rho
        M.addConstr(V_curr >= 1e-3,                        name="V_curr_pos")
        M.addConstr(V_next - V_curr + eps * V_curr >= 0,  name="stability")
        M.setObjective(0, GRB.MAXIMIZE)

        self.V_curr = V_curr
        self.V_next = V_next
        self.x0 = x0
        self.x1 = x1
        self.u0 = u0

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {"obj": None, "V_curr": None, "V_next": None,
                    "x0": None, "x1": None, "u0": None}
        return {
            "obj":    self.model.ObjVal,
            "V_curr": self.V_curr.getValue(),
            "V_next": self.V_next.getValue(),
            "x0":     [self.x0[i].X for i in range(2)],
            "x1":     [self.x1[i].X for i in range(2)],
            "u0":     self.u0[0].X,
        }


class CartpoleSCPPolicyPhase1:
    """
    Phase 1 for SCP: compute V_0_max = max_{x_0 in X_0} V_scp(x_0).
    V_scp(x_0) = cost of last SCP LQR solution from x_0 with n_iters iterations.
    """

    def __init__(self, T=5, r=0.0, dt=0.01, mass=1, length=1, g=9.8,
                 x_lo=-2.0, x_hi=2.0, verbose=False, time_limit=None,
                 n_iters=1, trust_region=None):
        n_x = 2
        u_max = 1000

        M = gp.Model("cartpole_scp_phase1")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")

        u_scp, x_lqr = _add_scp_iters(
            M, "scp", x0, n_iters, T, r, dt, mass, length, g, u_max,
            trust_region=trust_region)

        V_curr = (
            gp.quicksum(x_lqr[t][i] * x_lqr[t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(r * u_scp[t][0] * u_scp[t][0] for t in range(T))
        )
        M.setObjective(V_curr, GRB.MAXIMIZE)
        self.V_curr = V_curr

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def V0_max(self):
        if self.model.SolCount == 0:
            return None
        return self.model.ObjVal


# ---------------------------------------------------------------------------
# KKT: Phase 1 + verification (true nonlinear MPC — same as cartpole_ilqr.py)
# ---------------------------------------------------------------------------

class CartpolePhase1:
    """Phase 1: compute V_0_max = max_{x_0 in X_0} J*(x_0) under true MPC."""

    def __init__(self, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x_lo=0.0, x_hi=4.0, verbose=True, time_limit=None):
        n_x, n_u = 2, 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * r
        u_max = 1000

        M = gp.Model("cartpole_phase1")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        x0    = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")
        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"x_mpc_{t}") for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_mpc_{t}") for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lam_{t}") for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x0[i], name=f"init_{i}")

        sin_mpc, cos_mpc, tdd_mpc = {}, {}, {}
        for t in range(T):
            sin_mpc[t] = M.addVar(lb=-1, ub=1,      name=f"sin_mpc_{t}")
            cos_mpc[t] = M.addVar(lb=-1, ub=1,      name=f"cos_mpc_{t}")
            tdd_mpc[t] = M.addVar(lb=-GRB.INFINITY, name=f"tdd_mpc_{t}")
            M.addGenConstrSin(x_mpc[t][0], sin_mpc[t], name=f"sin_c_{t}")
            M.addGenConstrCos(x_mpc[t][0], cos_mpc[t], name=f"cos_c_{t}")
            M.addConstr(x_mpc[t+1][0] == x_mpc[t][0] + dt * x_mpc[t][1],      name=f"dyn0_{t}")
            M.addConstr(tdd_mpc[t] == g/length * sin_mpc[t]
                        + u_mpc[t][0] / (mass * length**2),                     name=f"tdd_{t}")
            M.addConstr(x_mpc[t+1][1] == x_mpc[t][1] + dt * tdd_mpc[t],       name=f"dyn1_{t}")

        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i] * x_mpc[T][i], name=f"term_{i}")

        for t in range(T - 1, -1, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0]*x_mpc[t][0] + lam[t+1][0]
                + dt*g/length * cos_mpc[t] * lam[t+1][1],
                name=f"costate0_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1]*x_mpc[t][1] + dt*lam[t+1][0] + lam[t+1][1],
                name=f"costate1_{t}")

        for t in range(T):
            M.addConstr(
                R[0, 0]*u_mpc[t][0] + dt/(mass*length**2)*lam[t+1][1] == 0,
                name=f"stat_{t}")

        V0 = (
            gp.quicksum(Q[i, i]*x_mpc[t][i]*x_mpc[t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R[j, j]*u_mpc[t][j]*u_mpc[t][j]
                          for t in range(T) for j in range(n_u))
        )
        M.setObjective(V0, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def V0_max(self):
        if self.model.SolCount == 0:
            return None
        return self.model.ObjVal


class CartpoleKKTVerify:
    """
    Lyapunov verification for true nonlinear MPC (K=1 step).
    V(x) = J*(x) = MPC value function.
    Checks V(x[1]) - rho*V(x[0]) >= 0 for x[0] in X_0.
    """

    def __init__(self, n=10, K=1, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 rho=0.2, x_lo=0.0, x_hi=4.0, verbose=False,
                 time_limit=None, V_0_max=None):
        self.K = K
        n_x, n_u = 2, 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * r
        u_max = 1000

        M = gp.Model("cartpole_kkt_verify")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        self.x = {}
        self.u = {}
        self.x_mpc     = {}
        self.u_mpc_var = {}
        self.lam_mpc   = {}
        self.sin_theta = {}
        self.tdd_theta = {}

        for k in range(K + 1):
            self.x[k] = M.addVars(n_x, lb=(x_lo if k == 0 else -GRB.INFINITY),
                                   ub=(x_hi if k == 0 else GRB.INFINITY), name=f"x_{k}")

        for k in range(K):
            self.u[k] = M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_{k}")

            # MPC at x[k]
            self.x_mpc[k]     = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xm_{k}_{t}")
                                  for t in range(T + 1)}
            self.u_mpc_var[k] = {t: M.addVars(n_u, lb=-u_max, ub=u_max,  name=f"um_{k}_{t}")
                                  for t in range(T)}
            self.lam_mpc[k]   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lm_{k}_{t}")
                                  for t in range(T + 1)}

            sin_m = {t: M.addVar(lb=-1, ub=1,      name=f"sinm_{k}_{t}") for t in range(T)}
            cos_m = {t: M.addVar(lb=-1, ub=1,      name=f"cosm_{k}_{t}") for t in range(T)}
            tdd_m = {t: M.addVar(lb=-GRB.INFINITY, name=f"tddm_{k}_{t}") for t in range(T)}

            for i in range(n_x):
                M.addConstr(self.x_mpc[k][0][i] == self.x[k][i], name=f"minit_{k}_{i}")

            for t in range(T):
                M.addGenConstrSin(self.x_mpc[k][t][0], sin_m[t], name=f"sinm_{k}_{t}")
                M.addGenConstrCos(self.x_mpc[k][t][0], cos_m[t], name=f"cosm_{k}_{t}")
                M.addConstr(self.x_mpc[k][t+1][0] == self.x_mpc[k][t][0] + dt*self.x_mpc[k][t][1],
                            name=f"mdyn0_{k}_{t}")
                M.addConstr(tdd_m[t] == g/length*sin_m[t] + self.u_mpc_var[k][t][0]/(mass*length**2),
                            name=f"mtdd_{k}_{t}")
                M.addConstr(self.x_mpc[k][t+1][1] == self.x_mpc[k][t][1] + dt*tdd_m[t],
                            name=f"mdyn1_{k}_{t}")

            for i in range(n_x):
                M.addConstr(self.lam_mpc[k][T][i] == Q[i, i]*self.x_mpc[k][T][i],
                            name=f"mterm_{k}_{i}")
            for t in range(T - 1, -1, -1):
                M.addConstr(
                    self.lam_mpc[k][t][0] == Q[0,0]*self.x_mpc[k][t][0]
                    + self.lam_mpc[k][t+1][0]
                    + dt*g/length * cos_m[t] * self.lam_mpc[k][t+1][1],
                    name=f"mcs0_{k}_{t}")
                M.addConstr(
                    self.lam_mpc[k][t][1] == Q[1,1]*self.x_mpc[k][t][1]
                    + dt*self.lam_mpc[k][t+1][0] + self.lam_mpc[k][t+1][1],
                    name=f"mcs1_{k}_{t}")
            for t in range(T):
                M.addConstr(
                    R[0,0]*self.u_mpc_var[k][t][0]
                    + dt/(mass*length**2)*self.lam_mpc[k][t+1][1] == 0,
                    name=f"mstat_{k}_{t}")

            M.addConstr(self.u[k][0] == self.u_mpc_var[k][0][0], name=f"link_{k}")

            # Closed-loop dynamics
            sin_k = M.addVar(lb=-1, ub=1,      name=f"sin_{k}")
            tdd_k = M.addVar(lb=-GRB.INFINITY, name=f"tdd_{k}")
            self.sin_theta[k] = sin_k
            self.tdd_theta[k] = tdd_k
            M.addGenConstrSin(self.x[k][0], sin_k, name=f"sin_{k}")
            M.addConstr(self.x[k+1][0] == self.x[k][0] + dt*self.x[k][1],         name=f"dyn0_{k}")
            M.addConstr(tdd_k == g/length*sin_k + self.u[k][0]/(mass*length**2),   name=f"tdd_{k}")
            M.addConstr(self.x[k+1][1] == self.x[k][1] + dt*tdd_k,                name=f"dyn1_{k}")

        # MPC at x[K] (for V_next)
        self.x_mpc[K]     = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xm_{K}_{t}")
                              for t in range(T + 1)}
        self.u_mpc_var[K] = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"um_{K}_{t}")
                              for t in range(T)}
        self.lam_mpc[K]   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lm_{K}_{t}")
                              for t in range(T + 1)}

        sin_K = {t: M.addVar(lb=-1, ub=1,      name=f"sinm_{K}_{t}") for t in range(T)}
        cos_K = {t: M.addVar(lb=-1, ub=1,      name=f"cosm_{K}_{t}") for t in range(T)}
        tdd_K = {t: M.addVar(lb=-GRB.INFINITY, name=f"tddm_{K}_{t}") for t in range(T)}

        for i in range(n_x):
            M.addConstr(self.x_mpc[K][0][i] == self.x[K][i], name=f"minit_{K}_{i}")
        for t in range(T):
            M.addGenConstrSin(self.x_mpc[K][t][0], sin_K[t], name=f"sinm_{K}_{t}")
            M.addGenConstrCos(self.x_mpc[K][t][0], cos_K[t], name=f"cosm_{K}_{t}")
            M.addConstr(self.x_mpc[K][t+1][0] == self.x_mpc[K][t][0] + dt*self.x_mpc[K][t][1],
                        name=f"mdyn0_{K}_{t}")
            M.addConstr(tdd_K[t] == g/length*sin_K[t] + self.u_mpc_var[K][t][0]/(mass*length**2),
                        name=f"mtdd_{K}_{t}")
            M.addConstr(self.x_mpc[K][t+1][1] == self.x_mpc[K][t][1] + dt*tdd_K[t],
                        name=f"mdyn1_{K}_{t}")
        for i in range(n_x):
            M.addConstr(self.lam_mpc[K][T][i] == Q[i,i]*self.x_mpc[K][T][i], name=f"mterm_{K}_{i}")
        for t in range(T - 1, -1, -1):
            M.addConstr(
                self.lam_mpc[K][t][0] == Q[0,0]*self.x_mpc[K][t][0]
                + self.lam_mpc[K][t+1][0]
                + dt*g/length * cos_K[t] * self.lam_mpc[K][t+1][1],
                name=f"mcs0_{K}_{t}")
            M.addConstr(
                self.lam_mpc[K][t][1] == Q[1,1]*self.x_mpc[K][t][1]
                + dt*self.lam_mpc[K][t+1][0] + self.lam_mpc[K][t+1][1],
                name=f"mcs1_{K}_{t}")
        for t in range(T):
            M.addConstr(
                R[0,0]*self.u_mpc_var[K][t][0]
                + dt/(mass*length**2)*self.lam_mpc[K][t+1][1] == 0,
                name=f"mstat_{K}_{t}")

        V_curr = (
            gp.quicksum(Q[i,i]*self.x_mpc[0][t][i]*self.x_mpc[0][t][i]
                        for t in range(T+1) for i in range(n_x))
            + gp.quicksum(R[0,0]*self.u_mpc_var[0][t][0]*self.u_mpc_var[0][t][0]
                          for t in range(T))
        )
        V_next = (
            gp.quicksum(Q[i,i]*self.x_mpc[K][t][i]*self.x_mpc[K][t][i]
                        for t in range(T+1) for i in range(n_x))
            + gp.quicksum(R[0,0]*self.u_mpc_var[K][t][0]*self.u_mpc_var[K][t][0]
                          for t in range(T))
        )

        if V_0_max is not None:
            M.addConstr(V_curr <= V_0_max, name="V0_bound")

        self.V_curr = V_curr
        self.V_next = V_next

        eps = 1 - rho
        M.addConstr(V_next - V_curr + eps*V_curr >= 1e-6, name="stability")
        M.setObjective(0, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {"obj": None, "V_curr": None, "V_next": None}
        return {
            "obj":    self.model.ObjVal,
            "V_curr": self.V_curr.getValue(),
            "V_next": self.V_next.getValue(),
            "x":      {k: [self.x[k][i].X for i in range(2)] for k in range(self.K + 1)},
            "u":      {k: [self.u[k][i].X for i in range(1)] for k in range(self.K)},
        }
