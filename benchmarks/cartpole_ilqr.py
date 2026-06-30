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

    T_vals_list = list(cfg.T_vals)
    n_ilqr_iters_list = list(cfg.n_ilqr_iters)
    r = cfg.r
    dt = cfg.dt

    x_min = x_mins[0]
    x_max = x_maxes[0]
    rho_max = cfg.rho_max

    n_T = len(T_vals_list)
    n_iters_axis = len(n_ilqr_iters_list)
    times    = np.zeros((n_T, n_iters_axis))
    opt_vals = np.zeros((n_T, n_iters_axis))
    kkt_rhos  = np.zeros(n_T)
    kkt_times = np.zeros(n_T)

    for ti, T in enumerate(T_vals_list):
        # --- KKT bisection (true nonlinear MPC policy, J* Lyapunov) ---
        print(f"\n=== KKT verification: T={T} ===")
        rho_lo, rho_hi = 0.0, rho_max
        best_kkt_rho = None
        total_kkt_time = 0.0
        while rho_hi - rho_lo > cfg.tol:
            rho_mid = (rho_lo + rho_hi) / 2.0
            ver_kkt = CartpoleILQRVerify(
                K=1, T=T, r=r, dt=dt,
                mass=1, length=1, g=9.8, rho=rho_mid,
                x_lo=x_min, x_hi=x_max, verbose=False,
                time_limit=cfg.time_limit,
            )
            status_kkt, t_kkt = ver_kkt.solve()
            total_kkt_time += t_kkt
            print(f"  KKT T={T}, rho={rho_mid:.6f}, status={status_kkt}")
            if status_kkt == GRB.OPTIMAL:
                rho_lo = rho_mid
            else:
                rho_hi = rho_mid
                best_kkt_rho = rho_mid
        kkt_rhos[ti]  = rho_hi
        kkt_times[ti] = total_kkt_time
        print(f"  KKT certified rho={rho_hi:.6f}, time={total_kkt_time:.1f}s")

        # --- Bisection over rho for each number of iLQR iterations ---
        for ii, n_iters in enumerate(n_ilqr_iters_list):
            rho_lo = 0.0
            rho_hi = rho_max
            best_rho = None
            best_sol = None
            total_time = 0.0

            while rho_hi - rho_lo > cfg.tol:
                rho_mid = (rho_lo + rho_hi) / 2.0

                ver = CartpoleILQRPolicyVerify(
                    T=T, r=r, dt=dt,
                    mass=1, length=1, g=9.8, rho=rho_mid,
                    x_lo=x_min, x_hi=x_max, verbose=False,
                    time_limit=cfg.time_limit,
                    n_iters=n_iters
                )

                status, solve_time = ver.solve()
                total_time += solve_time

                print(f"T={T}, n_iters={n_iters}, rho={rho_mid:.6f}, status:", status)
                sol = ver.solution_dict()
                print("Objective:", sol["obj"])

                if status == GRB.TIME_LIMIT:
                    print(f"T={T}, n_iters={n_iters}, rho={rho_mid:.6f}: time limit exceeded")
                if status == GRB.OPTIMAL:
                    rho_lo = rho_mid
                else:
                    rho_hi = rho_mid
                    best_rho = rho_mid
                    best_sol = sol

            if best_sol is not None:
                print(f"Best verified rho for T={T}, n_iters={n_iters}: {best_rho:.6f}")
                opt_vals[ti, ii] = rho_hi
            else:
                print(f"Could not verify any rho <= {rho_max} for T={T}, n_iters={n_iters}")
                opt_vals[ti, ii] = np.inf
            times[ti, ii] = total_time

    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    fig_rate, ax_rate = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        color = colors[ti]
        ax_rate.plot(n_ilqr_iters_list, opt_vals[ti],
                     marker=markers[ti % len(markers)], linewidth=2, color=color,
                     label=f'iLQR T={T_vals_list[ti]}')
        ax_rate.axhline(kkt_rhos[ti], color=color, linewidth=1.5, linestyle='--',
                        label=f'KKT T={T_vals_list[ti]}')
    ax_rate.set_xlabel('iLQR iterations')
    ax_rate.set_ylabel('rate $\\rho$')
    ax_rate.legend()
    ax_rate.grid(True)
    fig_rate.tight_layout()
    fig_rate.savefig('rates_ilqr.pdf', bbox_inches='tight')
    plt.close(fig_rate)

    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        color = colors[ti]
        ax_time.plot(n_ilqr_iters_list, times[ti],
                     marker=markers[ti % len(markers)], linewidth=2, color=color,
                     label=f'iLQR T={T_vals_list[ti]}')
        ax_time.axhline(kkt_times[ti], color=color, linewidth=1.5, linestyle='--',
                        label=f'KKT T={T_vals_list[ti]}')
    ax_time.set_xlabel('iLQR iterations')
    ax_time.set_ylabel('total solve time (sec)')
    ax_time.set_yscale('log')
    ax_time.legend()
    ax_time.grid(True)
    fig_time.tight_layout()
    fig_time.savefig('times_ilqr.pdf', bbox_inches='tight')
    plt.close(fig_time)


def _add_mpc_kkt(M, tag, x_init, T, r, dt, mass, length, g, u_max):
    """
    Add MPC KKT conditions to model M starting from x_init (Gurobi vars or dict).
    Returns (x_mpc, u_mpc, cost_expr) where cost_expr = J*(x_init).
    No state constraints; only KKT stationarity + dynamics.
    """
    n_x, n_u = 2, 1
    b_u = dt / (mass * length**2)  # B[1,0]: constant control-to-thetadot gain

    x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xm_{tag}_{t}") for t in range(T + 1)}
    u_mpc = {t: M.addVars(n_u, lb=-u_max, ub=u_max,  name=f"um_{tag}_{t}") for t in range(T)}
    lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lm_{tag}_{t}") for t in range(T + 1)}
    sin_m = {t: M.addVar(lb=-1, ub=1,            name=f"sin_m_{tag}_{t}") for t in range(T)}
    cos_m = {t: M.addVar(lb=-1, ub=1,            name=f"cos_m_{tag}_{t}") for t in range(T)}
    tdd_m = {t: M.addVar(lb=-GRB.INFINITY,       name=f"tdd_m_{tag}_{t}") for t in range(T)}

    for i in range(n_x):
        M.addConstr(x_mpc[0][i] == x_init[i], name=f"init_{tag}_{i}")

    for t in range(T):
        M.addGenConstrSin(x_mpc[t][0], sin_m[t], name=f"sin_{tag}_{t}")
        M.addGenConstrCos(x_mpc[t][0], cos_m[t], name=f"cos_{tag}_{t}")
        M.addConstr(x_mpc[t+1][0] == x_mpc[t][0] + dt * x_mpc[t][1],       name=f"dyn0_{tag}_{t}")
        M.addConstr(tdd_m[t] == g/length * sin_m[t] + u_mpc[t][0] / (mass * length**2),
                    name=f"tdd_{tag}_{t}")
        M.addConstr(x_mpc[t+1][1] == x_mpc[t][1] + dt * tdd_m[t],          name=f"dyn1_{tag}_{t}")

    for i in range(n_x):
        M.addConstr(lam[T][i] == x_mpc[T][i], name=f"term_{tag}_{i}")  # Q=I

    for t in range(T - 1, -1, -1):
        M.addConstr(
            lam[t][0] == x_mpc[t][0] + lam[t+1][0] + dt*g/length * cos_m[t] * lam[t+1][1],
            name=f"back0_{tag}_{t}")
        M.addConstr(
            lam[t][1] == x_mpc[t][1] + dt * lam[t+1][0] + lam[t+1][1],
            name=f"back1_{tag}_{t}")

    for t in range(T):
        M.addConstr(r * u_mpc[t][0] + b_u * lam[t+1][1] == 0, name=f"stat_{tag}_{t}")

    cost = (
        gp.quicksum(x_mpc[t][i] * x_mpc[t][i] for t in range(T + 1) for i in range(n_x))
        + gp.quicksum(r * u_mpc[t][0] * u_mpc[t][0] for t in range(T))
    )
    return x_mpc, u_mpc, cost


def _add_ilqr_iters(M, tag, x_0, n_iters, T, r, dt, mass, length, g, u_max):
    """
    Add n_iters of iLQR to model M.

    Each iteration:
      1. Forward pass: simulate nonlinear dynamics from x_0 with u_bar
         (u_bar = 0 for iter 0; u_bar = u_lin from previous iter otherwise)
      2. Linearize at forward-pass trajectory: A[t] uses cos(x_bar[t][0]), B is constant
      3. Solve time-varying LQR on linearized system from x_0 via its KKT conditions

    The bilinear terms cos_bar[t]*x_lin[t][0] and cos_bar[t]*lam[t+1][1] are handled
    by NonConvex=2.

    Returns (u_lin, x_lin): controls and states from the last LQR solve.
    """
    n_x, n_u = 2, 1
    b_u = dt / (mass * length**2)

    u_lin_prev = None  # None → u_bar = 0 for first iteration

    for it in range(n_iters):
        it_tag = f"{tag}_it{it}"

        # ------------------------------------------------------------------
        # 1. Forward pass: x_bar[t+1] = f(x_bar[t], u_bar[t])
        # ------------------------------------------------------------------
        x_bar = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xbar_{it_tag}_{t}")
                 for t in range(T + 1)}
        sin_bar = {t: M.addVar(lb=-1, ub=1,      name=f"sinbar_{it_tag}_{t}") for t in range(T)}
        cos_bar = {t: M.addVar(lb=-1, ub=1,      name=f"cosbar_{it_tag}_{t}") for t in range(T)}
        for i in range(n_x):
            M.addConstr(x_bar[0][i] == x_0[i], name=f"fp_init_{it_tag}_{i}")

        for t in range(T):
            M.addGenConstrSin(x_bar[t][0], sin_bar[t], name=f"fp_sin_{it_tag}_{t}")
            M.addGenConstrCos(x_bar[t][0], cos_bar[t], name=f"fp_cos_{it_tag}_{t}")
            M.addConstr(x_bar[t+1][0] == x_bar[t][0] + dt * x_bar[t][1],
                        name=f"fp_dyn0_{it_tag}_{t}")
            if it == 0:
                M.addConstr(
                    x_bar[t+1][1] == x_bar[t][1] + dt * g/length * sin_bar[t],
                    name=f"fp_dyn1_{it_tag}_{t}")
            else:
                M.addConstr(
                    x_bar[t+1][1] == x_bar[t][1] + dt * g/length * sin_bar[t]
                    + b_u * u_lin_prev[t][0],
                    name=f"fp_dyn1_{it_tag}_{t}")

        # ------------------------------------------------------------------
        # 3. LQR solve: KKT conditions on linearized system from x_0
        #    min  sum_t x^T Q x + u^T R u
        #    s.t. x[t+1] = A[t] x[t] + B u[t],  x[0] = x_0
        #    A[t] = [[1, dt], [dt*g/L*cos_bar[t], 1]],  B = [[0], [b_u]]
        # ------------------------------------------------------------------
        x_lin = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xlin_{it_tag}_{t}")
                 for t in range(T + 1)}
        u_lin = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"ulin_{it_tag}_{t}")
                 for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"laml_{it_tag}_{t}")
                 for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(x_lin[0][i] == x_0[i], name=f"lqr_init_{it_tag}_{i}")

        for t in range(T):
            M.addConstr(x_lin[t+1][0] == x_lin[t][0] + dt * x_lin[t][1],
                        name=f"lqr_dyn0_{it_tag}_{t}")
            # bilinear: cos_bar[t] * x_lin[t][0]
            M.addConstr(
                x_lin[t+1][1] == dt*g/length * cos_bar[t] * x_lin[t][0]
                + x_lin[t][1] + b_u * u_lin[t][0],
                name=f"lqr_dyn1_{it_tag}_{t}")

        for i in range(n_x):
            M.addConstr(lam[T][i] == x_lin[T][i], name=f"lqr_term_{it_tag}_{i}")  # Q=I

        for t in range(T - 1, -1, -1):
            # lam[t] = Q x_lin[t] + A[t]^T lam[t+1]
            # A[t]^T = [[1, dt*g/L*cos_bar[t]], [dt, 1]]
            # bilinear: cos_bar[t] * lam[t+1][1]
            M.addConstr(
                lam[t][0] == x_lin[t][0] + lam[t+1][0]
                + dt*g/length * cos_bar[t] * lam[t+1][1],
                name=f"lqr_back0_{it_tag}_{t}")
            M.addConstr(
                lam[t][1] == x_lin[t][1] + dt * lam[t+1][0] + lam[t+1][1],
                name=f"lqr_back1_{it_tag}_{t}")

        for t in range(T):
            # R u[t] + B^T lam[t+1] = 0  =>  r*u[t] + b_u*lam[t+1][1] = 0
            M.addConstr(r * u_lin[t][0] + b_u * lam[t+1][1] == 0,
                        name=f"lqr_stat_{it_tag}_{t}")

        u_lin_prev = u_lin
        x_lin_prev = x_lin

    return u_lin_prev, x_lin_prev  # controls and states from last LQR iteration


class CartpoleILQRPolicyVerify:
    """
    Verification problem for the iLQR policy (K=1).

    Policy at x[0]: run n_iters of iLQR (forward pass + time-varying LQR solve),
    apply u[0] = first control from last LQR solution.

    Lyapunov function V(x) = cost of the final iLQR LQR solution from x:
      V(x) = sum_{t=0}^T ||x_lin[t]||^2 + r * sum_{t=0}^{T-1} u_lin[t]^2

    Checks feasibility of V(x[1]) - rho*V(x[0]) >= eps > 0.
    Infeasible => contraction rate rho is certified.
    """

    def __init__(self, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 rho=0.2, x_lo=0.0, x_hi=4.0, verbose=True,
                 time_limit=None, n_iters=1):
        n_x, n_u = 2, 1
        u_max = 1000

        M = gp.Model("cartpole_ilqr_policy_verify")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # Initial state (x[0] in X_0)
        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")
        # Next state x[1] (free: no state constraints)
        x1 = M.addVars(n_x, lb=-GRB.INFINITY, name="x1")

        # iLQR at x[0]: n_iters iterations; u0_ilqr applied, x0_lqr used for V_curr
        u0_ilqr, x0_lqr = _add_ilqr_iters(M, "ilqr0", x0, n_iters, T, r, dt, mass, length, g, u_max)

        # Applied control = first step of last LQR solution at x[0]
        u0 = M.addVars(n_u, lb=-u_max, ub=u_max, name="u0")
        M.addConstr(u0[0] == u0_ilqr[0][0], name="apply_u")

        # True nonlinear dynamics: x[1] = f(x[0], u0)
        sin0 = M.addVar(lb=-1, ub=1,      name="sin0")
        tdd0 = M.addVar(lb=-GRB.INFINITY, name="tdd0")
        M.addGenConstrSin(x0[0], sin0, name="sin_dyn")
        M.addConstr(x1[0] == x0[0] + dt * x0[1],                          name="dyn_theta")
        M.addConstr(tdd0 == g/length * sin0 + u0[0] / (mass * length**2), name="tdd_dyn")
        M.addConstr(x1[1] == x0[1] + dt * tdd0,                           name="dyn_thetadot")

        # iLQR at x[1]: n_iters iterations; x1_lqr used for V_next
        u1_ilqr, x1_lqr = _add_ilqr_iters(M, "ilqr1", x1, n_iters, T, r, dt, mass, length, g, u_max)

        # Lyapunov values = cost of final LQR solution trajectory
        V_curr = (
            gp.quicksum(x0_lqr[t][i] * x0_lqr[t][i] for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(r * u0_ilqr[t][0] * u0_ilqr[t][0] for t in range(T))
        )
        V_next = (
            gp.quicksum(x1_lqr[t][i] * x1_lqr[t][i] for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(r * u1_ilqr[t][0] * u1_ilqr[t][0] for t in range(T))
        )

        # Feasibility = policy does NOT satisfy contraction with rate rho
        eps = 1 - rho
        # M.addConstr(V_next - V_curr + eps * V_curr >= 1e-6, name="stability")
        M.addConstr(V_curr >= 1e-3, name="stability")
        M.addConstr(V_next - V_curr + eps * V_curr >= 0, name="stability")
        M.setObjective(0, GRB.MAXIMIZE)

        self.V_curr = V_curr
        self.V_next = V_next
        self.x0 = x0
        self.x1 = x1

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {"obj": None}
        return {
            "obj":    self.model.ObjVal,
            "V_curr": self.V_curr.getValue(),
            "V_next": self.V_next.getValue(),
            "x0":     [self.x0[i].X for i in range(2)],
            "x1":     [self.x1[i].X for i in range(2)],
        }


class CartpolePhase1:
    """
    Phase 1: compute V_0_max = max_{x_0 in X_0, KKT} J*(x_0).

    J*(x_0) = sum_{t=0}^{T} x_t^T Q x_t + sum_{t=0}^{T-1} u_t^T R u_t
    is the MPC value function evaluated at x_0 (optimal cost under KKT
    conditions).  The resulting V_0_max is passed to Phase 2 as an upper
    bound on V_curr, restricting the worst-case search to initial conditions
    that are actually reachable under the MPC policy.
    """

    def __init__(self, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x_lo=0.0, x_hi=4.0, verbose=True, time_limit=None):
        n_x = 2
        n_u = 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * r
        u_max = 1000

        M = gp.Model("cartpole_phase1")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2   # bilinear costate term + quadratic objective
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # Initial state (free within X_0)
        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")

        # T-horizon MPC trajectory
        # x_mpc = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_mpc_{t}")
        #          for t in range(T + 1)}
        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_mpc_{t}")
                 for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_mpc_{t}")
                 for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lam_{t}")
                 for t in range(T + 1)}

        # MPC initial condition
        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x0[i], name=f"init_{i}")

        # MPC nonlinear dynamics
        sin_mpc = {}
        cos_mpc = {}
        tdd_mpc = {}
        for t in range(T):
            sin_mpc[t] = M.addVar(lb=-1, ub=1, name=f"sin_mpc_{t}")
            cos_mpc[t] = M.addVar(lb=-1, ub=1, name=f"cos_mpc_{t}")
            tdd_mpc[t] = M.addVar(lb=-GRB.INFINITY, name=f"tdd_mpc_{t}")
            M.addGenConstrSin(x_mpc[t][0], sin_mpc[t], name=f"sin_c_{t}")
            M.addGenConstrCos(x_mpc[t][0], cos_mpc[t], name=f"cos_c_{t}")
            M.addConstr(x_mpc[t+1][0] == x_mpc[t][0] + dt * x_mpc[t][1],
                        name=f"dyn_theta_{t}")
            M.addConstr(tdd_mpc[t] == g/length * sin_mpc[t] +
                        u_mpc[t][0] / (mass * length**2),
                        name=f"dyn_tdd_{t}")
            M.addConstr(x_mpc[t+1][1] == x_mpc[t][1] + dt * tdd_mpc[t],
                        name=f"dyn_thetadot_{t}")

        # KKT: terminal costate
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i] * x_mpc[T][i],
                        name=f"term_costate_{i}")

        # KKT: costate backward recursion (bilinear: cos_mpc[t] * lam[t+1][1])
        for t in range(T - 1, -1, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0] * x_mpc[t][0] + lam[t+1][0]
                + dt * g / length * cos_mpc[t] * lam[t+1][1],
                name=f"costate_0_{t}"
            )
            M.addConstr(
                lam[t][1] == Q[1, 1] * x_mpc[t][1]
                + dt * lam[t+1][0] + lam[t+1][1],
                name=f"costate_1_{t}"
            )

        # KKT: control stationarity
        for t in range(T):
            M.addConstr(
                R[0, 0] * u_mpc[t][0]
                + dt / (mass * length**2) * lam[t+1][1] == 0,
                name=f"ctrl_opt_{t}"
            )

        # Objective: maximize V_0 = J*(x_0)
        V0 = (
            gp.quicksum(Q[i, i] * x_mpc[t][i] * x_mpc[t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R[j, j] * u_mpc[t][j] * u_mpc[t][j]
                          for t in range(T) for j in range(n_u))
        )
        # M.addConstr(V0 <= 100)
        M.setObjective(V0, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        # import pdb
        # pdb.set_trace()
        return self.model.Status, self.model.Runtime

    def V0_max(self):
        if self.model.SolCount == 0:
            return None
        return self.model.ObjVal


class CartpoleILQRVerify:
    def __init__(self, n=10, K=3, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 rho=0.2, x_lo=0.0, x_hi=4.0, seed=None, verbose=True,
                 time_limit=None, V_0_max=None):
        self.n, self.K, self.rho = n, K, rho
        self.T = T
        self.verbose = bool(verbose)
        rng = np.random.default_rng(seed)

        n_x = 2
        n_u = 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * r
        u_max = 1000

        # Model
        M = gp.Model("cartpole_ilqr_verify")
        M.Params.OutputFlag = 1 if self.verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2   # bilinear costate term; quadratic V constraints
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # Store state/control trajectories
        self.x = {}      # states: x[k] for k=0..K
        self.u = {}      # controls: u[k] for k=0..K-1

        # Store MPC variables for each timestep
        self.x_mpc = {}      # MPC state variables (T+1 timesteps)
        self.u_mpc_var = {}  # MPC control variables (T timesteps)
        self.lambda_mpc = {} # MPC costate variables for optimality (T+1 timesteps)

        # Auxiliary variables for dynamics
        self.sin_theta = {}
        self.cos_theta = {}
        self.theta_ddot = {}

        # Auxiliary variables for MPC dynamics
        self.sin_theta_mpc = {}
        self.cos_theta_mpc = {}
        self.theta_ddot_mpc = {}

        # CREATE ALL STATE VARIABLES FIRST (k=0 to K)
        for k in range(K + 1):
            # self.x[k] = M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_{k}")
            if k == 0:
                self.x[k] = M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_{k}")
            else:
                self.x[k] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_{k}")

        # Create variables and constraints for K timesteps
        for k in range(K):
            # Control at timestep k (applied from state k)
            self.u[k] = M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_{k}")

            # MPC variables: solve T-horizon MPC at each timestep k
            # MPC states: x_mpc[k][t] for t=0..T
            self.x_mpc[k] = {}
            for t in range(T + 1):
                self.x_mpc[k][t] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_mpc_{k}_{t}")

            # MPC controls: u_mpc[k][t] for t=0..T-1
            self.u_mpc_var[k] = {}
            for t in range(T):
                self.u_mpc_var[k][t] = M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_mpc_{k}_{t}")

            # MPC initial condition: x_mpc[k][0] = x[k]
            for i in range(n_x):
                M.addConstr(self.x_mpc[k][0][i] == self.x[k][i], name=f"mpc_init_{k}_{i}")

            # Auxiliary variables for MPC nonlinear dynamics
            self.sin_theta_mpc[k] = {}
            self.cos_theta_mpc[k] = {}
            self.theta_ddot_mpc[k] = {}

            for t in range(T):
                self.sin_theta_mpc[k][t] = M.addVar(lb=-1, ub=1, name=f"sin_theta_mpc_{k}_{t}")
                self.cos_theta_mpc[k][t] = M.addVar(lb=-1, ub=1, name=f"cos_theta_mpc_{k}_{t}")
                self.theta_ddot_mpc[k][t] = M.addVar(lb=-GRB.INFINITY, name=f"theta_ddot_mpc_{k}_{t}")

                # Nonlinear dynamics constraints for MPC
                M.addGenConstrSin(self.x_mpc[k][t][0], self.sin_theta_mpc[k][t], name=f"sin_mpc_{k}_{t}")
                M.addGenConstrCos(self.x_mpc[k][t][0], self.cos_theta_mpc[k][t], name=f"cos_mpc_{k}_{t}")

                # theta_{t+1} = theta_t + dt * theta_dot_t
                M.addConstr(
                    self.x_mpc[k][t+1][0] == self.x_mpc[k][t][0] + dt * self.x_mpc[k][t][1],
                    name=f"mpc_theta_{k}_{t}"
                )

                # theta_ddot = (g/L) * sin(theta) + u / (m*L²)
                M.addConstr(
                    self.theta_ddot_mpc[k][t] == g/length * self.sin_theta_mpc[k][t] +
                    self.u_mpc_var[k][t][0] / (mass * length**2),
                    name=f"mpc_angular_accel_{k}_{t}"
                )

                # theta_dot_{t+1} = theta_dot_t + dt * theta_ddot
                M.addConstr(
                    self.x_mpc[k][t+1][1] == self.x_mpc[k][t][1] + dt * self.theta_ddot_mpc[k][t],
                    name=f"mpc_theta_dot_{k}_{t}"
                )

            # KKT optimality conditions for MPC
            # Add costate (dual) variables for dynamics constraints
            self.lambda_mpc[k] = {}
            for t in range(T + 1):
                self.lambda_mpc[k][t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"lambda_mpc_{k}_{t}")

            # Terminal costate condition: λ_T = Q x_T
            for i in range(n_x):
                M.addConstr(
                    self.lambda_mpc[k][T][i] == Q[i, i] * self.x_mpc[k][T][i],
                    name=f"terminal_costate_{k}_{i}"
                )

            # Backward sweep: costate equations
            # λ_t = Q x_t + (∂f/∂x_t)^T λ_{t+1}
            # For f = [x₁ + dt·x₂, x₂ + dt·(g/L·sin(x₁) + u/(m·L²))]
            # ∂f/∂x = [[1, dt], [dt·g/L·cos(x₁), 1]]
            for t in range(T - 1, -1, -1):
                # λ_t[0] = Q[0,0] x_t[0] + 1 · λ_{t+1}[0] + dt·g/L·cos(x_t[0]) · λ_{t+1}[1]
                M.addConstr(
                    self.lambda_mpc[k][t][0] == Q[0, 0] * self.x_mpc[k][t][0] +
                    self.lambda_mpc[k][t+1][0] +
                    dt * g / length * self.cos_theta_mpc[k][t] * self.lambda_mpc[k][t+1][1],
                    name=f"costate_0_{k}_{t}"
                )

                # λ_t[1] = Q[1,1] x_t[1] + dt · λ_{t+1}[0] + 1 · λ_{t+1}[1]
                M.addConstr(
                    self.lambda_mpc[k][t][1] == Q[1, 1] * self.x_mpc[k][t][1] +
                    dt * self.lambda_mpc[k][t+1][0] +
                    self.lambda_mpc[k][t+1][1],
                    name=f"costate_1_{k}_{t}"
                )

            # Control optimality: R u_t + (∂f/∂u_t)^T λ_{t+1} = 0
            # ∂f/∂u = [0, dt/(m·L²)]^T
            # So: R u_t + dt/(m·L²) · λ_{t+1}[1] = 0
            for t in range(T):
                M.addConstr(
                    R[0, 0] * self.u_mpc_var[k][t][0] +
                    dt / (mass * length**2) * self.lambda_mpc[k][t+1][1] == 0,
                    name=f"control_opt_{k}_{t}"
                )

            # Link the first control action to the applied control
            M.addConstr(self.u[k][0] == self.u_mpc_var[k][0][0], name=f"control_link_{k}")

            # Auxiliary variables for actual dynamics (state evolution)
            self.sin_theta[k] = M.addVar(lb=-1, ub=1, name=f"sin_theta_{k}")
            self.cos_theta[k] = M.addVar(lb=-1, ub=1, name=f"cos_theta_{k}")
            self.theta_ddot[k] = M.addVar(lb=-GRB.INFINITY, name=f"theta_ddot_{k}")

            # Nonlinear dynamics constraints for actual state evolution
            M.addGenConstrSin(self.x[k][0], self.sin_theta[k], name=f"sin_{k}")
            M.addGenConstrCos(self.x[k][0], self.cos_theta[k], name=f"cos_{k}")

            # theta_{k+1} = theta_k + dt * theta_dot_k
            M.addConstr(
                self.x[k+1][0] == self.x[k][0] + dt * self.x[k][1],
                name=f"integrate_theta_{k}"
            )

            # theta_ddot = (g/L) * sin(theta) + u / (m*L²)
            M.addConstr(
                self.theta_ddot[k] == g/length * self.sin_theta[k] + self.u[k][0] / (mass * length**2),
                name=f"angular_accel_{k}"
            )

            # theta_dot_{k+1} = theta_dot_k + dt * theta_ddot
            M.addConstr(
                self.x[k+1][1] == self.x[k][1] + dt * self.theta_ddot[k],
                name=f"integrate_theta_dot_{k}"
            )

        # ---------------------------------------------------------------
        # Phase 2: extra MPC at k=K to compute V_next = J*(x[K])
        # Only added when V_0_max is provided (two-phase mode).
        # ---------------------------------------------------------------
        # if V_0_max is not None:
        self.x_mpc[K] = {}
        for t in range(T + 1):
            self.x_mpc[K][t] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                            name=f"x_mpc_{K}_{t}")
        self.u_mpc_var[K] = {}
        for t in range(T):
            self.u_mpc_var[K][t] = M.addVars(n_u, lb=-u_max, ub=u_max,
                                                name=f"u_mpc_{K}_{t}")
        # Initial condition: MPC starts from actual state x[K]
        for i in range(n_x):
            M.addConstr(self.x_mpc[K][0][i] == self.x[K][i],
                        name=f"mpc_init_{K}_{i}")

        self.sin_theta_mpc[K] = {}
        self.cos_theta_mpc[K] = {}
        self.theta_ddot_mpc[K] = {}
        for t in range(T):
            self.sin_theta_mpc[K][t] = M.addVar(lb=-1, ub=1,
                                                    name=f"sin_theta_mpc_{K}_{t}")
            self.cos_theta_mpc[K][t] = M.addVar(lb=-1, ub=1,
                                                    name=f"cos_theta_mpc_{K}_{t}")
            self.theta_ddot_mpc[K][t] = M.addVar(lb=-GRB.INFINITY,
                                                    name=f"theta_ddot_mpc_{K}_{t}")
            M.addGenConstrSin(self.x_mpc[K][t][0], self.sin_theta_mpc[K][t],
                                name=f"sin_mpc_{K}_{t}")
            M.addGenConstrCos(self.x_mpc[K][t][0], self.cos_theta_mpc[K][t],
                                name=f"cos_mpc_{K}_{t}")
            M.addConstr(self.x_mpc[K][t+1][0] ==
                        self.x_mpc[K][t][0] + dt * self.x_mpc[K][t][1],
                        name=f"mpc_theta_{K}_{t}")
            M.addConstr(self.theta_ddot_mpc[K][t] ==
                        g/length * self.sin_theta_mpc[K][t]
                        + self.u_mpc_var[K][t][0] / (mass * length**2),
                        name=f"mpc_angular_accel_{K}_{t}")
            M.addConstr(self.x_mpc[K][t+1][1] ==
                        self.x_mpc[K][t][1] + dt * self.theta_ddot_mpc[K][t],
                        name=f"mpc_theta_dot_{K}_{t}")

        self.lambda_mpc[K] = {}
        for t in range(T + 1):
            self.lambda_mpc[K][t] = M.addVars(n_x, lb=-GRB.INFINITY,
                                                name=f"lambda_mpc_{K}_{t}")
        for i in range(n_x):
            M.addConstr(self.lambda_mpc[K][T][i] == Q[i, i] * self.x_mpc[K][T][i],
                        name=f"terminal_costate_{K}_{i}")
        for t in range(T - 1, -1, -1):
            M.addConstr(
                self.lambda_mpc[K][t][0] == Q[0, 0] * self.x_mpc[K][t][0]
                + self.lambda_mpc[K][t+1][0]
                + dt * g / length * self.cos_theta_mpc[K][t] * self.lambda_mpc[K][t+1][1],
                name=f"costate_0_{K}_{t}"
            )
            M.addConstr(
                self.lambda_mpc[K][t][1] == Q[1, 1] * self.x_mpc[K][t][1]
                + dt * self.lambda_mpc[K][t+1][0] + self.lambda_mpc[K][t+1][1],
                name=f"costate_1_{K}_{t}"
            )
        for t in range(T):
            M.addConstr(
                R[0, 0] * self.u_mpc_var[K][t][0]
                + dt / (mass * length**2) * self.lambda_mpc[K][t+1][1] == 0,
                name=f"control_opt_{K}_{t}"
            )

        # Worst-case objective
        # if V_0_max is None:
        #     # Original: Lyapunov candidate V(x) = ||x||^2
        #     V_curr = gp.quicksum(self.x[0][i] * self.x[0][i] for i in range(n_x))
        #     V_next = gp.quicksum(self.x[K][i] * self.x[K][i] for i in range(n_x))
        # else:
        # Phase 2: Lyapunov candidate V(x) = J*(x) = MPC value function
        V_curr = (
            gp.quicksum(Q[i, i] * self.x_mpc[0][t][i] * self.x_mpc[0][t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R[j, j] * self.u_mpc_var[0][t][j] * self.u_mpc_var[0][t][j]
                            for t in range(T) for j in range(n_u))
        )
        V_next = (
            gp.quicksum(Q[i, i] * self.x_mpc[K][t][i] * self.x_mpc[K][t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R[j, j] * self.u_mpc_var[K][t][j] * self.u_mpc_var[K][t][j]
                            for t in range(T) for j in range(n_u))
        )
        # # Restrict to initial conditions reachable under the MPC policy
        # M.addConstr(V_curr <= V_0_max, name="V0_bound")

        # V_curr = gp.quicksum(self.x[0][i] * self.x[0][i] for i in range(n_x))
        # V_next = gp.quicksum(self.x[K][i] * self.x[K][i] for i in range(n_x))

        # import pdb
        # pdb.set_trace()

        self.V_curr = V_curr
        self.V_next = V_next

        eps = 1 - rho
        self.orig_objective = 0
        M.addConstr(V_next - V_curr + eps * V_curr >= 1e-6, name="V_curr_pos")
        M.setObjective(self.orig_objective, GRB.MAXIMIZE)

    def solve(self):
        self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        def v2dict(vs): return {k: vs[k].X for k in vs}

        if self.model.SolCount == 0:
            return {"obj": None}

        out = {
            "obj": self.model.ObjVal,

            "V_next": self.V_next.getValue(),
            "V_curr": self.V_curr.getValue(),

            # State trajectory: x[0] to x[K]
            "x": {k: v2dict(self.x[k]) for k in range(self.K + 1)},

            # Control trajectory: u[0] to u[K-1]
            "u": {k: v2dict(self.u[k]) for k in range(self.K)},

            # MPC trajectories
            "x_mpc": {k: {t: v2dict(self.x_mpc[k][t]) for t in range(len(self.x_mpc[k]))}
                     for k in range(self.K)},
            "u_mpc": {k: {t: v2dict(self.u_mpc_var[k][t]) for t in range(len(self.u_mpc_var[k]))}
                     for k in range(self.K)},
            "lambda_mpc": {k: {t: v2dict(self.lambda_mpc[k][t]) for t in range(len(self.lambda_mpc[k]))}
                          for k in range(self.K)},

            # Auxiliary variables
            "sin_theta": {k: self.sin_theta[k].X for k in range(self.K)},
            "cos_theta": {k: self.cos_theta[k].X for k in range(self.K)},
            "theta_ddot": {k: self.theta_ddot[k].X for k in range(self.K)},
        }

        return out
