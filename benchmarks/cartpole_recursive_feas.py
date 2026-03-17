import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

cmap = plt.cm.Set1
colors = cmap.colors
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

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
    """
    Sequential recursive feasibility certification.

    For each step j in j_vals and each T in T_vals, runs two problems:
      1. CartpoleRecursiveFeas:  max r*(x_j) over x_0 in X_0
         If r_opt = 0 => step j is certifiably recursively feasible.
      2. CartpoleMaxStateNorm:  max ||x_j||_inf over x_0 in X_0
         Gives the worst-case state magnitude at step j under the MPC policy.
    """
    j_vals = list(cfg.j_vals)
    T_vals = list(cfg.T_vals)
    r_cost = cfg.r_cost
    dt = cfg.dt
    x0_lo = cfg.x0_mins[0]
    x0_hi = cfg.x0_maxes[0]
    x_feas_lo = cfg.x_feas_mins[0]
    x_feas_hi = cfg.x_feas_maxes[0]
    u_bound = cfg.u_bound
    feas_tol = cfg.feas_tol

    # Store results indexed by (T, j)
    feas_results = {}   # r_opt
    norm_results = {}   # max ||x_j||_inf

    for T in T_vals:
        for j in j_vals:
            common_kwargs = dict(
                j=j, T=T, r_cost=r_cost, dt=dt,
                mass=1, length=1, g=9.8,
                x0_lo=x0_lo, x0_hi=x0_hi,
                x_feas_lo=x_feas_lo, x_feas_hi=x_feas_hi,
                u_bound=u_bound, verbose=False,
                time_limit=cfg.time_limit,
            )

            # --- Problem 1: recursive feasibility (max violation r) ---
            # print(f"\n=== Feas: j={j}, T={T} ===")
            # cert = CartpoleRecursiveFeas(**common_kwargs)
            # status_f, t_f = cert.solve()
            # sol_f = cert.solution_dict()
            # r_opt = sol_f['r_opt']
            # certified = r_opt is not None and r_opt <= feas_tol
            # feas_results[(T, j)] = {'certified': certified, 'r_opt': r_opt,
            #                          'status': status_f, 'time': t_f}
            # print(f"  r_opt={r_opt}  {'CERTIFIED' if certified else 'FAIL'}")
            # import pdb
            # pdb.set_trace()

            # --- Problem 2: max ||x_j||_inf ---
            print(f"=== Norm: j={j}, T={T} ===")
            norm_prob = CartpoleMaxStateNorm(**common_kwargs)
            status_n, t_n = norm_prob.solve()
            sol_n = norm_prob.solution_dict()
            norm_results[(T, j)] = {'inf_norm': sol_n['inf_norm'],
                                     'status': status_n, 'time': t_n}
            print(f"  ||x_j||_inf={sol_n['inf_norm']}")

    # Print summary
    print("\n=== Summary ===")
    for T in T_vals:
        for j in j_vals:
            rf = feas_results[(T, j)]
            rn = norm_results[(T, j)]
            tag = "OK" if rf['certified'] else f"FAIL(r={rf['r_opt']:.2e})"
            norm_str = f"{rn['inf_norm']:.4f}" if rn['inf_norm'] is not None else "N/A"
            print(f"  T={T}, j={j}: {tag}  ||x_j||_inf={norm_str}")

    j_axis = list(j_vals)

    # --- Plot 1: worst-case constraint violation r_opt vs j ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for ti, T in enumerate(T_vals):
        r_vals = [feas_results[(T, j)]['r_opt'] if feas_results[(T, j)]['r_opt'] is not None
                  else float('nan') for j in j_vals]
        ax1.plot(j_axis, r_vals,
                 marker=markers[ti % len(markers)], linewidth=2, color=colors[ti])
    ax1.set_xlabel('step $j$')
    ax1.set_ylabel('max violation $r^*$')
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig('rec_feas_violation.pdf', bbox_inches='tight')
    plt.close(fig1)

    # --- Plot 2: max ||x_j||_inf vs j ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for ti, T in enumerate(T_vals):
        norm_vals = [norm_results[(T, j)]['inf_norm'] if norm_results[(T, j)]['inf_norm'] is not None
                     else float('nan') for j in j_vals]
        ax2.plot(j_axis, norm_vals,
                 marker=markers[ti % len(markers)], linewidth=2, color=colors[ti])
    ax2.set_xlabel('step $j$')
    ax2.set_ylabel(r'$\max \|x_j\|_\infty$')
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig('rec_feas_max_norm.pdf', bbox_inches='tight')
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Shared helper: build j steps of true (box-constrained) MPC chain in model M
# ---------------------------------------------------------------------------

def _add_true_mpc_chain(M, j, T, r_cost, dt, mass, length, g,
                         x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound):
    """
    Add j steps of true MPC chain to model M.

    Variables created:
      x_chain[0..j]: actual closed-loop states
        x_chain[0] in X_0, x_chain[1..j] constrained to X_feas (certified earlier)
      For each k in 0..j-1:
        T-horizon true MPC (box-constrained KKT) at x_chain[k]
        applied control u_chain[k] = u_mpc[k][0]
        nonlinear dynamics: x_chain[k] -> x_chain[k+1]

    Returns (x_chain, u_chain) where u_chain[k] is the applied control var at step k.
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)
    R = np.eye(n_u) * r_cost

    x_chain = {}
    u_chain = {}
    x_chain[0] = M.addVars(n_x, lb=x0_lo, ub=x0_hi, name="xc_0")
    for t in range(1, j + 1):
        x_chain[t] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xc_{t}")
        for idx in range(n_x):
            M.addConstr(x_chain[t][idx] >= x_feas_lo, name=f"xc_feas_lo_{t}_{idx}")
            M.addConstr(x_chain[t][idx] <= x_feas_hi, name=f"xc_feas_hi_{t}_{idx}")

    for k in range(j):
        u_k = M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uc_{k}")
        u_chain[k] = u_k

        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xmc_{k}_{t}")
                 for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"umc_{k}_{t}")
                 for t in range(T)}
        mu_up = {t: M.addVars(n_u, lb=0.0, name=f"muc_up_{k}_{t}") for t in range(T)}
        mu_lo = {t: M.addVars(n_u, lb=0.0, name=f"muc_lo_{k}_{t}") for t in range(T)}

        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x_chain[k][i], name=f"mc_init_{k}_{i}")

        sin_m = {}
        cos_m = {}
        tdd_m = {}
        for t in range(T):
            sin_m[t] = M.addVar(lb=-1, ub=1, name=f"smc_{k}_{t}")
            cos_m[t] = M.addVar(lb=-1, ub=1, name=f"cmc_{k}_{t}")
            tdd_m[t] = M.addVar(lb=-GRB.INFINITY, name=f"tmc_{k}_{t}")
            M.addGenConstrSin(x_mpc[t][0], sin_m[t], name=f"sinc_{k}_{t}")
            M.addGenConstrCos(x_mpc[t][0], cos_m[t], name=f"cosc_{k}_{t}")
            M.addConstr(x_mpc[t+1][0] == x_mpc[t][0] + dt * x_mpc[t][1],
                        name=f"cdth_{k}_{t}")
            M.addConstr(tdd_m[t] == g/length * sin_m[t] + u_mpc[t][0] / (mass * length**2),
                        name=f"ctdd_{k}_{t}")
            M.addConstr(x_mpc[t+1][1] == x_mpc[t][1] + dt * tdd_m[t],
                        name=f"cdtd_{k}_{t}")

        lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamc_{k}_{t}")
               for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i] * x_mpc[T][i], name=f"tc_{k}_{i}")

        for t in range(T - 1, -1, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0] * x_mpc[t][0] + lam[t+1][0]
                + dt * g / length * cos_m[t] * lam[t+1][1],
                name=f"cs0c_{k}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1] * x_mpc[t][1] + dt * lam[t+1][0] + lam[t+1][1],
                name=f"cs1c_{k}_{t}")

        for t in range(T):
            M.addConstr(
                R[0, 0] * u_mpc[t][0] + dt / (mass * length**2) * lam[t+1][1]
                + mu_up[t][0] - mu_lo[t][0] == 0,
                name=f"statc_{k}_{t}")
            M.addConstr(mu_up[t][0] * (u_mpc[t][0] - u_bound) == 0,
                        name=f"cup_c_{k}_{t}")
            M.addConstr(mu_lo[t][0] * (-u_bound - u_mpc[t][0]) == 0,
                        name=f"clo_c_{k}_{t}")

        M.addConstr(u_k[0] == u_mpc[0][0], name=f"link_{k}")

        sin_k = M.addVar(lb=-1, ub=1, name=f"sc_{k}")
        cos_k = M.addVar(lb=-1, ub=1, name=f"cc_{k}")
        tdd_k = M.addVar(lb=-GRB.INFINITY, name=f"tdc_{k}")
        M.addGenConstrSin(x_chain[k][0], sin_k, name=f"sincc_{k}")
        M.addGenConstrCos(x_chain[k][0], cos_k, name=f"coscc_{k}")
        M.addConstr(x_chain[k+1][0] == x_chain[k][0] + dt * x_chain[k][1],
                    name=f"ith_{k}")
        M.addConstr(tdd_k == g/length * sin_k + u_k[0] / (mass * length**2),
                    name=f"itdd_{k}")
        M.addConstr(x_chain[k+1][1] == x_chain[k][1] + dt * tdd_k,
                    name=f"itd_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# Problem 1: recursive feasibility — max constraint violation r at step j
# ---------------------------------------------------------------------------

class CartpoleRecursiveFeas:
    """
    Outer problem: max_{x_0 in X_0} r*(x_j)
    where r*(x_j) = optimal constraint violation of the feasibility MPC at x_j.

    Feasibility MPC: min r s.t. dynamics, x_t[i] in [xf_lo-r, xf_hi+r] for all i, u in U, r >= 0.

    KKT of feasibility MPC (no stage cost, Q=0 in obj):
      Terminal:   lambda_T[0] = nu_lo_0[T] - nu_up_0[T]
                  lambda_T[1] = nu_lo_1[T] - nu_up_1[T]
      Backward:   lambda_t[0] = lambda_{t+1}[0] + dt*g/L*cos(x_t[0])*lambda_{t+1}[1]
                                - nu_up_0[t] + nu_lo_0[t]
                  lambda_t[1] = dt*lambda_{t+1}[0] + lambda_{t+1}[1]
                                - nu_up_1[t] + nu_lo_1[t]
      Stationarity u_t:  dt/(m*L^2)*lambda_{t+1}[1] + mu_up_t - mu_lo_t = 0
      Stationarity r:    1 - sum(nu_up_0[t]+nu_lo_0[t]+nu_up_1[t]+nu_lo_1[t]) - sigma_r = 0
      Complementarity for state constraints (both dims), control bounds, and r >= 0.
    """

    def __init__(self, j=0, T=5, r_cost=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x0_lo=0.0, x0_hi=4.0, x_feas_lo=-1.0, x_feas_hi=1.0,
                 u_bound=10.0, verbose=True, time_limit=None):
        self.j = j
        self.T = T
        n_x, n_u = 2, 1

        M = gp.Model("cartpole_rec_feas")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        self.x_chain, self.u_chain = _add_true_mpc_chain(
            M, j, T, r_cost, dt, mass, length, g,
            x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound)

        # Feasibility MPC at x_chain[j]
        self.r_var = M.addVar(lb=0.0, name="r_feas")
        self.x_f = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xf_{t}")
                    for t in range(T + 1)}
        self.u_f = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uf_{t}")
                    for t in range(T)}

        # Duals for theta (dim 0) and theta_dot (dim 1) state constraints
        self.nu_up_0 = {t: M.addVar(lb=0.0, name=f"nu_up0_{t}") for t in range(1, T + 1)}
        self.nu_lo_0 = {t: M.addVar(lb=0.0, name=f"nu_lo0_{t}") for t in range(1, T + 1)}
        self.nu_up_1 = {t: M.addVar(lb=0.0, name=f"nu_up1_{t}") for t in range(1, T + 1)}
        self.nu_lo_1 = {t: M.addVar(lb=0.0, name=f"nu_lo1_{t}") for t in range(1, T + 1)}
        self.sigma_r = M.addVar(lb=0.0, name="sigma_r")
        self.mu_up_f = {t: M.addVar(lb=0.0, name=f"mu_up_f_{t}") for t in range(T)}
        self.mu_lo_f = {t: M.addVar(lb=0.0, name=f"mu_lo_f_{t}") for t in range(T)}
        self.lam_f   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamf_{t}")
                        for t in range(T + 1)}
        sin_f, cos_f, tdd_f = {}, {}, {}

        for i in range(n_x):
            M.addConstr(self.x_f[0][i] == self.x_chain[j][i], name=f"finit_{i}")

        for t in range(T):
            sin_f[t] = M.addVar(lb=-1, ub=1, name=f"sf_{t}")
            cos_f[t] = M.addVar(lb=-1, ub=1, name=f"cf_{t}")
            tdd_f[t] = M.addVar(lb=-GRB.INFINITY, name=f"tdf_{t}")
            M.addGenConstrSin(self.x_f[t][0], sin_f[t], name=f"sinf_{t}")
            M.addGenConstrCos(self.x_f[t][0], cos_f[t], name=f"cosf_{t}")
            M.addConstr(self.x_f[t+1][0] == self.x_f[t][0] + dt * self.x_f[t][1],
                        name=f"fdth_{t}")
            M.addConstr(tdd_f[t] == g/length * sin_f[t] + self.u_f[t][0] / (mass * length**2),
                        name=f"ftdd_{t}")
            M.addConstr(self.x_f[t+1][1] == self.x_f[t][1] + dt * tdd_f[t],
                        name=f"fdtd_{t}")

        # State constraints for both theta and theta_dot
        for t in range(1, T + 1):
            M.addConstr(self.x_f[t][0] <= x_feas_hi + self.r_var, name=f"sc0_hi_{t}")
            M.addConstr(self.x_f[t][0] >= x_feas_lo - self.r_var, name=f"sc0_lo_{t}")
            M.addConstr(self.x_f[t][1] <= x_feas_hi + self.r_var, name=f"sc1_hi_{t}")
            M.addConstr(self.x_f[t][1] >= x_feas_lo - self.r_var, name=f"sc1_lo_{t}")

        # KKT: terminal costate — both dims have state constraint duals
        M.addConstr(self.lam_f[T][0] == self.nu_lo_0[T] - self.nu_up_0[T],
                    name="term_lam_f_0")
        M.addConstr(self.lam_f[T][1] == self.nu_lo_1[T] - self.nu_up_1[T],
                    name="term_lam_f_1")

        for t in range(T - 1, 0, -1):
            M.addConstr(
                self.lam_f[t][0] == self.lam_f[t+1][0]
                + dt * g / length * cos_f[t] * self.lam_f[t+1][1]
                - self.nu_up_0[t] + self.nu_lo_0[t],
                name=f"cs0f_{t}")
            M.addConstr(
                self.lam_f[t][1] == dt * self.lam_f[t+1][0] + self.lam_f[t+1][1]
                - self.nu_up_1[t] + self.nu_lo_1[t],
                name=f"cs1f_{t}")

        for t in range(T):
            M.addConstr(
                dt / (mass * length**2) * self.lam_f[t+1][1]
                + self.mu_up_f[t] - self.mu_lo_f[t] == 0,
                name=f"statf_{t}")

        M.addConstr(
            1.0 - gp.quicksum(
                self.nu_up_0[t] + self.nu_lo_0[t] + self.nu_up_1[t] + self.nu_lo_1[t]
                for t in range(1, T + 1))
            - self.sigma_r == 0,
            name="stat_r")

        # Complementarity: theta constraints
        for t in range(1, T + 1):
            M.addConstr(self.nu_up_0[t] * (self.r_var + x_feas_hi - self.x_f[t][0]) == 0,
                        name=f"comp_up0_f_{t}")
            M.addConstr(self.nu_lo_0[t] * (self.r_var + self.x_f[t][0] - x_feas_lo) == 0,
                        name=f"comp_lo0_f_{t}")

        # Complementarity: theta_dot constraints
        for t in range(1, T + 1):
            M.addConstr(self.nu_up_1[t] * (self.r_var + x_feas_hi - self.x_f[t][1]) == 0,
                        name=f"comp_up1_f_{t}")
            M.addConstr(self.nu_lo_1[t] * (self.r_var + self.x_f[t][1] - x_feas_lo) == 0,
                        name=f"comp_lo1_f_{t}")

        for t in range(T):
            M.addConstr(self.mu_up_f[t] * (self.u_f[t][0] - u_bound) == 0,
                        name=f"comp_up_uf_{t}")
            M.addConstr(self.mu_lo_f[t] * (-u_bound - self.u_f[t][0]) == 0,
                        name=f"comp_lo_uf_{t}")

        M.addConstr(self.sigma_r * self.r_var == 0, name="comp_r")
        M.setObjective(self.r_var, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {'r_opt': None}
        return {
            'r_opt': self.r_var.X,
            # true MPC closed-loop chain: states and applied controls
            'x_chain': {k: {i: self.x_chain[k][i].X for i in range(2)}
                        for k in range(self.j + 1)},
            'u_chain': {k: self.u_chain[k][0].X for k in range(self.j)},
            # feasibility MPC at x_chain[j]: states, controls, costates, duals
            'x_feas':  {t: {i: self.x_f[t][i].X for i in range(2)}
                        for t in range(self.T + 1)},
            'u_feas':  {t: self.u_f[t][0].X for t in range(self.T)},
            'lam_feas': {t: {i: self.lam_f[t][i].X for i in range(2)}
                         for t in range(self.T + 1)},
            'nu_up_0': {t: self.nu_up_0[t].X for t in range(1, self.T + 1)},
            'nu_lo_0': {t: self.nu_lo_0[t].X for t in range(1, self.T + 1)},
            'nu_up_1': {t: self.nu_up_1[t].X for t in range(1, self.T + 1)},
            'nu_lo_1': {t: self.nu_lo_1[t].X for t in range(1, self.T + 1)},
            'mu_up_f': {t: self.mu_up_f[t].X for t in range(self.T)},
            'mu_lo_f': {t: self.mu_lo_f[t].X for t in range(self.T)},
            'sigma_r': self.sigma_r.X,
        }


# ---------------------------------------------------------------------------
# Problem 2: max ||x_j||_inf — worst-case state magnitude at step j
# ---------------------------------------------------------------------------

class CartpoleMaxStateNorm:
    """
    Outer problem: max_{x_0 in X_0} ||x_j||_inf
    subject to the same j-step true MPC chain (x_t in X_feas for t=1..j).

    Decomposes into 2*n_x sub-problems (one per component/sign):
        max_{x_0}  sign * x_chain[j][i]
    and returns the largest value found.

    Note: the single-model formulation "max z, z >= |x_j[i]|" is always
    unbounded because z has no implicit upper bound, so the decomposition
    approach is required.
    """

    def __init__(self, j=0, T=5, r_cost=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x0_lo=0.0, x0_hi=4.0, x_feas_lo=-1.0, x_feas_hi=1.0,
                 u_bound=10.0, verbose=True, time_limit=None):
        self.j = j
        n_x = 2
        self._subs = []  # list of (model, x_chain, component_idx, sign)

        for i in range(n_x):
            for sign in [1, -1]:
                M = gp.Model(f"cmn_{i}_{sign}")
                M.Params.OutputFlag = 1 if verbose else 0
                M.Params.FeasibilityTol = 1e-9
                M.Params.NonConvex = 2
                if time_limit is not None:
                    M.Params.TimeLimit = time_limit
                xc, uc = _add_true_mpc_chain(
                    M, j, T, r_cost, dt, mass, length, g,
                    x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound)
                M.setObjective(sign * xc[j][i], GRB.MAXIMIZE)
                self._subs.append((M, xc, uc, i, sign))

    def solve(self):
        t_total = 0.0
        for M, _, _, _, _ in self._subs:
            M.optimize()
            t_total += M.Runtime
        return [M.Status for M, _, _, _, _ in self._subs], t_total

    def solution_dict(self):
        best_val = None
        best_xc = None
        best_uc = None
        for M, xc, uc, i, sign in self._subs:
            if M.SolCount == 0:
                continue
            val = sign * xc[self.j][i].X
            if best_val is None or val > best_val:
                best_val = val
                best_xc = xc
                best_uc = uc
        if best_val is None:
            return {'inf_norm': None}
        return {
            'inf_norm': best_val,
            'x_chain': {k: {idx: best_xc[k][idx].X for idx in range(2)}
                        for k in range(self.j + 1)},
            'u_chain': {k: best_uc[k][0].X for k in range(self.j)},
        }
