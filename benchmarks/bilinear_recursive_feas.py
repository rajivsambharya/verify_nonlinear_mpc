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
    Sequential recursive feasibility certification for the bilinear system:
      x+[0] = 0.9*x[0] + u + 0.2*u*x[0]
      x+[1] = 0.85*x[1] + x[0]

    For each step j in j_vals and each T in T_vals, runs two problems:
      1. BilinearRecursiveFeas:  max r*(x_j) over x_0 in X_0
         If r_opt = 0 => step j is certifiably recursively feasible.
      2. BilinearMaxStateNorm:  max ||x_j||_inf over x_0 in X_0
         Gives the worst-case state magnitude at step j under the MPC policy.
    """
    j_vals = list(cfg.j_vals)
    T_vals = list(cfg.T_vals)
    r_cost = cfg.r_cost
    x0_lo = cfg.x0_mins[0]
    x0_hi = cfg.x0_maxes[0]
    x_feas_lo = cfg.x_feas_mins[0]
    x_feas_hi = cfg.x_feas_maxes[0]
    u_bound = cfg.u_bound
    feas_tol = cfg.feas_tol

    feas_results = {}
    norm_results = {}

    for T in T_vals:
        for j in j_vals:
            common_kwargs = dict(
                j=j, T=T, r_cost=r_cost,
                x0_lo=x0_lo, x0_hi=x0_hi,
                x_feas_lo=x_feas_lo, x_feas_hi=x_feas_hi,
                u_bound=u_bound, verbose=False,
                time_limit=cfg.time_limit,
            )

            # --- Problem 1: recursive feasibility (max violation r) ---
            # print(f"=== Feas: j={j}, T={T} ===")
            # cert = BilinearRecursiveFeas(**common_kwargs)
            # status_f, t_f = cert.solve()
            # sol_f = cert.solution_dict()
            # r_opt = sol_f['r_opt']
            # certified = r_opt is not None and r_opt <= feas_tol
            # feas_results[(T, j)] = {'certified': certified, 'r_opt': r_opt,
            #                          'status': status_f, 'time': t_f}
            # print(f"  r_opt={r_opt}  {'CERTIFIED' if certified else 'FAIL'}")

            # --- Problem 2: max ||x_j||_inf ---
            print(f"=== Norm: j={j}, T={T} ===")
            norm_prob = BilinearMaxStateNorm(**common_kwargs)
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
            if rf['certified']:
                tag = "OK"
            elif rf['r_opt'] is not None:
                tag = f"FAIL(r={rf['r_opt']:.2e})"
            else:
                tag = "FAIL(N/A)"
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
    fig1.savefig('bilinear_rec_feas_violation.pdf', bbox_inches='tight')
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
    fig2.savefig('bilinear_rec_feas_max_norm.pdf', bbox_inches='tight')
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Shared helper: build j steps of true (box-constrained) MPC chain in model M
# ---------------------------------------------------------------------------

def _add_true_mpc_chain(M, j, T, r_cost, x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound):
    """
    Add j steps of true MPC chain to model M for the bilinear system:
      x+[0] = 0.9*x[0] + u + 0.2*u*x[0]
      x+[1] = 0.85*x[1] + x[0]

    Variables created:
      x_chain[0..j]: actual closed-loop states
        x_chain[0] in X_0, x_chain[1..j] constrained to X_feas
      For each k in 0..j-1:
        T-horizon true MPC (box-constrained KKT) at x_chain[k]
        applied control u_chain[k] = u_mpc[k][0]
        bilinear dynamics: x_chain[k] -> x_chain[k+1]

    Returns (x_chain, u_chain).
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

        # Inner MPC variables
        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xmc_{k}_{t}")
                 for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"umc_{k}_{t}")
                 for t in range(T)}
        mu_up = {t: M.addVars(n_u, lb=0.0, name=f"muc_up_{k}_{t}") for t in range(T)}
        mu_lo = {t: M.addVars(n_u, lb=0.0, name=f"muc_lo_{k}_{t}") for t in range(T)}

        # Initial condition: x_mpc[0] = x_chain[k]
        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x_chain[k][i], name=f"mc_init_{k}_{i}")

        # Bilinear dynamics: x_mpc[t+1] = f(x_mpc[t], u_mpc[t])
        for t in range(T):
            M.addConstr(
                x_mpc[t+1][0] == 0.9*x_mpc[t][0] + u_mpc[t][0]
                + 0.2*u_mpc[t][0]*x_mpc[t][0],
                name=f"cdyn0_{k}_{t}")
            M.addConstr(
                x_mpc[t+1][1] == 0.85*x_mpc[t][1] + x_mpc[t][0],
                name=f"cdyn1_{k}_{t}")

        # Costates
        lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamc_{k}_{t}")
               for t in range(T + 1)}

        # Terminal costate: lam[T][i] = Q[i,i]*x_mpc[T][i]
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i] * x_mpc[T][i], name=f"tc_{k}_{i}")

        # Backward costate recursion (t = T-1 down to 0):
        #   lam[t][0] = Q[0,0]*x_mpc[t][0] + (0.9 + 0.2*u_mpc[t][0])*lam[t+1][0] + lam[t+1][1]
        #   lam[t][1] = Q[1,1]*x_mpc[t][1] + 0.85*lam[t+1][1]
        for t in range(T - 1, -1, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0] * x_mpc[t][0]
                + 0.9 * lam[t+1][0] + 0.2 * u_mpc[t][0] * lam[t+1][0]
                + lam[t+1][1],
                name=f"cs0c_{k}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1] * x_mpc[t][1] + 0.85 * lam[t+1][1],
                name=f"cs1c_{k}_{t}")

        # Stationarity w.r.t. u_mpc[t]:
        #   R*u + (1 + 0.2*x_mpc[t][0])*lam[t+1][0] + mu_up - mu_lo = 0
        for t in range(T):
            M.addConstr(
                R[0, 0] * u_mpc[t][0]
                + lam[t+1][0] + 0.2 * x_mpc[t][0] * lam[t+1][0]
                + mu_up[t][0] - mu_lo[t][0] == 0,
                name=f"statc_{k}_{t}")
            M.addConstr(mu_up[t][0] * (u_mpc[t][0] - u_bound) == 0,
                        name=f"cup_c_{k}_{t}")
            M.addConstr(mu_lo[t][0] * (-u_bound - u_mpc[t][0]) == 0,
                        name=f"clo_c_{k}_{t}")

        # Link applied control: u_k = u_mpc[0]
        M.addConstr(u_k[0] == u_mpc[0][0], name=f"link_{k}")

        # Chain dynamics: x_chain[k+1] = f(x_chain[k], u_k)
        M.addConstr(
            x_chain[k+1][0] == 0.9*x_chain[k][0] + u_k[0]
            + 0.2*u_k[0]*x_chain[k][0],
            name=f"idyn0_{k}")
        M.addConstr(
            x_chain[k+1][1] == 0.85*x_chain[k][1] + x_chain[k][0],
            name=f"idyn1_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# Problem 1: recursive feasibility — max constraint violation r at step j
# ---------------------------------------------------------------------------

class BilinearRecursiveFeas:
    """
    Outer problem: max_{x_0 in X_0} r*(x_j)
    where r*(x_j) = optimal constraint violation of the feasibility MPC at x_j.

    Feasibility MPC: min r s.t. dynamics, x_t[i] in [xf_lo-r, xf_hi+r] for all i, u in U, r >= 0.

    KKT of feasibility MPC (no stage cost):
      Terminal:   lambda_T[i] = nu_lo_i[T] - nu_up_i[T]  for i=0,1
      Backward:   lambda_t[0] = (0.9 + 0.2*u_f[t])*lambda_{t+1}[0] + lambda_{t+1}[1]
                                - nu_up_0[t] + nu_lo_0[t]
                  lambda_t[1] = 0.85*lambda_{t+1}[1] - nu_up_1[t] + nu_lo_1[t]
      Stationarity u_t:  (1 + 0.2*x_f[t][0])*lambda_{t+1}[0] + mu_up_t - mu_lo_t = 0
      Stationarity r:    1 - sum(nu_up_0+nu_lo_0+nu_up_1+nu_lo_1) - sigma_r = 0
      Complementarity for state constraints (both dims), control bounds, and r >= 0.
    """

    def __init__(self, j=0, T=5, r_cost=0.1,
                 x0_lo=0.0, x0_hi=1.0, x_feas_lo=-2.0, x_feas_hi=2.0,
                 u_bound=1.0, verbose=True, time_limit=None):
        self.j = j
        self.T = T
        n_x, n_u = 2, 1

        M = gp.Model("bilinear_rec_feas")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        self.x_chain, self.u_chain = _add_true_mpc_chain(
            M, j, T, r_cost, x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound)

        # Feasibility MPC at x_chain[j]
        self.r_var = M.addVar(lb=0.0, name="r_feas")
        self.x_f = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xf_{t}")
                    for t in range(T + 1)}
        self.u_f = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uf_{t}")
                    for t in range(T)}

        # Duals for dim 0 and dim 1 state constraints
        self.nu_up_0 = {t: M.addVar(lb=0.0, name=f"nu_up0_{t}") for t in range(1, T + 1)}
        self.nu_lo_0 = {t: M.addVar(lb=0.0, name=f"nu_lo0_{t}") for t in range(1, T + 1)}
        self.nu_up_1 = {t: M.addVar(lb=0.0, name=f"nu_up1_{t}") for t in range(1, T + 1)}
        self.nu_lo_1 = {t: M.addVar(lb=0.0, name=f"nu_lo1_{t}") for t in range(1, T + 1)}
        self.sigma_r = M.addVar(lb=0.0, name="sigma_r")
        self.mu_up_f = {t: M.addVar(lb=0.0, name=f"mu_up_f_{t}") for t in range(T)}
        self.mu_lo_f = {t: M.addVar(lb=0.0, name=f"mu_lo_f_{t}") for t in range(T)}
        self.lam_f = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamf_{t}")
                      for t in range(T + 1)}

        # Initial condition: x_f[0] = x_chain[j]
        for i in range(n_x):
            M.addConstr(self.x_f[0][i] == self.x_chain[j][i], name=f"finit_{i}")

        # Bilinear feasibility MPC dynamics
        for t in range(T):
            M.addConstr(
                self.x_f[t+1][0] == 0.9*self.x_f[t][0] + self.u_f[t][0]
                + 0.2*self.u_f[t][0]*self.x_f[t][0],
                name=f"fdyn0_{t}")
            M.addConstr(
                self.x_f[t+1][1] == 0.85*self.x_f[t][1] + self.x_f[t][0],
                name=f"fdyn1_{t}")

        # State constraints for both dims
        for t in range(1, T + 1):
            M.addConstr(self.x_f[t][0] <= x_feas_hi + self.r_var, name=f"sc0_hi_{t}")
            M.addConstr(self.x_f[t][0] >= x_feas_lo - self.r_var, name=f"sc0_lo_{t}")
            M.addConstr(self.x_f[t][1] <= x_feas_hi + self.r_var, name=f"sc1_hi_{t}")
            M.addConstr(self.x_f[t][1] >= x_feas_lo - self.r_var, name=f"sc1_lo_{t}")

        # KKT: terminal costate
        M.addConstr(self.lam_f[T][0] == self.nu_lo_0[T] - self.nu_up_0[T],
                    name="term_lam_f_0")
        M.addConstr(self.lam_f[T][1] == self.nu_lo_1[T] - self.nu_up_1[T],
                    name="term_lam_f_1")

        # KKT: backward costate recursion (t = T-1 down to 1):
        #   lam_f[t][0] = (0.9 + 0.2*u_f[t])*lam_f[t+1][0] + lam_f[t+1][1]
        #                 - nu_up_0[t] + nu_lo_0[t]
        #   lam_f[t][1] = 0.85*lam_f[t+1][1] - nu_up_1[t] + nu_lo_1[t]
        for t in range(T - 1, 0, -1):
            M.addConstr(
                self.lam_f[t][0] == 0.9*self.lam_f[t+1][0]
                + 0.2*self.u_f[t][0]*self.lam_f[t+1][0]
                + self.lam_f[t+1][1]
                - self.nu_up_0[t] + self.nu_lo_0[t],
                name=f"cs0f_{t}")
            M.addConstr(
                self.lam_f[t][1] == 0.85*self.lam_f[t+1][1]
                - self.nu_up_1[t] + self.nu_lo_1[t],
                name=f"cs1f_{t}")

        # KKT: stationarity w.r.t. u_f[t]:
        #   (1 + 0.2*x_f[t][0])*lam_f[t+1][0] + mu_up_f[t] - mu_lo_f[t] = 0
        for t in range(T):
            M.addConstr(
                self.lam_f[t+1][0] + 0.2*self.x_f[t][0]*self.lam_f[t+1][0]
                + self.mu_up_f[t] - self.mu_lo_f[t] == 0,
                name=f"statf_{t}")

        # KKT: stationarity w.r.t. r:
        #   1 - sum(nu_up_0 + nu_lo_0 + nu_up_1 + nu_lo_1) - sigma_r = 0
        M.addConstr(
            1.0 - gp.quicksum(
                self.nu_up_0[t] + self.nu_lo_0[t] + self.nu_up_1[t] + self.nu_lo_1[t]
                for t in range(1, T + 1))
            - self.sigma_r == 0,
            name="stat_r")

        # Complementarity: dim 0 state constraints
        for t in range(1, T + 1):
            M.addConstr(self.nu_up_0[t] * (self.r_var + x_feas_hi - self.x_f[t][0]) == 0,
                        name=f"comp_up0_f_{t}")
            M.addConstr(self.nu_lo_0[t] * (self.r_var + self.x_f[t][0] - x_feas_lo) == 0,
                        name=f"comp_lo0_f_{t}")

        # Complementarity: dim 1 state constraints
        for t in range(1, T + 1):
            M.addConstr(self.nu_up_1[t] * (self.r_var + x_feas_hi - self.x_f[t][1]) == 0,
                        name=f"comp_up1_f_{t}")
            M.addConstr(self.nu_lo_1[t] * (self.r_var + self.x_f[t][1] - x_feas_lo) == 0,
                        name=f"comp_lo1_f_{t}")

        # Complementarity: control bounds
        for t in range(T):
            M.addConstr(self.mu_up_f[t] * (self.u_f[t][0] - u_bound) == 0,
                        name=f"comp_up_uf_{t}")
            M.addConstr(self.mu_lo_f[t] * (-u_bound - self.u_f[t][0]) == 0,
                        name=f"comp_lo_uf_{t}")

        # Complementarity: r >= 0
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
            'x_chain': {k: {i: self.x_chain[k][i].X for i in range(2)}
                        for k in range(self.j + 1)},
            'u_chain': {k: self.u_chain[k][0].X for k in range(self.j)},
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

class BilinearMaxStateNorm:
    """
    Outer problem: max_{x_0 in X_0} ||x_j||_inf
    subject to the same j-step true MPC chain (x_t in X_feas for t=1..j).

    Decomposes into 2*n_x sub-problems (one per component/sign):
        max_{x_0}  sign * x_chain[j][i]
    and returns the largest value found.
    """

    def __init__(self, j=0, T=5, r_cost=0.1,
                 x0_lo=0.0, x0_hi=1.0, x_feas_lo=-2.0, x_feas_hi=2.0,
                 u_bound=1.0, verbose=True, time_limit=None):
        self.j = j
        n_x = 2
        self._subs = []  # list of (model, x_chain, u_chain, component_idx, sign)

        for i in range(n_x):
            for sign in [1, -1]:
                M = gp.Model(f"bmn_{i}_{sign}")
                M.Params.OutputFlag = 1 if verbose else 0
                M.Params.FeasibilityTol = 1e-9
                M.Params.NonConvex = 2
                if time_limit is not None:
                    M.Params.TimeLimit = time_limit
                xc, uc = _add_true_mpc_chain(
                    M, j, T, r_cost, x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound)
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
