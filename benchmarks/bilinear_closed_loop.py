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
    Recursive feasibility certification for SCP-linearized MPC on the bilinear system:
      x+[0] = 0.9*x[0] + u + 0.2*u*x[0]
      x+[1] = 0.7*x[1] + 0.5*x[0]

    Uses the Farkas lemma to search for the most problematic initial state x_0 in X_0.
    If the Farkas objective > 0, there exists an x_0 causing infeasibility of the
    SCP QP at step j.  If <= 0, feasibility is certified for all x_0 in X_0.
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

    farkas_results = {}
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

            # Max closed-loop suboptimality
            print(f"=== closed-loop suboptimality: j={j}, T={T} ===")
            norm_prob = BilinearMaxStateNorm(**common_kwargs)
            status_n, t_n = norm_prob.solve()
            sol_n = norm_prob.solution_dict()
            norm_results[(T, j)] = {
                'inf_norm': sol_n['inf_norm'],
                'status': status_n, 'time': t_n,
            }
            print(f"  ||x_j||_inf={sol_n['inf_norm']}")

    # Summary
    print("\n=== Summary ===")
    for T in T_vals:
        for j in j_vals:
            # rf = farkas_results[(T, j)]
            rn = norm_results[(T, j)]
            # if rf['certified']:
            #     tag = "OK"
            # elif rf['farkas_obj'] is not None:
            #     tag = f"FAIL(obj={rf['farkas_obj']:.2e})"
            # else:
            #     tag = "FAIL(N/A)"
            norm_str = f"{rn['inf_norm']:.4f}" if rn['inf_norm'] is not None else "N/A"
            # print(f"  T={T}, j={j}: {tag}  ||x_j||_inf={norm_str}")

    j_axis = list(j_vals)


    # Plot 2: max ||x_j||_inf vs j
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for ti, T in enumerate(T_vals):
        norm_vals = [
            norm_results[(T, j)]['inf_norm']
            if norm_results[(T, j)]['inf_norm'] is not None else float('nan')
            for j in j_vals
        ]
        ax2.plot(j_axis[1:], norm_vals[1:],
                 marker=markers[ti % len(markers)], linewidth=2, color=colors[ti],
                 label=f'T={T}')
    # import pdb; pdb.set_trace()
    ax2.set_xlabel('step $j$')
    ax2.set_yscale('log')
    ax2.set_ylabel('closed-loop suboptimality')
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig('bilinear_closed-loop_suboptimality.pdf', bbox_inches='tight')
    plt.close(fig2)



def _add_scp_chain(M, j, T, r_cost, x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound):
    """
    Build j steps of SCP MPC chain in model M.

    At each step k=0..j-1:
      - Linearize at x_chain[k]: B(x_chain[k]) = [[1+0.2*x_chain[k][0]], [0]]
      - Add KKT of the linearized QP to obtain optimal control u_k
      - Propagate true nonlinear dynamics: x_chain[k+1] = f_true(x_chain[k], u_k)

    x_chain[0] in [x0_lo, x0_hi]^2.
    x_chain[1..j] bounded in [x_feas_lo, x_feas_hi]^2.

    Returns x_chain (dict 0..j).
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)

    x_chain = {}
    x_chain[0] = M.addVars(n_x, lb=x0_lo, ub=x0_hi, name="xc_0")

    for k in range(j):
        x_k = x_chain[k]

        # Inner QP variables
        x_m = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xm_{k}_{t}")
               for t in range(T + 1)}
        u_m = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"um_{k}_{t}")
               for t in range(T)}
        lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lam_{k}_{t}")
               for t in range(T + 1)}
        mu_up = {t: M.addVar(lb=0.0, name=f"mu_up_{k}_{t}") for t in range(T)}
        mu_lo = {t: M.addVar(lb=0.0, name=f"mu_lo_{k}_{t}") for t in range(T)}
        xi_up = {t: M.addVars(n_x, lb=0.0, name=f"xi_up_{k}_{t}") for t in range(1, T + 1)}
        xi_lo = {t: M.addVars(n_x, lb=0.0, name=f"xi_lo_{k}_{t}") for t in range(1, T + 1)}

        # Initial condition
        for i in range(n_x):
            M.addConstr(x_m[0][i] == x_k[i], name=f"mc_init_{k}_{i}")

        # x_feas constraints on inner QP states
        for t in range(1, T + 1):
            for i in range(n_x):
                M.addConstr(x_m[t][i] >= x_feas_lo, name=f"mc_flo_{k}_{t}_{i}")
                M.addConstr(x_m[t][i] <= x_feas_hi, name=f"mc_fhi_{k}_{t}_{i}")

        # Linearized dynamics: x_{t+1} = A*x_m[t] + B(x_k)*u_m[t]
        # x_m[t+1][0] = 0.9*x_m[t][0] + (1 + 0.2*x_k[0])*u_m[t][0]
        # x_m[t+1][1] = 0.5*x_m[t][0] + 0.7*x_m[t][1]
        for t in range(T):
            M.addConstr(
                x_m[t+1][0] == 0.9*x_m[t][0] + u_m[t][0] + 0.2*x_k[0]*u_m[t][0],
                name=f"mc_dyn0_{k}_{t}")
            M.addConstr(
                x_m[t+1][1] == 0.5*x_m[t][0] + 0.7*x_m[t][1],
                name=f"mc_dyn1_{k}_{t}")

        # KKT terminal costate
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i]*x_m[T][i] - xi_up[T][i] + xi_lo[T][i],
                        name=f"mc_term_{k}_{i}")

        # KKT backward costate (t=T-1..1)
        # A^T = [[0.9, 0.5], [0, 0.7]]
        for t in range(T - 1, 0, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0]*x_m[t][0]
                + 0.9*lam[t+1][0] + 0.5*lam[t+1][1]
                - xi_up[t][0] + xi_lo[t][0],
                name=f"mc_back0_{k}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1]*x_m[t][1]
                + 0.7*lam[t+1][1]
                - xi_up[t][1] + xi_lo[t][1],
                name=f"mc_back1_{k}_{t}")

        # KKT stationarity w.r.t. u_m[t]:
        # R*u_m[t][0] + (1 + 0.2*x_k[0])*lam[t+1][0] + mu_up[t] - mu_lo[t] = 0
        for t in range(T):
            M.addConstr(
                r_cost*u_m[t][0]
                + lam[t+1][0] + 0.2*x_k[0]*lam[t+1][0]
                + mu_up[t] - mu_lo[t] == 0,
                name=f"mc_stat_{k}_{t}")

        # Complementarity: control
        for t in range(T):
            M.addConstr(mu_up[t]*(u_m[t][0] - u_bound) == 0, name=f"mc_cup_{k}_{t}")
            M.addConstr(mu_lo[t]*(-u_bound - u_m[t][0]) == 0, name=f"mc_clo_{k}_{t}")

        # Complementarity: state constraints
        for t in range(1, T + 1):
            for i in range(n_x):
                M.addConstr(xi_up[t][i]*(x_feas_hi - x_m[t][i]) == 0,
                            name=f"mc_cup_x_{k}_{t}_{i}")
                M.addConstr(xi_lo[t][i]*(x_m[t][i] - x_feas_lo) == 0,
                            name=f"mc_clo_x_{k}_{t}_{i}")

        # True nonlinear chain dynamics: x_chain[k+1] = f_true(x_chain[k], u_k)
        # u_k = u_m[0][0]
        u_k = u_m[0][0]
        x_chain[k+1] = M.addVars(n_x, lb=x_feas_lo, ub=x_feas_hi, name=f"xc_{k+1}")
        M.addConstr(
            x_chain[k+1][0] == 0.9*x_k[0] + u_k + 0.2*u_k*x_k[0],
            name=f"chain_dyn0_{k}")
        M.addConstr(
            x_chain[k+1][1] == 0.7*x_k[1] + 0.5*x_k[0],
            name=f"chain_dyn1_{k}")

    return x_chain




# ---------------------------------------------------------------------------
# Helper: true (nonlinear) MPC chain for BilinearMaxStateNorm
# ---------------------------------------------------------------------------

def _add_true_mpc_chain(M, j, T, r_cost, x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound):
    """
    Add j steps of true MPC chain to model M for the bilinear system:
      x+[0] = 0.9*x[0] + u + 0.2*u*x[0]
      x+[1] = 0.7*x[1] + 0.5*x[0]

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

    x_feas_lo = -GRB.INFINITY
    x_feas_hi = GRB.INFINITY
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
                x_mpc[t+1][1] == 0.7*x_mpc[t][1] + 0.5*x_mpc[t][0],
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
                + 0.5 * lam[t+1][1],
                name=f"cs0c_{k}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1] * x_mpc[t][1] + 0.7 * lam[t+1][1],
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
            x_chain[k+1][1] == 0.7*x_chain[k][1] + 0.5*x_chain[k][0],
            name=f"idyn1_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# Max ||x_j||_inf — worst-case state magnitude at step j under true MPC
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

        # for i in range(n_x):
        #     for sign in [1, -1]:
        #         M = gp.Model(f"bmn_{i}_{sign}")
        #         M.Params.OutputFlag = 1 if verbose else 0
        #         M.Params.FeasibilityTol = 1e-9
        #         M.Params.NonConvex = 2
        #         if time_limit is not None:
        #             M.Params.TimeLimit = time_limit
        #         xc, uc = _add_true_mpc_chain(
        #             M, j, T, r_cost, x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound)
        #         M.setObjective(sign * xc[j][i], GRB.MAXIMIZE)
        #         self._subs.append((M, xc, uc, i, sign))
        i = 0
        sign = 1
        M = gp.Model(f"bmn_{i}_{sign}")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        xc, uc = _add_true_mpc_chain(
            M, j, T, r_cost, x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound)
        # M.setObjective(sign * xc[j][i], GRB.MAXIMIZE)

        # Chain dynamics: x_chain[k+1] = f(x_chain[k], u_k)
        x_opt = {}
        u_opt = {}
        n_u = 1
        # x_opt[0] = M.addVars(n_x, lb=x0_lo, ub=x0_hi, name="xc_0")
        x_opt[0] = xc[0]
        for t in range(1, j + 1):
            x_opt[t] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xc_{t}")
            # x_opt[t] = M.addVars(n_x, lb=x_feas_lo, ub=x_feas_hi, name=f"xc_{t}")
            # for idx in range(n_x):
            #     M.addConstr(x_opt[t][idx] >= x_feas_lo, name=f"xc_feas_lo_{t}_{idx}")
            #     M.addConstr(x_opt[t][idx] <= x_feas_hi, name=f"xc_feas_hi_{t}_{idx}")

        for k in range(j):
            u_k = M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uc_{k}")
            u_opt[k] = u_k
            M.addConstr(
                x_opt[k+1][0] == 0.9*x_opt[k][0] + u_opt[k][0] + 0.2*u_opt[k][0]*x_opt[k][0], name=f"idyn0_{k}")
            M.addConstr(
                x_opt[k+1][1] == 0.7*x_opt[k][1] + 0.5*x_opt[k][0],
                name=f"idyn1_{k}")
        # import pdb; pdb.set_trace()
        obj = 0
        for jj in range(j + 1):
            for ii in range(n_x):
                obj += xc[jj][ii] * xc[jj][ii] - x_opt[jj][ii] * x_opt[jj][ii]
        for jj in range(j):
            # obj += r_cost * (uc[jj+1][0] * uc[jj+1])
            obj += r_cost * (uc[jj][0] * uc[jj][0] - u_opt[jj][0] * u_opt[jj][0])
        M.setParam('BestBdStop', 1e-5)
        M.setObjective(obj, GRB.MAXIMIZE)
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
            best_val = M.ObjVal
            # val =  xc[self.j][i].X
            # if best_val is None or val > best_val:
            #     best_val = val
            #     best_xc = xc
            #     best_uc = uc
        if best_val is None:
            return {'inf_norm': None}
        return {
            'inf_norm': best_val,
            # 'x_chain': {k: {idx: best_xc[k][idx].X for idx in range(2)}
            #             for k in range(self.j + 1)},
            # 'u_chain': {k: best_uc[k][0].X for k in range(self.j)},
        }
