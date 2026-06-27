import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class BilinearSystem:
    """
    Bilinear system: x+[0] = a00*x[0] + u + b*u*x[0]
                     x+[1] = a11*x[1] + a10*x[0]

    Linearized at (x_bar, u_bar):
      A(u_bar) = [[a00 + b*u_bar, 0], [a10, a11]]
      B(x_bar) = [[1 + b*x_bar[0]], [0]]
      c(x_bar, u_bar) = [-b*u_bar*x_bar[0], 0]   (affine offset)
    """
    a00: float  # x[0] self-transition
    a10: float  # coupling x[0] -> x[1]
    a11: float  # x[1] self-transition
    b:   float  # bilinear coefficient on u*x[0]

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
    Closed-loop suboptimality for the bilinear system:
      x+[0] = 0.9*x[0] + u + 0.2*u*x[0]
      x+[1] = 0.7*x[1] + 0.5*x[0]

    For each j and T, computes max_{x_0 in X_0} [J_policy(x_0) - J_opt(x_0)]
    for two policies:
      1. True nonlinear MPC (exact KKT)
      2. 1-iteration SCP MPC (linearize at current state, u_lin=0)
    """
    j_vals = list(cfg.j_vals)
    T_vals = list(cfg.T_vals)
    r_cost = cfg.r_cost
    x0_lo = cfg.x0_mins[0]
    x0_hi = cfg.x0_maxes[0]
    u_bound = cfg.u_bound
    sys = BilinearSystem(a00=cfg.a00, a10=cfg.a10, a11=cfg.a11, b=cfg.b)

    mpc_results = {}
    scp_results = {}

    for T in T_vals:
        for j in j_vals:
            common_kwargs = dict(
                j=j, T=T, r_cost=r_cost,
                x0_lo=x0_lo, x0_hi=x0_hi,
                u_bound=u_bound, sys=sys, verbose=False,
                time_limit=cfg.time_limit,
            )

            # True MPC suboptimality
            print(f"=== KKT suboptimality: j={j}, T={T} ===")
            mpc_prob = BilinearMaxStateNorm(**common_kwargs)
            status_m, t_m = mpc_prob.solve()
            sol_m = mpc_prob.solution_dict()
            mpc_results[(T, j)] = {'subopt': sol_m['subopt'], 'status': status_m, 'time': t_m}
            print(f"  subopt={sol_m['subopt']}")

            # SCP suboptimality (2 iterations)
            print(f"=== SCP suboptimality: j={j}, T={T} ===")
            scp_prob = BilinearSCPSubopt(**common_kwargs, n_iters=1)
            status_s, t_s = scp_prob.solve()
            sol_s = scp_prob.solution_dict()
            scp_results[(T, j)] = {'subopt': sol_s['subopt'], 'status': status_s, 'time': t_s}
            print(f"  subopt={sol_s['subopt']}")
            # import pdb; pdb.set_trace()

    # Summary
    print("\n=== Summary ===")
    for T in T_vals:
        for j in j_vals:
            m = mpc_results[(T, j)]
            s = scp_results[(T, j)]
            m_str = f"{m['subopt']:.4f}" if m['subopt'] is not None else "N/A"
            s_str = f"{s['subopt']:.4f}" if s['subopt'] is not None else "N/A"
            print(f"  T={T}, j={j}: MPC={m_str}  SCP={s_str}")

    j_axis = list(j_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ti, T in enumerate(T_vals):
        mpc_vals = [
            mpc_results[(T, j)]['subopt'] if mpc_results[(T, j)]['subopt'] is not None
            else float('nan') for j in j_vals
        ]
        scp_vals = [
            scp_results[(T, j)]['subopt'] if scp_results[(T, j)]['subopt'] is not None
            else float('nan') for j in j_vals
        ]
        color = colors[ti % len(colors)]
        ax.plot(j_axis[1:], mpc_vals[1:],
                marker=markers[ti % len(markers)], linewidth=2, color=color,
                linestyle='-', label=f'MPC T={T}')
        ax.plot(j_axis[1:], scp_vals[1:],
                marker=markers[ti % len(markers)], linewidth=2, color=color,
                linestyle='--', label=f'SCP T={T}')
    ax.set_xlabel('step $j$')
    ax.set_yscale('log')
    ax.set_ylabel('closed-loop suboptimality')
    # ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('bilinear_closed-loop_suboptimality.pdf', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bilinear system constants
#   x+[0] = 0.9*x[0] + u + 0.2*u*x[0]
#   x+[1] = 0.7*x[1] + 0.5*x[0]
# SCP linearizes at (x_k, u=0):
#   A = [[0.9, 0], [0.5, 0.7]]   (constant)
#   B(x_k) = [[1 + 0.2*x_k[0]], [0]]   (depends on x_k[0])
#
# Steady-state gain x[1]/x[0] = 0.5/(1-0.7) = 1.67, within x_feas=2.
# ---------------------------------------------------------------------------

# A_LIN = np.array([[0.9, 0.0], [0.5, 0.7]])   # constant A
# A^T = [[0.9, 0.5], [0.0, 0.7]]


def _add_one_scp_iter(M, tag, x_k, x_bar, u_bar, T, r_cost, u_bound, sys):
    """
    Add one SCP iteration KKT block to model M.

    Linearizes at trajectory (x_bar[t], u_bar[t]):
      A[t] = [[a00 + b*u_bar[t], 0], [a10, a11]]
      B[t] = [[1 + b*x_bar[t][0]], [0]]
      c[t] = [-b*u_bar[t]*x_bar[t][0], 0]

    x_bar[t] and u_bar[t] may be Gurobi variables (NonConvex=2).
    Returns (x_m, u_m): full solution trajectory dicts.
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)
    a00, a10, a11, b = sys.a00, sys.a10, sys.a11, sys.b

    x_m = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xm_{tag}_{t}")
           for t in range(T + 1)}
    u_m = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"um_{tag}_{t}")
           for t in range(T)}
    lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lam_{tag}_{t}")
           for t in range(T + 1)}
    mu_up = {t: M.addVar(lb=0.0, name=f"mu_up_{tag}_{t}") for t in range(T)}
    mu_lo = {t: M.addVar(lb=0.0, name=f"mu_lo_{tag}_{t}") for t in range(T)}

    for i in range(n_x):
        M.addConstr(x_m[0][i] == x_k[i], name=f"init_{tag}_{i}")

    # Linearized dynamics:
    #   x_m[t+1][0] = (a00 + b*ub)*x_m[t][0] + (1 + b*xb0)*u_m[t][0] - b*ub*xb0
    #   x_m[t+1][1] = a10*x_m[t][0] + a11*x_m[t][1]
    for t in range(T):
        xb0 = x_bar[t][0]
        ub  = u_bar[t]
        M.addConstr(
            x_m[t+1][0] == a00*x_m[t][0] + b*ub*x_m[t][0]
            + u_m[t][0] + b*xb0*u_m[t][0]
            - b*ub*xb0,
            name=f"dyn0_{tag}_{t}")
        M.addConstr(
            x_m[t+1][1] == a10*x_m[t][0] + a11*x_m[t][1],
            name=f"dyn1_{tag}_{t}")

    # KKT terminal costate (no state constraints)
    for i in range(n_x):
        M.addConstr(lam[T][i] == Q[i, i]*x_m[T][i],
                    name=f"term_{tag}_{i}")

    # KKT backward costate: A[t]^T = [[a00 + b*ub, a10], [0, a11]]
    for t in range(T - 1, 0, -1):
        ub = u_bar[t]
        M.addConstr(
            lam[t][0] == Q[0, 0]*x_m[t][0]
            + a00*lam[t+1][0] + b*ub*lam[t+1][0]
            + a10*lam[t+1][1],
            name=f"back0_{tag}_{t}")
        M.addConstr(
            lam[t][1] == Q[1, 1]*x_m[t][1]
            + a11*lam[t+1][1],
            name=f"back1_{tag}_{t}")

    # KKT stationarity: r_cost*u + (1 + b*xb0)*lam[t+1][0] + mu_up - mu_lo = 0
    for t in range(T):
        xb0 = x_bar[t][0]
        M.addConstr(
            r_cost*u_m[t][0]
            + lam[t+1][0] + b*xb0*lam[t+1][0]
            + mu_up[t] - mu_lo[t] == 0,
            name=f"stat_{tag}_{t}")

    for t in range(T):
        M.addConstr(mu_up[t]*(u_m[t][0] - u_bound) == 0, name=f"cup_{tag}_{t}")
        M.addConstr(mu_lo[t]*(-u_bound - u_m[t][0]) == 0, name=f"clo_{tag}_{t}")

    return x_m, u_m


def _add_scp_chain(M, j, T, r_cost, x0_lo, x0_hi, u_bound, sys, n_iters=2):
    """
    Build j steps of SCP MPC chain using n_iters SCP iterations per step.

    Iter 1 linearizes at (x_k, u_prev); iter i>1 linearizes at the full
    solution trajectory from the previous iteration.
    Chain propagates true nonlinear dynamics.

    Returns (x_chain, u_chain).
    """
    a00, a10, a11, b = sys.a00, sys.a10, sys.a11, sys.b
    x_chain = {}
    u_chain = {}
    n_x = 2
    x_chain[0] = M.addVars(n_x, lb=x0_lo, ub=x0_hi, name="xc_0")

    for k in range(j):
        x_k = x_chain[k]
        u_prev = 0.0 if k == 0 else u_chain[k - 1]
        x_bar = {t: x_k for t in range(T)}
        u_bar = {t: u_prev for t in range(T)}

        x_sol, u_sol = None, None
        for it in range(n_iters):
            x_sol, u_sol = _add_one_scp_iter(
                M, tag=f"{k}_{it}", x_k=x_k,
                x_bar=x_bar, u_bar=u_bar,
                T=T, r_cost=r_cost, u_bound=u_bound, sys=sys)
            x_bar = {t: x_sol[t] for t in range(T)}
            u_bar = {t: u_sol[t][0] for t in range(T)}

        u_k = u_sol[0][0]
        u_chain[k] = u_k
        x_chain[k+1] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xc_{k+1}")
        M.addConstr(
            x_chain[k+1][0] == a00*x_k[0] + u_k + b*u_k*x_k[0],
            name=f"chain_dyn0_{k}")
        M.addConstr(
            x_chain[k+1][1] == a11*x_k[1] + a10*x_k[0],
            name=f"chain_dyn1_{k}")

    return x_chain, u_chain




# ---------------------------------------------------------------------------
# Helper: true (nonlinear) MPC chain for BilinearMaxStateNorm
# ---------------------------------------------------------------------------

def _add_true_mpc_chain(M, j, T, r_cost, x0_lo, x0_hi, u_bound, sys):
    """
    Add j steps of true (nonlinear) MPC chain to model M.
    Uses exact bilinear dynamics and nonlinear KKT conditions.
    Returns (x_chain, u_chain).
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)
    R = np.eye(n_u) * r_cost
    a00, a10, a11, b = sys.a00, sys.a10, sys.a11, sys.b

    x_chain = {}
    u_chain = {}
    x_chain[0] = M.addVars(n_x, lb=x0_lo, ub=x0_hi, name="xc_0")

    for t in range(1, j + 1):
        x_chain[t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xc_{t}")

    for k in range(j):
        u_k = M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uc_{k}")
        u_chain[k] = u_k

        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xmc_{k}_{t}")
                 for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"umc_{k}_{t}")
                 for t in range(T)}
        mu_up = {t: M.addVars(n_u, lb=0.0, name=f"muc_up_{k}_{t}") for t in range(T)}
        mu_lo = {t: M.addVars(n_u, lb=0.0, name=f"muc_lo_{k}_{t}") for t in range(T)}
        lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamc_{k}_{t}")
               for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x_chain[k][i], name=f"mc_init_{k}_{i}")

        # True nonlinear dynamics
        for t in range(T):
            M.addConstr(
                x_mpc[t+1][0] == a00*x_mpc[t][0] + u_mpc[t][0] + b*u_mpc[t][0]*x_mpc[t][0],
                name=f"cdyn0_{k}_{t}")
            M.addConstr(
                x_mpc[t+1][1] == a11*x_mpc[t][1] + a10*x_mpc[t][0],
                name=f"cdyn1_{k}_{t}")

        # KKT terminal costate (no state constraints)
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i] * x_mpc[T][i], name=f"tc_{k}_{i}")

        # Backward costate: df/dx^T * lam, df/du = (1 + b*x[0])
        for t in range(T - 1, 0, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0] * x_mpc[t][0]
                + a00*lam[t+1][0] + b*u_mpc[t][0]*lam[t+1][0]
                + a10*lam[t+1][1],
                name=f"cs0c_{k}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1] * x_mpc[t][1] + a11*lam[t+1][1],
                name=f"cs1c_{k}_{t}")

        for t in range(T):
            M.addConstr(
                R[0, 0] * u_mpc[t][0]
                + lam[t+1][0] + b*x_mpc[t][0]*lam[t+1][0]
                + mu_up[t][0] - mu_lo[t][0] == 0,
                name=f"statc_{k}_{t}")
            M.addConstr(mu_up[t][0] * (u_mpc[t][0] - u_bound) == 0, name=f"cup_c_{k}_{t}")
            M.addConstr(mu_lo[t][0] * (-u_bound - u_mpc[t][0]) == 0, name=f"clo_c_{k}_{t}")

        M.addConstr(u_k[0] == u_mpc[0][0], name=f"link_{k}")

        M.addConstr(
            x_chain[k+1][0] == a00*x_chain[k][0] + u_k[0] + b*u_k[0]*x_chain[k][0],
            name=f"idyn0_{k}")
        M.addConstr(
            x_chain[k+1][1] == a11*x_chain[k][1] + a10*x_chain[k][0],
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
                 x0_lo=0.0, x0_hi=1.0,
                 u_bound=1.0, sys=None, verbose=True, time_limit=None):
        self.j = j
        n_x, n_u = 2, 1
        a00, a10, a11, b = sys.a00, sys.a10, sys.a11, sys.b
        self._subs = []

        M = gp.Model("bilinear_kkt_subopt")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit

        xc, uc = _add_true_mpc_chain(M, j, T, r_cost, x0_lo, x0_hi, u_bound, sys)

        # Unconstrained optimal chain (same x_0, free dynamics)
        x_opt = {0: xc[0]}
        u_opt = {}
        for t in range(1, j + 1):
            x_opt[t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xopt_{t}")
        for k in range(j):
            u_opt[k] = M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uopt_{k}")
            M.addConstr(
                x_opt[k+1][0] == a00*x_opt[k][0] + u_opt[k][0] + b*u_opt[k][0]*x_opt[k][0],
                name=f"idyn0_{k}")
            M.addConstr(
                x_opt[k+1][1] == a11*x_opt[k][1] + a10*x_opt[k][0],
                name=f"idyn1_{k}")

        obj = 0
        for jj in range(j + 1):
            for ii in range(n_x):
                obj += xc[jj][ii] * xc[jj][ii] - x_opt[jj][ii] * x_opt[jj][ii]
        for jj in range(j):
            obj += r_cost * (uc[jj][0] * uc[jj][0] - u_opt[jj][0] * u_opt[jj][0])
        M.Params.MIPGapAbs = 0.0001
        M.setObjective(obj, GRB.MAXIMIZE)
        self._subs.append((M, xc, uc, x_opt, u_opt, 0, 1))

    def solve(self):
        t_total = 0.0
        for M, *_ in self._subs:
            M.optimize()
            t_total += M.Runtime
        return [M.Status for M, *_ in self._subs], t_total

    def solution_dict(self):
        for M, xc, uc, x_opt, u_opt, i, sign in self._subs:
            if M.SolCount == 0:
                return {'subopt': None}
            return {
                'subopt': M.ObjVal,
                'x_kkt':  {k: [xc[k][i].X for i in range(2)] for k in range(self.j + 1)},
                'u_kkt':  {k: uc[k][0].X for k in range(self.j)},
                'x_opt':  {k: [x_opt[k][i].X for i in range(2)] for k in range(self.j + 1)},
                'u_opt':  {k: u_opt[k][0].X for k in range(self.j)},
            }
        return {'subopt': None}


# ---------------------------------------------------------------------------
# SCP closed-loop suboptimality: max J_scp(x_0) - J_opt(x_0)
# ---------------------------------------------------------------------------

class BilinearSCPSubopt:
    """
    Outer problem: max_{x_0 in X_0} [J_scp(x_0) - J_opt(x_0)]

    J_scp:  j-step cost under 1-iter SCP policy (linearize at x_k, u_lin=0)
    J_opt:  j-step cost under optimal (true nonlinear MPC) policy

    Both policies propagate true nonlinear dynamics.
    The SCP inner QP is convex (linearized dynamics, fixed x_k) and its KKT
    conditions are linear in inner variables — bilinear terms arise only
    from x_k being an outer variable.
    """

    def __init__(self, j=0, T=5, r_cost=0.1,
                 x0_lo=-1.0, x0_hi=1.0,
                 u_bound=1.0, sys=None, verbose=True, time_limit=None, n_iters=2):
        self.j = j
        n_x, n_u = 2, 1
        a00, a10, a11, b = sys.a00, sys.a10, sys.a11, sys.b

        M = gp.Model("bilinear_scp_subopt")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        M.Params.MIPGapAbs = 0.0001
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        x_scp, u_scp = _add_scp_chain(M, j, T, r_cost, x0_lo, x0_hi,
                                       u_bound, sys, n_iters=n_iters)

        # Unconstrained optimal chain (same x_0, free dynamics)
        x_opt = {0: x_scp[0]}
        u_opt = {}
        for t in range(1, j + 1):
            x_opt[t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xopt_{t}")
        for k in range(j):
            u_opt[k] = M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uopt_{k}")
            M.addConstr(
                x_opt[k+1][0] == a00*x_opt[k][0] + u_opt[k][0] + b*u_opt[k][0]*x_opt[k][0],
                name=f"opt_dyn0_{k}")
            M.addConstr(
                x_opt[k+1][1] == a11*x_opt[k][1] + a10*x_opt[k][0],
                name=f"opt_dyn1_{k}")

        # --- Objective: J_scp - J_opt ---
        # J = sum_{t=0}^{j} ||x_t||^2 + r_cost * sum_{t=0}^{j-1} u_t^2
        obj = gp.QuadExpr()
        for t in range(j + 1):
            for i in range(n_x):
                obj += x_scp[t][i] * x_scp[t][i] - x_opt[t][i] * x_opt[t][i]
        for k in range(j):
            obj += r_cost * (u_scp[k] * u_scp[k] - u_opt[k][0] * u_opt[k][0])

        M.setObjective(obj, GRB.MAXIMIZE)
        self.x_scp = x_scp
        self.u_scp = u_scp
        self.x_opt = x_opt
        self.u_opt = u_opt

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {'subopt': None}
        return {
            'subopt': self.model.ObjVal,
            'x_scp':  {k: [self.x_scp[k][i].X for i in range(2)] for k in range(self.j + 1)},
            'u_scp':  {k: self.u_scp[k].X for k in range(self.j)},
            'x_opt':  {k: [self.x_opt[k][i].X for i in range(2)] for k in range(self.j + 1)},
            'u_opt':  {k: self.u_opt[k][0].X for k in range(self.j)},
        }
