import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# System constants
#   x[t+1] = A*x[t] + B*u[t] + alpha*||x[t]||^2 * ones(2)
#   A = [[1,1],[0,1]],  B = [[0.5],[1]],  alpha = 0.1
#
# Linearized at x_bar[t]:
#   A_lin[t] = A + 2*alpha * ones(2)*x_bar[t]^T
#            = [[1+2a*xb0, 1+2a*xb1],
#               [  2a*xb0, 1+2a*xb1]]
#   B_lin    = B  (constant)
#   c[t]     = -alpha*||x_bar[t]||^2 * ones(2)   (affine offset)
#
# A_lin^T * lam:
#   lam[t][0] = Q*x[t][0] + (1+2a*xb0)*lam[t+1][0] + 2a*xb0*lam[t+1][1]
#   lam[t][1] = Q*x[t][1] + (1+2a*xb1)*lam[t+1][0] + (1+2a*xb1)*lam[t+1][1]
#
# Stationarity (unconstrained): r*u + 0.5*lam[t+1][0] + lam[t+1][1] = 0
# ---------------------------------------------------------------------------

ALPHA  = 0.1
A_MAT  = np.array([[1.0, 1.0], [0.0, 1.0]])
B_MAT  = np.array([[0.5], [1.0]])

cmap    = plt.cm.Set1
colors  = cmap.colors
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

FONT_SIZE = 33
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
})
plt.rc("xtick", labelsize=FONT_SIZE, labelcolor="black")
plt.rc("ytick", labelsize=FONT_SIZE, labelcolor="black")


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(cfg):
    """
    For each (T, j), compute max_{x_0 in X_0} [J_policy(x_0) - J_opt(x_0)]
    for iLQR with N=1, 2, ... iterations (unconstrained, no state/control bounds).
    """
    j_vals           = list(cfg.j_vals)
    T_vals           = list(cfg.T_vals)
    n_ilqr_iters_list = list(cfg.n_ilqr_iters)
    r                = cfg.r
    x0_lo            = list(cfg.x0_mins)
    x0_hi            = list(cfg.x0_maxes)

    # kkt_results  = {}
    ilqr_results = {n: {} for n in n_ilqr_iters_list}

    for T in T_vals:
        for j in j_vals:
            common = dict(j=j, T=T, r=r, x0_lo=x0_lo, x0_hi=x0_hi,
                          verbose=False, time_limit=cfg.time_limit)

            # print(f"=== KKT j={j} T={T} ===")
            # kkt = NLDIKKTSubopt(**common)
            # st, tt = kkt.solve()
            # sd = kkt.solution_dict()
            # kkt_results[(T, j)] = {'subopt': sd['subopt'], 'time': tt}
            # print(f"  subopt={sd['subopt']}")

            for n in n_ilqr_iters_list:
                print(f"=== iLQR N={n} j={j} T={T} ===")
                prob = NLDIILQRSubopt(**common, n_iters=n)
                st, tt = prob.solve()
                sd = prob.solution_dict()
                ilqr_results[n][(T, j)] = {'subopt': sd['subopt'], 'time': tt}
                print(f"  subopt={sd['subopt']}")
                if sd['subopt'] is not None:
                    print("  iLQR trajectory:")
                    for k in range(j + 1):
                        x = sd['x_ilqr'][k]
                        u_str = f"  u={sd['u_ilqr'][k]:.4f}" if k < j else ""
                        print(f"    x{k}=[{x[0]:.4f}, {x[1]:.4f}]{u_str}")
                    print("  Optimal trajectory:")
                    for k in range(j + 1):
                        x = sd['x_opt'][k]
                        u_str = f"  u={sd['u_opt'][k]:.4f}" if k < j else ""
                        print(f"    x{k}=[{x[0]:.4f}, {x[1]:.4f}]{u_str}")

    for T in T_vals:
        fig, ax = plt.subplots(figsize=(8, 5))
        j_axis = [j for j in j_vals if j > 0]

        for ci, n in enumerate(n_ilqr_iters_list):
            vals = [ilqr_results[n][(T, j)]['subopt'] or float('nan') for j in j_axis]
            ax.plot(j_axis, vals,
                    marker=markers[(ci + 1) % len(markers)],
                    linewidth=2, color=colors[ci],
                    linestyle='--', label=f'iLQR $N={n}$')

        ax.set_xlabel('step $j$')
        ax.set_yscale('log')
        ax.set_ylabel('closed-loop suboptimality')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f'nldi_ilqr_subopt_T{T}.pdf', bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# True dynamics helper
# ---------------------------------------------------------------------------

def _add_true_dyn(M, x_cur, u_cur, x_nxt, tag):
    """
    Add true nonlinear DI dynamics: x_nxt = A*x_cur + B*u_cur + alpha*||x_cur||^2*ones.
    Creates an auxiliary variable xnorm2 = ||x_cur||^2 to keep constraints quadratic.
    Returns xnorm2.
    """
    xnorm2 = M.addVar(lb=0.0, name=f"xn2_{tag}")
    M.addConstr(xnorm2 == x_cur[0]*x_cur[0] + x_cur[1]*x_cur[1],
                name=f"xn2def_{tag}")
    M.addConstr(x_nxt[0] == x_cur[0] + x_cur[1] + 0.5*u_cur + ALPHA*xnorm2,
                name=f"dyn0_{tag}")
    M.addConstr(x_nxt[1] == x_cur[1] + u_cur + ALPHA*xnorm2,
                name=f"dyn1_{tag}")
    return xnorm2


# ---------------------------------------------------------------------------
# KKT chain — true nonlinear MPC, unconstrained
# ---------------------------------------------------------------------------

def _add_kkt_chain(M, j, T, r, x0_lo, x0_hi):
    """
    j-step closed-loop true MPC KKT chain (unconstrained).

    At each step k, solves T-horizon MPC from x_chain[k] via exact nonlinear KKT:
      - True dynamics (with ||x||^2 auxiliary)
      - Terminal costate: lam[T] = Q*x[T]
      - Backward costate using true Jacobian A_true^T (bilinear: x_mpc*lam)
      - Stationarity: r*u + 0.5*lam[t+1][0] + lam[t+1][1] = 0

    Closed-loop chain propagates true nonlinear dynamics.
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)

    x_chain = {}
    u_chain = {}
    x_chain[0] = M.addVars(n_x,
                            lb={i: x0_lo[i] for i in range(n_x)},
                            ub={i: x0_hi[i] for i in range(n_x)},
                            name="xc_0")
    for t in range(1, j + 1):
        x_chain[t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xc_{t}")

    for k in range(j):
        u_k   = M.addVars(n_u, lb=-GRB.INFINITY, name=f"uc_{k}")
        u_chain[k] = u_k

        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xmc_{k}_{t}")
                 for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-GRB.INFINITY, name=f"umc_{k}_{t}")
                 for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamc_{k}_{t}")
                 for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x_chain[k][i], name=f"mc_init_{k}_{i}")

        for t in range(T):
            _add_true_dyn(M, x_mpc[t], u_mpc[t][0], x_mpc[t+1], f"mpc_{k}_{t}")

        # Terminal costate
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i]*x_mpc[T][i], name=f"tc_{k}_{i}")

        # Backward costate using true Jacobian A_true[t]^T (evaluated at x_mpc[t])
        # A_true^T * lam[t+1]:
        #   lam[t][0] += (1+2a*x[0])*lam[t+1][0] + 2a*x[0]*lam[t+1][1]
        #   lam[t][1] += (1+2a*x[1])*lam[t+1][0] + (1+2a*x[1])*lam[t+1][1]
        for t in range(T - 1, 0, -1):
            x0 = x_mpc[t][0]
            x1 = x_mpc[t][1]
            M.addConstr(
                lam[t][0] == Q[0, 0]*x0
                + lam[t+1][0]
                + 2*ALPHA*x0*lam[t+1][0] + 2*ALPHA*x0*lam[t+1][1],
                name=f"cs0_{k}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1]*x1
                + lam[t+1][0] + lam[t+1][1]
                + 2*ALPHA*x1*lam[t+1][0] + 2*ALPHA*x1*lam[t+1][1],
                name=f"cs1_{k}_{t}")

        # Stationarity (unconstrained): r*u + B^T*lam[t+1] = 0
        for t in range(T):
            M.addConstr(
                r*u_mpc[t][0] + 0.5*lam[t+1][0] + lam[t+1][1] == 0,
                name=f"stat_{k}_{t}")

        M.addConstr(u_k[0] == u_mpc[0][0], name=f"link_{k}")

        # Closed-loop chain step
        _add_true_dyn(M, x_chain[k], u_k[0], x_chain[k+1], f"chain_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# iLQR iterations
# ---------------------------------------------------------------------------

def _add_ilqr_iters_di(M, tag, x_k, n_iters, T, r):
    """
    n_iters of iLQR at state x_k (Gurobi vars) for the nonlinear DI.

    Each iteration:
      1. Forward pass: x_bar[t+1] = f(x_bar[t], u_bar[t])  (true dynamics)
         u_bar = 0 for iter 0; u_bar = u_lin from previous iter otherwise.
      2. Linearize at x_bar[t]:
           A_lin[t] = A + 2*alpha * ones(2)*x_bar[t]^T
           B_lin    = B (constant)
           c[t]     = -alpha*||x_bar[t]||^2 * ones
      3. Solve unconstrained LQR on linearized+offset system from x_k:
           min  sum_t x^T Q x + r*u^2
           s.t. x[t+1] = A_lin[t]*x[t] + B*u[t] + c[t],  x[0] = x_k
         KKT stationarity: r*u + 0.5*lam[t+1][0] + lam[t+1][1] = 0

    Bilinear products (xb*x_lin, xb*lam) handled by NonConvex=2.
    Returns (u_lin, x_lin) from last iteration.
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)

    u_lin_prev = None

    for it in range(n_iters):
        it_tag = f"{tag}_it{it}"

        # ------------------------------------------------------------------
        # 1. Forward pass: x_bar[t+1] = f(x_bar[t], u_bar[t])
        # ------------------------------------------------------------------
        x_bar    = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xbar_{it_tag}_{t}")
                    for t in range(T + 1)}
        xnorm2_bar = {}  # ||x_bar[t]||^2 auxiliary

        for i in range(n_x):
            M.addConstr(x_bar[0][i] == x_k[i], name=f"fp_init_{it_tag}_{i}")

        for t in range(T):
            xnorm2_bar[t] = M.addVar(lb=0.0, name=f"xbn2_{it_tag}_{t}")
            M.addConstr(
                xnorm2_bar[t] == x_bar[t][0]*x_bar[t][0] + x_bar[t][1]*x_bar[t][1],
                name=f"xbn2def_{it_tag}_{t}")
            if it == 0:
                # u_bar = 0
                M.addConstr(
                    x_bar[t+1][0] == x_bar[t][0] + x_bar[t][1] + ALPHA*xnorm2_bar[t],
                    name=f"fp_dyn0_{it_tag}_{t}")
                M.addConstr(
                    x_bar[t+1][1] == x_bar[t][1] + ALPHA*xnorm2_bar[t],
                    name=f"fp_dyn1_{it_tag}_{t}")
            else:
                ub = u_lin_prev[t][0]
                M.addConstr(
                    x_bar[t+1][0] == x_bar[t][0] + x_bar[t][1] + 0.5*ub + ALPHA*xnorm2_bar[t],
                    name=f"fp_dyn0_{it_tag}_{t}")
                M.addConstr(
                    x_bar[t+1][1] == x_bar[t][1] + ub + ALPHA*xnorm2_bar[t],
                    name=f"fp_dyn1_{it_tag}_{t}")

        # ------------------------------------------------------------------
        # 2/3. Unconstrained LQR via KKT on linearized + offset system
        # ------------------------------------------------------------------
        x_lin = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xlin_{it_tag}_{t}")
                 for t in range(T + 1)}
        u_lin = {t: M.addVars(n_u, lb=-GRB.INFINITY, name=f"ulin_{it_tag}_{t}")
                 for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"laml_{it_tag}_{t}")
                 for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(x_lin[0][i] == x_k[i], name=f"lqr_init_{it_tag}_{i}")

        for t in range(T):
            xb0 = x_bar[t][0]
            xb1 = x_bar[t][1]
            xn2 = xnorm2_bar[t]  # = xb0^2 + xb1^2

            # x_lin[t+1] = A_lin[t]*x_lin[t] + B*u_lin[t] + c[t]
            # c[t] = -alpha*||x_bar[t]||^2 * ones = -alpha*xn2
            M.addConstr(
                x_lin[t+1][0] == x_lin[t][0] + x_lin[t][1]
                + 2*ALPHA*xb0*x_lin[t][0] + 2*ALPHA*xb1*x_lin[t][1]
                + 0.5*u_lin[t][0]
                - ALPHA*xn2,
                name=f"lqr_dyn0_{it_tag}_{t}")
            M.addConstr(
                x_lin[t+1][1] == 2*ALPHA*xb0*x_lin[t][0]
                + x_lin[t][1] + 2*ALPHA*xb1*x_lin[t][1]
                + u_lin[t][0]
                - ALPHA*xn2,
                name=f"lqr_dyn1_{it_tag}_{t}")

        # Terminal costate
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i]*x_lin[T][i],
                        name=f"lqr_term_{it_tag}_{i}")

        # Backward costate: A_lin[t]^T * lam[t+1]
        for t in range(T - 1, -1, -1):
            xb0 = x_bar[t][0]
            xb1 = x_bar[t][1]
            M.addConstr(
                lam[t][0] == Q[0, 0]*x_lin[t][0]
                + lam[t+1][0]
                + 2*ALPHA*xb0*lam[t+1][0] + 2*ALPHA*xb0*lam[t+1][1],
                name=f"lqr_back0_{it_tag}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1]*x_lin[t][1]
                + lam[t+1][0] + lam[t+1][1]
                + 2*ALPHA*xb1*lam[t+1][0] + 2*ALPHA*xb1*lam[t+1][1],
                name=f"lqr_back1_{it_tag}_{t}")

        # Stationarity (unconstrained): r*u + B^T*lam[t+1] = 0
        for t in range(T):
            M.addConstr(
                r*u_lin[t][0] + 0.5*lam[t+1][0] + lam[t+1][1] == 0,
                name=f"lqr_stat_{it_tag}_{t}")

        u_lin_prev = u_lin

    return u_lin_prev, x_lin


# ---------------------------------------------------------------------------
# iLQR closed-loop chain
# ---------------------------------------------------------------------------

def _add_ilqr_chain(M, j, T, r, x0_lo, x0_hi, n_iters):
    """
    j-step closed-loop iLQR chain (n_iters iterations per step).
    Applied control: first control from last LQR iteration.
    Closed-loop dynamics: true nonlinear.
    """
    n_x = 2
    x_chain = {}
    u_chain = {}
    x_chain[0] = M.addVars(n_x,
                            lb={i: x0_lo[i] for i in range(n_x)},
                            ub={i: x0_hi[i] for i in range(n_x)},
                            name="xc_0")

    for k in range(j):
        x_k = x_chain[k]
        u_lin, _ = _add_ilqr_iters_di(M, tag=f"ilqr_{k}", x_k=x_k,
                                        n_iters=n_iters, T=T, r=r)
        u_k = u_lin[0][0]
        u_chain[k] = u_k

        x_chain[k+1] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xc_{k+1}")
        _add_true_dyn(M, x_k, u_k, x_chain[k+1], f"chain_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# Unconstrained optimal chain
# ---------------------------------------------------------------------------

def _add_opt_chain(M, j, x0_var, r):
    """
    Unconstrained optimal j-step chain (free u_opt).
    Outer maximisation minimises J_opt over u_opt automatically.
    """
    n_x, n_u = 2, 1
    x_opt = {0: x0_var}
    u_opt = {}
    for t in range(1, j + 1):
        x_opt[t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xopt_{t}")
    for k in range(j):
        u_opt[k] = M.addVars(n_u, lb=-GRB.INFINITY, name=f"uopt_{k}")
        x_opt[k+1] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xopt_{k+1}")
        _add_true_dyn(M, x_opt[k], u_opt[k][0], x_opt[k+1], f"opt_{k}")
    return x_opt, u_opt


# ---------------------------------------------------------------------------
# Verification problems
# ---------------------------------------------------------------------------

class NLDIKKTSubopt:
    """max_{x_0} J_kkt(x_0) - J_opt(x_0)  (unconstrained true MPC)."""

    def __init__(self, j=1, T=5, r=0.0, x0_lo=None, x0_hi=None,
                 verbose=True, time_limit=None):
        self.j = j
        n_x, n_u = 2, 1
        x0_lo = x0_lo or [-0.5, -0.5]
        x0_hi = x0_hi or [ 0.5,  0.5]

        M = gp.Model("nldi_kkt_subopt")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        M.Params.MIPGapAbs = 0.001
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        xc, uc = _add_kkt_chain(M, j, T, r, x0_lo, x0_hi)
        x_opt, u_opt = _add_opt_chain(M, j, xc[0], r)

        obj = gp.QuadExpr()
        for t in range(j + 1):
            for i in range(n_x):
                obj += xc[t][i]*xc[t][i] - x_opt[t][i]*x_opt[t][i]
        if r > 0:
            for k in range(j):
                obj += r*(uc[k][0]*uc[k][0] - u_opt[k][0]*u_opt[k][0])

        M.setObjective(obj, GRB.MAXIMIZE)
        self.xc = xc
        self.uc = uc
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
            'x_kkt': {k: [self.xc[k][i].X for i in range(2)] for k in range(self.j + 1)},
            'u_kkt': {k: self.uc[k][0].X for k in range(self.j)},
            'x_opt': {k: [self.x_opt[k][i].X for i in range(2)] for k in range(self.j + 1)},
            'u_opt': {k: self.u_opt[k][0].X for k in range(self.j)},
        }


class NLDIILQRSubopt:
    """max_{x_0} J_ilqr(x_0) - J_opt(x_0)  (N iLQR iterations, unconstrained)."""

    def __init__(self, j=1, T=5, r=0.0, x0_lo=None, x0_hi=None,
                 verbose=True, time_limit=None, n_iters=1):
        self.j = j
        n_x, n_u = 2, 1
        x0_lo = x0_lo or [-0.5, -0.5]
        x0_hi = x0_hi or [ 0.5,  0.5]

        M = gp.Model("nldi_ilqr_subopt")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        M.Params.MIPGapAbs = 0.001
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        x_ilqr, u_ilqr = _add_ilqr_chain(M, j, T, r, x0_lo, x0_hi, n_iters)
        x_opt,  u_opt   = _add_opt_chain(M, j, x_ilqr[0], r)

        obj = gp.QuadExpr()
        for t in range(j + 1):
            for i in range(n_x):
                obj += x_ilqr[t][i]*x_ilqr[t][i] - x_opt[t][i]*x_opt[t][i]
        if r > 0:
            for k in range(j):
                obj += r*(u_ilqr[k]*u_ilqr[k] - u_opt[k][0]*u_opt[k][0])

        M.setObjective(obj, GRB.MAXIMIZE)
        self.x_ilqr = x_ilqr
        self.u_ilqr = u_ilqr
        self.x_opt  = x_opt
        self.u_opt  = u_opt

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {'subopt': None}
        return {
            'subopt':  self.model.ObjVal,
            'x_ilqr': {k: [self.x_ilqr[k][i].X for i in range(2)] for k in range(self.j + 1)},
            'u_ilqr': {k: self.u_ilqr[k].X for k in range(self.j)},
            'x_opt':  {k: [self.x_opt[k][i].X for i in range(2)] for k in range(self.j + 1)},
            'u_opt':  {k: self.u_opt[k][0].X for k in range(self.j)},
        }
