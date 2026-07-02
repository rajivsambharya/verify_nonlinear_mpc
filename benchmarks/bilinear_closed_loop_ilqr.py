import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class BilinearSystem:
    """
    x+[0] = a00*x[0] + u + b*u*x[0]          (bilinear in x[0], u)
    x+[1] = a11*x[1] + a10*x[0] + b_2*x[1]^2 (nonlinear in x[1])

    Linearized at (x_bar[t], u_bar[t]):
      A[t] = [[a00 + b*u_bar[t],           0              ],
              [a10,              a11 + 2*b_2*x_bar[t][1]  ]]
      B[t] = [[1 + b*x_bar[t][0]], [0]]
      c[t] = [-b*u_bar[t]*x_bar[t][0],  -b_2*x_bar[t][1]^2]
    """
    a00: float
    a10: float
    a11: float
    b:   float
    b_2: float = 0.0


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
    for:
      - KKT: true nonlinear MPC (unconstrained)
      - iLQR N: standard iLQR with N iterations (unconstrained)
    """
    j_vals          = list(cfg.j_vals)
    T_vals          = list(cfg.T_vals)
    n_iters_list    = list(cfg.n_ilqr_iters)
    r_cost          = cfg.r_cost
    x0_lo           = list(cfg.x0_mins)
    x0_hi           = list(cfg.x0_maxes)
    sys             = BilinearSystem(a00=cfg.a00, a10=cfg.a10, a11=cfg.a11, b=cfg.b, b_2=cfg.b_2)

    kkt_results  = {}
    ilqr_results = {n: {} for n in n_iters_list}

    for T in T_vals:
        for j in j_vals:
            common = dict(j=j, T=T, r_cost=r_cost,
                          x0_lo=x0_lo, x0_hi=x0_hi, sys=sys,
                          verbose=False, time_limit=cfg.time_limit)

            print(f"=== KKT j={j} T={T} ===")
            kkt = BilinearKKTSubopt(**common)
            st, tt = kkt.solve()
            sd = kkt.solution_dict()
            kkt_results[(T, j)] = {'subopt': sd['subopt'], 'time': tt}
            print(f"  subopt={sd['subopt']}")

            for n in n_iters_list:
                print(f"=== iLQR N={n} j={j} T={T} ===")
                prob = BilinearILQRSubopt(**common, n_iters=n)
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

    # ------------------------------------------------------------------
    # Plot: one figure per T, curves over j
    # ------------------------------------------------------------------
    for T in T_vals:
        fig, ax = plt.subplots(figsize=(8, 5))
        j_axis = [j for j in j_vals if j > 0]

        kkt_vals = [kkt_results[(T, j)]['subopt'] or float('nan') for j in j_axis]
        ax.plot(j_axis, kkt_vals,
                marker=markers[0], linewidth=2.5, color='k',
                linestyle='-', label='KKT')

        for ci, n in enumerate(n_iters_list):
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
        fig.savefig(f'bilinear_ilqr_subopt_T{T}.pdf', bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# KKT chain — unconstrained true nonlinear MPC
# ---------------------------------------------------------------------------

def _add_kkt_chain(M, j, T, r_cost, x0_lo, x0_hi, sys):
    """
    j-step closed-loop KKT (true nonlinear MPC) chain, no control bounds.

    KKT stationarity (unconstrained):
      r*u[t] + (1 + b*x[t][0])*lam[t+1][0] = 0
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)
    a00, a10, a11, b, b_2 = sys.a00, sys.a10, sys.a11, sys.b, sys.b_2

    x_chain = {}
    u_chain = {}
    x_chain[0] = M.addVars(n_x,
                            lb={i: x0_lo[i] for i in range(n_x)},
                            ub={i: x0_hi[i] for i in range(n_x)},
                            name="xc_0")

    for t in range(1, j + 1):
        x_chain[t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xc_{t}")

    for k in range(j):
        u_k = M.addVars(n_u, lb=-GRB.INFINITY, name=f"uc_{k}")
        u_chain[k] = u_k

        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xmc_{k}_{t}")
                 for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-GRB.INFINITY, name=f"umc_{k}_{t}")
                 for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamc_{k}_{t}")
                 for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x_chain[k][i], name=f"mc_init_{k}_{i}")

        # True nonlinear dynamics
        for t in range(T):
            M.addConstr(
                x_mpc[t+1][0] == a00*x_mpc[t][0] + u_mpc[t][0] + b*u_mpc[t][0]*x_mpc[t][0],
                name=f"cdyn0_{k}_{t}")
            M.addConstr(
                x_mpc[t+1][1] == a11*x_mpc[t][1] + a10*x_mpc[t][0]
                + b_2*x_mpc[t][1]*x_mpc[t][1],
                name=f"cdyn1_{k}_{t}")

        # Terminal costate
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i]*x_mpc[T][i], name=f"tc_{k}_{i}")

        # Backward costate: A^T = [[a00+b*u, a10], [0, a11+2*b_2*x[1]]]
        for t in range(T - 1, 0, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0]*x_mpc[t][0]
                + a00*lam[t+1][0] + b*u_mpc[t][0]*lam[t+1][0]
                + a10*lam[t+1][1],
                name=f"cs0_{k}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1]*x_mpc[t][1]
                + a11*lam[t+1][1] + 2*b_2*x_mpc[t][1]*lam[t+1][1],
                name=f"cs1_{k}_{t}")

        # Stationarity (unconstrained): r*u + (1+b*x[0])*lam[1][0] = 0
        for t in range(T):
            M.addConstr(
                r_cost*u_mpc[t][0]
                + lam[t+1][0] + b*x_mpc[t][0]*lam[t+1][0] == 0,
                name=f"stat_{k}_{t}")

        M.addConstr(u_k[0] == u_mpc[0][0], name=f"link_{k}")

        M.addConstr(
            x_chain[k+1][0] == a00*x_chain[k][0] + u_k[0] + b*u_k[0]*x_chain[k][0],
            name=f"idyn0_{k}")
        M.addConstr(
            x_chain[k+1][1] == a11*x_chain[k][1] + a10*x_chain[k][0]
            + b_2*x_chain[k][1]*x_chain[k][1],
            name=f"idyn1_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# iLQR helpers
# ---------------------------------------------------------------------------

def _add_ilqr_iters_bilinear(M, tag, x_k, n_iters, T, r_cost, sys):
    """
    Add n_iters of standard iLQR at state x_k (Gurobi vars).

    Each iteration:
      1. Forward pass: x_bar[t+1] = f(x_bar[t], u_bar[t])  (true nonlinear)
      2. Linearize: A[t] = [[a00+b*u_bar[t], 0], [a10, a11]]
                    B[t] = [[1+b*x_bar[t][0]], [0]]
      3. Solve unconstrained LQR on linearized system from x_k (KKT conditions)

    Bilinear products (u_bar*x_bar, u_bar*lam, x_bar*u_lin, x_bar*lam)
    are handled by NonConvex=2.

    Returns (u_lin, x_lin) from the last iteration.
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)
    # Q[0,0] = 0.0   # only penalize x[1]
    a00, a10, a11, b, b_2 = sys.a00, sys.a10, sys.a11, sys.b, sys.b_2

    u_lin_prev = None   # None  →  u_bar = 0 for first iteration
    x_lin_prev = None

    for it in range(n_iters):
        it_tag = f"{tag}_it{it}"

        # ------------------------------------------------------------------
        # 1. Forward pass: x_bar[t+1] = f(x_bar[t], u_bar[t])
        # ------------------------------------------------------------------
        x_bar = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xbar_{it_tag}_{t}")
                 for t in range(T + 1)}
        for i in range(n_x):
            M.addConstr(x_bar[0][i] == x_k[i], name=f"fp_init_{it_tag}_{i}")

        for t in range(T):
            if it == 0:
                # u_bar = 0  →  linear dynamics
                M.addConstr(
                    x_bar[t+1][0] == a00 * x_bar[t][0],
                    name=f"fp_dyn0_{it_tag}_{t}")
            else:
                ub = u_lin_prev[t][0]   # Gurobi var from previous LQR
                M.addConstr(
                    x_bar[t+1][0] == a00*x_bar[t][0] + ub + b*ub*x_bar[t][0],
                    name=f"fp_dyn0_{it_tag}_{t}")
            M.addConstr(
                x_bar[t+1][1] == a11*x_bar[t][1] + a10*x_bar[t][0]
                + b_2*x_bar[t][1]*x_bar[t][1],
                name=f"fp_dyn1_{it_tag}_{t}")

        # ------------------------------------------------------------------
        # 2/3. Unconstrained LQR via KKT on linearized system from x_k
        #      min  sum_t x^T Q x + r*u^2
        #      s.t. x[t+1] = A[t]x[t] + B[t]u[t] + c[t],  x[0] = x_k
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
            xb0 = x_bar[t][0]   # Gurobi var (forward pass)
            if it == 0:
                # A[t][0,0] = a00, c[t] = 0
                # x_lin[t+1][0] = a00*x_lin[t][0] + (1+b*xb0)*u_lin[t][0]
                M.addConstr(
                    x_lin[t+1][0] == a00*x_lin[t][0]
                    + u_lin[t][0] + b*xb0*u_lin[t][0],
                    name=f"lqr_dyn0_{it_tag}_{t}")
            else:
                ub = u_lin_prev[t][0]
                # x[t+1][0] = (a00+b*ub)*x[t][0] + (1+b*xb0)*u[t] - b*ub*xb0
                M.addConstr(
                    x_lin[t+1][0] == a00*x_lin[t][0] + b*ub*x_lin[t][0]
                    + u_lin[t][0] + b*xb0*u_lin[t][0]
                    - b*ub*xb0,
                    name=f"lqr_dyn0_{it_tag}_{t}")
            xb1 = x_bar[t][1]
            M.addConstr(
                x_lin[t+1][1] == a10*x_lin[t][0]
                + a11*x_lin[t][1] + 2*b_2*xb1*x_lin[t][1]
                - b_2*xb1*xb1,
                name=f"lqr_dyn1_{it_tag}_{t}")

        # Terminal costate
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i]*x_lin[T][i],
                        name=f"lqr_term_{it_tag}_{i}")

        # Backward costate: A[t]^T = [[a00+b*ub, a10], [0, a11]]
        for t in range(T - 1, -1, -1):
            if it == 0:
                M.addConstr(
                    lam[t][0] == Q[0, 0]*x_lin[t][0]
                    + a00*lam[t+1][0] + a10*lam[t+1][1],
                    name=f"lqr_back0_{it_tag}_{t}")
            else:
                ub = u_lin_prev[t][0]
                M.addConstr(
                    lam[t][0] == Q[0, 0]*x_lin[t][0]
                    + a00*lam[t+1][0] + b*ub*lam[t+1][0] + a10*lam[t+1][1],
                    name=f"lqr_back0_{it_tag}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1]*x_lin[t][1]
                + a11*lam[t+1][1] + 2*b_2*x_bar[t][1]*lam[t+1][1],
                name=f"lqr_back1_{it_tag}_{t}")

        # Stationarity (unconstrained): r*u + (1+b*xb0)*lam[t+1][0] = 0
        for t in range(T):
            xb0 = x_bar[t][0]
            M.addConstr(
                r_cost*u_lin[t][0] + lam[t+1][0] + b*xb0*lam[t+1][0] == 0,
                name=f"lqr_stat_{it_tag}_{t}")

        u_lin_prev = u_lin
        x_lin_prev = x_lin

    return u_lin_prev, x_lin_prev


def _add_ilqr_chain(M, j, T, r_cost, x0_lo, x0_hi, sys, n_iters):
    """
    j-step closed-loop iLQR chain (n_iters iterations per step).
    Applied control at step k: first control of the last LQR iteration.
    Closed-loop propagates true nonlinear dynamics.
    Returns (x_chain, u_chain).
    """
    n_x = 2
    a00, a10, a11, b, b_2 = sys.a00, sys.a10, sys.a11, sys.b, sys.b_2

    x_chain = {}
    u_chain = {}
    x_chain[0] = M.addVars(n_x,
                            lb={i: x0_lo[i] for i in range(n_x)},
                            ub={i: x0_hi[i] for i in range(n_x)},
                            name="xc_0")

    for k in range(j):
        x_k = x_chain[k]

        u_lin, _ = _add_ilqr_iters_bilinear(
            M, tag=f"ilqr_{k}", x_k=x_k,
            n_iters=n_iters, T=T, r_cost=r_cost, sys=sys)

        u_k = u_lin[0][0]          # first control of last LQR iteration
        u_chain[k] = u_k

        x_chain[k+1] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xc_{k+1}")
        M.addConstr(
            x_chain[k+1][0] == a00*x_k[0] + u_k + b*u_k*x_k[0],
            name=f"chain_dyn0_{k}")
        M.addConstr(
            x_chain[k+1][1] == a11*x_k[1] + a10*x_k[0] + b_2*x_k[1]*x_k[1],
            name=f"chain_dyn1_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# Verification problems
# ---------------------------------------------------------------------------

def _add_opt_chain(M, j, x0_var, r_cost, sys):
    """
    Unconstrained optimal j-step chain (free u_opt, same x_0).
    Optimality comes for free: the outer maximisation minimises J_opt over u_opt.
    """
    n_x, n_u = 2, 1
    a00, a10, a11, b, b_2 = sys.a00, sys.a10, sys.a11, sys.b, sys.b_2

    x_opt = {0: x0_var}
    u_opt = {}
    for t in range(1, j + 1):
        x_opt[t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xopt_{t}")
    for k in range(j):
        u_opt[k] = M.addVars(n_u, lb=-GRB.INFINITY, name=f"uopt_{k}")
        M.addConstr(
            x_opt[k+1][0] == a00*x_opt[k][0] + u_opt[k][0] + b*u_opt[k][0]*x_opt[k][0],
            name=f"opt_dyn0_{k}")
        M.addConstr(
            x_opt[k+1][1] == a11*x_opt[k][1] + a10*x_opt[k][0] + b_2*x_opt[k][1]*x_opt[k][1],
            name=f"opt_dyn1_{k}")
    return x_opt, u_opt


class BilinearKKTSubopt:
    """max_{x_0} J_kkt(x_0) - J_opt(x_0)  (unconstrained controls)."""

    def __init__(self, j=1, T=5, r_cost=0.1,
                 x0_lo=0.0, x0_hi=1.0,
                 sys=None, verbose=True, time_limit=None):
        self.j = j
        n_x, n_u = 2, 1

        M = gp.Model("bilinear_kkt_subopt")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        M.Params.MIPGapAbs = 0.001
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        xc, uc = _add_kkt_chain(M, j, T, r_cost, x0_lo, x0_hi, sys)
        x_opt, u_opt = _add_opt_chain(M, j, xc[0], r_cost, sys)

        obj = gp.QuadExpr()
        for t in range(j + 1):
            for i in range(n_x):
                obj += xc[t][i]*xc[t][i] - x_opt[t][i]*x_opt[t][i]
        for k in range(j):
            obj += r_cost*(uc[k][0]*uc[k][0] - u_opt[k][0]*u_opt[k][0])

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


class BilinearILQRSubopt:
    """max_{x_0} J_ilqr(x_0) - J_opt(x_0)  (unconstrained controls, N iLQR iters)."""

    def __init__(self, j=1, T=5, r_cost=0.1,
                 x0_lo=0.0, x0_hi=1.0,
                 sys=None, verbose=True, time_limit=None, n_iters=1):
        self.j = j
        n_x, n_u = 2, 1

        M = gp.Model("bilinear_ilqr_subopt")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        M.Params.MIPGapAbs = 0.001
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        x_ilqr, u_ilqr = _add_ilqr_chain(M, j, T, r_cost, x0_lo, x0_hi, sys, n_iters)
        x_opt,  u_opt   = _add_opt_chain(M, j, x_ilqr[0], r_cost, sys)

        obj = gp.QuadExpr()
        for t in range(j + 1):
            for i in range(n_x):
                obj += x_ilqr[t][i]*x_ilqr[t][i] - x_opt[t][i]*x_opt[t][i]
            # obj += x_ilqr[t][1]*x_ilqr[t][1] - x_opt[t][1]*x_opt[t][1]
        if r_cost > 0:
            for k in range(j):
                # u_ilqr[k] is a single Gurobi Var (first control of last LQR iter)
                obj += r_cost*(u_ilqr[k]*u_ilqr[k] - u_opt[k][0]*u_opt[k][0])

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
            'subopt': self.model.ObjVal,
            'x_ilqr': {k: [self.x_ilqr[k][i].X for i in range(2)] for k in range(self.j + 1)},
            'u_ilqr': {k: self.u_ilqr[k].X for k in range(self.j)},
            'x_opt':  {k: [self.x_opt[k][i].X for i in range(2)] for k in range(self.j + 1)},
            'u_opt':  {k: self.u_opt[k][0].X for k in range(self.j)},
        }
