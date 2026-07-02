import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Pendulum (cartpole) dynamics — discrete-time, forward Euler
#   x[0] = theta,   x[1] = thetadot
#   x[t+1][0] = x[t][0] + dt * x[t][1]
#   x[t+1][1] = x[t][1] + dt * (g/L * sin(x[t][0]) + u[t] / (m*L^2))
#
# Linearized at (x_bar[t], u_bar[t]):
#   A[t] = [[1,  dt],
#            [dt*g/L*cos_bar[t],  1]]
#   B    = [[0], [b_u]]           b_u = dt/(m*L^2)  (constant)
#   c[t] = [0, dt*g/L*(sin_bar[t] - cos_bar[t]*x_bar[t][0])]
#
# A[t]^T * lam[t+1]:
#   lam[t][0] = Q*x[t][0] + lam[t+1][0] + dt*g/L*cos_bar[t]*lam[t+1][1]
#   lam[t][1] = Q*x[t][1] + dt*lam[t+1][0] + lam[t+1][1]
#
# Stationarity (unconstrained + optional proximal nu):
#   (r + nu)*u[t] - nu*u_bar[t] + b_u*lam[t+1][1] = 0
# ---------------------------------------------------------------------------

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
    For each (T, j), compute max_{x_0 in X_0} [J_ilqr(x_0) - J_opt(x_0)]
    for iLQR with N=1, 2, ... iterations (no state or control constraints).
    """
    j_vals            = list(cfg.j_vals)
    T_vals            = list(cfg.T_vals)
    n_ilqr_iters_list = list(cfg.n_ilqr_iters)
    r                 = cfg.r
    nu                = cfg.nu
    dt                = cfg.dt
    mass              = cfg.mass
    length            = cfg.length
    x0_lo             = list(cfg.x0_mins)
    x0_hi             = list(cfg.x0_maxes)
    g                 = 9.8

    ilqr_results = {n: {} for n in n_ilqr_iters_list}

    for T in T_vals:
        for j in j_vals:
            common = dict(j=j, T=T, r=r, nu=nu, dt=dt,
                          mass=mass, length=length, g=g,
                          x0_lo=x0_lo, x0_hi=x0_hi,
                          verbose=False, time_limit=cfg.time_limit)

            for n in n_ilqr_iters_list:
                print(f"=== iLQR N={n} j={j} T={T} ===")
                prob = CartpoleILQRSubopt(**common, n_iters=n)
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
        fig.savefig(f'cartpole_ilqr_subopt_T{T}.pdf', bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# iLQR iterations (forward pass + linearized LQR via KKT)
# ---------------------------------------------------------------------------

def _add_ilqr_iters(M, tag, x_0, n_iters, T, r, dt, mass, length, g, nu=0.0):
    """
    Add n_iters of iLQR at state x_0 (Gurobi vars), unconstrained controls.

    Each iteration:
      1. Forward pass from x_0 with u_bar (0 for iter 0, u_lin_prev for iter > 0)
      2. Linearize: A[t] = [[1,dt],[dt*g/L*cos_bar[t],1]], B = [[0],[b_u]]
         affine offset c[t][1] = dt*g/L*(sin_bar[t] - cos_bar[t]*x_bar[t][0])
      3. Solve unconstrained LQR via KKT (with optional proximal nu)

    Bilinear products (cos_bar*x_lin, cos_bar*lam, cos_bar*x_bar)
    handled by NonConvex=2.

    Returns (u_lin, x_lin) from the last iteration.
    """
    n_x, n_u = 2, 1
    b_u = dt / (mass * length**2)

    u_lin_prev = None

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
        # 2/3. Unconstrained LQR via KKT on linearized + offset system
        #   x[t+1][0] = x[t][0] + dt*x[t][1]
        #   x[t+1][1] = dt*g/L*cos_bar*x[t][0] + x[t][1] + b_u*u[t]
        #              + dt*g/L*(sin_bar - cos_bar*x_bar[0])   <- affine offset
        # ------------------------------------------------------------------
        x_lin = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"xlin_{it_tag}_{t}")
                 for t in range(T + 1)}
        u_lin = {t: M.addVars(n_u, lb=-GRB.INFINITY, name=f"ulin_{it_tag}_{t}")
                 for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"laml_{it_tag}_{t}")
                 for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(x_lin[0][i] == x_0[i], name=f"lqr_init_{it_tag}_{i}")

        for t in range(T):
            M.addConstr(
                x_lin[t+1][0] == x_lin[t][0] + dt * x_lin[t][1],
                name=f"lqr_dyn0_{it_tag}_{t}")
            M.addConstr(
                x_lin[t+1][1] == dt*g/length * cos_bar[t] * x_lin[t][0]
                + x_lin[t][1] + b_u * u_lin[t][0]
                + dt*g/length * sin_bar[t]
                - dt*g/length * cos_bar[t] * x_bar[t][0],
                name=f"lqr_dyn1_{it_tag}_{t}")

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

        # Stationarity: (r+nu)*u - nu*u_bar + b_u*lam[t+1][1] = 0
        for t in range(T):
            if it == 0:
                M.addConstr(
                    (r + nu)*u_lin[t][0] + b_u*lam[t+1][1] == 0,
                    name=f"lqr_stat_{it_tag}_{t}")
            else:
                M.addConstr(
                    (r + nu)*u_lin[t][0] - nu*u_lin_prev[t][0]
                    + b_u*lam[t+1][1] == 0,
                    name=f"lqr_stat_{it_tag}_{t}")

        u_lin_prev = u_lin

    return u_lin_prev, x_lin


# ---------------------------------------------------------------------------
# True pendulum dynamics helper
# ---------------------------------------------------------------------------

def _add_pendulum_dyn(M, x_cur, u_cur, x_nxt, dt, mass, length, g, tag):
    """
    Add one step of true pendulum dynamics: x_nxt = f(x_cur, u_cur).
    Creates auxiliary sin and tdd variables.
    """
    b_u  = dt / (mass * length**2)
    sin_ = M.addVar(lb=-1, ub=1,      name=f"sin_{tag}")
    tdd_ = M.addVar(lb=-GRB.INFINITY, name=f"tdd_{tag}")
    M.addGenConstrSin(x_cur[0], sin_,                                            name=f"gsin_{tag}")
    M.addConstr(x_nxt[0] == x_cur[0] + dt * x_cur[1],                           name=f"dyn0_{tag}")
    M.addConstr(tdd_ == g/length * sin_ + b_u * u_cur,                           name=f"tdd_{tag}")
    M.addConstr(x_nxt[1] == x_cur[1] + dt * tdd_,                               name=f"dyn1_{tag}")


# ---------------------------------------------------------------------------
# j-step closed-loop iLQR chain
# ---------------------------------------------------------------------------

def _add_ilqr_chain(M, j, T, r, nu, dt, mass, length, g, x0_lo, x0_hi, n_iters):
    """
    j-step closed-loop iLQR chain.
    Applied control at step k: first control from last LQR iteration.
    Propagates true pendulum dynamics.
    Returns (x_chain, u_chain).
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
        u_lin, _ = _add_ilqr_iters(
            M, tag=f"ilqr_{k}", x_0=x_k,
            n_iters=n_iters, T=T, r=r, dt=dt,
            mass=mass, length=length, g=g, nu=nu)

        u_k = u_lin[0][0]
        u_chain[k] = u_k

        x_chain[k+1] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xc_{k+1}")
        _add_pendulum_dyn(M, x_k, u_k, x_chain[k+1], dt, mass, length, g, f"chain_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# Unconstrained optimal chain
# ---------------------------------------------------------------------------

def _add_opt_chain(M, j, x0_var, r, dt, mass, length, g):
    """
    Unconstrained optimal j-step chain.
    u_opt is free; outer maximization minimizes J_opt automatically.
    """
    n_x, n_u = 2, 1
    x_opt = {0: x0_var}
    u_opt = {}
    for k in range(j):
        u_opt[k] = M.addVars(n_u, lb=-GRB.INFINITY, name=f"uopt_{k}")
        x_opt[k+1] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"xopt_{k+1}")
        _add_pendulum_dyn(M, x_opt[k], u_opt[k][0], x_opt[k+1],
                          dt, mass, length, g, f"opt_{k}")
    return x_opt, u_opt


# ---------------------------------------------------------------------------
# Verification problem
# ---------------------------------------------------------------------------

class CartpoleILQRSubopt:
    """max_{x_0 in X_0} J_ilqr(x_0) - J_opt(x_0)  (N iLQR iters, unconstrained)."""

    def __init__(self, j=1, T=5, r=0.0, nu=0.0, dt=0.01,
                 mass=1, length=1, g=9.8,
                 x0_lo=None, x0_hi=None,
                 verbose=True, time_limit=None, n_iters=1):
        self.j = j
        n_x, n_u = 2, 1
        x0_lo = x0_lo or [-1.0, -1.0]
        x0_hi = x0_hi or [ 1.0,  1.0]

        M = gp.Model("cartpole_ilqr_subopt")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        M.Params.MIPGapAbs = 0.001
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        x_ilqr, u_ilqr = _add_ilqr_chain(
            M, j, T, r, nu, dt, mass, length, g, x0_lo, x0_hi, n_iters)
        x_opt, u_opt = _add_opt_chain(M, j, x_ilqr[0], r, dt, mass, length, g)

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
            'x_ilqr': {k: [self.x_ilqr[k][i].X for i in range(2)]
                        for k in range(self.j + 1)},
            'u_ilqr': {k: self.u_ilqr[k].X for k in range(self.j)},
            'x_opt':  {k: [self.x_opt[k][i].X for i in range(2)]
                        for k in range(self.j + 1)},
            'u_opt':  {k: self.u_opt[k][0].X for k in range(self.j)},
        }
