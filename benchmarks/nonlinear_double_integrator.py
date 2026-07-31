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
    # T_max = cfg.T_max
    x_lo = cfg.x_mins
    x_hi = cfg.x_maxes
    r = getattr(cfg, 'r', 0.0)
    u_max = getattr(cfg, 'u_max', 10.0)
    alpha = getattr(cfg, 'alpha', 0.1)
    obj_tol = getattr(cfg, 'obj_tol', 1e-4)

    # T_vals = list(range(5, T_max + 1))
    T_vals = cfg.T_vals
    obj_vals = []
    time_vals = []

    for T in T_vals:
        ver = NonlinearDoubleIntegratorVerify(
            T=T, r=r, x_lo=x_lo, x_hi=x_hi, u_max=u_max, alpha=alpha,
            obj_tol=obj_tol, verbose=True
        )

        status, elapsed = ver.solve()
        sol = ver.solution_dict()
        obj = sol['obj'] if sol['obj'] is not None else float('nan')
        print(f"T={T}, status={status}, obj={obj:.6f}, time={elapsed:.3f}s")
        obj_vals.append(obj)
        time_vals.append(elapsed)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_vals, obj_vals, marker=markers[0], linewidth=2, color=colors[0])
    ax.set_xlabel('horizon $T$')
    # ax.set_yscale('log')
    ax.set_ylabel('worst-case subopt.')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('nonlinear_di_suboptimality.pdf', bbox_inches='tight')
    print("Plot saved to nonlinear_di_suboptimality.pdf")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(T_vals, time_vals, marker=markers[0], linewidth=2, color=colors[0])
    ax2.set_xlabel('horizon $T$')
    ax2.set_ylabel('solve time (seconds)')
    ax2.set_yscale('log')
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig('nonlinear_di_solve_time.pdf', bbox_inches='tight')
    print("Plot saved to nonlinear_di_solve_time.pdf")

    return obj_vals


class NonlinearDoubleIntegratorVerify:
    """
    Finds the worst-case performance gap between any KKT point and a feasible
    (optimal) solution for a T-horizon nonlinear double integrator MPC problem.

    Dynamics: x_{t+1} = A x_t + B u_t + f(x_t)
        A = [[1, 1], [0, 1]]
        B = [[0.5], [1]]
        f(x) = 0.025 * (x^T x) * ones(2)

    KKT trajectory: satisfies first-order optimality conditions for MPC.
    Optimal trajectory: only needs to satisfy dynamics and control constraints.
    Both share the same initial state x_0 (free within [x_lo, x_hi]^2).

    Objective: maximize ||x_kkt[T]||^2 - ||x_opt[T]||^2
    """

    def __init__(self, T=5, r=0.1, x_lo=-4.0, x_hi=4.0, u_max=10.0, alpha=0.1,
                 obj_tol=1e-4, verbose=True):
        n_x = 2
        n_u = 1
        # alpha: nonlinear gain: f(x) = alpha * ||x||^2 * ones(2)
        self.obj_tol = obj_tol

        # Dynamics
        A = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        B = np.array([[0.5],
                      [1.0]])

        # MPC cost (1/2 scaling): (1/2) sum_t x_t^T Q x_t + u_t^T R u_t
        Q = np.eye(n_x)
        R_mat = r * np.eye(n_u)

        # --- Gurobi model ---
        M = gp.Model("nonlinear_di_verify")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 5e-9
        # M.Params.OptimalityTol = 1e-9
        # M.Params.MIPGap = 1e-9
        # M.Params.NumericFocus = 3
        # M.Params.NonConvex = 2   # required for quadratic/bilinear constraints
        self.model = M
        self.T = T
        self.n_x = n_x
        self.n_u = n_u

        # ---------------------------------------------------------------
        # Shared initial state (free within box)
        # ---------------------------------------------------------------
        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")
        self.x0 = x0

        # ---------------------------------------------------------------
        # KKT-point trajectory variables
        # ---------------------------------------------------------------
        x_kkt = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"xk_{t}")
                 for t in range(T + 1)}
        u_kkt = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"uk_{t}")
                 for t in range(T)}
        # Costates (adjoint variables)
        lam_kkt = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lk_{t}")
                   for t in range(T + 1)}
        # Multipliers for box constraints: ν^u (upper), ν^l (lower)
        nu_up = {t: M.addVars(n_u, lb=0.0, name=f"nu_up_{t}") for t in range(T)}
        nu_lo = {t: M.addVars(n_u, lb=0.0, name=f"nu_lo_{t}") for t in range(T)}

        # ---------------------------------------------------------------
        # Feasible (optimal) trajectory variables — no KKT conditions
        # ---------------------------------------------------------------
        x_opt = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"xo_{t}")
                 for t in range(T + 1)}
        u_opt = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"uo_{t}")
                 for t in range(T)}

        # ---------------------------------------------------------------
        # Shared initial condition
        # ---------------------------------------------------------------
        for i in range(n_x):
            M.addConstr(x_kkt[0][i] == x0[i], name=f"ic_kkt_{i}")
            M.addConstr(x_opt[0][i] == x0[i], name=f"ic_opt_{i}")

        # ---------------------------------------------------------------
        # Dynamics: x_{t+1} = A x_t + B u_t + alpha * ||x_t||^2 * ones
        # ---------------------------------------------------------------
        xnorm2_kkt = {}   # auxiliary: ||x_kkt[t]||^2
        xnorm2_opt = {}   # auxiliary: ||x_opt[t]||^2

        for t in range(T):
            # KKT dynamics
            xnorm2_kkt[t] = M.addVar(lb=0.0, name=f"xnk_{t}")
            M.addConstr(
                xnorm2_kkt[t] == gp.quicksum(x_kkt[t][i] * x_kkt[t][i]
                                             for i in range(n_x)),
                name=f"xnk_def_{t}"
            )
            for i in range(n_x):
                rhs = (gp.quicksum(A[i, j] * x_kkt[t][j] for j in range(n_x))
                       + gp.quicksum(B[i, j] * u_kkt[t][j] for j in range(n_u))
                       + alpha * xnorm2_kkt[t])
                M.addConstr(x_kkt[t + 1][i] == rhs, name=f"dynk_{t}_{i}")

            # Optimal dynamics
            xnorm2_opt[t] = M.addVar(lb=0.0, name=f"xno_{t}")
            M.addConstr(
                xnorm2_opt[t] == gp.quicksum(x_opt[t][i] * x_opt[t][i]
                                             for i in range(n_x)),
                name=f"xno_def_{t}"
            )
            for i in range(n_x):
                rhs = (gp.quicksum(A[i, j] * x_opt[t][j] for j in range(n_x))
                       + gp.quicksum(B[i, j] * u_opt[t][j] for j in range(n_u))
                       + alpha * xnorm2_opt[t])
                M.addConstr(x_opt[t + 1][i] == rhs, name=f"dyno_{t}_{i}")

        # Terminal costate
        for i in range(n_x):
            M.addConstr(
                lam_kkt[T][i] == Q[i, i] * x_kkt[T][i],
                name=f"lam_term_{i}"
            )

        # Costate backward recursion (t = T-1, ..., 0)
        # Bilinear products p_{t+1}[j] * x_kkt[t][i] require auxiliary vars.
        for t in range(T - 1, -1, -1):
            for i in range(n_x):
                # A^T p_{t+1} component
                AT_lam = gp.quicksum(A[j, i] * lam_kkt[t + 1][j]
                                     for j in range(n_x))
                # Bilinear: sum_j p_{t+1}[j] * x_kkt[t][i]
                bl = []
                for j in range(n_x):
                    aux = M.addVar(lb=-GRB.INFINITY, name=f"bl_{t}_{i}_{j}")
                    M.addConstr(
                        aux == lam_kkt[t + 1][j] * x_kkt[t][i],
                        name=f"bl_def_{t}_{i}_{j}"
                    )
                    bl.append(aux)
                M.addConstr(
                    lam_kkt[t][i] == (Q[i, i] * x_kkt[t][i]
                                      + AT_lam
                                      + 2 * alpha * gp.quicksum(bl)),
                    name=f"costate_{t}_{i}"
                )

        # Control stationarity: R u_t + B^T p_{t+1} + nu_up - nu_lo = 0
        for t in range(T):
            for j in range(n_u):
                BT_lam = gp.quicksum(B[i, j] * lam_kkt[t + 1][i]
                                     for i in range(n_x))
                M.addConstr(
                    R_mat[j, j] * u_kkt[t][j] + BT_lam
                    + nu_up[t][j] - nu_lo[t][j] == 0,
                    name=f"station_{t}_{j}"
                )

        # Complementarity (bilinear)
        for t in range(T):
            for j in range(n_u):
                M.addConstr(
                    nu_up[t][j] * (u_max - u_kkt[t][j]) == 0,
                    name=f"comp_up_{t}_{j}"
                )
                M.addConstr(
                    nu_lo[t][j] * (u_max + u_kkt[t][j]) == 0,
                    name=f"comp_lo_{t}_{j}"
                )

        # ---------------------------------------------------------------
        # Store variable references
        # ---------------------------------------------------------------
        self.x_kkt_traj = x_kkt
        self.u_kkt_traj = u_kkt
        self.lam_kkt = lam_kkt
        self.nu_up = nu_up
        self.nu_lo = nu_lo
        self.x_opt_traj = x_opt
        self.u_opt_traj = u_opt

        # Keep terminal-state refs for backward compat
        self.x_kkt = x_kkt[T]
        self.x_opt = x_opt[T]

        # ---------------------------------------------------------------
        # Worst-case objective: maximize J_kkt - J_opt
        #   J = sum_{t=0}^{T-1} (x_t^T Q x_t + u_t^T R u_t) + x_T^T Q x_T
        # The KKT conditions encode stationarity for this exact cost, so
        # J_kkt - J_opt is the true suboptimality gap of the KKT point.
        # ---------------------------------------------------------------
        V_kkt = (
            gp.quicksum(Q[i, i] * x_kkt[t][i] * x_kkt[t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R_mat[j, j] * u_kkt[t][j] * u_kkt[t][j]
                          for t in range(T) for j in range(n_u))
        )
        V_opt = (
            gp.quicksum(Q[i, i] * x_opt[t][i] * x_opt[t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R_mat[j, j] * u_opt[t][j] * u_opt[t][j]
                          for t in range(T) for j in range(n_u))
        )

        self.orig_objective = V_kkt - V_opt
        M.setObjective(self.orig_objective, GRB.MAXIMIZE)

    def solve(self):
        def _callback(model, where):
            if where == GRB.Callback.MIP:
                obj_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                if obj_bnd < self.obj_tol:
                    model.terminate()

        self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
        # self.model.optimize(_callback)
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        def v2dict(vs):
            return {k: vs[k].X for k in vs}

        if self.model.SolCount == 0:
            return {"obj": None}

        return {
            "obj": self.model.ObjBound,
            "x0": {i: self.x0[i].X for i in range(self.n_x)},
            "x_kkt": {t: v2dict(self.x_kkt_traj[t]) for t in range(self.T + 1)},
            "u_kkt": {t: v2dict(self.u_kkt_traj[t]) for t in range(self.T)},
            "lam_kkt": {t: v2dict(self.lam_kkt[t]) for t in range(self.T + 1)},
            "nu_up": {t: v2dict(self.nu_up[t]) for t in range(self.T)},
            "nu_lo": {t: v2dict(self.nu_lo[t]) for t in range(self.T)},
            "x_opt": {t: v2dict(self.x_opt_traj[t]) for t in range(self.T + 1)},
            "u_opt": {t: v2dict(self.u_opt_traj[t]) for t in range(self.T)},
        }
