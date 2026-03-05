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
    T_max = cfg.T_max
    x_lo = cfg.x_mins
    x_hi = cfg.x_maxes
    r = getattr(cfg, 'r', 0.1)
    u_max = getattr(cfg, 'u_max', 10.0)

    T_vals = list(range(5, T_max + 1))
    obj_vals = []

    for T in T_vals:
        ver = NonlinearDoubleIntegratorVerify(
            T=T, r=r, x_lo=x_lo, x_hi=x_hi, u_max=u_max, verbose=True
        )
        # ver = NonlinearDoubleIntegratorVerifyFree(
        #     T=T, r=r, x_lo=x_lo, x_hi=x_hi, u_max=u_max, verbose=True
        # )
        # ver = NonlinearDoubleIntegratorVerifyFreeSimplified(
        #     T=T, r=r, x_lo=x_lo, x_hi=x_hi, u_max=u_max, verbose=True
        # )
        status, elapsed = ver.solve()
        sol = ver.solution_dict()
        obj = sol['obj'] if sol['obj'] is not None else float('nan')
        print(f"T={T}, status={status}, obj={obj:.6f}, time={elapsed:.3f}s")
        obj_vals.append(obj)
        # import pdb
        # pdb.set_trace()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_vals, obj_vals, marker='o', linewidth=2)
    ax.set_xlabel('Horizon $T$')
    # ax.set_yscale('log')
    ax.set_ylabel(r'$\|x_{\mathrm{kkt}}[T]\|^2 - \|x_{\mathrm{opt}}[T]\|^2$')
    ax.set_title('Worst-case KKT suboptimality vs.\ horizon')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('nonlinear_di_suboptimality.pdf', bbox_inches='tight')
    print("Plot saved to nonlinear_di_suboptimality.pdf")
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

    def __init__(self, T=5, r=0.1, x_lo=-4.0, x_hi=4.0, u_max=10.0, verbose=True):
        n_x = 2
        n_u = 1
        alpha = 0.025   # nonlinear gain: f(x) = alpha * ||x||^2 * ones(2)

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
        # M.Params.FeasibilityTol = 1e-9
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
        self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        def v2dict(vs):
            return {k: vs[k].X for k in vs}

        if self.model.SolCount == 0:
            return {"obj": None}

        return {
            "obj": self.model.ObjVal,
            "x0": {i: self.x0[i].X for i in range(self.n_x)},
            "x_kkt": {t: v2dict(self.x_kkt_traj[t]) for t in range(self.T + 1)},
            "u_kkt": {t: v2dict(self.u_kkt_traj[t]) for t in range(self.T)},
            "lam_kkt": {t: v2dict(self.lam_kkt[t]) for t in range(self.T + 1)},
            "nu_up": {t: v2dict(self.nu_up[t]) for t in range(self.T)},
            "nu_lo": {t: v2dict(self.nu_lo[t]) for t in range(self.T)},
            "x_opt": {t: v2dict(self.x_opt_traj[t]) for t in range(self.T + 1)},
            "u_opt": {t: v2dict(self.u_opt_traj[t]) for t in range(self.T)},
        }


class NonlinearDoubleIntegratorVerifyFree:
    """
    Same worst-case verification as NonlinearDoubleIntegratorVerify but with
    NO control constraints.  Because u is unconstrained, the KKT conditions
    collapse to pure stationarity (no complementarity multipliers):

        R u_t + B^T p_{t+1} = 0

    which — when r > 0 — uniquely determines u_t = -(1/r) B^T p_{t+1}.
    When r = 0 it becomes the constraint B^T p_{t+1} = 0.

    Dynamics, costate recursion, and objective are identical to the
    constrained variant.
    """

    def __init__(self, T=5, r=0.1, x_lo=-4.0, x_hi=4.0, u_max=None, verbose=True):
        n_x = 2
        n_u = 1
        alpha = 0.025   # f(x) = alpha * ||x||^2 * ones(2)

        A = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        B = np.array([[0.5],
                      [1.0]])

        Q = np.eye(n_x)
        R_mat = r * np.eye(n_u)

        M = gp.Model("nonlinear_di_verify_free")
        M.Params.OutputFlag = 1 if verbose else 0
        # M.Params.FeasibilityTol = 1e-9
        # M.Params.OptimalityTol = 1e-9
        # M.Params.MIPGap = 1e-9
        M.Params.NumericFocus = 3
        M.Params.NonConvex = 2
        # M.Params.Cuts = 2
        self.model = M
        self.T = T
        self.n_x = n_x
        self.n_u = n_u

        # Shared initial state
        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")
        self.x0 = x0

        # KKT trajectory — controls unconstrained
        x_kkt = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"xk_{t}")
                 for t in range(T + 1)}
        u_kkt = {t: M.addVars(n_u, lb=-GRB.INFINITY, name=f"uk_{t}")
                 for t in range(T)}
        lam_kkt = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lk_{t}")
                   for t in range(T + 1)}

        # Optimal trajectory — controls unconstrained, no KKT conditions
        x_opt = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"xo_{t}")
                 for t in range(T + 1)}
        u_opt = {t: M.addVars(n_u, lb=-GRB.INFINITY, name=f"uo_{t}")
                 for t in range(T)}

        # Shared initial condition
        for i in range(n_x):
            M.addConstr(x_kkt[0][i] == x0[i], name=f"ic_kkt_{i}")
            M.addConstr(x_opt[0][i] == x0[i], name=f"ic_opt_{i}")

        # Dynamics: x_{t+1} = A x_t + B u_t + alpha * ||x_t||^2 * ones
        xnorm2_kkt = {}
        xnorm2_opt = {}
        for t in range(T):
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

        # ---------------------------------------------------------------
        # KKT conditions — unconstrained u, so no complementarity.
        #
        # Terminal costate:  p_T = Q x_T
        # Costate recursion: p_t = Q x_t + A^T p_{t+1}
        #                         + 2*alpha*(p_{t+1}[0]+p_{t+1}[1])*x_t
        # Stationarity:      R u_t + B^T p_{t+1} = 0
        # ---------------------------------------------------------------

        # Terminal costate
        for i in range(n_x):
            M.addConstr(
                lam_kkt[T][i] == Q[i, i] * x_kkt[T][i],
                name=f"lam_term_{i}"
            )

        # Costate backward recursion
        for t in range(T - 1, -1, -1):
            for i in range(n_x):
                AT_lam = gp.quicksum(A[j, i] * lam_kkt[t + 1][j]
                                     for j in range(n_x))
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

        # Control stationarity (no multipliers): R u_t + B^T p_{t+1} = 0
        for t in range(T):
            for j in range(n_u):
                BT_lam = gp.quicksum(B[i, j] * lam_kkt[t + 1][i]
                                     for i in range(n_x))
                M.addConstr(
                    R_mat[j, j] * u_kkt[t][j] + BT_lam == 0,
                    name=f"station_{t}_{j}"
                )

        # Store references
        self.x_kkt_traj = x_kkt
        self.u_kkt_traj = u_kkt
        self.lam_kkt = lam_kkt
        self.x_opt_traj = x_opt
        self.u_opt_traj = u_opt
        self.x_kkt = x_kkt[T]
        self.x_opt = x_opt[T]

        # Objective: maximize J_kkt - J_opt  (full trajectory cost)
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
        self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        def v2dict(vs):
            return {k: vs[k].X for k in vs}

        if self.model.SolCount == 0:
            return {"obj": None}

        return {
            "obj": self.model.ObjVal,
            "x0": {i: self.x0[i].X for i in range(self.n_x)},
            "x_kkt": {t: v2dict(self.x_kkt_traj[t]) for t in range(self.T + 1)},
            "u_kkt": {t: v2dict(self.u_kkt_traj[t]) for t in range(self.T)},
            "lam_kkt": {t: v2dict(self.lam_kkt[t]) for t in range(self.T + 1)},
            "x_opt": {t: v2dict(self.x_opt_traj[t]) for t in range(self.T + 1)},
            "u_opt": {t: v2dict(self.u_opt_traj[t]) for t in range(self.T)},
        }


class NonlinearDoubleIntegratorVerifyFreeSimplified:
    """
    Faster variant: replaces the full bilinear costate backward recursion with
    a LINEAR first-order condition derived directly from grad_u J(u; x_0) = 0.

    For cost  J = sum_{t=0}^{T} x_t^T Q x_t + r * sum_{t=0}^{T-1} u_t^2,
    differentiating w.r.t. u_t and using the linearized sensitivity
    ∂x_s/∂u_t = A^{s-t-1} B  gives:

        ∂J/∂u_t = 2 r u_t + 2 sum_{s=t+1}^{T} B^T (A^T)^{s-t-1} Q x_s = 0

    i.e.   r u_t[j] + sum_{s=t+1}^{T} C[t,s][j,:] @ x_kkt[s] = 0

    where  C[t,s] = B^T (A^T)^{s-t-1} Q  are FIXED scalars precomputed offline.

    This constraint is LINEAR in the Gurobi variables {u_kkt[t], x_kkt[s]},
    so it eliminates every lambda*x bilinear product from the costate recursion
    while still coupling u to the actual (nonlinear) trajectory states.

    The TRUE nonlinear dynamics are used for both trajectories; the only
    nonlinear constraints are the quadratic  ||x_t||^2  auxiliary definitions.
    """

    def __init__(self, T=5, r=0.1, x_lo=-4.0, x_hi=4.0, u_max=None, verbose=True):
        n_x = 2
        n_u = 1
        alpha = 0.025

        A = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        B = np.array([[0.5],
                      [1.0]])
        Q = np.eye(n_x)
        R_mat = r * np.eye(n_u)

        # Precompute sensitivity coefficients C[t,s] = B^T (A^T)^{s-t-1} Q
        # shape (n_u, n_x) — fixed scalars, computed outside Gurobi
        C = {}
        for t in range(T):
            for s in range(t + 1, T + 1):
                AT_pow = np.linalg.matrix_power(A.T, s - t - 1)
                C[t, s] = B.T @ AT_pow @ Q  # (n_u, n_x)

        M = gp.Model("nonlinear_di_verify_simplified")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.OptimalityTol = 1e-9
        M.Params.MIPGap = 1e-9
        M.Params.NumericFocus = 3
        M.Params.NonConvex = 2
        self.model = M
        self.T = T
        self.n_x = n_x
        self.n_u = n_u

        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")
        self.x0 = x0

        x_kkt = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"xk_{t}")
                 for t in range(T + 1)}
        u_kkt = {t: M.addVars(n_u, lb=-GRB.INFINITY, name=f"uk_{t}")
                 for t in range(T)}

        x_opt = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"xo_{t}")
                 for t in range(T + 1)}
        u_opt = {t: M.addVars(n_u, lb=-GRB.INFINITY, name=f"uo_{t}")
                 for t in range(T)}

        # Shared initial condition
        for i in range(n_x):
            M.addConstr(x_kkt[0][i] == x0[i], name=f"ic_kkt_{i}")
            M.addConstr(x_opt[0][i] == x0[i], name=f"ic_opt_{i}")

        # Dynamics: x_{t+1} = A x_t + B u_t + alpha * ||x_t||^2 * ones
        xnorm2_kkt = {}
        xnorm2_opt = {}
        for t in range(T):
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

        # ---------------------------------------------------------------
        # FOC: grad_u J(u; x_0) = 0
        #   r * u_kkt[t][j] + sum_{s=t+1}^{T} C[t,s][j,:] @ x_kkt[s] = 0
        # C[t,s] = B^T (A^T)^{s-t-1} Q are precomputed fixed scalars.
        # This is linear in the Gurobi variables {u_kkt[t], x_kkt[s]}.
        # ---------------------------------------------------------------
        for t in range(T):
            for j in range(n_u):
                sens_sum = gp.quicksum(
                    float(C[t, s][j, i]) * x_kkt[s][i]
                    for s in range(t + 1, T + 1)
                    for i in range(n_x)
                )
                M.addConstr(
                    R_mat[j, j] * u_kkt[t][j] + sens_sum == 0,
                    name=f"foc_{t}_{j}"
                )

        # Store references
        self.x_kkt_traj = x_kkt
        self.u_kkt_traj = u_kkt
        self.x_opt_traj = x_opt
        self.u_opt_traj = u_opt
        self.x_kkt = x_kkt[T]
        self.x_opt = x_opt[T]

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
        self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        def v2dict(vs):
            return {k: vs[k].X for k in vs}

        if self.model.SolCount == 0:
            return {"obj": None}

        return {
            "obj": self.model.ObjVal,
            "x0": {i: self.x0[i].X for i in range(self.n_x)},
            "x_kkt": {t: v2dict(self.x_kkt_traj[t]) for t in range(self.T + 1)},
            "u_kkt": {t: v2dict(self.u_kkt_traj[t]) for t in range(self.T)},
            "x_opt": {t: v2dict(self.x_opt_traj[t]) for t in range(self.T + 1)},
            "u_opt": {t: v2dict(self.u_opt_traj[t]) for t in range(self.T)},
        }
