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


# ---------------------------------------------------------------------------
# Four tank system (Raff, Huber, Nagy, Allgower, "Nonlinear Model Predictive
# Control of a Four Tank System: An Experimental Stability Study", 2006).
#
#   x1' = -a1/A1 sqrt(2 g x1) + a3/A1 sqrt(2 g x3) + gamma1/A1 u1
#   x2' = -a2/A2 sqrt(2 g x2) + a4/A2 sqrt(2 g x4) + gamma2/A2 u2
#   x3' = -a3/A3 sqrt(2 g x3)                      + (1-gamma2)/A3 u2
#   x4' = -a4/A4 sqrt(2 g x4)                      + (1-gamma1)/A4 u1
#
# Indices below are 0-based: tank/pump i in the code is tank/pump (i+1) in
# the paper. Table I gives A_i (cross section) and a_i (outlet cross
# section); the setpoint xs, us and box constraints (9) are hardcoded from
# the paper (with u_max = 60 ml/s -> u in [0, 60]).
# ---------------------------------------------------------------------------

A_AREA = np.array([50.27, 50.27, 28.27, 28.27])   # cm^2
A_OUT  = np.array([0.233, 0.242, 0.127, 0.127])   # cm^2
G      = 981.0                                     # cm/s^2
GAMMA  = np.array([0.4, 0.4])

XS = np.array([14.0, 14.0, 14.2, 21.3])            # cm, setpoint
US = np.array([43.4, 35.4])                        # ml/s, setpoint input

# Box constraints (9), in deviation coordinates z = x - xs, v = u - us
ZMIN = 0 *np.array([-6.5, -6.5, -10.7, -16.8])
ZMAX = 0 * np.array([14.0, 14.0, 13.8, 6.7])
# ZMIN = np.array([-6.5, -6.5, -10.7, -16.8])
# ZMAX = np.array([14.0, 14.0, 13.8, 6.7])
# VMIN = np.array([-43.4, -35.4])
# VMAX = np.array([16.6, 24.6])
BB = 600
VMIN = np.array([-BB, -BB])
VMAX = np.array([BB, BB])

X_LO = XS + ZMIN
X_HI = XS + ZMAX
U_LO = US + VMIN
U_HI = US + VMAX

# Constant input matrix B (4x2): x' = ... + B @ u
B_MAT = np.zeros((4, 2))
B_MAT[0, 0] = GAMMA[0] / A_AREA[0]
B_MAT[1, 1] = GAMMA[1] / A_AREA[1]
B_MAT[2, 1] = (1 - GAMMA[1]) / A_AREA[2]
B_MAT[3, 0] = (1 - GAMMA[0]) / A_AREA[3]


def run(cfg):
    T_vals = list(cfg.T_vals)
    r_vals = list(cfg.r_vals)
    dt = cfg.dt
    feas_tol = cfg.feas_tol
    time_limit = cfg.time_limit
    dual_bound = getattr(cfg, 'dual_bound', 50.0)
    x0_nominal = np.array(cfg.x0_nominal) if getattr(cfg, 'x0_nominal', None) is not None \
        else 0.5 * (X_LO + X_HI)

    # -----------------------------------------------------------------
    # Feasibility certificate: for each T, search over xbar in
    # [X_LO, X_HI] for a Farkas certificate that the T-horizon linearized
    # MPC QP (dynamics linearized at xbar + box constraints) is infeasible.
    # farkas_obj >= -feas_tol for all xbar in the box  =>  feasibility
    # is guaranteed for every linearization point in the box (weak duality
    # guarantees farkas_obj >= 0 whenever the QP is feasible; a value below
    # -feas_tol is a genuine Farkas certificate of infeasibility).
    # -----------------------------------------------------------------
    farkas_obj = []
    farkas_time = []
    for T in T_vals:
        ver = FourTankFarkas(T=T, dt=dt, dual_bound=dual_bound, verbose=True,
                              time_limit=time_limit)
        status, elapsed = ver.solve()
        sol = ver.solution_dict()
        obj = sol['farkas_obj']
        certified = obj is not None and obj >= -feas_tol
        tag = 'CERTIFIED FEASIBLE' if certified else f'INFEASIBLE COUNTEREXAMPLE FOUND (obj={obj})'
        print(f"T={T}: farkas_obj={obj}  status={status}  time={elapsed:.3f}s  -> {tag}")
        if not certified and sol.get('xbar') is not None:
            xbar = sol['xbar']
            xbar_vec = [xbar[i] for i in range(4)]
            print(f"  counterexample initial state x0: "
                  f"[{xbar_vec[0]:.4f}, {xbar_vec[1]:.4f}, {xbar_vec[2]:.4f}, {xbar_vec[3]:.4f}] cm")
            Mmat_ce, bbar_ce = _numeric_Mmat_bbar(xbar_vec, dt)
            print(f"  Mmat at counterexample:\n{Mmat_ce}")
            print(f"  bbar at counterexample: {bbar_ce}")
        farkas_obj.append(obj if obj is not None else float('nan'))
        farkas_time.append(elapsed)

    # -----------------------------------------------------------------
    # Nominal MPC cost: a plain convex QP (linearized once at a *fixed*
    # x0_nominal, so no bilinear terms) solved for each (T, r). This does
    # not affect the feasibility certificate above, but shows how the
    # horizon T and the control weight r trade off nominal performance.
    # -----------------------------------------------------------------
    nominal_cost = np.zeros((len(r_vals), len(T_vals)))
    for ri, r in enumerate(r_vals):
        for ti, T in enumerate(T_vals):
            status, obj, elapsed = solve_nominal_mpc(
                T=T, r=r, x0=x0_nominal, dt=dt, verbose=False, time_limit=time_limit)
            print(f"[nominal] r={r}, T={T}: status={status}, obj={obj}, time={elapsed:.3f}s")
            nominal_cost[ri, ti] = obj if obj is not None else float('nan')

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(T_vals, farkas_obj, marker=markers[0], linewidth=2, color=colors[0])
    ax1.axhline(0, color='k', linestyle='--', linewidth=1)
    ax1.set_xlabel('horizon $T$')
    ax1.set_ylabel('Farkas objective')
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig('four_tank_farkas.pdf', bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for ri, r in enumerate(r_vals):
        ax2.plot(T_vals, nominal_cost[ri], marker=markers[ri % len(markers)],
                  linewidth=2, color=colors[ri % len(colors)], label=f'r={r}')
    ax2.set_xlabel('horizon $T$')
    ax2.set_ylabel('nominal MPC cost')
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig('four_tank_nominal_cost.pdf', bbox_inches='tight')
    plt.close(fig2)

    return {
        'T_vals': T_vals, 'r_vals': r_vals,
        'farkas_obj': farkas_obj, 'farkas_time': farkas_time,
        'nominal_cost': nominal_cost,
    }


# ---------------------------------------------------------------------------
# Sqrt linearization via quadratic constraints
#
# sqrt(2 g x) is linearized about xbar as sqrt(2 g x) ~= s + c*(x - xbar),
# where s = sqrt(2 g xbar) (the function value) and c = g/s (the
# derivative, d/dx sqrt(2 g x) = g / sqrt(2 g x)). Both s and c are tied to
# xbar through purely quadratic constraints:
#     s^2 == 2 g xbar      (s >= 0)
#     c*s == g
# which lets xbar be a free decision variable while keeping every
# constraint at most bilinear (degree 2), solvable with Gurobi's
# NonConvex=2 QCQP solver.
#
# Using x0_i = s_i^2/(2g) it also follows that s - c*xbar = s/2, so the
# affine residual term of the linearized dynamics only needs s (not xbar
# directly).
# ---------------------------------------------------------------------------

def _add_sqrt_linearization(M, tag, x_lo, x_hi):
    n = len(x_lo)
    # s_bnd_lo = np.sqrt(2 * G * np.asarray(x_lo, dtype=float))
    # s_bnd_hi = np.sqrt(2 * G * np.asarray(x_hi, dtype=float))
    s_bnd_lo = 0 #-GRB.INFINITY #np.sqrt(2 * G * np.asarray(x_lo, dtype=float))
    s_bnd_hi = GRB.INFINITY #np.sqrt(2 * G * np.asarray(x_hi, dtype=float))
    # c_bnd_lo = G / s_bnd_hi
    # c_bnd_hi = G / s_bnd_lo
    c_bnd_lo = -GRB.INFINITY
    c_bnd_hi = GRB.INFINITY

    xbar = {i: M.addVar(lb=x_lo[i], ub=x_hi[i], name=f"xbar_{tag}_{i}") for i in range(n)}
    # s    = {i: M.addVar(lb=s_bnd_lo[i], ub=s_bnd_hi[i], name=f"s_{tag}_{i}") for i in range(n)}
    # c    = {i: M.addVar(lb=c_bnd_lo[i], ub=c_bnd_hi[i], name=f"c_{tag}_{i}") for i in range(n)}
    s    = {i: M.addVar(lb=s_bnd_lo, ub=s_bnd_hi, name=f"s_{tag}_{i}") for i in range(n)}
    c    = {i: M.addVar(lb=c_bnd_lo, ub=c_bnd_hi, name=f"c_{tag}_{i}") for i in range(n)}
    for i in range(n):
        M.addConstr(s[i] * s[i] == 2 * G * xbar[i], name=f"sqrtdef_{tag}_{i}")
        M.addConstr(c[i] * s[i] == G, name=f"slopedef_{tag}_{i}")
    return xbar, s, c


def _build_Alin_bbar(c, s):
    """
    Linearized continuous-time dynamics x' ~= Alin(c) x + B u + bbar(s),
    where Alin, bbar are built from the sqrt-linearization variables.
    """
    Alin = [[0.0] * 4 for _ in range(4)]
    Alin[0][0] = -A_OUT[0] / A_AREA[0] * c[0]
    Alin[0][2] =  A_OUT[2] / A_AREA[0] * c[2]
    Alin[1][1] = -A_OUT[1] / A_AREA[1] * c[1]
    Alin[1][3] =  A_OUT[3] / A_AREA[1] * c[3]
    Alin[2][2] = -A_OUT[2] / A_AREA[2] * c[2]
    Alin[3][3] = -A_OUT[3] / A_AREA[3] * c[3]

    bbar = [0.0] * 4
    bbar[0] = -A_OUT[0] / (2 * A_AREA[0]) * s[0] + A_OUT[2] / (2 * A_AREA[0]) * s[2] \
        + GAMMA[0] / A_AREA[0] * US[0]
    bbar[1] = -A_OUT[1] / (2 * A_AREA[1]) * s[1] + A_OUT[3] / (2 * A_AREA[1]) * s[3] \
        + GAMMA[1] / A_AREA[1] * US[1]
    bbar[2] = -A_OUT[2] / (2 * A_AREA[2]) * s[2] + (1 - GAMMA[1]) / A_AREA[2] * US[1]
    bbar[3] = -A_OUT[3] / (2 * A_AREA[3]) * s[3] + (1 - GAMMA[0]) / A_AREA[3] * US[0]
    return Alin, bbar


def _numeric_Mmat_bbar(xbar, dt):
    """Evaluate the linearized discrete-time dynamics x_{t+1} = Mmat x_t + dt*B u_t
    + dt*bbar at a concrete (numeric) linearization point xbar."""
    xbar = np.asarray(xbar, dtype=float)
    s = np.sqrt(2 * G * xbar)
    c = G / s
    Alin, bbar = _build_Alin_bbar(c, s)
    Mmat = np.eye(4) + dt * np.array(Alin, dtype=float)
    return Mmat, np.array(bbar, dtype=float)


# ---------------------------------------------------------------------------
# Farkas feasibility certificate
# ---------------------------------------------------------------------------

class FourTankFarkas:
    """
    Certifies feasibility of the T-horizon linearized MPC QP

        find   x_1, ..., x_T,  u_0, ..., u_{T-1}
        s.t.   x_0 == xbar
               x_{t+1} = x_t + dt*(Alin(xbar) x_t + B u_t + bbar(xbar))
               x_lo <= x_t <= x_hi   (t = 0..T)
               u_lo <= u_t <= u_hi   (t = 0..T-1)

    for every linearization point xbar in [x_lo, x_hi] (the box (9)).

    The dynamics are linearized about xbar using the quadratic-constraint
    sqrt trick above. For fixed xbar (equivalently, fixed s, c), the
    remaining system is linear in (x_1..T, u_0..T-1), so Farkas' lemma
    gives an exact feasibility certificate: the system {A_eq z == b, C z <= d}
    (z free) is infeasible iff there exist free duals y_ic, y_0..T-1 and
    nonnegative duals s_up, s_lo, su_up, su_lo with

        A_eq^T y + C^T s == 0             (dual feasibility)
        b^T y + d^T s < 0                 (dual objective, strictly negative)

    (by weak duality, b^T y + d^T s >= 0 for *every* dual-feasible point
    whenever the primal is feasible, so a negative value is only reachable
    when the primal is infeasible). This dual-feasible set is a cone
    (closed under positive scaling of (y, s)), so rather than normalizing
    s to a fixed size we just cap the objective below at -1: any
    negative-objective ray can be rescaled to hit exactly -1, so the true
    minimum is either 0 (no such ray exists) or -1 (one does).

    Minimizing this dual objective jointly over (xbar, s, c, y, s_ineq)
    with Gurobi's NonConvex=2 QCQP solver finds the xbar (and dual
    direction) that comes closest to -- or succeeds at -- proving
    infeasibility. If the optimum is >= -feas_tol (i.e. ~0), the linearized
    MPC QP is feasible for every xbar in the box; if it is ~-1 (below
    -feas_tol), the returned xbar is a counterexample.
    """

    def __init__(self, T, x_lo=X_LO, x_hi=X_HI, u_lo=U_LO, u_hi=U_HI, dt=3.0,
                 dual_bound=50.0, verbose=True, time_limit=None):
        n_x, n_u = 4, 2
        self.T = T

        M = gp.Model("four_tank_farkas")
        M.Params.OutputFlag = 1 if verbose else 0
        # M.Params.NonConvex = 2
        M.Params.FeasibilityTol = 1e-9
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        xbar, s, c = _add_sqrt_linearization(M, "lin", x_lo, x_hi)
        Alin, bbar = _build_Alin_bbar(c, s)
        # Discrete-time (Euler) linearized dynamics: x_{t+1} = Mmat x_t + dt*B u_t + dt*bbar
        Mmat = [[(1.0 if i == j else 0.0) + dt * Alin[i][j] for j in range(n_x)]
                for i in range(n_x)]

        # ---------------- Farkas dual variables ----------------
        # y_ic, y are free in sign but bounded to a large-but-finite range:
        # after normalizing the inequality duals (sum s == 1) their true
        # magnitude is O(1), and a finite range lets Gurobi's spatial
        # branch-and-bound build tight McCormick relaxations for the
        # bilinear stationarity/objective terms -- unbounded duals were
        # observed to stall convergence.
        # db = dual_bound
        db = GRB.INFINITY
        y_ic  = {i: M.addVar(lb=-db, ub=db, name=f"yic_{i}") for i in range(n_x)}
        y     = {t: {i: M.addVar(lb=-db, ub=db, name=f"y_{t}_{i}") for i in range(n_x)}
                 for t in range(T)}
        s_up  = {t: {i: M.addVar(lb=0.0, name=f"sup_{t}_{i}") for i in range(n_x)}
                 for t in range(T + 1)}
        s_lo  = {t: {i: M.addVar(lb=0.0, name=f"slo_{t}_{i}") for i in range(n_x)}
                 for t in range(T + 1)}
        su_up = {t: {k: M.addVar(lb=0.0, name=f"suup_{t}_{k}") for k in range(n_u)}
                 for t in range(T)}
        su_lo = {t: {k: M.addVar(lb=0.0, name=f"sulo_{t}_{k}") for k in range(n_u)}
                 for t in range(T)}

        # Stationarity w.r.t. x_0 (appears in the IC eq. and in the t=0 dynamics eq.)
        for i in range(n_x):
            M.addConstr(
                y_ic[i] - gp.quicksum(Mmat[j][i] * y[0][j] for j in range(n_x))
                + s_up[0][i] - s_lo[0][i] == 0,
                name=f"stat_x0_{i}")

        # Stationarity w.r.t. x_tau, tau = 1..T-1 (appears as x_{t+1} in eq. tau-1,
        # and as x_t in eq. tau)
        for tau in range(1, T):
            for i in range(n_x):
                M.addConstr(
                    y[tau - 1][i] - gp.quicksum(Mmat[j][i] * y[tau][j] for j in range(n_x))
                    + s_up[tau][i] - s_lo[tau][i] == 0,
                    name=f"stat_x_{tau}_{i}")

        # Stationarity w.r.t. x_T (appears only as x_{t+1} in eq. T-1)
        for i in range(n_x):
            M.addConstr(
                y[T - 1][i] + s_up[T][i] - s_lo[T][i] == 0,
                name=f"stat_xT_{i}")

        # Stationarity w.r.t. u_t, t = 0..T-1
        for t in range(T):
            for k in range(n_u):
                M.addConstr(
                    -dt * gp.quicksum(B_MAT[i, k] * y[t][i] for i in range(n_x))
                    + su_up[t][k] - su_lo[t][k] == 0,
                    name=f"stat_u_{t}_{k}")

        # Farkas objective: b^T y + d^T s
        obj = gp.quicksum(xbar[i] * y_ic[i] for i in range(n_x))
        obj += dt * gp.quicksum(bbar[i] * y[t][i] for t in range(T) for i in range(n_x))
        obj += gp.quicksum(x_hi[i] * s_up[t][i] - x_lo[i] * s_lo[t][i]
                            for t in range(T + 1) for i in range(n_x))
        obj += gp.quicksum(u_hi[k] * su_up[t][k] - u_lo[k] * su_lo[t][k]
                            for t in range(T) for k in range(n_u))

        # The dual-feasible cone {A_eq^T y + C^T s == 0, s >= 0} is homogeneous
        # (closed under positive scaling), so instead of normalizing s to a
        # fixed size we just cap the objective below at -1: if any
        # negative-objective ray exists it can always be rescaled to hit
        # exactly -1, so the minimum is either 0 (no such ray -> feasible)
        # or -1 (such a ray exists -> infeasibility certificate).
        M.addConstr(obj >= -1.0, name="obj_floor")
        M.setObjective(obj, GRB.MINIMIZE)

        self.xbar, self.s, self.c = xbar, s, c
        self.y_ic, self.y = y_ic, y
        self.s_up, self.s_lo, self.su_up, self.su_lo = s_up, s_lo, su_up, su_lo
        self.obj_expr = obj

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {"farkas_obj": None}
        return {
            "farkas_obj": self.model.ObjVal,
            "xbar": {i: self.xbar[i].X for i in range(4)},
        }


# ---------------------------------------------------------------------------
# Nominal (fixed-linearization-point) MPC QP -- plain convex QP, no
# NonConvex needed, used only as a cheap diagnostic that actually depends
# on r and T.
# ---------------------------------------------------------------------------

def solve_nominal_mpc(T, r, x0, dt=3.0, x_lo=X_LO, x_hi=X_HI, u_lo=U_LO, u_hi=U_HI,
                       verbose=False, time_limit=None):
    n_x, n_u = 4, 2
    x0 = np.asarray(x0, dtype=float)

    Mmat, bbar = _numeric_Mmat_bbar(x0, dt)

    Q = np.diag([1.0, 1.0, 0.0, 0.0])   # stage cost (8) penalizes only z1, z2
    R = r * np.eye(n_u)

    M = gp.Model("four_tank_nominal_mpc")
    M.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        M.Params.TimeLimit = time_limit

    x = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_{t}") for t in range(T + 1)}
    u = {t: M.addVars(n_u, lb=u_lo, ub=u_hi, name=f"u_{t}") for t in range(T)}

    for i in range(n_x):
        M.addConstr(x[0][i] == x0[i], name=f"ic_{i}")

    for t in range(T):
        for i in range(n_x):
            M.addConstr(
                x[t + 1][i] ==
                gp.quicksum(Mmat[i, j] * x[t][j] for j in range(n_x))
                + dt * gp.quicksum(B_MAT[i, k] * u[t][k] for k in range(n_u))
                + dt * bbar[i],
                name=f"dyn_{t}_{i}")

    obj = 0.5 * (
        gp.quicksum(Q[i, i] * (x[t][i] - XS[i]) * (x[t][i] - XS[i])
                    for t in range(T + 1) for i in range(n_x))
        + gp.quicksum(R[k, k] * (u[t][k] - US[k]) * (u[t][k] - US[k])
                      for t in range(T) for k in range(n_u))
    )
    M.setObjective(obj, GRB.MINIMIZE)
    M.optimize()

    if M.SolCount == 0:
        return M.Status, None, M.Runtime
    return M.Status, M.ObjVal, M.Runtime
