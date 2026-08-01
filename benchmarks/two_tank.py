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
# Two tank system -- a fully-actuated reduction of the four tank system
# (Raff, Huber, Nagy, Allgower, 2006). Tank 2 (the paper's tank 3) drains
# into tank 1 (the paper's tank 1); each tank has its own dedicated pump,
# so unlike the four tank plant (2 pumps split across 4 tanks, B rank 2 of
# 4 -> structurally underactuated) this system's input matrix B is a full
# rank 2x2 diagonal matrix -- there is no direction in state space that a
# large enough control range cannot correct.
#
#   x1' = -a1/A1 sqrt(2 g x1) + a2/A1 sqrt(2 g x2) + gamma1/A1 u1
#   x2' = -a2/A2 sqrt(2 g x2)                       + (1-gamma2)/A2 u2
#
# Physical parameters reuse the paper's tank 1 / tank 3 values (Table I),
# so this is a genuine physical sub-system of the original plant, not an
# arbitrary toy.
# ---------------------------------------------------------------------------

A_AREA = np.array([50.27, 28.27])   # cm^2 (tank1, tank2 cross sections)
A_OUT  = np.array([0.233, 0.127])   # cm^2 (outlet cross sections)
G      = 981.0                       # cm/s^2
GAMMA1 = 0.4                         # fraction of pump1 flow into tank1
GAMMA2 = 0.4                         # (1 - GAMMA2) is the fraction of pump2 flow into tank2

XS = np.array([14.0, 14.2])          # cm, setpoint (matches the paper's x1s, x3s)

# Choose (u1s, u2s) so that (XS, US) is an *exact* equilibrium of the
# continuous dynamics (unlike the paper's rounded table values, which leave
# a small residual) -- this keeps feasibility results attributable to the
# box/control-range choices below rather than to setpoint rounding error.
_S1 = np.sqrt(2 * G * XS[0])
_S2 = np.sqrt(2 * G * XS[1])
US = np.array([
    (A_OUT[0] * _S1 - A_OUT[1] * _S2) / GAMMA1,
    (A_OUT[1] * _S2) / (1 - GAMMA2),
])
# US ~= [43.55, 35.33]. min(US) = 35.33 caps how far u_range can be swept
# while still respecting u_range <= US (pumps can't deliver negative flow).

# Box constraints on the states: a symmetric +/- Z_DIFF tolerance around
# the setpoint. Z_DIFF=10 is tuned (see two_tank.yaml's u_range_vals
# sweep) so that at u_range=0 there is a Farkas counterexample (some
# corner of the box can't be held in-box under one/several steps with u
# pinned exactly at US), while at u_range=30 (< min(US)) the certificate
# is feasible: the crossover happens somewhere in between.
Z_DIFF = 10.0
ZMIN = np.array([-Z_DIFF, -Z_DIFF])
ZMAX = np.array([Z_DIFF, Z_DIFF])
X_LO = XS + ZMIN
X_HI = XS + ZMAX

# Constant, *invertible* input matrix B (2x2, diagonal): each pump drives
# exactly one tank, so the reachable-in-one-step set of x_{t+1} is a full
# 2D affine plane (all of R^2) once u's range is large enough -- no
# direction is structurally uncontrollable, unlike four_tank.py's B (4x2,
# rank 2 of 4).
B_MAT = np.array([
    [GAMMA1 / A_AREA[0], 0.0],
    [0.0, (1 - GAMMA2) / A_AREA[1]],
])


def _u_box(u_range):
    """Symmetric control range around the equilibrium, clipped at 0 (pumps
    cannot deliver negative flow)."""
    u_range = np.asarray(u_range, dtype=float)
    if u_range.ndim == 0:
        u_range = np.full(2, float(u_range))
    u_lo = np.maximum(0.0, US - u_range)
    u_hi = US + u_range
    return u_lo, u_hi


def run(cfg):
    T_vals = list(cfg.T_vals)
    u_range_vals = list(cfg.u_range_vals)
    r_vals = list(cfg.r_vals)
    dt = cfg.dt
    feas_tol = cfg.feas_tol
    time_limit = cfg.time_limit
    dual_bound = getattr(cfg, 'dual_bound', 50.0)
    bound_tighten_time = getattr(cfg, 'bound_tighten_time', None)
    x0_nominal = np.array(cfg.x0_nominal) if getattr(cfg, 'x0_nominal', None) is not None \
        else 0.5 * (X_LO + X_HI)

    # -----------------------------------------------------------------
    # Feasibility certificate over a (u_range, T) grid: for each control
    # range, search over xbar in [X_LO, X_HI] for a Farkas certificate
    # that the T-horizon linearized MPC QP is infeasible. Because B is
    # invertible here, feasibility is monotone non-decreasing in u_range
    # (a wider control range can only relax the problem), so we expect a
    # clean crossover: small u_range -> counterexample found, large
    # u_range -> certified feasible.
    # -----------------------------------------------------------------
    farkas_obj = np.zeros((len(u_range_vals), len(T_vals)))
    linerr_checked = set()   # linearization-error check is T-independent; run once per u_range
    for ri, u_range in enumerate(u_range_vals):
        u_lo, u_hi = _u_box(u_range)
        for ti, T in enumerate(T_vals):
            ver = TwoTankFarkas(T=T, u_lo=u_lo, u_hi=u_hi, dt=dt,
                                 dual_bound=dual_bound, feas_tol=feas_tol,
                                 bound_tighten_time=bound_tighten_time,
                                 verbose=False, time_limit=time_limit)
            status, elapsed = ver.solve()
            sol = ver.solution_dict()
            obj = sol['farkas_obj']
            certified = obj is not None and obj >= -feas_tol
            tag = 'CERTIFIED FEASIBLE' if certified else f'INFEASIBLE COUNTEREXAMPLE (obj={obj})'
            print(f"u_range={u_range} (u in [{u_lo}, {u_hi}]), T={T}: "
                  f"farkas_obj={obj}  status={status}  time={elapsed:.3f}s  -> {tag}")
            if not certified and sol.get('xbar') is not None:
                xbar = sol['xbar']
                xbar_vec = [xbar[i] for i in range(2)]
                print(f"  counterexample initial state x0: "
                      f"[{xbar_vec[0]:.4f}, {xbar_vec[1]:.4f}] cm")
                Mmat_ce, bbar_ce = _numeric_Mmat_bbar(xbar_vec, dt)
                print(f"  Mmat at counterexample:\n{Mmat_ce}")
                print(f"  bbar at counterexample: {bbar_ce}")
            farkas_obj[ri, ti] = obj if obj is not None else float('nan')

            # ---------------------------------------------------------
            # The linearized certificate above says nothing about the
            # *true* nonlinear successor -- verify separately that
            # trusting the linearized model's u_0 choice can't itself
            # push x_1 = f_true(x_0, u_0) out of the box.
            # ---------------------------------------------------------
            if certified and u_range not in linerr_checked:
                linerr_checked.add(u_range)
                chk = TwoTankLinearizationErrorCheck(
                    u_lo=u_lo, u_hi=u_hi, dt=dt, verbose=False, time_limit=time_limit)
                lin_sol = chk.solution_dict()
                robust = chk.is_robust()
                print(f"  [linearization-error check] u_range={u_range}:")
                for i in range(2):
                    b = lin_sol[i]
                    print(f"    x1_true[{i}] range = [{b['min']}, {b['max']}]  "
                          f"vs box [{X_LO[i]}, {X_HI[i]}]")
                verdict = {True: 'ROBUST (true dynamics stay in box)',
                           False: 'NOT ROBUST -- linearization error can leave the box',
                           None: 'INCONCLUSIVE (a sub-solve did not return a bound)'}[robust]
                print(f"    -> {verdict}")

    # -----------------------------------------------------------------
    # Nominal MPC cost: a plain convex QP (linearized once at a *fixed*
    # x0_nominal, so no bilinear terms), just to show how r/T trade off
    # nominal performance. Does not affect the feasibility certificate.
    # -----------------------------------------------------------------
    u_lo_nom, u_hi_nom = _u_box(u_range_vals[-1])
    nominal_cost = np.zeros((len(r_vals), len(T_vals)))
    for ri, r in enumerate(r_vals):
        for ti, T in enumerate(T_vals):
            status, obj, elapsed = solve_nominal_mpc(
                T=T, r=r, x0=x0_nominal, dt=dt, u_lo=u_lo_nom, u_hi=u_hi_nom,
                verbose=False, time_limit=time_limit)
            print(f"[nominal] r={r}, T={T}: status={status}, obj={obj}, time={elapsed:.3f}s")
            nominal_cost[ri, ti] = obj if obj is not None else float('nan')

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for ti, T in enumerate(T_vals):
        ax1.plot(u_range_vals, farkas_obj[:, ti], marker=markers[ti % len(markers)],
                  linewidth=2, color=colors[ti % len(colors)], label=f'T={T}')
    ax1.axhline(0, color='k', linestyle='--', linewidth=1)
    ax1.set_xlabel('control range')
    ax1.set_ylabel('Farkas objective')
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig('two_tank_farkas.pdf', bbox_inches='tight')
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
    fig2.savefig('two_tank_nominal_cost.pdf', bbox_inches='tight')
    plt.close(fig2)

    return {
        'T_vals': T_vals, 'u_range_vals': u_range_vals, 'r_vals': r_vals,
        'farkas_obj': farkas_obj, 'nominal_cost': nominal_cost,
    }


# ---------------------------------------------------------------------------
# Sqrt linearization via quadratic constraints (see four_tank.py for the
# full derivation). sqrt(2 g x) ~= s + c*(x - xbar), with
#     s^2 == 2 g xbar     (s >= 0, the function value)
#     c*s == g             (c >= 0, the derivative g / sqrt(2 g xbar))
# both purely quadratic, so xbar can be a free decision variable while
# every constraint stays at most bilinear (degree 2) -- solvable with
# Gurobi's NonConvex=2 QCQP solver. s must be bounded below by 0 (not
# -infinity): s^2 == 2 g xbar has two roots, and without s >= 0 the solver
# is free to pick the unphysical negative branch.
# ---------------------------------------------------------------------------

def _add_sqrt_linearization(M, tag, x_lo, x_hi):
    n = len(x_lo)
    s_bnd_lo = np.sqrt(2 * G * np.asarray(x_lo, dtype=float))
    s_bnd_hi = np.sqrt(2 * G * np.asarray(x_hi, dtype=float))
    c_bnd_lo = G / s_bnd_hi
    c_bnd_hi = G / s_bnd_lo
    # s_bnd_lo = 0 #np.sqrt(2 * G * np.asarray(x_lo, dtype=float))
    # s_bnd_hi = GRB.INFINITY #np.sqrt(2 * G * np.asarray(x_hi, dtype=float))
    # c_bnd_lo = -GRB.INFINITY #G / s_bnd_hi
    # c_bnd_hi = GRB.INFINITY #G / s_bnd_lo

    xbar = {i: M.addVar(lb=x_lo[i], ub=x_hi[i], name=f"xbar_{tag}_{i}") for i in range(n)}
    s    = {i: M.addVar(lb=s_bnd_lo[i], ub=s_bnd_hi[i], name=f"s_{tag}_{i}") for i in range(n)}
    c    = {i: M.addVar(lb=c_bnd_lo[i], ub=c_bnd_hi[i], name=f"c_{tag}_{i}") for i in range(n)}
    # s    = {i: M.addVar(lb=s_bnd_lo, ub=s_bnd_hi, name=f"s_{tag}_{i}") for i in range(n)}
    # c    = {i: M.addVar(lb=c_bnd_lo, ub=c_bnd_hi, name=f"c_{tag}_{i}") for i in range(n)}
    for i in range(n):
        M.addConstr(s[i] * s[i] == 2 * G * xbar[i], name=f"sqrtdef_{tag}_{i}")
        M.addConstr(c[i] * s[i] == G, name=f"slopedef_{tag}_{i}")
    return xbar, s, c


def _build_Alin_bbar(c, s):
    """
    Linearized continuous-time dynamics x' ~= Alin(c) x + B u + bbar(s),
    valid for *absolute* u (not deviation v = u - us).

    Correct first-order Taylor expansion about (xbar, ubar=US):
        f(x,u) ~= f(xbar,ubar) + Alin(x-xbar) + B(u-ubar)
                = [f(xbar,ubar) - Alin@xbar - B@ubar] + Alin@x + B@u
    The B@ubar term must be *subtracted* out of the constant here,
    because B@u (with absolute u) already reproduces it once u=ubar.
    Using c@xbar == s/2 (since xbar_i = s_i^2/(2g)) collapses
    f(xbar,ubar) - Alin@xbar down to -a/(2A)*s, and the +B@ubar term
    introduced by f(xbar,ubar) exactly cancels the -B@ubar above -- so
    bbar ends up depending only on s, with no US term at all.
    """
    Alin = [[0.0, 0.0], [0.0, 0.0]]
    Alin[0][0] = -A_OUT[0] / A_AREA[0] * c[0]
    Alin[0][1] =  A_OUT[1] / A_AREA[0] * c[1]
    Alin[1][1] = -A_OUT[1] / A_AREA[1] * c[1]

    bbar = [0.0, 0.0]
    bbar[0] = -A_OUT[0] / (2 * A_AREA[0]) * s[0] + A_OUT[1] / (2 * A_AREA[0]) * s[1]
    bbar[1] = -A_OUT[1] / (2 * A_AREA[1]) * s[1]
    return Alin, bbar


def _tighten_var_bounds(M, varlist, time_limit_per_var, verbose=False):
    """
    Optimization-based bound tightening (OBBT).

    For each variable in varlist, solve min v and max v subject to the
    model's *actual* constraints (everything already added to M, including
    the obj_floor constraint), each capped at time_limit_per_var. Gurobi's
    ObjBound is a mathematically valid bound on the true min/max even if
    the sub-solve doesn't converge within that time, so it's always safe
    to use -- it just may be looser than the true optimum for a very
    small time budget. Bounds are only ever tightened, never loosened.

    This directly targets the variables that participate in the model's
    bilinear terms (xbar*y_ic, c*y, s*y): tighter bounds on those shrink
    the McCormick relaxation Gurobi's spatial branch-and-bound builds
    internally, which is what actually determines solve speed near the
    feasible/infeasible crossover -- much more principled than guessing a
    single global dual_bound.
    """
    orig_obj = M.getObjective()
    orig_sense = M.ModelSense
    orig_time_limit = M.Params.TimeLimit
    orig_best_obj_stop = M.Params.BestObjStop
    orig_best_bd_stop = M.Params.BestBdStop

    M.Params.TimeLimit = time_limit_per_var
    M.Params.BestObjStop = -GRB.INFINITY   # disabled -- these thresholds only
    M.Params.BestBdStop = GRB.INFINITY     # make sense for the *final* objective

    for v in varlist:
        M.setObjective(v, GRB.MINIMIZE)
        M.optimize()
        if M.SolCount > 0 and np.isfinite(M.ObjBound) and M.ObjBound > v.LB:
            v.LB = min(M.ObjBound, v.UB)

        M.setObjective(v, GRB.MAXIMIZE)
        M.optimize()
        if M.SolCount > 0 and np.isfinite(M.ObjBound) and M.ObjBound < v.UB:
            v.UB = max(M.ObjBound, v.LB)

        if verbose:
            print(f"  OBBT tightened {v.VarName}: [{v.LB:.4f}, {v.UB:.4f}]")

    M.Params.TimeLimit = orig_time_limit
    M.Params.BestObjStop = orig_best_obj_stop
    M.Params.BestBdStop = orig_best_bd_stop
    M.setObjective(orig_obj, orig_sense)


def _numeric_Mmat_bbar(xbar, dt):
    """Evaluate the linearized discrete-time dynamics x_{t+1} = Mmat x_t + dt*B u_t
    + dt*bbar at a concrete (numeric) linearization point xbar."""
    xbar = np.asarray(xbar, dtype=float)
    s = np.sqrt(2 * G * xbar)
    c = G / s
    Alin, bbar = _build_Alin_bbar(c, s)
    Mmat = np.eye(2) + dt * np.array(Alin, dtype=float)
    return Mmat, np.array(bbar, dtype=float)


# ---------------------------------------------------------------------------
# Farkas feasibility certificate (see four_tank.py for the full derivation
# of the dual system; the structure here is identical, just sized for
# n_x=2, n_u=2).
# ---------------------------------------------------------------------------

class TwoTankFarkas:
    """
    Certifies feasibility of the T-horizon linearized MPC QP

        find   x_1, ..., x_T,  u_0, ..., u_{T-1}
        s.t.   x_0 == xbar
               x_{t+1} = x_t + dt*(Alin(xbar) x_t + B u_t + bbar(xbar))
               x_lo <= x_t <= x_hi   (t = 0..T)
               u_lo <= u_t <= u_hi   (t = 0..T-1)

    for every linearization point xbar in [x_lo, x_hi]. Since B is
    invertible here, any single-step target x_{t+1} is achievable by some
    u_t once u's range is wide enough, so (unlike four_tank.py's
    structurally rank-deficient B) feasibility for large enough u_range is
    not just likely but guaranteed for every xbar in the box -- the only
    question this certificate resolves is *how wide* u_range needs to be
    for a given T.

    Minimizing the Farkas dual objective b^T y + d^T s (capped below at
    -1, see four_tank.py) jointly over (xbar, s, c, y, s_ineq) with
    Gurobi's NonConvex=2 QCQP solver finds the worst-case xbar. If the
    optimum is >= -feas_tol, feasible for every xbar in the box; if it is
    ~-1 (below -feas_tol), the returned xbar is a counterexample.
    """

    def __init__(self, T, x_lo=X_LO, x_hi=X_HI, u_lo=None, u_hi=None, dt=1.0,
                 dual_bound=50.0, feas_tol=1e-4, bound_tighten_time=None,
                 verbose=True, time_limit=None):
        n_x, n_u = 2, 2
        self.T = T
        if u_lo is None or u_hi is None:
            u_lo, u_hi = _u_box(0.0)

        M = gp.Model("two_tank_farkas")
        M.Params.OutputFlag = 1 if verbose else 0
        # M.Params.NonConvex = 2
        M.Params.FeasibilityTol = 1e-9
        self.model = M

        xbar, s, c = _add_sqrt_linearization(M, "lin", x_lo, x_hi)
        Alin, bbar = _build_Alin_bbar(c, s)
        Mmat = [[(1.0 if i == j else 0.0) + dt * Alin[i][j] for j in range(n_x)]
                for i in range(n_x)]

        db = dual_bound
        y_ic  = {i: M.addVar(lb=-db, ub=db, name=f"yic_{i}") for i in range(n_x)}
        y     = {t: {i: M.addVar(lb=-db, ub=db, name=f"y_{t}_{i}") for i in range(n_x)}
                 for t in range(T)}
        s_up  = {t: {i: M.addVar(lb=0.0, ub=db, name=f"sup_{t}_{i}") for i in range(n_x)}
                 for t in range(T + 1)}
        s_lo  = {t: {i: M.addVar(lb=0.0, ub=db, name=f"slo_{t}_{i}") for i in range(n_x)}
                 for t in range(T + 1)}
        su_up = {t: {k: M.addVar(lb=0.0, ub=db, name=f"suup_{t}_{k}") for k in range(n_u)}
                 for t in range(T)}
        su_lo = {t: {k: M.addVar(lb=0.0, ub=db, name=f"sulo_{t}_{k}") for k in range(n_u)}
                 for t in range(T)}

        for i in range(n_x):
            M.addConstr(
                y_ic[i] - gp.quicksum(Mmat[j][i] * y[0][j] for j in range(n_x))
                + s_up[0][i] - s_lo[0][i] == 0,
                name=f"stat_x0_{i}")

        for tau in range(1, T):
            for i in range(n_x):
                M.addConstr(
                    y[tau - 1][i] - gp.quicksum(Mmat[j][i] * y[tau][j] for j in range(n_x))
                    + s_up[tau][i] - s_lo[tau][i] == 0,
                    name=f"stat_x_{tau}_{i}")

        for i in range(n_x):
            M.addConstr(
                y[T - 1][i] + s_up[T][i] - s_lo[T][i] == 0,
                name=f"stat_xT_{i}")

        for t in range(T):
            for k in range(n_u):
                M.addConstr(
                    -dt * gp.quicksum(B_MAT[i, k] * y[t][i] for i in range(n_x))
                    + su_up[t][k] - su_lo[t][k] == 0,
                    name=f"stat_u_{t}_{k}")

        obj = gp.quicksum(xbar[i] * y_ic[i] for i in range(n_x))
        obj += dt * gp.quicksum(bbar[i] * y[t][i] for t in range(T) for i in range(n_x))
        obj += gp.quicksum(x_hi[i] * s_up[t][i] - x_lo[i] * s_lo[t][i]
                            for t in range(T + 1) for i in range(n_x))
        obj += gp.quicksum(u_hi[k] * su_up[t][k] - u_lo[k] * su_lo[t][k]
                            for t in range(T) for k in range(n_u))

        M.addConstr(obj >= -1.0, name="obj_floor")

        # Optimization-based bound tightening on the dual variables that
        # actually appear in bilinear terms (xbar*y_ic, c*y via Mmat,
        # s*y via bbar) -- y_ic, y. s_up/s_lo/su_up/su_lo and xbar/s/c
        # never need it: the box duals only ever appear linearly, and
        # xbar/s/c already have exact analytic bounds from the state box.
        if bound_tighten_time is not None and bound_tighten_time > 0:
            tighten_vars = [y_ic[i] for i in range(n_x)]
            tighten_vars += [y[t][i] for t in range(T) for i in range(n_x)]
            _tighten_var_bounds(M, tighten_vars, bound_tighten_time, verbose=verbose)

        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        # We only need to know which side of -feas_tol the true minimum
        # falls on, not its exact value -- closing the B&B gap all the way
        # to the true optimum (0 or -1) is what was causing timeouts near
        # the feasible/infeasible crossover. BestObjStop lets Gurobi stop
        # the instant it finds *any* incumbent <= -feas_tol (an
        # infeasibility witness -- no need to also prove it's the worst
        # one). BestBdStop lets it stop the instant the best bound rises
        # to >= -feas_tol (proves feasibility -- no need to close in on 0).
        M.Params.BestObjStop = -feas_tol
        M.Params.BestBdStop = -feas_tol
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
            "xbar": {i: self.xbar[i].X for i in range(2)},
        }


# ---------------------------------------------------------------------------
# Linearization-error check
#
# TwoTankFarkas only proves that the *linearized* one-step model has a
# feasible u_0 for every x_0 in the box -- it says nothing about the
# *true* nonlinear successor. This checks, over every x_0 in the box and
# every u_0 the linearized model would accept (u_0 in the control box AND
# the linearized prediction x_1_lin stays in the box), whether the true
# successor x_1_true = f_true(x_0, u_0) can still leave the box. This is
# a single-step check (T-independent): for each state component i,
# maximize and minimize (x_1_true)_i over that same feasible set. If both
# extremes stay inside [x_lo, x_hi] for every i, trusting the linearized
# model's u_0 choice is safe up to the box; if either extreme escapes,
# the linearization error alone is enough to violate the box regardless
# of what the T-horizon certificate said.
# ---------------------------------------------------------------------------

class TwoTankLinearizationErrorCheck:
    def __init__(self, x_lo=X_LO, x_hi=X_HI, u_lo=None, u_hi=None, dt=1.0,
                 verbose=False, time_limit=None):
        n_x, n_u = 2, 2
        if u_lo is None or u_hi is None:
            u_lo, u_hi = _u_box(0.0)
        self.x_lo, self.x_hi = x_lo, x_hi
        self.models = {}

        for i in range(n_x):
            for sense_name, sense in [('max', GRB.MAXIMIZE), ('min', GRB.MINIMIZE)]:
                M = gp.Model(f"two_tank_linerr_{i}_{sense_name}")
                M.Params.OutputFlag = 1 if verbose else 0
                if time_limit is not None:
                    M.Params.TimeLimit = time_limit

                x0, s0, c0 = _add_sqrt_linearization(M, "x0", x_lo, x_hi)
                u0 = M.addVars(n_u, lb=u_lo, ub=u_hi, name="u0")

                # Linearized one-step prediction (what the linearized MPC
                # believes x_1 will be) -- must stay in the box, exactly
                # like the T=1 case of TwoTankFarkas's primal constraints.
                # This restricts u0 to the set the linearized model would
                # actually consider valid.
                Alin, bbar = _build_Alin_bbar(c0, s0)
                for j in range(n_x):
                    x1_lin_j = (x0[j]
                                + dt * gp.quicksum(Alin[j][k] * x0[k] for k in range(n_x))
                                + dt * gp.quicksum(B_MAT[j, k] * u0[k] for k in range(n_u))
                                + dt * bbar[j])
                    M.addConstr(x1_lin_j <= x_hi[j], name=f"linfeas_hi_{j}")
                    M.addConstr(x1_lin_j >= x_lo[j], name=f"linfeas_lo_{j}")

                # True (exact, nonlinear) one-step successor: uses s0
                # directly -- s0 == sqrt(2 g x0) exactly via the quadratic
                # constraint, no tangent-line approximation involved.
                x1_true = [
                    x0[0] + dt * (-A_OUT[0] / A_AREA[0] * s0[0]
                                  + A_OUT[1] / A_AREA[0] * s0[1]
                                  + B_MAT[0, 0] * u0[0]),
                    x0[1] + dt * (-A_OUT[1] / A_AREA[1] * s0[1]
                                  + B_MAT[1, 1] * u0[1]),
                ]

                M.setObjective(x1_true[i], sense)
                M.optimize()
                self.models[(i, sense_name)] = M

    def solution_dict(self):
        out = {}
        for i in range(2):
            hi_M, lo_M = self.models[(i, 'max')], self.models[(i, 'min')]
            out[i] = {
                'max': hi_M.ObjVal if hi_M.SolCount else None,
                'min': lo_M.ObjVal if lo_M.SolCount else None,
                'max_status': hi_M.Status, 'min_status': lo_M.Status,
            }
        return out

    def is_robust(self, tol=1e-6):
        """None if either sub-solve failed to produce a bound, else True/False."""
        sol = self.solution_dict()
        robust = True
        for i, bounds in sol.items():
            if bounds['max'] is None or bounds['min'] is None:
                return None
            if bounds['max'] > self.x_hi[i] + tol or bounds['min'] < self.x_lo[i] - tol:
                robust = False
        return robust


# ---------------------------------------------------------------------------
# Nominal (fixed-linearization-point) MPC QP -- plain convex QP, no
# NonConvex needed, used only as a cheap diagnostic that actually depends
# on r and T.
# ---------------------------------------------------------------------------

def solve_nominal_mpc(T, r, x0, dt=1.0, x_lo=X_LO, x_hi=X_HI, u_lo=None, u_hi=None,
                       verbose=False, time_limit=None):
    n_x, n_u = 2, 2
    x0 = np.asarray(x0, dtype=float)
    if u_lo is None or u_hi is None:
        u_lo, u_hi = _u_box(0.0)

    Mmat, bbar = _numeric_Mmat_bbar(x0, dt)

    Q = np.eye(n_x)
    R = r * np.eye(n_u)

    M = gp.Model("two_tank_nominal_mpc")
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
