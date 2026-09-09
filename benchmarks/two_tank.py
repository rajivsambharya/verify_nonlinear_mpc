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
# US ~= [43.55, 35.33].
#
# The equilibrium flow US is *not* a good anchor for a u_range sweep: this
# system is open-loop self-stabilizing at u=US (outflow ~ sqrt(level) is
# concave, so a fixed equilibrium-sized inflow already pulls any level in
# the box back toward xs -- verified directly: Mmat's diagonal, which is
# also its eigenvalues since Mmat is upper-triangular, stays in (0,1)
# across the whole box). So a control window *centered at US* is feasible
# even at zero width, and only gets easier as it widens -- there is no
# infeasible regime to sweep out of.
#
# To get a genuine small-window-infeasible / large-window-feasible
# crossover, anchor the sweep at a deliberately *under-powered* baseline
# flow instead, U_ANCHOR = 0.5*US: held there with no deviation, the
# tanks can't sustain the setpoint and drift out of the box; the window
# has to widen enough to let u climb back up (not all the way to US
# itself -- somewhere in between suffices) before it's feasible again.
# U_ANCHOR = 0.5 * US
U_ANCHOR = US

# Box constraints on the states: a symmetric +/- Z_DIFF tolerance around
# the setpoint.
Z_DIFF = 5.0
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
    """Symmetric control window around the under-powered anchor U_ANCHOR
    (not the equilibrium US -- see the comment above), clipped at 0 (pumps
    cannot deliver negative flow)."""
    u_range = np.asarray(u_range, dtype=float)
    if u_range.ndim == 0:
        u_range = np.full(2, float(u_range))
    u_lo = np.maximum(0.0, U_ANCHOR - u_range)
    u_hi = U_ANCHOR + u_range
    return u_lo, u_hi


def _x_box(z_diff, floor=0.1):
    """Symmetric state box +/- z_diff around the setpoint XS, matching the
    module-level Z_DIFF/X_LO/X_HI default. Clipped at `floor` (a small
    positive level, not 0) so sqrt(2 g x) stays well away from the
    singular slope at x=0 regardless of how large z_diff is swept."""
    z_diff = np.asarray(z_diff, dtype=float)
    if z_diff.ndim == 0:
        z_diff = np.full(2, float(z_diff))
    x_lo = np.maximum(floor, XS - z_diff)
    x_hi = XS + z_diff
    return x_lo, x_hi


def run(cfg):
    T_vals = list(cfg.T_vals)
    u_range_vals = list(cfg.u_range_vals)
    z_diff_vals = list(cfg.z_diff_vals)
    r_vals = list(cfg.r_vals)
    dt = cfg.dt
    feas_tol = cfg.feas_tol
    time_limit = cfg.time_limit
    dual_bound = getattr(cfg, 'dual_bound', 50.0)
    bound_tighten_time = getattr(cfg, 'bound_tighten_time', None)
    recfeas_r = getattr(cfg, 'recfeas_r', 0.1)
    x0_nominal = np.array(cfg.x0_nominal) if getattr(cfg, 'x0_nominal', None) is not None \
        else 0.5 * (X_LO + X_HI)

    # -----------------------------------------------------------------
    # Feasibility certificate over a (Z_DIFF, u_range, T) grid: for each
    # state-box half-width and control range, search over xbar in
    # [x_lo, x_hi] for a Farkas certificate that the T-horizon linearized
    # MPC QP is infeasible. Because B is invertible here, feasibility is
    # monotone non-decreasing in u_range (wider control range can only
    # relax the problem) and non-increasing in Z_DIFF (a bigger box is
    # harder to stay inside).
    # -----------------------------------------------------------------
    farkas_obj = np.zeros((len(z_diff_vals), len(u_range_vals), len(T_vals)))
    guaranteed = np.zeros((len(z_diff_vals), len(u_range_vals), len(T_vals)), dtype=bool)
    for zi, z_diff in enumerate(z_diff_vals):
        x_lo, x_hi = _x_box(z_diff)
        for ri, u_range in enumerate(u_range_vals):
            u_lo, u_hi = _u_box(u_range)
            for ti, T in enumerate(T_vals):
                ver = TwoTankFarkas(T=T, x_lo=x_lo, x_hi=x_hi, u_lo=u_lo, u_hi=u_hi, dt=dt,
                                     dual_bound=dual_bound, feas_tol=feas_tol,
                                     bound_tighten_time=bound_tighten_time,
                                     verbose=False, time_limit=time_limit)
                status, elapsed = ver.solve()
                sol = ver.solution_dict()
                obj = sol['farkas_obj']
                certified = obj is not None and obj >= -feas_tol
                guaranteed[zi, ri, ti] = certified
                tag = 'CERTIFIED FEASIBLE' if certified else f'INFEASIBLE COUNTEREXAMPLE (obj={obj})'
                print(f"Z_DIFF={z_diff} (x in [{x_lo}, {x_hi}]), u_range={u_range} "
                      f"(u in [{u_lo}, {u_hi}]), T={T}: "
                      f"farkas_obj={obj}  status={status}  time={elapsed:.3f}s  -> {tag}")
                if not certified and sol.get('xbar') is not None:
                    xbar = sol['xbar']
                    xbar_vec = [xbar[i] for i in range(2)]
                    print(f"  counterexample initial state x0: "
                          f"[{xbar_vec[0]:.4f}, {xbar_vec[1]:.4f}] cm")
                    Mmat_ce, bbar_ce = _numeric_Mmat_bbar(xbar_vec, dt)
                    print(f"  Mmat at counterexample:\n{Mmat_ce}")
                    print(f"  bbar at counterexample: {bbar_ce}")
                farkas_obj[zi, ri, ti] = obj if obj is not None else float('nan')

                # -----------------------------------------------------
                # The Farkas certificate above only proves a linearized-
                # feasible control sequence *exists* -- it says nothing
                # about what the actual MPC law (the KKT-optimal u_0(x_0)
                # of the linearized-at-x_0 QP) does to the *true*
                # nonlinear plant. Check recursive feasibility instead.
                # -----------------------------------------------------
                if certified:
                    chk = TwoTankRecursiveFeasibilityCheck(
                        T=T, x_lo=x_lo, x_hi=x_hi, u_lo=u_lo, u_hi=u_hi, dt=dt, r=recfeas_r,
                        verbose=False, time_limit=time_limit)
                    robust, violation = chk.is_robust()
                    print(f"  [recursive-feasibility check] Z_DIFF={z_diff}, u_range={u_range}, T={T}:")
                    if robust is None:
                        print("    -> INCONCLUSIVE (a sub-solve did not return a bound)")
                    elif robust:
                        print("    -> ROBUST (the KKT-optimal u_0(x_0) always keeps the true "
                              "successor in the box)")
                    else:
                        i_v, side, val = violation
                        bound = x_hi[i_v] if side == 'max' else x_lo[i_v]
                        print(f"    -> NOT ROBUST: true x_1[{i_v}] can reach {val:.4f} "
                              f"({side}), outside box bound {bound}")

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
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    grid_cmap = ListedColormap(['#d62728', '#1f77b4'])   # red = not guaranteed, blue = guaranteed
    legend_handles = [mpatches.Patch(color='#1f77b4', label='feasibility guaranteed'),
                       mpatches.Patch(color='#d62728', label='not guaranteed')]

    for ti, T in enumerate(T_vals):
        fig, ax = plt.subplots(figsize=(8, 6))
        grid = guaranteed[:, :, ti].astype(int)   # rows = Z_DIFF, cols = u_range
        ax.imshow(grid, origin='lower', aspect='auto', cmap=grid_cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(u_range_vals)))
        ax.set_xticklabels(u_range_vals)
        ax.set_yticks(range(len(z_diff_vals)))
        ax.set_yticklabels(z_diff_vals)
        ax.set_xlabel(r'$\Delta u$)')
        ax.set_ylabel('Z\\_DIFF (state box half-width)')
        # ax.set_title(f'T={T}')
        # ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(f'two_tank_feasibility_grid_T{T}.pdf', bbox_inches='tight')
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for ri, r in enumerate(r_vals):
        ax2.plot(T_vals, nominal_cost[ri], marker=markers[ri % len(markers)],
                  linewidth=2, color=colors[ri % len(colors)], label=f'r={r}')
    ax2.set_xlabel('horizon $T$')
    ax2.set_ylabel('nominal MPC cost')
    # ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig('two_tank_nominal_cost.pdf', bbox_inches='tight')
    plt.close(fig2)

    return {
        'T_vals': T_vals, 'u_range_vals': u_range_vals, 'z_diff_vals': z_diff_vals,
        'r_vals': r_vals, 'farkas_obj': farkas_obj, 'guaranteed': guaranteed,
        'nominal_cost': nominal_cost,
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
# Recursive-feasibility check via the KKT-optimal MPC law
#
# At a free x_0 in the box, build the convex QP that results from
# linearizing the dynamics *once*, at x_0 itself, and using that fixed
# (Alin, bbar) for the whole T-step prediction:
#
#     min  0.5*( sum_{t=1}^T (x_t-xs)'Q(x_t-xs) + sum_{t=0}^{T-1}(u_t-us)'R(u_t-us) )
#     s.t. x_{t+1} = Mmat(x_0) x_t + dt B u_t + dt bbar(x_0),  t = 0..T-1
#          x_lo <= x_t <= x_hi,  t = 1..T
#          u_lo <= u_t <= u_hi,  t = 0..T-1
#
# This QP is convex, so its KKT conditions (stationarity, costate
# recursion, complementarity -- encoded here directly as bilinear
# equalities, the same style as bilinear_recursive_feas.py's
# _add_scp_chain, rather than Big-M) are necessary *and* sufficient:
# solving them pins down the unique optimal u_0(x_0), the action a
# receding-horizon MPC controller linearizing at the current state would
# actually apply.
#
# We then ask whether the *true* nonlinear successor
# f_true(x_0, u_0(x_0)) can leave the box, by maximizing/minimizing it
# over x_0 in the box (subject to the whole KKT system). This is a
# single-step check; if it holds for every x_0 in the box, the standard
# recursive argument applies -- the same check at the (now known
# in-box) successor state re-establishes the property one step further,
# so the receding-horizon closed loop never leaves the box.
# ---------------------------------------------------------------------------

class TwoTankRecursiveFeasibilityCheck:
    def __init__(self, T, x_lo=X_LO, x_hi=X_HI, u_lo=None, u_hi=None, dt=1.0,
                 r=0.1, tol=1e-4, verbose=False, time_limit=None):
        n_x, n_u = 2, 2
        if u_lo is None or u_hi is None:
            u_lo, u_hi = _u_box(0.0)
        self.T = T
        self.x_lo, self.x_hi = x_lo, x_hi

        Q = np.eye(n_x)
        R = r * np.eye(n_u)

        M = gp.Model("two_tank_recfeas")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-7
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # x_0 is free in the box, and also *is* the linearization point.
        x0, s0, c0 = _add_sqrt_linearization(M, "x0", x_lo, x_hi)
        Alin, bbar = _build_Alin_bbar(c0, s0)
        Mmat = [[(1.0 if i == j else 0.0) + dt * Alin[i][j] for j in range(n_x)]
                for i in range(n_x)]

        u = {t: M.addVars(n_u, lb=u_lo, ub=u_hi, name=f"u_{t}") for t in range(T)}
        x = {0: x0}
        for t in range(1, T + 1):
            x[t] = M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_{t}")

        for t in range(T):
            for i in range(n_x):
                M.addConstr(
                    x[t + 1][i] ==
                    gp.quicksum(Mmat[i][j] * x[t][j] for j in range(n_x))
                    + dt * gp.quicksum(B_MAT[i, k] * u[t][k] for k in range(n_u))
                    + dt * bbar[i],
                    name=f"dyn_{t}_{i}")

        # KKT multipliers: costate (free), and bilinear complementarity
        # duals (>= 0) for the state box (t=1..T) and control box (t=0..T-1).
        lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lam_{t}") for t in range(1, T + 1)}
        xi_up = {t: M.addVars(n_x, lb=0.0, name=f"xiup_{t}") for t in range(1, T + 1)}
        xi_lo = {t: M.addVars(n_x, lb=0.0, name=f"xilo_{t}") for t in range(1, T + 1)}
        mu_up = {t: M.addVars(n_u, lb=0.0, name=f"muup_{t}") for t in range(T)}
        mu_lo = {t: M.addVars(n_u, lb=0.0, name=f"mulo_{t}") for t in range(T)}

        # Terminal costate.
        for i in range(n_x):
            M.addConstr(
                lam[T][i] == Q[i, i] * (x[T][i] - XS[i]) + xi_up[T][i] - xi_lo[T][i],
                name=f"termcs_{i}")

        # Backward costate recursion, t = T-1 .. 1 (bilinear: Mmat depends on c0).
        for t in range(T - 1, 0, -1):
            for i in range(n_x):
                M.addConstr(
                    lam[t][i] == Q[i, i] * (x[t][i] - XS[i])
                    + gp.quicksum(Mmat[j][i] * lam[t + 1][j] for j in range(n_x))
                    + xi_up[t][i] - xi_lo[t][i],
                    name=f"cs_{t}_{i}")

        # Stationarity w.r.t. u_t, t = 0..T-1.
        for t in range(T):
            for i in range(n_u):
                M.addConstr(
                    R[i, i] * (u[t][i] - US[i])
                    + dt * gp.quicksum(B_MAT[j, i] * lam[t + 1][j] for j in range(n_x))
                    + mu_up[t][i] - mu_lo[t][i] == 0,
                    name=f"stat_u_{t}_{i}")

        # Complementarity (bilinear equalities, no Big-M -- consistent with
        # this file's NonConvex=2 QCQP style throughout).
        for t in range(1, T + 1):
            for i in range(n_x):
                M.addConstr(xi_up[t][i] * (x_hi[i] - x[t][i]) == 0, name=f"cup_x_{t}_{i}")
                M.addConstr(xi_lo[t][i] * (x[t][i] - x_lo[i]) == 0, name=f"clo_x_{t}_{i}")
        for t in range(T):
            for i in range(n_u):
                M.addConstr(mu_up[t][i] * (u_hi[i] - u[t][i]) == 0, name=f"cup_u_{t}_{i}")
                M.addConstr(mu_lo[t][i] * (u[t][i] - u_lo[i]) == 0, name=f"clo_u_{t}_{i}")

        # True (exact, nonlinear) one-step successor under u_0(x_0), computed
        # independently of the QP's own x_1 -- via s0 directly, no
        # tangent-line approximation (the true dynamics don't need one).
        x1_true = [
            x0[0] + dt * (-A_OUT[0] / A_AREA[0] * s0[0] + A_OUT[1] / A_AREA[0] * s0[1]
                          + B_MAT[0, 0] * u[0][0]),
            x0[1] + dt * (-A_OUT[1] / A_AREA[1] * s0[1] + B_MAT[1, 1] * u[0][1]),
        ]

        self.x0, self.u, self.x, self.x1_true = x0, u, x, x1_true

        # Reframe "what is the worst-case x1_true[i]" as a pure
        # feasibility question instead of an optimization: temporarily
        # add a constraint that *forces* a violation (x1_true[i] on the
        # wrong side of the bound by more than tol) and just ask whether
        # that's satisfiable, with SolutionLimit=1 so Gurobi stops the
        # instant it finds *any* feasible point. Finding a violation is
        # then cheap (no need to also prove it's the worst one); proving
        # robustness (infeasibility of the violation constraint) costs
        # the same either way, since that always requires exhausting the
        # search space regardless of whether we ask for it via a bound
        # comparison or a direct feasibility query.
        M.Params.SolutionLimit = 1
        M.setObjective(0.0)

        self.results = {}
        for i in range(n_x):
            for side, exceeds in [('hi', x1_true[i] - x_hi[i] - tol),
                                   ('lo', x_lo[i] - tol - x1_true[i])]:
                c = M.addConstr(exceeds >= 0, name=f"viol_{side}_{i}")
                M.optimize()
                violated = M.SolCount > 0
                proven_safe = M.Status == GRB.INFEASIBLE
                val = x1_true[i].getValue() if violated else None
                self.results[(i, side)] = {
                    'violated': violated, 'val': val,
                    'inconclusive': not violated and not proven_safe,
                    'status': M.Status,
                }
                if verbose:
                    print(f"  recfeas T={T} i={i} {side}: violated={violated} val={val}")
                M.remove(c)
                M.update()

    def solution_dict(self):
        return self.results

    def is_robust(self):
        """None if any sub-check was inconclusive (and none found a
        violation), else True/False. Also returns the first violation
        found (i, side, value), if any."""
        inconclusive = False
        for i in range(2):
            for side in ('hi', 'lo'):
                r = self.results[(i, side)]
                if r['violated']:
                    return False, (i, 'max' if side == 'hi' else 'min', r['val'])
                if r['inconclusive']:
                    inconclusive = True
        return (None, None) if inconclusive else (True, None)


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
