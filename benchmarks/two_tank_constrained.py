import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

import benchmarks.two_tank as tt

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
# Same physical system as two_tank.py (reused directly, not re-derived), but
# now comparing two linearized-MPC terminal-constraint schemes -- the two
# approaches to guaranteed closed-loop stability described in Raff, Huber,
# Nagy, Allgower (Section III):
#
#   1. Box-terminal (tt.TwoTankFarkas): x_T only has to stay in the box
#      [x_lo, x_hi], same as every other predicted state.
#   2. Zero terminal state constraint (TwoTankTerminalFarkas below): x_T
#      must equal the setpoint XS *exactly*. The paper notes this "does
#      require in general infinite number of iterations in the
#      optimization" and "leads to feasibility problems especially for
#      short control/prediction horizons" -- we verify that directly.
#
# Since {x_T == XS} => {x_lo <= x_T <= x_hi} (XS is always inside the box
# by construction), the zero-terminal feasible region is a *subset* of the
# box-terminal feasible region for every (Z_DIFF, u_range, T): whenever the
# zero-terminal scheme is certified feasible, the box-terminal scheme must
# be too. The comparison grid below checks that this containment actually
# holds and visualizes exactly where the two schemes disagree.
# ---------------------------------------------------------------------------


class TwoTankTerminalFarkas:
    """
    Same Farkas feasibility certificate as tt.TwoTankFarkas, except the
    terminal state x_T is pinned by the equality x_T == XS (the "zero
    terminal state constraint" scheme) instead of the box inequalities
    x_lo <= x_T <= x_hi. Concretely: x_T's box-dual pair (s_up[T], s_lo[T])
    is replaced by one free dual y_term for the new equality constraint,
    which also removes x_T's box terms from the Farkas objective and adds
    a term for the equality's RHS (XS) instead. Everything else -- the
    IC/dynamics stationarity for x_0..x_{T-1}, the u_t stationarity, the
    objective-floor-at-(-1) reformulation, OBBT, the early-stopping
    BestObjStop/BestBdStop thresholds -- is identical to TwoTankFarkas;
    see that class's docstring for the full Farkas-lemma derivation.
    """

    def __init__(self, T, x_lo=tt.X_LO, x_hi=tt.X_HI, u_lo=None, u_hi=None, dt=1.0,
                 dual_bound=50.0, feas_tol=1e-4, bound_tighten_time=None,
                 verbose=True, time_limit=None):
        n_x, n_u = 2, 2
        self.T = T
        if u_lo is None or u_hi is None:
            u_lo, u_hi = tt._u_box(0.0)

        M = gp.Model("two_tank_terminal_farkas")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        self.model = M

        xbar, s, c = tt._add_sqrt_linearization(M, "lin", x_lo, x_hi)
        Alin, bbar = tt._build_Alin_bbar(c, s)
        Mmat = [[(1.0 if i == j else 0.0) + dt * Alin[i][j] for j in range(n_x)]
                for i in range(n_x)]

        db = dual_bound
        y_ic = {i: M.addVar(lb=-db, ub=db, name=f"yic_{i}") for i in range(n_x)}
        y = {t: {i: M.addVar(lb=-db, ub=db, name=f"y_{t}_{i}") for i in range(n_x)}
             for t in range(T)}
        y_term = {i: M.addVar(lb=-db, ub=db, name=f"yterm_{i}") for i in range(n_x)}
        # Box duals only for x_0..x_{T-1} (T entries, t=0..T-1) -- x_T no
        # longer has box inequalities, it has the equality above instead.
        s_up = {t: {i: M.addVar(lb=0.0, ub=db, name=f"sup_{t}_{i}") for i in range(n_x)}
                for t in range(T)}
        s_lo = {t: {i: M.addVar(lb=0.0, ub=db, name=f"slo_{t}_{i}") for i in range(n_x)}
                for t in range(T)}
        su_up = {t: {k: M.addVar(lb=0.0, ub=db, name=f"suup_{t}_{k}") for k in range(n_u)}
                 for t in range(T)}
        su_lo = {t: {k: M.addVar(lb=0.0, ub=db, name=f"sulo_{t}_{k}") for k in range(n_u)}
                 for t in range(T)}

        # Stationarity w.r.t. x_0 (identical to TwoTankFarkas).
        for i in range(n_x):
            M.addConstr(
                y_ic[i] - gp.quicksum(Mmat[j][i] * y[0][j] for j in range(n_x))
                + s_up[0][i] - s_lo[0][i] == 0,
                name=f"stat_x0_{i}")

        # Stationarity w.r.t. x_tau, tau = 1..T-1 (identical to TwoTankFarkas).
        for tau in range(1, T):
            for i in range(n_x):
                M.addConstr(
                    y[tau - 1][i] - gp.quicksum(Mmat[j][i] * y[tau][j] for j in range(n_x))
                    + s_up[tau][i] - s_lo[tau][i] == 0,
                    name=f"stat_x_{tau}_{i}")

        # Stationarity w.r.t. x_T: x_T now appears only in the terminal
        # equality (coeff +1, dual y_term) and as the "+1" term of
        # dynamics eq. T-1 (dual y[T-1]) -- no box duals anymore.
        for i in range(n_x):
            M.addConstr(y[T - 1][i] + y_term[i] == 0, name=f"stat_xT_{i}")

        # Stationarity w.r.t. u_t (identical to TwoTankFarkas).
        for t in range(T):
            for k in range(n_u):
                M.addConstr(
                    -dt * gp.quicksum(tt.B_MAT[i, k] * y[t][i] for i in range(n_x))
                    + su_up[t][k] - su_lo[t][k] == 0,
                    name=f"stat_u_{t}_{k}")

        # Farkas objective: b^T y + d^T s, now including the terminal
        # equality's RHS (XS) contribution via y_term, and with the box
        # terms only over t=0..T-1 (x_T has none).
        obj = gp.quicksum(xbar[i] * y_ic[i] for i in range(n_x))
        obj += dt * gp.quicksum(bbar[i] * y[t][i] for t in range(T) for i in range(n_x))
        obj += gp.quicksum(tt.XS[i] * y_term[i] for i in range(n_x))
        obj += gp.quicksum(x_hi[i] * s_up[t][i] - x_lo[i] * s_lo[t][i]
                            for t in range(T) for i in range(n_x))
        obj += gp.quicksum(u_hi[k] * su_up[t][k] - u_lo[k] * su_lo[t][k]
                            for t in range(T) for k in range(n_u))

        M.addConstr(obj >= -1.0, name="obj_floor")

        if bound_tighten_time is not None and bound_tighten_time > 0:
            tighten_vars = [y_ic[i] for i in range(n_x)]
            tighten_vars += [y[t][i] for t in range(T) for i in range(n_x)]
            tighten_vars += [y_term[i] for i in range(n_x)]
            tt._tighten_var_bounds(M, tighten_vars, bound_tighten_time, verbose=verbose)

        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        M.Params.BestObjStop = -feas_tol
        M.Params.BestBdStop = -feas_tol
        M.setObjective(obj, GRB.MINIMIZE)

        self.xbar, self.s, self.c = xbar, s, c
        self.y_ic, self.y, self.y_term = y_ic, y, y_term
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


def run(cfg):
    T_vals = list(cfg.T_vals)
    u_range_vals = list(cfg.u_range_vals)
    z_diff_vals = list(cfg.z_diff_vals)
    dt = cfg.dt
    feas_tol = cfg.feas_tol
    time_limit = cfg.time_limit
    dual_bound = getattr(cfg, 'dual_bound', 50.0)
    bound_tighten_time = getattr(cfg, 'bound_tighten_time', None)

    n_z, n_u, n_T = len(z_diff_vals), len(u_range_vals), len(T_vals)
    box_obj = np.zeros((n_z, n_u, n_T))
    term_obj = np.zeros((n_z, n_u, n_T))
    box_guaranteed = np.zeros((n_z, n_u, n_T), dtype=bool)
    term_guaranteed = np.zeros((n_z, n_u, n_T), dtype=bool)

    for zi, z_diff in enumerate(z_diff_vals):
        x_lo, x_hi = tt._x_box(z_diff)
        for ri, u_range in enumerate(u_range_vals):
            u_lo, u_hi = tt._u_box(u_range)
            for ti, T in enumerate(T_vals):
                ver_box = tt.TwoTankFarkas(
                    T=T, x_lo=x_lo, x_hi=x_hi, u_lo=u_lo, u_hi=u_hi, dt=dt,
                    dual_bound=dual_bound, feas_tol=feas_tol,
                    bound_tighten_time=bound_tighten_time,
                    verbose=False, time_limit=time_limit)
                ver_box.solve()
                obj_b = ver_box.solution_dict()['farkas_obj']
                cert_b = obj_b is not None and obj_b >= -feas_tol
                box_obj[zi, ri, ti] = obj_b if obj_b is not None else float('nan')
                box_guaranteed[zi, ri, ti] = cert_b

                ver_term = TwoTankTerminalFarkas(
                    T=T, x_lo=x_lo, x_hi=x_hi, u_lo=u_lo, u_hi=u_hi, dt=dt,
                    dual_bound=dual_bound, feas_tol=feas_tol,
                    bound_tighten_time=bound_tighten_time,
                    verbose=False, time_limit=time_limit)
                ver_term.solve()
                obj_t = ver_term.solution_dict()['farkas_obj']
                cert_t = obj_t is not None and obj_t >= -feas_tol
                term_obj[zi, ri, ti] = obj_t if obj_t is not None else float('nan')
                term_guaranteed[zi, ri, ti] = cert_t

                print(f"Z_DIFF={z_diff} u_range={u_range} T={T}: "
                      f"box-terminal obj={obj_b} ({'FEASIBLE' if cert_b else 'INFEASIBLE'})   "
                      f"zero-terminal obj={obj_t} ({'FEASIBLE' if cert_t else 'INFEASIBLE'})")

                if cert_t and not cert_b:
                    print("  ** WARNING: zero-terminal certified feasible but box-terminal was "
                          "not -- should be impossible (zero-terminal implies box-terminal), "
                          "likely a numerical artifact right at the feasibility boundary.")
                # import pdb
                # pdb.set_trace()

    # -----------------------------------------------------------------
    # Comparison grid, one per T: three categories --
    #   both guaranteed / box-terminal only / neither guaranteed.
    # "zero-terminal only" should never occur (see the containment
    # argument above); if it ever does the WARNING above will have fired.
    # -----------------------------------------------------------------
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    category = np.zeros((n_z, n_u, n_T), dtype=int)
    category[box_guaranteed & ~term_guaranteed] = 1
    category[box_guaranteed & term_guaranteed] = 2

    cmap3 = ListedColormap(['#d62728', '#ff7f0e', '#1f77b4'])
    legend_handles = [
        mpatches.Patch(color='#1f77b4', label='both guaranteed'),
        mpatches.Patch(color='#ff7f0e', label='box-terminal only'),
        mpatches.Patch(color='#d62728', label='neither guaranteed'),
    ]

    for ti, T in enumerate(T_vals):
        fig, ax = plt.subplots(figsize=(8, 6))
        grid = category[:, :, ti]
        ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap3, vmin=0, vmax=2)
        ax.set_xticks(range(len(u_range_vals)))
        ax.set_xticklabels(u_range_vals)
        ax.set_yticks(range(len(z_diff_vals)))
        ax.set_yticklabels(z_diff_vals)
        ax.set_xlabel(r'$\Delta u$')
        ax.set_ylabel(r'$\Delta x$')
        ax.set_title(f'T={T}')
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(f'two_tank_constrained_comparison_T{T}.pdf', bbox_inches='tight')
        plt.close(fig)

    # -----------------------------------------------------------------
    # (T, u_range) grids, one per Z_DIFF and per certificate: plain
    # blue (guaranteed) / red (not guaranteed), T on the x-axis and
    # u_range on the y-axis.
    # -----------------------------------------------------------------
    grid_cmap = ListedColormap(['#d62728', '#1f77b4'])   # red = not guaranteed, blue = guaranteed
    bw_legend = [mpatches.Patch(color='#1f77b4', label='feasibility guaranteed'),
                 mpatches.Patch(color='#d62728', label='not guaranteed')]

    for zi, z_diff in enumerate(z_diff_vals):
        for name, guaranteed_arr in [('boxterm', box_guaranteed), ('zeroterm', term_guaranteed)]:
            fig, ax = plt.subplots(figsize=(8, 6))
            grid = guaranteed_arr[zi].astype(int)   # rows = u_range, cols = T
            ax.imshow(grid, origin='lower', aspect='auto', cmap=grid_cmap, vmin=0, vmax=1)
            ax.set_xticks(range(len(T_vals)))
            ax.set_xticklabels(T_vals)
            ax.set_yticks(range(len(u_range_vals)))
            ax.set_yticklabels(u_range_vals)
            ax.set_xlabel(r'$T$')
            ax.set_ylabel(r'$\Delta u$')
            ax.set_title(f'{"box-terminal" if name == "boxterm" else "zero-terminal-state"}, '
                          f'Z_DIFF={z_diff}')
            ax.legend(handles=bw_legend, loc='center left', bbox_to_anchor=(1.02, 0.5))
            fig.tight_layout()
            fig.savefig(f'two_tank_constrained_grid_{name}_zdiff{z_diff}.pdf', bbox_inches='tight')
            plt.close(fig)

    # -----------------------------------------------------------------
    # Side-by-side (T, u_range) grids: no terminal constraint vs. terminal
    # constraint, one figure per Z_DIFF, no legend.
    # -----------------------------------------------------------------
    for zi, z_diff in enumerate(z_diff_vals):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, arr, title, show_ylabel in [(axes[0], box_guaranteed, 'no terminal constraint', True),
                                             (axes[1], term_guaranteed, 'terminal constraint', False)]:
            grid = arr[zi].astype(int)   # rows = u_range, cols = T
            ax.imshow(grid, origin='lower', aspect='auto', cmap=grid_cmap, vmin=0, vmax=1)
            ax.set_xticks(range(len(T_vals)))
            ax.set_xticklabels(T_vals)
            ax.set_yticks(range(len(u_range_vals)))
            ax.set_yticklabels(u_range_vals)
            ax.set_xlabel(r'$T$')
            if show_ylabel:
                ax.set_ylabel(r'$\Delta u$')
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(f'two_tank_constrained_side_by_side_zdiff{z_diff}.pdf', bbox_inches='tight')
        plt.close(fig)

    return {
        'T_vals': T_vals, 'u_range_vals': u_range_vals, 'z_diff_vals': z_diff_vals,
        'box_obj': box_obj, 'term_obj': term_obj,
        'box_guaranteed': box_guaranteed, 'term_guaranteed': term_guaranteed,
    }
