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
    """
    Sequential recursive feasibility certification.

    For each step j in j_vals and each T in T_vals, runs two problems:
      1. CartpoleRecursiveFeas:  max r*(x_j) over x_0 in X_0
         If r_opt = 0 => step j is certifiably recursively feasible.
      2. CartpoleMaxStateNorm:  max ||x_j||_inf over x_0 in X_0
         Gives the worst-case state magnitude at step j under the MPC policy.
    """
    j_vals = list(cfg.j_vals)
    T_vals = list(cfg.T_vals)
    r_cost = cfg.r_cost
    dt = cfg.dt
    x0_lo = cfg.x0_mins[0]
    x0_hi = cfg.x0_maxes[0]
    x_feas_lo = cfg.x_feas_mins[0]
    x_feas_hi = cfg.x_feas_maxes[0]
    u_bound = cfg.u_bound
    feas_tol = cfg.feas_tol
    trust_delta = getattr(cfg, 'trust_delta', None)

    # Store results indexed by (T, j)
    feas_results = {}   # r_opt
    norm_results = {}   # max ||x_j||_inf

    for T in T_vals:
        for j in j_vals:
            common_kwargs = dict(
                j=j, T=T, r_cost=r_cost, dt=dt,
                mass=1, length=1, g=9.8,
                x0_lo=x0_lo, x0_hi=x0_hi,
                x_feas_lo=x_feas_lo, x_feas_hi=x_feas_hi,
                u_bound=u_bound, verbose=True,
                time_limit=cfg.time_limit,
            )

            # --- Problem 1: recursive feasibility (max violation r) ---
            # print(f"\n=== Feas: j={j}, T={T} ===")
            # cert = CartpoleRecursiveFeas(**common_kwargs)
            # status_f, t_f = cert.solve()
            # sol_f = cert.solution_dict()
            # r_opt = sol_f['r_opt']
            # certified = r_opt is not None and r_opt <= feas_tol
            # feas_results[(T, j)] = {'certified': certified, 'r_opt': r_opt,
            #                          'status': status_f, 'time': t_f}
            # print(f"  r_opt={r_opt}  {'CERTIFIED' if certified else 'FAIL'}")
            # --- Problem 2: max ||x_j||_inf (SCP-based, linearized QP KKT) ---
            print(f"=== Norm (SCP): j={j}, T={T} ===")
            norm_prob = CartpoleMaxNormSCP(**common_kwargs, delta=trust_delta)
            status_n, t_n = norm_prob.solve()
            sol_n = norm_prob.solution_dict()
            norm_results[(T, j)] = {'inf_norm': sol_n['inf_norm'],
                                     'status': status_n, 'time': t_n}
            print(f"  ||x_j||_inf={sol_n['inf_norm']}")

    # Print summary
    print("\n=== Summary ===")
    for T in T_vals:
        for j in j_vals:
            rf = feas_results[(T, j)]
            rn = norm_results[(T, j)]
            tag = "OK" if rf['certified'] else f"FAIL(r={rf['r_opt']:.2e})"
            norm_str = f"{rn['inf_norm']:.4f}" if rn['inf_norm'] is not None else "N/A"
            print(f"  T={T}, j={j}: {tag}  ||x_j||_inf={norm_str}")

    j_axis = list(j_vals)

    # --- Plot 1: worst-case constraint violation r_opt vs j ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for ti, T in enumerate(T_vals):
        r_vals = [feas_results[(T, j)]['r_opt'] if feas_results[(T, j)]['r_opt'] is not None
                  else float('nan') for j in j_vals]
        ax1.plot(j_axis, r_vals,
                 marker=markers[ti % len(markers)], linewidth=2, color=colors[ti])
    ax1.set_xlabel('step $j$')
    ax1.set_ylabel('max violation $r^*$')
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig('rec_feas_violation.pdf', bbox_inches='tight')
    plt.close(fig1)

    # --- Plot 2: max ||x_j||_inf vs j ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for ti, T in enumerate(T_vals):
        norm_vals = [norm_results[(T, j)]['inf_norm'] if norm_results[(T, j)]['inf_norm'] is not None
                     else float('nan') for j in j_vals]
        ax2.plot(j_axis, norm_vals,
                 marker=markers[ti % len(markers)], linewidth=2, color=colors[ti])
    ax2.set_xlabel('step $j$')
    ax2.set_ylabel(r'$\max \|x_j\|_\infty$')
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig('rec_feas_max_norm.pdf', bbox_inches='tight')
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Shared helper: build j steps of true (box-constrained) MPC chain in model M
# ---------------------------------------------------------------------------

def _add_true_mpc_chain(M, j, T, r_cost, dt, mass, length, g,
                         x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound):
    """
    Add j steps of true MPC chain to model M.

    Variables created:
      x_chain[0..j]: actual closed-loop states
        x_chain[0] in X_0, x_chain[1..j] constrained to X_feas (certified earlier)
      For each k in 0..j-1:
        T-horizon true MPC (box-constrained KKT) at x_chain[k]
        applied control u_chain[k] = u_mpc[k][0]
        nonlinear dynamics: x_chain[k] -> x_chain[k+1]

    Returns (x_chain, u_chain) where u_chain[k] is the applied control var at step k.
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)
    R = np.eye(n_u) * r_cost

    x_chain = {}
    u_chain = {}
    x_chain[0] = M.addVars(n_x, lb=x0_lo, ub=x0_hi, name="xc_0")
    for t in range(1, j + 1):
        x_chain[t] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xc_{t}")
        for idx in range(n_x):
            M.addConstr(x_chain[t][idx] >= x_feas_lo, name=f"xc_feas_lo_{t}_{idx}")
            M.addConstr(x_chain[t][idx] <= x_feas_hi, name=f"xc_feas_hi_{t}_{idx}")

    for k in range(j):
        u_k = M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uc_{k}")
        u_chain[k] = u_k

        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xmc_{k}_{t}")
                 for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"umc_{k}_{t}")
                 for t in range(T)}
        mu_up = {t: M.addVars(n_u, lb=0.0, name=f"muc_up_{k}_{t}") for t in range(T)}
        mu_lo = {t: M.addVars(n_u, lb=0.0, name=f"muc_lo_{k}_{t}") for t in range(T)}

        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x_chain[k][i], name=f"mc_init_{k}_{i}")

        sin_m = {}
        cos_m = {}
        tdd_m = {}
        for t in range(T):
            sin_m[t] = M.addVar(lb=-1, ub=1, name=f"smc_{k}_{t}")
            cos_m[t] = M.addVar(lb=-1, ub=1, name=f"cmc_{k}_{t}")
            tdd_m[t] = M.addVar(lb=-GRB.INFINITY, name=f"tmc_{k}_{t}")
            M.addGenConstrSin(x_mpc[t][0], sin_m[t], name=f"sinc_{k}_{t}")
            M.addGenConstrCos(x_mpc[t][0], cos_m[t], name=f"cosc_{k}_{t}")
            M.addConstr(x_mpc[t+1][0] == x_mpc[t][0] + dt * x_mpc[t][1],
                        name=f"cdth_{k}_{t}")
            M.addConstr(tdd_m[t] == g/length * sin_m[t] + u_mpc[t][0] / (mass * length**2),
                        name=f"ctdd_{k}_{t}")
            M.addConstr(x_mpc[t+1][1] == x_mpc[t][1] + dt * tdd_m[t],
                        name=f"cdtd_{k}_{t}")

        lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamc_{k}_{t}")
               for t in range(T + 1)}

        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i] * x_mpc[T][i], name=f"tc_{k}_{i}")

        for t in range(T - 1, -1, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0] * x_mpc[t][0] + lam[t+1][0]
                + dt * g / length * cos_m[t] * lam[t+1][1],
                name=f"cs0c_{k}_{t}")
            M.addConstr(
                lam[t][1] == Q[1, 1] * x_mpc[t][1] + dt * lam[t+1][0] + lam[t+1][1],
                name=f"cs1c_{k}_{t}")

        for t in range(T):
            M.addConstr(
                R[0, 0] * u_mpc[t][0] + dt / (mass * length**2) * lam[t+1][1]
                + mu_up[t][0] - mu_lo[t][0] == 0,
                name=f"statc_{k}_{t}")
            M.addConstr(mu_up[t][0] * (u_mpc[t][0] - u_bound) == 0,
                        name=f"cup_c_{k}_{t}")
            M.addConstr(mu_lo[t][0] * (-u_bound - u_mpc[t][0]) == 0,
                        name=f"clo_c_{k}_{t}")

        M.addConstr(u_k[0] == u_mpc[0][0], name=f"link_{k}")

        sin_k = M.addVar(lb=-1, ub=1, name=f"sc_{k}")
        cos_k = M.addVar(lb=-1, ub=1, name=f"cc_{k}")
        tdd_k = M.addVar(lb=-GRB.INFINITY, name=f"tdc_{k}")
        M.addGenConstrSin(x_chain[k][0], sin_k, name=f"sincc_{k}")
        M.addGenConstrCos(x_chain[k][0], cos_k, name=f"coscc_{k}")
        M.addConstr(x_chain[k+1][0] == x_chain[k][0] + dt * x_chain[k][1],
                    name=f"ith_{k}")
        M.addConstr(tdd_k == g/length * sin_k + u_k[0] / (mass * length**2),
                    name=f"itdd_{k}")
        M.addConstr(x_chain[k+1][1] == x_chain[k][1] + dt * tdd_k,
                    name=f"itd_{k}")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# Linear MPC chain helper (linearized QP at theta=0, u=0 — zero initialization)
# ---------------------------------------------------------------------------

def _add_linear_mpc_chain(M, j, T, r_cost, dt, mass, length, g,
                           x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound,
                           delta=None):
    """
    Add j steps of the SCP-linearized MPC chain to model M.

    x_chain[0] ∈ [x0_lo, x0_hi]  — outer optimization searches over X_0.
    x_chain[1..j] are fully determined by nonlinear dynamics + KKT policy.

    At each chain step k the inner MPC is a convex QP linearized at x_chain[k]:
        min  (1/2) sum_t [ x_t^T Q x_t + u_t^T R u_t ]
        s.t. x_{t+1} = A_k x_t + B0 u_t + c_k   (linearized at x_chain[k], u=0)
             x_0     = x_chain[k]
             x_t     in [x_feas_lo, x_feas_hi]   t=1..T
             u_t     in [-u_bound, u_bound]
             ||x_t - x_chain[k]||_inf <= delta    t=1..T   (if delta not None)
             ||u_t||_inf              <= delta    t=0..T-1 (if delta not None)

    Full KKT (necessary AND sufficient) is embedded, with duals for all constraints.

    Returns (x_chain, u_chain).
    """
    n_x, n_u = 2, 1
    Q = np.eye(n_x)
    R = np.eye(n_u) * r_cost
    B0_1 = dt / (mass * length ** 2)   # scalar: B0[1, 0]

    x_chain = {}
    u_chain = {}
    x_chain[0] = M.addVars(n_x, lb=x0_lo, ub=x0_hi, name="xl_0")
    for t in range(1, j + 1):
        # Fully determined by dynamics + policy; no bounds.
        x_chain[t] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xl_{t}")

    for k in range(j):
        u_k = M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"ul_{k}")
        u_chain[k] = u_k

        # sin / cos of x_chain[k][0] — used for linearization and chain propagation
        sin_k = M.addVar(lb=-1.0, ub=1.0, name=f"lsin_{k}")
        cos_k = M.addVar(lb=-1.0, ub=1.0, name=f"lcos_{k}")
        M.addGenConstrSin(x_chain[k][0], sin_k, name=f"lsinc_{k}")
        M.addGenConstrCos(x_chain[k][0], cos_k, name=f"lcosc_{k}")

        x_m = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xlm_{k}_{t}")
               for t in range(T + 1)}
        u_m = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"ulm_{k}_{t}")
               for t in range(T)}
        # Costates for t=1..T (x_0 fixed by IC; no stationarity at t=0)
        lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"laml_{k}_{t}")
               for t in range(1, T + 1)}
        mu_up = {t: M.addVar(lb=0.0, name=f"mup_{k}_{t}") for t in range(T)}
        mu_lo = {t: M.addVar(lb=0.0, name=f"mlo_{k}_{t}") for t in range(T)}

        # Duals for x_feas state constraints: x_m[t] ∈ [x_feas_lo, x_feas_hi], t=1..T
        xi_up = {t: M.addVars(n_x, lb=0.0, name=f"xi_up_{k}_{t}") for t in range(1, T + 1)}
        xi_lo = {t: M.addVars(n_x, lb=0.0, name=f"xi_lo_{k}_{t}") for t in range(1, T + 1)}

        # Trust region duals (only when delta is set)
        if delta is not None:
            nu_up  = {t: M.addVars(n_x, lb=0.0, name=f"nu_up_{k}_{t}") for t in range(1, T + 1)}
            nu_lo  = {t: M.addVars(n_x, lb=0.0, name=f"nu_lo_{k}_{t}") for t in range(1, T + 1)}
            eta_up = {t: M.addVar(lb=0.0, name=f"eta_up_{k}_{t}") for t in range(T)}
            eta_lo = {t: M.addVar(lb=0.0, name=f"eta_lo_{k}_{t}") for t in range(T)}

        # Initial condition: x_m[0] = x_chain[k]
        for i in range(n_x):
            M.addConstr(x_m[0][i] == x_chain[k][i], name=f"lmic_{k}_{i}")

        # Linearized dynamics:
        #   x_m[t+1][0] = x_m[t][0] + dt * x_m[t][1]
        #   x_m[t+1][1] = dt*g/L*cos_k*x_m[t][0] + x_m[t][1] + B0_1*u_m[t][0]
        #                + dt*g/L*(sin_k - cos_k*x_chain[k][0])   ← affine offset
        # Bilinear terms (cos_k*x_m, cos_k*x_chain[k][0]) handled by NonConvex=2.
        gdtL = g * dt / length
        for t in range(T):
            M.addConstr(
                x_m[t + 1][0] == x_m[t][0] + dt * x_m[t][1],
                name=f"ldyn_{k}_{t}_0")
            M.addConstr(
                x_m[t + 1][1] == gdtL * cos_k * x_m[t][0]
                                  + x_m[t][1]
                                  + B0_1 * u_m[t][0]
                                  + gdtL * sin_k
                                  - gdtL * cos_k * x_chain[k][0],
                name=f"ldyn_{k}_{t}_1")

        # x_feas primal constraints: x_m[t] ∈ [x_feas_lo, x_feas_hi], t=1..T
        for t in range(1, T + 1):
            for i in range(n_x):
                M.addConstr(x_m[t][i] >= x_feas_lo, name=f"xf_lo_{k}_{t}_{i}")
                M.addConstr(x_m[t][i] <= x_feas_hi, name=f"xf_hi_{k}_{t}_{i}")

        # -----------------------------------------------------------
        # KKT conditions for the linearized QP
        # Dual contributions per constraint type:
        #   xi_up[t][i]:  x_m[t][i] <= x_feas_hi   → subtract xi_up
        #   xi_lo[t][i]:  x_m[t][i] >= x_feas_lo   → add    xi_lo
        #   nu_up[t][i]:  x_m[t][i] <= x_chain[k][i] + delta → subtract nu_up  (if delta)
        #   nu_lo[t][i]:  x_m[t][i] >= x_chain[k][i] - delta → add    nu_lo    (if delta)
        # -----------------------------------------------------------
        # Terminal costate: lam[T][i] = Q x_m[T][i] - xi_up[T][i] + xi_lo[T][i] [±nu]
        for i in range(n_x):
            rhs = Q[i, i] * x_m[T][i] - xi_up[T][i] + xi_lo[T][i]
            if delta is not None:
                rhs = rhs - nu_up[T][i] + nu_lo[T][i]
            M.addConstr(lam[T][i] == rhs, name=f"ltc_{k}_{i}")

        # Backward costate: lam[t] = Q x_m[t] + A_k^T lam[t+1] - xi_up[t] + xi_lo[t] [±nu]
        #   A_k^T[0,:] = [1, gdtL*cos_k],  A_k^T[1,:] = [dt, 1]
        for t in range(T - 1, 0, -1):
            rhs0 = (Q[0, 0] * x_m[t][0]
                    + lam[t + 1][0]
                    + gdtL * cos_k * lam[t + 1][1]
                    - xi_up[t][0] + xi_lo[t][0])
            rhs1 = (Q[1, 1] * x_m[t][1]
                    + dt * lam[t + 1][0]
                    + lam[t + 1][1]
                    - xi_up[t][1] + xi_lo[t][1])
            if delta is not None:
                rhs0 = rhs0 - nu_up[t][0] + nu_lo[t][0]
                rhs1 = rhs1 - nu_up[t][1] + nu_lo[t][1]
            M.addConstr(lam[t][0] == rhs0, name=f"lcs0_{k}_{t}")
            M.addConstr(lam[t][1] == rhs1, name=f"lcs1_{k}_{t}")

        # Stationarity w.r.t. u_t:
        #   R u_t + B0^T lam[t+1] + mu_up - mu_lo [+ eta_up - eta_lo] = 0
        for t in range(T):
            stat = (R[0, 0] * u_m[t][0]
                    + B0_1 * lam[t + 1][1]
                    + mu_up[t] - mu_lo[t])
            if delta is not None:
                stat = stat + eta_up[t] - eta_lo[t]
            M.addConstr(stat == 0, name=f"lstat_{k}_{t}")

        # Complementarity for control bound
        for t in range(T):
            M.addConstr(mu_up[t] * (u_bound - u_m[t][0]) == 0, name=f"lcu_{k}_{t}")
            M.addConstr(mu_lo[t] * (u_bound + u_m[t][0]) == 0, name=f"lcl_{k}_{t}")

        # Complementarity for x_feas constraints
        for t in range(1, T + 1):
            for i in range(n_x):
                M.addConstr(xi_up[t][i] * (x_feas_hi - x_m[t][i]) == 0, name=f"xfc_up_{k}_{t}_{i}")
                M.addConstr(xi_lo[t][i] * (x_m[t][i] - x_feas_lo) == 0, name=f"xfc_lo_{k}_{t}_{i}")

        # Trust region: primal bounds + complementarity
        if delta is not None:
            for t in range(1, T + 1):
                for i in range(n_x):
                    M.addConstr(x_m[t][i] >= x_chain[k][i] - delta, name=f"tr_xlo_{k}_{t}_{i}")
                    M.addConstr(x_m[t][i] <= x_chain[k][i] + delta, name=f"tr_xhi_{k}_{t}_{i}")
                    # nu_up*(x_chain[k][i] + delta - x_m[t][i]) = 0  (quadratic)
                    M.addConstr(
                        nu_up[t][i] * x_chain[k][i] - nu_up[t][i] * x_m[t][i]
                        + delta * nu_up[t][i] == 0,
                        name=f"tr_cup_{k}_{t}_{i}")
                    # nu_lo*(x_m[t][i] - x_chain[k][i] + delta) = 0  (quadratic)
                    M.addConstr(
                        nu_lo[t][i] * x_m[t][i] - nu_lo[t][i] * x_chain[k][i]
                        + delta * nu_lo[t][i] == 0,
                        name=f"tr_clo_{k}_{t}_{i}")
            for t in range(T):
                M.addConstr(u_m[t][0] >= -delta, name=f"tr_ulo_{k}_{t}")
                M.addConstr(u_m[t][0] <= delta,  name=f"tr_uhi_{k}_{t}")
                M.addConstr(eta_up[t] * (delta - u_m[t][0]) == 0, name=f"tr_eup_{k}_{t}")
                M.addConstr(eta_lo[t] * (delta + u_m[t][0]) == 0, name=f"tr_elo_{k}_{t}")

        # Link: applied control is first MPC control
        M.addConstr(u_k[0] == u_m[0][0], name=f"llink_{k}")

        # Chain propagation: true nonlinear Euler step using sin_k, cos_k
        tdd_k = M.addVar(lb=-GRB.INFINITY, name=f"ltdd_{k}")
        M.addConstr(tdd_k == g / length * sin_k + u_k[0] / (mass * length ** 2),
                    name=f"ltddc_{k}")
        M.addConstr(x_chain[k + 1][0] == x_chain[k][0] + dt * x_chain[k][1],
                    name=f"lchain_{k}_0")
        M.addConstr(x_chain[k + 1][1] == x_chain[k][1] + dt * tdd_k,
                    name=f"lchain_{k}_1")

    return x_chain, u_chain


# ---------------------------------------------------------------------------
# Problem 1: recursive feasibility — max constraint violation r at step j
# ---------------------------------------------------------------------------
class CartpoleRecursiveFeas:
    """
    Outer problem:  max_{x_0 ∈ X_0}  r*(x_j)
    where r*(x_j) = optimal constraint violation of the feasibility MPC at x_j.
 
    Feasibility MPC:
        min  r
        s.t. dynamics (forward Euler on cartpole),
             x_t[i] ∈ [x_feas_lo − r,  x_feas_hi + r]  for i ∈ {0,1}, t = 1..T,
             u_t ∈ [−u_bound, u_bound],
             r ≥ 0.
 
    The inner MPC is replaced by its KKT conditions.
    """
 
    def __init__(
        self,
        j: int = 0,
        T: int = 5,
        r_cost: float = 0.1,
        dt: float = 0.1,
        mass: float = 1.0,
        length: float = 1.0,
        g: float = 9.8,
        x0_lo: float = 0.0,
        x0_hi: float = 4.0,
        x_feas_lo: float = -1.0,
        x_feas_hi: float = 1.0,
        u_bound: float = 10.0,
        verbose: bool = True,
        time_limit: float = 1000.0,
    ):
        self.j = j
        self.T = T
        self.dt = dt
        self.mass = mass
        self.length = length
        self.g = g
        self.x_feas_lo = x_feas_lo
        self.x_feas_hi = x_feas_hi
        self.u_bound = u_bound
        n_x, n_u = 2, 1
 
        # ----- Gurobi model -----
        M = gp.Model("cartpole_rec_feas")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.OptimalityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M
 
        # ----- True MPC closed-loop chain (0 → j) -----
        self.x_chain, self.u_chain = _add_true_mpc_chain(
            M, j, T, r_cost, dt, mass, length, g,
            x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound,
        )
 
        # =================================================================
        #  Feasibility MPC at state x_chain[j]
        # =================================================================
        self.r_var = M.addVar(lb=0.0, name="r_feas")
 
        # Primal variables
        self.x_f = {
            t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xf_{t}")
            for t in range(T + 1)
        }
        self.u_f = {
            t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uf_{t}")
            for t in range(T)
        }
 
        # Dual variables – state constraint multipliers (θ and θ̇, upper/lower)
        self.nu_up_0 = {t: M.addVar(lb=0.0, name=f"nu_up0_{t}")
                        for t in range(1, T + 1)}
        self.nu_lo_0 = {t: M.addVar(lb=0.0, name=f"nu_lo0_{t}")
                        for t in range(1, T + 1)}
        self.nu_up_1 = {t: M.addVar(lb=0.0, name=f"nu_up1_{t}")
                        for t in range(1, T + 1)}
        self.nu_lo_1 = {t: M.addVar(lb=0.0, name=f"nu_lo1_{t}")
                        for t in range(1, T + 1)}
 
        # Dual for r ≥ 0
        self.sigma_r = M.addVar(lb=0.0, name="sigma_r")
 
        # Dual for control bounds
        self.mu_up_f = {t: M.addVar(lb=0.0, name=f"mu_up_f_{t}") for t in range(T)}
        self.mu_lo_f = {t: M.addVar(lb=0.0, name=f"mu_lo_f_{t}") for t in range(T)}
 
        # Costates
        self.lam_f = {
            t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamf_{t}")
            for t in range(T + 1)
        }
 
        # Auxiliary nonlinear terms
        sin_f = {}
        cos_f = {}
        tdd_f = {}
 
        # ----- Initial condition: feasibility MPC starts at x_chain[j] -----
        for i in range(n_x):
            M.addConstr(self.x_f[0][i] == self.x_chain[j][i], name=f"finit_{i}")
 
        # ----- Dynamics -----
        for t in range(T):
            sin_f[t] = M.addVar(lb=-1, ub=1, name=f"sf_{t}")
            cos_f[t] = M.addVar(lb=-1, ub=1, name=f"cf_{t}")
            tdd_f[t] = M.addVar(lb=-GRB.INFINITY, name=f"tdf_{t}")
            M.addGenConstrSin(self.x_f[t][0], sin_f[t], name=f"sinf_{t}")
            M.addGenConstrCos(self.x_f[t][0], cos_f[t], name=f"cosf_{t}")
            M.addConstr(
                self.x_f[t + 1][0] == self.x_f[t][0] + dt * self.x_f[t][1],
                name=f"fdth_{t}")
            M.addConstr(
                tdd_f[t] == g / length * sin_f[t]
                + self.u_f[t][0] / (mass * length ** 2),
                name=f"ftdd_{t}")
            M.addConstr(
                self.x_f[t + 1][1] == self.x_f[t][1] + dt * tdd_f[t],
                name=f"fdtd_{t}")
 
        # ----- State constraints (relaxed by r) for t = 1..T -----
        for t in range(1, T + 1):
            M.addConstr(self.x_f[t][0] <= x_feas_hi + self.r_var, name=f"sc0_hi_{t}")
            M.addConstr(self.x_f[t][0] >= x_feas_lo - self.r_var, name=f"sc0_lo_{t}")
            M.addConstr(self.x_f[t][1] <= x_feas_hi + self.r_var, name=f"sc1_hi_{t}")
            M.addConstr(self.x_f[t][1] >= x_feas_lo - self.r_var, name=f"sc1_lo_{t}")
 
        # =================================================================
        #  KKT of the feasibility MPC (objective = r, no stage cost on x/u)
        # =================================================================
 
        # Terminal costate (no stage cost ⇒ only dual terms)
        M.addConstr(self.lam_f[T][0] == self.nu_lo_0[T] - self.nu_up_0[T],
                    name="term_lam_f_0")
        M.addConstr(self.lam_f[T][1] == self.nu_lo_1[T] - self.nu_up_1[T],
                    name="term_lam_f_1")
 
        # Backward costate recursion: t = T-1, ..., 1  (with state-constraint duals)
        for t in range(T - 1, 1 - 1, -1):  # T-1 down to 1 inclusive
            M.addConstr(
                self.lam_f[t][0] == self.lam_f[t + 1][0]
                + dt * g / length * cos_f[t] * self.lam_f[t + 1][1]
                - self.nu_up_0[t] + self.nu_lo_0[t],
                name=f"cs0f_{t}")
            M.addConstr(
                self.lam_f[t][1] == dt * self.lam_f[t + 1][0]
                + self.lam_f[t + 1][1]
                - self.nu_up_1[t] + self.nu_lo_1[t],
                name=f"cs1f_{t}")
 
        # *** FIX: costate at t = 0 (no state constraints at t = 0) ***
        M.addConstr(
            self.lam_f[0][0] == self.lam_f[1][0]
            + dt * g / length * cos_f[0] * self.lam_f[1][1],
            name="cs0f_0")
        M.addConstr(
            self.lam_f[0][1] == dt * self.lam_f[1][0] + self.lam_f[1][1],
            name="cs1f_0")
 
        # Stationarity w.r.t. u_t
        for t in range(T):
            M.addConstr(
                dt / (mass * length ** 2) * self.lam_f[t + 1][1]
                + self.mu_up_f[t] - self.mu_lo_f[t] == 0,
                name=f"statf_{t}")
 
        # Stationarity w.r.t. r
        M.addConstr(
            1.0 - gp.quicksum(
                self.nu_up_0[t] + self.nu_lo_0[t]
                + self.nu_up_1[t] + self.nu_lo_1[t]
                for t in range(1, T + 1))
            - self.sigma_r == 0,
            name="stat_r")
 
        # ----- Complementarity: state constraints (θ) -----
        for t in range(1, T + 1):
            M.addConstr(
                self.nu_up_0[t] * (x_feas_hi + self.r_var - self.x_f[t][0]) == 0,
                name=f"comp_up0_f_{t}")
            M.addConstr(
                self.nu_lo_0[t] * (self.x_f[t][0] - x_feas_lo + self.r_var) == 0,
                name=f"comp_lo0_f_{t}")
 
        # ----- Complementarity: state constraints (θ̇) -----
        for t in range(1, T + 1):
            M.addConstr(
                self.nu_up_1[t] * (x_feas_hi + self.r_var - self.x_f[t][1]) == 0,
                name=f"comp_up1_f_{t}")
            M.addConstr(
                self.nu_lo_1[t] * (self.x_f[t][1] - x_feas_lo + self.r_var) == 0,
                name=f"comp_lo1_f_{t}")
 
        # ----- Complementarity: control bounds -----
        for t in range(T):
            M.addConstr(
                self.mu_up_f[t] * (u_bound - self.u_f[t][0]) == 0,
                name=f"comp_up_uf_{t}")
            M.addConstr(
                self.mu_lo_f[t] * (self.u_f[t][0] + u_bound) == 0,
                name=f"comp_lo_uf_{t}")
 
        # ----- Complementarity: r ≥ 0 -----
        M.addConstr(self.sigma_r * self.r_var == 0, name="comp_r")
 
        # ----- Outer objective: maximise the feasibility violation -----
        M.setObjective(self.r_var, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {'r_opt': None}
        return {
            'r_opt': self.r_var.X,
            'x_chain': {k: {i: self.x_chain[k][i].X for i in range(2)}
                        for k in range(self.j + 1)},
            'u_chain': {k: self.u_chain[k][0].X for k in range(self.j)},
            'x_feas':  {t: {i: self.x_f[t][i].X for i in range(2)}
                        for t in range(self.T + 1)},
            'u_feas':  {t: self.u_f[t][0].X for t in range(self.T)},
            'lam_feas': {t: {i: self.lam_f[t][i].X for i in range(2)}
                         for t in range(self.T + 1)},
            'nu_up_0': {t: self.nu_up_0[t].X for t in range(1, self.T + 1)},
            'nu_lo_0': {t: self.nu_lo_0[t].X for t in range(1, self.T + 1)},
            'nu_up_1': {t: self.nu_up_1[t].X for t in range(1, self.T + 1)},
            'nu_lo_1': {t: self.nu_lo_1[t].X for t in range(1, self.T + 1)},
            'mu_up_f': {t: self.mu_up_f[t].X for t in range(self.T)},
            'mu_lo_f': {t: self.mu_lo_f[t].X for t in range(self.T)},
            'sigma_r': self.sigma_r.X,
        }


class CartpoleRecursiveFeasOLD:
    """
    Outer problem: max_{x_0 in X_0} r*(x_j)
    where r*(x_j) = optimal constraint violation of the feasibility MPC at x_j.

    Feasibility MPC: min r s.t. dynamics, x_t[i] in [xf_lo-r, xf_hi+r] for all i, u in U, r >= 0.

    KKT of feasibility MPC (no stage cost, Q=0 in obj):
      Terminal:   lambda_T[0] = nu_lo_0[T] - nu_up_0[T]
                  lambda_T[1] = nu_lo_1[T] - nu_up_1[T]
      Backward:   lambda_t[0] = lambda_{t+1}[0] + dt*g/L*cos(x_t[0])*lambda_{t+1}[1]
                                - nu_up_0[t] + nu_lo_0[t]
                  lambda_t[1] = dt*lambda_{t+1}[0] + lambda_{t+1}[1]
                                - nu_up_1[t] + nu_lo_1[t]
      Stationarity u_t:  dt/(m*L^2)*lambda_{t+1}[1] + mu_up_t - mu_lo_t = 0
      Stationarity r:    1 - sum(nu_up_0[t]+nu_lo_0[t]+nu_up_1[t]+nu_lo_1[t]) - sigma_r = 0
      Complementarity for state constraints (both dims), control bounds, and r >= 0.
    """

    def __init__(self, j=0, T=5, r_cost=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x0_lo=0.0, x0_hi=4.0, x_feas_lo=-1.0, x_feas_hi=1.0,
                 u_bound=10.0, verbose=True, time_limit=None):
        self.j = j
        self.T = T
        n_x, n_u = 2, 1

        M = gp.Model("cartpole_rec_feas")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        self.x_chain, self.u_chain = _add_true_mpc_chain(
            M, j, T, r_cost, dt, mass, length, g,
            x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound)

        # Feasibility MPC at x_chain[j]
        self.r_var = M.addVar(lb=0.0, name="r_feas")
        self.x_f = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xf_{t}")
                    for t in range(T + 1)}
        self.u_f = {t: M.addVars(n_u, lb=-u_bound, ub=u_bound, name=f"uf_{t}")
                    for t in range(T)}

        # Duals for theta (dim 0) and theta_dot (dim 1) state constraints
        self.nu_up_0 = {t: M.addVar(lb=0.0, name=f"nu_up0_{t}") for t in range(1, T + 1)}
        self.nu_lo_0 = {t: M.addVar(lb=0.0, name=f"nu_lo0_{t}") for t in range(1, T + 1)}
        self.nu_up_1 = {t: M.addVar(lb=0.0, name=f"nu_up1_{t}") for t in range(1, T + 1)}
        self.nu_lo_1 = {t: M.addVar(lb=0.0, name=f"nu_lo1_{t}") for t in range(1, T + 1)}
        self.sigma_r = M.addVar(lb=0.0, name="sigma_r")
        self.mu_up_f = {t: M.addVar(lb=0.0, name=f"mu_up_f_{t}") for t in range(T)}
        self.mu_lo_f = {t: M.addVar(lb=0.0, name=f"mu_lo_f_{t}") for t in range(T)}
        self.lam_f   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lamf_{t}")
                        for t in range(T + 1)}
        sin_f, cos_f, tdd_f = {}, {}, {}

        for i in range(n_x):
            M.addConstr(self.x_f[0][i] == self.x_chain[j][i], name=f"finit_{i}")

        for t in range(T):
            sin_f[t] = M.addVar(lb=-1, ub=1, name=f"sf_{t}")
            cos_f[t] = M.addVar(lb=-1, ub=1, name=f"cf_{t}")
            tdd_f[t] = M.addVar(lb=-GRB.INFINITY, name=f"tdf_{t}")
            M.addGenConstrSin(self.x_f[t][0], sin_f[t], name=f"sinf_{t}")
            M.addGenConstrCos(self.x_f[t][0], cos_f[t], name=f"cosf_{t}")
            M.addConstr(self.x_f[t+1][0] == self.x_f[t][0] + dt * self.x_f[t][1],
                        name=f"fdth_{t}")
            M.addConstr(tdd_f[t] == g/length * sin_f[t] + self.u_f[t][0] / (mass * length**2),
                        name=f"ftdd_{t}")
            M.addConstr(self.x_f[t+1][1] == self.x_f[t][1] + dt * tdd_f[t],
                        name=f"fdtd_{t}")

        # State constraints for both theta and theta_dot
        for t in range(1, T + 1):
            M.addConstr(self.x_f[t][0] <= x_feas_hi + self.r_var, name=f"sc0_hi_{t}")
            M.addConstr(self.x_f[t][0] >= x_feas_lo - self.r_var, name=f"sc0_lo_{t}")
            M.addConstr(self.x_f[t][1] <= x_feas_hi + self.r_var, name=f"sc1_hi_{t}")
            M.addConstr(self.x_f[t][1] >= x_feas_lo - self.r_var, name=f"sc1_lo_{t}")

        # KKT: terminal costate — both dims have state constraint duals
        M.addConstr(self.lam_f[T][0] == self.nu_lo_0[T] - self.nu_up_0[T],
                    name="term_lam_f_0")
        M.addConstr(self.lam_f[T][1] == self.nu_lo_1[T] - self.nu_up_1[T],
                    name="term_lam_f_1")

        for t in range(T - 1, 0, -1):
            M.addConstr(
                self.lam_f[t][0] == self.lam_f[t+1][0]
                + dt * g / length * cos_f[t] * self.lam_f[t+1][1]
                - self.nu_up_0[t] + self.nu_lo_0[t],
                name=f"cs0f_{t}")
            M.addConstr(
                self.lam_f[t][1] == dt * self.lam_f[t+1][0] + self.lam_f[t+1][1]
                - self.nu_up_1[t] + self.nu_lo_1[t],
                name=f"cs1f_{t}")

        for t in range(T):
            M.addConstr(
                dt / (mass * length**2) * self.lam_f[t+1][1]
                + self.mu_up_f[t] - self.mu_lo_f[t] == 0,
                name=f"statf_{t}")

        M.addConstr(
            1.0 - gp.quicksum(
                self.nu_up_0[t] + self.nu_lo_0[t] + self.nu_up_1[t] + self.nu_lo_1[t]
                for t in range(1, T + 1))
            - self.sigma_r == 0,
            name="stat_r")

        # Complementarity: theta constraints
        for t in range(1, T + 1):
            M.addConstr(self.nu_up_0[t] * (self.r_var + x_feas_hi - self.x_f[t][0]) == 0,
                        name=f"comp_up0_f_{t}")
            M.addConstr(self.nu_lo_0[t] * (self.r_var + self.x_f[t][0] - x_feas_lo) == 0,
                        name=f"comp_lo0_f_{t}")

        # Complementarity: theta_dot constraints
        for t in range(1, T + 1):
            M.addConstr(self.nu_up_1[t] * (self.r_var + x_feas_hi - self.x_f[t][1]) == 0,
                        name=f"comp_up1_f_{t}")
            M.addConstr(self.nu_lo_1[t] * (self.r_var + self.x_f[t][1] - x_feas_lo) == 0,
                        name=f"comp_lo1_f_{t}")

        for t in range(T):
            M.addConstr(self.mu_up_f[t] * (self.u_f[t][0] - u_bound) == 0,
                        name=f"comp_up_uf_{t}")
            M.addConstr(self.mu_lo_f[t] * (-u_bound - self.u_f[t][0]) == 0,
                        name=f"comp_lo_uf_{t}")
            
        M.addConstr(
            self.lam_f[0][0] == self.lam_f[1][0]
            + dt * g / length * cos_f[0] * self.lam_f[1][1],
            name="cs0f_0")
        M.addConstr(
            self.lam_f[0][1] == dt * self.lam_f[1][0] + self.lam_f[1][1],
            name="cs1f_0")

        M.addConstr(self.sigma_r * self.r_var == 0, name="comp_r")
        M.setObjective(self.r_var, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {'r_opt': None}
        return {
            'r_opt': self.r_var.X,
            # true MPC closed-loop chain: states and applied controls
            'x_chain': {k: {i: self.x_chain[k][i].X for i in range(2)}
                        for k in range(self.j + 1)},
            'u_chain': {k: self.u_chain[k][0].X for k in range(self.j)},
            # feasibility MPC at x_chain[j]: states, controls, costates, duals
            'x_feas':  {t: {i: self.x_f[t][i].X for i in range(2)}
                        for t in range(self.T + 1)},
            'u_feas':  {t: self.u_f[t][0].X for t in range(self.T)},
            'lam_feas': {t: {i: self.lam_f[t][i].X for i in range(2)}
                         for t in range(self.T + 1)},
            'nu_up_0': {t: self.nu_up_0[t].X for t in range(1, self.T + 1)},
            'nu_lo_0': {t: self.nu_lo_0[t].X for t in range(1, self.T + 1)},
            'nu_up_1': {t: self.nu_up_1[t].X for t in range(1, self.T + 1)},
            'nu_lo_1': {t: self.nu_lo_1[t].X for t in range(1, self.T + 1)},
            'mu_up_f': {t: self.mu_up_f[t].X for t in range(self.T)},
            'mu_lo_f': {t: self.mu_lo_f[t].X for t in range(self.T)},
            'sigma_r': self.sigma_r.X,
        }


# ---------------------------------------------------------------------------
# Problem 2: max ||x_j||_inf — worst-case state magnitude at step j
# ---------------------------------------------------------------------------

class CartpoleMaxStateNorm:
    """
    Outer problem: max_{x_0 in X_0} ||x_j||_inf
    subject to the same j-step true MPC chain (x_t in X_feas for t=1..j).

    Decomposes into 2*n_x sub-problems (one per component/sign):
        max_{x_0}  sign * x_chain[j][i]
    and returns the largest value found.

    Note: the single-model formulation "max z, z >= |x_j[i]|" is always
    unbounded because z has no implicit upper bound, so the decomposition
    approach is required.
    """

    def __init__(self, j=0, T=5, r_cost=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x0_lo=0.0, x0_hi=4.0, x_feas_lo=-1.0, x_feas_hi=1.0,
                 u_bound=10.0, verbose=True, time_limit=None):
        self.j = j
        n_x = 2
        self._subs = []  # list of (model, x_chain, component_idx, sign)

        for i in range(n_x):
            for sign in [1, -1]:
                M = gp.Model(f"cmn_{i}_{sign}")
                M.Params.OutputFlag = 1 if verbose else 0
                M.Params.FeasibilityTol = 1e-9
                M.Params.NonConvex = 2
                if time_limit is not None:
                    M.Params.TimeLimit = time_limit
                xc, uc = _add_true_mpc_chain(
                    M, j, T, r_cost, dt, mass, length, g,
                    x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound)
                M.setObjective(sign * xc[j][i], GRB.MAXIMIZE)
                self._subs.append((M, xc, uc, i, sign))

    def solve(self):
        t_total = 0.0
        for M, _, _, _, _ in self._subs:
            M.optimize()
            t_total += M.Runtime
        return [M.Status for M, _, _, _, _ in self._subs], t_total

    def solution_dict(self):
        best_val = None
        best_xc = None
        best_uc = None
        for M, xc, uc, i, sign in self._subs:
            if M.SolCount == 0:
                continue
            val = sign * xc[self.j][i].X
            if best_val is None or val > best_val:
                best_val = val
                best_xc = xc
                best_uc = uc
        if best_val is None:
            return {'inf_norm': None}
        return {
            'inf_norm': best_val,
            'x_chain': {k: {idx: best_xc[k][idx].X for idx in range(2)}
                        for k in range(self.j + 1)},
            'u_chain': {k: best_uc[k][0].X for k in range(self.j)},
        }


# ---------------------------------------------------------------------------
# Problem 2b: max ||x_j||_inf using SCP-linearized QP KKT (no spurious solutions)
# ---------------------------------------------------------------------------

class CartpoleMaxNormSCP:
    """
    Outer verification:  max_{x_0 ∈ X_0}  ||x_j||_inf
    subject to j steps of the SCP-linearized MPC policy.

    The inner MPC is a convex QP (linearized at theta=0, u=0):
        min  (1/2) [x^T Q x + u^T R u]
        s.t. x_{t+1} = A0 x_t + B0 u_t,  u_t ∈ [-u_bound, u_bound]

    Its KKT is necessary AND sufficient, so embedding it gives no spurious solutions
    (unlike the nonlinear MPC KKT used in CartpoleMaxStateNorm).

    Decomposes into 2*n_x sub-problems, one per (component, sign).
    Requires NonConvex=2 for bilinear complementarity in the QP KKT.
    """

    def __init__(self, j=0, T=5, r_cost=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x0_lo=-1.0, x0_hi=1.0, x_feas_lo=-1.0, x_feas_hi=1.0,
                 u_bound=10.0, delta=None, verbose=True, time_limit=None):
        self.j = j
        n_x = 2
        self._subs = []

        for i in range(n_x):
            for sign in [1, -1]:
                M = gp.Model(f"cmnscp_{i}_{sign}")
                M.Params.OutputFlag = 1 if verbose else 0
                M.Params.FeasibilityTol = 1e-9
                M.Params.OptimalityTol = 1e-9
                M.Params.NonConvex = 2   # needed: bilinear complementarity in QP KKT
                M.Params.MIPGap = 0.01   # terminate within 1% of optimal
                if time_limit is not None:
                    M.Params.TimeLimit = time_limit
                xc, uc = _add_linear_mpc_chain(
                    M, j, T, r_cost, dt, mass, length, g,
                    x0_lo, x0_hi, x_feas_lo, x_feas_hi, u_bound,
                    delta=delta)
                M.setObjective(sign * xc[j][i], GRB.MAXIMIZE)
                self._subs.append((M, xc, uc, i, sign))

    def solve(self):
        t_total = 0.0
        for M, _, _, _, _ in self._subs:
            M.optimize()
            t_total += M.Runtime
        return [M.Status for M, _, _, _, _ in self._subs], t_total

    def solution_dict(self):
        best_val = None
        best_xc = None
        best_uc = None
        for M, xc, uc, i, sign in self._subs:
            if M.SolCount == 0:
                continue
            val = sign * xc[self.j][i].X
            if best_val is None or val > best_val:
                best_val = val
                best_xc = xc
                best_uc = uc
        if best_val is None:
            return {'inf_norm': None}
        return {
            'inf_norm': best_val,
            'x_chain': {k: {idx: best_xc[k][idx].X for idx in range(2)}
                        for k in range(self.j + 1)},
            'u_chain': {k: best_uc[k][0].X for k in range(self.j)},
        }


# ---------------------------------------------------------------------------
# SCP-based feasibility MPC
# ---------------------------------------------------------------------------

class FeasibilitySCPSolver:
    """
    Solves the cartpole feasibility MPC via Sequential Convex Programming
    with an inf-norm trust region.

    Feasibility MPC:
        min  r
        s.t. x_{t+1} = f(x_t, u_t)                       (nonlinear cartpole)
             x_t[i] ∈ [x_feas_lo − r, x_feas_hi + r]     t = 1..T
             u_t     ∈ [−u_bound, u_bound]
             r ≥ 0.

    Each SCP iteration linearizes f around the current trajectory (x_bar, u_bar),
    then solves the resulting LP subject to a trust-region constraint
        ||x_t − x_bar_t||_inf ≤ delta,  ||u_t − u_bar_t||_inf ≤ delta.

    The LP at each iteration is convex, so its KKT conditions are necessary
    AND sufficient (unlike the nonlinear NLP).  Gurobi computes the full dual
    solution; the KKT residual is reported at convergence.
    """

    def __init__(self, T, dt, mass, length, g, x_feas_lo, x_feas_hi, u_bound):
        self.T = T
        self.dt = dt
        self.mass = mass
        self.length = length
        self.g = g
        self.x_feas_lo = x_feas_lo
        self.x_feas_hi = x_feas_hi
        self.u_bound = u_bound

    # ------------------------------------------------------------------
    # Nonlinear cartpole helpers
    # ------------------------------------------------------------------

    def dynamics(self, x, u):
        """One Euler step of the nonlinear cartpole."""
        theta, theta_dot = x
        theta_ddot = (self.g / self.length * np.sin(theta)
                      + float(u) / (self.mass * self.length ** 2))
        return np.array([theta + self.dt * theta_dot,
                         theta_dot + self.dt * theta_ddot])

    def rollout(self, x0, u_seq):
        """Nonlinear rollout: returns (T+1) x 2 array."""
        xs = [np.asarray(x0, dtype=float)]
        for t in range(self.T):
            xs.append(self.dynamics(xs[-1], u_seq[t]))
        return np.array(xs)                   # shape (T+1, 2)

    def linearize(self, x_bar, u_bar):
        """
        Linearize f around (x_bar, u_bar).
        Returns A (2x2), B (2x1), c (2,) such that
            f(x, u) ≈ A @ x + B @ u + c.
        """
        theta = float(x_bar[0])
        A = np.array([[1.0, self.dt],
                      [self.dt * self.g / self.length * np.cos(theta), 1.0]])
        B = np.array([[0.0],
                      [self.dt / (self.mass * self.length ** 2)]])
        f_bar = self.dynamics(x_bar, u_bar)
        c = f_bar - A @ x_bar - B @ np.atleast_1d(u_bar)
        return A, B, c

    # ------------------------------------------------------------------
    # LP subproblem (one SCP iteration)
    # ------------------------------------------------------------------

    def solve_lp(self, x0, x_bar, u_bar, delta, verbose=False):
        """
        Solve one linearized LP subproblem.

        Parameters
        ----------
        x0    : (2,) current state (fixed initial condition)
        x_bar : (T+1, 2) linearization state trajectory
        u_bar : (T, 1) linearization control trajectory
        delta : inf-norm trust-region radius

        Returns
        -------
        r_opt   : float or None
        x_sol   : (T+1, 2) optimal state trajectory
        u_sol   : (T, 1)   optimal control trajectory
        kkt_res : float  max |stationarity residual| over primal vars
        lam_sol : (T+1, 2) LP costates  (dual of linearized dynamics)
        """
        T, n_x, n_u = self.T, 2, 1

        M = gp.Model("feas_scp_lp")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.InfUnbdInfo = 1

        r_var = M.addVar(lb=0.0, name="r")
        x = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"x_{t}") for t in range(T + 1)}
        u = {t: M.addVars(n_u, lb=-self.u_bound, ub=self.u_bound, name=f"u_{t}")
             for t in range(T)}

        # Fixed initial condition
        for i in range(n_x):
            M.addConstr(x[0][i] == float(x0[i]), name=f"ic_{i}")

        # Linearized dynamics + trust region for t = 0..T-1
        dyn_constrs = {}
        for t in range(T):
            A_t, B_t, c_t = self.linearize(x_bar[t], u_bar[t])
            row_c = []
            for i in range(n_x):
                lhs = (gp.quicksum(A_t[i, j] * x[t][j] for j in range(n_x))
                       + gp.quicksum(B_t[i, k] * u[t][k] for k in range(n_u))
                       + float(c_t[i]))
                c_obj = M.addConstr(x[t + 1][i] == lhs, name=f"dyn_{t}_{i}")
                row_c.append(c_obj)
            dyn_constrs[t] = row_c

            # Trust region on x[t]
            for i in range(n_x):
                M.addConstr(x[t][i] >= float(x_bar[t][i]) - delta, name=f"tr_xlo_{t}_{i}")
                M.addConstr(x[t][i] <= float(x_bar[t][i]) + delta, name=f"tr_xhi_{t}_{i}")
            # Trust region on u[t]
            for k in range(n_u):
                M.addConstr(u[t][k] >= float(u_bar[t][k]) - delta, name=f"tr_ulo_{t}_{k}")
                M.addConstr(u[t][k] <= float(u_bar[t][k]) + delta, name=f"tr_uhi_{t}_{k}")

        # Trust region on x[T]
        for i in range(n_x):
            M.addConstr(x[T][i] >= float(x_bar[T][i]) - delta, name=f"tr_xlo_{T}_{i}")
            M.addConstr(x[T][i] <= float(x_bar[T][i]) + delta, name=f"tr_xhi_{T}_{i}")

        # Relaxed state constraints for t = 1..T
        for t in range(1, T + 1):
            for i in range(n_x):
                M.addConstr(x[t][i] <= self.x_feas_hi + r_var, name=f"sc_hi_{t}_{i}")
                M.addConstr(x[t][i] >= self.x_feas_lo - r_var, name=f"sc_lo_{t}_{i}")

        M.setObjective(r_var, GRB.MINIMIZE)
        M.optimize()

        if M.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return None, None, None, None, None

        r_opt = r_var.X
        x_sol = np.array([[x[t][i].X for i in range(n_x)] for t in range(T + 1)])
        u_sol = np.array([[u[t][k].X for k in range(n_u)] for t in range(T)])

        # LP dual: costate = Pi of dynamics equality constraints
        # (sign: Pi for "x_{t+1} = A x_t + B u_t + c" gives the negative Lagrange
        #  multiplier in the minimization Lagrangian, i.e., lambda in our convention)
        lam_sol = np.zeros((T + 1, n_x))
        for t in range(T):
            for i in range(n_x):
                lam_sol[t + 1][i] = dyn_constrs[t][i].Pi

        # KKT stationarity residual for u_t:
        #   B_t^T lam_{t+1} + mu_up - mu_lo + eta_up - eta_lo = 0
        # With Gurobi LP, stationarity is encoded via reduced costs (RC).
        # RC of a basic variable = 0; for non-basic: RC = obj_coeff + A^T * pi.
        # For u[t][0] (obj coeff = 0):  RC = B_t^T lam_{t+1} + (trust/bound duals).
        # We use var.RC directly as the stationarity residual for each u.
        u_rc = np.array([u[t][0].RC for t in range(T)])
        kkt_res = float(np.max(np.abs(u_rc)))

        return r_opt, x_sol, u_sol, kkt_res, lam_sol

    # ------------------------------------------------------------------
    # SCP loop
    # ------------------------------------------------------------------

    def run_scp(self, x0, delta=0.5, max_iter=50, tol=1e-4, verbose=False):
        """
        Run SCP until convergence (change in r < tol) or max_iter.

        Returns
        -------
        r_opt   : float          optimal violation at convergence
        x_sol   : (T+1, 2)      state trajectory
        u_sol   : (T, 1)         control trajectory
        n_iters : int
        kkt_res : float          max stationarity residual of final LP
        """
        x0 = np.asarray(x0, dtype=float)

        # Initialise: zero controls, nonlinear rollout
        u_bar = np.zeros((self.T, 1))
        x_bar = self.rollout(x0, u_bar)

        r_prev = np.inf
        r_opt, x_sol, u_sol, kkt_res = None, None, None, None

        for it in range(max_iter):
            r_opt, x_sol, u_sol, kkt_res, _ = self.solve_lp(
                x0, x_bar, u_bar, delta, verbose=verbose)

            if r_opt is None:
                if verbose:
                    print(f"  SCP iter {it}: LP infeasible/unbounded")
                break

            if verbose:
                print(f"  SCP iter {it}: r={r_opt:.6f}  kkt_res={kkt_res:.2e}")

            # Update linearisation point: roll out nonlinearly with new u
            u_bar = u_sol
            x_bar = self.rollout(x0, u_bar)

            if abs(r_opt - r_prev) < tol:
                if verbose:
                    print(f"  Converged at iter {it+1}")
                break
            r_prev = r_opt

        return r_opt, x_sol, u_sol, it + 1, kkt_res


# ---------------------------------------------------------------------------
# Closed-loop simulation with SCP feasibility MPC
# ---------------------------------------------------------------------------

def run_closedloop_scp(cfg, N_cl=30, delta=0.5, scp_max_iter=30, scp_tol=1e-4,
                       x0=None, verbose=False):
    """
    Run the nonlinear cartpole in closed-loop using the SCP feasibility MPC
    as the controller.

    At each step k:
      1. Run SCP from x_k to solve the feasibility MPC (min r).
      2. Apply the first optimal control u*_0 to the true nonlinear system.
      3. Propagate: x_{k+1} = f(x_k, u*_0).

    Plots and saves:
      - State trajectory and feasibility bounds
      - Constraint violation r at each step
      - KKT residual at each step
    """
    T_vals = list(cfg.T_vals)
    dt = cfg.dt
    x_feas_lo = cfg.x_feas_mins[0]
    x_feas_hi = cfg.x_feas_maxes[0]
    u_bound = cfg.u_bound

    if x0 is None:
        # Default: worst-case corner of X_0
        x0 = np.array([cfg.x0_maxes[0], cfg.x0_maxes[0]])

    results = {}
    for T in T_vals:
        solver = FeasibilitySCPSolver(
            T=T, dt=dt, mass=1.0, length=1.0, g=9.8,
            x_feas_lo=x_feas_lo, x_feas_hi=x_feas_hi, u_bound=u_bound,
        )

        x_cl = [x0.copy()]
        u_cl, r_cl, kkt_cl = [], [], []
        x_cur = x0.copy()

        for step in range(N_cl):
            r_opt, _, u_scp, _, kkt_res = solver.run_scp(
                x_cur, delta=delta, max_iter=scp_max_iter, tol=scp_tol,
                verbose=verbose,
            )
            if r_opt is None:
                print(f"  [T={T}] step {step}: SCP failed — stopping")
                break

            u_apply = float(u_scp[0, 0])
            u_cl.append(u_apply)
            r_cl.append(r_opt)
            kkt_cl.append(kkt_res if kkt_res is not None else np.nan)

            x_next = solver.dynamics(x_cur, u_apply)
            x_cl.append(x_next.copy())
            x_cur = x_next

            if verbose:
                print(f"  [T={T}] step {step}: r={r_opt:.4f}  "
                      f"u={u_apply:.3f}  x={x_cur}  kkt={kkt_res:.2e}")

        results[T] = {
            'x': np.array(x_cl),
            'u': np.array(u_cl),
            'r': np.array(r_cl),
            'kkt': np.array(kkt_cl),
        }


    # --- Plot: state trajectory ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    labels = [r'$\theta$', r'$\dot\theta$']
    for ti, T in enumerate(T_vals):
        xs = results[T]['x']
        n = min(len(xs), N_cl + 1)
        t_ax = np.arange(n)
        for i, ax in enumerate(axes):
            ax.plot(t_ax, xs[:n, i],
                    marker=markers[ti % len(markers)], linewidth=2,
                    color=colors[ti % len(colors)], label=f'T={T}')
    for ax, lbl in zip(axes, labels):
        ax.axhline(x_feas_hi, color='k', linestyle='--', linewidth=1)
        ax.axhline(x_feas_lo, color='k', linestyle='--', linewidth=1)
        ax.set_ylabel(lbl)
        ax.grid(True)
        ax.legend(fontsize=14)
    axes[-1].set_xlabel('closed-loop step')
    fig.tight_layout()
    fig.savefig('scp_cl_states.pdf', bbox_inches='tight')
    plt.close(fig)

    # --- Plot: violation r and KKT residual ---
    fig2, (ax_r, ax_k) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ti, T in enumerate(T_vals):
        rs = results[T]['r']
        ks = results[T]['kkt']
        n = len(rs)
        ax_r.plot(np.arange(n), rs,
                  marker=markers[ti % len(markers)], linewidth=2,
                  color=colors[ti % len(colors)], label=f'T={T}')
        ax_k.semilogy(np.arange(n), np.maximum(ks, 1e-16),
                      marker=markers[ti % len(markers)], linewidth=2,
                      color=colors[ti % len(colors)], label=f'T={T}')
    ax_r.set_ylabel('violation $r$')
    ax_r.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_r.grid(True)
    ax_r.legend(fontsize=14)
    ax_k.set_ylabel('KKT residual')
    ax_k.set_xlabel('closed-loop step')
    ax_k.grid(True)
    fig2.tight_layout()
    fig2.savefig('scp_cl_kkt.pdf', bbox_inches='tight')
    plt.close(fig2)

    return results
