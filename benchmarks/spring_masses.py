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
    K           = cfg.K
    T_vals_list = list(cfg.T_vals)
    r           = cfg.r
    dt          = cfg.dt
    x_min       = cfg.x_mins[0]
    x_max       = cfg.x_maxes[0]
    rho_max     = cfg.rho_max
    u_max       = getattr(cfg, 'u_max', 1.0)
    n_masses    = getattr(cfg, 'n_masses', 6)
    c_spring    = getattr(cfg, 'c', 1.0)
    d_damp      = getattr(cfg, 'd', 0.1)

    A_c, B_c, _ = oscillating_masses_system(n_masses=n_masses, c=c_spring, d=d_damp)
    A_d, B_d    = discretize_euler(A_c, B_c, dt)
    n_x, n_u    = A_d.shape[0], B_d.shape[1]

    Q = np.eye(n_x)
    R = r * np.eye(n_u)

    n_T      = len(T_vals_list)
    times    = np.zeros((n_T, K))
    opt_vals = np.zeros((n_T, K))

    for ti, T in enumerate(T_vals_list):
        # Phase 1: upper bound on V_0 = J*(x_0) for this horizon T
        phase1 = SpringMassPhase1(
            T=T, A_d=A_d, B_d=B_d, Q=Q, R=R, u_max=u_max,
            x_lo=x_min, x_hi=x_max, verbose=True, time_limit=cfg.time_limit
        )
        phase1.solve()
        V_0_max = phase1.V0_max()
        print(f"T={T}: Phase 1 V_0_max = {V_0_max}")

        # Phase 2: bisection over rho
        for k in range(K):
            rho_lo     = 0.0
            rho_hi     = rho_max
            best_rho   = None
            best_sol   = None
            total_time = 0.0

            while rho_hi - rho_lo > cfg.tol:
                rho_mid = (rho_lo + rho_hi) / 2.0

                ver = SpringMassVerify(
                    K=k+1, T=T, A_d=A_d, B_d=B_d, Q=Q, R=R,
                    u_max=u_max, x_lo=x_min, x_hi=x_max, rho=rho_mid,
                    verbose=True, time_limit=cfg.time_limit, V_0_max=V_0_max
                )
                status, solve_time = ver.solve()
                total_time += solve_time

                print(f"T={T}, k={k+1}, rho={rho_mid:.6f}, status={status}")
                sol = ver.solution_dict()
                print(f"  V_curr={sol.get('V_curr')}  V_next={sol.get('V_next')}")

                if status == GRB.TIME_LIMIT:
                    print(f"T={T}, k={k+1}: time limit exceeded, cannot certify rate")
                if status == GRB.OPTIMAL:
                    rho_lo = rho_mid
                else:
                    rho_hi   = rho_mid
                    best_rho = rho_mid
                    best_sol = sol

            if best_sol is not None:
                print(f"Best verified rho for T={T}, k={k+1}: {best_rho:.6f}")
                opt_vals[ti, k] = rho_hi
            else:
                print(f"Could not verify stability for T={T}, k={k+1}")
                opt_vals[ti, k] = np.inf
            times[ti, k] = total_time
            import pdb
            pdb.set_trace()

    k_axis = np.arange(K) + 1

    fig_rate, ax_rate = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        ax_rate.plot(k_axis, opt_vals[ti],
                     marker=markers[ti % len(markers)], linewidth=2,
                     color=colors[ti % len(colors)])
    ax_rate.set_xlabel('iterations $k$')
    ax_rate.set_ylabel('contraction rate $\\rho$')
    ax_rate.grid(True)
    fig_rate.tight_layout()
    fig_rate.savefig('spring_mass_rates.pdf', bbox_inches='tight')
    plt.close(fig_rate)

    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        ax_time.plot(k_axis, times[ti],
                     marker=markers[ti % len(markers)], linewidth=2,
                     color=colors[ti % len(colors)])
    ax_time.set_xlabel('iterations $k$')
    ax_time.set_ylabel('total solve time (sec)')
    ax_time.set_yscale('log')
    ax_time.grid(True)
    fig_time.tight_layout()
    fig_time.savefig('spring_mass_times.pdf', bbox_inches='tight')
    plt.close(fig_time)

    return opt_vals


def oscillating_masses_system(n_masses=6, c=1.0, d=0.1):
    """
    Oscillating masses system (continuous-time).
    n_masses = 6, n_springs = 3, mass = 1.
    a = -2c, b = -2.
    State: x = [positions; velocities] in R^12
    Input: u in R^3
    """
    a = -2 * c
    b = -2.0
    n = n_masses  # 6

    # Build L_6: the spring coupling matrix in R^{6x6}
    # 3 springs connecting pairs (1,2), (3,5), (4,6) (1-indexed):
    #   spring 1: mass 1 (+e1) and mass 2 (-e1)
    #   spring 2: mass 3 (+e2) and mass 5 (-e2)
    #   spring 3: mass 4 (+e3) and mass 6 (-e3)
    # F in R^{6x3}: rows are masses, columns are springs
    F = np.zeros((n, 3))
    F[0, 0] =  1.0   # +e1 (mass 1, spring 1)
    F[1, 0] = -1.0   # -e1 (mass 2, spring 1)
    F[2, 1] =  1.0   # +e2 (mass 3, spring 2)
    F[3, 2] =  1.0   # +e3 (mass 4, spring 3)
    F[4, 1] = -1.0   # -e2 (mass 5, spring 2)
    F[5, 2] = -1.0   # -e3 (mass 6, spring 3)

    # L_6 = F @ F^T  (the graph Laplacian of the spring network)
    L6 = F @ F.T

    I6 = np.eye(n)
    O6 = np.zeros((n, n))

    # A_c = [[0_6,                          I_6                    ],
    #        [a*I_6 + c*L_6 + c*L_6^T,   b*I_6 + d*L_6 + d*L_6^T]]
    A_top = np.hstack([O6, I6])
    A_bot = np.hstack([a * I6 + c * L6 + c * L6.T,
                       b * I6 + d * L6 + d * L6.T])
    A_c = np.vstack([A_top, A_bot])

    # B_c = [[0], [F]]  where F is 6x3
    B_c = np.vstack([np.zeros((n, 3)), F])

    return A_c, B_c, F


def discretize_euler(A_c, B_c, dt):
    n = A_c.shape[0]
    A_d = np.eye(n) + dt * A_c
    B_d = dt * B_c
    return A_d, B_d


def _add_mpc_kkt(M, tag, T, n_x, n_u, A_d, B_d, Q, R, u_max, x_init):
    """
    Add variables and KKT constraints for one T-horizon linear MPC problem.

    MPC (convex QP):
        min  sum_{t=0}^{T} x_t^T Q x_t + sum_{t=0}^{T-1} u_t^T R u_t
        s.t. x_{t+1} = A_d x_t + B_d u_t
             -u_max <= u_t <= u_max

    KKT conditions (necessary AND sufficient for this convex QP):
        Dynamics:      x_{t+1} = A_d x_t + B_d u_t
        Terminal:      lam[T]  = Q x_T
        Costate:       lam[t]  = Q x_t + A_d^T lam[t+1]    (fully linear)
        Stationarity:  R u_t + B_d^T lam[t+1] + nu_up[t] - nu_lo[t] = 0
        Complementarity (bilinear):
                       nu_up[t] * (u_t - u_max) = 0
                       nu_lo[t] * (-u_max - u_t) = 0

    Returns x_traj, u_traj (dicts keyed by time index t).
    """
    x_traj = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"x_{tag}_{t}")
               for t in range(T + 1)}
    u_traj = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_{tag}_{t}")
               for t in range(T)}
    lam    = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lam_{tag}_{t}")
               for t in range(T + 1)}
    nu_up  = {t: M.addVars(n_u, lb=0.0, name=f"nup_{tag}_{t}") for t in range(T)}
    nu_lo  = {t: M.addVars(n_u, lb=0.0, name=f"nlo_{tag}_{t}") for t in range(T)}

    # Initial condition
    for i in range(n_x):
        M.addConstr(x_traj[0][i] == x_init[i], name=f"ic_{tag}_{i}")

    # Linear dynamics
    for t in range(T):
        for i in range(n_x):
            M.addConstr(
                x_traj[t + 1][i] ==
                gp.quicksum(A_d[i, j] * x_traj[t][j] for j in range(n_x))
                + gp.quicksum(B_d[i, j] * u_traj[t][j] for j in range(n_u)),
                name=f"dyn_{tag}_{t}_{i}"
            )

    # Terminal costate: lam[T] = Q x_T
    for i in range(n_x):
        M.addConstr(
            lam[T][i] == gp.quicksum(Q[i, j] * x_traj[T][j] for j in range(n_x)),
            name=f"tc_{tag}_{i}"
        )

    # Costate backward recursion: lam[t] = Q x_t + A_d^T lam[t+1]
    # Fully linear because dynamics are linear — no bilinear products needed.
    for t in range(T - 1, -1, -1):
        for i in range(n_x):
            M.addConstr(
                lam[t][i] == (
                    gp.quicksum(Q[i, j] * x_traj[t][j] for j in range(n_x))
                    + gp.quicksum(A_d[j, i] * lam[t + 1][j] for j in range(n_x))
                ),
                name=f"cs_{tag}_{t}_{i}"
            )

    # Control stationarity: R u_t + B_d^T lam[t+1] + nu_up[t] - nu_lo[t] = 0
    for t in range(T):
        for j in range(n_u):
            M.addConstr(
                gp.quicksum(R[j, l] * u_traj[t][l] for l in range(n_u))
                + gp.quicksum(B_d[i, j] * lam[t + 1][i] for i in range(n_x))
                + nu_up[t][j] - nu_lo[t][j] == 0,
                name=f"stat_{tag}_{t}_{j}"
            )

    # Complementarity (only nonlinearity in the convex-QP KKT)
    for t in range(T):
        for j in range(n_u):
            M.addConstr(nu_up[t][j] * (u_traj[t][j] - u_max) == 0,
                        name=f"cup_{tag}_{t}_{j}")
            M.addConstr(nu_lo[t][j] * (-u_max - u_traj[t][j]) == 0,
                        name=f"clo_{tag}_{t}_{j}")

    return x_traj, u_traj


def _mpc_cost(x_traj, u_traj, T, n_x, n_u, Q, R):
    """MPC cost: sum_{t=0}^T x_t^T Q x_t + sum_{t=0}^{T-1} u_t^T R u_t."""
    return (
        gp.quicksum(Q[i, i] * x_traj[t][i] * x_traj[t][i]
                    for t in range(T + 1) for i in range(n_x))
        + gp.quicksum(R[j, j] * u_traj[t][j] * u_traj[t][j]
                      for t in range(T) for j in range(n_u))
    )


class SpringMassPhase1:
    """
    Phase 1: compute V_0_max = max_{x_0 in X_0} J*(x_0).

    J*(x_0) is the T-horizon MPC optimal cost from x_0.
    Since the MPC is a convex QP, KKT conditions are necessary and sufficient,
    so we can enforce them as equality/complementarity constraints without
    losing any optimal solutions.

    The only nonlinearity in the KKT is complementarity (bilinear), handled
    via NonConvex=2.  The costate recursion is fully linear.
    """

    def __init__(self, T, A_d, B_d, Q, R, u_max, x_lo, x_hi,
                 verbose=True, time_limit=None):
        n_x = A_d.shape[0]
        n_u = B_d.shape[1]

        M = gp.Model("spring_mass_phase1")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.NonConvex  = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")

        x_traj, u_traj = _add_mpc_kkt(
            M, tag="p1", T=T, n_x=n_x, n_u=n_u,
            A_d=A_d, B_d=B_d, Q=Q, R=R, u_max=u_max, x_init=x0
        )

        V0 = _mpc_cost(x_traj, u_traj, T, n_x, n_u, Q, R)
        self.orig_objective = V0
        M.setObjective(V0, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def V0_max(self):
        if self.model.SolCount == 0:
            return None
        return self.model.ObjVal


class SpringMassVerify:
    """
    Phase 2: verify closed-loop Lyapunov stability under T-horizon MPC.

    Searches for a counterexample: an initial state x_0 in X_0 and K steps
    of the MPC-controlled system such that V(x_K) >= rho * V(x_0), where
    V(x) = J*(x) is the MPC optimal cost (Lyapunov candidate).

    Infeasibility certifies that V decreases by factor rho in K steps for
    all initial conditions in X_0.

    Since the MPC is a convex QP:
      - KKT conditions are both necessary and sufficient for optimality.
      - The costate recursion is fully linear (linear dynamics => no bilinear
        adjoint products).
      - The only nonlinearity is complementarity (bilinear), handled via
        NonConvex=2.
    """

    def __init__(self, K, T, A_d, B_d, Q, R, u_max, x_lo, x_hi, rho,
                 verbose=True, time_limit=None, V_0_max=None):
        self.K = K
        n_x = A_d.shape[0]
        n_u = B_d.shape[1]

        M = gp.Model("spring_mass_verify")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.NonConvex  = 2
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # Closed-loop state trajectory x[0], ..., x[K]
        x = {0: M.addVars(n_x, lb=x_lo, ub=x_hi, name="x_0")}
        for k in range(1, K + 1):
            x[k] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"x_{k}")

        # Applied controls u[k] = first MPC action at step k
        u = {k: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_{k}")
             for k in range(K)}

        # MPC KKT at each closed-loop step k = 0, ..., K
        x_traj, u_traj = {}, {}
        for k in range(K + 1):
            xt, ut = _add_mpc_kkt(
                M, tag=f"cl{k}", T=T, n_x=n_x, n_u=n_u,
                A_d=A_d, B_d=B_d, Q=Q, R=R, u_max=u_max, x_init=x[k]
            )
            x_traj[k] = xt
            u_traj[k] = ut

        # Link first MPC control to applied control
        for k in range(K):
            for j in range(n_u):
                M.addConstr(u[k][j] == u_traj[k][0][j], name=f"ulink_{k}_{j}")

        # Closed-loop dynamics: x[k+1] = A_d x[k] + B_d u[k]
        for k in range(K):
            for i in range(n_x):
                M.addConstr(
                    x[k + 1][i] ==
                    gp.quicksum(A_d[i, j] * x[k][j] for j in range(n_x))
                    + gp.quicksum(B_d[i, j] * u[k][j] for j in range(n_u)),
                    name=f"cldyn_{k}_{i}"
                )

        # Lyapunov values: V(x[0]) and V(x[K])
        V_curr = _mpc_cost(x_traj[0], u_traj[0], T, n_x, n_u, Q, R)
        V_next = _mpc_cost(x_traj[K], u_traj[K], T, n_x, n_u, Q, R)
        self.V_curr = V_curr
        self.V_next = V_next

        # Restrict search to the Phase-1 bound on V(x_0)
        if V_0_max is not None:
            M.addConstr(V_curr <= V_0_max, name="V0_bound")
            M.addConstr(V_curr >= 1e-2, name="V0_bound2")

        # Counterexample constraint: V_next >= rho * V_curr
        # Infeasibility of this problem => stability verified at rate rho.
        eps = 1.0 - rho
        M.addConstr(V_next - V_curr + eps * V_curr >= 1e-6,
                    name="lyap_violation")

        M.setObjective(0, GRB.MAXIMIZE)
        # M.setObjective(V_next - V_curr, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        if self.model.SolCount == 0:
            return {"obj": None}
        return {
            "obj":    self.model.ObjVal,
            "V_curr": self.V_curr.getValue(),
            "V_next": self.V_next.getValue(),
        }


# Module-level system construction (used for quick testing)
A_c, B_c, F = oscillating_masses_system()
dt = 0.1
A_d, B_d = discretize_euler(A_c, B_c, dt)
