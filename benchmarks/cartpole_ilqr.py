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
    K = cfg.K
    x_mins = cfg.x_mins
    x_maxes = cfg.x_maxes

    T_vals_list = list(cfg.T_vals)
    r = cfg.r
    dt = cfg.dt

    x_min = x_mins[0]
    x_max = x_maxes[0]
    rho_max = cfg.rho_max

    n_T = len(T_vals_list)
    times    = np.zeros((n_T, K))
    opt_vals = np.zeros((n_T, K))

    for ti, T in enumerate(T_vals_list):
        # --- Phase 1: upper bound on V_0 = J*(x_0) for this horizon T ---
        phase1 = CartpolePhase1(
            T=T, r=r, dt=dt, mass=1, length=1, g=9.8,
            x_lo=x_min, x_hi=x_max, verbose=True,
            time_limit=cfg.time_limit
        )
        phase1.solve()
        V_0_max = phase1.V0_max()
        print(f"T={T}: Phase 1 V_0_max = {V_0_max}")

        # --- Phase 2: bisection over rho using V_0_max from Phase 1 ---
        for k in range(K):
            rho_lo = 0.0
            rho_hi = rho_max
            best_rho = None
            best_sol = None
            total_time = 0.0

            while rho_hi - rho_lo > cfg.tol:
                rho_mid = (rho_lo + rho_hi) / 2.0

                ver = CartpoleILQRVerify(
                    n=cfg.n, K=k+1, T=T, r=r, dt=dt,
                    mass=1, length=1, g=9.8, rho=rho_mid,
                    x_lo=x_min, x_hi=x_max, seed=42, verbose=False,
                    time_limit=cfg.time_limit, V_0_max=V_0_max
                )

                status, solve_time = ver.solve()
                total_time += solve_time

                print(f"T={T}, k={k+1}, rho={rho_mid:.6f}, status:", status)
                sol = ver.solution_dict()
                print("Objective:", sol["obj"])

                if status == GRB.TIME_LIMIT:
                    print(f"T={T}, k={k+1}, rho={rho_mid:.6f}: time limit exceeded, cannot certify rate")
                if status == GRB.OPTIMAL:
                    rho_lo = rho_mid
                else:
                    rho_hi = rho_mid
                    best_rho = rho_mid
                    best_sol = sol
                # import pdb; pdb.set_trace()

            if best_sol is not None:
                print(f"Best verified rho for T={T}, k={k+1}: {best_rho:.6f}")
                opt_vals[ti, k] = rho_hi
            else:
                print(f"Could not verify any rho <= {rho_max} for T={T}, k={k+1}")
                opt_vals[ti, k] = np.inf
            times[ti, k] = total_time

    k_axis = np.arange(K) + 1
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    fig_rate, ax_rate = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        ax_rate.plot(k_axis, opt_vals[ti], marker=markers[ti % len(markers)], linewidth=2, color=colors[ti])
    ax_rate.set_xlabel('iterations')
    ax_rate.set_ylabel('rate')
    ax_rate.grid(True)
    fig_rate.tight_layout()
    fig_rate.savefig('rates_ilqr.pdf', bbox_inches='tight')
    plt.close(fig_rate)

    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    for ti in range(n_T):
        ax_time.plot(k_axis, times[ti], marker=markers[ti % len(markers)], linewidth=2, color=colors[ti])
    ax_time.set_xlabel('iterations $k$')
    ax_time.set_ylabel('total solve time (sec)')
    ax_time.set_yscale('log')
    ax_time.grid(True)
    fig_time.tight_layout()
    fig_time.savefig('times_ilqr.pdf', bbox_inches='tight')
    plt.close(fig_time)


class CartpolePhase1:
    """
    Phase 1: compute V_0_max = max_{x_0 in X_0, KKT} J*(x_0).

    J*(x_0) = sum_{t=0}^{T} x_t^T Q x_t + sum_{t=0}^{T-1} u_t^T R u_t
    is the MPC value function evaluated at x_0 (optimal cost under KKT
    conditions).  The resulting V_0_max is passed to Phase 2 as an upper
    bound on V_curr, restricting the worst-case search to initial conditions
    that are actually reachable under the MPC policy.
    """

    def __init__(self, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x_lo=0.0, x_hi=4.0, verbose=True, time_limit=None):
        n_x = 2
        n_u = 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * r
        u_max = 1000

        M = gp.Model("cartpole_phase1")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2   # bilinear costate term + quadratic objective
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # Initial state (free within X_0)
        x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x0")

        # T-horizon MPC trajectory
        # x_mpc = {t: M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_mpc_{t}")
        #          for t in range(T + 1)}
        x_mpc = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_mpc_{t}")
                 for t in range(T + 1)}
        u_mpc = {t: M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_mpc_{t}")
                 for t in range(T)}
        lam   = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lam_{t}")
                 for t in range(T + 1)}

        # MPC initial condition
        for i in range(n_x):
            M.addConstr(x_mpc[0][i] == x0[i], name=f"init_{i}")

        # MPC nonlinear dynamics
        sin_mpc = {}
        cos_mpc = {}
        tdd_mpc = {}
        for t in range(T):
            sin_mpc[t] = M.addVar(lb=-1, ub=1, name=f"sin_mpc_{t}")
            cos_mpc[t] = M.addVar(lb=-1, ub=1, name=f"cos_mpc_{t}")
            tdd_mpc[t] = M.addVar(lb=-GRB.INFINITY, name=f"tdd_mpc_{t}")
            M.addGenConstrSin(x_mpc[t][0], sin_mpc[t], name=f"sin_c_{t}")
            M.addGenConstrCos(x_mpc[t][0], cos_mpc[t], name=f"cos_c_{t}")
            M.addConstr(x_mpc[t+1][0] == x_mpc[t][0] + dt * x_mpc[t][1],
                        name=f"dyn_theta_{t}")
            M.addConstr(tdd_mpc[t] == g/length * sin_mpc[t] +
                        u_mpc[t][0] / (mass * length**2),
                        name=f"dyn_tdd_{t}")
            M.addConstr(x_mpc[t+1][1] == x_mpc[t][1] + dt * tdd_mpc[t],
                        name=f"dyn_thetadot_{t}")

        # KKT: terminal costate
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i] * x_mpc[T][i],
                        name=f"term_costate_{i}")

        # KKT: costate backward recursion (bilinear: cos_mpc[t] * lam[t+1][1])
        for t in range(T - 1, -1, -1):
            M.addConstr(
                lam[t][0] == Q[0, 0] * x_mpc[t][0] + lam[t+1][0]
                + dt * g / length * cos_mpc[t] * lam[t+1][1],
                name=f"costate_0_{t}"
            )
            M.addConstr(
                lam[t][1] == Q[1, 1] * x_mpc[t][1]
                + dt * lam[t+1][0] + lam[t+1][1],
                name=f"costate_1_{t}"
            )

        # KKT: control stationarity
        for t in range(T):
            M.addConstr(
                R[0, 0] * u_mpc[t][0]
                + dt / (mass * length**2) * lam[t+1][1] == 0,
                name=f"ctrl_opt_{t}"
            )

        # Objective: maximize V_0 = J*(x_0)
        V0 = (
            gp.quicksum(Q[i, i] * x_mpc[t][i] * x_mpc[t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R[j, j] * u_mpc[t][j] * u_mpc[t][j]
                          for t in range(T) for j in range(n_u))
        )
        # M.addConstr(V0 <= 100)
        M.setObjective(V0, GRB.MAXIMIZE)

    def solve(self):
        self.model.optimize()
        # import pdb
        # pdb.set_trace()
        return self.model.Status, self.model.Runtime

    def V0_max(self):
        if self.model.SolCount == 0:
            return None
        return self.model.ObjVal


class CartpoleILQRVerify:
    def __init__(self, n=10, K=3, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 rho=0.2, x_lo=0.0, x_hi=4.0, seed=None, verbose=True,
                 time_limit=None, V_0_max=None):
        self.n, self.K, self.rho = n, K, rho
        self.T = T
        self.verbose = bool(verbose)
        rng = np.random.default_rng(seed)

        n_x = 2
        n_u = 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * r
        u_max = 1000

        # Model
        M = gp.Model("cartpole_ilqr_verify")
        M.Params.OutputFlag = 1 if self.verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2   # bilinear costate term; quadratic V constraints
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # Store state/control trajectories
        self.x = {}      # states: x[k] for k=0..K
        self.u = {}      # controls: u[k] for k=0..K-1

        # Store MPC variables for each timestep
        self.x_mpc = {}      # MPC state variables (T+1 timesteps)
        self.u_mpc_var = {}  # MPC control variables (T timesteps)
        self.lambda_mpc = {} # MPC costate variables for optimality (T+1 timesteps)

        # Auxiliary variables for dynamics
        self.sin_theta = {}
        self.cos_theta = {}
        self.theta_ddot = {}

        # Auxiliary variables for MPC dynamics
        self.sin_theta_mpc = {}
        self.cos_theta_mpc = {}
        self.theta_ddot_mpc = {}

        # CREATE ALL STATE VARIABLES FIRST (k=0 to K)
        for k in range(K + 1):
            # self.x[k] = M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_{k}")
            if k == 0:
                self.x[k] = M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_{k}")
            else:
                self.x[k] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_{k}")

        # Create variables and constraints for K timesteps
        for k in range(K):
            # Control at timestep k (applied from state k)
            self.u[k] = M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_{k}")

            # MPC variables: solve T-horizon MPC at each timestep k
            # MPC states: x_mpc[k][t] for t=0..T
            self.x_mpc[k] = {}
            for t in range(T + 1):
                self.x_mpc[k][t] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_mpc_{k}_{t}")

            # MPC controls: u_mpc[k][t] for t=0..T-1
            self.u_mpc_var[k] = {}
            for t in range(T):
                self.u_mpc_var[k][t] = M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_mpc_{k}_{t}")

            # MPC initial condition: x_mpc[k][0] = x[k]
            for i in range(n_x):
                M.addConstr(self.x_mpc[k][0][i] == self.x[k][i], name=f"mpc_init_{k}_{i}")

            # Auxiliary variables for MPC nonlinear dynamics
            self.sin_theta_mpc[k] = {}
            self.cos_theta_mpc[k] = {}
            self.theta_ddot_mpc[k] = {}

            for t in range(T):
                self.sin_theta_mpc[k][t] = M.addVar(lb=-1, ub=1, name=f"sin_theta_mpc_{k}_{t}")
                self.cos_theta_mpc[k][t] = M.addVar(lb=-1, ub=1, name=f"cos_theta_mpc_{k}_{t}")
                self.theta_ddot_mpc[k][t] = M.addVar(lb=-GRB.INFINITY, name=f"theta_ddot_mpc_{k}_{t}")

                # Nonlinear dynamics constraints for MPC
                M.addGenConstrSin(self.x_mpc[k][t][0], self.sin_theta_mpc[k][t], name=f"sin_mpc_{k}_{t}")
                M.addGenConstrCos(self.x_mpc[k][t][0], self.cos_theta_mpc[k][t], name=f"cos_mpc_{k}_{t}")

                # theta_{t+1} = theta_t + dt * theta_dot_t
                M.addConstr(
                    self.x_mpc[k][t+1][0] == self.x_mpc[k][t][0] + dt * self.x_mpc[k][t][1],
                    name=f"mpc_theta_{k}_{t}"
                )

                # theta_ddot = (g/L) * sin(theta) + u / (m*L²)
                M.addConstr(
                    self.theta_ddot_mpc[k][t] == g/length * self.sin_theta_mpc[k][t] +
                    self.u_mpc_var[k][t][0] / (mass * length**2),
                    name=f"mpc_angular_accel_{k}_{t}"
                )

                # theta_dot_{t+1} = theta_dot_t + dt * theta_ddot
                M.addConstr(
                    self.x_mpc[k][t+1][1] == self.x_mpc[k][t][1] + dt * self.theta_ddot_mpc[k][t],
                    name=f"mpc_theta_dot_{k}_{t}"
                )

            # KKT optimality conditions for MPC
            # Add costate (dual) variables for dynamics constraints
            self.lambda_mpc[k] = {}
            for t in range(T + 1):
                self.lambda_mpc[k][t] = M.addVars(n_x, lb=-GRB.INFINITY, name=f"lambda_mpc_{k}_{t}")

            # Terminal costate condition: λ_T = Q x_T
            for i in range(n_x):
                M.addConstr(
                    self.lambda_mpc[k][T][i] == Q[i, i] * self.x_mpc[k][T][i],
                    name=f"terminal_costate_{k}_{i}"
                )

            # Backward sweep: costate equations
            # λ_t = Q x_t + (∂f/∂x_t)^T λ_{t+1}
            # For f = [x₁ + dt·x₂, x₂ + dt·(g/L·sin(x₁) + u/(m·L²))]
            # ∂f/∂x = [[1, dt], [dt·g/L·cos(x₁), 1]]
            for t in range(T - 1, -1, -1):
                # λ_t[0] = Q[0,0] x_t[0] + 1 · λ_{t+1}[0] + dt·g/L·cos(x_t[0]) · λ_{t+1}[1]
                M.addConstr(
                    self.lambda_mpc[k][t][0] == Q[0, 0] * self.x_mpc[k][t][0] +
                    self.lambda_mpc[k][t+1][0] +
                    dt * g / length * self.cos_theta_mpc[k][t] * self.lambda_mpc[k][t+1][1],
                    name=f"costate_0_{k}_{t}"
                )

                # λ_t[1] = Q[1,1] x_t[1] + dt · λ_{t+1}[0] + 1 · λ_{t+1}[1]
                M.addConstr(
                    self.lambda_mpc[k][t][1] == Q[1, 1] * self.x_mpc[k][t][1] +
                    dt * self.lambda_mpc[k][t+1][0] +
                    self.lambda_mpc[k][t+1][1],
                    name=f"costate_1_{k}_{t}"
                )

            # Control optimality: R u_t + (∂f/∂u_t)^T λ_{t+1} = 0
            # ∂f/∂u = [0, dt/(m·L²)]^T
            # So: R u_t + dt/(m·L²) · λ_{t+1}[1] = 0
            for t in range(T):
                M.addConstr(
                    R[0, 0] * self.u_mpc_var[k][t][0] +
                    dt / (mass * length**2) * self.lambda_mpc[k][t+1][1] == 0,
                    name=f"control_opt_{k}_{t}"
                )

            # Link the first control action to the applied control
            M.addConstr(self.u[k][0] == self.u_mpc_var[k][0][0], name=f"control_link_{k}")

            # Auxiliary variables for actual dynamics (state evolution)
            self.sin_theta[k] = M.addVar(lb=-1, ub=1, name=f"sin_theta_{k}")
            self.cos_theta[k] = M.addVar(lb=-1, ub=1, name=f"cos_theta_{k}")
            self.theta_ddot[k] = M.addVar(lb=-GRB.INFINITY, name=f"theta_ddot_{k}")

            # Nonlinear dynamics constraints for actual state evolution
            M.addGenConstrSin(self.x[k][0], self.sin_theta[k], name=f"sin_{k}")
            M.addGenConstrCos(self.x[k][0], self.cos_theta[k], name=f"cos_{k}")

            # theta_{k+1} = theta_k + dt * theta_dot_k
            M.addConstr(
                self.x[k+1][0] == self.x[k][0] + dt * self.x[k][1],
                name=f"integrate_theta_{k}"
            )

            # theta_ddot = (g/L) * sin(theta) + u / (m*L²)
            M.addConstr(
                self.theta_ddot[k] == g/length * self.sin_theta[k] + self.u[k][0] / (mass * length**2),
                name=f"angular_accel_{k}"
            )

            # theta_dot_{k+1} = theta_dot_k + dt * theta_ddot
            M.addConstr(
                self.x[k+1][1] == self.x[k][1] + dt * self.theta_ddot[k],
                name=f"integrate_theta_dot_{k}"
            )

        # ---------------------------------------------------------------
        # Phase 2: extra MPC at k=K to compute V_next = J*(x[K])
        # Only added when V_0_max is provided (two-phase mode).
        # ---------------------------------------------------------------
        # if V_0_max is not None:
        self.x_mpc[K] = {}
        for t in range(T + 1):
            self.x_mpc[K][t] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                            name=f"x_mpc_{K}_{t}")
        self.u_mpc_var[K] = {}
        for t in range(T):
            self.u_mpc_var[K][t] = M.addVars(n_u, lb=-u_max, ub=u_max,
                                                name=f"u_mpc_{K}_{t}")
        # Initial condition: MPC starts from actual state x[K]
        for i in range(n_x):
            M.addConstr(self.x_mpc[K][0][i] == self.x[K][i],
                        name=f"mpc_init_{K}_{i}")

        self.sin_theta_mpc[K] = {}
        self.cos_theta_mpc[K] = {}
        self.theta_ddot_mpc[K] = {}
        for t in range(T):
            self.sin_theta_mpc[K][t] = M.addVar(lb=-1, ub=1,
                                                    name=f"sin_theta_mpc_{K}_{t}")
            self.cos_theta_mpc[K][t] = M.addVar(lb=-1, ub=1,
                                                    name=f"cos_theta_mpc_{K}_{t}")
            self.theta_ddot_mpc[K][t] = M.addVar(lb=-GRB.INFINITY,
                                                    name=f"theta_ddot_mpc_{K}_{t}")
            M.addGenConstrSin(self.x_mpc[K][t][0], self.sin_theta_mpc[K][t],
                                name=f"sin_mpc_{K}_{t}")
            M.addGenConstrCos(self.x_mpc[K][t][0], self.cos_theta_mpc[K][t],
                                name=f"cos_mpc_{K}_{t}")
            M.addConstr(self.x_mpc[K][t+1][0] ==
                        self.x_mpc[K][t][0] + dt * self.x_mpc[K][t][1],
                        name=f"mpc_theta_{K}_{t}")
            M.addConstr(self.theta_ddot_mpc[K][t] ==
                        g/length * self.sin_theta_mpc[K][t]
                        + self.u_mpc_var[K][t][0] / (mass * length**2),
                        name=f"mpc_angular_accel_{K}_{t}")
            M.addConstr(self.x_mpc[K][t+1][1] ==
                        self.x_mpc[K][t][1] + dt * self.theta_ddot_mpc[K][t],
                        name=f"mpc_theta_dot_{K}_{t}")

        self.lambda_mpc[K] = {}
        for t in range(T + 1):
            self.lambda_mpc[K][t] = M.addVars(n_x, lb=-GRB.INFINITY,
                                                name=f"lambda_mpc_{K}_{t}")
        for i in range(n_x):
            M.addConstr(self.lambda_mpc[K][T][i] == Q[i, i] * self.x_mpc[K][T][i],
                        name=f"terminal_costate_{K}_{i}")
        for t in range(T - 1, -1, -1):
            M.addConstr(
                self.lambda_mpc[K][t][0] == Q[0, 0] * self.x_mpc[K][t][0]
                + self.lambda_mpc[K][t+1][0]
                + dt * g / length * self.cos_theta_mpc[K][t] * self.lambda_mpc[K][t+1][1],
                name=f"costate_0_{K}_{t}"
            )
            M.addConstr(
                self.lambda_mpc[K][t][1] == Q[1, 1] * self.x_mpc[K][t][1]
                + dt * self.lambda_mpc[K][t+1][0] + self.lambda_mpc[K][t+1][1],
                name=f"costate_1_{K}_{t}"
            )
        for t in range(T):
            M.addConstr(
                R[0, 0] * self.u_mpc_var[K][t][0]
                + dt / (mass * length**2) * self.lambda_mpc[K][t+1][1] == 0,
                name=f"control_opt_{K}_{t}"
            )

        # Worst-case objective
        # if V_0_max is None:
        #     # Original: Lyapunov candidate V(x) = ||x||^2
        #     V_curr = gp.quicksum(self.x[0][i] * self.x[0][i] for i in range(n_x))
        #     V_next = gp.quicksum(self.x[K][i] * self.x[K][i] for i in range(n_x))
        # else:
        # Phase 2: Lyapunov candidate V(x) = J*(x) = MPC value function
        V_curr = (
            gp.quicksum(Q[i, i] * self.x_mpc[0][t][i] * self.x_mpc[0][t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R[j, j] * self.u_mpc_var[0][t][j] * self.u_mpc_var[0][t][j]
                            for t in range(T) for j in range(n_u))
        )
        V_next = (
            gp.quicksum(Q[i, i] * self.x_mpc[K][t][i] * self.x_mpc[K][t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R[j, j] * self.u_mpc_var[K][t][j] * self.u_mpc_var[K][t][j]
                            for t in range(T) for j in range(n_u))
        )
        # # Restrict to initial conditions reachable under the MPC policy
        # M.addConstr(V_curr <= V_0_max, name="V0_bound")

        # V_curr = gp.quicksum(self.x[0][i] * self.x[0][i] for i in range(n_x))
        # V_next = gp.quicksum(self.x[K][i] * self.x[K][i] for i in range(n_x))

        # import pdb
        # pdb.set_trace()

        self.V_curr = V_curr
        self.V_next = V_next

        eps = 1 - rho
        self.orig_objective = 0
        M.addConstr(V_next - V_curr + eps * V_curr >= 1e-6, name="V_curr_pos")
        M.setObjective(self.orig_objective, GRB.MAXIMIZE)

    def solve(self):
        self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        def v2dict(vs): return {k: vs[k].X for k in vs}

        if self.model.SolCount == 0:
            return {"obj": None}

        out = {
            "obj": self.model.ObjVal,

            "V_next": self.V_next.getValue(),
            "V_curr": self.V_curr.getValue(),

            # State trajectory: x[0] to x[K]
            "x": {k: v2dict(self.x[k]) for k in range(self.K + 1)},

            # Control trajectory: u[0] to u[K-1]
            "u": {k: v2dict(self.u[k]) for k in range(self.K)},

            # MPC trajectories
            "x_mpc": {k: {t: v2dict(self.x_mpc[k][t]) for t in range(len(self.x_mpc[k]))}
                     for k in range(self.K)},
            "u_mpc": {k: {t: v2dict(self.u_mpc_var[k][t]) for t in range(len(self.u_mpc_var[k]))}
                     for k in range(self.K)},
            "lambda_mpc": {k: {t: v2dict(self.lambda_mpc[k][t]) for t in range(len(self.lambda_mpc[k]))}
                          for k in range(self.K)},

            # Auxiliary variables
            "sin_theta": {k: self.sin_theta[k].X for k in range(self.K)},
            "cos_theta": {k: self.cos_theta[k].X for k in range(self.K)},
            "theta_ddot": {k: self.theta_ddot[k].X for k in range(self.K)},
        }

        return out
