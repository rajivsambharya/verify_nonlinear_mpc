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

    T = cfg.T
    r = 0 #0.001 #0.01  # Small positive value to avoid degenerate optimality conditions
    dt = cfg.dt

    x_min = x_mins[0]
    x_max = x_maxes[0]
    rho_max = cfg.rho_max

    sols = []
    times = np.zeros(K)
    opt_vals = np.zeros(K)

    for k in range(K):
        rho_lo = 0.0
        rho_hi = rho_max
        best_rho = None
        best_sol = None
        best_time = None

        while rho_hi - rho_lo > 1e-3:
            rho_mid = (rho_lo + rho_hi) / 2.0

            ver = CartpoleILQRVerify(
                n=cfg.n, K=k+1, T=T, r=r, dt=dt,
                mass=1, length=1, g=9.8, rho=rho_mid,
                x_lo=x_min, x_hi=x_max, seed=42, verbose=True
            )

            status, time = ver.solve()

            print(f"k={k+1}, rho={rho_mid:.6f}, status:", status)
            sol = ver.solution_dict()
            print("Objective:", sol["obj"])

            if status == GRB.OPTIMAL:
                rho_lo = rho_mid
            else:
                rho_hi = rho_mid
                best_rho = rho_mid
                best_sol = sol
                best_time = time

        if best_sol is not None:
            print(f"Best verified rho for k={k+1}: {best_rho:.6f}")
            opt_vals[k] = rho_hi
            sols.append(best_sol)
            times[k] = best_time
        else:
            print(f"Could not verify any rho <= {rho_max} for k={k+1}")
            opt_vals[k] = np.inf
            sols.append(None)
            times[k] = 0

    plt.plot(np.arange(K) + 1, opt_vals)
    plt.xlabel('iterations')
    plt.ylabel('rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rates_ilqr.pdf', bbox_inches='tight')


class CartpoleILQRVerify:
    def __init__(self, n=10, K=3, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 rho=0.2, x_lo=0.0, x_hi=4.0, seed=None, verbose=True):
        self.n, self.K, self.rho = n, K, rho
        self.verbose = bool(verbose)
        rng = np.random.default_rng(seed)

        n_x = 2
        n_u = 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * r
        u_max = 10

        # Model
        M = gp.Model("cartpole_ilqr_verify")
        M.Params.OutputFlag = 1 if self.verbose else 0
        M.Params.FeasibilityTol = 1e-9
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
            self.x[k] = M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_{k}")

        # Create variables and constraints for K timesteps
        for k in range(K):
            # Control at timestep k (applied from state k)
            self.u[k] = M.addVars(n_u, lb=-u_max, ub=u_max, name=f"u_{k}")

            # MPC variables: solve T-horizon MPC at each timestep k
            # MPC states: x_mpc[k][t] for t=0..T
            self.x_mpc[k] = {}
            for t in range(T + 1):
                self.x_mpc[k][t] = M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_mpc_{k}_{t}")

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

        # Worst-case objective
        V_curr = gp.quicksum(self.x[0][i] * self.x[0][i] for i in range(n_x))
        V_next = gp.quicksum(self.x[K][i] * self.x[K][i] for i in range(n_x))
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
