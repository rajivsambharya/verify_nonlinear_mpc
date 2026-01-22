import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
cmap = plt.cm.Set1
colors = cmap.colors
import cvxpy as cp
from scipy import sparse
# Solve MPC QP with linearized dynamics
from scipy.optimize import minimize



FONT_SIZE = 44 #33 #36 # 38
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE
})
# Optional: give ticks their own family
plt.rc("xtick", labelsize=FONT_SIZE, labelcolor="black")
plt.rc("ytick", labelsize=FONT_SIZE, labelcolor="black")


def run(cfg):
    K = cfg.K
    
    inits = ['warm_start'] #, 'cold_start']
    all_opt_vals = []
    all_sols = []
    all_times = []
    all_sample_maxes = []
    x_mins = cfg.x_mins
    x_maxes = cfg.x_maxes

    T = 5
    # K_sim = cfg.K
    r = 0 #0.00001 #0.1
    dt = 0.1
    mass = 1
    length = 1

    x_min = x_mins[0]
    x_max = x_maxes[0]
    rho_max = cfg.rho_max

    for j in range(len(x_mins)):
        
        curr_init_opt_vals = []
        curr_init_sols = []
        curr_init_times = []
        curr_sample_maxes = []
        for i in range(len(inits)):
            sols = []
            times = np.zeros(K)
            opt_vals = np.zeros(K)
            warm_start_x = None
            warm_start_bd = None
            for k in range(K):
                
                ver = CartpoleVerify(n=cfg.n, K=k+1, T=T, r=r, dt=dt, mass=1, length=1, g=9.8, rho=cfg.rho, x_lo=x_min, x_hi=x_max, seed=42, verbose=True)

                status, time = ver.solve()
                print("solve() status:", status)
                sol = ver.solution_dict()
                print("Objective:", sol["obj"])

                x0_init = np.array([sol['x'][0][0], sol['x'][0][1]])
                sim_out = simulate_layered_controller(x0_init, k+1, T=T, r=r, dt=dt, mass=1, length=1, g=9.8)
                import pdb
                pdb.set_trace()

                opt_vals[j] = sol["obj"]
                sols.append(sol)
                times[j] = time

            curr_init_opt_vals.append(opt_vals)
            curr_init_sols.append(sols)
            curr_init_times.append(times)

            sample_maxes = find_sample_maxes(K, cfg.num_samples, ver.P, ver.H, ver.z0, x_min, x_max, cfg.rho)
            curr_sample_maxes.append(sample_maxes)

        all_opt_vals.append(curr_init_opt_vals)
        all_sols.append(curr_init_sols)
        all_times.append(curr_init_times)
        all_sample_maxes.append(curr_sample_maxes)

    ################################ journal figure
    FONT_SIZE = 42 #33 #36 # 38
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",   # For talks, use sans-serif
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE
    })
    # Optional: give ticks their own family
    plt.rc("xtick", labelsize=FONT_SIZE, labelcolor="black")
    plt.rc("ytick", labelsize=FONT_SIZE, labelcolor="black")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # First subplot (linear scale)
    axes[0].plot(np.arange(K), all_opt_vals[0][0], color=colors[0], marker='s', markevery=(0, 1))
    axes[0].plot(np.arange(K), all_opt_vals[0][1], color=colors[1], marker='o', markevery=(0, 1))
    axes[0].plot(np.arange(K), all_sample_maxes[0][0], color=colors[0], linestyle=':', marker='s', markerfacecolor='none', markevery=(0, 1))
    axes[0].plot(np.arange(K), all_sample_maxes[0][1], color=colors[1], linestyle=':', marker='o', markerfacecolor='none', markevery=(0, 1))
    axes[0].set_xlabel('iterations')
    axes[0].set_ylabel('worst-case suboptimality')
    axes[0].grid(True)
    axes[0].set_title(r'parameter set $\mathcal{X}_1=[2,4]^d$')
    
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Force at least 3 ticks
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=3))

    # Second subplot (log scale)
    axes[1].plot(np.arange(K), all_opt_vals[1][0], color=colors[0], marker='s', markevery=(0, 1))
    axes[1].plot(np.arange(K), all_opt_vals[1][1], color=colors[1], marker='o', markevery=(0, 1))
    axes[1].plot(np.arange(K), all_sample_maxes[1][0], color=colors[0], linestyle=':', marker='s', markerfacecolor='none', markevery=(0, 1))
    axes[1].plot(np.arange(K), all_sample_maxes[1][1], color=colors[1], linestyle=':', marker='o', markerfacecolor='none', markevery=(0, 1))
    axes[1].set_xlabel('iterations')
    axes[1].grid(True)
    axes[1].set_yscale('log')
    axes[1].set_title(r'parameter set $\mathcal{X}_2=[5,8]^d$')

    plt.tight_layout()
    plt.savefig('suboptimality_journal.pdf', bbox_inches='tight')
    plt.clf()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # First subplot (linear scale)
    axes[0].plot(np.arange(K), all_times[0][0], color=colors[0], marker='s', markevery=(0, 1))
    axes[0].plot(np.arange(K), all_times[0][1], color=colors[1], marker='o', markevery=(0, 1))
    axes[0].set_xlabel('iterations')
    axes[0].set_ylabel('solve time (seconds)')
    axes[0].grid(True)
    axes[0].set_title(r'parameter set $\mathcal{X}_1$')
    
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Second subplot (log scale)
    axes[1].plot(np.arange(K), all_times[1][0], color=colors[0], marker='s', markevery=(0, 1))
    axes[1].plot(np.arange(K), all_times[1][1], color=colors[1], marker='o', markevery=(0, 1))
    axes[1].set_xlabel('iterations')
    axes[1].grid(True)
    axes[1].set_yscale('log')
    axes[1].set_title(r'parameter set $\mathcal{X}_2$')

    plt.tight_layout()
    plt.savefig('times_journal.pdf')


    ################################ preprint figure
    FONT_SIZE = 33 #36 # 38
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",   # For talks, use sans-serif
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE
    })
    # Optional: give ticks their own family
    plt.rc("xtick", labelsize=FONT_SIZE, labelcolor="black")
    plt.rc("ytick", labelsize=FONT_SIZE, labelcolor="black")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    # fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # First subplot (linear scale)
    axes[0].plot(np.arange(K), all_opt_vals[0][0], color=colors[0], marker='s', markevery=(0, 1))
    axes[0].plot(np.arange(K), all_opt_vals[0][1], color=colors[1], marker='o', markevery=(0, 1))
    axes[0].plot(np.arange(K), all_sample_maxes[0][0], color=colors[0], linestyle=':', marker='s', markerfacecolor='none', markevery=(0, 1))
    axes[0].plot(np.arange(K), all_sample_maxes[0][1], color=colors[1], linestyle=':', marker='o', markerfacecolor='none', markevery=(0, 1))
    axes[0].set_xlabel('iterations')
    axes[0].set_ylabel('worst-case suboptimality')
    axes[0].grid(True)
    axes[0].set_title(r'parameter set $\mathcal{X}_1=[2,4]^d$')
    
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Force at least 3 ticks
    # axes[0].yaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=3))

    # Second subplot (log scale)
    axes[1].plot(np.arange(K), all_opt_vals[1][0], color=colors[0], marker='s', markevery=(0, 1))
    axes[1].plot(np.arange(K), all_opt_vals[1][1], color=colors[1], marker='o', markevery=(0, 1))
    axes[1].plot(np.arange(K), all_sample_maxes[1][0], color=colors[0], linestyle=':', marker='s', markerfacecolor='none', markevery=(0, 1))
    axes[1].plot(np.arange(K), all_sample_maxes[1][1], color=colors[1], linestyle=':', marker='o', markerfacecolor='none', markevery=(0, 1))
    axes[1].set_xlabel('iterations')
    axes[1].grid(True)
    axes[1].set_yscale('log')
    axes[1].set_title(r'parameter set $\mathcal{X}_2=[5,8]^d$')

    plt.tight_layout()
    plt.savefig('suboptimality_preprint.pdf', bbox_inches='tight')
    plt.clf()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # First subplot (linear scale)
    axes[0].plot(np.arange(K), all_times[0][0], color=colors[0], marker='s', markevery=(0, 1))
    axes[0].plot(np.arange(K), all_times[0][1], color=colors[1], marker='o', markevery=(0, 1))
    axes[0].set_xlabel('iterations')
    axes[0].set_ylabel('solve time (seconds)')
    axes[0].grid(True)
    axes[0].set_title(r'parameter set $\mathcal{X}_1$')
    
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Second subplot (log scale)
    axes[1].plot(np.arange(K), all_times[1][0], color=colors[0], marker='s', markevery=(0, 1))
    axes[1].plot(np.arange(K), all_times[1][1], color=colors[1], marker='o', markevery=(0, 1))
    axes[1].set_xlabel('iterations')
    axes[1].grid(True)
    axes[1].set_yscale('log')
    axes[1].set_title(r'parameter set $\mathcal{X}_2$')

    plt.tight_layout()
    plt.savefig('times_preprint.pdf')




class CartpoleVerify:
    def __init__(self, n=10, K=3, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8, rho=0.2, x_lo=0.0, x_hi=4.0, warm_start_bd=None, seed=None, verbose=True):
        self.n, self.K, self.rho = n, K, rho
        self.verbose = bool(verbose)
        rng = np.random.default_rng(seed)

        # generate problem data
        # g = 9.8
        # length = 1
        # dt = 0.1
        # mass = 1
        n_x = 2
        n_u = 1
        Q = np.eye(n_x)
        # R = np.eye(n_u) * .01
        R = np.eye(n_u) * r #.1
        # T = 5
        u_max = 10

        theta_0 = 0
        self.A_dyn, self.B_dyn = get_dynamics_at_angle(theta_0, dt=dt, mass=mass, length=length, g=g)
        P, c, A, b_const, cone_dims, z_slices = form_mpc_qp(self.A_dyn, self.B_dyn, Q, R, T, u_max)
        n_eq = cone_dims['zero']

        n = A.shape[1]
        m = A.shape[0]

        # Model
        M = gp.Model("cartpole_verify")
        M.Params.OutputFlag = 1 if self.verbose else 0
        M.Params.FeasibilityTol = 1e-9
        self.model = M

        # Store state/control/reference trajectories
        self.x = {}      # states: x[k] for k=0..K
        self.u_lower = {}  # lower-level controls: u_lower[k] for k=0..K-1
        self.x_ref = {}  # reference states from MPC: x_ref[k] for k=1..K
        
        # Store MPC variables for each timestep
        self.u_mpc = {}      # MPC decision variables
        self.s = {}          # MPC slack variables
        self.mu = {}         # MPC dual variables
        self.b = {}          # MPC RHS
        self.A_lin = {}      # Linearized A matrices
        
        # Auxiliary variables for dynamics
        self.sin_theta = {}
        self.cos_theta = {}
        self.theta_ddot = {}
        self.pred_error = {}
        
        # Fixed B matrix (doesn't depend on state)
        self.B_lin = np.zeros((2, 1))
        self.B_lin[0, 0] = 0
        self.B_lin[1, 0] = dt / (mass * length ** 2)

        # CREATE ALL STATE VARIABLES FIRST (k=0 to K)
        for k in range(K + 1):
            self.x[k] = M.addVars(n_x, lb=x_lo, ub=x_hi, name=f"x_{k}")
        
        # Create variables and constraints for K timesteps
        for k in range(K):
            # Lower-level control at timestep k
            self.u_lower[k] = M.addVars(n_u, lb=-GRB.INFINITY, name=f"u_lower_{k}")
            
            # Reference state from MPC (for k+1)
            self.x_ref[k+1] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_ref_{k+1}")
            
            # MPC variables for timestep k
            self.u_mpc[k] = M.addVars(n, lb=-GRB.INFINITY, name=f"u_mpc_{k}")
            self.s[k] = M.addVars(m, lb=-GRB.INFINITY, name=f"s_{k}")
            self.mu[k] = M.addVars(m, lb=-GRB.INFINITY, name=f"mu_{k}")
            self.b[k] = M.addVars(m, lb=-GRB.INFINITY, name=f"b_{k}")
            
            # Cone constraints (equality vs inequality)
            for i in range(m):
                if i < n_eq:
                    M.addConstr(self.s[k][i] == 0, name=f"s_{k}_{i}")
                else:
                    M.addConstr(self.s[k][i] >= 0, name=f"s_{k}_{i}")
                    M.addConstr(self.mu[k][i] >= 0, name=f"mu_{k}_{i}")
            
            # Linearized dynamics matrices
            self.A_lin[k] = M.addVars(n_x, n_x, lb=-GRB.INFINITY, name=f"A_{k}")
            
            # Set RHS vector b for MPC (initial condition + constants)
            for i in range(n_x):
                M.addConstr(self.b[k][i] == self.x[k][i], name=f"b_{k}_{i}")
            for i in range(n_x, m):
                M.addConstr(self.b[k][i] == b_const[i], name=f"b_{k}_{i}")
            
            # MPC KKT conditions
            # Primal feasibility: A u + s = b
            for r in range(m):
                lhs = gp.quicksum(A[r, j] * self.u_mpc[k][j] for j in range(n))
                M.addConstr(lhs + self.s[k][r] == self.b[k][r], name=f"pr_{k}_{r}")
            
            # Dual feasibility: A^T mu + P u = -c
            for i in range(n):
                Atu = gp.quicksum(A[r, i] * self.mu[k][r] for r in range(m))
                Pu = gp.quicksum(P[i, j] * self.u_mpc[k][j] for j in range(n))
                M.addConstr(Atu + Pu == -c[i], name=f"du_{k}_{i}")
            
            # Complementarity: s_r * mu_r = 0
            for r in range(m):
                M.addConstr(self.s[k][r] * self.mu[k][r] == 0, name=f"comp_{k}_{r}")
            
            # Link MPC solution to reference state
            M.addConstr(self.x_ref[k+1][0] == self.u_mpc[k][n_x + n_u], name=f"link_xref_{k}_0")
            M.addConstr(self.x_ref[k+1][1] == self.u_mpc[k][n_x + n_u + 1], name=f"link_xref_{k}_1")
            
            # Auxiliary variables for nonlinear dynamics
            self.sin_theta[k] = M.addVar(lb=-1, ub=1, name=f"sin_theta_{k}")
            self.cos_theta[k] = M.addVar(lb=-1, ub=1, name=f"cos_theta_{k}")
            self.theta_ddot[k] = M.addVar(lb=-GRB.INFINITY, name=f"theta_ddot_{k}")
            
            # Nonlinear dynamics constraints
            M.addGenConstrSin(self.x[k][0], self.sin_theta[k], name=f"sin_{k}")
            M.addGenConstrCos(self.x[k][0], self.cos_theta[k], name=f"cos_{k}")
            
            # theta_{k+1} = theta_k + dt * theta_dot_k
            M.addConstr(
                self.x[k+1][0] == self.x[k][0] + dt * self.x[k][1],
                name=f"integrate_theta_{k}"
            )
            
            # theta_ddot = (g/L) * sin(theta) + u_lower / (m*L²)
            M.addConstr(
                self.theta_ddot[k] == g/length * self.sin_theta[k] + self.u_lower[k][0] / (mass * length**2),
                name=f"angular_accel_{k}"
            )
            
            # theta_dot_{k+1} = theta_dot_k + dt * theta_ddot
            M.addConstr(
                self.x[k+1][1] == self.x[k][1] + dt * self.theta_ddot[k],
                name=f"integrate_theta_dot_{k}"
            )
            
            # Linearized dynamics A matrix
            M.addConstr(self.A_lin[k][0, 0] == 1, name=f"A_lin_{k}_0_0")
            M.addConstr(self.A_lin[k][0, 1] == dt, name=f"A_lin_{k}_0_1")
            M.addConstr(self.A_lin[k][1, 0] == dt * g / length * self.cos_theta[k], name=f"A_lin_{k}_1_0")
            M.addConstr(self.A_lin[k][1, 1] == 1, name=f"A_lin_{k}_1_1")
            
            # Prediction error: e = x_{k+1} - x_ref_{k+1}
            self.pred_error[k] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"pred_error_{k}")
            for i in range(n_x):
                Bu = gp.quicksum(self.B_lin[i, r] * self.u_lower[k][r] for r in range(n_u))
                Ax = gp.quicksum(self.A_lin[k][i, r] * self.x[k][r] for r in range(n_x))
                M.addConstr(
                    # self.pred_error[k][i] == self.x[k+1][i] + Bu - self.x_ref[k+1][i], 
                    self.pred_error[k][i] == Ax + Bu - self.x_ref[k+1][i],
                    name=f"pred_error_{k}_{i}"
                )
            
            # LQR controller optimality: B^T Q e + R u_lower = 0
            B_lin_Q = self.B_lin.T @ Q
            for i in range(n_u):
                BQe = gp.quicksum(B_lin_Q[i, r] * self.pred_error[k][r] for r in range(n_x))
                M.addConstr(BQe + R[0, 0] * self.u_lower[k][i] == 0, name=f"lqr_opt_{k}_{i}")
        
        # Objective: maximize terminal state cost
        # self.orig_objective = gp.quicksum(self.x[K][i] * self.x[K][i] for i in range(n_x))
        # M.setObjective(self.orig_objective, GRB.MAXIMIZE)


        # Worst-case objective
        V_curr = gp.quicksum(self.x[0][i] * self.x[0][i] for i in range(n_x)) #+ R[0,0] * self.u_lower[0] * self.u_lower[0]
        V_next = gp.quicksum(self.x[K][i] * self.x[K][i] for i in range(n_x)) #+ R[0,0] * self.u_lower[0] * self.u_lower[0]
        # self.orig_objective = V_next - V_curr + (-.1) * V_curr
        self.orig_objective = V_next - V_curr + rho * V_curr

        # M.addConstr(V_curr >= 1e-4)
        # self.orig_objective = - + gp.quicksum(self.x1[i] * self.x1[i] for i in range(n_x))
        # self.orig_objective = gp.quicksum((self.x1[i]  - self.x_ref1[i]) * (self.x1[i] - self.x_ref1[i]) for i in range(n_x))


        # if warm_start_bd is not None:
        #     M.addConstr(self.orig_objective <= warm_start_bd)
        M.setObjective(self.orig_objective, GRB.MAXIMIZE)

        # Cache flat var list for OBBT convenience
        self.all_vars = []
        # for coll in (self.x, *self.u, *self.u0, *self.c, *self.b, *self.s, *self.mu):
        #     self.all_vars += list(coll.values())
        # for coll in (self.x0, *self.u, *self.b, *self.s, *self.mu):
        #     self.all_vars += list(coll.values())

    def solve(self):
        self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    # def solution_dict(self):
    #     def v2dict(vs): return {k: vs[k].X for k in vs}
    #     out = {
    #         "obj": None if self.model.SolCount == 0 else self.model.ObjVal,
    #         "x0":   None if self.model.SolCount == 0 else v2dict(self.x[0]),
    #         "x1":   None if self.model.SolCount == 0 else v2dict(self.x[1]),
    #         "x2":   None if self.model.SolCount == 0 else v2dict(self.x[2]),
    #         "x_fin":   None if self.model.SolCount == 0 else v2dict(self.x[self.K]),
    #         "x_ref1":   None if self.model.SolCount == 0 else v2dict(self.x_ref[1]),
    #         "u_lower0":   None if self.model.SolCount == 0 else v2dict(self.u_lower[0]),
    #         "u_mpc0":   None if self.model.SolCount == 0 else v2dict(self.u_mpc[0]),
    #         # # "cos_theta":  None if self.model.SolCount == 0 else v2dict(self.cos_theta),
    #         # "cos_theta":  None if self.model.SolCount == 0 else self.cos_theta.X,
    #         # "sin_theta":  None if self.model.SolCount == 0 else self.sin_theta.X,
    #         # "x_ref1":   None if self.model.SolCount == 0 else v2dict(self.x_ref1),
    #         # "u":  None if self.model.SolCount == 0 else v2dict(self.u),
    #         # "s":  None if self.model.SolCount == 0 else v2dict(self.s),
    #         # "mu":  None if self.model.SolCount == 0 else v2dict(self.mu),
    #         # "b":  None if self.model.SolCount == 0 else v2dict(self.b),
    #         # "u_lower":  None if self.model.SolCount == 0 else v2dict(self.u_lower),
    #     }
    #     return out

    def solution_dict(self):
        def v2dict(vs): return {k: vs[k].X for k in vs}
        
        if self.model.SolCount == 0:
            return {"obj": None}
        
        out = {
            "obj": self.model.ObjVal,
            
            # State trajectory: x[0] to x[K]
            "x": {k: v2dict(self.x[k]) for k in range(self.K + 1)},
            
            # Control trajectory: u_lower[0] to u_lower[K-1]
            "u_lower": {k: v2dict(self.u_lower[k]) for k in range(self.K)},
            
            # Reference trajectory: x_ref[1] to x_ref[K]
            "x_ref": {k: v2dict(self.x_ref[k]) for k in range(1, self.K + 1)},
            
            # MPC decision variables: u_mpc[0] to u_mpc[K-1]
            "u_mpc": {k: v2dict(self.u_mpc[k]) for k in range(self.K)},
            
            # MPC slack/dual variables
            "s": {k: v2dict(self.s[k]) for k in range(self.K)},
            "mu": {k: v2dict(self.mu[k]) for k in range(self.K)},
            "b": {k: v2dict(self.b[k]) for k in range(self.K)},
            
            # Linearized dynamics matrices
            "A_lin": {k: v2dict(self.A_lin[k]) for k in range(self.K)},
            
            # Auxiliary variables
            "sin_theta": {k: self.sin_theta[k].X for k in range(self.K)},
            "cos_theta": {k: self.cos_theta[k].X for k in range(self.K)},
            "theta_ddot": {k: self.theta_ddot[k].X for k in range(self.K)},
            
            # Prediction errors
            "pred_error": {k: v2dict(self.pred_error[k]) for k in range(self.K)},
        }
        
        return out
    

def simulate_layered_controller(x0_init, K_sim, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8):
    """
    Simulate the layered controller starting from x0_init for K_sim timesteps.
    Uses the EXACT same MPC formulation as the verification problem.
    
    Args:
        x0_init: Initial state [theta, theta_dot]
        K_sim: Number of simulation steps (defaults to self.K)
        dt, mass, length, g: Dynamics parameters
    
    Returns:
        Dictionary with state/control/reference trajectories
    """
    
    # Get MPC problem data - use the EXACT same formulation
    Q = np.eye(2)
    R = np.eye(1) * r
    u_max = 1
    
    # Get the QP matrices from your form_mpc_qp function
    theta_0 = 0  # Linearization point (you're relinearizing at each step)
    A_dyn_ref, B_dyn_ref = get_dynamics_at_angle(theta_0, dt=dt, mass=mass, length=length, g=g)
    P, c, A, b_const, cone_dims, z_slices = form_mpc_qp(A_dyn_ref, B_dyn_ref, Q, R, T, u_max)
    n_eq = cone_dims['zero']
    
    n = A.shape[1]
    m = A.shape[0]
    n_x = 2
    n_u = 1
    
    # Storage for trajectories
    x_traj = np.zeros((K_sim + 1, 2))
    u_lower_traj = np.zeros((K_sim, 1))
    x_ref_traj = np.zeros((K_sim + 1, 2))
    u_mpc_traj = []
    pred_error_traj = np.zeros((K_sim, 2))
    A_lin_traj = []
    
    x_traj[0] = x0_init
    
    # B matrix for linearized dynamics (constant)
    B_lin = np.zeros((2, 1))
    B_lin[0, 0] = 0
    B_lin[1, 0] = dt / (mass * length ** 2)
    
    for k in range(K_sim):
        x_k = x_traj[k]
        
        # === UPPER LAYER: MPC with linearized dynamics ===
        # Linearize dynamics around current state
        theta_k = x_k[0]
        A_lin = np.array([
            [1, dt],
            [dt * g / length * np.cos(theta_k), 1]
        ])
        A_lin_traj.append(A_lin)
        
        # Solve MPC using the EXACT same QP formulation as verification
        # The QP is: min 0.5 * u^T P u + c^T u
        #            s.t. A u + s = b
        #                 s in K (cone constraints)
        
        # Update b vector with current initial condition
        b = b_const.copy()
        b[:n_x] = x_k  # Set initial condition
        
        # Solve using cvxpy with Clarabel for high accuracy
        u = cp.Variable(n)
        s = cp.Variable(m)
        
        # Objective: 0.5 * u^T P u + c^T u
        objective = cp.Minimize(0.5 * cp.quad_form(u, P) + c @ u)
        
        # Constraints
        constraints = [A @ u + s == b]
        
        # Cone constraints
        for i in range(m):
            if i < n_eq:
                # Equality constraint (zero cone)
                constraints.append(s[i] == 0)
            else:
                # Inequality constraint (non-negative cone)
                constraints.append(s[i] >= 0)
        
        # Solve with Clarabel
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"MPC failed at step {k}: {problem.status}")
            break
        
        # Extract solution
        u_sol = u.value
        u_mpc_traj.append(u_sol)
        
        # Extract reference for next timestep (x_1 from the MPC solution)
        # The MPC variables are ordered as: [x_0, u_0, x_1, u_1, x_2, ...]
        x_ref_k1 = u_sol[n_x + n_u : n_x + n_u + n_x]
        x_ref_traj[k + 1] = x_ref_k1
        
        # === LOWER LAYER: LQR tracking controller ===
        # Solve: B_lin^T Q (x_{k+1} - x_ref_{k+1}) + R u_lower = 0
        # where x_{k+1} depends on u_lower through true dynamics
        
        # For the inverted pendulum, we can solve this analytically or numerically
        # The optimality condition is:
        # B_lin^T Q (x_{k+1}(u_lower) - x_ref_{k+1}) + R u_lower = 0
        
        # Since only theta_dot is affected by u_lower:
        # B_lin^T Q e + R u_lower = 0
        # where e = [e_theta, e_theta_dot]
        # and e_theta_dot = theta_dot_{k+1} - x_ref_{k+1}[1]
        
        # Note: B_lin = [0; dt/(m*L^2)]
        # So: B_lin^T Q e = [0, dt/(m*L^2)] @ Q @ [e_theta, e_theta_dot]
        #                 = Q[1,0] * e_theta + Q[1,1] * e_theta_dot
        #                 = Q[1,1] * e_theta_dot  (since Q is diagonal)
        
        # The optimality condition becomes:
        # Q[1,1] * e_theta_dot + R * u_lower = 0
        # where e_theta_dot = (theta_dot_k + dt * theta_ddot) - x_ref_{k+1}[1]
        # and theta_ddot = g/L * sin(theta_k) + u_lower / (m*L^2)
        
        # Solving:
        # Q[1,1] * (theta_dot_k + dt * (g/L * sin(theta_k) + u_lower/(m*L^2)) - x_ref_{k+1}[1]) + R * u_lower = 0
        # Q[1,1] * (theta_dot_k + dt*g/L*sin(theta_k) - x_ref_{k+1}[1]) + Q[1,1] * dt/(m*L^2) * u_lower + R * u_lower = 0
        # u_lower * (Q[1,1] * dt/(m*L^2) + R) = -Q[1,1] * (theta_dot_k + dt*g/L*sin(theta_k) - x_ref_{k+1}[1])
        

        def lqr_cost(u_lower_val):
            u_lower_val = u_lower_val[0]
            
            # True dynamics
            x_k1_true = np.array([
                x_k[0] + dt * x_k[1],
                x_k[1] + dt * (g / length * np.sin(x_k[0]) + u_lower_val / (mass * length ** 2))
            ])
            
            # Prediction error as defined in your code
            Bu = B_lin[:, 0] * u_lower_val
            pred_error = x_k1_true + Bu - x_ref_k1
            
            # LQR optimality condition: B_lin^T Q pred_error + R u_lower = 0
            # We want this to be satisfied, so let's return the residual
            gradient = B_lin[:, 0] @ Q @ pred_error + R[0, 0] * u_lower_val
            
            return gradient ** 2  # Minimize squared residual
        
        # from scipy.optimize import minimize
        result = minimize(lqr_cost, [0.0], method='BFGS', tol=1e-12)
        u_lower_k = result.x[0]
        u_lower_traj[k] = u_lower_k
        
        # === TRUE DYNAMICS: Propagate state ===
        theta_k1 = x_k[0] + dt * x_k[1]
        theta_ddot = g / length * np.sin(x_k[0]) + u_lower_k / (mass * length ** 2)
        theta_dot_k1 = x_k[1] + dt * theta_ddot
        x_traj[k + 1] = np.array([theta_k1, theta_dot_k1])
        
        # Compute prediction error (as defined in your code)
        Bu = B_lin[:, 0] * u_lower_k
        pred_error_traj[k] = x_traj[k + 1] + Bu - x_ref_traj[k + 1]
    
    return {
        "x": x_traj,
        "u_lower": u_lower_traj,
        "x_ref": x_ref_traj,
        "u_mpc": u_mpc_traj,
        "pred_error": pred_error_traj,
        "A_lin": A_lin_traj,
    }







        # theta_ddot_passive = g / length * np.sin(x_k[0])
        # residual = x_k[1] + dt * theta_ddot_passive - x_ref_k1[1]
        
        # # Coefficient of u_lower
        # coeff = Q[1, 1] * B_lin[1, 0] + R[0, 0]
        
        # # Solve for u_lower
        # u_lower_k = -Q[1, 1] * residual / coeff
        # u_lower_traj[k] = u_lower_k
        
        # # === TRUE DYNAMICS: Propagate state ===
        # theta_k1 = x_k[0] + dt * x_k[1]
        # theta_ddot = g / length * np.sin(x_k[0]) + u_lower_k / (mass * length ** 2)
        # theta_dot_k1 = x_k[1] + dt * theta_ddot
        # x_traj[k + 1] = np.array([theta_k1, theta_dot_k1])
        
        # # Compute prediction error
        # pred_error_traj[k] = x_traj[k + 1] - x_ref_traj[k + 1]
    
    return {
        "x": x_traj,
        "u_lower": u_lower_traj,
        "x_ref": x_ref_traj,
        "u_mpc": u_mpc_traj,
        "pred_error": pred_error_traj,
        "A_lin": A_lin_traj,
    }
    

def simulate_layered_controller_old(x0_init, K_sim, dt=0.01, mass=1, length=1, g=9.8):
    """
    Simulate the layered controller starting from x0_init for K_sim timesteps.
    
    Args:
        x0_init: Initial state [theta, theta_dot]
        K_sim: Number of simulation steps (defaults to self.K)
        dt, mass, length, g: Dynamics parameters
    
    Returns:
        Dictionary with state/control/reference trajectories
    """

    
    # Get MPC problem data (assuming it's already set up)
    Q = np.eye(2)
    R = np.eye(1) * 0.1
    T = 5
    u_max = 1
    
    # Storage for trajectories
    x_traj = np.zeros((K_sim + 1, 2))
    u_lower_traj = np.zeros((K_sim, 1))
    x_ref_traj = np.zeros((K_sim + 1, 2))
    u_mpc_traj = []
    pred_error_traj = np.zeros((K_sim, 2))
    
    x_traj[0] = x0_init
    
    # B matrix for linearized dynamics (constant)
    B_lin = np.zeros((2, 1))
    B_lin[0, 0] = 0
    B_lin[1, 0] = dt / (mass * length ** 2)
    
    for k in range(K_sim):
        x_k = x_traj[k]
        
        # === UPPER LAYER: MPC with linearized dynamics ===
        # Linearize dynamics around current state
        theta_k = x_k[0]
        A_lin = np.array([
            [1, dt],
            [dt * g / length * np.cos(theta_k), 1]
        ])
        
        # Build and solve MPC problem
        n_x = 2
        n_u = 1
        
        # Decision variables: [x_0, u_0, x_1, u_1, ..., x_T]
        n_vars = (T + 1) * n_x + T * n_u
        
        # For simplicity, use cvxpy to solve the MPC QP
        x_mpc = cp.Variable((T + 1, n_x))
        u_mpc = cp.Variable((T, n_u))
        
        # Cost: sum_t (x_t^T Q x_t + u_t^T R u_t)
        cost = 0
        for t in range(T):
            cost += .5 * cp.quad_form(x_mpc[t], Q) + .5 * cp.quad_form(u_mpc[t], R)
        cost += .5 * cp.quad_form(x_mpc[T], Q)  # Terminal cost
        
        # Constraints
        constraints = [
            x_mpc[0] == x_k  # Initial condition
        ]
        
        for t in range(T):
            # Linearized dynamics: x_{t+1} = A_lin @ x_t + B_lin @ u_t
            constraints.append(x_mpc[t + 1] == A_lin @ x_mpc[t] + B_lin @ u_mpc[t])
            # Control bounds
            constraints.append(u_mpc[t] <= u_max)
            constraints.append(u_mpc[t] >= -u_max)
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"MPC failed at step {k}: {problem.status}")
            break
        
        # Extract reference for next timestep
        x_ref_k1 = x_mpc.value[1]  # Second state in the trajectory
        x_ref_traj[k + 1] = x_ref_k1
        u_mpc_traj.append(u_mpc.value)
        
        # === LOWER LAYER: LQR tracking controller ===
        # The lower layer uses true nonlinear dynamics and tries to track x_ref_k1
        
        # We need to solve: min_{u_lower} ||x_{k+1} - x_ref_{k+1}||_Q^2 + ||u_lower||_R^2
        # where x_{k+1} = f_true(x_k, u_lower)
        
        # For the inverted pendulum:
        # theta_{k+1} = theta_k + dt * theta_dot_k
        # theta_dot_{k+1} = theta_dot_k + dt * (g/L * sin(theta_k) + u_lower / (m*L^2))
        
        # This is a simple 1D optimization problem
        def lower_cost(u_lower):
            u_lower = u_lower[0]
            # True dynamics
            theta_k1 = x_k[0] + dt * x_k[1]
            theta_ddot = g / length * np.sin(x_k[0]) + u_lower / (mass * length ** 2)
            theta_dot_k1 = x_k[1] + dt * theta_ddot
            x_k1_true = np.array([theta_k1, theta_dot_k1])
            
            # Tracking error
            error = x_k1_true - x_ref_k1
            
            # Cost
            return error.T @ Q @ error + u_lower ** 2 * R[0, 0]
        
        # Optimize
        result = minimize(lower_cost, [0.0], method='BFGS')
        u_lower_k = result.x[0]
        u_lower_traj[k] = u_lower_k
        
        # === TRUE DYNAMICS: Propagate state ===
        theta_k1 = x_k[0] + dt * x_k[1]
        theta_ddot = g / length * np.sin(x_k[0]) + u_lower_k / (mass * length ** 2)
        theta_dot_k1 = x_k[1] + dt * theta_ddot
        x_traj[k + 1] = np.array([theta_k1, theta_dot_k1])
        
        # Compute prediction error
        pred_error_traj[k] = x_traj[k + 1] - x_ref_traj[k + 1]
    
    return {
        "x": x_traj,
        "u_lower": u_lower_traj,
        "x_ref": x_ref_traj,
        "u_mpc": u_mpc_traj,
        "pred_error": pred_error_traj,
    }

# def encode_dyn(x1, x0, u):
#     x1[0] == x0[0] + dt * x0[1]
#     theta_ddot = g / L * np.sin(theta) + u / (m * L**2)
#     x1[1] == x0[1] + dt * theta_ddot     

def find_sample_maxes(K, num_samples, P, H, z0, x_min, x_max, rho):
    sample_maxes = np.zeros(K)
    sample_vals = np.zeros((num_samples, K))
    d = P.shape[0]

    x_samples = x_min + np.random.rand(num_samples, d) * (x_max - x_min)

    for i in range(num_samples):
        curr_x = x_samples[i, :]

        z_star, z_sols, suboptimalities = run_trust_region_method(K, P, H, z0, curr_x, rho)

        sample_vals[i, :] = suboptimalities
        
    sample_maxes = np.max(sample_vals, axis=0)
    return sample_maxes


def solve_exact(P, x):
    n_orig = P.shape[1]

    # Create model
    m = gp.Model("warm_start_min")
    m.Params.NonConvex = 2  # allow nonconvex quadratic objectives

    # Variables
    zz = m.addMVar(n_orig, lb=-1, ub=1, name="uu")

    m.setObjective(.5 * zz @ P @ zz + x @ zz, GRB.MINIMIZE)

    # Solve
    m.optimize()

    z_star = zz.x
    return z_star
    

def run_trust_region_method(K, P, H, z0, x, rho):
    z_sol = z0
    z_sols = []
    suboptimalities = np.zeros(K)

    # get the optimal solution
    z_star = solve_exact(P, x)

    suboptimalities[0] = 0.5 * z0 @ P @ z0 - 0.5 * z_star @ P @ z_star + x @ (z0 - z_star)
    
    for k in range(1, K):
        z_sol = trust_region_iter(P, H, z0, x, rho)
        z_sols.append(z_sol)
        suboptimalities[k] = 0.5 * z_sol @ P @ z_sol - 0.5 * z_star @ P @ z_star + x @ (z_sol - z_star)
        z0 = z_sol

    suboptimalities[suboptimalities <= 1e-9] = 0
    return z_star, z_sols, suboptimalities


def trust_region_iter(P, H, z0, x, rho):
    n_orig = z0.size
    z = cp.Variable(n_orig)

    constraints = [z >= -1, z <= 1, z <= z0 + rho, z >= z0 - rho]

    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(z, H) +  z0 @ (P - H) @ z + x @ z), constraints)
    prob.solve(verbose=True)
    z_next = z.value

    return z_next


def get_dynamics(dt=0.05, M=1.0, m=0.1, L=0.5, g=9.81):
    """
    Get discrete-time linearized cart-pole dynamics around upright equilibrium.
    
    Linearization point: x_eq = [p, p_dot, theta, theta_dot] = [0, 0, 0, 0]
                        u_eq = 0
    
    State: x = [p, p_dot, theta, theta_dot]^T
        p: cart position [m]
        p_dot: cart velocity [m/s]
        theta: pendulum angle from vertical [rad] (positive = clockwise)
        theta_dot: pendulum angular velocity [rad/s]
    
    Control: u = horizontal force on cart [N]
    
    Dynamics: x_{k+1} = A_dyn @ x_k + B_dyn @ u_k
    
    Parameters:
        dt: sampling time [s] (default: 0.05)
        M: cart mass [kg] (default: 1.0)
        m: pendulum mass [kg] (default: 0.1)
        L: pendulum length to center of mass [m] (default: 0.5)
        g: gravitational acceleration [m/s^2] (default: 9.81)
    
    Returns:
        A_dyn: (4, 4) discrete-time state matrix
        B_dyn: (4, 1) discrete-time input matrix
    """
    
    # Continuous-time linearized dynamics around upright equilibrium
    # Using small-angle approximation: sin(theta) ≈ theta, cos(theta) ≈ 1
    
    A_cont = np.array([
        [0,  1,              0,              0],
        [0,  0,  -m*g/M,                     0],
        [0,  0,              0,              1],
        [0,  0,  (M+m)*g/(M*L),              0]
    ])
    
    B_cont = np.array([
        [0],
        [1/M],
        [0],
        [-1/(M*L)]
    ])
    
    # Discretize using matrix exponential (exact discretization)
    # Method: x_{k+1} = e^{A*dt} x_k + (∫_0^dt e^{A*τ} dτ) B u_k
    
    n_states = A_cont.shape[0]
    n_inputs = B_cont.shape[1]
    
    # Augmented matrix for simultaneous discretization
    # [A  B]
    # [0  0]
    F = np.zeros((n_states + n_inputs, n_states + n_inputs))
    F[:n_states, :n_states] = A_cont
    F[:n_states, n_states:] = B_cont
    
    # Matrix exponential
    F_exp = expm(F * dt)
    
    # Extract discrete-time matrices
    A_dyn = F_exp[:n_states, :n_states]
    B_dyn = F_exp[:n_states, n_states:]
    
    return A_dyn, B_dyn


def get_dynamics_at_angle(theta_0, dt=0.05, mass=1.0, length=1.0, g=9.81):
    """Get A_lin when linearizing at theta = theta_0 (not at upright)."""
    A_cont = np.array([
        [0, 1],
        [(g/length) * np.cos(theta_0), 0]  # derivative of sin is cos
    ])
    
    B_cont = np.array([[0], [1 / (mass * length**2)]])
    
    A_dyn = np.eye(2) + dt * A_cont
    B_dyn = dt * B_cont
    
    return A_dyn, B_dyn


# def get_dynamics_euler(dt=0.05, M=1.0, m=0.1, L=0.5, g=9.81):
#     """
#     Get discrete-time linearized cart-pole dynamics using Euler approximation.
    
#     This is a simpler but less accurate alternative to get_dynamics().
#     Use for comparison or when exact discretization is not needed.
    
#     Parameters: same as get_dynamics()
    
#     Returns:
#         A_dyn: (4, 4) discrete-time state matrix
#         B_dyn: (4, 1) discrete-time input matrix
#     """
    
#     # Continuous-time matrices
#     A_cont = np.array([
#         [0,  1,              0,              0],
#         [0,  0,  -m*g/M,                     0],
#         [0,  0,              0,              1],
#         [0,  0,  (M+m)*g/(M*L),              0]
#     ])
    
#     B_cont = np.array([
#         [0],
#         [1/M],
#         [0],
#         [-1/(M*L)]
#     ])
    
#     # Euler discretization: A_d = I + dt*A_c, B_d = dt*B_c
#     A_dyn = np.eye(4) + dt * A_cont
#     B_dyn = dt * B_cont
    
#     return A_dyn, B_dyn


def form_mpc_qp(A_dyn, B_dyn, Q, r, T, u_max):
    """
    Form MPC problem in standard conic QP form for solvers like CVXPY, Clarabel, SCS.
    
    Original problem:
        min  Σ_{t=0}^{T-1} (x_t^T Q x_t + u_t^T r u_t)
        s.t. x_{t+1} = A_dyn x_t + B_dyn u_t,  t=0,...,T-1
             -u_max <= u_t <= u_max,  t=0,...,T-1
             x_0 = x_init
    
    Standard conic form:
        min  (1/2) z^T P z + c^T z
        s.t. A_cone z + s = b
             s ∈ K
    
    where z = [x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T]
    and K = K_zero × K_+ (zero cone for equality, nonnegative orthant for inequality)
    
    Args:
        A_dyn: (n_x, n_x) discrete-time state matrix
        B_dyn: (n_x, n_u) discrete-time input matrix
        Q: (n_x, n_x) state cost matrix
        r: scalar or (n_u, n_u) input cost (if scalar, assumes R = r*I)
        T: time horizon (number of time steps)
        x_init: (n_x,) initial state
        u_max: scalar, control box constraint: -u_max <= u <= u_max
    
    Returns:
        P: (n_z, n_z) sparse quadratic cost matrix (symmetric)
        c: (n_z,) linear cost vector
        A_cone: (n_constraints, n_z) sparse constraint matrix
        b: (n_constraints,) constraint RHS
        cone_dims: dict with keys 'zero' (equality) and 'nonneg' (inequality)
        z_slices: dict with 'x' and 'u' indices for extracting solution
    """
    
    n_x = A_dyn.shape[0]  # state dimension
    n_u = B_dyn.shape[1]  # control dimension
    x_init = np.ones(n_x)  # initial state (assumed zero for this function)
    
    # Handle scalar r
    if np.isscalar(r):
        R = r * np.eye(n_u)
    else:
        R = r
    
    # Total number of decision variables
    # z = [x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T]
    n_z = (T + 1) * n_x + T * n_u
    
    # ========================================
    # BUILD QUADRATIC COST: (1/2) z^T P z + c^T z
    # ========================================
    
    # Cost is: Σ_{t=0}^{T-1} (x_t^T Q x_t + u_t^T R u_t)
    # Note: x_T has no cost (could add terminal cost if desired)
    
    P_blocks = []
    row_indices = []
    col_indices = []
    data_values = []
    
    for t in range(T):
        # State cost at time t: x_t^T Q x_t
        x_idx = t * (n_x + n_u)  # starting index of x_t in z
        
        # Add Q block (symmetric, so we store full matrix)
        for i in range(n_x):
            for j in range(n_x):
                if Q[i, j] != 0:
                    row_indices.append(x_idx + i)
                    col_indices.append(x_idx + j)
                    data_values.append(Q[i, j])
        
        # Control cost at time t: u_t^T R u_t
        u_idx = t * (n_x + n_u) + n_x  # starting index of u_t in z
        
        for i in range(n_u):
            for j in range(n_u):
                if R[i, j] != 0:
                    row_indices.append(u_idx + i)
                    col_indices.append(u_idx + j)
                    data_values.append(R[i, j])
    
    # Create sparse P matrix
    P = sparse.coo_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(n_z, n_z)
    ).tocsr()
    
    # Make P symmetric (in case of numerical issues)
    P = (P + P.T) / 2
    
    # Linear cost c (zero for this problem)
    c = np.zeros(n_z)
    
    # ========================================
    # BUILD CONSTRAINTS: A_cone z + s = b, s ∈ K
    # ========================================
    
    # Three types of constraints:
    # 1. Initial condition: x_0 = x_init  (n_x equality constraints)
    # 2. Dynamics: x_{t+1} = A_dyn x_t + B_dyn u_t  (T * n_x equality constraints)
    # 3. Control bounds: -u_max <= u_t <= u_max  (2*T*n_u inequality constraints)
    
    n_eq = n_x + T * n_x  # initial condition + dynamics
    n_ineq = 2 * T * n_u  # control box constraints (upper and lower)
    n_constraints = n_eq + n_ineq
    
    constraint_rows = []
    constraint_cols = []
    constraint_data = []
    b_constraint = np.zeros(n_constraints)
    
    constraint_idx = 0
    
    # ----------------------------------------
    # 1. Initial condition: x_0 = x_init
    # ----------------------------------------
    for i in range(n_x):
        constraint_rows.append(constraint_idx)
        constraint_cols.append(i)  # x_0 is at indices 0:n_x
        constraint_data.append(1.0)
        b_constraint[constraint_idx] = x_init[i]
        constraint_idx += 1
    
    # ----------------------------------------
    # 2. Dynamics: x_{t+1} - A_dyn x_t - B_dyn u_t = 0
    # ----------------------------------------
    for t in range(T):
        x_t_idx = t * (n_x + n_u)
        u_t_idx = t * (n_x + n_u) + n_x
        x_tp1_idx = (t + 1) * (n_x + n_u)
        
        for i in range(n_x):
            # x_{t+1}[i]
            constraint_rows.append(constraint_idx)
            constraint_cols.append(x_tp1_idx + i)
            constraint_data.append(1.0)
            
            # -A_dyn x_t
            for j in range(n_x):
                if A_dyn[i, j] != 0:
                    constraint_rows.append(constraint_idx)
                    constraint_cols.append(x_t_idx + j)
                    constraint_data.append(-A_dyn[i, j])
            
            # -B_dyn u_t
            for j in range(n_u):
                if B_dyn[i, j] != 0:
                    constraint_rows.append(constraint_idx)
                    constraint_cols.append(u_t_idx + j)
                    constraint_data.append(-B_dyn[i, j])
            
            b_constraint[constraint_idx] = 0.0
            constraint_idx += 1
    
    # ----------------------------------------
    # 3. Control bounds: -u_max <= u_t <= u_max
    # Reformulated as: u_t <= u_max  AND  -u_t <= u_max
    # In conic form: u_t - u_max <= 0  AND  -u_t - u_max <= 0
    # With slack: u_t + s_upper = u_max, s_upper >= 0
    #            -u_t + s_lower = u_max, s_lower >= 0
    # ----------------------------------------
    for t in range(T):
        u_t_idx = t * (n_x + n_u) + n_x
        
        # Upper bound: u_t <= u_max  -->  u_t + s = u_max, s >= 0
        for j in range(n_u):
            constraint_rows.append(constraint_idx)
            constraint_cols.append(u_t_idx + j)
            constraint_data.append(1.0)
            b_constraint[constraint_idx] = u_max
            constraint_idx += 1
        
        # Lower bound: -u_max <= u_t  -->  -u_t + s = u_max, s >= 0
        for j in range(n_u):
            constraint_rows.append(constraint_idx)
            constraint_cols.append(u_t_idx + j)
            constraint_data.append(-1.0)
            b_constraint[constraint_idx] = u_max
            constraint_idx += 1
    
    # Create sparse constraint matrix
    A_cone = sparse.coo_matrix(
        (constraint_data, (constraint_rows, constraint_cols)),
        shape=(n_constraints, n_z)
    ).tocsr()
    
    b = b_constraint
    
    # ========================================
    # CONE DIMENSIONS
    # ========================================
    
    # K = K_zero × K_+
    # K_zero: equality constraints (initial condition + dynamics)
    # K_+: nonnegative orthant (control bounds)
    
    cone_dims = {
        'zero': n_eq,      # first n_eq constraints are equalities
        'nonneg': n_ineq   # remaining n_ineq constraints are inequalities
    }
    
    # ========================================
    # SOLUTION EXTRACTION INDICES
    # ========================================
    
    z_slices = {
        'x': [],  # list of slices for each x_t
        'u': []   # list of slices for each u_t
    }
    
    for t in range(T):
        x_t_start = t * (n_x + n_u)
        u_t_start = t * (n_x + n_u) + n_x
        
        z_slices['x'].append(slice(x_t_start, x_t_start + n_x))
        z_slices['u'].append(slice(u_t_start, u_t_start + n_u))
    
    # Final state x_T
    x_T_start = T * (n_x + n_u)
    z_slices['x'].append(slice(x_T_start, x_T_start + n_x))
    
    return P.todense(), c, A_cone.todense(), b, cone_dims, z_slices