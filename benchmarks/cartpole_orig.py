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

    for j in range(len(x_mins)):
        x_min = x_mins[j]
        x_max = x_maxes[j]
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
            for j in range(K):
                ver = CartpoleVerify(n=cfg.n, K=j, rho=cfg.rho, x_lo=x_min, x_hi=x_max, warm_start_bd=warm_start_bd, seed=42, verbose=True)
                
                status, time = ver.solve()
                print("solve() status:", status)
                sol = ver.solution_dict()
                print("Objective:", sol["obj"])
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
    def __init__(self, n=10, K=3, rho=0.2, x_lo=0.0, x_hi=4.0, warm_start_bd=None, seed=None, verbose=True):
        self.n, self.K, self.rho = n, K, rho
        self.verbose = bool(verbose)
        rng = np.random.default_rng(seed)

        # generate problem data
        g = 9.8
        length = 1
        dt = 0.01
        mass = 1
        n_x = 2
        n_u = 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * .01
        T = 10
        u_max = 1

        # A_dyn, B_dyn = get_dynamics_euler(dt=dt, M=1.0, m=m, L=L, g=g)
        theta_0 = 0
        self.A_dyn, self.B_dyn = get_dynamics_at_angle(theta_0, dt=dt, mass=mass, length=length, g=g)
        P, c, A, b_const, cone_dims, z_slices = form_mpc_qp(self.A_dyn, self.B_dyn, Q, R, T, u_max)
        n_eq = cone_dims['zero']


        # P, A, c, b_const = generate_cartpole_mpc_qp_matrices(n, K, dt, m, L, g, Q, R)
        # x0 constraint should be the first n_x entries of b
        # x1 variable should be the entries (n_x, 2 n_x)

        # Constraint matrix A (box + trust region): shape (4n, n)
        n = A.shape[1] #self.n
        m = A.shape[0]

        # Model
        M = gp.Model("cartpole_verify")
        M.Params.OutputFlag = 1 if self.verbose else 0
        M.Params.FeasibilityTol = 1e-9
        self.model = M

        # initial state x0
        self.x0 = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x")

        # lower-level controller u_lower
        self.u_lower = M.addVars(n_u, lb=-GRB.INFINITY, name="u_lower")
        # self.u_lower = M.addVars(n_u, lb=-1, ub=1, name="u_lower")

        # next state x1
        self.x1 = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

        # reference state x_ref1
        self.x_ref1 = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x_ref1")


        # MPC variables
        self.u      = M.addVars(n, lb=-GRB.INFINITY, name=f"u")
        self.s      = M.addVars(m, lb=-GRB.INFINITY, name=f"s")
        self.mu     = M.addVars(m, lb=-GRB.INFINITY, name=f"mu")
        self.b      = M.addVars(m, lb=-GRB.INFINITY, name=f"b")

        # separate cones: equality vs inequality
        for i in range(m):
            if i < n_eq:
                M.addConstr(self.s[i] == 0, name=f"s_{i}")
                # mu is free
            else:
                M.addConstr(self.s[i] >= 0, name=f"s_{i}")
                M.addConstr(self.mu[i] >= 0, name=f"mu_{i}")

        # linearized dynamics variables
        self.A_lin = M.addVars(n_x, n_x,   lb=-GRB.INFINITY, name=f"A")
        self.B_lin = np.zeros((2, 1))
        self.B_lin[0, 0] = 0
        self.B_lin[1, 0] = dt / (mass * length ** 2)

        import pdb
        pdb.set_trace()


        # c^{(k)} = x + (P - H) u0^{(k)}
        for i in range(n_x):
            M.addConstr(self.b[i] == self.x0[i], name=f"x0_{i}")
        for i in range(n_x, m):
            M.addConstr(self.b[i] == b_const[i], name=f"x0_{i}")

        # Primal feasibility: A u + s = b
        for r in range(m):
            lhs = gp.quicksum(A[r, j] * self.u[j] for j in range(n))
            M.addConstr(lhs + self.s[r] == self.b[r], name=f"pr_{r}")

        # Dual feasibility: A^T mu + H u = -c
        # row i: sum_r A[r,i]*mu[r] + sum_j H[i,j]*u[j] = -c[i]
        for i in range(n):
            Atu = gp.quicksum(A[r, i] * self.mu[r] for r in range(m))
            Pu  = gp.quicksum(P[i, j] * self.u[j]  for j in range(n))
            M.addConstr(Atu + Pu == -c[i], name=f"du_{i}")

        # Complementarity: s_r * mu_r = 0
        for r in range(m):
            M.addConstr(self.s[r] * self.mu[r] == 0, name=f"comp_{r}")

        # connect opt sol of mpc to x_ref1
        # for i in range(n_x + n_u, n_x + n_u + n_x):
        M.addConstr(self.x_ref1[0] == self.u[n_x + n_u], name=f"link_xref1_{i}")
        M.addConstr(self.x_ref1[1] == self.u[n_x + n_u + 1], name=f"link_xref1_{i}")


        # Create auxiliary variable for sin(x0[0])
        self.sin_theta = M.addVar(lb=-1, ub=1, name="sin_theta")

        M.addConstr(
            self.x1[0] == self.x0[0] + dt * self.x0[1],
            name="integrate_theta"
        )
        
        # For theta_ddot, we need sin(theta)
        M.addGenConstrSin(self.x0[0], self.sin_theta, name="sin_constraint")
        
        # Create auxiliary variable for theta_ddot
        self.theta_ddot = M.addVar(lb=-GRB.INFINITY, name="theta_ddot")
        
        # Constraint 2: theta_ddot = (g/L) * sin(theta) + u_lower / (m*L²)
        M.addConstr(
            self.theta_ddot == g/length * self.sin_theta + self.u_lower[0] / (mass * length**2),
            name="angular_acceleration"
        )
        
        # Constraint 3: theta_dot_{k+1} = theta_dot_k + dt * theta_ddot
        M.addConstr(
            self.x1[1] == self.x0[1] + dt * self.theta_ddot,
            name="integrate_theta_dot"
        )


        # constraints for the linearized dynamics
        self.cos_theta = M.addVar(lb=-1, ub=1, name="cos_theta")
        M.addGenConstrCos(self.x0[0], self.cos_theta, name="cos_constraint")
        M.addConstr(self.A_lin[0, 0] == 1, name=f"A_dyn_{0}_{0}")
        M.addConstr(self.A_lin[0, 1] == dt, name=f"A_dyn_{0}_{1}")
        M.addConstr(self.A_lin[1, 0] == dt * g / length * self.cos_theta, name=f"A_dyn_{1}_{0}")
        M.addConstr(self.A_lin[1, 1] == 1, name=f"A_dyn_{1}_{1}")

        # constraints for the prediction error
        self.pred_error = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="pred_error")
        for i in range(n_x):
            Ax = gp.quicksum(self.A_lin[i, r] * self.x0[r] for r in range(n_x))
            Bu = gp.quicksum(self.B_lin[i, r] * self.u_lower[r] for r in range(n_u))
            # M.addConstr(self.pred_error[i] == Ax + Bu - self.x_ref1[i], name=f"pred_error_{i}")
            M.addConstr(self.pred_error[i] == self.x1[i] + Bu - self.x_ref1[i], name=f"pred_error_{i}")

        # constraints for the LQR controller
        B_lin_Q = self.B_lin.T @ Q
        for i in range(n_u):
            BQe = gp.quicksum(B_lin_Q[i, r] * self.pred_error[r] for r in range(n_x))
            M.addConstr(BQe + R[0, 0] * self.u_lower[i] == 0, name=f"lqr_opt_{i}")
            # M.addConstr(BQe == 0, name=f"lqr_opt_{i}")

        # actual dynamics
        # x_1 = f_dyn(x_0, u_lower)
        # def encode_dyn(x1, x0, u):
        # x1[0] == x0[0] + dt * x0[1]
        # theta_ddot = g / L * np.sin(theta) + u_lower / (m * L**2)
        # x1[1] == x0[1] + dt * theta_ddot 
        # Constraint 1: theta_{k+1} = theta_k + dt * theta_dot_k



        # Worst-case objective
        V_curr = gp.quicksum(self.x0[i] * self.x0[i] for i in range(n_x)) #+ R[0,0] * self.u_lower[0] * self.u_lower[0]
        V_next = gp.quicksum(self.x1[i] * self.x1[i] for i in range(n_x)) #+ R[0,0] * self.u_lower[0] * self.u_lower[0]
        self.orig_objective = V_next - V_curr
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

    def solution_dict(self):
        def v2dict(vs): return {k: vs[k].X for k in vs}
        out = {
            "obj": None if self.model.SolCount == 0 else self.model.ObjVal,
            "x0":   None if self.model.SolCount == 0 else v2dict(self.x0),
            "x1":   None if self.model.SolCount == 0 else v2dict(self.x1),
            # "cos_theta":  None if self.model.SolCount == 0 else v2dict(self.cos_theta),
            "cos_theta":  None if self.model.SolCount == 0 else self.cos_theta.X,
            "sin_theta":  None if self.model.SolCount == 0 else self.sin_theta.X,
            "x_ref1":   None if self.model.SolCount == 0 else v2dict(self.x_ref1),
            "u":  None if self.model.SolCount == 0 else v2dict(self.u),
            "s":  None if self.model.SolCount == 0 else v2dict(self.s),
            "mu":  None if self.model.SolCount == 0 else v2dict(self.mu),
            "b":  None if self.model.SolCount == 0 else v2dict(self.b),
            "u_lower":  None if self.model.SolCount == 0 else v2dict(self.u_lower),
        }
        return out

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