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
    """
    For each (K, T) pair, search for a K-periodic orbit of the MPC policy.
    A feasible solution certifies that stability is NOT obtained.
    """
    K_vals_list = list(cfg.K_vals)
    T_vals_list = list(cfg.T_vals)
    r = cfg.r
    dt = cfg.dt
    x_min = cfg.x_mins[0]
    x_max = cfg.x_maxes[0]
    u_bound = cfg.u_bound

    n_K = len(K_vals_list)
    n_T = len(T_vals_list)

    # results[ti, ki] = (found: bool, obj_val: float or None, solve_time: float)
    results = [[None] * n_K for _ in range(n_T)]

    for ti, T in enumerate(T_vals_list):
        for ki, K in enumerate(K_vals_list):
            print(f"\n=== Searching for K={K}-cycle with T={T} ===")
            cyc = CartpoleILQRCycle(
                K=K, T=T, r=r, dt=dt,
                mass=1, length=1, g=9.8,
                x_lo=x_min, x_hi=x_max,
                u_bound=u_bound,
                verbose=True,
                time_limit=cfg.time_limit,
                nontrivial_eps=cfg.nontrivial_eps,
            )
            status, solve_time = cyc.solve()
            sol = cyc.solution_dict()

            found = sol['obj'] is not None and sol['obj'] > cfg.get('nontrivial_eps', 1e-4)
            results[ti][ki] = (found, sol['obj'], solve_time)

            if found:
                print(f"  CYCLE FOUND: ||x_0||^2 = {sol['obj']:.6f}")
                x0 = sol['x'][0]
                print(f"  x_0 = {x0}")
            else:
                print(f"  No cycle found (status={status})")
            import pdb
            pdb.set_trace()

    # Print summary table
    print("\n=== Summary: cycle found? (||x_0||^2 if yes) ===")
    header = "T\\K  " + "  ".join(f"K={K:2d}" for K in K_vals_list)
    print(header)
    for ti, T in enumerate(T_vals_list):
        row = f"T={T:2d}  "
        for ki in range(n_K):
            found, obj, _ = results[ti][ki]
            row += f"{'YES':>6}" if found else f"{'no':>6}"
            row += "  "
        print(row)

    # Save a grid figure: green = cycle found, red = no cycle
    fig, ax = plt.subplots(figsize=(max(4, n_K * 1.5), max(3, n_T * 1.2)))
    grid = np.array([[1.0 if results[ti][ki][0] else 0.0
                      for ki in range(n_K)] for ti in range(n_T)])
    ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n_K))
    ax.set_xticklabels([str(K) for K in K_vals_list])
    ax.set_yticks(range(n_T))
    ax.set_yticklabels([str(T) for T in T_vals_list])
    ax.set_xlabel('cycle length $K$')
    ax.set_ylabel('horizon $T$')
    for ti in range(n_T):
        for ki in range(n_K):
            found, obj, _ = results[ti][ki]
            label = f"{obj:.2f}" if found and obj is not None else "—"
            ax.text(ki, ti, label, ha='center', va='center', fontsize=14)
    fig.tight_layout()
    fig.savefig('cycles_ilqr.pdf', bbox_inches='tight')
    plt.close(fig)


class CartpoleILQRCycle:
    """
    Searches for a K-periodic orbit of the MPC closed-loop system.

    Find x_0 in X_0 such that:
        x_{t+1} = f_MPC(x_t)   for t = 0, ..., K-1   (MPC dynamics + KKT)
        x_K = x_0                                      (cycle closure)
        ||x_0 - x_1||^2 >= nontrivial_eps             (non-trivial cycle)

    Objective: maximize ||x_0||^2.
    Feasibility + positive objective => K-cycle found => stability not certified.

    When u_bound is given, KKT includes complementarity for box-constrained MPC.
    When u_bound is None, unconstrained MPC (simple stationarity).
    """

    def __init__(self, K=3, T=5, r=0.1, dt=0.1, mass=1, length=1, g=9.8,
                 x_lo=0.0, x_hi=4.0, u_bound=None, seed=None, verbose=True,
                 time_limit=None, nontrivial_eps=1e-4):
        self.K = K
        self.T = T
        self.verbose = bool(verbose)

        n_x = 2
        n_u = 1
        Q = np.eye(n_x)
        R = np.eye(n_u) * r
        constrained = u_bound is not None
        _u_max = u_bound if constrained else 1000.0

        M = gp.Model("cartpole_ilqr_cycle")
        M.Params.OutputFlag = 1 if self.verbose else 0
        M.Params.FeasibilityTol = 1e-9
        M.Params.NonConvex = 2   # bilinear costate terms + complementarity + cycle quadratics
        if time_limit is not None:
            M.Params.TimeLimit = time_limit
        self.model = M

        # State trajectory x[0], ..., x[K]  (x[K] will be constrained == x[0])
        # x[0] is the candidate cycle start, bounded within X_0
        self.x = {}
        self.x[0] = M.addVars(n_x, lb=x_lo, ub=x_hi, name="x_0")
        for k in range(1, K + 1):
            self.x[k] = M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_{k}")

        self.u = {}
        self.x_mpc = {}
        self.u_mpc_var = {}
        self.lambda_mpc = {}
        self.sin_theta = {}
        self.cos_theta = {}
        self.theta_ddot = {}
        self.sin_theta_mpc = {}
        self.cos_theta_mpc = {}
        self.theta_ddot_mpc = {}

        if constrained:
            self.mu_up_mpc = {}
            self.mu_lo_mpc = {}

        # Build K steps of MPC-controlled dynamics
        for k in range(K):
            self.u[k] = M.addVars(n_u, lb=-_u_max, ub=_u_max, name=f"u_{k}")

            # T-horizon MPC at x[k]
            self.x_mpc[k] = {t: M.addVars(n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                           name=f"x_mpc_{k}_{t}")
                              for t in range(T + 1)}
            self.u_mpc_var[k] = {t: M.addVars(n_u, lb=-_u_max, ub=_u_max,
                                               name=f"u_mpc_{k}_{t}")
                                  for t in range(T)}

            # MPC initial condition
            for i in range(n_x):
                M.addConstr(self.x_mpc[k][0][i] == self.x[k][i], name=f"mpc_init_{k}_{i}")

            # MPC nonlinear dynamics
            self.sin_theta_mpc[k] = {}
            self.cos_theta_mpc[k] = {}
            self.theta_ddot_mpc[k] = {}
            for t in range(T):
                self.sin_theta_mpc[k][t] = M.addVar(lb=-1, ub=1, name=f"sin_mpc_{k}_{t}")
                self.cos_theta_mpc[k][t] = M.addVar(lb=-1, ub=1, name=f"cos_mpc_{k}_{t}")
                self.theta_ddot_mpc[k][t] = M.addVar(lb=-GRB.INFINITY,
                                                      name=f"tdd_mpc_{k}_{t}")
                M.addGenConstrSin(self.x_mpc[k][t][0], self.sin_theta_mpc[k][t],
                                  name=f"sin_mpc_c_{k}_{t}")
                M.addGenConstrCos(self.x_mpc[k][t][0], self.cos_theta_mpc[k][t],
                                  name=f"cos_mpc_c_{k}_{t}")
                M.addConstr(
                    self.x_mpc[k][t+1][0] == self.x_mpc[k][t][0] + dt * self.x_mpc[k][t][1],
                    name=f"mpc_theta_{k}_{t}")
                M.addConstr(
                    self.theta_ddot_mpc[k][t] == g/length * self.sin_theta_mpc[k][t]
                    + self.u_mpc_var[k][t][0] / (mass * length**2),
                    name=f"mpc_tdd_{k}_{t}")
                M.addConstr(
                    self.x_mpc[k][t+1][1] == self.x_mpc[k][t][1]
                    + dt * self.theta_ddot_mpc[k][t],
                    name=f"mpc_tdot_{k}_{t}")

            # KKT costate variables
            self.lambda_mpc[k] = {t: M.addVars(n_x, lb=-GRB.INFINITY,
                                                name=f"lam_{k}_{t}")
                                   for t in range(T + 1)}

            # Terminal costate
            for i in range(n_x):
                M.addConstr(
                    self.lambda_mpc[k][T][i] == Q[i, i] * self.x_mpc[k][T][i],
                    name=f"term_costate_{k}_{i}")

            # Backward sweep (bilinear: cos * lambda)
            for t in range(T - 1, -1, -1):
                M.addConstr(
                    self.lambda_mpc[k][t][0] == Q[0, 0] * self.x_mpc[k][t][0]
                    + self.lambda_mpc[k][t+1][0]
                    + dt * g / length * self.cos_theta_mpc[k][t] * self.lambda_mpc[k][t+1][1],
                    name=f"costate_0_{k}_{t}")
                M.addConstr(
                    self.lambda_mpc[k][t][1] == Q[1, 1] * self.x_mpc[k][t][1]
                    + dt * self.lambda_mpc[k][t+1][0] + self.lambda_mpc[k][t+1][1],
                    name=f"costate_1_{k}_{t}")

            if constrained:
                # Box-constrained KKT: stationarity + complementarity
                self.mu_up_mpc[k] = {t: M.addVars(n_u, lb=0.0, name=f"mu_up_{k}_{t}")
                                      for t in range(T)}
                self.mu_lo_mpc[k] = {t: M.addVars(n_u, lb=0.0, name=f"mu_lo_{k}_{t}")
                                      for t in range(T)}
                for t in range(T):
                    M.addConstr(
                        R[0, 0] * self.u_mpc_var[k][t][0]
                        + dt / (mass * length**2) * self.lambda_mpc[k][t+1][1]
                        + self.mu_up_mpc[k][t][0] - self.mu_lo_mpc[k][t][0] == 0,
                        name=f"ctrl_stat_{k}_{t}")
                    M.addConstr(
                        self.mu_up_mpc[k][t][0] * (self.u_mpc_var[k][t][0] - u_bound) == 0,
                        name=f"comp_up_{k}_{t}")
                    M.addConstr(
                        self.mu_lo_mpc[k][t][0] * (-u_bound - self.u_mpc_var[k][t][0]) == 0,
                        name=f"comp_lo_{k}_{t}")
            else:
                # Unconstrained KKT: simple stationarity
                for t in range(T):
                    M.addConstr(
                        R[0, 0] * self.u_mpc_var[k][t][0]
                        + dt / (mass * length**2) * self.lambda_mpc[k][t+1][1] == 0,
                        name=f"ctrl_stat_{k}_{t}")

            # Link first MPC action to applied control
            M.addConstr(self.u[k][0] == self.u_mpc_var[k][0][0], name=f"ctrl_link_{k}")

            # Actual nonlinear dynamics: x[k] -> x[k+1]
            self.sin_theta[k] = M.addVar(lb=-1, ub=1, name=f"sin_{k}")
            self.cos_theta[k] = M.addVar(lb=-1, ub=1, name=f"cos_{k}")
            self.theta_ddot[k] = M.addVar(lb=-GRB.INFINITY, name=f"tdd_{k}")
            M.addGenConstrSin(self.x[k][0], self.sin_theta[k], name=f"sin_c_{k}")
            M.addGenConstrCos(self.x[k][0], self.cos_theta[k], name=f"cos_c_{k}")
            M.addConstr(
                self.x[k+1][0] == self.x[k][0] + dt * self.x[k][1],
                name=f"int_theta_{k}")
            M.addConstr(
                self.theta_ddot[k] == g/length * self.sin_theta[k]
                + self.u[k][0] / (mass * length**2),
                name=f"tdd_{k}")
            M.addConstr(
                self.x[k+1][1] == self.x[k][1] + dt * self.theta_ddot[k],
                name=f"int_tdot_{k}")

        # Cycle closure: x[K] = x[0]
        for i in range(n_x):
            M.addConstr(self.x[K][i] == self.x[0][i], name=f"cycle_{i}")

        # Non-trivial: ||x_0 - x_1||^2 >= nontrivial_eps
        # diff_sq = gp.quicksum(
        #     (self.x[0][i] - self.x[1][i]) * (self.x[0][i] - self.x[1][i])
        #     for i in range(n_x)
        # )
        diff_sq = gp.quicksum(
            (self.x[0][i]) * (self.x[0][i])
            for i in range(n_x)
        )
        M.addConstr(diff_sq >= nontrivial_eps, name="nontrivial")

        # Objective: maximize ||x_0||^2
        # obj = gp.quicksum(
        #     (self.x[0][i] - self.x[K][i]) * (self.x[0][i] - self.x[K][i])
        #     for i in range(n_x)
        # )
        obj = 0
        # obj = gp.quicksum(self.x[0][i] * self.x[0][i] for i in range(n_x))
        self.obj_expr = obj
        M.setObjective(obj, GRB.MINIMIZE)

    def solve(self):
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        def v2dict(vs): return {k: vs[k].X for k in vs}

        if self.model.SolCount == 0:
            return {"obj": None}

        out = {
            "obj": self.model.ObjVal,
            "x": {k: v2dict(self.x[k]) for k in range(self.K + 1)},
            "u": {k: v2dict(self.u[k]) for k in range(self.K)},
            "x_mpc": {k: {t: v2dict(self.x_mpc[k][t])
                          for t in range(self.T + 1)}
                      for k in range(self.K)},
        }
        return out
