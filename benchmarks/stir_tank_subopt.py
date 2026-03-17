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

# ---- Stir tank reactor parameters (from paper) ----
THETA   = 20.0
K_REACT = 300.0
M_PARAM = 5.0
X_F     = 0.3947
X_C     = 0.3816
GAMMA   = 0.117

XE = np.array([0.2632, 0.6519])  # equilibrium state
UE = 0.7853                       # equilibrium control

U_LO = 0.0
U_HI = 2.0


def _stir_tank_discrete(dt):
    """
    Linearize the continuous-time stir tank dynamics around (XE, UE):
        xdot1 = (1 - x1) / theta - k * x1 * exp(-M / x2)
        xdot2 = (xf - x2) / theta + k * x1 * exp(-M / x2) - gamma * u * (x2 - xc)

    and discretize using Euler's method with step dt.

    In error coordinates  e = x - XE,  v = u - UE:
        e_{t+1} = A_d * e_t + B_d * v_t

    Returns (A_d, B_d) as numpy arrays of shape (2,2) and (2,1).
    """
    E_eq = np.exp(-M_PARAM / XE[1])
    # Jacobian w.r.t. state at equilibrium
    A_c = np.array([
        [-1/THETA - K_REACT * E_eq,
          K_REACT * XE[0] * E_eq * M_PARAM / XE[1]**2],
        [ K_REACT * E_eq,
         -1/THETA + K_REACT * XE[0] * E_eq * M_PARAM / XE[1]**2 - GAMMA * UE]
    ])
    # Jacobian w.r.t. control at equilibrium
    B_c = np.array([[0.0],
                    [-GAMMA * (XE[1] - X_C)]])

    A_d = np.eye(2) + dt * A_c
    B_d = dt * B_c
    return A_d, B_d


def run(cfg):
    # T_max = cfg.T_max
    e_lim = getattr(cfg, 'e_lim', 0.2)
    r     = getattr(cfg, 'r', 0.1)
    dt    = getattr(cfg, 'dt', 1.0)

    # T_vals  = list(range(5, T_max + 1))
    T_vals = cfg.T_vals
    obj_vals = []

    for T in T_vals:
        ver = StirTankVerify(T=T, r=r, e_lim=e_lim, dt=dt, verbose=True)
        status, elapsed = ver.solve()
        sol = ver.solution_dict()
        obj = sol['obj'] if sol['obj'] is not None else float('nan')
        print(f"T={T}, status={status}, obj={obj:.6f}, time={elapsed:.3f}s")
        obj_vals.append(obj)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_vals, obj_vals, marker='o', linewidth=2)
    ax.set_xlabel('Horizon $T$')
    ax.set_ylabel(r'$J_{\mathrm{kkt}} - J_{\mathrm{opt}}$')
    ax.set_title(r'Stir tank: worst-case KKT suboptimality vs.\ horizon')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('stir_tank_suboptimality.pdf', bbox_inches='tight')
    print("Plot saved to stir_tank_suboptimality.pdf")
    return obj_vals


class StirTankVerify:
    """
    Finds the worst-case performance gap between any KKT point and a feasible
    (optimal) solution for a T-horizon MPC problem on the nonlinear stir tank
    reactor.

    Continuous-time dynamics:
        xdot1 = (1 - x1) / theta - k * x1 * exp(-M / x2)
        xdot2 = (xf - x2) / theta + k * x1 * exp(-M / x2) - gamma * u * (x2 - xc)

    Parameters: theta=20, k=300, M=5, xf=0.3947, xc=0.3816, gamma=0.117
    Equilibrium: xe=[0.2632, 0.6519], ue=0.7853

    The MPC is formulated in error coordinates e = x - xe, v = u - ue using a
    linearization of the dynamics around (xe, ue) discretized with Euler step dt:
        e_{t+1} = A_d * e_t + B_d * v_t

    MPC cost (without the 1/2 factor, consistent with NDI convention):
        J = sum_{t=0}^{T} e_t^T Q e_t + sum_{t=0}^{T-1} v_t^T R v_t

    Constraints:
        ||e_t||_inf <= e_lim  (state box around equilibrium, enforced via bounds)
        0 <= u_t <= 2  <=>  v_t in [U_LO - UE, U_HI - UE]

    KKT conditions (for control constraints only, consistent with NDI formulation):
        Terminal costate:   lambda_T = Q * e_T
        Costate recursion:  lambda_t = Q * e_t + A_d^T * lambda_{t+1}  (LINEAR)
        Stationarity:       R * v_t + B_d^T * lambda_{t+1} + nu_up - nu_lo = 0  (LINEAR)
        Complementarity:    nu_up_t * (v_hi - v_t) = 0  (bilinear)
                            nu_lo_t * (v_t - v_lo) = 0  (bilinear)

    Objective: maximize J_kkt - J_opt
    """

    def __init__(self, T=5, r=0.1, e_lim=0.2, dt=1.0, verbose=True):
        n_x = 2
        n_u = 1

        A_d, B_d = _stir_tank_discrete(dt)

        Q     = np.eye(n_x)
        R_mat = r * np.eye(n_u)

        # Control error bounds: v = u - ue, u in [U_LO, U_HI]
        v_lo = U_LO - UE   # = -0.7853
        v_hi = U_HI - UE   # =  1.2147

        M = gp.Model("stir_tank_verify")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.NonConvex  = 2   # required for bilinear complementarity & non-convex obj
        self.model = M
        self.T     = T
        self.n_x   = n_x
        self.n_u   = n_u

        # ---------------------------------------------------------------
        # Shared initial state error (free within box)
        # ---------------------------------------------------------------
        e0 = M.addVars(n_x, lb=-e_lim, ub=e_lim, name="e0")
        self.e0 = e0

        # ---------------------------------------------------------------
        # KKT-point trajectory variables (error coordinates)
        # ---------------------------------------------------------------
        e_kkt  = {t: M.addVars(n_x, lb=-e_lim, ub=e_lim, name=f"ek_{t}")
                  for t in range(T + 1)}
        v_kkt  = {t: M.addVars(n_u, lb=v_lo, ub=v_hi, name=f"vk_{t}")
                  for t in range(T)}
        lam_kkt = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lk_{t}")
                   for t in range(T + 1)}
        nu_up  = {t: M.addVars(n_u, lb=0.0, name=f"nu_up_{t}") for t in range(T)}
        nu_lo  = {t: M.addVars(n_u, lb=0.0, name=f"nu_lo_{t}") for t in range(T)}

        # ---------------------------------------------------------------
        # Feasible (optimal) trajectory variables
        # ---------------------------------------------------------------
        e_opt = {t: M.addVars(n_x, lb=-e_lim, ub=e_lim, name=f"eo_{t}")
                 for t in range(T + 1)}
        v_opt = {t: M.addVars(n_u, lb=v_lo, ub=v_hi, name=f"vo_{t}")
                 for t in range(T)}

        # ---------------------------------------------------------------
        # Shared initial condition
        # ---------------------------------------------------------------
        for i in range(n_x):
            M.addConstr(e_kkt[0][i] == e0[i], name=f"ic_kkt_{i}")
            M.addConstr(e_opt[0][i] == e0[i], name=f"ic_opt_{i}")

        # ---------------------------------------------------------------
        # Dynamics: e_{t+1} = A_d * e_t + B_d * v_t  (LINEAR)
        # ---------------------------------------------------------------
        for t in range(T):
            for i in range(n_x):
                rhs_kkt = (gp.quicksum(A_d[i, j] * e_kkt[t][j] for j in range(n_x))
                           + gp.quicksum(B_d[i, j] * v_kkt[t][j] for j in range(n_u)))
                M.addConstr(e_kkt[t + 1][i] == rhs_kkt, name=f"dynk_{t}_{i}")

                rhs_opt = (gp.quicksum(A_d[i, j] * e_opt[t][j] for j in range(n_x))
                           + gp.quicksum(B_d[i, j] * v_opt[t][j] for j in range(n_u)))
                M.addConstr(e_opt[t + 1][i] == rhs_opt, name=f"dyno_{t}_{i}")

        # ---------------------------------------------------------------
        # Terminal costate: lambda_T = Q * e_T
        # ---------------------------------------------------------------
        for i in range(n_x):
            M.addConstr(lam_kkt[T][i] == Q[i, i] * e_kkt[T][i], name=f"lam_term_{i}")

        # ---------------------------------------------------------------
        # Costate backward recursion: lambda_t = Q * e_t + A_d^T * lambda_{t+1}
        # (fully LINEAR — no bilinear products because dynamics are linear)
        # ---------------------------------------------------------------
        for t in range(T - 1, -1, -1):
            for i in range(n_x):
                AT_lam = gp.quicksum(A_d[j, i] * lam_kkt[t + 1][j] for j in range(n_x))
                M.addConstr(
                    lam_kkt[t][i] == Q[i, i] * e_kkt[t][i] + AT_lam,
                    name=f"costate_{t}_{i}"
                )

        # ---------------------------------------------------------------
        # Control stationarity: R * v_t + B_d^T * lambda_{t+1} + nu_up - nu_lo = 0
        # (LINEAR)
        # ---------------------------------------------------------------
        for t in range(T):
            for j in range(n_u):
                BT_lam = gp.quicksum(B_d[i, j] * lam_kkt[t + 1][i] for i in range(n_x))
                M.addConstr(
                    R_mat[j, j] * v_kkt[t][j] + BT_lam + nu_up[t][j] - nu_lo[t][j] == 0,
                    name=f"station_{t}_{j}"
                )

        # ---------------------------------------------------------------
        # Complementarity (bilinear)
        # ---------------------------------------------------------------
        for t in range(T):
            for j in range(n_u):
                M.addConstr(
                    nu_up[t][j] * (v_hi - v_kkt[t][j]) == 0,
                    name=f"comp_up_{t}_{j}"
                )
                M.addConstr(
                    nu_lo[t][j] * (v_kkt[t][j] - v_lo) == 0,
                    name=f"comp_lo_{t}_{j}"
                )

        # ---------------------------------------------------------------
        # Store variable references
        # ---------------------------------------------------------------
        self.e_kkt_traj = e_kkt
        self.v_kkt_traj = v_kkt
        self.lam_kkt    = lam_kkt
        self.nu_up      = nu_up
        self.nu_lo      = nu_lo
        self.e_opt_traj = e_opt
        self.v_opt_traj = v_opt

        self.e_kkt = e_kkt[T]
        self.e_opt = e_opt[T]

        # ---------------------------------------------------------------
        # Worst-case objective: maximize J_kkt - J_opt
        #   J = sum_{t=0}^{T} e_t^T Q e_t + sum_{t=0}^{T-1} v_t^T R v_t
        # ---------------------------------------------------------------
        V_kkt = (
            gp.quicksum(Q[i, i] * e_kkt[t][i] * e_kkt[t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R_mat[j, j] * v_kkt[t][j] * v_kkt[t][j]
                          for t in range(T) for j in range(n_u))
        )
        V_opt = (
            gp.quicksum(Q[i, i] * e_opt[t][i] * e_opt[t][i]
                        for t in range(T + 1) for i in range(n_x))
            + gp.quicksum(R_mat[j, j] * v_opt[t][j] * v_opt[t][j]
                          for t in range(T) for j in range(n_u))
        )

        self.orig_objective = V_kkt - V_opt
        M.setObjective(self.orig_objective, GRB.MAXIMIZE)

    def solve(self):
        self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
        self.model.optimize()
        return self.model.Status, self.model.Runtime

    def solution_dict(self):
        def v2dict(vs):
            return {k: vs[k].X for k in vs}

        if self.model.SolCount == 0:
            return {"obj": None}

        return {
            "obj":     self.model.ObjVal,
            "e0":      {i: self.e0[i].X for i in range(self.n_x)},
            "e_kkt":   {t: v2dict(self.e_kkt_traj[t]) for t in range(self.T + 1)},
            "v_kkt":   {t: v2dict(self.v_kkt_traj[t]) for t in range(self.T)},
            "lam_kkt": {t: v2dict(self.lam_kkt[t])    for t in range(self.T + 1)},
            "nu_up":   {t: v2dict(self.nu_up[t])       for t in range(self.T)},
            "nu_lo":   {t: v2dict(self.nu_lo[t])       for t in range(self.T)},
            "e_opt":   {t: v2dict(self.e_opt_traj[t])  for t in range(self.T + 1)},
            "v_opt":   {t: v2dict(self.v_opt_traj[t])  for t in range(self.T)},
        }
