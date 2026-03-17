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
    ax.set_yscale('log')
    ax.set_title(r'Stir tank: worst-case KKT suboptimality vs.\ horizon')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('stir_tank_suboptimality.pdf', bbox_inches='tight')
    print("Plot saved to stir_tank_suboptimality.pdf")
    return obj_vals


# class StirTankVerify:
#     """
#     Finds the worst-case performance gap between any KKT point and a feasible
#     (optimal) solution for a T-horizon MPC problem on the nonlinear stir tank
#     reactor.

#     Continuous-time dynamics:
#         xdot1 = (1 - x1) / theta - k * x1 * exp(-M / x2)
#         xdot2 = (xf - x2) / theta + k * x1 * exp(-M / x2) - gamma * u * (x2 - xc)

#     Parameters: theta=20, k=300, M=5, xf=0.3947, xc=0.3816, gamma=0.117
#     Equilibrium: xe=[0.2632, 0.6519], ue=0.7853

#     The MPC is formulated in error coordinates e = x - xe, v = u - ue using a
#     linearization of the dynamics around (xe, ue) discretized with Euler step dt:
#         e_{t+1} = A_d * e_t + B_d * v_t

#     MPC cost (without the 1/2 factor, consistent with NDI convention):
#         J = sum_{t=0}^{T} e_t^T Q e_t + sum_{t=0}^{T-1} v_t^T R v_t

#     Constraints:
#         ||e_t||_inf <= e_lim  (state box around equilibrium, enforced via bounds)
#         0 <= u_t <= 2  <=>  v_t in [U_LO - UE, U_HI - UE]

#     KKT conditions (for control constraints only, consistent with NDI formulation):
#         Terminal costate:   lambda_T = Q * e_T
#         Costate recursion:  lambda_t = Q * e_t + A_d^T * lambda_{t+1}  (LINEAR)
#         Stationarity:       R * v_t + B_d^T * lambda_{t+1} + nu_up - nu_lo = 0  (LINEAR)
#         Complementarity:    nu_up_t * (v_hi - v_t) = 0  (bilinear)
#                             nu_lo_t * (v_t - v_lo) = 0  (bilinear)

#     Objective: maximize J_kkt - J_opt
#     """

#     def __init__(self, T=5, r=0.1, e_lim=0.2, dt=1.0, verbose=True):
#         n_x = 2
#         n_u = 1

#         A_d, B_d = _stir_tank_discrete(dt)

#         Q     = np.eye(n_x)
#         R_mat = r * np.eye(n_u)

#         # Control error bounds: v = u - ue, u in [U_LO, U_HI]
#         v_lo = U_LO - UE   # = -0.7853
#         v_hi = U_HI - UE   # =  1.2147

#         M = gp.Model("stir_tank_verify")
#         M.Params.OutputFlag = 1 if verbose else 0
#         M.Params.NonConvex  = 2   # required for bilinear complementarity & non-convex obj
#         self.model = M
#         self.T     = T
#         self.n_x   = n_x
#         self.n_u   = n_u

#         # ---------------------------------------------------------------
#         # Shared initial state error (free within box)
#         # ---------------------------------------------------------------
#         e0 = M.addVars(n_x, lb=-e_lim, ub=e_lim, name="e0")
#         self.e0 = e0

#         # ---------------------------------------------------------------
#         # KKT-point trajectory variables (error coordinates)
#         # ---------------------------------------------------------------
#         e_kkt  = {t: M.addVars(n_x, lb=-e_lim, ub=e_lim, name=f"ek_{t}")
#                   for t in range(T + 1)}
#         v_kkt  = {t: M.addVars(n_u, lb=v_lo, ub=v_hi, name=f"vk_{t}")
#                   for t in range(T)}
#         lam_kkt = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lk_{t}")
#                    for t in range(T + 1)}
#         nu_up  = {t: M.addVars(n_u, lb=0.0, name=f"nu_up_{t}") for t in range(T)}
#         nu_lo  = {t: M.addVars(n_u, lb=0.0, name=f"nu_lo_{t}") for t in range(T)}

#         # ---------------------------------------------------------------
#         # Feasible (optimal) trajectory variables
#         # ---------------------------------------------------------------
#         e_opt = {t: M.addVars(n_x, lb=-e_lim, ub=e_lim, name=f"eo_{t}")
#                  for t in range(T + 1)}
#         v_opt = {t: M.addVars(n_u, lb=v_lo, ub=v_hi, name=f"vo_{t}")
#                  for t in range(T)}

#         # ---------------------------------------------------------------
#         # Shared initial condition
#         # ---------------------------------------------------------------
#         for i in range(n_x):
#             M.addConstr(e_kkt[0][i] == e0[i], name=f"ic_kkt_{i}")
#             M.addConstr(e_opt[0][i] == e0[i], name=f"ic_opt_{i}")

#         # ---------------------------------------------------------------
#         # Dynamics: e_{t+1} = A_d * e_t + B_d * v_t  (LINEAR)
#         # ---------------------------------------------------------------
#         for t in range(T):
#             for i in range(n_x):
#                 rhs_kkt = (gp.quicksum(A_d[i, j] * e_kkt[t][j] for j in range(n_x))
#                            + gp.quicksum(B_d[i, j] * v_kkt[t][j] for j in range(n_u)))
#                 M.addConstr(e_kkt[t + 1][i] == rhs_kkt, name=f"dynk_{t}_{i}")

#                 rhs_opt = (gp.quicksum(A_d[i, j] * e_opt[t][j] for j in range(n_x))
#                            + gp.quicksum(B_d[i, j] * v_opt[t][j] for j in range(n_u)))
#                 M.addConstr(e_opt[t + 1][i] == rhs_opt, name=f"dyno_{t}_{i}")

#         # ---------------------------------------------------------------
#         # Terminal costate: lambda_T = Q * e_T
#         # ---------------------------------------------------------------
#         for i in range(n_x):
#             M.addConstr(lam_kkt[T][i] == Q[i, i] * e_kkt[T][i], name=f"lam_term_{i}")

#         # ---------------------------------------------------------------
#         # Costate backward recursion: lambda_t = Q * e_t + A_d^T * lambda_{t+1}
#         # (fully LINEAR — no bilinear products because dynamics are linear)
#         # ---------------------------------------------------------------
#         for t in range(T - 1, -1, -1):
#             for i in range(n_x):
#                 AT_lam = gp.quicksum(A_d[j, i] * lam_kkt[t + 1][j] for j in range(n_x))
#                 M.addConstr(
#                     lam_kkt[t][i] == Q[i, i] * e_kkt[t][i] + AT_lam,
#                     name=f"costate_{t}_{i}"
#                 )

#         # ---------------------------------------------------------------
#         # Control stationarity: R * v_t + B_d^T * lambda_{t+1} + nu_up - nu_lo = 0
#         # (LINEAR)
#         # ---------------------------------------------------------------
#         for t in range(T):
#             for j in range(n_u):
#                 BT_lam = gp.quicksum(B_d[i, j] * lam_kkt[t + 1][i] for i in range(n_x))
#                 M.addConstr(
#                     R_mat[j, j] * v_kkt[t][j] + BT_lam + nu_up[t][j] - nu_lo[t][j] == 0,
#                     name=f"station_{t}_{j}"
#                 )

#         # ---------------------------------------------------------------
#         # Complementarity (bilinear)
#         # ---------------------------------------------------------------
#         for t in range(T):
#             for j in range(n_u):
#                 M.addConstr(
#                     nu_up[t][j] * (v_hi - v_kkt[t][j]) == 0,
#                     name=f"comp_up_{t}_{j}"
#                 )
#                 M.addConstr(
#                     nu_lo[t][j] * (v_kkt[t][j] - v_lo) == 0,
#                     name=f"comp_lo_{t}_{j}"
#                 )

#         # ---------------------------------------------------------------
#         # Store variable references
#         # ---------------------------------------------------------------
#         self.e_kkt_traj = e_kkt
#         self.v_kkt_traj = v_kkt
#         self.lam_kkt    = lam_kkt
#         self.nu_up      = nu_up
#         self.nu_lo      = nu_lo
#         self.e_opt_traj = e_opt
#         self.v_opt_traj = v_opt

#         self.e_kkt = e_kkt[T]
#         self.e_opt = e_opt[T]

#         # ---------------------------------------------------------------
#         # Worst-case objective: maximize J_kkt - J_opt
#         #   J = sum_{t=0}^{T} e_t^T Q e_t + sum_{t=0}^{T-1} v_t^T R v_t
#         # ---------------------------------------------------------------
#         V_kkt = (
#             gp.quicksum(Q[i, i] * e_kkt[t][i] * e_kkt[t][i]
#                         for t in range(T + 1) for i in range(n_x))
#             + gp.quicksum(R_mat[j, j] * v_kkt[t][j] * v_kkt[t][j]
#                           for t in range(T) for j in range(n_u))
#         )
#         V_opt = (
#             gp.quicksum(Q[i, i] * e_opt[t][i] * e_opt[t][i]
#                         for t in range(T + 1) for i in range(n_x))
#             + gp.quicksum(R_mat[j, j] * v_opt[t][j] * v_opt[t][j]
#                           for t in range(T) for j in range(n_u))
#         )

#         self.orig_objective = V_kkt - V_opt
#         M.setObjective(self.orig_objective, GRB.MAXIMIZE)

#     def solve(self):
#         self.model.setObjective(self.orig_objective, GRB.MAXIMIZE)
#         self.model.optimize()
#         return self.model.Status, self.model.Runtime

#     def solution_dict(self):
#         def v2dict(vs):
#             return {k: vs[k].X for k in vs}

#         if self.model.SolCount == 0:
#             return {"obj": None}

#         return {
#             "obj":     self.model.ObjVal,
#             "e0":      {i: self.e0[i].X for i in range(self.n_x)},
#             "e_kkt":   {t: v2dict(self.e_kkt_traj[t]) for t in range(self.T + 1)},
#             "v_kkt":   {t: v2dict(self.v_kkt_traj[t]) for t in range(self.T)},
#             "lam_kkt": {t: v2dict(self.lam_kkt[t])    for t in range(self.T + 1)},
#             "nu_up":   {t: v2dict(self.nu_up[t])       for t in range(self.T)},
#             "nu_lo":   {t: v2dict(self.nu_lo[t])       for t in range(self.T)},
#             "e_opt":   {t: v2dict(self.e_opt_traj[t])  for t in range(self.T + 1)},
#             "v_opt":   {t: v2dict(self.v_opt_traj[t])  for t in range(self.T)},
#         }

# import numpy as np
# import gurobipy as gp
# from gurobipy import GRB

# # ---------------------------------------------------------------
# # Stir tank parameters
# # ---------------------------------------------------------------
THETA = 20.0; K = 300.0; M_par = 5.0
XF = 0.3947; XC = 0.3816; GAMMA = 0.117
XE = np.array([0.2632, 0.6519])
UE = 0.7853
U_LO, U_HI = 0.0, 2.0


def _f_nonlinear(x1, x2, u):
    """True continuous-time RHS as symbolic-friendly expressions.
    Returns (f1, f2) as Gurobi LinExpr / QuadExpr / float depending on inputs.
    x1, x2, u may be Gurobi Var objects or floats.
    """
    raise NotImplementedError  # see _add_nonlinear_dynamics below


def _euler_step(M, e_t, v_t, dt, n_x=2):
    """
    Add Euler-step constraints for the TRUE nonlinear dynamics in error coords.

    Continuous dynamics in absolute coords x = e + xe, u = v + ue:
        xdot1 = (1 - x1)/theta - k*x1*exp(-M_par/x2)
        xdot2 = (xf - x2)/theta + k*x1*exp(-M_par/x2) - gamma*u*(x2-xc)

    We introduce auxiliary variables for the nonlinear terms:
        phi  = x1 * exp(-M_par / x2)    [Arrhenius reaction rate]
        psi  = u * (x2 - xc)            [cooling term]

    phi is handled by encoding exp(-M_par/x2) via a general constraint (GenConstrExp),
    or by letting NonConvex=2 handle the product x1 * exp_term.

    Returns the expressions for e_{t+1} as GRB expressions.
    """
    x1 = e_t[0] + XE[0]
    x2 = e_t[1] + XE[1]
    u  = v_t[0] + UE

    # --- auxiliary: z = -M_par / x2 (nonlinear division) ---
    # We model exp(-M_par/x2) as an auxiliary exp_var using addGenConstrExp.
    # But x2 is a decision variable, so -M_par/x2 is nonlinear.
    # Strategy: introduce aux_z s.t. x2 * aux_z = -M_par  (bilinear = const)
    # then exp_var = exp(aux_z).  This works with NonConvex=2.

    # aux_z: x2 * aux_z = -M_par
    aux_z   = M.addVar(lb=-GRB.INFINITY, ub=0.0, name="aux_z")
    M.addConstr(e_t[1] + XE[1], GRB.EQUAL, 0)  # placeholder; see below
    # Actually encode directly: introduce phi = x1 * exp(-M/x2) as:
    #   aux_ratio = M_par / x2   (bilinear: x2 * aux_ratio = M_par)
    #   aux_neg   = -aux_ratio
    #   exp_term  = exp(aux_neg)  via addGenConstrExp
    #   phi       = x1 * exp_term (bilinear)
    raise NotImplementedError  # stub; full version below


class StirTankVerify:
    """
    Worst-case suboptimality between any KKT point and the optimal solution
    for the TRUE NONLINEAR stir-tank MPC.

    Key differences from the linearized version:
      1. Dynamics: true nonlinear Euler steps, not A_d*e + B_d*v
      2. Costate recursion: uses ∂f/∂e(e_t, v_t)^T λ_{t+1}, state-dependent
      3. Stationarity: uses ∂f/∂u(e_t, v_t)^T λ_{t+1}, also state-dependent
      4. All three above introduce bilinear products → NonConvex=2

    Nonlinear terms in the dynamics (absolute coords x = e+xe, u = v+ue):
        reaction  φ(x1,x2)   = x1 · exp(-M/x2)
        cooling   ψ(x2, u)   = u  · (x2 - xc)

    Their Jacobians (needed for KKT):
        ∂φ/∂x1 = exp(-M/x2)
        ∂φ/∂x2 = x1 · (M/x2²) · exp(-M/x2)
        ∂ψ/∂x2 = u
        ∂ψ/∂u  = x2 - xc

    Continuous dynamics:
        f1(x,u) = (1-x1)/θ  - k·φ
        f2(x,u) = (xf-x2)/θ + k·φ  - γ·ψ

    Jacobian ∂f/∂x (2×2):
        df1/dx1 = -1/θ - k·∂φ/∂x1
        df1/dx2 =       - k·∂φ/∂x2
        df2/dx1 =         k·∂φ/∂x1
        df2/dx2 = -1/θ  + k·∂φ/∂x2 - γ·∂ψ/∂x2

    Jacobian ∂f/∂u (2×1):
        df1/du = 0
        df2/du = -γ·∂ψ/∂u = -γ·(x2 - xc)

    Discretized Jacobian (Euler, dt):
        Jx_d = I + dt · ∂f/∂x
        Ju_d = dt · ∂f/∂u
    """

    def __init__(self, T=5, r=0.1, e_lim=0.2, dt=1.0, verbose=True):
        n_x = 2
        n_u = 1

        Q     = np.eye(n_x)
        R_mat = r * np.eye(n_u)

        v_lo = U_LO - UE
        v_hi = U_HI - UE

        M = gp.Model("stir_tank_verify_nonlinear")
        M.Params.OutputFlag = 1 if verbose else 0
        M.Params.NonConvex  = 2
        self.model = M
        self.T = T; self.n_x = n_x; self.n_u = n_u

        # ---------------------------------------------------------------
        # State/control trajectory variables
        # ---------------------------------------------------------------
        e0 = M.addVars(n_x, lb=-e_lim, ub=e_lim, name="e0")

        e_kkt = {t: M.addVars(n_x, lb=-e_lim, ub=e_lim, name=f"ek_{t}")
                 for t in range(T + 1)}
        v_kkt = {t: M.addVars(n_u, lb=v_lo,   ub=v_hi,  name=f"vk_{t}")
                 for t in range(T)}

        e_opt = {t: M.addVars(n_x, lb=-e_lim, ub=e_lim, name=f"eo_{t}")
                 for t in range(T + 1)}
        v_opt = {t: M.addVars(n_u, lb=v_lo,   ub=v_hi,  name=f"vo_{t}")
                 for t in range(T)}

        # Dual variables for KKT
        lam = {t: M.addVars(n_x, lb=-GRB.INFINITY, name=f"lam_{t}")
               for t in range(T + 1)}
        nu_up = {t: M.addVars(n_u, lb=0.0, name=f"nu_up_{t}") for t in range(T)}
        nu_lo = {t: M.addVars(n_u, lb=0.0, name=f"nu_lo_{t}") for t in range(T)}

        # ---------------------------------------------------------------
        # Shared initial condition
        # ---------------------------------------------------------------
        for i in range(n_x):
            M.addConstr(e_kkt[0][i] == e0[i])
            M.addConstr(e_opt[0][i] == e0[i])

        # ---------------------------------------------------------------
        # Helper: add auxiliary variables for the nonlinear terms at (e,v)
        # Returns (phi, exp_term, psi) as Gurobi Var
        #
        # phi  = (e[0]+xe[0]) * exp(-M_par / (e[1]+xe[1]))
        # psi  = (v[0]+ue)    * ((e[1]+xe[1]) - xc)
        # ---------------------------------------------------------------
        def add_nl_aux(e, v, tag):
            x1 = M.addVar(lb=XE[0]-e_lim, ub=XE[0]+e_lim, name=f"x1_{tag}")
            x2 = M.addVar(lb=XE[1]-e_lim, ub=XE[1]+e_lim, name=f"x2_{tag}")
            u  = M.addVar(lb=U_LO, ub=U_HI, name=f"u_{tag}")
            M.addConstr(x1 == e[0] + XE[0])
            M.addConstr(x2 == e[1] + XE[1])
            M.addConstr(u  == v[0] + UE)

            # ratio = M_par / x2  (bilinear: x2 * ratio = M_par)
            ratio = M.addVar(lb=M_par/(XE[1]+e_lim),
                             ub=M_par/(XE[1]-e_lim), name=f"ratio_{tag}")
            M.addConstr(x2 * ratio == M_par)

            # neg_ratio = -ratio
            neg_ratio = M.addVar(lb=-M_par/(XE[1]-e_lim),
                                 ub=-M_par/(XE[1]+e_lim), name=f"nratio_{tag}")
            M.addConstr(neg_ratio == -ratio)

            # exp_term = exp(neg_ratio) = exp(-M_par/x2)
            exp_term = M.addVar(lb=0.0, ub=GRB.INFINITY, name=f"exp_{tag}")
            M.addGenConstrExp(neg_ratio, exp_term, name=f"gc_exp_{tag}")

            # phi = x1 * exp_term  (bilinear)
            phi = M.addVar(lb=0.0, ub=GRB.INFINITY, name=f"phi_{tag}")
            M.addConstr(phi == x1 * exp_term)

            # psi = u * (x2 - xc)  (bilinear)
            psi = M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"psi_{tag}")
            M.addConstr(psi == u * (x2 - XC))

            return x1, x2, u, ratio, exp_term, phi, psi

        # ---------------------------------------------------------------
        # True nonlinear Euler dynamics:
        #   e_{t+1}[0] = e_t[0] + dt * [ (1-x1)/θ - k*φ ]  - xe[0]*(step)
        # In error coords, since e = x - xe and xe is the equilibrium:
        #   ė = f(x,u) - f(xe,ue)  (error dynamics, but we just use Euler on x)
        #   x_{t+1} = x_t + dt * f(x_t, u_t)
        #   e_{t+1} = x_{t+1} - xe
        # ---------------------------------------------------------------
        def add_euler_dynamics(e_traj, v_traj, suffix):
            for t in range(T):
                x1, x2, u, _, _, phi, psi = add_nl_aux(e_traj[t], v_traj[t],
                                                        f"{suffix}_{t}")
                # f1 = (1 - x1)/theta - k*phi
                # f2 = (xf - x2)/theta + k*phi - gamma*u*(x2-xc)  = ... - gamma*psi
                # e_{t+1}[i] = x_{t+1}[i] - xe[i]
                #             = (x_t[i] + dt*fi(x_t,u_t)) - xe[i]
                #             = e_t[i] + dt*fi  (since xe fixed)
                # fi itself contains (xe+e) so we already used x1,x2 above
                f1 = (1.0 - x1) / THETA - K * phi
                f2 = (XF  - x2) / THETA + K * phi - GAMMA * psi

                M.addConstr(e_traj[t+1][0] == e_traj[t][0] + dt * f1)
                M.addConstr(e_traj[t+1][1] == e_traj[t][1] + dt * f2)

        add_euler_dynamics(e_kkt, v_kkt, "kkt")
        add_euler_dynamics(e_opt, v_opt, "opt")

        # ---------------------------------------------------------------
        # KKT conditions for the nonlinear MPC
        # ---------------------------------------------------------------

        # Terminal costate: λ_T = Q * e_T  (linear, same as before)
        for i in range(n_x):
            M.addConstr(lam[T][i] == Q[i, i] * e_kkt[T][i])

        # ---------------------------------------------------------------
        # Costate backward recursion (nonlinear adjoint equation):
        #   λ_t = Q*e_t + Jx_d(e_t,v_t)^T * λ_{t+1}
        # where Jx_d = I + dt * ∂f/∂x  (discrete-time Jacobian)
        #
        # ∂f/∂x entries (in terms of phi and x2):
        #   df1/dx1 = -1/θ - k*exp_term
        #   df1/dx2 = -k * x1 * (M/x2²) * exp_term  = -k * phi * ratio/x2
        #           = -k * phi * (ratio / x2)   — use aux: phi * ratio bilinear
        #   df2/dx1 =  k*exp_term
        #   df2/dx2 = -1/θ + k*phi*(ratio/x2) - γ*u
        #
        # All products of lam[t+1][j] with state-dependent Jacobian entries
        # are bilinear → introduce auxiliary products w[t][i,j] = Jx[i,j]*lam[t+1][j]
        # ---------------------------------------------------------------

        def add_costate_recursion():
            for t in range(T - 1, -1, -1):
                # Re-use aux vars if available; here we create fresh ones tagged "jac"
                x1, x2, u, ratio, exp_term, phi, psi = add_nl_aux(
                    e_kkt[t], v_kkt[t], f"jac_{t}")

                # ---- Jacobian entries of continuous f ----
                # phi_ratio = phi * ratio  (bilinear: phi * ratio, both aux vars)
                phi_ratio = M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"phi_ratio_{t}")
                M.addConstr(phi_ratio == phi * ratio)

                # phi_ratio_over_x2 = phi_ratio / x2 = phi * (M/x2²) * (x2/M)
                # Easier: phi_ratio / x2 → bilinear: x2 * phr_x2 = phi_ratio
                phr_x2 = M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                  name=f"phr_x2_{t}")
                M.addConstr(x2 * phr_x2 == phi_ratio)

                # Continuous Jacobian entries
                # df/dx  (2x2):
                #   [0,0]: -1/θ - k*exp_term
                #   [0,1]: -k * phr_x2
                #   [1,0]:  k * exp_term
                #   [1,1]: -1/θ + k*phr_x2 - γ*u
                #
                # Discrete Jacobian Jx_d = I + dt*df/dx:
                #   [0,0]: 1 + dt*(-1/θ - k*exp_term)    = 1 - dt/θ - dt*k*exp_term
                #   [0,1]: -dt*k * phr_x2
                #   [1,0]:  dt*k * exp_term
                #   [1,1]: 1 - dt/θ + dt*k*phr_x2 - dt*γ*u

                # Bilinear products:  Jx_d[i,j] * lam[t+1][j]
                # We create aux vars w[i] = sum_j Jx_d[i,j] * lam[t+1][j]

                # w00 = (1 - dt/θ - dt*k*exp_term) * lam[t+1][0]
                #     = (1-dt/θ)*lam[t+1][0]  - dt*k * (exp_term * lam[t+1][0])
                el0 = lam[t+1][0]; el1 = lam[t+1][1]

                # Bilinear aux: exp_term * el0, exp_term * el1, phr_x2 * el0,
                #               phr_x2 * el1, u * el1
                def bil(a, b, name):
                    w = M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=name)
                    M.addConstr(w == a * b)
                    return w

                et_el0 = bil(exp_term, el0, f"et_el0_{t}")
                et_el1 = bil(exp_term, el1, f"et_el1_{t}")
                prx2_el0 = bil(phr_x2, el0, f"prx2_el0_{t}")
                prx2_el1 = bil(phr_x2, el1, f"prx2_el1_{t}")
                u_el1    = bil(u,   el1,    f"u_el1_{t}")

                # w0 = Jx_d[0,:] * lam[t+1]
                #    = (1-dt/θ)*el0 - dt*k*et_el0   (row 0, col 0 term)
                #    + (-dt*k*phr_x2)*el1            (row 0, col 1 term)
                w0 = ((1.0 - dt/THETA) * el0
                      - dt * K * et_el0
                      - dt * K * prx2_el1)

                # w1 = Jx_d[1,:] * lam[t+1]
                #    = dt*k*exp_term*el0              (row 1, col 0 term)
                #    + (1-dt/θ + dt*k*phr_x2 - dt*γ*u)*el1
                w1 = (dt * K * et_el0
                      + (1.0 - dt/THETA) * el1
                      + dt * K * prx2_el1
                      - dt * GAMMA * u_el1)

                # λ_t = Q*e_t + Jx_d^T * λ_{t+1}
                M.addConstr(lam[t][0] == Q[0,0]*e_kkt[t][0] + w0)
                M.addConstr(lam[t][1] == Q[1,1]*e_kkt[t][1] + w1)

        add_costate_recursion()

        # ---------------------------------------------------------------
        # Stationarity: R*v_t + Ju_d(e_t,v_t)^T * λ_{t+1} + ν_up - ν_lo = 0
        #
        # Continuous ∂f/∂u (2×1):
        #   df1/du = 0
        #   df2/du = -γ*(x2 - xc)
        # Discrete Ju_d = dt * ∂f/∂u:
        #   Ju_d[0] = 0
        #   Ju_d[1] = -dt*γ*(x2 - xc)
        #
        # Stationarity (scalar v, scalar u component j=0):
        #   R*v + Ju_d^T * λ_{t+1} + ν_up - ν_lo = 0
        #   R*v + (0)*λ_{t+1,0} + (-dt*γ*(x2-xc))*λ_{t+1,1} + ν_up - ν_lo = 0
        #
        # New bilinear: (x2 - xc) * lam[t+1][1]
        # ---------------------------------------------------------------
        def add_stationarity():
            for t in range(T):
                # Reuse geometry: need x2 at time t for KKT trajectory
                x1, x2, u, ratio, exp_term, phi, psi = add_nl_aux(
                    e_kkt[t], v_kkt[t], f"stat_{t}")

                el1 = lam[t+1][1]
                # bilinear: (x2 - xc) * lam[t+1][1]
                x2_el1 = M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                  name=f"x2_el1_stat_{t}")
                M.addConstr(x2_el1 == x2 * el1)

                # Stationarity for j=0:
                M.addConstr(
                    R_mat[0,0] * v_kkt[t][0]
                    + (-dt * GAMMA) * (x2_el1 - XC * el1)
                    + nu_up[t][0] - nu_lo[t][0] == 0.0
                )

        add_stationarity()

        # ---------------------------------------------------------------
        # Complementarity (unchanged — same bilinear structure)
        # ---------------------------------------------------------------
        for t in range(T):
            M.addConstr(nu_up[t][0] * (v_hi - v_kkt[t][0]) == 0)
            M.addConstr(nu_lo[t][0] * (v_kkt[t][0] - v_lo) == 0)

        # ---------------------------------------------------------------
        # Store references
        # ---------------------------------------------------------------
        self.e0 = e0
        self.e_kkt_traj = e_kkt; self.v_kkt_traj = v_kkt
        self.e_opt_traj = e_opt; self.v_opt_traj = v_opt
        self.lam = lam; self.nu_up = nu_up; self.nu_lo = nu_lo

        # ---------------------------------------------------------------
        # Objective: maximize J_kkt - J_opt  (same quadratic form, true states)
        # ---------------------------------------------------------------
        def quadratic_cost(e_traj, v_traj):
            return (
                gp.quicksum(Q[i,i] * e_traj[t][i] * e_traj[t][i]
                            for t in range(T+1) for i in range(n_x))
                + gp.quicksum(R_mat[j,j] * v_traj[t][j] * v_traj[t][j]
                              for t in range(T) for j in range(n_u))
            )

        self.orig_objective = quadratic_cost(e_kkt, v_kkt) - quadratic_cost(e_opt, v_opt)
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
            "lam":     {t: v2dict(self.lam[t])         for t in range(self.T + 1)},
            "nu_up":   {t: v2dict(self.nu_up[t])       for t in range(self.T)},
            "nu_lo":   {t: v2dict(self.nu_lo[t])       for t in range(self.T)},
            "e_opt":   {t: v2dict(self.e_opt_traj[t])  for t in range(self.T + 1)},
            "v_opt":   {t: v2dict(self.v_opt_traj[t])  for t in range(self.T)},
        }