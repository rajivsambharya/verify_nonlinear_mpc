"""
Closed-loop iLQR on the Van der Pol oscillator:

  x1_{t+1} = x1_t + dt * x2_t
  x2_{t+1} = x2_t + dt * (mu*(1 - x1_t^2)*x2_t - x1_t + u_t)

The nonlinear damping term mu*(1-x1^2)*x2 creates a limit cycle at amplitude ~2.
Without control, all trajectories are attracted to this limit cycle.

The iLQR here is standard (Tassa/Li-Todorov style):
  backward pass computes feedforward k_t and feedback K_t via second-order
  expansion of the value function, then the forward pass applies:
    u_t = u_bar_t + k_t + K_t @ (x_t - x_bar_t)

Why iterations matter:
  Iter 0 (u_bar=0): x_bar spirals toward the limit cycle.  The Jacobian
    A[1,0] = dt*(-2*mu*x1*x2 - 1) and A[1,1] = 1 + dt*mu*(1-x1^2)
    vary strongly along the spiral, giving an LQR that is only a rough
    approximation of the optimal policy.
  Iter 1: x_bar follows the controlled trajectory (moving toward origin),
    giving far more accurate Jacobians near x=0 where the linearization is
    nearly exact.  The resulting policy matches the KKT solution closely.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# System: Van der Pol oscillator with scalar control on x2
# ---------------------------------------------------------------------------
DT  = 0.05
MU  = 2.0
R   = 0.1
U_MAX = 1e3
Q   = np.eye(2)


def step(x, u):
    x1, x2 = x
    return np.array([x1 + DT * x2,
                     x2 + DT * (MU * (1 - x1**2) * x2 - x1 + u)])


def jac_A(x_bar):
    x1, x2 = x_bar
    return np.array([[1.0,  DT],
                     [DT * (-2*MU*x1*x2 - 1),  1 + DT*MU*(1 - x1**2)]])


B = np.array([0.0, DT])   # df/du (constant)


# ---------------------------------------------------------------------------
# Standard iLQR (feedforward + feedback, no line-search)
# ---------------------------------------------------------------------------

def ilqr_control(x0, n_iters, T):
    """
    Run n_iters of standard iLQR from x0 with planning horizon T.
    Returns first control u[0].

    Each iteration:
      1. Forward pass with current u_bar → x_bar
      2. Backward pass: compute feedforward k_t and feedback K_t via
         second-order expansion of the value function
      3. Forward rollout update: u_t ← u_bar_t + k_t + K_t(x_t - x_bar_t)
    """
    x0 = np.asarray(x0, dtype=float)
    u_bar = np.zeros(T)

    for _ in range(n_iters):
        # 1. Forward pass
        x_bar = [x0.copy()]
        for t in range(T):
            x_bar.append(step(x_bar[-1], u_bar[t]))

        As = [jac_A(x_bar[t]) for t in range(T)]

        # 2. Backward pass
        V_x  = 2.0 * Q @ x_bar[T]   # gradient of terminal cost l_T = x^T Q x
        V_xx = 2.0 * Q               # hessian of terminal cost

        ks = []
        Ks = []
        for t in range(T - 1, -1, -1):
            A = As[t]
            Q_x  = 2.0 * Q @ x_bar[t]  + A.T @ V_x
            Q_u  = 2.0 * R * u_bar[t]  + float(B @ V_x)
            Q_xx = 2.0 * Q             + A.T @ V_xx @ A
            Q_uu = 2.0 * R             + float(B @ V_xx @ B)
            Q_ux = B @ V_xx @ A          # shape (n_x,)

            k =  -Q_u  / Q_uu            # scalar feedforward
            K =  -Q_ux / Q_uu            # (n_x,) feedback row

            V_x  = Q_x  + K * Q_u
            V_xx = Q_xx + np.outer(K, Q_ux)

            ks.insert(0, k)
            Ks.insert(0, K)

        # 3. Forward rollout with updated policy
        x = x0.copy()
        u_new = np.zeros(T)
        for t in range(T):
            dx = x - x_bar[t]
            u_new[t] = float(np.clip(u_bar[t] + ks[t] + Ks[t] @ dx, -U_MAX, U_MAX))
            x = step(x, u_new[t])

        u_bar = u_new

    return float(u_bar[0])


# ---------------------------------------------------------------------------
# KKT (true optimal MPC) via direct nonlinear optimization
# ---------------------------------------------------------------------------

def kkt_control(x0, T):
    """Solve the true nonlinear MPC; warm-start from 5-iter iLQR."""
    x0 = np.asarray(x0, dtype=float)

    def cost(u_seq):
        x = x0.copy()
        c = 0.0
        for u in u_seq:
            c += float(x @ Q @ x) + R * u**2
            x = np.clip(step(x, u), -1e4, 1e4)
        return c + float(x @ Q @ x)

    u_warm = np.zeros(T)
    u_warm[0] = ilqr_control(x0, 5, T)

    res = minimize(
        cost, u_warm,
        method='L-BFGS-B',
        bounds=[(-U_MAX, U_MAX)] * T,
        options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 2000},
    )
    return float(res.x[0])


# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------

def simulate(x0, N_sim, T_plan, n_iters=None, use_kkt=False):
    x = np.asarray(x0, dtype=float).copy()
    stage_costs = np.zeros(N_sim)
    states      = np.zeros((N_sim + 1, 2))
    states[0]   = x

    for k in range(N_sim):
        u = kkt_control(x, T_plan) if use_kkt else ilqr_control(x, n_iters, T_plan)
        stage_costs[k] = float(x @ Q @ x) + R * u**2
        x = step(x, u)
        states[k + 1] = x

    return stage_costs, states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    x0_vals      = [[2.0, 0.0], [2.5, 0.0], [1.5, 1.5]]
    n_iters_list = [1, 2, 3, 4, 5]
    N_sim        = 60
    T_plan       = 30

    FONT_SIZE = 22
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
    })
    plt.rc("xtick", labelsize=FONT_SIZE, labelcolor="black")
    plt.rc("ytick", labelsize=FONT_SIZE, labelcolor="black")

    iter_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_iters_list)))
    x0_colors   = plt.cm.Set1.colors
    markers     = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

    # -----------------------------------------------------------------------
    # Figure 1: stage cost trajectories for each x0
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(x0_vals), figsize=(5 * len(x0_vals), 5), sharey=False)
    times = np.arange(N_sim) * DT

    for ax, x0 in zip(axes, x0_vals):
        print(f"\nx0 = {x0}")
        for i, n in enumerate(n_iters_list):
            sc, _ = simulate(x0, N_sim, T_plan, n_iters=n)
            total = np.sum(sc)
            print(f"  n_iters={n}: total={total:.2f}")
            ax.semilogy(times, sc, label=f'$N={n}$', color=iter_colors[i], linewidth=2)
        sc_kkt, _ = simulate(x0, N_sim, T_plan, use_kkt=True)
        print(f"  KKT:      total={np.sum(sc_kkt):.2f}")
        ax.semilogy(times, sc_kkt, label='KKT', color='k', linewidth=2.5, linestyle='--')
        ax.set_title(f'$x_0 = {x0}$')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('stage cost $\\|x\\|^2 + r u^2$')
        ax.legend(fontsize=14)
        ax.grid(True)

    fig.suptitle(
        rf'Van der Pol: $\dot{{x}}_2 = \mu(1-x_1^2)x_2 - x_1 + u$, $\mu={MU}$',
        fontsize=FONT_SIZE - 2
    )
    fig.tight_layout()
    fig.savefig('ilqr_sim2_stage.pdf', bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 2: total cost vs n_iters
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for j, x0 in enumerate(x0_vals):
        totals = [np.sum(simulate(x0, N_sim, T_plan, n_iters=n)[0]) for n in n_iters_list]
        kkt_total = np.sum(simulate(x0, N_sim, T_plan, use_kkt=True)[0])
        color = x0_colors[j % len(x0_colors)]
        ax2.plot(n_iters_list, totals,
                 marker=markers[j % len(markers)], linewidth=2, color=color,
                 label=f'$x_0={x0}$')
        ax2.axhline(kkt_total, color=color, linewidth=1.5, linestyle='--')
    ax2.set_xlabel('iLQR iterations $N$')
    ax2.set_ylabel('total cost')
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig('ilqr_sim2_total.pdf', bbox_inches='tight')
    plt.close(fig2)

    print("\nSaved ilqr_sim2_stage.pdf and ilqr_sim2_total.pdf")
