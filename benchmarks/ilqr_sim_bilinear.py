"""
Closed-loop iLQR simulation on the bilinear system from bilinear_closed_loop2:

  x+[0] = (a00 + b*u) * x[0] + u
  x+[1] = a11 * x[1] + a10 * x[0]

Linearization at (x_bar[t], u_bar[t]):
  A[t] = [[a00 + b*u_bar[t],  0  ],   <- A[0,0] depends on u_bar (updates each iter)
           [a10,               a11]]
  B[t] = [[1 + b*x_bar[t][0],        <- B[0] depends on x_bar (updates each iter)
           [0                ]]

Why iLQR iterations matter here:
  Iter 0 (u_bar=0): A[t][0,0] = a00 (constant, ignores bilinear coupling).
    B[t][0] = 1 + b*x_bar_0[t][0] correctly reflects the input gain,
    but A is wrong because the true closed-loop A depends on u.
  Iter 1 (u_bar=u_lin_0): A[t][0,0] = a00 + b*u_lin_0[t], which can differ
    significantly from a00 when b is large, giving a much better-calibrated
    LQR and substantially lower closed-loop cost.

Note on original yaml params (r=1e-3, b=0.2, x0 in [2,3]):
  With very small r, the optimal control saturates at ±u_bound for all x,
  so the LQR converges in 1 iteration regardless. The parameters below
  (b=2.0, r=0.5) keep the same system structure but make the bilinear
  coupling strong enough, and the control cost large enough, that the
  solution is interior and the linearization quality matters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
A00     = 0.9
A10     = 0.85
A11     = 0.9
B_COEF  = 2.0
R       = 0.5
U_BOUND = 1.0
Q       = np.eye(2)


# ---------------------------------------------------------------------------
# Dynamics and Jacobians
# ---------------------------------------------------------------------------

def step(x, u):
    u = np.clip(u, -U_BOUND, U_BOUND)
    return np.array([(A00 + B_COEF * u) * x[0] + u,
                     A11 * x[1] + A10 * x[0]])


def jac_A(u_bar):
    """df/dx at nominal (x_bar, u_bar) — depends on u_bar."""
    return np.array([[A00 + B_COEF * u_bar, 0.0],
                     [A10,                   A11]])


def jac_B(x_bar):
    """df/du at nominal (x_bar, u_bar) — depends on x_bar[0]."""
    return np.array([1.0 + B_COEF * x_bar[0], 0.0])


# ---------------------------------------------------------------------------
# Standard iLQR (feedforward + feedback)
# ---------------------------------------------------------------------------

def ilqr_control(x0, n_iters, T):
    """
    Standard iLQR from x0 with horizon T.  Returns first control u[0].

    Each iteration:
      1. Forward pass with u_bar → x_bar
      2. Backward pass: second-order expansion gives feedforward k_t, feedback K_t
      3. Rollout: u_t ← u_bar_t + k_t + K_t @ (x_t - x_bar_t)

    A[t] = jac_A(u_bar[t]) and B[t] = jac_B(x_bar[t]) both update each iter.
    """
    x0    = np.asarray(x0, dtype=float)
    u_bar = np.zeros(T)

    for _ in range(n_iters):
        # 1. Forward pass
        x_bar = [x0.copy()]
        for t in range(T):
            x_bar.append(step(x_bar[-1], u_bar[t]))

        As = [jac_A(u_bar[t]) for t in range(T)]
        Bs = [jac_B(x_bar[t]) for t in range(T)]

        # 2. Backward pass
        V_x  = 2.0 * Q @ x_bar[T]
        V_xx = 2.0 * Q

        ks = []
        Ks = []
        for t in range(T - 1, -1, -1):
            A  = As[t]
            Bv = Bs[t]

            Q_u  = 2.0 * R * u_bar[t] + float(Bv @ V_x)
            Q_uu = 2.0 * R             + float(Bv @ V_xx @ Bv)
            Q_ux = Bv @ V_xx @ A
            Q_x  = 2.0 * Q @ x_bar[t] + A.T @ V_x
            Q_xx = 2.0 * Q             + A.T @ V_xx @ A

            k = -Q_u  / Q_uu
            K = -Q_ux / Q_uu

            V_x  = Q_x  + K * Q_u
            V_xx = Q_xx + np.outer(K, Q_ux)

            ks.insert(0, k)
            Ks.insert(0, K)

        # 3. Forward rollout
        x     = x0.copy()
        u_new = np.zeros(T)
        for t in range(T):
            u_new[t] = float(np.clip(u_bar[t] + ks[t] + Ks[t] @ (x - x_bar[t]),
                                     -U_BOUND, U_BOUND))
            x = step(x, u_new[t])

        u_bar = u_new

    return float(u_bar[0])


# ---------------------------------------------------------------------------
# KKT — true optimal MPC via direct nonlinear optimisation
# ---------------------------------------------------------------------------

def kkt_control(x0, T):
    """Multi-start L-BFGS-B; warm-start from 3-iter iLQR."""
    x0 = np.asarray(x0, dtype=float)

    def cost(u_seq):
        x = x0.copy()
        c = 0.0
        for u in u_seq:
            c += float(x @ Q @ x) + R * u**2
            x = step(x, u)
        return c + float(x @ Q @ x)

    starts = [np.zeros(T),
              np.full(T, -U_BOUND),
              np.full(T,  U_BOUND),
              np.array([ilqr_control(x0, 3, T)] + [0.0] * (T - 1))]

    best_val, best_u = np.inf, 0.0
    for u0 in starts:
        res = minimize(cost, u0, method='L-BFGS-B',
                       bounds=[(-U_BOUND, U_BOUND)] * T,
                       options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 3000})
        if res.fun < best_val:
            best_val, best_u = res.fun, float(res.x[0])

    return best_u


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
    x0_vals      = [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]
    n_iters_list = [1, 2, 3, 4, 5]
    N_sim        = 20
    T_plan       = 5

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
    markers     = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    # -----------------------------------------------------------------------
    # Figure 1: stage cost trajectories
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(x0_vals), figsize=(5 * len(x0_vals), 5))
    steps_ax  = np.arange(N_sim)

    for ax, x0 in zip(axes, x0_vals):
        print(f"\nx0 = {x0}")
        for i, n in enumerate(n_iters_list):
            sc, _ = simulate(x0, N_sim, T_plan, n_iters=n)
            print(f"  n_iters={n}: total={np.sum(sc):.4f}")
            ax.semilogy(steps_ax, sc, label=f'$N={n}$',
                        color=iter_colors[i], linewidth=2)
        sc_kkt, _ = simulate(x0, N_sim, T_plan, use_kkt=True)
        print(f"  KKT:      total={np.sum(sc_kkt):.4f}")
        ax.semilogy(steps_ax, sc_kkt, label='KKT',
                    color='k', linewidth=2.5, linestyle='--')
        ax.set_title(f'$x_0 = {x0}$')
        ax.set_xlabel('step $k$')
        ax.set_ylabel('stage cost $\\|x\\|^2 + r u^2$')
        ax.legend(fontsize=14)
        ax.grid(True)

    fig.suptitle(
        rf'Bilinear: $x^+_1=(a_{{00}}+bu)x_1+u$, $x^+_2=a_{{11}}x_2+a_{{10}}x_1$'
        rf', $b={B_COEF}$, $r={R}$',
        fontsize=FONT_SIZE - 4
    )
    fig.tight_layout()
    fig.savefig('ilqr_sim_bilinear_stage.pdf', bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 2: total cost vs n_iters
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for j, x0 in enumerate(x0_vals):
        totals    = [np.sum(simulate(x0, N_sim, T_plan, n_iters=n)[0])
                     for n in n_iters_list]
        kkt_total = np.sum(simulate(x0, N_sim, T_plan, use_kkt=True)[0])
        color     = x0_colors[j % len(x0_colors)]
        ax2.plot(n_iters_list, totals,
                 marker=markers[j % len(markers)], linewidth=2, color=color,
                 label=f'$x_0={x0}$')
        ax2.axhline(kkt_total, color=color, linewidth=1.5, linestyle='--')

    ax2.set_xlabel('iLQR iterations $N$')
    ax2.set_ylabel('total cost')
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig('ilqr_sim_bilinear_total.pdf', bbox_inches='tight')
    plt.close(fig2)

    print("\nSaved ilqr_sim_bilinear_stage.pdf  ilqr_sim_bilinear_total.pdf")
