"""
Closed-loop simulation of iLQR on the inverted pendulum.

Dynamics (theta=0 is unstable upright equilibrium):
  theta_{t+1}     = theta_t + dt * theta_dot_t
  theta_dot_{t+1} = theta_dot_t + dt * (g/L * sin(theta_t) + u_t / (m*L^2))

iLQR policy at state x_k (repeated each closed-loop step, warm-start off):
  Initialize u_bar = 0.
  For each iteration:
    1. Forward pass:  x_bar[t+1] = f(x_bar[t], u_bar[t])   (nonlinear)
    2. Linearize:     A[t] = df/dx|(x_bar[t])                (B is constant)
    3. Solve LQR:     time-varying LQR on linearized system from x_k
  Set u_bar = u_lin from LQR.
  Apply u_k = u_bar[0].
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

def step(x, u, dt, g, L, mass):
    theta, tdot = x
    tddot = g / L * np.sin(theta) + u / (mass * L**2)
    return np.array([theta + dt * tdot, tdot + dt * tddot])


# ---------------------------------------------------------------------------
# iLQR
# ---------------------------------------------------------------------------

def ilqr_control(x0, n_iters, T, dt, Q, r, g, L, mass, u_max):
    """
    Run n_iters of iLQR from x0 with planning horizon T.
    Returns first control u[0].

    Cartpole B matrix is constant: B = [[0], [dt/(m*L^2)]].
    A[t] = [[1, dt], [dt*g/L*cos(theta_bar[t]), 1]].
    """
    b_u = dt / (mass * L**2)
    B = np.array([0.0, b_u])          # shape (2,) — B col-vector flattened

    u_bar = np.zeros(T)

    for _ in range(n_iters):
        # 1. Forward pass
        x_bar = np.zeros((T + 1, 2))
        x_bar[0] = x0
        for t in range(T):
            x_bar[t + 1] = step(x_bar[t], u_bar[t], dt, g, L, mass)

        # 2. Linearize: A[t] depends only on theta_bar[t] (B is constant)
        cos_bar = np.cos(x_bar[:T, 0])
        A_list = [np.array([[1.0, dt], [dt * g / L * c, 1.0]]) for c in cos_bar]

        # 3. Backward Riccati sweep
        P = Q.copy()            # P[T] = Q_terminal
        K_list = [None] * T
        for t in range(T - 1, -1, -1):
            A = A_list[t]
            BtP  = B @ P            # (2,):  B^T P
            S    = r + float(BtP @ B)  # scalar: R + B^T P B
            K    = (BtP @ A) / S    # (2,):  K = S^{-1} B^T P A  (gain row)
            P    = Q + A.T @ P @ (A - np.outer(B, K))   # Riccati update
            K_list[t] = K

        # Forward LQR from x0 on linearized dynamics
        x_lin = np.zeros((T + 1, 2))
        u_lin = np.zeros(T)
        x_lin[0] = x0
        for t in range(T):
            u_t = float(np.clip(-K_list[t] @ x_lin[t], -u_max, u_max))
            u_lin[t] = u_t
            x_lin[t + 1] = A_list[t] @ x_lin[t] + B * u_t

        u_bar = u_lin

    return float(u_bar[0])


# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------

def simulate(x0, N_sim, T_plan, dt, n_iters, Q, r, g, L, mass, u_max):
    x = x0.copy()
    stage_costs = np.zeros(N_sim)
    states = np.zeros((N_sim + 1, 2))
    controls = np.zeros(N_sim)
    states[0] = x

    for k in range(N_sim):
        u = ilqr_control(x, n_iters, T_plan, dt, Q, r, g, L, mass, u_max)
        stage_costs[k] = float(x @ Q @ x) + r * u ** 2
        controls[k] = u
        x = step(x, u, dt, g, L, mass)
        states[k + 1] = x

    return stage_costs, states, controls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # System
    g, L, mass = 9.8, 1.0, 1.0
    dt = 0.01
    Q = np.eye(2)
    r = 0.00
    u_max = 1e3

    # Simulation
    x0 = np.array([2., 2.0])   # 1.5 rad from upright, zero angular velocity
    N_sim = 100
    T_plan = 10                  # iLQR planning horizon (1 second lookahead)
    n_iters_list = [1, 2, 3, 4, 5]

    FONT_SIZE = 22
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
    })
    plt.rc("xtick", labelsize=FONT_SIZE, labelcolor="black")
    plt.rc("ytick", labelsize=FONT_SIZE, labelcolor="black")

    cmap = plt.cm.Set1
    colors = cmap.colors

    all_stage, all_states, all_controls = {}, {}, {}
    for n in n_iters_list:
        sc, st, uc = simulate(x0, N_sim, T_plan, dt, n, Q, r, g, L, mass, u_max)
        all_stage[n] = sc
        all_states[n] = st
        all_controls[n] = uc
        print(f"n_iters={n}: total cost = {np.sum(sc):.4f}, "
              f"final theta = {st[-1, 0]:.4f} rad")

    steps = np.arange(N_sim)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Stage cost (log scale)
    for i, n in enumerate(n_iters_list):
        axes[0].semilogy(steps * dt, all_stage[n],
                         label=f'$N={n}$', color=colors[i], linewidth=2)
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('stage cost')
    axes[0].legend()
    axes[0].grid(True)

    # Cumulative cost
    for i, n in enumerate(n_iters_list):
        axes[1].plot(steps * dt, np.cumsum(all_stage[n]),
                     label=f'$N={n}$', color=colors[i], linewidth=2)
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('cumulative cost')
    axes[1].legend()
    axes[1].grid(True)

    # Angle trajectory
    time_full = np.arange(N_sim + 1) * dt
    for i, n in enumerate(n_iters_list):
        axes[2].plot(time_full, all_states[n][:, 0],
                     label=f'$N={n}$', color=colors[i], linewidth=2)
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel(r'$\theta$ (rad)')
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig('ilqr_closed_loop.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved ilqr_closed_loop.pdf")
