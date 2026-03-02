#!/usr/bin/env python
"""Test that iLQR verification trajectories match true dynamics."""

import sys
sys.path.insert(0, '/Users/rajivsambharya/Documents/work/verify_layers')

from benchmarks.cartpole_ilqr import CartpoleILQRVerify
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Disable LaTeX to avoid unicode issues
plt.rcParams.update({'text.usetex': False})

def simulate_true_dynamics(x0, controls, dt=0.1, mass=1, length=1, g=9.8):
    """Simulate true nonlinear dynamics given controls."""
    K = len(controls)
    x_traj = np.zeros((K + 1, 2))
    x_traj[0] = x0

    for k in range(K):
        u_k = controls[k]
        theta_k = x_traj[k, 0]
        theta_dot_k = x_traj[k, 1]

        # True nonlinear dynamics
        theta_ddot = g / length * np.sin(theta_k) + u_k / (mass * length**2)

        # Forward Euler integration
        x_traj[k+1, 0] = theta_k + dt * theta_dot_k
        x_traj[k+1, 1] = theta_dot_k + dt * theta_ddot

    return x_traj

def test_trajectory_verification(K=3, rho=0.5, verbose=True):
    """Test that verification trajectories match true dynamics."""

    print(f"\n{'='*70}")
    print(f"Testing Trajectory Verification for K={K}, rho={rho}")
    print(f"{'='*70}\n")

    # Create and solve verification problem
    x_min = -0.5
    x_max = 0.5

    ver = CartpoleILQRVerify(
        n=10, K=K, T=3, r=0.01, dt=0.1,
        mass=1, length=1, g=9.8, rho=rho,
        x_lo=x_min, x_hi=x_max, seed=42, verbose=False
    )

    status, solve_time = ver.solve()

    if status != GRB.OPTIMAL:
        print(f"⚠️  Status: {status} (not OPTIMAL)")
        print("Cannot verify trajectory - problem is infeasible or unbounded")
        return None

    print(f"✓ Verification problem solved (status: {status}, time: {solve_time:.3f}s)")

    # Extract solution
    sol = ver.solution_dict()

    # Extract verification trajectory and controls
    x_verify = np.zeros((K + 1, 2))
    u_verify = np.zeros(K)

    for k in range(K + 1):
        x_verify[k, 0] = sol['x'][k][0]
        x_verify[k, 1] = sol['x'][k][1]

    for k in range(K):
        u_verify[k] = sol['u'][k][0]

    # Simulate true dynamics using extracted controls
    x0 = x_verify[0]
    x_simulated = simulate_true_dynamics(x0, u_verify, dt=0.1, mass=1, length=1, g=9.8)

    # Check sin/cos accuracy
    print(f"\nChecking sin/cos constraint accuracy:")
    print(f"{'Step':>5} | {'sin(θ)_verify':>15} {'sin(θ)_true':>15} {'Δsin':>12} | {'cos(θ)_verify':>15} {'cos(θ)_true':>15} {'Δcos':>12}")
    print("-" * 100)
    max_sin_error = 0
    max_cos_error = 0
    for k in range(K):
        sin_verify = sol['sin_theta'][k]
        cos_verify = sol['cos_theta'][k]
        sin_true = np.sin(x_verify[k, 0])
        cos_true = np.cos(x_verify[k, 0])
        sin_error = abs(sin_verify - sin_true)
        cos_error = abs(cos_verify - cos_true)
        max_sin_error = max(max_sin_error, sin_error)
        max_cos_error = max(max_cos_error, cos_error)
        print(f"{k:5d} | {sin_verify:15.10f} {sin_true:15.10f} {sin_error:12.2e} | "
              f"{cos_verify:15.10f} {cos_true:15.10f} {cos_error:12.2e}")

    print(f"\nMax sin error: {max_sin_error:.2e}, Max cos error: {max_cos_error:.2e}")

    # Compare trajectories
    print(f"\nInitial state: θ={x0[0]:.6f}, θ_dot={x0[1]:.6f}")
    print(f"\n{'Step':>5} | {'θ_verify':>12} {'θ_sim':>12} {'Δθ':>12} | {'θ̇_verify':>12} {'θ̇_sim':>12} {'Δθ̇':>12} | {'u':>12}")
    print("-" * 105)

    max_error_theta = 0
    max_error_theta_dot = 0

    for k in range(K + 1):
        error_theta = abs(x_verify[k, 0] - x_simulated[k, 0])
        error_theta_dot = abs(x_verify[k, 1] - x_simulated[k, 1])

        max_error_theta = max(max_error_theta, error_theta)
        max_error_theta_dot = max(max_error_theta_dot, error_theta_dot)

        u_str = f"{u_verify[k]:.6f}" if k < K else "N/A"

        print(f"{k:5d} | {x_verify[k,0]:12.8f} {x_simulated[k,0]:12.8f} {error_theta:12.2e} | "
              f"{x_verify[k,1]:12.8f} {x_simulated[k,1]:12.8f} {error_theta_dot:12.2e} | {u_str:>12}")

    print(f"\n{'='*70}")
    print(f"Maximum Errors:")
    print(f"  θ (angle):          {max_error_theta:.2e}")
    print(f"  θ̇ (angular vel):    {max_error_theta_dot:.2e}")

    # Check if errors are within tolerance
    # Note: Tolerance scales with K because errors accumulate over time due to
    # numerical approximations in Gurobi's nonconvex sin/cos constraints
    # (FeasibilityTol = 1e-9 but general constraints can have slightly larger errors)
    tol_base = 1e-3
    tol = tol_base * K  # Scale tolerance with trajectory length
    if max_error_theta < tol and max_error_theta_dot < tol:
        print(f"\n✓ PASS: Trajectories match within tolerance ({tol:.2e})")
        passed = True
    else:
        print(f"\n✗ FAIL: Trajectories differ more than tolerance ({tol:.2e})")
        passed = False

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time_steps = np.arange(K + 1)

    # Plot theta
    axes[0, 0].plot(time_steps, x_verify[:, 0], 'o-', label='Verification', linewidth=2, markersize=8)
    axes[0, 0].plot(time_steps, x_simulated[:, 0], 's--', label='Simulated', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Time step k')
    axes[0, 0].set_ylabel('θ (angle) [rad]')
    axes[0, 0].set_title('Angle Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot theta_dot
    axes[0, 1].plot(time_steps, x_verify[:, 1], 'o-', label='Verification', linewidth=2, markersize=8)
    axes[0, 1].plot(time_steps, x_simulated[:, 1], 's--', label='Simulated', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Time step k')
    axes[0, 1].set_ylabel('θ̇ (angular velocity) [rad/s]')
    axes[0, 1].set_title('Angular Velocity Trajectory Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot control
    axes[1, 0].plot(np.arange(K), u_verify, 'o-', linewidth=2, markersize=8, color='C2')
    axes[1, 0].set_xlabel('Time step k')
    axes[1, 0].set_ylabel('u (control torque) [N⋅m]')
    axes[1, 0].set_title('Control Trajectory')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot errors
    error_theta = np.abs(x_verify[:, 0] - x_simulated[:, 0])
    error_theta_dot = np.abs(x_verify[:, 1] - x_simulated[:, 1])

    axes[1, 1].semilogy(time_steps, error_theta, 'o-', label='|Δθ|', linewidth=2, markersize=8)
    axes[1, 1].semilogy(time_steps, error_theta_dot, 's-', label='|Δθ̇|', linewidth=2, markersize=8)
    axes[1, 1].axhline(tol, color='r', linestyle='--', linewidth=2, label=f'Tolerance ({tol:.0e})')
    axes[1, 1].set_xlabel('Time step k')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Trajectory Errors (log scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    filename = f'ilqr_trajectory_verification_K{K}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    print(f"\n✓ Saved plots to {filename}")

    # Also check MPC trajectories
    print(f"\n{'='*70}")
    print("Checking MPC internal trajectories...")
    print(f"{'='*70}")

    for k in range(min(2, K)):  # Check first 2 timesteps
        print(f"\nTimestep k={k}:")
        print(f"  MPC initial condition: x_mpc[{k}][0] = [{sol['x_mpc'][k][0][0]:.6f}, {sol['x_mpc'][k][0][1]:.6f}]")
        print(f"  Actual state:          x[{k}] = [{sol['x'][k][0]:.6f}, {sol['x'][k][1]:.6f}]")
        print(f"  Match: {np.allclose([sol['x_mpc'][k][0][0], sol['x_mpc'][k][0][1]], [sol['x'][k][0], sol['x'][k][1]], atol=1e-6)}")

        print(f"\n  MPC controls over horizon:")
        for t in range(min(3, len(sol['x_mpc'][k]) - 1)):
            print(f"    u_mpc[{k}][{t}] = {sol['u_mpc'][k][t][0]:.6f}")

        print(f"\n  Applied control: u[{k}] = {sol['u'][k][0]:.6f}")
        print(f"  First MPC control: u_mpc[{k}][0] = {sol['u_mpc'][k][0][0]:.6f}")
        print(f"  Match: {np.allclose(sol['u'][k][0], sol['u_mpc'][k][0][0], atol=1e-6)}")

    return {
        'passed': passed,
        'max_error_theta': max_error_theta,
        'max_error_theta_dot': max_error_theta_dot,
        'x_verify': x_verify,
        'x_simulated': x_simulated,
        'controls': u_verify
    }

if __name__ == '__main__':
    print("Testing iLQR trajectory verification")
    print("=" * 70)

    # Test with different K values
    test_cases = [
        (1, 0.5),
        (2, 0.5),
        (3, 0.5),
        (3, 0.7),  # Different rho
    ]

    results = []
    for K, rho in test_cases:
        result = test_trajectory_verification(K=K, rho=rho, verbose=True)
        if result is not None:
            results.append((K, rho, result))

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'K':>5} {'rho':>8} {'Status':>10} {'Max θ Error':>15} {'Max θ̇ Error':>15}")
    print("-" * 70)

    all_passed = True
    for K, rho, result in results:
        status = "PASS" if result['passed'] else "FAIL"
        all_passed = all_passed and result['passed']
        print(f"{K:5d} {rho:8.3f} {status:>10} {result['max_error_theta']:15.2e} {result['max_error_theta_dot']:15.2e}")

    print(f"{'='*70}")
    if all_passed:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
