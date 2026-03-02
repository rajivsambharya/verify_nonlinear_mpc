#!/usr/bin/env python
"""Full run of iLQR verification mimicking the actual experiment."""

import sys
sys.path.insert(0, '/Users/rajivsambharya/Documents/work/verify_layers')

from benchmarks.cartpole_ilqr import CartpoleILQRVerify
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("Running iLQR verification experiment (K=1 to 5)...")

x_min = -0.001
x_max = 0.001
rho_max = 5.0
K_max = 5

opt_vals = np.zeros(K_max)
times = np.zeros(K_max)

for k in range(K_max):
    print(f"\n{'='*60}")
    print(f"K = {k+1}")
    print(f"{'='*60}")

    rho_lo = 0.0
    rho_hi = rho_max
    best_rho = None
    best_time = None

    iteration = 0
    max_iterations = 20

    while rho_hi - rho_lo > 1e-2 and iteration < max_iterations:
        rho_mid = (rho_lo + rho_hi) / 2.0
        iteration += 1

        ver = CartpoleILQRVerify(
            n=10, K=k+1, T=5, r=0.01, dt=0.1,
            mass=1, length=1, g=9.8, rho=rho_mid,
            x_lo=x_min, x_hi=x_max, seed=42, verbose=False
        )

        status, solve_time = ver.solve()

        print(f"  k={k+1}, rho={rho_mid:.6f}, status={status}")

        if status == GRB.OPTIMAL:
            rho_lo = rho_mid
        else:
            rho_hi = rho_mid
            best_rho = rho_mid
            best_time = solve_time

    if best_rho is not None:
        print(f"  ✓ Best verified rho: {best_rho:.6f} (time: {best_time:.3f}s)")
        opt_vals[k] = rho_hi
        times[k] = best_time
    else:
        print(f"  ✗ Could not verify any rho <= {rho_max}")
        opt_vals[k] = np.inf
        times[k] = 0

# Plot results
plt.rcParams.update({'text.usetex': False})  # Disable LaTeX to avoid unicode issues
plt.figure(figsize=(8, 6))
plt.plot(np.arange(K_max) + 1, opt_vals, 'o-', linewidth=2, markersize=8)
plt.xlabel('K (iterations)')
plt.ylabel('Verified contraction rate rho')
plt.title('iLQR Stability Verification')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rates_ilqr_test.pdf', bbox_inches='tight')
print(f"\n✓ Saved plot to rates_ilqr_test.pdf")

print(f"\n{'='*60}")
print("Summary:")
print(f"{'='*60}")
for k in range(K_max):
    if opt_vals[k] == np.inf:
        print(f"  K={k+1}: Could not verify (ρ > {rho_max})")
    else:
        print(f"  K={k+1}: ρ = {opt_vals[k]:.6f}, time = {times[k]:.3f}s")
