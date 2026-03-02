#!/usr/bin/env python
"""Test binary search for multiple K values in iLQR verification."""

import sys
sys.path.insert(0, '/Users/rajivsambharya/Documents/work/verify_layers')

from benchmarks.cartpole_ilqr import CartpoleILQRVerify
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Test with K=1, 2, 3
print("Testing CartpoleILQRVerify for multiple K values...")

x_min = -0.001
x_max = 0.001
rho_max = 5.0

K_values = [1, 2, 3]
results = {}

for K in K_values:
    print(f"\n{'='*60}")
    print(f"Testing K = {K}")
    print(f"{'='*60}")

    rho_lo = 0.0
    rho_hi = rho_max
    best_rho = None

    iteration = 0
    max_iterations = 15

    while rho_hi - rho_lo > 1e-2 and iteration < max_iterations:
        rho_mid = (rho_lo + rho_hi) / 2.0
        iteration += 1

        print(f"  Iter {iteration}: rho = {rho_mid:.6f}... ", end='', flush=True)

        try:
            ver = CartpoleILQRVerify(
                n=10, K=K, T=5, r=0.01, dt=0.1,
                mass=1, length=1, g=9.8, rho=rho_mid,
                x_lo=x_min, x_hi=x_max, seed=42, verbose=False
            )

            status, solve_time = ver.solve()

            if status == GRB.OPTIMAL:
                print(f"OPTIMAL (need larger rho)")
                rho_lo = rho_mid
            else:
                print(f"INFEASIBLE (verified!)")
                rho_hi = rho_mid
                best_rho = rho_mid

        except Exception as e:
            print(f"ERROR: {e}")
            break

    if best_rho is not None:
        print(f"\n  ✓ Best verified rho for K={K}: {best_rho:.6f}")
        results[K] = best_rho
    else:
        print(f"\n  ✗ Could not verify any rho <= {rho_max} for K={K}")
        results[K] = np.inf

print(f"\n{'='*60}")
print("Summary:")
print(f"{'='*60}")
for K in K_values:
    rho = results[K]
    if rho == np.inf:
        print(f"  K={K}: Could not verify (rho > {rho_max})")
    else:
        print(f"  K={K}: rho = {rho:.6f}")
