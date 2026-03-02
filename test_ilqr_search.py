#!/usr/bin/env python
"""Test binary search for rho in iLQR verification."""

import sys
sys.path.insert(0, '/Users/rajivsambharya/Documents/work/verify_layers')

from benchmarks.cartpole_ilqr import CartpoleILQRVerify
import gurobipy as gp
from gurobipy import GRB

# Test with K=1, small problem
print("Testing CartpoleILQRVerify binary search for K=1...")

x_min = -0.001
x_max = 0.001
rho_max = 5.0

rho_lo = 0.0
rho_hi = rho_max
best_rho = None

iteration = 0
while rho_hi - rho_lo > 1e-2 and iteration < 10:
    rho_mid = (rho_lo + rho_hi) / 2.0
    iteration += 1

    print(f"\n{'='*60}")
    print(f"Iteration {iteration}: Testing rho = {rho_mid:.6f}")
    print(f"{'='*60}")

    try:
        ver = CartpoleILQRVerify(
            n=10, K=1, T=5, r=0.01, dt=0.1,
            mass=1, length=1, g=9.8, rho=rho_mid,
            x_lo=x_min, x_hi=x_max, seed=42, verbose=False
        )

        status, solve_time = ver.solve()

        print(f"Status: {status}, Time: {solve_time:.4f}s")

        if status == GRB.OPTIMAL:
            sol = ver.solution_dict()
            print(f"OPTIMAL (counterexample found) - obj: {sol['obj']:.6e}")
            print("Need larger rho")
            rho_lo = rho_mid
        else:
            print(f"INFEASIBLE or other status - verification succeeded!")
            print("Can try smaller rho")
            rho_hi = rho_mid
            best_rho = rho_mid

    except Exception as e:
        print(f"Error: {e}")
        break

print(f"\n{'='*60}")
if best_rho is not None:
    print(f"Best verified rho for K=1: {best_rho:.6f}")
else:
    print(f"Could not verify any rho <= {rho_max}")
print(f"{'='*60}")
