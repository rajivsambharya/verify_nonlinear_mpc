#!/usr/bin/env python
"""Quick test script for cartpole_ilqr without hydra."""

import sys
sys.path.insert(0, '/Users/rajivsambharya/Documents/work/verify_layers')

from benchmarks.cartpole_ilqr import CartpoleILQRVerify
import gurobipy as gp
from gurobipy import GRB

# Test with K=1, small problem
print("Testing CartpoleILQRVerify with K=1...")

try:
    ver = CartpoleILQRVerify(
        n=10, K=1, T=5, r=0.01, dt=0.1,
        mass=1, length=1, g=9.8, rho=0.5,
        x_lo=-0.001, x_hi=0.001, seed=42, verbose=True
    )

    print("Model created successfully!")
    print(f"Number of variables: {ver.model.NumVars}")
    print(f"Number of constraints: {ver.model.NumConstrs}")
    print(f"Number of general constraints: {ver.model.NumGenConstrs}")

    print("\nSolving...")
    status, solve_time = ver.solve()

    print(f"Status: {status}")
    print(f"Solve time: {solve_time:.4f}s")

    if status == GRB.INFEASIBLE:
        print("Model is INFEASIBLE - verification succeeded (no counterexample found)")
    elif status == GRB.OPTIMAL:
        sol = ver.solution_dict()
        print(f"Model is OPTIMAL - found counterexample with objective: {sol['obj']}")
        print("Verification failed at this rho")
    else:
        print(f"Unexpected status: {status}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
