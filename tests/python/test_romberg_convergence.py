#!/usr/bin/env python3
"""
Debug script to show convergence history at each Romberg refinement level.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp
import numpy as np

def test_convergence_history():
    """Show phase velocity at each refinement level."""
    
    # Use the linear gradient model
    depths = [0.0, 10.0, 20.0, 30.0]
    params = []
    for z in depths:
        vs = 2.5 + 0.05 * z
        vp = 4.3 + 0.08 * z
        rho = 2.4 + 0.01 * z
        params.append(dp.create_isotropic_parameters(vs, vp, rho))
    
    is_discontinuity = [False, False, False, False]
    period = 20.0
    c_guess = 3.0
    
    print("=" * 70)
    print("Romberg Convergence History")
    print("=" * 70)
    print(f"{'Level':>5} {'Subdivisions':>12} {'Phase Vel':>14} {'Rel. Error':>12}")
    print("-" * 70)
    
    prev_c = None
    for max_level in range(6):
        result = dp.compute_dispersion_romberg(
            depths=depths,
            params=params,
            is_discontinuity=is_discontinuity,
            period=period,
            wave_type="Rayleigh",
            c_guess=c_guess,
            max_steps=max_level,
            tolerance=1e-15  # Very tight to force all levels
        )
        c, U, HV, level, rel_err = result
        subdivisions = 2 ** max_level
        
        if prev_c is not None:
            diff = abs(c - prev_c) / c
            print(f"{max_level:>5} {subdivisions:>12} {c:>14.10f} {diff:>12.2e}")
        else:
            print(f"{max_level:>5} {subdivisions:>12} {c:>14.10f}       ---")
        
        prev_c = c
    
    print("=" * 70)
    print("\nInterpretation:")
    print("  1e-10未満になったレベルで十分な精度に達しています")
    print("  Early stopping would be optimal at the level where rel_error < 1e-10")


if __name__ == "__main__":
    test_convergence_history()
