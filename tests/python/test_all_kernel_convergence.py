#!/usr/bin/env python3
"""
Verify O(h^2) convergence of dU/dp and dHV/dp kernels.
"""

import disper26py as dp
import numpy as np


def test_all_kernel_convergence():
    """Test dc/dp, dU/dp, dHV/dp convergence with refinement."""
    
    depths = [0.0, 10.0, 20.0, 30.0]
    params = []
    for z in depths:
        vs = 2.5 + 0.05 * z
        vp = 4.3 + 0.08 * z
        rho = 2.4 + 0.01 * z
        params.append(dp.create_isotropic_parameters(vs, vp, rho))
    
    is_discontinuity = [False, False, False, False]
    period = 20.0
    
    print("=" * 80)
    print("Convergence Test: dc/dp, dU/dp, dHV/dp (Node 0, Vs)")
    print("=" * 80)
    
    # Collect kernels at each level
    dc_history = []
    dU_history = []
    dHV_history = []
    
    for max_level in range(1, 6):
        result = dp.compute_dispersion_romberg_with_kernels(
            depths=depths,
            params=params,
            is_discontinuity=is_discontinuity,
            period=period,
            wave_type="Rayleigh",
            c_guess=3.0,
            sensitivity_mode="Isotropic",
            max_steps=max_level,
            tolerance=1e-15  # Force to run max_level
        )
        
        c, U, HV, level, rel_err, kernels = result
        
        # Find Vs kernel for node 0
        for k in kernels:
            if k.parameter_name == "Vs" and k.layer_index == 0:
                dc_history.append((max_level, k.d_c))
                dU_history.append((max_level, k.d_U))
                dHV_history.append((max_level, k.d_HV))
                break
    
    # Print results
    print(f"{'Level':>5} {'dc/dVs':>14} {'dU/dVs':>14} {'dHV/dVs':>14}")
    print("-" * 60)
    for i, (level, dc) in enumerate(dc_history):
        dU = dU_history[i][1]
        dHV = dHV_history[i][1]
        print(f"{level:>5} {dc:>14.8f} {dU:>14.8f} {dHV:>14.8f}")
    
    print()
    
    # Compute convergence ratios
    def compute_ratios(history, name):
        values = [h[1] for h in history]
        diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        
        print(f"{name} convergence (ratio ≈ 4 for O(h²)):")
        for i in range(1, len(diffs)):
            if diffs[i] > 1e-15:
                ratio = diffs[i-1] / diffs[i]
                print(f"  Level {i+1} → {i+2}: ratio = {ratio:.2f}")
            else:
                print(f"  Level {i+1} → {i+2}: Δ too small ({diffs[i]:.2e})")
        print()
    
    compute_ratios(dc_history, "dc/dp")
    compute_ratios(dU_history, "dU/dp")
    compute_ratios(dHV_history, "dHV/dp")


if __name__ == "__main__":
    test_all_kernel_convergence()
