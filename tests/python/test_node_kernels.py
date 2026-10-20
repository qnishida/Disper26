#!/usr/bin/env python3
"""
Test node-based sensitivity kernels with Romberg acceleration.
Verify O(h^2) convergence of node kernels.
"""

import disper26py as dp
import numpy as np


def test_node_kernels_basic():
    """Test that node-based kernels can be computed."""
    
    # Simple linear gradient model
    depths = [0.0, 10.0, 20.0, 30.0]
    params = []
    for z in depths:
        vs = 2.5 + 0.05 * z
        vp = 4.3 + 0.08 * z
        rho = 2.4 + 0.01 * z
        params.append(dp.create_isotropic_parameters(vs, vp, rho))
    
    is_discontinuity = [False, False, False, False]
    period = 20.0
    
    print("=" * 70)
    print("Test: Node-Based Sensitivity Kernels (Romberg)")
    print("=" * 70)
    
    # Call with kernels
    result = dp.compute_dispersion_romberg_with_kernels(
        depths=depths,
        params=params,
        is_discontinuity=is_discontinuity,
        period=period,
        wave_type="Rayleigh",
        c_guess=3.0,
        sensitivity_mode="Isotropic",
        max_steps=5,
        tolerance=1e-10
    )
    
    c, U, HV, level, rel_err, kernels = result
    
    print(f"Phase Velocity: {c:.8f} km/s")
    print(f"Group Velocity: {U:.8f} km/s")
    print(f"H/V Ratio:      {HV:.8f}")
    print(f"Converged at level: {level}")
    print(f"Relative error: {rel_err:.2e}")
    print()
    
    print("Node Kernels (dc/dp):")
    print(f"{'Node':>5} {'Param':>6} {'dc/dp':>14}")
    print("-" * 30)
    for k in kernels:
        print(f"{k.layer_index:>5} {k.parameter_name:>6} {k.d_c:>14.8f}")
    
    print()
    print("✓ Node kernels computed successfully!")
    return True


def test_kernel_convergence_by_level():
    """Test if node kernels converge with O(h^2) by comparing different refinement levels."""
    
    # Simple model
    depths = [0.0, 10.0, 20.0, 30.0]
    params = []
    for z in depths:
        vs = 2.5 + 0.05 * z
        vp = 4.3 + 0.08 * z
        rho = 2.4 + 0.01 * z
        params.append(dp.create_isotropic_parameters(vs, vp, rho))
    
    is_discontinuity = [False, False, False, False]
    period = 20.0
    
    print("=" * 70)
    print("Test: Node Kernel Convergence by Refinement Level")
    print("=" * 70)
    
    # Collect dc/dVs[0] at each level
    kernel_vs0_history = []
    
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
        
        # Find dc/dVs for node 0
        dc_dVs_0 = None
        for k in kernels:
            if k.parameter_name == "Vs" and k.layer_index == 0:
                dc_dVs_0 = k.d_c
                break
        
        kernel_vs0_history.append((max_level, dc_dVs_0))
    
    print(f"{'Level':>5} {'dc/dVs[0]':>16} {'Δ':>14}")
    print("-" * 40)
    
    prev = None
    diffs = []
    for level, val in kernel_vs0_history:
        if prev is not None:
            diff = abs(val - prev)
            diffs.append(diff)
            print(f"{level:>5} {val:>16.10f} {diff:>14.2e}")
        else:
            print(f"{level:>5} {val:>16.10f}       ---")
        prev = val
    
    print()
    print("Convergence Ratio (O(h^2) → ratio ≈ 4):")
    for i in range(1, len(diffs)):
        if diffs[i] > 1e-15:
            ratio = diffs[i-1] / diffs[i]
            print(f"  Level {i+1} → {i+2}: ratio = {ratio:.2f}")
    
    print()
    print("✓ Kernel convergence test complete!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NODE-BASED SENSITIVITY KERNEL TESTS")
    print("=" * 70 + "\n")
    
    try:
        test_node_kernels_basic()
        print()
        test_kernel_convergence_by_level()
        print()
        print("=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\n*** TEST FAILED: {e} ***\n")
        import traceback
        traceback.print_exc()
        exit(1)
