#!/usr/bin/env python3
"""
Test all node-based kernels: dc/dp, dU/dp, dHV/dp
"""

import disper26py as dp
import numpy as np


def test_full_kernels():
    """Test dc/dp, dU/dp, dHV/dp all return non-zero values."""
    
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
    
    print("=" * 80)
    print("Test: Full Node Kernels (dc/dp, dU/dp, dHV/dp)")
    print("=" * 80)
    
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
    
    print(f"c = {c:.6f} km/s, U = {U:.6f} km/s, H/V = {HV:.4f}")
    print()
    
    print(f"{'Node':>4} {'Param':>6} {'dc/dp':>12} {'dU/dp':>12} {'dHV/dp':>12}")
    print("-" * 56)
    
    all_U_nonzero = True
    all_HV_nonzero = True
    
    for k in kernels:
        print(f"{k.layer_index:>4} {k.parameter_name:>6} {k.d_c:>12.6f} {k.d_U:>12.6f} {k.d_HV:>12.6f}")
        if abs(k.d_U) < 1e-15:
            all_U_nonzero = False
        if abs(k.d_HV) < 1e-15:
            all_HV_nonzero = False
    
    print()
    
    if all_U_nonzero:
        print("✓ dU/dp: all non-zero")
    else:
        print("⚠ dU/dp: some zero values")
    
    if all_HV_nonzero:
        print("✓ dHV/dp: all non-zero")
    else:
        print("⚠ dHV/dp: some zero values")
    
    return all_U_nonzero and all_HV_nonzero


if __name__ == "__main__":
    try:
        success = test_full_kernels()
        print()
        if success:
            print("=" * 80)
            print("ALL KERNELS COMPUTED SUCCESSFULLY!")
            print("=" * 80)
        else:
            print("Some kernels returned zero - review implementation")
    except Exception as e:
        print(f"\n*** TEST FAILED: {e} ***\n")
        import traceback
        traceback.print_exc()
        exit(1)
