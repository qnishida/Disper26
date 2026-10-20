#!/usr/bin/env python3
"""
Test script for compute_dispersion_romberg function.
Tests Romberg acceleration on a simple linear gradient model.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp
import numpy as np

def test_basic_romberg():
    """Test Romberg acceleration on a simple 3-node model."""
    
    # Create a simple linear gradient model
    # depths: 0, 10, 20, 30 km (halfspace at 30km)
    depths = [0.0, 10.0, 20.0, 30.0]
    
    # Parameters: linear gradient in Vs
    # Vs = 2.5 + 0.05*z, Vp = 4.3 + 0.08*z, Rho = 2.4 + 0.01*z
    params = []
    for z in depths:
        vs = 2.5 + 0.05 * z
        vp = 4.3 + 0.08 * z
        rho = 2.4 + 0.01 * z
        params.append(dp.create_isotropic_parameters(vs, vp, rho))
    
    # No discontinuities
    is_discontinuity = [False, False, False, False]
    
    # Test at period 20s with initial guess c=3.0 km/s
    period = 20.0
    c_guess = 3.0
    
    print("=" * 60)
    print("Test: Romberg Acceleration on Linear Gradient Model")
    print("=" * 60)
    print(f"  Depths: {depths}")
    print(f"  Period: {period} s")
    print(f"  Initial c guess: {c_guess} km/s")
    print()
    
    # Call Romberg function
    result = dp.compute_dispersion_romberg(
        depths=depths,
        params=params,
        is_discontinuity=is_discontinuity,
        period=period,
        wave_type="Rayleigh",
        c_guess=c_guess,
        max_steps=5,
        tolerance=1e-10
    )
    
    c, U, HV, level, rel_err = result
    
    print("Results:")
    print(f"  Phase Velocity: {c:.8f} km/s")
    print(f"  Group Velocity: {U:.8f} km/s")
    print(f"  H/V Ratio:      {HV:.8f}")
    print(f"  Converged at level: {level}")
    print(f"  Relative error: {rel_err:.2e}")
    print()
    
    # Basic sanity checks
    assert 2.0 < c < 5.0, f"Phase velocity {c} out of reasonable range"
    assert 2.0 < U < 6.0, f"Group velocity {U} out of reasonable range"
    assert 0 < HV < 10, f"H/V ratio {HV} out of reasonable range"
    
    print("✓ All basic sanity checks passed!")
    return True


def test_discontinuity():
    """Test Romberg with a discontinuity (Moho)."""
    
    # depths: 0, 15, 30, 30, 50 km (Moho at 30km)
    depths = [0.0, 15.0, 30.0, 30.0, 50.0]
    
    # Crust: Vs~3.5, Mantle: Vs~4.5
    params = [
        dp.create_isotropic_parameters(3.2, 5.5, 2.7),   # 0 km
        dp.create_isotropic_parameters(3.5, 6.0, 2.8),   # 15 km
        dp.create_isotropic_parameters(3.7, 6.3, 2.9),   # 30 km (crust side)
        dp.create_isotropic_parameters(4.5, 8.0, 3.3),   # 30 km (mantle side)
        dp.create_isotropic_parameters(4.6, 8.1, 3.35),  # 50 km (halfspace)
    ]
    
    # Discontinuity at index 2 (between 30km crust and 30km mantle)
    is_discontinuity = [False, False, True, False, False]
    
    period = 30.0
    c_guess = 3.5
    
    print("=" * 60)
    print("Test: Romberg with Moho Discontinuity")
    print("=" * 60)
    print(f"  Depths: {depths}")
    print(f"  Discontinuity at index 2 (Moho)")
    print()
    
    result = dp.compute_dispersion_romberg(
        depths=depths,
        params=params,
        is_discontinuity=is_discontinuity,
        period=period,
        wave_type="Rayleigh",
        c_guess=c_guess,
        max_steps=5,
        tolerance=1e-10
    )
    
    c, U, HV, level, rel_err = result
    
    print("Results:")
    print(f"  Phase Velocity: {c:.8f} km/s")
    print(f"  Group Velocity: {U:.8f} km/s")
    print(f"  H/V Ratio:      {HV:.8f}")
    print(f"  Converged at level: {level}")
    print(f"  Relative error: {rel_err:.2e}")
    print()
    
    assert 3.0 < c < 5.0, f"Phase velocity {c} out of reasonable range"
    print("✓ Discontinuity test passed!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ROMBERG ACCELERATION VERIFICATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_basic_romberg()
        print()
        test_discontinuity()
        print()
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n*** TEST FAILED: {e} ***\n")
        import traceback
        traceback.print_exc()
        exit(1)
