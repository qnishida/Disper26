#!/usr/bin/env python3
"""
Verify Adaptive Romberg Stopping.
Ensures that the engine stops when error increases, rather than continuing to max_steps.
"""

import disper26py as dp
import numpy as np

def test_adaptive_stop():
    print("=" * 60)
    print("Testing Adaptive Romberg Stopping")
    print("=" * 60)
    
    # Model configuration that showed kernel precision degradation at Level 7
    # in test_romberg_accuracy.py
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
    
    # Use max_steps=12 to allow deep search
    # Use strict tolerance 1e-15 to force it to continue until error degrades
    max_steps = 12
    tolerance = 1e-15
    
    print(f"Running with max_steps={max_steps}, tolerance={tolerance}")
    
    res = dp.compute_dispersion_romberg(
        depths, params, is_discontinuity, period, "Rayleigh", c_guess, max_steps, tolerance
    )
    
    c, U, HV, level, error = res
    
    print(f"Result:")
    print(f"  Phase Velocity: {c:.8f}")
    print(f"  Converged Level: {level}")
    print(f"  Relative Error:  {error:.4e}")
    
    # If adaptive stops works:
    # It should stop when error starts increasing relative to previous step.
    # Level 6 is usually best -> approx 1e-9 error (for kernels, maybe c is similar)
    # Level 7 -> error increases?
    
    if level < max_steps:
        print(f"SUCCESS: Stopped early at Level {level} (max_steps={max_steps}).")
        
        # Check if the error is reasonably small
        if error < 1e-6:
             print("  Error is small, indicating convergence.")
        else:
             print("  Warning: Stopped with large error.")
             
    else:
        print(f"FAILURE: Reached max_steps ({max_steps}). Adaptive logic did not trigger.")
        
    # We expect it to stop before max_steps
    assert level < max_steps, f"Should stop before max_steps={max_steps} if error increases"

if __name__ == "__main__":
    test_adaptive_stop()
