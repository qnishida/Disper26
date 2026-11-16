#!/usr/bin/env python3
"""
Verification: Check if sensitivity kernels converge with O(h^2) error.
This is prerequisite for Romberg acceleration to work on kernels.

If kernels converge as O(h^2), then Richardson extrapolation (Romberg) should work.
"""

import disper26py as dp
import numpy as np


def get_refined_model(base_depths, base_params, refinement_level):
    """
    Refine a model by subdividing each segment into 2^n parts.
    Returns (layers, thicknesses) for LAYER_CAKE-style calculation.
    """
    n_subdivisions = 2 ** refinement_level
    
    layers = []
    thicknesses = []
    
    for i in range(len(base_depths) - 1):
        z_start = base_depths[i]
        z_end = base_depths[i + 1]
        p_start = base_params[i]
        p_end = base_params[i + 1]
        
        segment_thickness = z_end - z_start
        sub_thickness = segment_thickness / n_subdivisions
        
        for j in range(n_subdivisions):
            # Linear interpolation at midpoint
            t = (j + 0.5) / n_subdivisions
            
            # Interpolate parameters
            vs = (1 - t) * p_start.get_vs() + t * p_end.get_vs()
            vp = (1 - t) * p_start.get_vp() + t * p_end.get_vp()
            rho = (1 - t) * p_start.rho + t * p_end.rho
            
            layers.append(dp.create_isotropic_parameters(vs, vp, rho))
            thicknesses.append(sub_thickness)
    
    return layers, thicknesses


def test_kernel_convergence():
    """
    Test if sensitivity kernels follow O(h^2) convergence.
    """
    
    # Simple linear gradient model
    base_depths = [0.0, 10.0, 20.0, 30.0]
    base_params = []
    for z in base_depths:
        vs = 2.5 + 0.05 * z
        vp = 4.3 + 0.08 * z
        rho = 2.4 + 0.01 * z
        base_params.append(dp.create_isotropic_parameters(vs, vp, rho))
    
    halfspace = base_params[-1]
    
    period = 20.0
    omega = 2.0 * np.pi / period
    
    print("=" * 80)
    print("Verification: Sensitivity Kernel O(h^2) Convergence")
    print("=" * 80)
    
    # First, find phase velocity at high refinement (reference)
    layers_ref, thicknesses_ref = get_refined_model(base_depths, base_params, 5)
    roots = dp.search_roots(
        layers_ref, thicknesses_ref, halfspace, omega, 2.5, 4.0, 200, "Rayleigh")
    assert roots, "No reference root found"
    c_ref = roots[0]
    print(f"Reference phase velocity (level 5): {c_ref:.10f} km/s")
    print()
    
    # Compute kernels at different refinement levels
    print(f"{'Level':>5} {'Layers':>8} {'c (km/s)':>14} {'dc/dVs[0]':>14} {'Δ(dc/dVs[0])':>14}")
    print("-" * 80)
    
    prev_kernel_val = None
    kernel_values = []
    
    for level in range(6):
        layers, thicknesses = get_refined_model(base_depths, base_params, level)
        
        # Find root
        roots = dp.search_roots(
            layers, thicknesses, halfspace, omega, 2.5, 4.0, 200, "Rayleigh")
        assert roots, f"Level {level}: No root found"
        c = roots[0]
        
        # Compute kernels
        kernels = dp.compute_kernels_ad_layer_cake(
            layers, thicknesses, halfspace, omega, c,
            wave_type="Rayleigh", sensitivity_mode="Isotropic"
        )
        
        # Find dc/dVs for first layer
        dc_dVs_0 = None
        for k in kernels:
            if k.parameter_name == "Vs" and k.layer_index == 0:
                dc_dVs_0 = k.d_c
                break
        
        assert dc_dVs_0 is not None, f"Level {level}: Kernel not found"
        assert np.isfinite(dc_dVs_0), f"Level {level}: Nonfinite kernel"
        
        kernel_values.append((level, dc_dVs_0))
        
        if prev_kernel_val is not None:
            diff = abs(dc_dVs_0 - prev_kernel_val)
            print(f"{level:>5} {len(layers):>8} {c:>14.10f} {dc_dVs_0:>14.10f} {diff:>14.2e}")
        else:
            print(f"{level:>5} {len(layers):>8} {c:>14.10f} {dc_dVs_0:>14.10f}       ---")
        
        prev_kernel_val = dc_dVs_0
    
    print()
    
    # Check O(h^2) convergence by looking at ratio of successive differences
    # For O(h^2), ratio should be approximately 4
    print("Convergence Ratio Analysis (O(h^2) → ratio ≈ 4):")
    print("-" * 50)
    
    diffs = []
    for i in range(1, len(kernel_values)):
        diff = abs(kernel_values[i][1] - kernel_values[i-1][1])
        diffs.append(diff)
    
    for i in range(1, len(diffs)):
        if diffs[i] > 1e-15:
            ratio = diffs[i-1] / diffs[i]
            print(f"  Level {i} → {i+1}: ratio = {ratio:.2f}")
    
    print()
    
    # Try Romberg extrapolation on kernels
    print("Romberg Extrapolation Test for dc/dVs[0]:")
    print("-" * 50)
    
    if len(kernel_values) >= 3:
        values = [kv[1] for kv in kernel_values]
        
        # Neville's algorithm
        def neville_extrapolation(table):
            n = len(table)
            P = [[0.0] * n for _ in range(n)]
            for i in range(n):
                P[i][0] = table[i]
            
            for j in range(1, n):
                ratio = 4.0 ** j  # O(h^2) convergence
                for i in range(j, n):
                    P[i][j] = P[i][j-1] + (P[i][j-1] - P[i-1][j-1]) / (ratio - 1.0)
            
            return P[n-1][n-1], abs(P[n-1][n-1] - P[n-2][n-2]) if n >= 2 else 1.0
        
        for n_levels in range(2, len(values) + 1):
            ext_val, err = neville_extrapolation(values[:n_levels])
            print(f"  Using levels 0-{n_levels-1}: extrapolated = {ext_val:.10f}, err = {err:.2e}")
    
    print()
    print("=" * 80)
    print("Conclusion:")
    print("  If ratio ≈ 4 consistently, Romberg acceleration is viable for kernels.")
    print("=" * 80)


if __name__ == "__main__":
    test_kernel_convergence()
