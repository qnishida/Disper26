#!/usr/bin/env python3
"""
Verification: Kernel convergence with FIXED phase velocity (Romberg-converged c).
This tests if the discretization error in kernels is O(h^2) when c is held constant.
"""

import disper26py as dp
import numpy as np


def get_refined_model(base_depths, base_params, refinement_level):
    """Refine a model by subdividing each segment into 2^n parts."""
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
            t = (j + 0.5) / n_subdivisions
            vs = (1 - t) * p_start.get_vs() + t * p_end.get_vs()
            vp = (1 - t) * p_start.get_vp() + t * p_end.get_vp()
            rho = (1 - t) * p_start.rho + t * p_end.rho
            layers.append(dp.create_isotropic_parameters(vs, vp, rho))
            thicknesses.append(sub_thickness)
    
    return layers, thicknesses


def test_kernel_convergence_fixed_c():
    """Test kernel O(h^2) convergence with fixed (converged) c."""
    
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
    print("Verification: Kernel O(h^2) with FIXED converged c")
    print("=" * 80)
    
    # Get Romberg-converged phase velocity
    result = dp.compute_dispersion_romberg(
        depths=base_depths,
        params=base_params,
        is_discontinuity=[False] * 4,
        period=period,
        wave_type="Rayleigh",
        c_guess=3.0,
        max_steps=5,
        tolerance=1e-12
    )
    c_converged = result[0]
    print(f"Romberg-converged c: {c_converged:.10f} km/s")
    print()
    
    # Compute kernels at each level using FIXED c_converged
    print(f"{'Level':>5} {'Layers':>8} {'dc/dVs[0]':>16} {'Δ(dc/dVs[0])':>16}")
    print("-" * 60)
    
    prev_kernel_val = None
    kernel_values = []
    diffs = []
    
    for level in range(6):
        layers, thicknesses = get_refined_model(base_depths, base_params, level)
        
        # Use FIXED c_converged for all levels
        kernels = dp.compute_sensitivity_kernels_full_ad(
            layers, thicknesses, halfspace, omega, c_converged,
            wave_type="Rayleigh", sensitivity_mode="Isotropic"
        )
        
        dc_dVs_0 = None
        for k in kernels:
            if k.parameter_name == "Vs" and k.layer_index == 0:
                dc_dVs_0 = k.d_c
                break
        
        if dc_dVs_0 is None:
            print(f"  Level {level}: Kernel not found")
            continue
        
        kernel_values.append((level, dc_dVs_0))
        
        if prev_kernel_val is not None:
            diff = abs(dc_dVs_0 - prev_kernel_val)
            diffs.append(diff)
            print(f"{level:>5} {len(layers):>8} {dc_dVs_0:>16.10f} {diff:>16.2e}")
        else:
            print(f"{level:>5} {len(layers):>8} {dc_dVs_0:>16.10f}       ---")
        
        prev_kernel_val = dc_dVs_0
    
    print()
    
    # Convergence ratio
    print("Convergence Ratio (O(h^2) → ratio ≈ 4):")
    print("-" * 40)
    for i in range(1, len(diffs)):
        if diffs[i] > 1e-15:
            ratio = diffs[i-1] / diffs[i]
            print(f"  Level {i} → {i+1}: ratio = {ratio:.2f}")
    
    print()
    
    # Romberg extrapolation
    print("Romberg Extrapolation:")
    print("-" * 40)
    
    if len(kernel_values) >= 3:
        values = [kv[1] for kv in kernel_values]
        
        def neville_extrapolation(table):
            n = len(table)
            P = [[0.0] * n for _ in range(n)]
            for i in range(n):
                P[i][0] = table[i]
            for j in range(1, n):
                ratio = 4.0 ** j
                for i in range(j, n):
                    P[i][j] = P[i][j-1] + (P[i][j-1] - P[i-1][j-1]) / (ratio - 1.0)
            return P[n-1][n-1], abs(P[n-1][n-1] - P[n-2][n-2]) if n >= 2 else 1.0
        
        for n_levels in range(2, len(values) + 1):
            ext_val, err = neville_extrapolation(values[:n_levels])
            print(f"  Using 0-{n_levels-1}: extrapolated = {ext_val:.10f}, Δ = {err:.2e}")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    test_kernel_convergence_fixed_c()
