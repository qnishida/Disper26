#!/usr/bin/env python3
"""
Verification: INTEGRATED kernel convergence.
The per-layer kernel decreases with refinement (more layers = smaller individual contribution).
But the TOTAL kernel (integral of dc/dVs over depth) should converge.
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


def test_integrated_kernel():
    """Test if TOTAL (integrated) kernel converges with O(h^2)."""
    
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
    print("Verification: INTEGRATED Kernel O(h^2) Convergence")
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
    
    print(f"{'Level':>5} {'Layers':>8} {'Σ(dc/dVs)':>16} {'Σ(dc/dVp)':>16} {'Σ(dc/dRho)':>16}")
    print("-" * 80)
    
    kernel_history = {'Vs': [], 'Vp': [], 'Rho': []}
    
    for level in range(6):
        layers, thicknesses = get_refined_model(base_depths, base_params, level)
        
        kernels = dp.compute_sensitivity_kernels_full_ad(
            layers, thicknesses, halfspace, omega, c_converged,
            wave_type="Rayleigh", sensitivity_mode="Isotropic"
        )
        
        # Sum all kernels by parameter type (excluding halfspace)
        total_dc_dVs = sum(k.d_c for k in kernels if k.parameter_name == "Vs" and k.layer_index >= 0)
        total_dc_dVp = sum(k.d_c for k in kernels if k.parameter_name == "Vp" and k.layer_index >= 0)
        total_dc_dRho = sum(k.d_c for k in kernels if k.parameter_name == "Rho" and k.layer_index >= 0)
        
        kernel_history['Vs'].append(total_dc_dVs)
        kernel_history['Vp'].append(total_dc_dVp)
        kernel_history['Rho'].append(total_dc_dRho)
        
        print(f"{level:>5} {len(layers):>8} {total_dc_dVs:>16.10f} {total_dc_dVp:>16.10f} {total_dc_dRho:>16.10f}")
    
    print()
    
    # Convergence analysis for Vs
    print("Convergence Analysis for Σ(dc/dVs):")
    print("-" * 50)
    
    values = kernel_history['Vs']
    diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
    
    for i, diff in enumerate(diffs):
        print(f"  Level {i} → {i+1}: Δ = {diff:.2e}", end="")
        if i > 0 and diffs[i] > 1e-15:
            ratio = diffs[i-1] / diffs[i]
            print(f", ratio = {ratio:.2f}")
        else:
            print()
    
    print()
    
    # Romberg extrapolation for integrated kernels
    print("Romberg Extrapolation for Σ(dc/dVs):")
    print("-" * 50)
    
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
    test_integrated_kernel()
