#!/usr/bin/env python3
"""
Show actual Romberg extrapolation accuracy for all kernels.
"""

import disper26py as dp
import numpy as np


def neville_extrapolation(table):
    """Neville's algorithm for Richardson extrapolation (O(h^2) convergence)."""
    n = len(table)
    if n == 0:
        return 0.0, 1.0
    if n == 1:
        return table[0], 1.0
    
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        P[i][0] = table[i]
    
    for j in range(1, n):
        ratio = 4.0 ** j  # O(h^2)
        for i in range(j, n):
            P[i][j] = P[i][j-1] + (P[i][j-1] - P[i-1][j-1]) / (ratio - 1.0)
    
    extrapolated = P[n-1][n-1]
    error_estimate = abs(P[n-1][n-1] - P[n-2][n-2]) if n >= 2 else 1.0
    return extrapolated, error_estimate


def test_romberg_accuracy():
    """Demonstrate Romberg extrapolation accuracy for kernels."""
    
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
    print("Romberg Extrapolation Accuracy for Node 0, Vs Kernels")
    print("=" * 80)
    
    # Collect values at each level
    dc_vals = []
    dU_vals = []
    dHV_vals = []
    
    for max_level in range(8):  # 0 to 7 (step 6, 7 added)
        result = dp.compute_dispersion_romberg_with_kernels(
            depths=depths,
            params=params,
            is_discontinuity=is_discontinuity,
            period=period,
            wave_type="Rayleigh",
            c_guess=3.0,
            sensitivity_mode="Isotropic",
            max_steps=max_level,
            tolerance=1e-15
        )
        
        _, _, _, level, _, kernels = result
        
        for k in kernels:
            if k.parameter_name == "Vs" and k.layer_index == 0:
                dc_vals.append(k.d_c)
                dU_vals.append(k.d_U)
                dHV_vals.append(k.d_HV)
                break
    
    # Show Romberg extrapolation progression
    def show_romberg_progression(name, values):
        print(f"\n{name}:")
        print(f"  {'Levels':>10} {'Extrapolated':>16} {'Est. Error':>14} {'Rel. Error':>14}")
        print("-" * 60)
        
        for n_levels in range(2, len(values) + 1):
            ext, err = neville_extrapolation(values[:n_levels])
            rel_err = err / abs(ext) if abs(ext) > 1e-20 else 0.0
            print(f"  0-{n_levels-1:>7} {ext:>16.10f} {err:>14.2e} {rel_err:>14.2e}")
        
        # Final result
        ext_final, err_final = neville_extrapolation(values)
        rel_final = err_final / abs(ext_final) if abs(ext_final) > 1e-20 else 0.0
        print(f"\n  Final: {ext_final:.12f} ± {err_final:.2e} (rel: {rel_final:.2e})")
    
    show_romberg_progression("dc/dVs", dc_vals)
    show_romberg_progression("dU/dVs", dU_vals)
    show_romberg_progression("dHV/dVs", dHV_vals)
    
    print()
    print("=" * 80)
    print("Summary: Romberg Extrapolation Relative Precision")
    print("=" * 80)
    
    dc_ext, dc_err = neville_extrapolation(dc_vals)
    dU_ext, dU_err = neville_extrapolation(dU_vals)
    dHV_ext, dHV_err = neville_extrapolation(dHV_vals)
    
    print(f"  dc/dVs: {dc_err/abs(dc_ext):.2e}")
    print(f"  dU/dVs: {dU_err/abs(dU_ext):.2e}")
    print(f"  dHV/dVs: {dHV_err/abs(dHV_ext):.2e}")


if __name__ == "__main__":
    test_romberg_accuracy()
