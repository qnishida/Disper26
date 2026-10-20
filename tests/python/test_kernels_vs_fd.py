#!/usr/bin/env python3
"""
Verify node-based kernels by comparing against Finite Difference (FD) method.
Checks dc/dp, dU/dp, and dHV/dp correctness.
"""

import disper26py as dp
import numpy as np


def compute_fd_derivative(func, params, idx, attr_name, eps=1e-4):
    """Compute central difference derivative."""
    
    # Save original value
    original_val = getattr(params[idx], attr_name)
    
    # Forward
    setattr(params[idx], attr_name, original_val * (1 + eps))
    val_plus = func(params)
    
    # Backward
    setattr(params[idx], attr_name, original_val * (1 - eps))
    val_minus = func(params)
    
    # Restore
    setattr(params[idx], attr_name, original_val)
    
    # Derivative w.r.t parameter value (not scale factor)
    h = original_val * 2 * eps
    return (val_plus - val_minus) / h


def test_kernels_vs_fd():
    """Compare analytical kernels (AD) with Finite Difference."""
    
    depths = [0.0, 10.0, 20.0, 30.0]
    # Base parameters
    def make_params():
        params = []
        for z in depths:
            vs = 2.5 + 0.05 * z
            vp = 4.3 + 0.08 * z
            rho = 2.4 + 0.01 * z
            params.append(dp.create_isotropic_parameters(vs, vp, rho))
        return params
    
    params = make_params()
    is_discontinuity = [False, False, False, False]
    period = 20.0
    
    print("=" * 80)
    print("Verification: Node Kernels (AD) vs Finite Difference (FD)")
    print("=" * 80)
    
    # 1. Compute Analytical Kernels (AD) with Romberg
    result = dp.compute_dispersion_romberg_with_kernels(
        depths=depths,
        params=params,
        is_discontinuity=is_discontinuity,
        period=period,
        wave_type="Rayleigh",
        c_guess=3.0,
        sensitivity_mode="Isotropic",
        max_steps=5,
        tolerance=1e-12
    )
    c_ad, U_ad, HV_ad, _, _, kernels_ad = result
    
    # Target node/param to check
    target_node = 1
    target_param = "Vs"
    
    print(f"Checking Node {target_node}, Parameter {target_param}")
    print(f"Base Values: c={c_ad:.6f}, U={U_ad:.6f}, H/V={HV_ad:.6f}")
    print()
    
    # Find analytical values
    ad_vals = {}
    for k in kernels_ad:
        if k.layer_index == target_node and k.parameter_name == target_param:
            ad_vals = {'dc': k.d_c, 'dU': k.d_U, 'dHV': k.d_HV}
            break
            
    # 2. Compute Numerical Derivatives (FD)
    # We need a wrapper function that takes params and returns c, U, HV
    # Note: We must use Romberg for FD too to ensure consistency!
    
    def get_c(p_list):
        res = dp.compute_dispersion_romberg(
            depths=depths, params=p_list, is_discontinuity=is_discontinuity,
            period=period, wave_type="Rayleigh", c_guess=3.0, max_steps=5, tolerance=1e-12
        )
        return res[0]
        
    def get_U(p_list):
        res = dp.compute_dispersion_romberg(
            depths=depths, params=p_list, is_discontinuity=is_discontinuity,
            period=period, wave_type="Rayleigh", c_guess=3.0, max_steps=5, tolerance=1e-12
        )
        return res[1]

    def get_HV(p_list):
        res = dp.compute_dispersion_romberg(
            depths=depths, params=p_list, is_discontinuity=is_discontinuity,
            period=period, wave_type="Rayleigh", c_guess=3.0, max_steps=5, tolerance=1e-12
        )
        return res[2]

    # Vs in PhysicalParameters is computed from rho/mu, so we need to be careful.
    # Disper26 PhysicalParameters stores rho, c11, c33... 
    # For isotropic, changing Vs means changing c44 and c66.
    # But compute_fd_derivative accesses attributes directly.
    # The python binding might expose 'get_vs()' but maybe not setter 'set_vs()'.
    # We'll assume we modify the parameters directly in a copy-calculate way.
    
    eps = 1e-4
    
    # perturb +
    params_plus = make_params()
    # Change Vs at target node implies changing c44, c66 (and c13, c33, c11 if Vp/Rho fixed? No)
    # Isotropic: mu = rho * vs^2, lambda = rho * (vp^2 - 2*vs^2)
    # To change Vs only (keeping Vp, Rho constant):
    # New mu, New lambda.
    
    p_orig = params[target_node]
    vs_orig = p_orig.get_vs()
    vp_orig = p_orig.get_vp()
    rho_orig = p_orig.rho
    
    vs_plus = vs_orig * (1 + eps)
    params_plus[target_node] = dp.create_isotropic_parameters(vs_plus, vp_orig, rho_orig)
    
    # perturb -
    params_minus = make_params()
    vs_minus = vs_orig * (1 - eps)
    params_minus[target_node] = dp.create_isotropic_parameters(vs_minus, vp_orig, rho_orig)
    
    # Calculate FD
    c_plus = get_c(params_plus)
    c_minus = get_c(params_minus)
    dc_fd = (c_plus - c_minus) / (2 * vs_orig * eps)
    
    U_plus = get_U(params_plus)
    U_minus = get_U(params_minus)
    dU_fd = (U_plus - U_minus) / (2 * vs_orig * eps)
    
    HV_plus = get_HV(params_plus)
    HV_minus = get_HV(params_minus)
    dHV_fd = (HV_plus - HV_minus) / (2 * vs_orig * eps)
    
    # Compare
    print(f"{'Kernel':>10} {'AD (Analytic)':>16} {'FD (Numerical)':>16} {'Diff (%)':>10}")
    print("-" * 60)
    
    def print_cmp(name, ad, fd):
        diff_pct = abs(ad - fd) / abs(fd) * 100 if abs(fd) > 1e-10 else 0.0
        print(f"{name:>10} {ad:>16.8f} {fd:>16.8f} {diff_pct:>9.4f}%")
        
    print_cmp("dc/dVs", ad_vals['dc'], dc_fd)
    print_cmp("dU/dVs", ad_vals['dU'], dU_fd)
    print_cmp("dHV/dVs", ad_vals['dHV'], dHV_fd)
    
    print()
    print("Note: FD accuracy depends on epsilon and Romberg convergence.")
    print("      Small differences (< 0.1%) indicate correct implementation.")


if __name__ == "__main__":
    test_kernels_vs_fd()
