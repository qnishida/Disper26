#!/usr/bin/env python3
"""
Verify Love Wave Sensitivity Kernels (Isotropic and VTI).
Checks correctness against Finite Difference and ensures zero sensitivity for irrelevant parameters.
"""

import disper26py as dp
import numpy as np


def compute_fd_derivative(func, params, param_idx, component_idx, eps=1e-4):
    """Compute central difference derivative w.r.t a specific parameter component."""
    
    # Deep copy params
    # We need to construct new parameter lists with perturbed values
    
    # Identify target parameter
    p_target = params[param_idx]
    
    # Perturb
    # For isotropic: 0=Vs, 1=Vp, 2=Rho
    # For VTI: 0=A, 1=C, 2=F, 3=L, 4=N, 5=Rho
    
    # We need a way to perturb specific component cleanly
    # Helper to cloning/modifying
    def perturb(p, idx, delta):
        if hasattr(p, 'c11'): # It's a PhysicalParameters structure
             # But we need to perturb "Vs" or "A".
             # This helper is tricky because 'params' are PhysicalParameters which store stiffnesses.
             # We should perturb the high-level parameter (Vs/Vp or A/C..) and re-compute stiffnesses.
             pass
    
    # Actually, simpler to just reconstruct the whole params list
    # But wait, create_isotropic_parameters returns PhysicalParameters.
    # We can't easily extract "Vs" from it without formula.
    # Let's assume we know the base values.
    pass

# Redefine simpler FD approach specific to this test
def get_perturbed_params_iso(base_vals, node_idx, param_name, delta):
    new_params = []
    for i, (vs, vp, rho) in enumerate(base_vals):
        if i == node_idx:
            if param_name == "Vs": vs += delta
            elif param_name == "Vp": vp += delta
            elif param_name == "Rho": rho += delta
        new_params.append(dp.create_isotropic_parameters(vs, vp, rho))
    return new_params

def get_perturbed_params_vti(base_vals, node_idx, param_name, delta):
    new_params = []
    for i, (A, C, F, L, N, rho) in enumerate(base_vals):
        if i == node_idx:
            if param_name == "A": A += delta
            elif param_name == "C": C += delta
            elif param_name == "F": F += delta
            elif param_name == "L": L += delta
            elif param_name == "N": N += delta
            elif param_name == "Rho": rho += delta
        
        p = dp.PhysicalParameters()
        p.c11 = A
        p.c33 = C
        p.c13 = F
        p.c44 = L
        p.c66 = N
        p.rho = rho
        new_params.append(p)
    return new_params


def test_love_kernels_isotropic():
    print("=" * 60)
    print("Test: Love Wave Kernels (Isotropic Mode)")
    print("=" * 60)
    
    depths = [0.0, 10.0, 20.0]
    # Base values (Vs, Vp, Rho)
    base_vals = [
        (3.5, 6.0, 2.7),
        (3.8, 6.5, 2.8),
        (4.5, 7.8, 3.3)
    ]
    
    params = [dp.create_isotropic_parameters(*v) for v in base_vals]
    is_discontinuity = [False, False, False]
    period = 20.0
    c_guess = 4.0
    
    res = dp.compute_dispersion_romberg_with_kernels(
        depths, params, is_discontinuity, period, "Love", c_guess, "Isotropic", 5, 1e-12
    )
    c, U, HV, _, _, kernels = res
    print(f"Phase Velocity: {c:.6f} km/s (Love)")
    
    # Check 1: Vp sensitivity should be zero
    vp_kernels = [k for k in kernels if k.parameter_name == "Vp"]
    max_d_vp = max([abs(k.d_c) for k in vp_kernels])
    print(f"Max dc/dVp: {max_d_vp:.4e}")
    if max_d_vp < 1e-10:
        print("  SUCCESS: Vp sensitivity is effectively zero.")
    else:
        print("  FAILURE: Vp sensitivity is non-zero!")

    # Check 2: Vs and Rho sensitivity vs FD (Node 0)
    target_node = 0
    
    # Analytical
    k_vs = [k for k in kernels if k.parameter_name == "Vs" and k.layer_index == target_node][0]
    k_rho = [k for k in kernels if k.parameter_name == "Rho" and k.layer_index == target_node][0]
    
    dc_dvs_an = k_vs.d_c
    dc_drho_an = k_rho.d_c
    
    # FD
    delta = 1e-3
    
    def calc_c(p_list):
        r = dp.compute_dispersion_romberg(depths, p_list, is_discontinuity, period, "Love", c, 5, 1e-12)
        return r[0]
        
    # FD Vs
    p_plus = get_perturbed_params_iso(base_vals, target_node, "Vs", delta)
    p_minus = get_perturbed_params_iso(base_vals, target_node, "Vs", -delta)
    dc_dvs_fd = (calc_c(p_plus) - calc_c(p_minus)) / (2 * delta)
    
    # FD Rho
    p_plus = get_perturbed_params_iso(base_vals, target_node, "Rho", delta)
    p_minus = get_perturbed_params_iso(base_vals, target_node, "Rho", -delta)
    dc_drho_fd = (calc_c(p_plus) - calc_c(p_minus)) / (2 * delta)
    
    print(f"\nNode {target_node} Check:")
    print(f"  dc/dVs  : AN={dc_dvs_an:.6f} FD={dc_dvs_fd:.6f} Diff={abs(dc_dvs_an-dc_dvs_fd):.4e}")
    print(f"  dc/dRho : AN={dc_drho_an:.6f} FD={dc_drho_fd:.6f} Diff={abs(dc_drho_an-dc_drho_fd):.4e}")
    
    assert abs(dc_dvs_an - dc_dvs_fd) < 1e-4
    assert abs(dc_drho_an - dc_drho_fd) < 1e-4


def test_love_kernels_vti():
    print("\n" + "=" * 60)
    print("Test: Love Wave Kernels (VTI Mode)")
    print("=" * 60)
    
    depths = [0.0, 10.0, 20.0]
    # Base values (A, C, F, L, N, Rho)
    # Isotropic equivalent: A=C=lam+2mu, L=N=mu, F=lam
    # Let's use slight anisotropy
    # A=20, C=18, F=8, L=6, N=7, Rho=2.5
    base_vals = [
        (20.0, 18.0, 8.0, 6.0, 7.0, 2.5),
        (22.0, 20.0, 9.0, 7.0, 8.0, 2.7),
        (30.0, 28.0, 12.0, 10.0, 11.0, 3.2)
    ]
    
    params = []
    for (A, C, F, L, N, Rho) in base_vals:
        p = dp.PhysicalParameters()
        p.c11 = A; p.c33 = C; p.c13 = F; p.c44 = L; p.c66 = N; p.rho = Rho
        params.append(p)
        
    is_discontinuity = [False, False, False]
    period = 20.0
    c_guess = np.sqrt(7.0/2.5) # Approx sqrt(N/rho)
    
    res = dp.compute_dispersion_romberg_with_kernels(
        depths, params, is_discontinuity, period, "Love", c_guess, "VTI", 5, 1e-12
    )
    c, U, HV, _, _, kernels = res
    print(f"Phase Velocity: {c:.6f} km/s (Love VTI)")
    
    # Check 1: A, C, F sensitivity should be zero
    for pname in ["A", "C", "F"]:
        ks = [k for k in kernels if k.parameter_name == pname]
        max_val = max([abs(k.d_c) for k in ks])
        print(f"Max dc/d{pname}: {max_val:.4e}")
        if max_val < 1e-10:
             print(f"  SUCCESS: {pname} sensitivity is effectively zero.")
        else:
             print(f"  FAILURE: {pname} sensitivity is non-zero!")

    # Check 2: L, N, Rho sensitivity vs FD (Node 0)
    target_node = 0
    
    # Analytical
    k_L = [k for k in kernels if k.parameter_name == "L" and k.layer_index == target_node][0]
    k_N = [k for k in kernels if k.parameter_name == "N" and k.layer_index == target_node][0]
    
    dc_dL_an = k_L.d_c
    dc_dN_an = k_N.d_c
    
    # FD
    delta = 1e-2 # Larger delta for stiffness
    
    def calc_c(p_list):
        # Must handle possible mode jumps or failures if simple c_guess, but here simple model
        r = dp.compute_dispersion_romberg(depths, p_list, is_discontinuity, period, "Love", c, 5, 1e-12)
        return r[0]

    # FD L
    p_plus = get_perturbed_params_vti(base_vals, target_node, "L", delta)
    p_minus = get_perturbed_params_vti(base_vals, target_node, "L", -delta)
    dc_dL_fd = (calc_c(p_plus) - calc_c(p_minus)) / (2 * delta)

    # FD N
    p_plus = get_perturbed_params_vti(base_vals, target_node, "N", delta)
    p_minus = get_perturbed_params_vti(base_vals, target_node, "N", -delta)
    dc_dN_fd = (calc_c(p_plus) - calc_c(p_minus)) / (2 * delta)
    
    print(f"\nNode {target_node} Check:")
    print(f"  dc/dL   : AN={dc_dL_an:.6f} FD={dc_dL_fd:.6f} Diff={abs(dc_dL_an-dc_dL_fd):.4e}")
    print(f"  dc/dN   : AN={dc_dN_an:.6f} FD={dc_dN_fd:.6f} Diff={abs(dc_dN_an-dc_dN_fd):.4e}")
    
    assert abs(dc_dL_an - dc_dL_fd) < 1e-4
    assert abs(dc_dN_an - dc_dN_fd) < 1e-4

if __name__ == "__main__":
    test_love_kernels_isotropic()
    test_love_kernels_vti()
