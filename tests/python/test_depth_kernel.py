#!/usr/bin/env python3
"""
Verify discontinuity depth sensitivity kernels (dc/dz).
"""

import disper26py as dp
import numpy as np


def compute_fd_derivative(func, depths, params, is_disc, disc_idx, eps=1e-4):
    """Compute central difference derivative w.r.t discontinuity depth."""
    
    # Original depth
    # Find which index in depths[] corresponds to discontinuity disc_idx
    # The convention in our test is that we just pass the vector
    # and know which index to modify.
    
    # Identify depth index
    # We iterate is_discontinuity. If true, it's a discontinuity.
    count = 0
    target_idx = -1
    for i, flag in enumerate(is_disc):
        if flag:
            if count == disc_idx:
                target_idx = i
                break
            count += 1
            
    if target_idx == -1:
        raise ValueError(f"Discontinuity {disc_idx} not found")
        
    z_orig = depths[target_idx]
    
    # Perturb depths
    # We must move BOTH the node at the discontinuity and the one after it (since they represent the interface)
    # Actually, in Mesher logic, is_disc[i]=true means segment i is skipped.
    # The depth at i is the interface depth.
    # We usually model it as: node i (bottom of upper layer) and node i+1 (top of lower layer) are at same depth.
    
    depths_plus = list(depths)
    depths_minus = list(depths)
    
    h = 1.0 * eps  # Absolute perturbation 1m if eps=1e-3? No, relative.
    delta = max(0.001, z_orig * eps)
    
    # Perturb target node (and i+1 usually)
    depths_plus[target_idx] += delta
    if target_idx + 1 < len(depths):
        depths_plus[target_idx+1] += delta
        
    depths_minus[target_idx] -= delta
    if target_idx + 1 < len(depths):
        depths_minus[target_idx+1] -= delta
    
    # Compute values
    def run(d):
        res = dp.compute_dispersion_romberg(
            d, params, is_disc, 20.0, "Rayleigh", 3.0, 5, 1e-12
        )
        return res[0], res[1], res[2] # c, U, HV
        
    c_plus, U_plus, HV_plus = run(depths_plus)
    c_minus, U_minus, HV_minus = run(depths_minus)
    
    dc = (c_plus - c_minus) / (2 * delta)
    dU = (U_plus - U_minus) / (2 * delta)
    dHV = (HV_plus - HV_minus) / (2 * delta)
    
    return dc, dU, dHV


def test_depth_kernel():
    """Test dc/dz kernel."""
    
    # Model with discontinuity at 10km and 20km
    depths = [0.0, 10.0, 10.0, 20.0, 20.0, 30.0]
    
    # Params: Layer 1, Layer 2, Layer 3
    # Node 0 (0km)
    p1 = dp.create_isotropic_parameters(2.5, 4.3, 2.45)
    # Node 1 (10km - Top of disc)
    p2 = dp.create_isotropic_parameters(2.6, 4.4, 2.5) 
    # Node 2 (10km - Bottom of disc)
    p3 = dp.create_isotropic_parameters(3.2, 5.5, 2.7)
    # Node 3 (20km - Top of disc)
    p4 = dp.create_isotropic_parameters(3.3, 5.6, 2.75)
    # Node 4 (20km - Bottom of disc)
    p5 = dp.create_isotropic_parameters(3.8, 6.5, 3.0)
    # Node 5 (30km - Halfspace)
    p6 = dp.create_isotropic_parameters(3.9, 6.6, 3.1)
    
    params = [p1, p2, p3, p4, p5, p6]
    
    # In GRID mode logic:
    # Segment 0: 0->10 (Active)
    # Segment 1: 10->10 (Discontinuity -> Skipped)
    # Segment 2: 10->20 (Active)
    # Segment 3: 20->20 (Discontinuity -> Skipped)
    # Segment 4: 20->30 (Active)
    
    is_discontinuity = [False, True, False, True, False, False]
    
    print("=" * 80)
    print("Verification: Discontinuity Depth Sensitivity (dc/dz)")
    print("=" * 80)
    
    # 1. Analytical (AD)
    result = dp.compute_dispersion_romberg_with_kernels(
        depths=depths,
        params=params,
        is_discontinuity=is_discontinuity,
        period=20.0,
        wave_type="Rayleigh",
        c_guess=3.0,
        sensitivity_mode="Isotropic",
        max_steps=5,
        tolerance=1e-12
    )
    c_ad, U_ad, HV_ad, _, _, kernels = result
    
    print(f"Base Phase Velocity: {c_ad:.6f} km/s")
    print()
    
    # Find Depth Kernels
    depth_kernels = []
    for k in kernels:
        if k.parameter_name == "Depth":
            depth_kernels.append(k)
            
    print(f"Found {len(depth_kernels)} depth kernels (expected 2)")
    
    # 2. Check against FD
    for i, k in enumerate(depth_kernels):
        # Determine depth value for reporting
        node_idx = k.layer_index
        z_val = depths[node_idx]
        
        print(f"\n--- Discontinuity {i+1} at {z_val} km (Node {node_idx}) ---")
        
        dc_an = k.d_c
        dU_an = k.d_U
        dHV_an = k.d_HV
        
        # FD
        dc_fd, dU_fd, dHV_fd = compute_fd_derivative(None, depths, params, is_discontinuity, i)
        
        def print_cmp(name, ad, fd):
            diff_pct = abs(ad - fd) / abs(fd) * 100 if abs(fd) > 1e-10 else 0.0
            print(f"{name:>10} {ad:>16.8f} {fd:>16.8f} {diff_pct:>9.4f}%")
            
        print_cmp("dc/dz", dc_an, dc_fd)
        print_cmp("dU/dz", dU_an, dU_fd)
        print_cmp("dHV/dz", dHV_an, dHV_fd)


if __name__ == "__main__":
    test_depth_kernel()
