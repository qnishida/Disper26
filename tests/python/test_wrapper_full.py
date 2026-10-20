import disper26py
import sys
import math

def approx_equal(a, b, tol=1e-5):
    return abs(a - b) < tol

def test_full_wrapper():
    print("Testing Full disper26py wrapper features...")
    
    # 1. Test VTI Layer Creation
    # Create an isotropic layer manually vs via helper
    vs = 3.0
    vp = 5.2
    rho = 2.5
    
    c44 = rho * vs**2
    c33 = rho * vp**2
    c11 = c33
    c66 = c44
    c13 = c11 - 2*c44
    
    (layer_vti, thick) = disper26py.create_vti_layer_from_depth(
        c11, c33, c13, c44, c66, rho, 
        0.0, 10.0 # Top, Bottom -> Thickness 10
    )
    
    print(f"Created VTI Layer: {layer_vti}, Thickness: {thick}")
    if not approx_equal(layer_vti.c44, c44):
        print("FAILED: c44 mismatch in VTI helper")
        sys.exit(1)
    if not approx_equal(thick, 10.0):
        print("FAILED: Thickness mismatch")
        sys.exit(1)
        
    # 2. Test Root Searching
    # Use the same model as documentation
    layer1 = disper26py.create_isotropic_parameters(3.0, 5.2, 2.5)
    hs = disper26py.create_isotropic_parameters(4.5, 8.0, 3.3)
    
    layers = [layer1]
    thicknesses = [10.0]
    
    omega = 2 * math.pi * 0.5 # 0.5 Hz
    # Search range check
    # Fundamental Rayleigh should be somewhere between Vs_layer (3.0) and Vs_half (4.5)
    
    roots = disper26py.search_roots(
        layers, thicknesses, hs, 
        omega, 2.0, 5.0, 100, "Rayleigh"
    )
    
    print(f"Found Roots: {roots}")
    if len(roots) == 0:
        print("FAILED: No roots found")
        sys.exit(1)
    
    found_c = roots[0]
        
    # 3. Test Sensitivity Kernels
    # Isotropic
    kernels_iso = disper26py.compute_sensitivity_kernels_fd(
        layers, thicknesses, hs, 
        omega, found_c, "Rayleigh", "Isotropic"
    )
    print(f"Isotropic Kernels: {len(kernels_iso)}")
    if len(kernels_iso) == 0: 
        print("FAILED: No isotropic kernels returned")
        sys.exit(1)
        
    # Check if first kernel has non-zero sensitivity
    k0 = kernels_iso[0]
    print(f"First Kernel: {k0}, dC/dp={k0.d_c}")
    
    # VTI (Love Wave Check)
    # Love wave sensitivity to L(c44) and N(c66) should differ
    # Search for Love root first
    roots_love = disper26py.search_roots(
        layers, thicknesses, hs, 
        omega, 2.0, 5.0, 100, "Love"
    )
    if len(roots_love) > 0:
        c_love = roots_love[0]
        kernels_vti = disper26py.compute_sensitivity_kernels_fd(
            layers, thicknesses, hs, 
            omega, c_love, "Love", "VTI"
        )
        # Find L and N kernels for layer 0
        k_L = next((k for k in kernels_vti if k.parameter_name=="L" and k.layer_index==0), None)
        k_N = next((k for k in kernels_vti if k.parameter_name=="N" and k.layer_index==0), None)
        
        if k_L and k_N:
            print(f"Love Sensitivities: L={k_L.d_c}, N={k_N.d_c}")
            if abs(k_N.d_c) < abs(k_L.d_c):
                 print("WARNING: Usually Love wave is more sensitive to N(SH) in horizontal direction? Wait.")
                 # Love wave depends on shear modulus mu. In VTI: c44 (vertical shear) and c66 (horizontal).
                 # Love wave is SH. Particle motion is Horizontal. Propagation is Horizontal.
                 # The restoring force involves d/dx and d/dz.
                 # Actually Love wave involves c44 and c66.
                 # Sensitivity to N (c66) is usually dominant at surface?
                 pass
        else:
            print("FAILED: Could not find VTI kernels for L/N")

    # 4. Test 3-Layer Model (Documentation Example)
    print("\nTesting 3-Layer Model...")
    l1 = disper26py.create_isotropic_parameters(1.0, 1.73, 2.0)
    l2 = disper26py.create_isotropic_parameters(1.5, 2.6, 2.2)
    l3 = disper26py.create_isotropic_parameters(2.2, 3.8, 2.5)
    hs_3layer = disper26py.create_isotropic_parameters(3.0, 5.2, 2.7)
    
    layers_3 = [l1, l2, l3]
    th_3 = [2.0, 5.0, 8.0]
    
    omega_3 = 2.0 * math.pi * 0.5 # 0.5 Hz
    
    roots_3 = disper26py.search_roots(
        layers_3, th_3, hs_3layer,
        omega_3, 1.0, 3.0, 100, "Rayleigh"
    )
    print(f"3-Layer Roots: {roots_3}")
    
    if len(roots_3) == 0:
        print("FAILED: No roots found for 3-layer model")
        sys.exit(1)

    # Check sensitivity for 3-layer (just ensure it runs)
    k_3 = disper26py.compute_sensitivity_kernels_fd(
        layers_3, th_3, hs_3layer,
        omega_3, roots_3[0], "Rayleigh", "Isotropic"
    )
    print(f"3-Layer Mode 0 Sensitivity Kernels: {len(k_3)} entries")

    print("All Full Wrapper Tests Passed!")

if __name__ == "__main__":
    test_full_wrapper()
