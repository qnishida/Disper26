
import disper26py
import numpy as np
import matplotlib.pyplot as plt
import sys

def test_romberg_kernel():
    max_iter = 10
    n0 = 0
    tol = 1e-12

    # Model: 0-20km gradient over halfspace
    thick_total = 20.0
    vs_top, vs_bot = 3.0, 4.0
    vp_top, vp_bot = 5.0, 7.0
    rho_top, rho_bot = 2.5, 2.8

    # Halfspace
    hs_params = disper26py.create_isotropic_parameters(4.5, 8.0, 3.0)

    target_depth = 10.0 # Middle of the layer (10 km)

    period = 20.0
    omega = 2 * np.pi / period
    
    # Range for root search
    c_min = 3.0
    c_max = 4.5

    # Table for Kernel Density
    # K_table[n, m] where n is row (refinement level), m is order of extrapolation
    K_table = np.zeros((max_iter, max_iter))
    
    # Store errors for plotting
    errors = []
    
    print(f"{'n':<4} {'Layers':<8} {'Phase Vel':<12} {'dC/dVs':<15} {'K_density':<15}")
    print("-" * 60)

    for n in range(n0, max_iter):
        num_layers = 2**n
        h = thick_total / num_layers

        layers = []
        thicknesses = []

        # Build layers (Midpoint rule for parameters ideally, or just simple gradient)
        # Using midpoint to be consistent with typical Romberg schemes
        for i in range(num_layers):
            z_mid = (i + 0.5) * h
            ratio = z_mid / thick_total

            vs = vs_top + ratio * (vs_bot - vs_top)
            vp = vp_top + ratio * (vp_bot - vp_top)
            rho = rho_top + ratio * (rho_bot - rho_top)

            layers.append(disper26py.create_isotropic_parameters(vs, vp, rho))
            thicknesses.append(h)

        # 1. Compute Phase Velocity
        # search_roots returns a list of roots
        roots = disper26py.search_roots(
            layers, thicknesses, hs_params, omega, c_min, c_max, 100, 'Rayleigh'
        )
        
        if not roots:
            print(f"{n:<4} {num_layers:<8} {'No Root':<12}")
            continue
            
        c_phase = roots[0]

        # 2. Compute Sensitivity Kernels
        # This returns a list of SensitivityKernel objects
        # We need to find the kernel for the layer containing target_depth
        kernels = disper26py.compute_sensitivity_kernels_fd(
            layers, thicknesses, hs_params, omega, c_phase, 
            'Rayleigh', 'Isotropic', False
        )
        
        # 3. Extract Kernel for Vs at target depth
        i_target = int(target_depth / h)
        if i_target >= num_layers: i_target = num_layers - 1
        
        k_val = 0.0
        found = False
        for k in kernels:
            if k.layer_index == i_target and k.parameter_name == 'Vs':
                k_val = k.d_c  # d_PhaseVelocity is exposed as d_c in bindings
                found = True
                break

        if not found:
            print(f"Kernel not found for n={n}")
            continue

        # Kernel Density = K / h
        k_density = k_val / h
        
        print(f"{n:<4} {num_layers:<8} {c_phase:<12.6f} {k_val:<15.6e} {k_density:<15.6e}")

        K_table[n, 0] = k_density
        
        # Extrapolation
        for m in range(1, n + 1):
            multiplier = 4**m
            K_table[n, m] = (multiplier * K_table[n, m-1] - K_table[n-1, m-1]) / (multiplier - 1)
            
        # Check convergence
        if n > n0:
            diff = abs(K_table[n, n] - K_table[n, n-1])
            errors.append((2**-n, diff))
            # if diff < tol * abs(K_table[n, n]):
            #    print("Converged!")
            #    break

    # Result at max_iter
    best_val = K_table[max_iter-1, max_iter-1]
    print(f"\nFinal Extrapolated Value: {best_val}")

    # Plotting
    if len(errors) > 0:
        h_vals = [e[0] for e in errors]
        diffs = [e[1] for e in errors]
        
        plt.figure()
        plt.loglog(h_vals, diffs, '.-', label='Error (Difference)')
        
        # Reference line for O(h^2) or O(h^4) ?
        # Romberg should give high order.
        # Initial trapezoidal is O(h^2).
        # We are checking the diagonal diff, which gets better very fast.
        
        plt.xlabel('h (relative size)')
        plt.ylabel('Difference')
        plt.grid(True, which="both", ls="-")
        plt.title('Convergence of Romberg Extrapolation for Kernel Density')
        plt.savefig('kernel_convergence_plot.png')
        print("Plot saved to kernel_convergence_plot.png")

if __name__ == "__main__":
    test_romberg_kernel()
