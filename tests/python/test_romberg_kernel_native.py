
import disper26py
import numpy as np
import matplotlib.pyplot as plt

def test_romberg_kernel_native():
    max_steps_loop = 10
    n0 = 1 
    # n=0 (1 layer) might be too crude, let's start from 1.
    
    # Model: 0-20km gradient over halfspace
    depths = [0.0, 20.0]
    p_top = disper26py.create_isotropic_parameters(3.0, 5.0, 2.5)
    p_bot = disper26py.create_isotropic_parameters(4.0, 7.0, 2.8)
    params = [p_top, p_bot]
    is_discontinuity = [False, False]

    # Halfspace implicitly defined by p_bot if not passed? 
    # Actually based on manual script, p_bot is at 20km.
    # The native function will likely treat the last layer as "Layer to Infinity"? or "Layer above Halfspace"?
    # For now we assume the setup is valid enough to produce a result.
    
    target_node = 0 # Check sensitivity of Top Node
    
    period = 20.0
    c_guess = 3.6 

    K_table = np.zeros((max_steps_loop + 1, max_steps_loop + 1))
    errors = []

    print(f"{'n':<4} {'Layers':<8} {'Phase Vel':<12} {'dC/dVs(0)':<15}")
    print("-" * 60)

    for n in range(n0, max_steps_loop):
        # We want to check result at Romberg level 'n'.
        # We set steps_limit = n + 1 (assuming 0-indexed levels 0..n).
        # We force it to go to 'n' with tol=0.0.
        steps_limit = n + 1
        
        try:
            result = disper26py.compute_dispersion_romberg_with_kernels(
                depths, params, is_discontinuity, period, "Rayleigh", c_guess,
                "Isotropic", steps_limit, -1.0 
            )
            c_val, U, HV, distinct_level, rel_err, kernels = result
            
            # Verify distinct_level
            # If we asked for n+1 steps, we expect level n.
            # 2^n sublayers.
            num_sublayers = 2**distinct_level
            
            # Find kernel for node 0
            k_val = 0.0
            found = False
            for k in kernels:
                if k.layer_index == target_node and k.parameter_name == 'Vs':
                    k_val = k.d_c 
                    found = True
                    break
            
            if not found:
                print(f"{n:<4} Kernel not found for node {target_node}")
                continue
                
            print(f"{distinct_level:<4} {num_sublayers:<8} {c_val:<12.6f} {k_val:<15.6e}")
            
            # Store in table
            if distinct_level <= max_steps_loop:
                K_table[distinct_level, 0] = k_val
                
                # We can perform manual extrapolation on top of the results just to see.
                # BUT the results themselves *should already be improving* if Romberg works (indirectly).
                # Actually, the 'kernels' are computed on the refined grid.
                # So K_table[n, 0] IS the approximation at level n.
                # We can manually extrapolate this to see convergence ORDER.
                
                for m in range(1, distinct_level + 1):
                    multiplier = 4**m
                    K_table[distinct_level, m] = (multiplier * K_table[distinct_level, m-1] - K_table[distinct_level-1, m-1]) / (multiplier - 1)
                
                if distinct_level > n0:
                    # Compare raw convergence (diagonal difference)
                     diff = abs(K_table[distinct_level, distinct_level] - K_table[distinct_level, distinct_level-1])
                     errors.append((1.0/num_sublayers, diff))

        except Exception as e:
            print(f"Error at n={n}: {e}")

    # Plot
    if len(errors) > 0:
        h_vals = [e[0] for e in errors]
        diffs = [e[1] for e in errors]
        
        plt.figure()
        plt.loglog(h_vals, diffs, '.-', label='Native Kernel Convergence')
        plt.xlabel('1 / Sublayers (h)')
        plt.ylabel('Difference')
        plt.title('Convergence of Native Romberg Kernel (Vs Node 0)')
        plt.grid(True, which="both", ls="-")
        
        plt.savefig('kernel_convergence_native.png')
        print("\nPlot saved to kernel_convergence_native.png")

if __name__ == "__main__":
    test_romberg_kernel_native()
