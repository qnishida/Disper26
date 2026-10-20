from pathlib import Path

import disper26py
import numpy as np
import matplotlib.pyplot as plt

def test_romberg_group_velocity_kernel(tmp_path):
    max_steps_loop = 10
    n0 = 1 
    
    # Model: 0-20km gradient over halfspace
    depths = [0.0, 20.0]
    p_top = disper26py.create_isotropic_parameters(3.0, 5.0, 2.5)
    p_bot = disper26py.create_isotropic_parameters(4.0, 7.0, 2.8)
    params = [p_top, p_bot]
    is_discontinuity = [False, False]
    
    target_node = 0 # Check sensitivity of Top Node
    
    period = 20.0
    c_guess = 3.6 

    # Use max_steps_loop + 1 to avoid IndexError
    K_table = np.zeros((max_steps_loop + 1, max_steps_loop + 1))
    errors = []

    print(f"{'n':<4} {'Layers':<8} {'Phase Vel':<12} {'Group Vel':<12} {'dU/dVs(0)':<15}")
    print("-" * 75)

    for n in range(n0, max_steps_loop):
        steps_limit = n + 1
        
        try:
            result = disper26py.compute_dispersion_romberg_with_kernels(
                depths, params, is_discontinuity, period, "Rayleigh", c_guess,
                "Isotropic", steps_limit, -1.0 
            )
            c_val, U_val, HV, distinct_level, rel_err, kernels = result
            
            num_sublayers = 2**distinct_level
            
            # Find Group Velocity kernel for node 0
            k_val = 0.0
            found = False
            for k in kernels:
                if k.layer_index == target_node and k.parameter_name == 'Vs':
                    k_val = k.d_U
                    found = True
                    break
            
            if not found:
                print(f"{n:<4} Kernel not found for node {target_node}")
                continue
                
            print(f"{distinct_level:<4} {num_sublayers:<8} {c_val:<12.6f} {U_val:<12.6f} {k_val:<15.6e}")
            
            # Store in table
            if distinct_level <= max_steps_loop:
                K_table[distinct_level, 0] = k_val
                
                # Manual Extrapolation Check
                for m in range(1, distinct_level + 1):
                    multiplier = 4**m
                    K_table[distinct_level, m] = (multiplier * K_table[distinct_level, m-1] - K_table[distinct_level-1, m-1]) / (multiplier - 1)
                
                if distinct_level > n0:
                     diff = abs(K_table[distinct_level, distinct_level] - K_table[distinct_level, distinct_level-1])
                     errors.append((1.0/num_sublayers, diff))

        except Exception as e:
            print(f"Error at n={n}: {e}")

    # Plot
    if len(errors) > 0:
        h_vals = [e[0] for e in errors]
        diffs = [e[1] for e in errors]
        
        plt.figure()
        plt.loglog(h_vals, diffs, '.-', label='Native U Kernel Convergence')
        plt.xlabel('1 / Sublayers (h)')
        plt.ylabel('Difference')
        plt.title('Convergence of Native Romberg Group Vel Kernel (Vs Node 0)')
        plt.grid(True, which="both", ls="-")
        
        output_path = tmp_path / 'kernel_U_convergence_native.png'
        plt.savefig(output_path)
        plt.close()
        print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[2] / 'build/test-results/romberg'
    output_dir.mkdir(parents=True, exist_ok=True)
    test_romberg_group_velocity_kernel(output_dir)
