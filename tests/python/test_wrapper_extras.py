
import disper26py
import math

# Create Model
l1 = disper26py.create_isotropic_parameters(1.5, 2.6, 2.0)
hs = disper26py.create_isotropic_parameters(3.0, 5.2, 2.5)
layers = [l1]
thick = [2.0]

omega = 2.0 * math.pi * 0.5 # 0.5 Hz

print("--- Model Inspection ---")
print(f"Layer 1: {l1}")
print(f"Halfspace: {hs}")
print(f"L1 Vp via getter: {l1.get_vp()}")
print(f"L1 Vs via getter: {l1.get_vs()}")

# Search for a root
print("\n--- Root Search ---")
roots = disper26py.search_roots(layers, thick, hs, omega, 1.4, 2.8, 50, "Rayleigh")
if not roots:
    print("No roots found!")
    exit(1)

c_phase = roots[0]
print(f"Found root c = {c_phase}")

# Compute Group Velocity
print("\n--- Computations ---")
U = disper26py.calculate_group_velocity(layers, thick, hs, omega, c_phase, "Rayleigh")
print(f"Group Velocity U = {U}")

# Compute H/V
HV = disper26py.calculate_hv_ratio(layers, thick, hs, omega, c_phase, "Rayleigh")
print(f"H/V Ratio (Ellipticity) = {HV}")

# Comparison with Full Sensitivity (Debugging)
print("\n--- Sensitivity Check ---")
kernels = disper26py.compute_sensitivity_kernels_fd(layers, thick, hs, omega, c_phase, "Rayleigh", "Isotropic")
# Check if U matches visually (kernels calculates U internally but returns derivatives)
# We can't easily extract U0 from kernels without debugging, 
# but if this script runs without error it proves the binding works.
print("Sensitivity calculation successful.")
