import disper26py
import numpy as np

# Surface Layer
l1 = disper26py.create_isotropic_parameters(1.0, 1.73, 2.0)
# Second Layer
l2 = disper26py.create_isotropic_parameters(1.5, 2.6, 2.2)
# Third Layer
l3 = disper26py.create_isotropic_parameters(2.2, 3.8, 2.5)
# Half-space
half_space = disper26py.create_isotropic_parameters(3.0, 5.2, 2.7)

layers = [l1, l2, l3]
thicknesses = [2.0, 5.0, 8.0] # km
omega = 2 * np.pi * 0.5

print("--- Scan (500 pts, wider range) ---")
c_min = 0.50
c_max = 3.0

roots = disper26py.search_roots(
    layers, thicknesses, half_space,
    omega, c_min, c_max,
    500, "Rayleigh"
)
print(f"Found {len(roots)} Phase Velocities: {roots}")

print("\n--- Mode Counts at Roots ---")
print(f"{'Velocity':<12} {'Cont Mode':<12} {'Int Mode':<10}")
for efreq in roots:
    c_mode = disper26py.calculate_continuous_mode_count(layers, thicknesses, half_space, omega, efreq, 'Rayleigh')
    i_mode = disper26py.calculate_total_mode_count(layers, thicknesses, half_space, omega, efreq, 'Rayleigh')
    print(f"{efreq:<12.6f} {c_mode:<12.6f} {i_mode:<10}")

print("\n--- Checking if mode counts are near half-integers and integer labels ---")
ideal_match = True
for i, efreq in enumerate(roots):
    c_mode = disper26py.calculate_continuous_mode_count(layers, thicknesses, half_space, omega, efreq, 'Rayleigh')
    i_mode = disper26py.calculate_total_mode_count(layers, thicknesses, half_space, omega, efreq, 'Rayleigh')
    expected_cont = i + 0.5
    cont_deviation = abs(c_mode - expected_cont)
    int_match = (i_mode == i)
    status = "OK" if cont_deviation < 0.15 and int_match else "MISMATCH"
    if status == "MISMATCH":
        ideal_match = False
    print(
        f"Root {i}: c={efreq:.4f}, count={c_mode:.4f}, expected={expected_cont:.4f}, "
        f"deviation={cont_deviation:.4f}, int_mode={i_mode} [{status}]"
    )

if ideal_match:
    print("\n=== SUCCESS: All roots have half-integer continuous counts and matching integer labels ===")
else:
    print("\n=== WARNING: Some roots have inconsistent mode counts ===")
