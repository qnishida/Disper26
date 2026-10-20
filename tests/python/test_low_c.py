import disper26py
import numpy as np
import math

# Test model
l1 = disper26py.create_isotropic_parameters(1.0, 1.73, 2.0)
l2 = disper26py.create_isotropic_parameters(1.5, 2.6, 2.2)
l3 = disper26py.create_isotropic_parameters(2.2, 3.8, 2.5)
half_space = disper26py.create_isotropic_parameters(3.0, 5.2, 2.7)

layers = [l1, l2, l3]
thicknesses = [2.0, 5.0, 8.0]
omega = 2 * np.pi * 0.5

print("=== Phase Monotonicity in Low Velocity Region ===")
print("Testing if count is monotonically increasing with velocity")
print()

prev = None
c_vals = np.linspace(0.9, 1.8, 20)
print(f"{'c (km/s)':<12} {'Count':<12} {'Delta':<12} {'Status'}")
for c in c_vals:
    count = disper26py.calculate_continuous_mode_count(layers, thicknesses, half_space, omega, c, 'Rayleigh')
    if math.isnan(count):
        print(f"{c:.3f}        NaN")
        continue
    if prev is not None:
        diff = count - prev
        status = "OK" if diff >= -0.001 else "DROP!"
    else:
        diff = 0
        status = ""
    print(f"{c:<12.3f} {count:<12.4f} {diff:<+12.4f} {status}")
    prev = count
