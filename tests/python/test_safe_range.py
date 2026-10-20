import disper26py
import numpy as np

l1 = disper26py.create_isotropic_parameters(1.0, 1.73, 2.0)
l2 = disper26py.create_isotropic_parameters(1.5, 2.6, 2.2)
l3 = disper26py.create_isotropic_parameters(2.2, 3.8, 2.5)
half_space = disper26py.create_isotropic_parameters(3.0, 5.2, 2.7)

layers = [l1, l2, l3]
thicknesses = [2.0, 5.0, 8.0]
omega = 2 * np.pi * 0.5

# Check monotonicity while avoiding exact layer boundaries
c_values = np.linspace(1.05, 2.95, 100)
prev = None
drops = 0
max_drop = 0
for c in c_values:
    count = disper26py.calculate_continuous_mode_count(layers, thicknesses, half_space, omega, c, 'Rayleigh')
    if prev is not None and not np.isnan(count) and not np.isnan(prev):
        if count < prev - 0.001:
            drops += 1
            max_drop = max(max_drop, prev - count)
    prev = count
print(f'Drops in [1.05, 2.95]: {drops}')
print(f'Max drop magnitude: {max_drop:.4f}')
print('Range at roots:')
print(f'  c=1.05: {disper26py.calculate_continuous_mode_count(layers, thicknesses, half_space, omega, 1.05, "Rayleigh"):.4f}')
print(f'  c=2.95: {disper26py.calculate_continuous_mode_count(layers, thicknesses, half_space, omega, 2.95, "Rayleigh"):.4f}')
