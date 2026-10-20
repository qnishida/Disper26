import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import disper26py

# Create Spherical Model (PREM-ish Crust/Mantle)
# R=6371km
radius = 6371.0

# Layer 1: Crust (Thickness 20km, Top=6371, Bot=6351)
l1 = disper26py.create_isotropic_parameters(3.2, 5.8, 2.6)
# Layer 2: Mantle Top (Thickness 50km, Top=6351, Bot=6301)
l2 = disper26py.create_isotropic_parameters(4.5, 8.1, 3.4)
# (Halfspace logic needs flat model, for EFT we just output flat layers list)

layers_spherical = [l1, l2]
spherical_thicknesses = [20.0, 50.0]

print("--- Original Spherical Layers ---")
print(f"L1: {l1} h={spherical_thicknesses[0]}")
print(f"L2: {l2} h={spherical_thicknesses[1]}")

# Transform
print("\n--- Transforming to Flat BA97 (p=-2) ---")
(flat_layers, flat_thick) = disper26py.earth_flattening_transformation_layers(
    layers_spherical, spherical_thicknesses, radius, -2.0
)

for i, (l, h) in enumerate(zip(flat_layers, flat_thick)):
    print(f"Flat L{i+1}: {l} h={h:.4f}")
    
    # Check scaling logic
    r_top = radius - sum(spherical_thicknesses[:i])
    r_bot = radius - sum(spherical_thicknesses[:i+1])
    r_mid = (r_top + r_bot) / 2.0
    
    scale_elastic = pow(r_mid / radius, 2.0)
    scale_rho = pow(r_mid / radius, 4.0)
    
    expected_vp = l.get_vp() # transformed vp
    original_vp = layers_spherical[i].get_vp()
    
    # In inverse BA97 mode, v_flat/v_spherical = sqrt((r/R)^(-p) / (r/R)^(2-p)) = R/r.
    ratio_v = expected_vp / original_vp
    print(f"  Vp Scaling: {ratio_v:.6f} vs Expected {radius / r_mid:.6f}")
    
    # Rho Scaling
    ratio_rho = l.rho / layers_spherical[i].rho
    print(f"  Rho Scaling: {ratio_rho:.6f} vs Expected {scale_rho:.6f}")
    print(f"  C44 Scaling: {l.c44 / layers_spherical[i].c44:.6f} vs Expected {scale_elastic:.6f}")
    
    # Thickness
    expected_h = radius * math.log(r_top/r_bot)
    print(f"  Thick: {h:.4f} vs Expected {expected_h:.4f}")
