#!/usr/bin/env python3
"""Analytical Rayleigh layer integrals for models with top liquid layers.

A liquid layer carries only [U, Tzz]; its quadratic moments follow in closed
form from the state at the layer top (energy_entire.hpp, fluid_moments). The tests compare these integrals with independent
non-gravitating references:

  kernels -- Full AD of the secular function gives the exact derivative of the
  layered-model phase velocity. The analytical variational kernels must agree
  to roundoff when the liquid carries no gravity.

  group velocity and energy balance -- U = I3/(c I1) against the AD group
  velocity, and I2 = omega^2 I1.

The derivative API rejects a Cowling reference for either quadrature.
The relative tolerance of 1e-9 covers root and propagation roundoff, as in
test_love_energy_integrals.py.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp  # noqa: E402


EXACT_RTOL = 1e-9
HALFSPACE = dp.create_isotropic_parameters(4.5, 8.1, 3.35)


def iso(vs, vp, rho):
    return dp.create_isotropic_parameters(vs, vp, rho)


def fluid(vp, rho, gravity=0.0):
    return dp.create_fluid_parameters(vp, rho, gravity, 1.0, 0.0)


def models(gravity=0.0):
    solid = [iso(3.2, 5.8, 2.6), iso(3.6, 6.3, 2.8)]
    return [
        ("ocean 4km", [fluid(1.5, 1.03, gravity)] + solid, [4.0, 8.0, 20.0]),
        ("ocean 1km", [fluid(1.5, 1.03, gravity)] + solid, [1.0, 8.0, 20.0]),
        ("ocean 2 layers", [fluid(1.48, 1.02, gravity), fluid(1.52, 1.04, gravity)]
         + solid, [1.5, 2.5, 8.0, 20.0]),
    ]


def root(layers, thick, omega):
    return dp.search_roots(layers, thick, HALFSPACE, omega, 0.5, 4.4, 6000,
                           "Rayleigh")[0]


def kernels(result):
    return {(k.layer_index, k.parameter_name): k.d_c for k in result}


def weakform(layers, thick, omega, c, analytical, sublayers=5):
    return kernels(dp.compute_sensitivity_kernels_weakform(
        layers, thick, HALFSPACE, omega, c, wave_type="Rayleigh",
        sensitivity_mode="Isotropic", num_sub_layers=sublayers,
        use_analytical=analytical))


def scaled_difference(a, b):
    scale = max(abs(v) for v in b.values())
    return max(abs(a[key] - b[key]) for key in b) / scale


def test_kernels_match_full_ad_without_gravity():
    for name, layers, thick in models():
        for period in (20.0, 7.0, 3.0):
            omega = 2.0 * math.pi / period
            c = root(layers, thick, omega)
            reference = kernels(dp.compute_sensitivity_kernels_full_ad(
                layers, thick, HALFSPACE, omega, c, "Rayleigh", "Isotropic"))
            error = scaled_difference(weakform(layers, thick, omega, c, True),
                                      reference)
            assert error < EXACT_RTOL, f"{name} T={period}: {error:.2e}"
            print(f"OK {name} T={period}: kernels vs Full AD {error:.1e}")


def test_energy_integrals_match_group_velocity_and_balance():
    for name, layers, thick in models():
        for period in (20.0, 7.0, 3.0):
            omega = 2.0 * math.pi / period
            c = root(layers, thick, omega)
            trace = dp.debug_kernels_variational_rayleigh_layer_cake(
                layers, thick, HALFSPACE, omega, c, num_sub_layers=1,
                use_analytical=True)
            group = dp.compute_group_velocity_ad(layers, thick, HALFSPACE,
                                                 omega, c, "Rayleigh")
            u_error = abs(trace["I3"] / (c * trace["I1"]) / group - 1.0)
            balance = abs(trace["I2"] / (omega * omega * trace["I1"]) - 1.0)
            assert u_error < EXACT_RTOL, f"{name} T={period}: U {u_error:.2e}"
            assert balance < EXACT_RTOL, f"{name} T={period}: balance {balance:.2e}"
            print(f"OK {name} T={period}: U {u_error:.1e}, balance {balance:.1e}")


def test_gravitating_material_kernels_rejected():
    import pytest
    for _, layers, thick in models(gravity=9.8e-3):
        omega = 2.0 * math.pi / 20.0
        c = root(layers, thick, omega)
        for analytical in (True, False):
            with pytest.raises(ValueError, match="Cowling gravity"):
                weakform(layers, thick, omega, c, analytical)


if __name__ == "__main__":
    test_kernels_match_full_ad_without_gravity()
    test_energy_integrals_match_group_velocity_and_balance()
    test_gravitating_material_kernels_rejected()
    print("OK")
