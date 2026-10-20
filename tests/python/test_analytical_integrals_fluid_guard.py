#!/usr/bin/env python3
"""The analytical Rayleigh energy integrals must refuse fluid models, not
return NaN.

They assume a solid layer -- the form divides by c44 and uses the P-SV
eigenfunction shape, and a water layer has c44 = 0.

compute_sensitivity_kernels_weakform(use_analytical=True) was already covered:
the variational-gradient entry point has thrown for fluid since it was written.
compute_rayleigh_energy_integrals_analytical was not, and returned (nan, nan,
nan) with no indication of why. Silent NaN is the worst of the outcomes -- a
smoke test that only checks the call succeeds will pass, and the value
propagates into whatever consumes it.

The guard now sits inside both computeRayleighEnergyIntegrals_Analytical
overloads rather than at the entry points, so a future caller cannot route
around it. This test covers both ways in, including the one that was already
safe, so that neither regresses.

Fluid models are not left without an option: the trapezoidal integrals and
compute_sensitivity_kernels_full_ad both handle them. See
docs/fluid_gravity_status.md for which path to use when.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp  # noqa: E402


def iso(vs, vp, rho):
    return dp.create_isotropic_parameters(vs, vp, rho)


HALFSPACE = iso(4.5, 8.1, 3.35)
SOLID = ([iso(3.2, 5.8, 2.6), iso(3.6, 6.3, 2.8)], [10.0, 20.0], HALFSPACE)
OCEAN = ([dp.create_fluid_parameters(1.5, 1.03, 0.0), iso(3.2, 5.8, 2.6)],
         [4.0, 20.0], HALFSPACE)
OMEGA = 2.0 * math.pi / 20.0


def setup(model):
    layers, thick, halfspace = model
    c = dp.search_roots(layers, thick, halfspace, OMEGA, 0.5, 4.4, 6000, "Rayleigh")[0]
    efs = dp.compute_rayleigh_eigenfunctions(layers, thick, halfspace, OMEGA, c, 10)
    return layers, thick, halfspace, c, efs


def has_nan(values):
    return any(v != v for v in values)


def test_analytical_paths_reject_fluid():
    layers, thick, halfspace, c, efs = setup(OCEAN)

    for label, call in (
        ("compute_rayleigh_energy_integrals_analytical",
         lambda: dp.compute_rayleigh_energy_integrals_analytical(
             efs, layers, thick, halfspace, OMEGA, c)),
        ("compute_sensitivity_kernels_weakform(use_analytical=True)",
         lambda: dp.compute_sensitivity_kernels_weakform(
             layers, thick, halfspace, OMEGA, c, wave_type="Rayleigh",
             sensitivity_mode="Isotropic", num_sub_layers=5, use_analytical=True)),
    ):
        try:
            call()
        except RuntimeError as exc:
            assert "fluid" in str(exc), f"{label}: unhelpful message {exc!r}"
            print(f"OK {label} raises: {str(exc)[:60]}...")
        else:
            raise AssertionError(f"{label} accepted a fluid model instead of raising")


def test_fluid_still_has_working_paths():
    """The guard must not leave fluid models without a route to kernels."""
    layers, thick, halfspace, c, efs = setup(OCEAN)

    integrals = dp.compute_energy_integrals(efs, layers, thick,
                                            halfspace, OMEGA, c, "Rayleigh")
    assert not has_nan(integrals), f"trapezoidal integrals gave NaN: {integrals}"

    for label, call in (
        ("weakform trapezoidal",
         lambda: dp.compute_sensitivity_kernels_weakform(
             layers, thick, halfspace, OMEGA, c, wave_type="Rayleigh",
             sensitivity_mode="Isotropic", num_sub_layers=5, use_analytical=False)),
        ("full_ad",
         lambda: dp.compute_sensitivity_kernels_full_ad(
             layers, thick, halfspace, OMEGA, c, "Rayleigh", "Isotropic")),
    ):
        kernels = call()
        assert kernels, f"{label} returned nothing"
        assert not has_nan([k.d_c for k in kernels]), f"{label} gave NaN kernels"
        print(f"OK {label}: {len(kernels)} kernels, no NaN")


def test_solid_analytical_is_unaffected():
    """The guard is on fluid detection only; solid accuracy must be untouched.
    The analytical kernels are exact here, so this pins the value rather than a
    tolerance -- if the guard ever starts firing on solids this fails loudly."""
    layers, thick, halfspace, c, efs = setup(SOLID)

    integrals = dp.compute_rayleigh_energy_integrals_analytical(
        efs, layers, thick, halfspace, OMEGA, c)
    assert not has_nan(integrals), "solid analytical integrals gave NaN"

    analytic = dp.compute_sensitivity_kernels_weakform(
        layers, thick, halfspace, OMEGA, c, wave_type="Rayleigh",
        sensitivity_mode="Isotropic", num_sub_layers=5, use_analytical=True)
    full_ad = dp.compute_sensitivity_kernels_full_ad(
        layers, thick, halfspace, OMEGA, c, "Rayleigh", "Isotropic")

    def vs_kernel(kernels, layer):
        return [k.d_c for k in kernels
                if k.layer_index == layer and k.parameter_name == "Vs"][0]

    for layer in (0, 1):
        a, f = vs_kernel(analytic, layer), vs_kernel(full_ad, layer)
        rel = abs(a - f) / abs(f)
        assert rel < 1e-8, f"layer {layer}: analytic {a} vs full_ad {f} (rel {rel:.2e})"
        print(f"OK solid layer {layer}: analytic dc/dVs = {a:.12f} "
              f"(rel to full_ad {rel:.1e})")


if __name__ == "__main__":
    test_analytical_paths_reject_fluid()
    test_fluid_still_has_working_paths()
    test_solid_analytical_is_unaffected()
    print("OK")
