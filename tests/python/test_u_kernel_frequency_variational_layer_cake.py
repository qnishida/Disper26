#!/usr/bin/env python3
"""Frequency-AD variational LAYER_CAKE group-kernel regression tests."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp


D_U_RTOL = 1.0e-5


def solid_model():
    layers = [
        dp.create_isotropic_parameters(3.2, 5.8, 2.6),
        dp.create_isotropic_parameters(3.7, 6.5, 2.9),
    ]
    thicknesses = [10.0, 20.0]
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    return layers, thicknesses, halfspace


def phase_root(layers, thicknesses, halfspace, omega, wave, hint=None):
    if hint is None:
        lower, upper = (1.0, 5.0)
    else:
        lower, upper = (0.95 * hint, 1.05 * hint)
    roots = dp.search_roots(
        layers, thicknesses, halfspace, omega,
        lower, upper, 1500, wave)
    assert roots
    return min(roots, key=lambda value: abs(value - hint)) if hint else roots[0]


def compute(mode, wave, sensitivity_mode="Isotropic", thickness=True,
            analytical=False, **targets):
    layers, thicknesses, halfspace = solid_model()
    omega = 2.0 * math.pi / 20.0
    c = phase_root(layers, thicknesses, halfspace, omega, wave)
    kernels = dp.compute_kernels_variational_layer_cake(
        layers, thicknesses, halfspace, omega, c, wave,
        sensitivity_mode, thickness, 10, analytical,
        u_kernel_mode=mode, **targets)
    return layers, thicknesses, halfspace, omega, c, kernels


def normalized_du_error(reference, candidate):
    scale = max(abs(kernel.d_U) for kernel in reference)
    return max(abs(a.d_U - b.d_U)
               for a, b in zip(reference, candidate)) / scale


def compare_paths(wave, sensitivity_mode, analytical):
    old = compute("variational", wave, sensitivity_mode,
                  thickness=True, analytical=analytical)[-1]
    new = compute("variational_frequency_ad", wave, sensitivity_mode,
                  thickness=True, analytical=analytical)[-1]
    assert [(k.parameter_name, k.layer_index) for k in old] == [
        (k.parameter_name, k.layer_index) for k in new]
    assert all(a.d_c == b.d_c for a, b in zip(old, new))
    for a, b in zip(old, new):
        if math.isnan(a.d_HV):
            assert math.isnan(b.d_HV)
        else:
            assert a.d_HV == b.d_HV
    error = normalized_du_error(old, new)
    assert error < D_U_RTOL, (wave, sensitivity_mode, analytical, error)
    return error


def test_solid_material_and_thickness_kernels():
    cases = [
        ("Rayleigh", False),  # explicit legacy fallback: trapezoidal phase K
        ("Rayleigh", True),
        ("Love", False),
    ]
    for wave, analytical in cases:
        for sensitivity_mode in ("Isotropic", "VTI"):
            error = compare_paths(wave, sensitivity_mode, analytical)
            print("OK", wave, sensitivity_mode, "analytical", analytical,
                  f"max scaled dU error={error:.3e}")


def test_against_full_ad_reference():
    for wave, analytical in (("Rayleigh", True), ("Love", False)):
        layers, thicknesses, halfspace, omega, c, new = compute(
            "variational_frequency_ad", wave, "VTI",
            thickness=True, analytical=analytical)
        reference = dp.compute_kernels_ad_layer_cake(
            layers, thicknesses, halfspace, omega, c,
            wave, "VTI", True)
        assert normalized_du_error(reference, new) < 1.0e-8


def test_centered_frequency_identity():
    for wave, analytical in (("Rayleigh", True), ("Love", False)):
        layers, thicknesses, halfspace, omega, c, kernels = compute(
            "variational_frequency_ad", wave, "VTI",
            thickness=True, analytical=analytical,
            want_kernel_hv=False)
        U = dp.compute_group_velocity_ad(
            layers, thicknesses, halfspace, omega, c, wave)
        eps = omega * 1.0e-4
        samples = []
        for shifted_omega in (omega - eps, omega + eps):
            shifted_c = phase_root(
                layers, thicknesses, halfspace, shifted_omega, wave, c)
            shifted = dp.compute_kernels_variational_layer_cake(
                layers, thicknesses, halfspace, shifted_omega, shifted_c,
                wave, "VTI", True, 10, analytical,
                u_kernel_mode="variational",
                want_kernel_u=False, want_kernel_hv=False)
            samples.append((shifted_omega, shifted_c, shifted))

        predicted = []
        for index in range(len(kernels)):
            minus = (samples[0][0] * samples[0][2][index].d_c /
                     samples[0][1] ** 2)
            plus = (samples[1][0] * samples[1][2][index].d_c /
                    samples[1][1] ** 2)
            predicted.append(U * U * (plus - minus) / (2.0 * eps))
        scale = max(abs(kernel.d_U) for kernel in kernels)
        error = max(abs(value - kernel.d_U)
                    for value, kernel in zip(predicted, kernels)) / scale
        assert error < 1.0e-6, (wave, error)


def test_fluid_and_trapezoidal_fallbacks_are_exact():
    layers = [
        dp.create_fluid_parameters(1.5, 1.03),
        dp.create_isotropic_parameters(3.2, 5.8, 2.6),
    ]
    thicknesses = [4.0, 20.0]
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    omega = 2.0 * math.pi / 20.0
    c = phase_root(layers, thicknesses, halfspace, omega, "Rayleigh")
    args = (layers, thicknesses, halfspace, omega, c,
            "Rayleigh", "Isotropic", True, 5, False)
    old = dp.compute_kernels_variational_layer_cake(
        *args, u_kernel_mode="variational")
    new = dp.compute_kernels_variational_layer_cake(
        *args, u_kernel_mode="variational_frequency_ad")
    assert all(a.d_c == b.d_c and a.d_U == b.d_U and a.d_HV == b.d_HV
               for a, b in zip(old, new))


def test_defaults_targets_and_invalid_mode():
    explicit = compute("variational_frequency_ad", "Love")[-1]
    layers, thicknesses, halfspace, omega, c, _ = compute(
        "variational_frequency_ad", "Love")
    default = dp.compute_kernels_variational_layer_cake(
        layers, thicknesses, halfspace, omega, c, "Love",
        "Isotropic", True, 10, False)
    assert all(a.d_U == b.d_U for a, b in zip(default, explicit))

    selected = dp.compute_kernels_variational_layer_cake(
        layers, thicknesses, halfspace, omega, c, "Love",
        "Isotropic", True, 10, False,
        compute_c=False, want_kernel_u=True, want_kernel_hv=False)
    assert all(k.d_c == 0.0 and k.d_HV == 0.0 for k in selected)
    omitted = dp.compute_kernels_variational_layer_cake(
        layers, thicknesses, halfspace, omega, c, "Love",
        "Isotropic", True, 10, False,
        want_kernel_u=False, want_kernel_hv=False)
    assert all(k.d_U == 0.0 and k.d_HV == 0.0 for k in omitted)

    try:
        dp.compute_kernels_variational_layer_cake(
            layers, thicknesses, halfspace, omega, c,
            u_kernel_mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid LAYER_CAKE u_kernel_mode must raise")


if __name__ == "__main__":
    test_solid_material_and_thickness_kernels()
    test_against_full_ad_reference()
    test_centered_frequency_identity()
    test_fluid_and_trapezoidal_fallbacks_are_exact()
    test_defaults_targets_and_invalid_mode()
    print("All frequency-variational LAYER_CAKE U-kernel tests passed.")
