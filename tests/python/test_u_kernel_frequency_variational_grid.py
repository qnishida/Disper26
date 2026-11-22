#!/usr/bin/env python3
"""Frequency-AD variational GRID group-velocity kernel regression tests."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp


D_U_RTOL = 1.0e-5


def smooth_model(n=4):
    depths = np.linspace(0.0, 30.0 if n == 4 else 100.0, n).tolist()
    params = []
    for i, depth in enumerate(depths):
        scale = 1.0 + 0.5 * i / max(1, n - 1)
        params.append(dp.create_isotropic_parameters(
            2.5 * scale, 4.3 * scale, 2.4 + 0.4 * i / max(1, n - 1)))
    return depths, params, [False] * n


def discontinuous_model():
    depths = [0.0, 10.0, 10.0, 40.0]
    triples = [(2.5, 4.3, 2.4), (2.7, 4.6, 2.5),
               (3.5, 6.2, 2.8), (4.0, 7.0, 3.0)]
    return depths, [dp.create_isotropic_parameters(*v) for v in triples], \
        [False, True, False, False]


def normalized_du_error(reference, candidate):
    scale = max(abs(k.d_U) for k in reference[5])
    return max(abs(a.d_U - b.d_U)
               for a, b in zip(reference[5], candidate[5])) / scale


def compare_main(depths, params, disc, wave, sensitivity_mode):
    kwargs = dict(
        depths=depths, params=params, is_discontinuity=disc,
        period=20.0, wave_type=wave, c_guess=3.0,
        sensitivity_mode=sensitivity_mode, max_steps=5, tolerance=1.0e-10,
        compute_c=True, want_kernel_u=True, want_kernel_hv=False,
        kernel_extrapolation=True)
    reference = dp.compute_kernels_ad_grid(
        **kwargs, u_kernel_mode="variational")
    candidate = dp.compute_kernels_ad_grid(
        **kwargs, u_kernel_mode="variational_frequency_ad")
    for actual, expected in zip(candidate[:5], reference[:5]):
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == expected
    assert [k.parameter_name for k in candidate[5]] == \
        [k.parameter_name for k in reference[5]]
    assert [k.layer_index for k in candidate[5]] == \
        [k.layer_index for k in reference[5]]
    assert all(a.d_c == b.d_c for a, b in zip(candidate[5], reference[5]))
    error = normalized_du_error(reference, candidate)
    assert error < D_U_RTOL, (wave, sensitivity_mode, error)
    return reference, candidate, error


def test_solid_material_and_depth_kernels():
    for model in (smooth_model(), smooth_model(20), discontinuous_model()):
        for wave in ("Rayleigh", "Love"):
            for sensitivity_mode in ("Isotropic", "VTI"):
                _, candidate, error = compare_main(
                    *model, wave, sensitivity_mode)
                assert all(math.isfinite(k.d_U) for k in candidate[5])
                print("OK", len(model[0]), wave, sensitivity_mode,
                      f"max scaled dU error={error:.3e}")


def test_love_interface_depth_kernel_against_group_velocity_difference():
    depths, params, disc = discontinuous_model()
    result = dp.compute_kernels_ad_grid(
        depths, params, disc, period=20.0, wave_type="Love", c_guess=3.0,
        sensitivity_mode="Isotropic", max_steps=5, tolerance=1.0e-10,
        want_kernel_u=True, want_kernel_hv=False)
    depth_kernel, = [k for k in result[5] if k.parameter_name == "Depth"]

    # Shift both nodes at the interface. Steps of 0.0001 to 0.001 km give
    # stable centered differences on this model at the same refinement level.
    step_km = 1.0e-4
    group_velocities = []
    for shift in (-step_km, step_km):
        shifted = [depths[0], depths[1] + shift, depths[2] + shift, depths[3]]
        _, group, _, level, _ = dp.compute_dispersion_romberg(
            shifted, params, disc, 20.0, "Love", result[0], 5, 1.0e-10)
        assert level == result[3]
        group_velocities.append(group)
    centered = (group_velocities[1] - group_velocities[0]) / (2.0 * step_km)
    # The existing Romberg group-velocity finite-difference checks use 1e-6.
    assert math.isclose(depth_kernel.d_U, centered, rel_tol=1e-6)


def test_velocity_parameterizations():
    depths, _, disc = smooth_model()
    rhos = [2.4 + 0.01 * z for z in depths]
    vps = [4.3 + 0.08 * z for z in depths]
    vss = [2.5 + 0.05 * z for z in depths]
    for wave in ("Rayleigh", "Love"):
        args = (depths, rhos, vps, vss, disc, 20.0, wave, 3.0, 5, 1.0e-10,
                True, True, False)
        reference = dp.compute_kernels_ad_grid_velocity_iso(
            *args, u_kernel_mode="variational")
        candidate = dp.compute_kernels_ad_grid_velocity_iso(
            *args, u_kernel_mode="variational_frequency_ad")
        assert normalized_du_error(reference, candidate) < D_U_RTOL

    vph = [v + 0.2 for v in vps]
    vpv = vps
    vsh = [v + 0.2 for v in vss]
    vsv = vss
    eta = [0.9 + 0.05 * i for i in range(len(depths))]
    for wave in ("Rayleigh", "Love"):
        args = (depths, rhos, vph, vpv, vsh, vsv, eta, disc,
                20.0, wave, 3.0, 5, 1.0e-10, True, True, False)
        reference = dp.compute_kernels_ad_grid_velocity_vti(
            *args, u_kernel_mode="variational")
        candidate = dp.compute_kernels_ad_grid_velocity_vti(
            *args, u_kernel_mode="variational_frequency_ad")
        assert normalized_du_error(reference, candidate) < D_U_RTOL


def test_frequency_identity_against_centered_difference():
    depths, params, disc = smooth_model()
    frequency = 0.05
    omega = 2.0 * math.pi * frequency
    result = dp.compute_kernels_ad_grid(
        depths, params, disc, 1.0 / frequency, "Rayleigh", 3.0, "VTI",
        5, 1.0e-10, True, True, False,
        u_kernel_mode="variational_frequency_ad")
    level = result[3]
    eps = omega * 1.0e-4
    samples = []
    for shifted_omega in (omega - eps, omega + eps):
        shifted_frequency = shifted_omega / (2.0 * math.pi)
        dispersion = dp.compute_dispersion_romberg(
            depths, params, disc, 1.0 / shifted_frequency, "Rayleigh",
            result[0], level, -1.0)
        gradient = dp.compute_kernels_variational_grid(
            depths, params, disc, shifted_frequency, dispersion[0],
            "Rayleigh", level, -1.0)["gradient_node"]
        samples.append((shifted_omega, dispersion[0], gradient))

    # API VTI order [A,C,F,L,N,Rho] versus variational order
    # [Rho,A,C,F,L,N].
    order = (1, 2, 3, 4, 5, 0)
    predicted = []
    for index in range(len(result[5])):
        node, component = divmod(index, 6)
        q_index = node * 6 + order[component]
        minus = samples[0][0] * samples[0][2][q_index] / samples[0][1] ** 2
        plus = samples[1][0] * samples[1][2][q_index] / samples[1][1] ** 2
        predicted.append(result[1] ** 2 * (plus - minus) / (2.0 * eps))
    scale = max(abs(k.d_U) for k in result[5])
    error = max(abs(value - kernel.d_U)
                for value, kernel in zip(predicted, result[5])) / scale
    assert error < 1.0e-6, error


def test_fluid_fallback_and_target_switch():
    ocean = dp.create_fluid_parameters(1.5, 1.03)
    crust_top = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    crust_bottom = dp.create_isotropic_parameters(3.5, 6.2, 2.8)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    depths = [0.0, 3.0, 3.0, 23.0, 23.0]
    params = [ocean, ocean, crust_top, crust_bottom, halfspace]
    disc = [False, True, False, True, False]
    roots = dp.search_roots(
        [ocean, crust_top], [3.0, 20.0], halfspace,
        2.0 * math.pi / 20.0, 0.5, 5.0, 500, "Rayleigh")
    reference = dp.compute_kernels_ad_grid(
        depths, params, disc, 20.0, "Rayleigh", roots[0], "Isotropic",
        5, 1.0e-10, False, True, False, u_kernel_mode="full_ad")
    candidate = dp.compute_kernels_ad_grid(
        depths, params, disc, 20.0, "Rayleigh", roots[0], "Isotropic",
        5, 1.0e-10, False, True, False,
        u_kernel_mode="variational_frequency_ad")
    assert all(a.d_U == b.d_U for a, b in zip(reference[5], candidate[5]))
    assert all(k.d_c == 0.0 and k.d_HV == 0.0 for k in candidate[5])


def test_frequency_mode_is_default():
    depths, params, disc = smooth_model()
    args = (depths, params, disc, 20.0, "Rayleigh", 3.0, "Isotropic",
            5, 1.0e-10, False, True, False)
    default = dp.compute_kernels_ad_grid(*args)
    explicit = dp.compute_kernels_ad_grid(
        *args, u_kernel_mode="variational_frequency_ad")
    assert all(a.d_U == b.d_U for a, b in zip(default[5], explicit[5]))


if __name__ == "__main__":
    test_solid_material_and_depth_kernels()
    test_velocity_parameterizations()
    test_frequency_identity_against_centered_difference()
    test_fluid_fallback_and_target_switch()
    test_frequency_mode_is_default()
    print("All frequency-variational GRID U-kernel tests passed.")
