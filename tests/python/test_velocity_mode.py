#!/usr/bin/env python3
"""Tests for Grid/Romberg velocity interpolation mode."""

import math

import disper26py as dp


def make_vti(rho, vph, vpv, vsh, vsv, eta):
    p = dp.PhysicalParameters()
    p.rho = rho
    p.c11 = rho * vph * vph
    p.c33 = rho * vpv * vpv
    p.c44 = rho * vsv * vsv
    p.c66 = rho * vsh * vsh
    p.c13 = eta * (p.c11 - 2.0 * p.c44)
    return p


def assert_close_tuple(left, right, tol=1e-10):
    for idx, (a, b) in enumerate(zip(left[:3], right[:3])):
        if math.isnan(a) and math.isnan(b):
            continue
        assert abs(a - b) < tol, f"component {idx}: {a} vs {b}"


def find_nearest_root(layers, thicknesses, halfspace, period, wave_type, c_guess):
    omega = 2.0 * math.pi / period
    roots = dp.search_roots(
        layers,
        thicknesses,
        halfspace,
        omega,
        max(0.1, 0.8 * c_guess),
        1.2 * c_guess,
        100,
        wave_type,
    )
    assert roots, "manual refined model did not produce a root"
    return min(roots, key=lambda c: abs(c - c_guess))


def manual_level0_iso(depths, rhos, vps, vss, is_disc, period, wave_type, c_guess):
    layers = []
    thicknesses = []
    for i in range(len(depths) - 1):
        if is_disc[i]:
            continue
        h = depths[i + 1] - depths[i]
        if h < 1e-12:
            continue
        rho = 0.5 * (rhos[i] + rhos[i + 1])
        vp = 0.5 * (vps[i] + vps[i + 1])
        vs = 0.5 * (vss[i] + vss[i + 1])
        layers.append(dp.create_isotropic_parameters(vs, vp, rho))
        thicknesses.append(h)
    halfspace = dp.create_isotropic_parameters(vss[-1], vps[-1], rhos[-1])
    c = find_nearest_root(layers, thicknesses, halfspace, period, wave_type, c_guess)
    omega = 2.0 * math.pi / period
    u = dp.calculate_group_velocity(layers, thicknesses, halfspace, omega, c, wave_type)
    return c, u


def manual_level0_vti(depths, rhos, vphs, vpvs, vshs, vsvs, etas, is_disc, period, wave_type, c_guess):
    layers = []
    thicknesses = []
    for i in range(len(depths) - 1):
        if is_disc[i]:
            continue
        h = depths[i + 1] - depths[i]
        if h < 1e-12:
            continue
        layers.append(
            make_vti(
                0.5 * (rhos[i] + rhos[i + 1]),
                0.5 * (vphs[i] + vphs[i + 1]),
                0.5 * (vpvs[i] + vpvs[i + 1]),
                0.5 * (vshs[i] + vshs[i + 1]),
                0.5 * (vsvs[i] + vsvs[i + 1]),
                0.5 * (etas[i] + etas[i + 1]),
            )
        )
        thicknesses.append(h)
    halfspace = make_vti(rhos[-1], vphs[-1], vpvs[-1], vshs[-1], vsvs[-1], etas[-1])
    c = find_nearest_root(layers, thicknesses, halfspace, period, wave_type, c_guess)
    omega = 2.0 * math.pi / period
    u = dp.calculate_group_velocity(layers, thicknesses, halfspace, omega, c, wave_type)
    return c, u


def kernel_value(kernels, node_index, parameter_name, attr="d_c"):
    for kernel in kernels:
        if kernel.layer_index == node_index and kernel.parameter_name == parameter_name:
            return getattr(kernel, attr)
    raise AssertionError(f"missing kernel {parameter_name}[{node_index}]")


def rel_close(actual, expected, rel_tol=2e-3, abs_tol=2e-5):
    scale = max(abs(actual), abs(expected), 1.0)
    assert abs(actual - expected) <= max(abs_tol, rel_tol * scale), (
        f"actual={actual:.12g}, expected={expected:.12g}, diff={abs(actual - expected):.3e}"
    )


def test_velocity_iso_constant_matches_modulus():
    depths = [0.0, 10.0]
    is_disc = [False, False]
    rhos = [2.7, 2.7]
    vps = [6.0, 6.0]
    vss = [3.5, 3.5]
    params = [dp.create_isotropic_parameters(3.5, 6.0, 2.7) for _ in depths]

    mod = dp.compute_dispersion_romberg(depths, params, is_disc, 20.0, "Love", 3.3, 5, 1e-12)
    vel = dp.compute_dispersion_romberg_velocity_iso(depths, rhos, vps, vss, is_disc, 20.0, "Love", 3.3, 5, 1e-12)

    assert_close_tuple(mod, vel)


def test_velocity_vti_constant_matches_modulus():
    depths = [0.0, 10.0]
    is_disc = [False, False]
    rhos = [2.7, 2.7]
    vphs = [6.2, 6.2]
    vpvs = [6.0, 6.0]
    vshs = [3.7, 3.7]
    vsvs = [3.5, 3.5]
    etas = [0.95, 0.95]
    params = [make_vti(2.7, 6.2, 6.0, 3.7, 3.5, 0.95) for _ in depths]

    mod = dp.compute_dispersion_romberg(depths, params, is_disc, 20.0, "Rayleigh", 3.2, 5, 1e-12)
    vel = dp.compute_dispersion_romberg_velocity_vti(
        depths, rhos, vphs, vpvs, vshs, vsvs, etas, is_disc, 20.0, "Rayleigh", 3.2, 5, 1e-12
    )

    assert_close_tuple(mod, vel)


def test_velocity_gradient_runs():
    depths = [0.0, 10.0, 20.0]
    is_disc = [False, False, False]

    iso = dp.compute_dispersion_romberg_velocity_iso(
        depths,
        [2.6, 2.7, 2.8],
        [5.8, 6.1, 6.4],
        [3.3, 3.5, 3.7],
        is_disc,
        20.0,
        "Love",
        3.4,
        5,
        1e-12,
    )
    assert 3.0 < iso[0] < 4.5

    vti = dp.compute_dispersion_romberg_velocity_vti(
        depths,
        [2.6, 2.7, 2.8],
        [5.9, 6.2, 6.5],
        [5.8, 6.1, 6.4],
        [3.45, 3.65, 3.85],
        [3.3, 3.5, 3.7],
        [0.95, 1.0, 1.05],
        is_disc,
        20.0,
        "Rayleigh",
        3.3,
        5,
        1e-12,
    )
    assert 2.8 < vti[0] < 4.5


def test_velocity_iso_level0_uses_velocity_midpoints():
    depths = [0.0, 8.0, 20.0]
    is_disc = [False, False, False]
    rhos = [2.5, 2.7, 2.9]
    vps = [5.6, 6.0, 6.5]
    vss = [3.1, 3.4, 3.8]
    period = 18.0
    c_guess = 3.2

    vel = dp.compute_dispersion_romberg_velocity_iso(
        depths, rhos, vps, vss, is_disc, period, "Love", c_guess, 0, 1e-12
    )
    manual_c, manual_u = manual_level0_iso(depths, rhos, vps, vss, is_disc, period, "Love", c_guess)

    assert abs(vel[0] - manual_c) < 1e-10
    assert abs(vel[1] - manual_u) < 1e-10


def test_velocity_vti_level0_uses_velocity_midpoints():
    depths = [0.0, 9.0, 21.0]
    is_disc = [False, False, False]
    rhos = [2.55, 2.7, 2.85]
    vphs = [5.9, 6.2, 6.6]
    vpvs = [5.7, 6.0, 6.4]
    vshs = [3.35, 3.6, 3.9]
    vsvs = [3.15, 3.4, 3.75]
    etas = [0.92, 0.98, 1.04]
    period = 20.0
    c_guess = 3.25

    vel = dp.compute_dispersion_romberg_velocity_vti(
        depths, rhos, vphs, vpvs, vshs, vsvs, etas, is_disc, period, "Rayleigh", c_guess, 0, 1e-12
    )
    manual_c, manual_u = manual_level0_vti(
        depths, rhos, vphs, vpvs, vshs, vsvs, etas, is_disc, period, "Rayleigh", c_guess
    )

    assert abs(vel[0] - manual_c) < 1e-10
    assert abs(vel[1] - manual_u) < 1e-10


def test_velocity_iso_phase_kernel_matches_finite_difference():
    depths = [0.0, 10.0, 20.0]
    is_disc = [False, False, False]
    rhos = [2.6, 2.7, 2.8]
    vps = [5.8, 6.1, 6.4]
    vss = [3.3, 3.5, 3.7]
    period = 20.0
    c_guess = 3.4

    result = dp.compute_dispersion_romberg_velocity_iso_with_kernels(
        depths, rhos, vps, vss, is_disc, period, "Love", c_guess, 5, 1e-12, True, False, False
    )
    c0, kernels = result[0], result[5]

    delta = 1e-4
    vss_plus = list(vss)
    vss_minus = list(vss)
    vss_plus[1] += delta
    vss_minus[1] -= delta
    c_plus = dp.compute_dispersion_romberg_velocity_iso(
        depths, rhos, vps, vss_plus, is_disc, period, "Love", c0, 5, 1e-12
    )[0]
    c_minus = dp.compute_dispersion_romberg_velocity_iso(
        depths, rhos, vps, vss_minus, is_disc, period, "Love", c0, 5, 1e-12
    )[0]
    fd = (c_plus - c_minus) / (2.0 * delta)

    rel_close(kernel_value(kernels, 1, "vs"), fd)
    assert kernel_value(kernels, 1, "vp") == 0.0


def test_velocity_vti_love_phase_kernel_matches_finite_difference():
    depths = [0.0, 10.0, 20.0]
    is_disc = [False, False, False]
    rhos = [2.6, 2.7, 2.8]
    vphs = [5.9, 6.2, 6.5]
    vpvs = [5.8, 6.1, 6.4]
    vshs = [3.45, 3.65, 3.85]
    vsvs = [3.3, 3.5, 3.7]
    etas = [0.95, 1.0, 1.05]
    period = 20.0
    c_guess = 3.5

    result = dp.compute_dispersion_romberg_velocity_vti_with_kernels(
        depths, rhos, vphs, vpvs, vshs, vsvs, etas, is_disc, period, "Love", c_guess, 5, 1e-12, True, False, False
    )
    c0, kernels = result[0], result[5]

    delta = 1e-4
    vshs_plus = list(vshs)
    vshs_minus = list(vshs)
    vshs_plus[1] += delta
    vshs_minus[1] -= delta
    c_plus = dp.compute_dispersion_romberg_velocity_vti(
        depths, rhos, vphs, vpvs, vshs_plus, vsvs, etas, is_disc, period, "Love", c0, 5, 1e-12
    )[0]
    c_minus = dp.compute_dispersion_romberg_velocity_vti(
        depths, rhos, vphs, vpvs, vshs_minus, vsvs, etas, is_disc, period, "Love", c0, 5, 1e-12
    )[0]
    fd = (c_plus - c_minus) / (2.0 * delta)

    rel_close(kernel_value(kernels, 1, "Vsh"), fd)
    for name in ("Vph", "Vpv", "eta"):
        assert abs(kernel_value(kernels, 1, name)) < 1e-8


if __name__ == "__main__":
    test_velocity_iso_constant_matches_modulus()
    test_velocity_vti_constant_matches_modulus()
    test_velocity_gradient_runs()
    test_velocity_iso_level0_uses_velocity_midpoints()
    test_velocity_vti_level0_uses_velocity_midpoints()
    test_velocity_iso_phase_kernel_matches_finite_difference()
    test_velocity_vti_love_phase_kernel_matches_finite_difference()
