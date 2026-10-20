#!/usr/bin/env python3
"""Verification for the isolated GRID frequency-curvature AD path."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))

import disper26py as dp  # noqa: E402


EXPECTED_KEYS = {
    "phase_velocity",
    "group_velocity",
    "group_velocity_omega_derivative",
    "wavenumber_omega_second_derivative",
    "dimensionless_phase_group_dispersion",
    "dimensionless_frequency_curvature",
    "accepted_level",
    "phase_velocity_relative_error",
    "group_velocity_omega_derivative_error",
    "maximum_root_residual",
    "status",
}


def build_solid_model():
    depths = [0.0, 10.0, 20.0, 30.0]
    params = []
    for depth in depths:
        vs = 2.5 + 0.05 * depth
        vp = 4.3 + 0.08 * depth
        rho = 2.4 + 0.01 * depth
        params.append(dp.create_isotropic_parameters(vs, vp, rho))
    return depths, params, [False] * len(depths)


def build_ocean_model():
    gravity = 0.00980665
    params = [
        dp.create_fluid_parameters(1.5, 1.03, gravity),
        dp.create_fluid_parameters(1.5, 1.03, gravity),
        dp.create_isotropic_parameters(3.2, 5.8, 2.6),
        dp.create_isotropic_parameters(3.5, 6.2, 2.8),
        dp.create_isotropic_parameters(4.0, 7.0, 3.2),
    ]
    return [0.0, 4.0, 4.0, 30.0, 60.0], params, [False, True, False, False, False]


def compute_at_omega(model, omega, wave_type, c_guess, max_steps):
    depths, params, is_discontinuity = model
    return dp.compute_frequency_curvature_ad_grid(
        depths=depths,
        params=params,
        is_discontinuity=is_discontinuity,
        period=2.0 * math.pi / omega,
        wave_type=wave_type,
        c_guess=c_guess,
        max_steps=max_steps,
        tolerance=1e-10,
    )


def five_point_estimates(model, omega, h, wave_type, c_guess, max_steps):
    values = {}
    for offset in (-2, -1, 0, 1, 2):
        values[offset] = compute_at_omega(
            model, omega + offset * h, wave_type, c_guess, max_steps
        )
        assert values[offset]["status"] == "ok"

    group_derivative = (
        values[-2]["group_velocity"]
        - 8.0 * values[-1]["group_velocity"]
        + 8.0 * values[1]["group_velocity"]
        - values[2]["group_velocity"]
    ) / (12.0 * h)

    wavenumbers = {
        offset: (omega + offset * h) / values[offset]["phase_velocity"]
        for offset in values
    }
    wavenumber_second_derivative = (
        -wavenumbers[2]
        + 16.0 * wavenumbers[1]
        - 30.0 * wavenumbers[0]
        + 16.0 * wavenumbers[-1]
        - wavenumbers[-2]
    ) / (12.0 * h * h)
    return group_derivative, wavenumber_second_derivative


def assert_within_fd_uncertainty(ad_value, fine_fd, medium_fd):
    # The difference of two successively refined five-point formulas is an
    # a-posteriori truncation/round-off estimate.  The small scale-aware floor
    # covers root and Romberg noise when that difference crosses zero.
    uncertainty = abs(fine_fd - medium_fd)
    floor = 2.0e-7 * max(1.0, abs(ad_value))
    assert abs(ad_value - fine_fd) <= 8.0 * uncertainty + floor, (
        ad_value,
        fine_fd,
        medium_fd,
        uncertainty,
    )


def test_fixed_mesh_ad_matches_root_resolved_five_point_fd():
    model = build_solid_model()
    omega = 2.0 * math.pi / 20.0
    center = compute_at_omega(model, omega, "Rayleigh", 3.0, max_steps=0)
    assert center["status"] == "ok"
    assert set(center) == EXPECTED_KEYS

    estimates = []
    for relative_step in (4.0e-3, 2.0e-3, 1.0e-3):
        estimates.append(
            five_point_estimates(
                model,
                omega,
                relative_step * omega,
                "Rayleigh",
                center["phase_velocity"],
                max_steps=0,
            )
        )

    medium, fine = estimates[-2], estimates[-1]
    assert_within_fd_uncertainty(
        center["group_velocity_omega_derivative"], fine[0], medium[0]
    )
    assert_within_fd_uncertainty(
        center["wavenumber_omega_second_derivative"], fine[1], medium[1]
    )


def test_romberg_result_identities_and_existing_api_isolation():
    model = build_solid_model()
    depths, params, is_discontinuity = model
    old_before = dp.compute_dispersion_romberg(
        depths, params, is_discontinuity, 20.0, "Rayleigh", 3.0, 5, 1e-10
    )
    result = compute_at_omega(
        model, 2.0 * math.pi / 20.0, "Rayleigh", 3.0, max_steps=5
    )
    old_after = dp.compute_dispersion_romberg(
        depths, params, is_discontinuity, 20.0, "Rayleigh", 3.0, 5, 1e-10
    )

    assert old_before == old_after
    assert len(old_after) == 5
    assert result["status"] == "ok"
    assert result["accepted_level"] == 5
    assert result["maximum_root_residual"] < 0.1

    omega = 2.0 * math.pi / 20.0
    expected_k_second = -result["group_velocity_omega_derivative"] / (
        result["group_velocity"] ** 2
    )
    expected_d1 = abs(1.0 - result["phase_velocity"] / result["group_velocity"])
    expected_d2 = abs(
        omega
        * result["group_velocity_omega_derivative"]
        / result["group_velocity"]
    )
    assert math.isclose(
        result["wavenumber_omega_second_derivative"],
        expected_k_second,
        rel_tol=2e-15,
    )
    assert math.isclose(
        result["dimensionless_phase_group_dispersion"], expected_d1, rel_tol=2e-15
    )
    assert math.isclose(
        result["dimensionless_frequency_curvature"], expected_d2, rel_tol=2e-15
    )


def test_love_and_cowling_ocean_are_finite_and_deterministic():
    love = compute_at_omega(
        build_solid_model(), 2.0 * math.pi / 20.0, "Love", 3.0, max_steps=4
    )
    assert love["status"] == "ok"
    assert np.isfinite(
        [
            love["phase_velocity"],
            love["group_velocity"],
            love["group_velocity_omega_derivative"],
            love["wavenumber_omega_second_derivative"],
        ]
    ).all()

    ocean_model = build_ocean_model()
    first = compute_at_omega(
        ocean_model, 2.0 * math.pi / 8.0, "Rayleigh", 3.0, max_steps=4
    )
    second = compute_at_omega(
        ocean_model, 2.0 * math.pi / 8.0, "Rayleigh", 3.0, max_steps=4
    )
    assert first == second
    assert first["status"] == "ok"


def test_missing_root_is_reported_without_a_surrogate_value():
    result = compute_at_omega(
        build_solid_model(), 2.0 * math.pi / 20.0, "Rayleigh", 20.0, max_steps=0
    )
    assert result["status"] == "unsafe_root"
    assert result["accepted_level"] == -1
    for key in (
        "phase_velocity",
        "group_velocity",
        "group_velocity_omega_derivative",
        "wavenumber_omega_second_derivative",
        "dimensionless_frequency_curvature",
    ):
        assert math.isnan(result[key])


if __name__ == "__main__":
    test_fixed_mesh_ad_matches_root_resolved_five_point_fd()
    test_romberg_result_identities_and_existing_api_isolation()
    test_love_and_cowling_ocean_are_finite_and_deterministic()
    test_missing_root_is_reported_without_a_surrogate_value()
    print("test_frequency_curvature_ad_grid: PASS")
