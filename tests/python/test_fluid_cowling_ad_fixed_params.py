"""Cowling GRID propagation is available for forward checks."""

import pytest
import disper26py as dp


def build_model(gravity):
    fluid_top = dp.create_fluid_parameters(1.5, 1.03, gravity)
    fluid_bot = dp.create_fluid_parameters(1.5, 1.03, gravity)
    solid_top = dp.create_isotropic_parameters(3.2, 5.8, 2.6)
    solid_bot = dp.create_isotropic_parameters(3.5, 6.2, 2.8)
    halfspace = dp.create_isotropic_parameters(4.0, 7.0, 3.2)
    depths = [0.0, 4.0, 4.0, 30.0, 60.0]
    params = [fluid_top, fluid_bot, solid_top, solid_bot, halfspace]
    is_discontinuity = [False, True, False, False, False]
    return depths, params, is_discontinuity


def test_fluid_cowling_ad_fixed_params():
    period = 8.0
    c_guess = 3.0
    depths, params, is_discontinuity = build_model(0.00980665)
    with pytest.raises(ValueError, match="Cowling gravity"):
        dp.compute_dispersion_romberg_with_kernels(
            depths, params, is_discontinuity, period, "Rayleigh", c_guess,
            "Isotropic", 4, 1e-8)
    c = dp.compute_dispersion_romberg(
        depths, params, is_discontinuity, period, "Rayleigh", c_guess, 4, 1e-8)[0]

    depths0, params0, is_discontinuity0 = build_model(0.0)
    c0 = dp.compute_dispersion_romberg(
        depths0, params0, is_discontinuity0, period, "Rayleigh", c_guess, 4, 1e-8
    )[0]

    assert abs(c - c0) > 1e-5


if __name__ == "__main__":
    test_fluid_cowling_ad_fixed_params()
    print("OK")
