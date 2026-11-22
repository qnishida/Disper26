#!/usr/bin/env python3
"""Smoke tests for EFT defaults and Mineos-style Q physical dispersion."""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import disper26py as dp


def close(a, b, tol=1e-12):
    assert abs(a - b) <= tol, f"{a} != {b}"


def make_vti():
    params = dp.PhysicalParameters()
    params.rho = 3.3
    params.c11 = 220.0
    params.c13 = 75.0
    params.c33 = 205.0
    params.c44 = 62.0
    params.c66 = 71.0
    return params


def test_physical_dispersion_factor():
    tref_s = 1.0
    period_s = 20.0
    expected = 2.0 * math.log(tref_s / period_s) / math.pi
    close(dp.mineos_physical_dispersion_factor(tref_s, period_s), expected)
    close(dp.mineos_physical_dispersion_factor(0.0, period_s), 0.0)


def test_physical_dispersion_changes_shear_moduli():
    params = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    corrected = dp.apply_mineos_physical_dispersion(params, 57823.0, 600.0, 1.0, 20.0)
    fct = dp.mineos_physical_dispersion_factor(1.0, 20.0)
    expected_shear_scale = 1.0 + (1.0 / 600.0) * fct
    close(corrected.c44 / params.c44, expected_shear_scale)
    close(corrected.c66 / params.c66, expected_shear_scale)
    close(corrected.rho, params.rho)


def test_eft_default_ba97_exponent():
    radius = 6371.0
    layers = [dp.create_isotropic_parameters(3.2, 5.8, 2.6)]
    thicknesses = [20.0]
    default_layers, default_thick = dp.earth_flattening_transformation_layers(layers, thicknesses, radius)
    explicit_layers, explicit_thick = dp.earth_flattening_transformation_layers(layers, thicknesses, radius, -2.0)
    close(default_thick[0], explicit_thick[0])
    close(default_layers[0].rho, explicit_layers[0].rho)
    close(default_layers[0].c44, explicit_layers[0].c44)


def test_grid_eft_matches_node_scaling():
    radius = 6371.0
    depths = [0.0, 20.0, 20.0]
    params = [
        dp.create_isotropic_parameters(3.2, 5.8, 2.6),
        dp.create_isotropic_parameters(4.5, 8.1, 3.4),
        dp.create_isotropic_parameters(4.7, 8.3, 3.5),
    ]
    is_disc = [False, True, False]
    flat_depths, flat_params, flat_disc = dp.earth_flattening_transformation_grid(
        depths, params, is_disc, radius, -2.0
    )
    expected_depth = radius * math.log(radius / (radius - 20.0))
    radius_ratio = (radius - 20.0) / radius
    scale_rho = radius_ratio ** 4.0
    scale_elastic = radius_ratio ** 2.0
    close(flat_depths[0], 0.0)
    close(flat_depths[1], expected_depth)
    close(flat_params[1].rho / params[1].rho, scale_rho)
    close(flat_params[1].c44 / params[1].c44, scale_elastic)
    close(flat_depths[2], expected_depth)
    close(flat_params[2].c44 / params[2].c44, scale_elastic)
    assert flat_disc == is_disc
    assert not hasattr(dp, "earth_flattening_transformation")
    assert hasattr(dp, "earth_flattening_transformation_grid_experimental_density")


def test_grid_ba97_fluid_scaling_keeps_density_fixed():
    radius = 6371.0
    depth = 20.0
    fluid_top = dp.create_fluid_parameters(1.5, 1.03)
    fluid_bottom = dp.create_fluid_parameters(1.6, 1.04)
    solid = dp.create_isotropic_parameters(3.2, 5.8, 2.6)
    flat_depths, flat_params, flat_disc = dp.earth_flattening_transformation_grid(
        [0.0, depth, 2.0 * depth], [fluid_top, fluid_bottom, solid], [False, False, False], radius, -2.0
    )
    radius_ratio = (radius - depth) / radius
    solid_radius_ratio = (radius - 2.0 * depth) / radius
    close(flat_depths[1], radius * math.log(radius / (radius - depth)))
    close(flat_params[1].rho, fluid_bottom.rho)
    close(flat_params[1].c33 / fluid_bottom.c33, radius_ratio ** -2.0)
    close(flat_params[2].rho / solid.rho, solid_radius_ratio ** 4.0)
    close(flat_params[2].c44 / solid.c44, solid_radius_ratio ** 2.0)
    assert flat_disc == [False, False, False]


def test_layer_eft_scales_vti_coefficients():
    radius = 6371.0
    thickness = 20.0
    params = make_vti()
    flat_layers, flat_thick = dp.earth_flattening_transformation_layers(
        [params], [thickness], radius, -2.0
    )
    r_mid = radius - 0.5 * thickness
    radius_ratio = r_mid / radius
    scale_rho = radius_ratio ** 4.0
    scale_elastic = radius_ratio ** 2.0
    close(flat_thick[0], radius * math.log(radius / (radius - thickness)))
    close(flat_layers[0].rho / params.rho, scale_rho)
    for name in ("c11", "c13", "c33", "c44", "c66"):
        close(getattr(flat_layers[0], name) / getattr(params, name), scale_elastic)


def test_grid_experimental_density_eft_scales_vti_coefficients():
    radius = 6371.0
    depth = 20.0
    params = [make_vti(), make_vti()]
    flat_depths, flat_params, flat_disc = dp.earth_flattening_transformation_grid_experimental_density(
        [0.0, depth], params, [False, False], radius, 5.0
    )
    r_node = radius - depth
    scale_vel = radius / r_node
    scale_rho = (r_node / radius) ** 5.0
    scale_elastic = scale_rho * scale_vel * scale_vel
    close(flat_depths[1], radius * math.log(radius / r_node))
    close(flat_params[1].rho / params[1].rho, scale_rho)
    for name in ("c11", "c13", "c33", "c44", "c66"):
        close(getattr(flat_params[1], name) / getattr(params[1], name), scale_elastic)
    assert flat_disc == [False, False]


if __name__ == "__main__":
    test_physical_dispersion_factor()
    test_physical_dispersion_changes_shear_moduli()
    test_eft_default_ba97_exponent()
    test_grid_eft_matches_node_scaling()
    test_grid_ba97_fluid_scaling_keeps_density_fixed()
    test_layer_eft_scales_vti_coefficients()
    test_grid_experimental_density_eft_scales_vti_coefficients()
