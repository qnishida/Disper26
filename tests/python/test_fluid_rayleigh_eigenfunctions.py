import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import disper26py as dp


@pytest.mark.parametrize("gravity", [0.0, 0.00980665])
@pytest.mark.parametrize("solid_layer", [False, True])
def test_seabed_lagrangian_traction_and_displacement_are_continuous(gravity, solid_layer):
    ocean = dp.create_fluid_parameters(1.5, 1.03, gravity)
    crust = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    layers = [ocean, crust] if solid_layer else [ocean]
    thicknesses = [3.0, 20.0] if solid_layer else [3.0]
    omega = 2.0 * math.pi / 20.0
    c = dp.search_roots(layers, thicknesses, halfspace, omega, 0.5, 4.4, 300, "Rayleigh")[0]
    ef = dp.compute_rayleigh_eigenfunctions(layers, thicknesses, halfspace, omega, c, 8)

    is_fluid = np.asarray(ef["is_fluid"], dtype=bool)
    fluid_side = np.flatnonzero(is_fluid)[-1]
    solid_side = fluid_side + 1
    assert not is_fluid[solid_side]
    assert ef["depth"][fluid_side] == ef["depth"][solid_side] == 3.0
    # Existing eigenfunction adapter tolerances also resolve the two traces.
    for component in ("uz", "szz"):
        np.testing.assert_allclose(ef[component][fluid_side], ef[component][solid_side],
                                   rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(ef["szr"][solid_side], 0.0, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(ef["szz"][0], 0.0, rtol=0.0, atol=1e-13)


def test_fluid_rayleigh_eigenfunctions_and_energy_are_finite():
    ocean = dp.create_fluid_parameters(1.5, 1.03)
    crust = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    layers = [ocean, crust]
    thicknesses = [3.0, 20.0]
    omega = 2.0 * math.pi * 0.05

    roots = dp.search_roots(layers, thicknesses, halfspace, omega, 0.5, 5.0, 300, "Rayleigh")
    assert roots
    eigenfunctions = dp.compute_rayleigh_eigenfunctions(
        layers, thicknesses, halfspace, omega, roots[0], 4
    )

    assert "is_fluid" in eigenfunctions
    is_fluid = np.asarray(eigenfunctions["is_fluid"], dtype=bool)
    assert np.any(is_fluid)
    for key in ("ur", "uz", "szz", "szr"):
        assert np.all(np.isfinite(np.asarray(eigenfunctions[key])))
    assert np.max(np.abs(np.asarray(eigenfunctions["szr"])[is_fluid])) < 1e-14

    i1, i2, i3, _c, u_energy = dp.compute_energy_integrals(
        eigenfunctions, layers, thicknesses, halfspace, omega, roots[0], "Rayleigh"
    )
    assert np.isfinite([i1, i2, i3, u_energy]).all()
    assert i1 > 0.0


def test_fluid_rayleigh_horizontal_displacement_uses_lagrangian_stress_relation():
    gravity = 0.00980665
    ocean = dp.create_fluid_parameters(1.5, 1.03, gravity)
    crust = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    layers = [ocean, crust]
    thicknesses = [3.0, 20.0]
    omega = 2.0 * math.pi * 0.05

    roots = dp.search_roots(layers, thicknesses, halfspace, omega, 0.5, 5.0, 300, "Rayleigh")
    assert roots
    eigenfunctions = dp.compute_rayleigh_eigenfunctions(
        layers, thicknesses, halfspace, omega, roots[0], 4
    )

    is_fluid = np.asarray(eigenfunctions["is_fluid"], dtype=bool)
    ur = np.asarray(eigenfunctions["ur"])
    uz = np.asarray(eigenfunctions["uz"])
    szz = np.asarray(eigenfunctions["szz"])
    k = omega / roots[0]
    expected = k * (ocean.rho * gravity * uz[is_fluid] - szz[is_fluid]) / (ocean.rho * omega**2)

    assert np.max(np.abs(ur[is_fluid] - expected)) < 1e-13


def test_solid_rayleigh_eigenfunctions_mark_no_fluid_points():
    layer = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    layers = [layer]
    thicknesses = [20.0]
    omega = 2.0 * math.pi * 0.05

    roots = dp.search_roots(layers, thicknesses, halfspace, omega, 1.0, 5.0, 300, "Rayleigh")
    assert roots
    eigenfunctions = dp.compute_rayleigh_eigenfunctions(
        layers, thicknesses, halfspace, omega, roots[0], 4
    )
    assert "is_fluid" in eigenfunctions
    assert not np.any(np.asarray(eigenfunctions["is_fluid"], dtype=bool))
    for key in ("ur", "uz", "szz", "szr"):
        assert np.all(np.isfinite(np.asarray(eigenfunctions[key])))


def test_fluid_rayleigh_variational_gradient_is_finite_and_matches_fd():
    crust = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    thicknesses = [3.0, 20.0]
    omega = 2.0 * math.pi * 0.05

    def root_for(vp=1.5, rho=1.03):
        ocean = dp.create_fluid_parameters(vp, rho)
        roots = dp.search_roots(
            [ocean, crust], thicknesses, halfspace, omega, 0.5, 5.0, 500, "Rayleigh"
        )
        assert roots
        return roots[0]

    ocean = dp.create_fluid_parameters(1.5, 1.03)
    c0 = root_for()
    result = dp.compute_phase_velocity_hessian_weakform_rayleigh(
        [ocean, crust], thicknesses, halfspace, omega, c0, "Isotropic", False, 16, False
    )
    gradient = np.asarray(result["gradient"], dtype=float)

    assert np.all(np.isfinite(gradient))
    assert abs(gradient[0]) < 1e-14

    dvp = 1.5e-4
    drho = 1.03e-4
    fd_vp = (root_for(vp=1.5 + dvp) - root_for(vp=1.5 - dvp)) / (2.0 * dvp)
    fd_rho = (root_for(rho=1.03 + drho) - root_for(rho=1.03 - drho)) / (2.0 * drho)

    assert abs(gradient[1] - fd_vp) / abs(fd_vp) < 2e-2
    assert abs(gradient[2] - fd_rho) / abs(fd_rho) < 2e-2


def test_fluid_rayleigh_grid_weakform_hessian_treats_fluid_nodes_as_fixed():
    freq = 0.05
    period = 1.0 / freq
    ocean = dp.create_fluid_parameters(1.5, 1.03)
    crust_top = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    crust_bottom = dp.create_isotropic_parameters(3.5, 6.2, 2.8)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    depths = [0.0, 3.0, 3.0, 23.0, 23.0]
    params = [ocean, ocean, crust_top, crust_bottom, halfspace]
    is_disc = [False, True, False, True, False]

    roots = dp.search_roots(
        [ocean, crust_top], [3.0, 20.0], halfspace,
        2.0 * math.pi * freq, 0.5, 5.0, 500, "Rayleigh"
    )
    assert roots
    c_guess = roots[0]

    result = dp.compute_phase_velocity_hessian_weakform_rayleigh_grid(
        depths, params, is_disc, freq, c_guess, "Isotropic", 1, 1e-6
    )
    gradient = np.asarray(result["gradient"], dtype=float)
    hessian = np.asarray(result["hessian"], dtype=float)

    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    assert np.allclose(gradient[:6], 0.0, atol=1e-14)
    assert np.allclose(hessian[:6, :], 0.0, atol=1e-14)
    assert np.allclose(hessian[:, :6], 0.0, atol=1e-14)
    assert np.linalg.norm(gradient[6:]) > 0.0

    def romberg_c(vp_top):
        perturbed_top = dp.create_isotropic_parameters(3.2, vp_top, 2.7)
        perturbed_params = [ocean, ocean, perturbed_top, crust_bottom, halfspace]
        c, *_ = dp.compute_dispersion_romberg(
            depths, perturbed_params, is_disc, period, "Rayleigh", c_guess, 1, 1e-8
        )
        return c

    dvp = 5.8e-4
    fd_vp = (romberg_c(5.8 + dvp) - romberg_c(5.8 - dvp)) / (2.0 * dvp)
    assert abs(gradient[2 * 3 + 1] - fd_vp) / max(abs(fd_vp), 1e-12) < 1e-1

    result_vti = dp.compute_phase_velocity_hessian_weakform_rayleigh_grid(
        depths, params, is_disc, freq, c_guess, "VTI", 1, 1e-6
    )
    gradient_vti = np.asarray(result_vti["gradient"], dtype=float)
    hessian_vti = np.asarray(result_vti["hessian"], dtype=float)
    assert np.all(np.isfinite(gradient_vti))
    assert np.all(np.isfinite(hessian_vti))
    assert np.allclose(gradient_vti[:12], 0.0, atol=1e-14)
    assert np.allclose(hessian_vti[:12, :], 0.0, atol=1e-14)
    assert np.allclose(hessian_vti[:, :12], 0.0, atol=1e-14)


def test_fluid_rayleigh_grid_weakform_hessian_requires_seabed_discontinuity():
    freq = 0.05
    ocean = dp.create_fluid_parameters(1.5, 1.03)
    crust = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    depths = [0.0, 3.0, 23.0]
    params = [ocean, crust, halfspace]
    is_disc = [False, False, False]

    try:
        dp.compute_phase_velocity_hessian_weakform_rayleigh_grid(
            depths, params, is_disc, freq, 2.0, "Isotropic", 1, 1e-6
        )
    except RuntimeError as exc:
        assert "duplicate-depth discontinuity" in str(exc)
    else:
        raise AssertionError("Expected fluid-solid grid without discontinuity to be rejected")


if __name__ == "__main__":
    test_fluid_rayleigh_eigenfunctions_and_energy_are_finite()
    test_fluid_rayleigh_horizontal_displacement_uses_lagrangian_stress_relation()
    test_solid_rayleigh_eigenfunctions_mark_no_fluid_points()
    test_fluid_rayleigh_variational_gradient_is_finite_and_matches_fd()
    test_fluid_rayleigh_grid_weakform_hessian_treats_fluid_nodes_as_fixed()
    test_fluid_rayleigh_grid_weakform_hessian_requires_seabed_discontinuity()
