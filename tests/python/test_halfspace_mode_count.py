"""Halfspace initialization and cumulative counts checked against actual roots."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

import disper26 as api
import disper26py as dp


def isotropic_rayleigh_speed(vs, vp):
    # Squaring (2-x)^2 = 4 sqrt(1-x) sqrt(1-q*x), removing the x=0
    # factor, gives this cubic in x=(c/vs)^2, q=(vs/vp)^2.
    q = (vs / vp)**2
    roots = np.roots([1., -8., 24. - 16.*q, -16.*(1. - q)])
    physical = [x.real for x in roots if abs(x.imag) < 1e-12 and 0 < x.real < 1]
    assert len(physical) == 1
    return vs * math.sqrt(physical[0])


@pytest.mark.parametrize("vs,vp,rho", [(1., math.sqrt(3.), 2.), (4.5, 8., 3.3)])
@pytest.mark.parametrize("period", [0.1, 1., 10., 100., 1000.])
def test_analytic_halfspace_and_identical_layer_splitting(vs, vp, rho, period):
    h = dp.create_isotropic_parameters(vs, vp, rho)
    omega = 2 * math.pi / period
    root = isotropic_rayleigh_speed(vs, vp)
    # Probe outside root-roundoff ambiguity, on both sides of the analytical
    # root. Splitting a homogeneous medium cannot change its spectrum.
    for thicknesses in ([], [1.], [2.5]*4):
        layers = [h] * len(thicknesses)
        for c, expected in ((root - 1e-5, 0), (root + 1e-5, 1), (0.999*vs, 1)):
            assert dp.compute_mode_count(layers, thicknesses, h, omega, c, "Rayleigh") == expected
            assert dp.compute_mode_count(layers, thicknesses, h, omega, c, "Love") == 0
    # The zero-based label at the homogeneous fundamental root is zero.
    assert dp.compute_mode_count([], [], h, omega, root, "Rayleigh") == 0


def comparison_models():
    iso = dp.create_isotropic_parameters

    def vti(vs, vp, rho):
        p = iso(vs, vp, rho)
        p.c11 *= 1.08
        p.c66 *= 1.06
        p.c13 *= 0.96
        return p

    h = iso(4.5, 8., 3.3)
    return [
        ("halfspace", [], [], h),
        ("homogeneous", [h], [10.], h),
        ("split", [h]*4, [2.5]*4, h),
        ("solid", [iso(1., 1.73, 2.), iso(1.5, 2.6, 2.2), iso(2.2, 3.8, 2.5)],
         [2., 5., 8.], iso(3., 5.2, 2.7)),
        ("ocean", [dp.create_fluid_parameters(1.5, 1.03), iso(2.2, 3.8, 2.5)],
         [3., 12.], iso(3., 5.2, 2.7)),
        ("fast_cap", [iso(5., 9., 3.5)], [2.], h),
        ("vti", [vti(2.8, 5., 2.5), vti(3.7, 6.5, 2.9)], [5., 10.], vti(4.5, 8., 3.3)),
        ("vti_half", [], [], vti(4.5, 8., 3.3)),
        ("lvz", [iso(3.5, 6., 2.6), iso(2.8, 4.8, 2.4), iso(3.8, 6.8, 2.8)],
         [3., 5., 8.], h),
    ]


@pytest.mark.parametrize("frequency", [0.02, 0.1, 0.5])
@pytest.mark.parametrize("name,layers,thicknesses,halfspace", comparison_models(),
                         ids=lambda value: value if isinstance(value, str) else None)
def test_counts_match_independent_secular_scan(name, layers, thicknesses, halfspace, frequency):
    omega = 2 * math.pi * frequency
    beta = math.sqrt(halfspace.c44 / halfspace.rho)
    # search_roots brackets the dispersion residual, independently of the
    # integer counter. Double its mesh to verify reference completeness here.
    roots = dp.search_roots(layers, thicknesses, halfspace, omega, 0.5, 0.999*beta, 2000, "Rayleigh")
    fine = dp.search_roots(layers, thicknesses, halfspace, omega, 0.5, 0.999*beta, 4000, "Rayleigh")
    # These well-separated roots should agree to 1e-8 km/s on the two meshes,
    # well inside the 1e-5 km/s offset used to test each side of a root.
    np.testing.assert_allclose(roots, fine, rtol=0, atol=1e-8)
    probes = [(c + side*1e-5, i + int(side > 0))
              for i, c in enumerate(fine) for side in (-1, 1)]
    probes += [(c, sum(root < c for root in fine))
               for c in np.linspace(0.6, 0.997*beta, 120)
               if all(abs(c-root) > 1e-4 for root in fine)]
    for c, expected in probes:
        actual = dp.compute_mode_count(layers, thicknesses, halfspace, omega, c, "Rayleigh")
        assert actual == expected, (name, frequency, c, actual, expected)

    if name in ("solid", "ocean", "fast_cap", "lvz"):
        # Crossing the halfspace's own Rayleigh speed must not create a mode
        # of the full stack. Include the boundary itself, where the initial
        # contribution exchanges with a crossing at the bottom interface.
        c_hs = isotropic_rayleigh_speed(beta, math.sqrt(halfspace.c33/halfspace.rho))
        for c in (c_hs - 1e-6, c_hs, c_hs + 1e-6):
            expected = sum(root < c for root in fine)
            assert dp.compute_mode_count(layers, thicknesses, halfspace, omega, c, "Rayleigh") == expected


@pytest.mark.parametrize("kind,interpolation,depths", [
    ("pancake", None, [0.]), ("pancake", None, [0., 10.]),
    ("grid", "stiffness", [0., 10.]), ("grid", "velocity", [0., 10.]),
])
def test_public_mode_order_solver_on_homogeneous_medium(kind, interpolation, depths):
    model = api.create_model(depths, kind=kind, interpolation=interpolation,
                             vs=[4.5]*len(depths), vp=[8.]*len(depths), rho=[3.3]*len(depths))
    expected = isotropic_rayleigh_speed(4.5, 8.)
    for period in (2., 10., 100.):
        result = api.compute_dispersion(model, period=period)
        assert result.mode == 0
        # Require agreement well inside the 1e-5 km/s count-probe offsets,
        # allowing for the independent reference and refinement paths.
        assert abs(result.phase_velocity - expected) < 1e-8
    with pytest.raises(RuntimeError, match="outside"):
        api.compute_dispersion(model, period=10., mode=1)


def test_debug_separates_initial_contribution_from_depth_crossings():
    h = dp.create_isotropic_parameters(4.5, 8., 3.3)
    for c, initial in ((4., 0), (4.3, -1)):
        info = dp.debug_mode_count([], [], h, 2*math.pi/10., c, "Rayleigh")
        assert info["status"] == "ok"
        assert info["initial_signed_count"] == initial
        assert info["signed_count"] == 0
        assert info["crossings"] == []
        assert info["n_integer"] == abs(initial)


@pytest.mark.parametrize("ocean", [False, True])
def test_public_solver_retains_highest_supra_rayleigh_mode(ocean):
    if ocean:
        model = api.create_model([0., 3., 15.], kind="pancake", vs=[0., 2.2, 3.],
                                 vp=[1.5, 3.8, 5.2], rho=[1.03, 2.5, 2.7])
        name = "ocean"
    else:
        model = api.create_model([0., 2., 7., 15.], kind="pancake", vs=[1., 1.5, 2.2, 3.],
                                 vp=[1.73, 2.6, 3.8, 5.2], rho=[2., 2.2, 2.5, 2.7])
        name = "solid"
    _, layers, ds, h = next(x for x in comparison_models() if x[0] == name)
    roots = dp.search_roots(layers, ds, h, math.pi, 0.5, 2.999, 4000, "Rayleigh")
    assert roots[-1] > isotropic_rayleigh_speed(3., 5.2)
    solution = api.compute_dispersion(model, period=2., mode=len(roots)-1)
    assert abs(solution.phase_velocity - roots[-1]) < 1e-8


@pytest.mark.parametrize("frequency", [0.02, 0.05, 0.1])
def test_prem_counts_match_secular_scan(frequency):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inversion"))
    from disper26_inv.prem import load_prem_flat

    model = load_prem_flat(max_depth_km=400., max_spacing_km=20.)
    layers, ds, halfspace = model.refined_layers(level=1)
    test_counts_match_independent_secular_scan("prem", layers, ds, halfspace, frequency)


@pytest.mark.parametrize("frequency", [0.02, 0.1, 0.5])
def test_initial_boundary_crossing_directly_into_water(frequency):
    layers = [dp.create_fluid_parameters(1.5, 1.03)]
    halfspace = dp.create_isotropic_parameters(4.5, 8., 3.3)
    omega = 2*math.pi*frequency
    roots = dp.search_roots(layers, [3.], halfspace, omega, 0.5, 4.499, 4000, "Rayleigh")
    c_hs = isotropic_rayleigh_speed(4.5, 8.)
    for c in (c_hs - 1e-6, c_hs, c_hs + 1e-6):
        assert dp.compute_mode_count(layers, [3.], halfspace, omega, c, "Rayleigh") == sum(r < c for r in roots)
