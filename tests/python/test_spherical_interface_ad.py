"""Interface motion in physical spherical depth, including a moving halfspace."""
from pathlib import Path
import sys

import numpy as np
import pytest

import disper26 as dp
import disper26py as raw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inversion"))
from disper26_inv.data import DatasetSpec, Datum
from disper26_inv.model import ForwardGridModel, GridRefinementPolicy, InterfaceShiftBasis, InversionBasis
from disper26_inv.pipeline import run_phase1


VS = np.array([3., 3.55, 4.04, 4.25, 4.55])
VP = np.array([5.4, 6.3, 7.2, 7.6, 8.22])
RHO = np.array([2.55, 2.75, 3.05, 3.18, 3.35])
FLAGS = [False, True, False, True, False]
PERIOD = 26.
OPTIONS = dp.Options(max_steps=8, tolerance=1.e-10)


def solve(geometry, wave, upper=18., lower=43., vs_bottom=VS[-1]):
    vs = VS.copy()
    vs[-1] = vs_bottom
    model = dp.create_grid_model([0., upper, upper, lower, lower], vs=vs, vp=VP, rho=RHO,
                                 interpolation="stiffness", geometry=geometry)
    return dp.compute_dispersion(model, period=PERIOD, wave=wave, options=OPTIONS,
                                 outputs=("phase_velocity", "group_velocity", "hv_ratio"))


def values(solution):
    result = [solution.phase_velocity, solution.group_velocity]
    if solution.hv_ratio is not None:
        result.append(solution.hv_ratio)
    return np.array(result)


@pytest.mark.parametrize("scheme,radius,exponent", [
    ("spherical_ba97", 6371., -2.), ("spherical_experimental_density", 3200., -1.3)])
@pytest.mark.parametrize("wave", ["Rayleigh", "Love"])
def test_spherical_interface_derivatives(wave, scheme, radius, exponent):
    geo = dp.Geometry(scheme, radius, exponent)
    ref = solve(geo, wave)
    opt = dp.Options(max_levels=ref.diagnostics.accepted_level)
    first = dp.compute_interface_kernels(ref, options=opt)
    rows = [dp.compute_interface_hessian_row(ref, interface_index=i, basis="velocity_iso",
                                             options=opt) for i in range(2)]
    assert first.diagnostics.method == rows[0].diagnostics.method == "ad"
    assert [p.name for p in rows[0].parameters[-2:]] == ["Depth", "Depth"]
    h = np.vstack([r.hessian for r in rows])
    np.testing.assert_allclose(h[0, -1], h[1, -2], rtol=1e-7, atol=1e-12)

    params = [raw.create_isotropic_parameters(*v) for v in zip(VS, VP, RHO)]
    kernels = raw.compute_kernels_ad_grid(
        ref.model.depths, params, FLAGS, PERIOD, wave, ref.phase_velocity, "VTI",
        8, 1e-10, grid_policy=scheme, planet_radius=radius, flattening_exponent=exponent)[5]
    depths = [k for k in kernels if k.parameter_name == "Depth"]
    assert len(depths) == 2
    for step in (.02, .01):
        for i, (name, value) in enumerate([("upper", 18.), ("lower", 43.)]):
            plus, minus = [solve(geo, wave, **{name: value + sign*step}) for sign in (1, -1)]
            assert plus.diagnostics.accepted_level == minus.diagnostics.accepted_level == opt.max_levels
            g = (values(plus) - values(minus)) / (2*step)
            k = depths[i]
            calculated = [k.d_c, k.d_U] + ([k.d_HV] if wave == "Rayleigh" else [])
            np.testing.assert_allclose(calculated, g, rtol=3e-5, atol=1e-7)
            np.testing.assert_allclose(first.gradient[i], g[0], rtol=3e-5, atol=1e-7)
            diag = (plus.phase_velocity - 2*ref.phase_velocity + minus.phase_velocity) / step**2
            np.testing.assert_allclose(h[i, -2+i], diag, rtol=3e-5, atol=1e-7)

        # Four independent phase solves for each mixed derivative. Exercise both
        # the neighbouring interface and the material of the moving halfspace.
        for name, center, delta, column in [("lower", 43., step, -1),
                                             ("vs_bottom", VS[-1], .02*step, 12)]:
            mixed = sum(a*b*solve(geo, wave, upper=18.+a*step, **{name: center+b*delta}).phase_velocity
                        for a in (-1, 1) for b in (-1, 1)) / (4*step*delta)
            np.testing.assert_allclose(h[0, column], mixed, rtol=3e-5, atol=1e-7)


def test_spherical_phase1_taylor_prediction():
    depths = [0., 18., 18., 43., 43.]
    policy = GridRefinementPolicy("spherical_ba97")
    model = ForwardGridModel(depths, RHO, VP, VP, VS, VS, np.ones(5), FLAGS,
                             refinement_policy=policy)
    basis = InterfaceShiftBasis(InversionBasis(model, [0., 43.], ("vsv",)), 1)
    data = DatasetSpec([Datum(wave, 0, PERIOD) for wave in ("Rayleigh", "Love")])
    results = [run_phase1(model, basis, data, reduced_hessian=reduced, processes=workers,
                          ad_kernel_every=every)
               for reduced, workers, every in [(False, 1, 1), (True, 2, 0)]]
    ref, reduced = results
    assert ref.meta["depth_backend"]["backend"] == "spherical_ad"
    np.testing.assert_allclose(reduced.J, ref.J, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(reduced.H, ref.H, rtol=1e-6, atol=1e-10)
    direction = np.array([.003, -.001, 1.])
    errors = []
    for scale in (1., .5):
        x = scale*direction
        m = basis.m_to_forward_model(x)
        varied = dp.create_model(m.depths, kind="grid", rho=m.rho, vpv=m.vpv, vph=m.vph,
                                      vsv=m.vsv, vsh=m.vsh, eta=m.eta, interpolation="stiffness",
                                      geometry=dp.Geometry(policy.kind))
        exact = [dp.compute_dispersion(varied, wave=d.wave, period=d.period, options=OPTIONS).phase_velocity
                 for d in data.data]
        predicted = ref.surrogate().predict(x)
        errors.append(np.abs(exact-predicted))
    # Second-order Taylor remainder is cubic in the perturbation on a simple root.
    assert np.all(errors[0] > 5*errors[1])


@pytest.mark.parametrize("wave,g,h", [
    ("Rayleigh", -.011886779368170354, -.0007393870067461366),
    ("Love", -.028156734242854812, -.00016834987166952178)])
def test_flat_interface_reference(wave, g, h):
    # Values from the flat variational paths before spherical interface support.
    indices = [0, 1, 2, 4]
    model = dp.create_grid_model([0., 18., 18., 55.], vs=VS[indices], vp=VP[indices],
                                 rho=RHO[indices], interpolation="stiffness")
    ref = dp.compute_dispersion(model, period=PERIOD, wave=wave, options=OPTIONS)
    opt = dp.Options(max_levels=5)
    grad = dp.compute_interface_kernels(ref, options=opt)
    row = dp.compute_interface_hessian_row(ref, interface_index=0, basis="stiffness", options=opt)
    assert grad.diagnostics.method == "variational"
    assert row.diagnostics.method == "variational_ad"
    # The flat variational/AD regressions use a relative tolerance of 2e-6.
    np.testing.assert_allclose([grad.gradient[0], row.hessian.ravel()[-1]], [g, h], rtol=2e-6)


def test_interface_capabilities_and_material_only_requests():
    geo = dp.Geometry("spherical_ba97")
    ref = solve(geo, "Rayleigh")
    advertised = {(c.operation, c.method): c.supported for c in dp.get_capabilities(ref.model)
                  if c.basis in ("interface_depth", "velocity_iso")}
    assert advertised["interface_kernels", "ad"] and advertised["interface_hessian_row", "ad"]
    assert not advertised["interface_hessian_row", "variational_ad"]
    with pytest.raises(dp.Error, match="require"):
        dp.compute_interface_kernels(ref, options=dp.Options(method="variational"))

    # A surface fluid still supports its established material kernels, while
    # explicit interface derivatives require an all-solid model.
    model = dp.create_grid_model([0., 3., 3., 35.], vs=[0., 0., 3.5, 4.5],
                                 vp=[1.5, 1.5, 6.1, 8.1], rho=[1., 1., 2.8, 3.3],
                                 interpolation="stiffness", geometry=geo)
    ocean = dp.compute_dispersion(model, period=PERIOD, options=OPTIONS)
    g = dp.compute_kernels(ocean, basis="stiffness", options=dp.Options(method="ad"))
    assert np.isfinite(g.gradient).all()
    with pytest.raises(dp.Error, match="fluid"):
        dp.compute_interface_kernels(ocean)


@pytest.mark.parametrize("wave", ["Love", "Rayleigh"])
def test_interface_gradient_tracks_first_overtone(wave):
    def mode(upper):
        m = dp.create_grid_model([0., upper, upper, 43., 43.], vs=VS, vp=VP, rho=RHO,
                                 interpolation="stiffness", geometry=dp.Geometry("spherical_ba97"))
        return dp.compute_dispersion(m, period=8., mode=1, wave=wave, options=OPTIONS)

    ref = mode(18.)
    opts = dp.Options(max_levels=ref.diagnostics.accepted_level)
    step = .01
    measured = (mode(18.+step).phase_velocity - mode(18.-step).phase_velocity) / (2*step)
    np.testing.assert_allclose(dp.compute_interface_kernels(ref, options=opts).gradient[0],
                               measured, rtol=3e-5, atol=1e-7)
