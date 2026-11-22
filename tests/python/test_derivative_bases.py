"""Physical coordinates at ordinary and zero-Lamé-parameter reference models."""
import numpy as np
import pytest

import disper26 as dp


@pytest.mark.parametrize("lam", [12., 0.])
@pytest.mark.parametrize("basis", ["stiffness_iso", "velocity_iso"])
def test_isotropic_derivatives(lam, basis):
    # A = 2L exactly in floating point for this velocity/density combination.
    vs, rho = 2.5, 2.8
    mu = rho * vs * vs
    coords = np.array([[lam, mu, rho], [55., 65., 3.3]])
    if basis == "velocity_iso":
        coords[:, 0], coords[:, 1] = (np.sqrt(coords[:, 1] / coords[:, 2]),
                                      np.sqrt((coords[:, 0] + 2 * coords[:, 1]) / coords[:, 2]))

    def solve(x):
        a, b, r = x.T
        fields = (dict(vs=a, vp=b, rho=r) if basis == "velocity_iso" else
                  dict(A=a+2*b, C=a+2*b, F=a, L=b, N=b, rho=r))
        return dp.compute_dispersion(dp.create_model([0., 10.], kind="layer_cake", **fields), period=5.)

    reference = solve(coords)
    result = dp.compute_hessian(reference, basis=basis)
    assert [p.name for p in result.parameters[:3]] == (
        ["Vs", "Vp", "Rho"] if basis == "velocity_iso" else ["Lambda", "Mu", "Rho"])
    assert [p.unit for p in result.parameters[:3]] == (
        ["km/s", "km/s", "g/cm^3"] if basis == "velocity_iso" else ["GPa", "GPa", "g/cm^3"])
    if lam == 0:
        material = reference.model.materials[0]
        assert material.A == 2 * material.L
    for index in range(coords.size):
        step = 1e-4
        plus, minus = coords.copy(), coords.copy()
        plus.flat[index] += step
        minus.flat[index] -= step
        sp, sm = solve(plus), solve(minus)
        slope = (sp.phase_velocity - sm.phase_velocity) / (2 * step)
        column = (dp.compute_kernels(sp, basis=basis).gradient
                  - dp.compute_kernels(sm, basis=basis).gradient) / (2 * step)
        np.testing.assert_allclose(result.gradient[index], slope, rtol=3e-5, atol=1e-7)
        np.testing.assert_allclose(result.hessian[:, index], column, rtol=3e-5, atol=1e-7)


def solve_grid(model):
    # These sparse grids span up to 15 km between adjacent nodes.
    return dp.compute_dispersion(model, period=5., options=dp.Options(max_steps=8, strict=True))


@pytest.mark.parametrize("interpolation", ["stiffness", "velocity"])
def test_zero_lambda_velocity_input_and_capabilities(interpolation):
    vs, rho = [2.5, 4.5], [2.8, 3.3]
    vp = [np.sqrt(2.) * vs[0], 8.]

    def model(eta):
        return dp.create_model([0., 15.], kind="grid", interpolation=interpolation, rho=rho,
                               vsh=vs, vsv=vs, vph=vp, vpv=vp, eta=eta)

    def kernels(solution, basis):
        # The variational stiffness kernel needs finer subdivision than the root.
        options = dp.Options(max_levels=12, strict=(interpolation == "stiffness"))
        return dp.compute_kernels(solution, basis=basis, options=options)

    m = model([1., 1.])
    assert m.materials[0].A == 2 * m.materials[0].L
    solution = solve_grid(m)
    for alias, standard in [("isotropic", "velocity_iso"), ("stiffness", "stiffness_vti")]:
        available = {r.basis: r.supported for r in dp.get_capabilities(m)
                     if r.operation == "kernels" and r.method == "standard"}
        assert available[alias] == available[standard]
        if standard == "stiffness_vti" and interpolation == "velocity":
            assert not available[standard]
            with pytest.raises(dp.Error, match="invertible"):
                kernels(solution, basis=standard)
        else:
            legacy = kernels(solution, basis=alias)
            current = kernels(solution, basis=standard)
            assert (legacy.basis, current.basis) == (alias, standard)
            np.testing.assert_array_equal(legacy.gradient, current.gradient)
    g = kernels(solution, basis="velocity_vti").gradient.reshape(-1, 6)
    iso = kernels(solution, basis="velocity_iso").gradient.reshape(-1, 3)
    np.testing.assert_allclose(iso, np.array([g[:, 3]+g[:, 4], g[:, 1]+g[:, 2], g[:, 0]]).T)
    elastic = kernels(solution, basis="stiffness_iso").gradient.reshape(-1, 3)
    np.testing.assert_allclose(elastic[:, 2], iso[:, 2] - (iso[:, 0]*vs + iso[:, 1]*vp)/(2*np.array(rho)))
    # A physically different eta can be specified even when F = 0 at the first node.
    alt = solve_grid(model([.8, 1.]))
    g_alt = kernels(alt, basis="velocity_vti").gradient
    step = 1e-4

    def perturbed(delta):
        varied = dp.create_model([0., 15.], kind="grid", interpolation=interpolation, rho=rho,
                                 vsh=vs, vsv=vs, vph=[vp[0]+delta, vp[1]], vpv=vp, eta=[.8, 1.])
        return solve_grid(varied).phase_velocity

    derivative = (perturbed(step) - perturbed(-step)) / (2 * step)
    np.testing.assert_allclose(g_alt[1], derivative, rtol=3e-5, atol=1e-7)
    if interpolation == "velocity":
        assert alt.phase_velocity != solution.phase_velocity
        with pytest.raises(dp.Error, match="eta=1"):
            kernels(alt, basis="velocity_iso")
    raw = dp.create_grid_model([0., 15.], m.materials, interpolation="stiffness")
    with pytest.raises(dp.Error, match="supply velocity arrays"):
        kernels(solve_grid(raw), basis="velocity_vti")


def test_lame_interface_row_projection():
    vs, vp, rho = [3., 3.4, 3.7, 4.5], [5., 6., 6.5, 8.], [2.7, 2.9, 3., 3.3]
    model = dp.create_grid_model([0., 5., 5., 15.], vs=vs, vp=vp, rho=rho, interpolation="stiffness")
    solution = solve_grid(model)

    def row(basis):
        return dp.compute_interface_hessian_row(solution, interface_index=0, basis=basis).hessian.ravel()

    full, iso = row("stiffness_vti"), row("stiffness_iso")
    A,C,F,L,N,r = full[:-1].reshape(-1,6).T
    np.testing.assert_allclose(iso[:-1].reshape(-1,3), np.column_stack([A+C+F,2*(A+C)+L+N,r]))
    assert iso[-1] == full[-1]


def test_transformed_velocity_coordinates_match_corrected_stiffness():
    fields = dict(rho=[2.5, 3.3], vph=[5.3, 8.1], vpv=[5., 8.], vsh=[3.2, 4.6],
                  vsv=[3., 4.5], eta=[.9, 1.1])
    source = dp.create_model([0., 15.], kind="grid", interpolation="velocity",
                             geometry=dp.Geometry("spherical_ba97"), **fields)
    flat = dp.earth_flattening_transformation(source)
    corrected = dp.apply_mineos_physical_dispersion(flat, [400., 550.], [300., 480.],
                                                     reference_period=1., period=5.)
    for m in (flat, corrected):
        raw = dp.create_grid_model(m.depths, m.materials, interpolation="velocity")
        expected = solve_grid(raw)
        actual = solve_grid(m)
        np.testing.assert_allclose(actual.phase_velocity, expected.phase_velocity, rtol=1e-9)
        np.testing.assert_allclose(dp.compute_kernels(actual, basis="velocity_vti").gradient,
                                   dp.compute_kernels(expected, basis="velocity_vti").gradient,
                                   rtol=1e-8, atol=1e-10)
        for basis in ("velocity_iso", "stiffness_iso"):
            with pytest.raises(dp.Error):
                dp.compute_kernels(actual, basis=basis)


def test_q_keeps_supplied_vti_coordinates_at_a_persistent_zero_lambda():
    fields = dict(rho=[2.8, 3.3], vph=[np.sqrt(2.)*2.5, 8.], vpv=[np.sqrt(2.)*2.5, 8.],
                  vsv=[2.5, 4.5], vsh=[2.5, 4.5], eta=[.8, 1.])
    m = dp.create_model([0., 15.], kind="grid", interpolation="velocity", **fields)
    corrected = dp.apply_mineos_physical_dispersion(m, [500., 500.], [500., 500.],
                                                     reference_period=1., period=5.)
    p = corrected.materials[0]
    assert p.A == 2*p.L
    sol = solve_grid(corrected)
    assert np.isfinite(dp.compute_kernels(sol, basis="velocity_vti").gradient).all()
