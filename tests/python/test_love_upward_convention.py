"""Public compatibility boundaries for the upward-coordinate SH refactor.

1e-12 matches the regular SH value tests for direct elementary solutions;
1e-10 matches energy-integral tests when numerical quadrature is involved.
"""
import math

import numpy as np
import pytest

import disper26 as api
import disper26py as dp


def elementary_matrix(p, omega, c, distance):
    q = (p.c66 * (omega / c)**2 - p.rho * omega**2) / p.c44
    if q > 0:
        nu = math.sqrt(q)
        C, S = math.cosh(nu * distance), math.sinh(nu * distance) / nu
    elif q < 0:
        mu = math.sqrt(-q)
        C, S = math.cos(mu * distance), math.sin(mu * distance) / mu
    else:
        C, S = 1., distance
    return np.array([[C, S / p.c44], [p.c44 * q * S, C]])


@pytest.mark.parametrize('c', [.4, .5, .7])
def test_signed_layer_matrix_and_legacy_depth_utilities(c):
    p = dp.create_isotropic_parameters(.5, 1., 2.)
    omega, h, depth = .7, .3, .2
    # Matrix arguments are signed upward increments. The generic depth APIs
    # retain their historical [u,c44*du/ddepth] vector at the Python boundary.
    for distance in (-depth, 0., depth):
        np.testing.assert_allclose(dp.compute_layer_matrix_sh(p, distance, omega, c),
                                   elementary_matrix(p, omega, c, distance),
                                   rtol=1e-12, atol=1e-12)
    y_depth = np.array([.7, -.2], dtype=complex)
    original = y_depth.copy()
    expected = elementary_matrix(p, omega, c, depth) @ y_depth
    for mode in ('LAYER_CAKE', 'GRID'):
        actual = dp.eval_layer_state(y_depth, p, p, h, depth, omega, c, 'Love', mode, 4, 1e-12)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    actual = dp.propagate_layer_quadrature_vector(y_depth, p, p, h, depth, omega, c, 'Love', 4, 1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    actual = dp.eval_global_state([p], [h], p, y_depth, depth, omega, c, 'Love', 'LAYER_CAKE', 4, 1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(y_depth, original)

    nodes, weights = np.polynomial.legendre.leggauss(24)
    integrals = np.zeros(3)
    for x, weight in zip(h * .5 * (1 + nodes), h * .5 * weights):
        u, tau_depth = elementary_matrix(p, omega, c, x) @ y_depth
        integrals += weight * np.real([p.rho*u*u,
            tau_depth*tau_depth/p.c44+(omega/c)**2*p.c66*u*u, p.c66*u*u])
    result = dp.integrate_grid_layer_quadrature(y_depth, p, p, h, omega, c, 'Love', 6, 1e-12, True)
    np.testing.assert_allclose(result[:3], integrals, rtol=1e-10, atol=1e-12)


def test_public_secular_keeps_its_historical_sign_off_root():
    p = dp.create_isotropic_parameters(.5, 1., 2.)
    hs = dp.create_isotropic_parameters(1., 2., 3.)
    omega, c, h = .7, .6, .3
    eta = math.sqrt(hs.c44 * (hs.c66*(omega/c)**2-hs.rho*omega**2))
    W, T = elementary_matrix(p, omega, c, h) @ np.array([1., eta])
    assert abs(T) > .1
    np.testing.assert_allclose(dp.compute_dispersion_value([p], [h], hs, omega, c, 'Love'),
                               -T, rtol=1e-12, atol=1e-12)


def test_legacy_phase_and_signed_sequence_keep_their_values():
    # Pre-refactor diagnostic value with nonzero traction at both interfaces.
    # The old phase samples the reverse direction within each layer. It is a
    # compatibility diagnostic, independent of the production crossing signs.
    iso = dp.create_isotropic_parameters
    layers = [iso(2.2, 3.8, 2.5), iso(1., 1.73, 2.)]
    hs = iso(3., 5.2, 2.7)
    args = (layers, [3., 1.], hs, math.pi, 1.2, 'Love')
    np.testing.assert_allclose(dp.compute_continuous_mode_count(*args),
                               .6184148809783636, rtol=1e-12, atol=1e-12)
    info = dp.debug_mode_count(*args)
    assert info['n_integer'] == info['signed_count'] == 0
    assert [x['contribution'] for x in info['crossings']] == [-1, 1]
    for crossing in info['crossings']:
        assert crossing['contribution'] == int(np.sign(crossing['test_value']))


@pytest.mark.parametrize('anisotropic', [False, True])
@pytest.mark.parametrize('mode', [0, 1])
def test_public_stress_is_the_upward_displacement_derivative(anisotropic, mode):
    # Same multilayer fixture as the energy regression, with nonzero traction
    # at both material interfaces. A displacement-only comparison misses a
    # simultaneous sign error in the reconstructed traction.
    vs, vp, rho = np.linspace(3., 4.5, 4), np.linspace(5.2, 8., 4), np.linspace(2.5, 3.3, 4)
    args = dict(rho=rho)
    if anisotropic:
        args.update(vsv=vs, vsh=1.07*vs, vpv=vp, vph=1.03*vp, eta=np.full(4, .95))
    else:
        args.update(vs=vs, vp=vp)
    model = api.create_model(np.linspace(0., 100., 4), kind='layer_cake', **args)
    solution = api.compute_dispersion(model, period=20., wave='Love', mode=mode)
    ef = api.compute_eigenfunctions(solution, num_sub_layers=400)
    for i, material in enumerate(model.materials):
        # Keep every finite-difference stencil inside one homogeneous layer.
        u = ef.components['uy'][i*400:(i+1)*400+1]
        T = ef.components['tyz'][i*400:(i+1)*400+1]
        dd = model.thicknesses[i] / 400
        du_dd = (u[:-4]-8*u[1:-3]+8*u[3:-1]-u[4:]) / (12*dd)
        # Fourth-order truncation is bounded by (|nu|*dd)^4/30 relative
        # (<1e-10 for these fixtures). 1e-9 also allows propagation/stencil
        # roundoff; the absolute term covers points near a traction zero.
        np.testing.assert_allclose(T[2:-2], -material.L*du_dd, rtol=1e-9, atol=1e-10)
