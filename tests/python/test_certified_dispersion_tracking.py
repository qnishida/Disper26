"""Checked predictors must retain mode identity, including close GRID roots."""
import numpy as np
import pytest

import disper26 as d26
import disper26py as dp
from test_romberg_mode_order_tracking import (
    close_love_overtone_model, validation_failure_model,
)
from test_halfspace_mode_count import comparison_models


@pytest.mark.parametrize('wave', ['Rayleigh', 'Love'])
@pytest.mark.parametrize('mode', [0, 1])
def test_layer_cake_candidates_are_certified_against_fresh_search(wave, mode):
    model = d26.create_model([0., 20., 60.], kind='layer_cake',
        vs=[3., 3.5, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
    periods = [8., 8.02, 8.04, 8.06]
    options = d26.Options(root_tolerance=1e-12)
    curve = d26.compute_dispersion_curve(model, periods=periods, wave=wave,
        mode=mode, options=options)
    # Single LAYER_CAKE solves have no history or coarse level: they still use
    # the original integer-count bracket search, independently of this path.
    reference = [d26.compute_dispersion(model, period=p, wave=wave,
                 mode=mode, options=options) for p in periods]
    np.testing.assert_allclose(curve.phase_velocity,
        [s.phase_velocity for s in reference], rtol=1e-9, atol=1e-11)
    for s in curve.solutions:
        assert s.diagnostics.mode_verified
        assert all(s.diagnostics.level_mode_verified)
    # On an uncomplicated branch the two endpoint checks replace bisection.
    if wave == 'Rayleigh':
        assert any(s.diagnostics.mode_count_evaluations == 2
                   and s.diagnostics.fallback_count == 0 for s in curve.solutions[1:])


@pytest.mark.parametrize('name', ['solid', 'ocean', 'vti', 'lvz'])
@pytest.mark.parametrize('method', ['mode_order', 'continuation'])
def test_rayleigh_highest_branch_tracks_without_relabeling(name, method):
    _, layers, thicknesses, halfspace = next(m for m in comparison_models() if m[0] == name)
    params = layers + [halfspace]
    depths = np.r_[0., np.cumsum(thicknesses)].tolist()
    fields = {key: [getattr(p, attr) for p in params] for key, attr in
              [('rho','rho'), ('A','c11'), ('C','c33'), ('F','c13'), ('L','c44'), ('N','c66')]}
    model = d26.create_model(depths, kind='layer_cake', **fields)
    roots = dp.search_roots(layers, thicknesses, halfspace, np.pi, .05,
        .9995*np.sqrt(halfspace.c44/halfspace.rho), 4000, 'Rayleigh')
    assert len(roots) > 1
    mode = len(roots)-1
    periods = [1.999, 2., 2.001]
    reference = [d26.compute_dispersion(model, period=p, mode=mode).phase_velocity
                 for p in periods]
    curve = d26.compute_dispersion_curve(model, periods=periods, mode=mode,
        options=d26.Options(method=method, c_guess=reference[0]))
    np.testing.assert_allclose(curve.phase_velocity, reference, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(curve.phase_velocity[1], sorted(roots)[mode],
                               rtol=1e-9, atol=1e-11)
    assert curve.solutions[0].diagnostics.mode_verified == (method == "mode_order")
    if method == 'mode_order':
        assert all(s.diagnostics.mode_verified for s in curve.solutions)


@pytest.mark.parametrize('close', [False, True])
def test_grid_overtone_matches_original_count_search(close):
    source = close_love_overtone_model() if close else validation_failure_model()
    depths, params, discontinuities = source.to_params()
    fields = {name: [getattr(p, attr) for p in params] for name, attr in
              [('rho','rho'), ('A','c11'), ('C','c33'), ('F','c13'), ('L','c44'), ('N','c66')]}
    policy = source.refinement_kwargs()
    model = d26.create_model(depths, kind='grid', interpolation='stiffness',
        geometry=d26.Geometry(policy['grid_policy'], policy['planet_radius'],
                              policy['flattening_exponent']), **fields)
    centre = 1. / (0.17 if close else 0.12625)
    periods = [centre-.01, centre, centre+.01]
    # The raw GRID API still uses the original count-bracket solver.
    reference = [dp.compute_dispersion_romberg(depths, params, discontinuities,
        p, 'Love', 4.53, 6, 1e-9, branch=1, phase_only=True,
        want_value_u=False, want_value_hv=False, **policy)[0] for p in periods]
    curve = dp._api.compute_dispersion_curve(model, periods, 'Love', 1,
        ['phase_velocity'], dp._api.Options(max_steps=6, tolerance=1e-9, c_guess=4.53))
    np.testing.assert_allclose(curve.phase_velocity, reference, rtol=1e-9, atol=1e-11)
    for s in curve.solutions:
        assert s.diagnostics.mode_verified
        assert len(s.diagnostics.level_mode_verified) > 1
        assert all(s.diagnostics.level_mode_verified)
