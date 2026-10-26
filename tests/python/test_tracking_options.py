"""Direction, initial brackets and explicitly selected mode guarantees."""
import warnings

import numpy as np
import pytest

import disper26 as d26


def solid(kind='pancake', interpolation=None):
    fields = dict(vs=[3., 3.5, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
    if interpolation:
        fields['interpolation'] = interpolation
    return d26.create_model([0., 20., 60.], kind=kind, **fields)


def solve(model, periods, wave='Rayleigh', mode=0, context=None, **settings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', d26.ConvergenceWarning)
        return d26.compute_dispersion_curve(model, periods=periods, wave=wave, mode=mode,
            options=d26.Options(max_steps=4, tolerance=1e-8, **settings), context=context)


@pytest.mark.parametrize('kind,interpolation', [('pancake', None), ('grid', 'stiffness'), ('grid', 'velocity')])
@pytest.mark.parametrize('wave', ['Rayleigh', 'Love'])
@pytest.mark.parametrize('mode', [0, 1])
def test_both_directions_match_fresh_mode_order(kind, interpolation, wave, mode):
    model = solid(kind, interpolation)
    periods = [8.2, 8., 8.4, 8.2]
    short = solve(model, periods, wave, mode, start='short', verify='every')
    long = solve(model, periods, wave, mode, start='long', verify='every')
    np.testing.assert_allclose(short.phase_velocity, long.phase_velocity, rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(long.periods, periods)
    assert long.phase_velocity[0] == long.phase_velocity[3]
    assert all(s.diagnostics.mode_verified for s in long.solutions)
    reused = solve(model, periods, wave, mode, verify='every', context=d26.create_dispersion_context())
    np.testing.assert_array_equal(reused.phase_velocity, long.phase_velocity)


def test_default_starts_at_longest_period_and_legacy_frequency_is_kept():
    model = solid()
    periods = [8., 8.02, 8.04]
    result = solve(model, periods, method='continuation_recover')
    assert result.solutions[-1].diagnostics.mode_verified
    assert not result.solutions[0].diagnostics.mode_verified
    np.testing.assert_array_equal(result.phase_velocity,
        solve(model, periods, start='long', method='continuation_recover').phase_velocity)
    first = result.solutions[-1].phase_velocity
    unchecked = solve(model, periods, method='continuation', c_guess=first)
    assert all(s.diagnostics.mode_count_evaluations == 0 for s in unchecked.solutions)


@pytest.mark.parametrize('wave', ['Rayleigh', 'Love'])
@pytest.mark.parametrize('start', ['short', 'long'])
def test_initial_range_uses_two_counts_and_ignores_guess(wave, start):
    model = solid()
    periods = [8., 8.02, 8.04]
    reference = solve(model, periods, wave, start=start, verify='every')
    index = 0 if start == 'short' else -1
    c = reference.phase_velocity[index]
    result = solve(model, periods, wave, start=start, verify='endpoints',
                   c_range=(c*.999, c*1.001), c_guess=1.)
    np.testing.assert_allclose(result.phase_velocity, reference.phase_velocity, rtol=1e-9, atol=1e-11)
    assert result.solutions[index].diagnostics.mode_count_evaluations == 2
    assert result.solutions[index].diagnostics.predictors == ['initial_range']
    assert result.solutions[0].diagnostics.mode_verified
    assert result.solutions[-1].diagnostics.mode_verified
    assert not result.solutions[1].diagnostics.mode_verified


@pytest.mark.parametrize('response', ['fallback', 'error', 'missing'])
def test_wrong_range_obeys_violation_policy(response):
    model = solid()
    correct = d26.compute_dispersion(model, period=8.)
    adjacent = d26.compute_dispersion(model, period=8., mode=1).phase_velocity
    settings = dict(verify='every', c_range=(adjacent*.999, adjacent*1.001), on_violation=response)
    if response == 'error':
        with pytest.raises(d26.Error) as failure:
            solve(model, [8.], **settings)
        assert failure.value.code == 'tracking_violation'
    else:
        result = solve(model, [8.], **settings).solutions[0]
        if response == 'fallback':
            assert result.phase_velocity == pytest.approx(correct.phase_velocity, rel=1e-9, abs=1e-11)
            assert result.diagnostics.mode_verified
            assert 'initial_range_mode_count' in result.diagnostics.fallback_reasons
        else:
            assert np.isnan(result.phase_velocity)
            assert result.diagnostics.missing_reason == 'tracking_violation'
            assert not result.diagnostics.mode_verified
            with pytest.raises(d26.Error):
                solve(model, [8.], strict=True, **settings)


def test_unchecked_range_has_no_number_guarantee_but_recovers_ambiguity():
    model = solid()
    c0 = d26.compute_dispersion(model, period=8.).phase_velocity
    c1 = d26.compute_dispersion(model, period=8., mode=1).phase_velocity
    unverified = solve(model, [8.], verify='none', c_range=(c1*.999, c1*1.001)).solutions[0]
    assert unverified.mode == 0
    assert unverified.phase_velocity == pytest.approx(c1, rel=1e-9, abs=1e-11)
    assert unverified.diagnostics.mode_count_evaluations == 0
    assert not unverified.diagnostics.mode_verified
    recovered = solve(model, [8.], verify='none', c_range=(c0*.999, c1*1.001)).solutions[0]
    assert recovered.phase_velocity == pytest.approx(c0, rel=1e-9, abs=1e-11)
    assert recovered.diagnostics.mode_verified
    assert recovered.diagnostics.mode_count_evaluations > 0
    with pytest.raises(d26.Error):
        solve(model, [8.], verify='none', c_range=(c0*.999, c1*1.001), on_violation='error')


@pytest.mark.parametrize('start', ['short', 'long'])
def test_cutoff_missing_periods_do_not_consume_initial_range(start):
    model = d26.create_model([0., 10.], kind='pancake', vs=[3., 4.5], vp=[5.2, 8.], rho=[2.6, 3.3])
    periods = [2., 3., 8., 10., 8.]
    seed_period = 2. if start == 'short' else 3.
    root = d26.compute_dispersion(model, period=seed_period, wave='Love', mode=1).phase_velocity
    result = solve(model, periods, wave='Love', mode=1, start=start,
                   verify='endpoints', c_range=(root*.999, root*1.001))
    assert np.isfinite(result.phase_velocity[:2]).all()
    assert np.isnan(result.phase_velocity[2:]).all()
    first = result.solutions[0 if start == 'short' else 1]
    assert first.diagnostics.mode_count_evaluations == 2
    assert first.diagnostics.predictors == ['initial_range']
    assert all(s.diagnostics.mode_verified for s in result.solutions[:2])
    assert all(s.diagnostics.missing_reason == 'mode_not_found' for s in result.solutions[2:])
    with pytest.raises(d26.Error) as failure:
        solve(model, periods, wave='Love', mode=1, start=start, strict=True)
    assert failure.value.code == 'mode_not_found'


def test_missing_solution_rejects_derived_calculation_and_context_survives():
    model = solid()
    context = d26.create_dispersion_context()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', d26.ConvergenceWarning)
        result = d26.compute_dispersion_curve(model, periods=[8., 8.2], mode=100,
            outputs=['phase_velocity', 'group_velocity', 'hv_ratio'], context=context)
    assert np.isnan(result.phase_velocity).all()
    for solution in result.solutions:
        assert solution.diagnostics.status == 'missing'
        assert solution.diagnostics.missing_reason == 'mode_not_found'
        assert not solution.diagnostics.mode_verified
        assert np.isnan(solution.group_velocity) and np.isnan(solution.hv_ratio)
        for operation in (lambda: d26.compute_mode_count(solution),
                          lambda: d26.compute_kernels(solution, basis='isotropic'),
                          lambda: d26.compute_hessian(solution, basis='isotropic')):
            with pytest.raises(d26.Error) as failure:
                operation()
            assert failure.value.code == 'missing_solution'
    np.testing.assert_array_equal(solve(model, [8., 8.2], context=context).phase_velocity,
                                  solve(model, [8., 8.2]).phase_velocity)


@pytest.mark.parametrize('settings', [dict(start='invalid'), dict(verify='invalid'),
    dict(on_violation='invalid'), dict(c_range=(0., 4.)), dict(c_range=(4., 3.)),
    dict(c_range=(3., np.inf)), dict(c_range=(np.nan, 4.)), dict(c_range=(4.5, 5.)),
    dict(verify='every', verify_mode=True)])
def test_invalid_options_are_errors_instead_of_missing(settings):
    with pytest.raises(d26.Error) as failure:
        solve(solid(), [8.], **settings)
    assert failure.value.code == 'invalid_argument'


def test_duplicate_only_endpoints_are_checked_once():
    model = solid()
    single = solve(model, [8.], verify='endpoints')
    repeated = solve(model, [8., 8., 8.], verify='endpoints')
    assert all(s.diagnostics.mode_count_evaluations == single.solutions[0].diagnostics.mode_count_evaluations
               for s in repeated.solutions)
    assert all(s.diagnostics.mode_verified for s in repeated.solutions)


def test_checked_love_initialization_is_not_a_tracking_violation():
    result = solve(solid(), [8., 8.02, 8.04], wave='Love', verify='every', on_violation='error')
    assert all(s.diagnostics.mode_verified for s in result.solutions)


def test_unchecked_initial_window_above_cap_can_recover():
    result = solve(solid(), [8.], verify='none', c_guess=6.).solutions[0]
    assert result.phase_velocity < .9995*4.5
    assert result.diagnostics.mode_verified


@pytest.mark.parametrize('interpolation', ['stiffness', 'velocity'])
def test_initial_range_is_certified_on_each_grid_level(interpolation):
    model = solid('grid', interpolation)
    seed = solve(model, [8.], verify='every').phase_velocity[0]
    result = solve(model, [8.], verify='every', c_range=(seed-.1, seed+.1)).solutions[0]
    assert result.diagnostics.mode_verified
    assert all(p == 'initial_range' for p in result.diagnostics.predictors)
    assert result.diagnostics.fallback_count == 0
    assert result.diagnostics.mode_count_evaluations == 2*len(result.diagnostics.level_mode_verified)


@pytest.mark.parametrize('name', ['solid', 'ocean', 'vti', 'lvz'])
@pytest.mark.parametrize('start', ['short', 'long'])
def test_ranges_cover_vti_fluid_and_low_velocity_models(name, start):
    from test_halfspace_mode_count import comparison_models
    _, layers, thicknesses, halfspace = next(m for m in comparison_models() if m[0] == name)
    parameters = layers+[halfspace]
    fields = {key: [getattr(p, attr) for p in parameters] for key, attr in
              [('rho', 'rho'), ('A', 'c11'), ('C', 'c33'), ('F', 'c13'), ('L', 'c44'), ('N', 'c66')]}
    model = d26.create_model(np.r_[0., np.cumsum(thicknesses)], kind='pancake', **fields)
    periods = [1.999, 2., 2.001]
    checked = solve(model, periods, start=start, verify='every')
    seed = checked.phase_velocity[0 if start == 'short' else -1]
    ranged = solve(model, periods, start=start, verify='every', c_range=(seed*.98, seed*1.02))
    np.testing.assert_allclose(ranged.phase_velocity, checked.phase_velocity, rtol=1e-9, atol=1e-11)
    assert all(s.diagnostics.mode_verified for s in ranged.solutions)


@pytest.mark.parametrize('close', [False, True])
def test_initial_range_preserves_close_or_tangential_love_root(close):
    from test_romberg_mode_order_tracking import close_love_overtone_model, validation_failure_model
    source = close_love_overtone_model() if close else validation_failure_model()
    depths, parameters, _ = source.to_params()
    fields = {key: [getattr(p, attr) for p in parameters] for key, attr in
              [('rho', 'rho'), ('A', 'c11'), ('C', 'c33'), ('F', 'c13'), ('L', 'c44'), ('N', 'c66')]}
    policy = source.refinement_kwargs()
    model = d26.create_model(depths, kind='grid', interpolation='stiffness',
        geometry=d26.Geometry(policy['grid_policy'], policy['planet_radius'], policy['flattening_exponent']), **fields)
    period = 1./(.17 if close else .12625)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', d26.ConvergenceWarning)
        reference = d26.compute_dispersion(model, period=period, wave='Love', mode=1,
            options=d26.Options(max_steps=6, tolerance=1e-9, c_guess=4.53))
        ranged = d26.compute_dispersion(model, period=period, wave='Love', mode=1,
            options=d26.Options(max_steps=6, tolerance=1e-9, verify='every', c_range=(4.4, 4.6)))
    assert abs(ranged.phase_velocity-reference.phase_velocity) < 5e-7
    assert ranged.diagnostics.mode_verified


@pytest.mark.parametrize('wave', ['Rayleigh', 'Love'])
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('surface_vs', [2.9, 3.1])
def test_range_from_previous_model_recovers_after_model_change(wave, mode, surface_vs):
    periods = [8., 8.2, 8.4]
    previous = solve(solid(), periods, wave, mode, verify='every')
    seed = previous.phase_velocity[-1]
    changed = d26.create_model([0., 20., 60.], kind='pancake',
        vs=[surface_vs, 3.5, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
    reference = solve(changed, periods, wave, mode, verify='every')
    reused = solve(changed, periods, wave, mode, verify='every', c_range=(seed-1e-5, seed+1e-5))
    np.testing.assert_allclose(reused.phase_velocity, reference.phase_velocity, rtol=1e-9, atol=1e-11)
    assert 'initial_range_mode_count' in reused.solutions[-1].diagnostics.fallback_reasons
    assert all(s.diagnostics.mode_verified for s in reused.solutions)


@pytest.mark.parametrize('kind,interpolation', [('pancake', None), ('grid', 'stiffness'), ('grid', 'velocity')])
@pytest.mark.parametrize('start', ['short', 'long'])
@pytest.mark.parametrize('settings', [dict(method='standard'), dict(method='mode_order'),
    dict(method='continuation', verify_mode=True), dict(verify='every')])
def test_checked_auto_computes_only_the_usable_startup_tangent(kind, interpolation, start, settings):
    model = solid(kind, interpolation)
    periods = [8., 8.2, 8.4]
    automatic = solve(model, periods, start=start, **settings)
    reference = solve(model, periods, start=start, predictor='history', **settings)
    np.testing.assert_allclose(automatic.phase_velocity, reference.phase_velocity, rtol=1e-9, atol=1e-11)
    ordered = automatic.solutions if start == 'short' else list(reversed(automatic.solutions))
    assert [s.diagnostics.predictor_ad_evaluations for s in ordered] == [1, 0, 0]
    assert ordered[1].diagnostics.predictors[0] == 'group_velocity'
    assert all(s.diagnostics.mode_verified for s in automatic.solutions)
    assert automatic.group_velocity is None
    if kind == 'grid':
        assert all(p == 'coarse_level' for p in ordered[1].diagnostics.predictors[1:])


def test_checked_long_start_tangent_avoids_the_second_point_recovery():
    count = 64
    midpoint = (np.arange(count, dtype=float)+.5)/count
    vs = 2.5+1.5*midpoint
    model = d26.create_model(np.linspace(0., 80., count+1), kind='pancake',
        vs=np.r_[vs, 4.5], vp=np.r_[1.75*vs, 8.], rho=np.r_[2.4+.7*midpoint, 3.3])
    periods = np.geomspace(2., 50., 64)
    automatic = solve(model, periods, start='long', verify='every', strict=True)
    history = solve(model, periods, start='long', verify='every', predictor='history', strict=True)
    np.testing.assert_allclose(automatic.phase_velocity, history.phase_velocity, rtol=1e-9, atol=1e-11)
    second = automatic.solutions[-2].diagnostics
    assert second.predictors == ['group_velocity']
    assert second.mode_count_evaluations == 2
    assert second.fallback_count == 0
    assert history.solutions[-2].diagnostics.fallback_count > 0
    assert sum(s.diagnostics.mode_count_evaluations for s in automatic.solutions) < sum(
        s.diagnostics.mode_count_evaluations for s in history.solutions)
    assert sum(s.diagnostics.predictor_ad_evaluations for s in automatic.solutions) == 1
    assert all(s.diagnostics.mode_verified for s in automatic.solutions)
