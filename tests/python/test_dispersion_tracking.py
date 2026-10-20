"""Mode identity and root accuracy are independent contracts for curve tracking."""
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np
import pytest
import disper26 as d26


def model(kind='pancake', interpolation=None):
    kwargs = dict(kind=kind, vs=[3.,3.5,4.5], vp=[5.2,6.,8.], rho=[2.5,2.8,3.3])
    if interpolation:
        kwargs['interpolation'] = interpolation
    return d26.create_model([0.,20.,60.], **kwargs)


def curve(m, periods, wave='Rayleigh', mode=0, outputs=('phase_velocity',), **options):
    # Unchecked tracking needs a branch-local seed. Establish it outside the
    # operation under test on level 0, without certifying the later trace.
    if options.get('method') == 'continuation' and not options.get('verify_mode') and 'c_guess' not in options:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', d26.ConvergenceWarning)
            options['c_guess'] = d26.compute_dispersion(m, period=min(periods), wave=wave,
                mode=mode, options=d26.Options(max_steps=0)).phase_velocity
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', d26.ConvergenceWarning)
        return d26.compute_dispersion_curve(m, periods=periods, wave=wave, mode=mode,
            outputs=outputs, options=d26.Options(max_steps=4, tolerance=1e-8, **options))


@pytest.mark.parametrize('kind,interpolation', [('pancake',None),('grid','stiffness'),('grid','velocity')])
@pytest.mark.parametrize('wave', ['Rayleigh','Love'])
@pytest.mark.parametrize('mode', [0,1])
def test_tracking_matches_checked(kind, interpolation, wave, mode):
    m = model(kind, interpolation)
    periods = [8.4,8.,8.2,8.2,8.6]
    checked = curve(m, periods, wave, mode, method='mode_order')
    fast = curve(m, periods, wave, mode, method='continuation')
    np.testing.assert_allclose(fast.phase_velocity, checked.phase_velocity, rtol=1e-9, atol=1e-11)
    assert all(s.diagnostics.mode_verified for s in checked.solutions)
    assert not any(s.diagnostics.mode_verified for s in fast.solutions)
    assert all(s.diagnostics.mode_count_evaluations == 0 for s in fast.solutions)
    assert fast.phase_velocity[2] == fast.phase_velocity[3]
    assert any(not s.diagnostics.mode_verified for s in fast.solutions)
    assert sum(s.diagnostics.mode_count_evaluations for s in fast.solutions) < sum(
        s.diagnostics.mode_count_evaluations for s in checked.solutions)
    for s in fast.solutions:
        assert s.diagnostics.mode_verified == all(s.diagnostics.level_mode_verified)


def test_forced_verification_and_predictor_cost():
    m = model('grid','stiffness')
    periods = [8.,8.2,8.4]
    forced = curve(m, periods, method='continuation', verify_mode=True)
    assert all(s.diagnostics.mode_verified for s in forced.solutions)
    auto = curve(m, periods)
    tangent = curve(m, periods, predictor='group_velocity')
    assert auto.group_velocity is None and tangent.group_velocity is None
    assert all(s.diagnostics.predictor_ad_evaluations == 0 for s in auto.solutions)
    assert all(s.diagnostics.predictor_ad_evaluations > 0 for s in tangent.solutions)
    assert any('group_velocity' in s.diagnostics.predictors for s in tangent.solutions[1:])
    reused = curve(m, periods, outputs=('phase_velocity','group_velocity'))
    assert all(s.diagnostics.predictor_ad_evaluations == 0 for s in reused.solutions)
    assert any('group_velocity' in s.diagnostics.predictors for s in reused.solutions[1:])
    np.testing.assert_allclose(tangent.phase_velocity, auto.phase_velocity, rtol=1e-9, atol=1e-11)


def test_fallback_and_invalid_settings():
    m = model()
    fast = curve(m, [5.,5.1,15.], method='continuation', predictor='history', c_guess=3.)
    assert fast.solutions[-1].diagnostics.fallback_count > 0
    assert not fast.solutions[-1].diagnostics.mode_verified
    assert all(s.diagnostics.mode_count_evaluations == 0 for s in fast.solutions)
    checked = curve(m, [5.,5.1,15.])
    np.testing.assert_allclose(fast.phase_velocity, checked.phase_velocity, rtol=1e-9, atol=1e-11)
    # A large jump can leave the local search domain. It must fail instead
    # of silently counting to recover the requested mode or returning a guess.
    with pytest.raises(d26.Error, match='No secular root'):
        curve(m, [5.,5.1,40.], method='continuation', predictor='history', c_guess=3.)
    for options in ({'method':'mode_order','verify_mode':False}, {'predictor':'invalid'}):
        with pytest.raises(Exception):
            curve(m, [8.], **options)
    with pytest.raises(Exception):
        curve(m, [8.], mode=100, method='mode_order')


@pytest.mark.parametrize('kind,interpolation', [('pancake',None),('grid','stiffness')])
def test_derivatives_verify_saved_trace_without_mutation(kind, interpolation):
    m = model(kind, interpolation)
    fast = curve(m, [8.,8.2], method='continuation').solutions[-1]
    assert not fast.diagnostics.mode_verified
    before = fast.phase_velocity
    def kernels(_):
        return d26.compute_kernels(fast, basis='isotropic', options=d26.Options(max_levels=4))
    with warnings.catch_warnings(), ThreadPoolExecutor(2) as pool:
        warnings.simplefilter('ignore', d26.ConvergenceWarning)
        results = list(pool.map(kernels, range(2)))
    assert all(r.diagnostics.mode_verified for r in results)
    assert not fast.diagnostics.mode_verified and fast.phase_velocity == before
    np.testing.assert_array_equal(results[0].gradient, results[1].gradient)


@pytest.mark.parametrize('interpolation', ['stiffness','velocity'])
def test_vti_discontinuity_and_spherical_grid(interpolation):
    m = d26.create_model([0.,20.,20.,60.], kind='grid', interpolation=interpolation,
        rho=[2.5,2.8,3.,3.3], vph=[5.4,6.2,6.5,8.2], vpv=[5.2,6.,6.3,8.],
        vsh=[3.1,3.6,3.8,4.6], vsv=[3.,3.5,3.7,4.5], eta=[.95,1.,.98,1.],
        geometry=d26.Geometry('spherical_ba97'))
    for wave in ['Rayleigh','Love']:
        checked = curve(m, [8.,8.2,8.4], wave)
        fast = curve(m, [8.,8.2,8.4], wave, method='continuation')
        np.testing.assert_allclose(fast.phase_velocity,checked.phase_velocity,rtol=1e-9,atol=1e-11)


def test_ocean_and_homogeneous_halfspace():
    ocean = d26.create_model([0.,3.,20.,60.], kind='pancake',
        vs=[0.,3.,3.5,4.5],vp=[1.5,5.2,6.,8.],rho=[1.,2.5,2.8,3.3])
    halfspace = d26.create_model([0.],kind='pancake',vs=[4.5],vp=[8.],rho=[3.3])
    for m in (ocean, halfspace):
        checked = curve(m,[8.,8.2,8.4],outputs=('phase_velocity','group_velocity','hv_ratio'))
        fast = curve(m,[8.,8.2,8.4],method='continuation',outputs=('phase_velocity','group_velocity','hv_ratio'))
        np.testing.assert_allclose(fast.phase_velocity,checked.phase_velocity,rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(fast.group_velocity,checked.group_velocity,rtol=1e-9,atol=1e-11)


def test_hessian_checks_tracked_pancake():
    m = model()
    tracked = curve(m,[8.,8.2],method='continuation').solutions[-1]
    checked = curve(m,[8.,8.2]).solutions[-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',d26.ConvergenceWarning)
        fast_h = d26.compute_hessian(tracked,basis='isotropic')
        checked_h = d26.compute_hessian(checked,basis='isotropic')
    assert fast_h.diagnostics.mode_verified
    np.testing.assert_allclose(fast_h.hessian,checked_h.hessian,rtol=1e-9,atol=1e-11)
    assert not tracked.diagnostics.mode_verified


def test_strict_does_not_imply_mode_verification():
    tracked = curve(model(),[8.,8.2],method='continuation',strict=True).solutions[-1]
    assert tracked.diagnostics.convergence == 'converged'
    assert not tracked.diagnostics.mode_verified


def test_frequency_curvature_checks_tracked_root():
    tracked = curve(model(),[8.,8.2],method='continuation').solutions[-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',d26.ConvergenceWarning)
        result = d26.compute_frequency_curvature(tracked)
    assert result.diagnostics.mode_verified
    assert np.isfinite(result.group_velocity)
    assert not tracked.diagnostics.mode_verified


def test_prem_second_period_uses_coarse_roots_before_history_is_ready():
    # A one-point history predicts the previous root, which is too far from
    # the 6 s root. Fine levels can still follow the current coarse roots.
    import json
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'fixtures/prem_tracking_model.json'
    nodes = json.loads(source.read_text())['nodes']
    fields = {k: [n[k] for n in nodes] for k in ('rho','vph','vpv','vsh','vsv','eta')}
    m = d26.create_model([n['depth'] for n in nodes], kind='grid',
                        interpolation='velocity', **fields)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', d26.ConvergenceWarning)
        fast = d26.compute_dispersion_curve(m, periods=[5.,6.,7.],
            options=d26.Options(max_steps=3,tolerance=1e-12,method='continuation',c_guess=3.))
        checked = d26.compute_dispersion_curve(m, periods=[5.,6.,7.],
            options=d26.Options(max_steps=3,tolerance=1e-12))
    np.testing.assert_allclose(fast.phase_velocity,checked.phase_velocity,
                               rtol=1e-9,atol=1e-11)
    assert all(s.diagnostics.mode_count_evaluations == 0 for s in fast.solutions)
    assert list(fast.solutions[1].diagnostics.level_mode_verified)[2:] == [False,False]


@pytest.mark.parametrize('kind,interpolation', [('pancake',None),('grid','stiffness'),('grid','velocity')])
@pytest.mark.parametrize('wave', ['Rayleigh','Love'])
def test_explicit_counts_saved_levels(kind, interpolation, wave):
    m = model(kind, interpolation)
    fast = curve(m, [8.,8.2,8.4], wave, mode=1, method='continuation')
    for solution in (fast.solutions[0], fast.solutions[-1]):
        before = solution.phase_velocity
        assert d26.compute_mode_count(solution) == 1
        levels = range(solution.diagnostics.generated_level+1) if kind == 'grid' else [0]
        for level in levels:
            assert d26.compute_mode_count(solution, level=level) == 1
        assert solution.phase_velocity == before
        assert solution.diagnostics.mode_count_evaluations == 0
        assert not solution.diagnostics.mode_verified
        for invalid in (-2, 99):
            with pytest.raises(d26.Error):
                d26.compute_mode_count(solution, level=invalid)


def test_unchecked_label_does_not_select_or_repair_branch():
    m = model()
    seed = d26.compute_dispersion(m, period=8., mode=1).phase_velocity
    result = curve(m, [8.,8.2], mode=0, method='continuation', c_guess=seed)
    assert all(s.diagnostics.mode_count_evaluations == 0 for s in result.solutions)
    assert d26.compute_mode_count(result.solutions[-1]) == 1
    assert result.solutions[-1].mode == 0
    with pytest.raises(d26.Error, match='No secular root'):
        curve(m, [8.], method='continuation', c_guess=.1)


@pytest.mark.parametrize('kind,interpolation', [('pancake',None),('grid','stiffness'),('grid','velocity')])
@pytest.mark.parametrize('wave', ['Rayleigh','Love'])
@pytest.mark.parametrize('mode', [0,1])
def test_recovery_tracking_selects_initial_mode(kind, interpolation, wave, mode):
    m = model(kind, interpolation)
    periods = [8.4,8.,8.2,8.2]
    # Deliberately use the same guess for both branches; recovery must select mode.
    result = curve(m, periods, wave, mode, method='continuation_recover', c_guess=3.)
    checked = curve(m, periods, wave, mode)
    np.testing.assert_allclose(result.phase_velocity, checked.phase_velocity, rtol=1e-9, atol=1e-11)
    first = result.solutions[1]
    assert first.diagnostics.mode_verified
    assert first.diagnostics.mode_count_evaluations > 0
    assert any(not s.diagnostics.mode_verified for s in result.solutions)
    for s in result.solutions:
        for level in range(s.diagnostics.generated_level+1):
            assert d26.compute_mode_count(s, level=level) == mode
    forced = curve(m, periods, wave, mode, method='continuation_recover', verify_mode=True)
    assert all(s.diagnostics.mode_verified for s in forced.solutions)


def test_recovery_after_large_period_jump():
    m = model()
    result = curve(m, [5.,5.1,40.], method='continuation_recover', predictor='history', c_guess=3.)
    last = result.solutions[-1]
    assert last.diagnostics.fallback_count > 0
    assert last.diagnostics.mode_verified
    assert last.diagnostics.mode_count_evaluations > 0
    np.testing.assert_allclose(result.phase_velocity, curve(m,[5.,5.1,40.]).phase_velocity,
                               rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize('method', ['continuation','continuation_recover'])
@pytest.mark.parametrize('kind,interpolation,n_ad', [('pancake',None,1),('grid','velocity',2)])
def test_startup_tangent_cost_and_context(method, kind, interpolation, n_ad):
    m = model(kind, interpolation)
    result = curve(m, [8.,8.2,8.4], method=method, c_guess=3.)
    assert [s.diagnostics.predictor_ad_evaluations for s in result.solutions] == [n_ad,0,0]
    assert 'group_velocity' in result.solutions[1].diagnostics.predictors
    for periods in ([8.], [8.,8.], [8.,16.]):
        result = curve(m, periods, method=method, c_guess=3.)
        assert all(s.diagnostics.predictor_ad_evaluations == 0 for s in result.solutions)
    for kwargs in ({'predictor':'history'}, {'outputs':('phase_velocity','group_velocity')}):
        result = curve(m, [8.,8.2,8.4], method=method, c_guess=3., **kwargs)
        assert all(s.diagnostics.predictor_ad_evaluations == 0 for s in result.solutions)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', d26.ConvergenceWarning)
        opts = d26.Options(method=method, max_steps=3, c_guess=3.)
        scalar = d26.compute_dispersion(m, period=8., options=opts)
        assert scalar.diagnostics.predictor_ad_evaluations == 0
        context = d26.create_dispersion_context()
        a = d26.compute_dispersion_curve(m, periods=[8.2,8.,8.], options=opts)
        for _ in range(2):
            b = d26.compute_dispersion_curve(m, periods=[8.2,8.,8.], options=opts, context=context)
            np.testing.assert_array_equal(a.phase_velocity,b.phase_velocity)
            assert [s.diagnostics.predictor_ad_evaluations for s in b.solutions] == [0,n_ad,n_ad]
