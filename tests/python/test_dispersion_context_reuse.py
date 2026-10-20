"""Explicit contexts retain capacity, never roots or caller-visible results."""
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np
import pytest
import disper26 as d26


def models():
    for kind, interpolation in [('pancake', None), ('grid', 'stiffness'), ('grid', 'velocity')]:
        for shift in (0., .1):
            yield d26.create_model([0., 20., 60.], kind=kind, interpolation=interpolation,
                vs=[3.+shift, 3.5, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])


def solve(model, context=None, mode=0, wave='Rayleigh', full=False, method='mode_order', steps=4):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', d26.ConvergenceWarning)
        return d26.compute_dispersion_curve(model, periods=[8.4, 8., 8.2, 8.2],
            wave=wave, mode=mode, outputs=('phase_velocity', 'group_velocity') if full else ('phase_velocity',),
            options=d26.Options(max_steps=steps, tolerance=1e-8, method=method), context=context)


def snapshot(curve):
    return [(s.phase_velocity, s.group_velocity, s.diagnostics.accepted_level,
        s.diagnostics.generated_level, s.diagnostics.mode_verified,
        list(s.diagnostics.predictors), list(s.diagnostics.level_mode_verified),
        s.diagnostics.mode_count_evaluations, s.diagnostics.fallback_count)
        for s in curve.solutions]


def test_changes_and_saved_results():
    context = d26.create_dispersion_context()
    saved = []
    for model in models():
        for wave in ('Rayleigh', 'Love'):
            for mode in (0, 1):
                for method in ('mode_order', 'continuation'):
                    for full in (False, True):
                        actual = solve(model, context, mode, wave, full, method)
                        assert snapshot(actual) == snapshot(solve(model, None, mode, wave, full, method))
                        saved.append((actual, snapshot(actual)))
    del context
    for result, before in saved:
        assert snapshot(result) == before


def test_errors_do_not_poison_reuse():
    context = d26.create_dispersion_context()
    model = list(models())[-1]
    first = solve(model, context)
    before = snapshot(first)
    for periods in ([], [8., float('nan')], [8., -1.]):
        with pytest.raises(Exception):
            d26.compute_dispersion_curve(model, periods=periods, context=context)
    with pytest.raises(RuntimeError, match='mode-order branch 100'):
        d26.compute_dispersion(model, period=8., mode=100, context=context)
    assert snapshot(solve(model, context)) == before
    # A smaller and then a larger level limit must rebuild valid contents.
    for steps in (0, 4, 2, 4):
        assert snapshot(solve(model, context, steps=steps)) == snapshot(solve(model, steps=steps))


@pytest.mark.parametrize('shared', [False, True])
def test_parallel_contexts(shared):
    context = d26.create_dispersion_context()
    inputs = list(models())
    def run(i):
        ctx = context if shared else d26.create_dispersion_context()
        return snapshot(solve(inputs[i % len(inputs)], ctx, wave='Love' if i % 2 else 'Rayleigh', full=True))
    reference = [run(i) for i in range(12)]
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(run, range(12))) == reference


def test_scalar_and_derivative_ownership():
    model = next(models())
    context = d26.create_dispersion_context()
    first = d26.compute_dispersion(model, period=8., context=context)
    gradient = d26.compute_kernels(first, basis='isotropic').gradient
    for other in models():
        solve(other, context)
    del context
    np.testing.assert_array_equal(d26.compute_kernels(first, basis='isotropic').gradient, gradient)


def test_geometry_fluids_halfspace_and_q_changes():
    context = d26.create_dispersion_context()
    a = d26.create_isotropic_parameters(3., 5.2, 2.5)
    b = d26.create_isotropic_parameters(4.5, 8., 3.3)
    water = d26.create_fluid_parameters(1.5, 1.03)
    inputs = [d26.create_pancake_model([], [], b),
              d26.create_pancake_model([water, a], [1., 20.], b)]
    for interpolation in ('stiffness', 'velocity'):
        for geometry in (d26.Geometry(), d26.Geometry('spherical_ba97')):
            inputs.append(d26.create_grid_model([0., 1., 1., 20., 60.],
                [water, water, a, a, b], interpolation=interpolation, geometry=geometry))
    base = d26.create_pancake_model([a], [20.], b)
    corrected = d26.apply_mineos_physical_dispersion(base, [600., 800.], [300., 400.],
                                     reference_period=1., period=8.)
    inputs.extend([corrected, base, inputs[0]])
    for model in inputs:
        actual = d26.compute_dispersion(model, period=8., context=context)
        expected = d26.compute_dispersion(model, period=8.)
        assert actual.phase_velocity == expected.phase_velocity
        assert actual.diagnostics.accepted_level == expected.diagnostics.accepted_level
    with pytest.raises(d26.Error):
        d26.compute_dispersion(corrected, period=9., context=context)
    assert snapshot(solve(base, context)) == snapshot(solve(base))
