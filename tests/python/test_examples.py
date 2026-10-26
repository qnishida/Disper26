"""The public practice examples must run and demonstrate their stated result."""
import numpy as np

from examples import basic_dispersion, gradient_convergence, kernels_hessian, tracking_options


def test_layered_dispersion_has_physical_fundamental_velocities():
    curves = basic_dispersion.run()
    for wave, curve in curves.items():
        assert np.all(np.isfinite(curve.phase_velocity))
        assert np.all(curve.phase_velocity > 0.0)
        assert np.all(curve.phase_velocity < 4.5)  # Halfspace shear velocity.
        assert all(s.diagnostics.mode_verified for s in curve.solutions)
        if wave == "Love":
            assert np.all(curve.phase_velocity > 3.0)  # Layer shear velocity.


def test_tracking_example_checks_endpoints_and_keeps_input_order():
    curve = tracking_options.run()
    np.testing.assert_array_equal(curve.periods, [8., 8.2, 8.4, 8.6])
    assert np.isfinite(curve.phase_velocity).all()
    assert curve.solutions[0].diagnostics.mode_verified
    assert curve.solutions[-1].diagnostics.mode_verified
    assert not any(s.diagnostics.mode_verified for s in curve.solutions[1:-1])


def test_gradient_example_reaches_its_requested_phase_tolerance():
    solutions = gradient_convergence.run()
    final = solutions[-1]
    assert all(np.isfinite(s.phase_velocity) for s in solutions)
    assert final.diagnostics.convergence == "converged"
    assert final.diagnostics.relative_error <= final.options.tolerance
    assert final.diagnostics.accepted_level <= final.options.max_steps


def test_hessian_improves_the_small_vs_perturbation_predictions():
    solution, kernels, hessian, rows = kernels_hessian.run()
    kernel_axes = [(p.entity, p.index, p.name) for p in kernels.parameters]
    assert kernel_axes == [(p.entity, p.index, p.name) for p in hessian.parameters]
    assert hessian.hessian.shape == (len(kernel_axes), len(kernel_axes))
    for _, exact, linear, quadratic in rows:
        assert np.isfinite([exact, linear, quadratic]).all()
        assert exact > solution.phase_velocity
        assert abs(quadratic - exact) < abs(linear - exact)
