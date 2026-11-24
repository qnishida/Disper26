"""Public related operations versus established low-level implementations.

Run in the project environment: python -m unittest discover -s tests/python -p test_public_api_related.py
Direct adapter comparisons use 1e-12 relative tolerance. Gravity allows 1e-9
relative/1e-12 km/s absolute for machine-polished root and frequency-difference
roundoff. Frequency FD uses the existing curvature suite's step-error bound.
"""
import importlib.util
import math
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

import disper26 as dp
import disper26py as raw


def legacy_material(material):
    result = raw.PhysicalParameters()
    for public, old in (("rho", "rho"), ("A", "c11"), ("C", "c33"),
                        ("F", "c13"), ("L", "c44"), ("N", "c66"),
                        ("gravity", "gravity"),
                        ("gravity_internal_scale", "gravity_internal_scale"),
                        ("gravity_surface_scale", "gravity_surface_scale")):
        setattr(result, old, getattr(material, public))
    return result


def fields(material, legacy=False):
    names = ("rho", "c11", "c33", "c13", "c44", "c66") if legacy else (
        "rho", "A", "C", "F", "L", "N")
    return np.array([getattr(material, name) for name in names])


class RelatedPublicAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layer = dp.create_isotropic_parameters(3.2, 5.8, 2.6)
        cls.halfspace = dp.create_isotropic_parameters(4.5, 8.1, 3.35)
        cls.model = dp.create_layer_cake_model([cls.layer], [20.], cls.halfspace)
        cls.raw_layer = legacy_material(cls.layer)
        cls.raw_halfspace = legacy_material(cls.halfspace)
        cls.fluid = dp.create_fluid_parameters(1.5, 1.03)
        cls.ocean = dp.create_layer_cake_model([cls.fluid, cls.layer], [4., 20.], cls.halfspace)

    def test_eigenfunctions_and_both_energy_methods_match_legacy(self):
        for wave in ("Rayleigh", "Love"):
            with self.subTest(wave=wave):
                solution = dp.compute_dispersion(self.model, period=5., wave=wave)
                omega = 2 * math.pi / solution.period
                direct = getattr(raw, f"compute_eigenfunctions_{wave.lower()}")(
                    [self.raw_layer], [20.], self.raw_halfspace, omega,
                    solution.phase_velocity, 8)
                result = dp.compute_eigenfunctions(solution, num_sub_layers=8)
                np.testing.assert_array_equal(result.depth, direct["depth"])
                for component, values in result.components.items():
                    np.testing.assert_allclose(values, direct[component], rtol=1e-12, atol=1e-13)
                self.assertFalse(np.any(result.is_fluid))
                for integration in ("analytical", "trapezoidal"):
                    result = dp.compute_energy_integrals(solution,
                        options=dp.Options(integration=integration, num_sub_layers=8))
                    args = (direct, [self.raw_layer], [20.], self.raw_halfspace,
                            omega, solution.phase_velocity)
                    expected = (getattr(raw, f"compute_energy_integrals_analytical_{wave.lower()}")(*args)
                                if integration == "analytical" else raw.compute_energy_integrals(*args, wave))
                    np.testing.assert_allclose([result.I1, result.I2, result.I3],
                                               expected[:3], rtol=1e-12, atol=1e-13)
                    self.assertAlmostEqual(result.group_velocity,
                        result.I3 / (result.phase_velocity * result.I1), places=13)
                    self.assertEqual(result.diagnostics.method, integration)

    def test_strict_solution_and_gravity_mode_validation(self):
        sol = dp.compute_dispersion(self.model, period=5., options=dp.Options(strict=True))
        for operation in (dp.compute_eigenfunctions, dp.compute_gravity_correction):
            self.assertEqual(operation(sol).diagnostics.convergence, "not_evaluated")
        missing = dp.compute_dispersion_curve(self.model, periods=[5.], mode=99).solutions[0]
        unchecked = dp.compute_dispersion(self.model, period=5., mode=1,
            options=dp.Options(method="continuation", c_guess=sol.phase_velocity, verify_mode=False))
        for invalid, code in ((missing, "missing_solution"), (unchecked, "mode_mismatch")):
            with self.assertRaises(dp.Error) as raised:
                dp.compute_gravity_correction(invalid)
            self.assertEqual(raised.exception.code, code)

    def test_fluid_energy_uses_analytical_and_keeps_fluid_flags(self):
        solution = dp.compute_dispersion(self.ocean, period=5.)
        result = dp.compute_eigenfunctions(solution)
        self.assertTrue(np.any(result.is_fluid))
        self.assertTrue(np.any(~result.is_fluid))
        # A top ocean stack selects the closed-form liquid moments by default;
        # the trapezoidal integrals remain available as an explicit check.
        automatic = dp.compute_energy_integrals(solution)
        analytical = dp.compute_energy_integrals(solution,
            options=dp.Options(integration="analytical"))
        trapezoidal = dp.compute_energy_integrals(solution,
            options=dp.Options(integration="trapezoidal"))
        self.assertEqual(automatic.diagnostics.method, "analytical")
        self.assertEqual(automatic.I1, analytical.I1)
        self.assertEqual(trapezoidal.diagnostics.method, "trapezoidal")
        self.assertEqual(analytical.diagnostics.method, "analytical")
        reference = dp.compute_dispersion(self.ocean, period=5.,
            outputs=("phase_velocity", "group_velocity")).group_velocity
        np.testing.assert_allclose(analytical.group_velocity, reference, rtol=1e-9, atol=0.)
        np.testing.assert_allclose(analytical.I2,
            (2 * np.pi / solution.period)**2 * analytical.I1, rtol=1e-9, atol=0.)

    def test_grid_trace_extension_is_shared_without_changing_solution(self):
        grid = dp.create_grid_model([0., 20., 40.], interpolation="stiffness",
            vs=[3.2, 3.6, 4.5], vp=[5.8, 6.4, 8.1], rho=[2.6, 2.8, 3.35])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", dp.ConvergenceWarning)
            # Repeated periods produce separate Python handles of one shared solution.
            curve = dp.compute_dispersion_curve(grid, periods=[5., 5.],
                options=dp.Options(max_steps=0))
        first, second = curve.solutions
        initial = first.phase_velocity
        original_level = first.diagnostics.accepted_level
        options = dp.Options(max_steps=2)
        raw.reset_dispersion_eval_counts()
        a = dp.compute_frequency_curvature(first, options=options)
        first_count = raw.get_dispersion_eval_counts()["double"]
        raw.reset_dispersion_eval_counts()
        b = dp.compute_frequency_curvature(second, options=options)
        repeat_count = raw.get_dispersion_eval_counts()["double"]
        self.assertGreater(first_count, repeat_count)
        self.assertEqual(a.group_velocity_omega_derivative, b.group_velocity_omega_derivative)
        self.assertEqual(first.phase_velocity, initial)
        self.assertEqual(second.phase_velocity, initial)
        self.assertEqual(first.diagnostics.accepted_level, original_level)
        energy = dp.compute_energy_integrals(first)
        self.assertEqual(energy.diagnostics.accepted_level, original_level)
        self.assertEqual(energy.phase_velocity, initial)  # Original level is zero.
        self.assertTrue(any("not extrapolated" in text for text in energy.diagnostics.messages))

    def test_layer_cake_frequency_curvature_against_resolved_frequency_fd(self):
        solution = dp.compute_dispersion(self.model, period=5.)
        result = dp.compute_frequency_curvature(solution)
        omega = 2 * math.pi / solution.period
        estimates = []
        for relative_step in (2e-3, 1e-3):
            step = omega * relative_step
            values = {}
            for offset in (-2, -1, 1, 2):
                shifted = dp.compute_dispersion(self.model,
                    period=2 * math.pi / (omega + offset * step),
                    outputs=("group_velocity",))
                values[offset] = shifted.group_velocity
            estimates.append((values[-2] - 8 * values[-1] +
                              8 * values[1] - values[2]) / (12 * step))
        fine, medium = estimates[-1], estimates[-2]
        uncertainty = 8 * abs(fine - medium) + 2e-7 * max(
            1., abs(result.group_velocity_omega_derivative))
        self.assertLessEqual(abs(result.group_velocity_omega_derivative - fine), uncertainty)
        self.assertAlmostEqual(result.wavenumber_omega_second_derivative,
            -result.group_velocity_omega_derivative / result.group_velocity**2, places=13)

    def test_eft_matches_legacy_and_transforms_layer_cake_halfspace_at_its_top(self):
        for scheme, exponent, suffix in (("spherical_ba97", -2., ""),
                ("spherical_experimental_density", 2.275, "_experimental_density")):
            with self.subTest(scheme=scheme):
                geometry = dp.Geometry(scheme, flattening_exponent=exponent)
                source = dp.create_layer_cake_model([self.layer], [20.],
                    self.halfspace, geometry=geometry)
                result = dp.earth_flattening_transformation(source)
                layers, thicknesses = getattr(raw, "earth_flattening_transformation_layers" + suffix)(
                    [self.raw_layer], [20.], 6371., exponent)
                np.testing.assert_allclose(fields(result.materials[0]), fields(layers[0], True), rtol=1e-12)
                np.testing.assert_allclose(result.thicknesses, thicknesses, rtol=1e-12)
                # A GRID node at the halfspace top provides an independent
                # legacy check of the chosen homogeneous-halfspace convention.
                _, materials, _ = getattr(raw, "earth_flattening_transformation_grid" + suffix)(
                    [0., 20.], [self.raw_layer, self.raw_halfspace], [False, False], 6371., exponent)
                np.testing.assert_allclose(fields(result.halfspace), fields(materials[-1], True), rtol=1e-12)
                self.assertTrue(result.flattened)
                self.assertEqual(result.flattening_policy, scheme)
                self.assertEqual(result.geometry.kind, "flat")
                self.assertEqual(source.geometry.kind, scheme)
                self.assertFalse(source.flattened)
                with self.assertRaises(dp.Error):
                    dp.earth_flattening_transformation(result)
                grid = dp.create_grid_model([0., 20., 20., 40.],
                    [self.layer, self.layer, self.halfspace, self.halfspace],
                    interpolation="velocity", geometry=geometry)
                flat_grid = dp.earth_flattening_transformation(grid)
                depths, materials, discontinuities = getattr(raw,
                    "earth_flattening_transformation_grid" + suffix)(
                    grid.depths, [legacy_material(p) for p in grid.materials],
                    grid.discontinuities, 6371., exponent)
                np.testing.assert_allclose(flat_grid.depths, depths, rtol=1e-12)
                np.testing.assert_array_equal(flat_grid.discontinuities, discontinuities)
                np.testing.assert_allclose([fields(p) for p in flat_grid.materials],
                    [fields(p, True) for p in materials], rtol=1e-12)
                np.testing.assert_array_equal(fields(flat_grid.halfspace),
                                               fields(flat_grid.materials[-1]))
                self.assertEqual(flat_grid.interpolation, "velocity")

    def test_q_correction_covers_halfspace_preserves_metadata_and_rejects_repeat(self):
        grid = dp.create_grid_model([0., 20.], [self.layer, self.halfspace],
                                    interpolation="stiffness")
        for source in (self.model, grid):
            with self.subTest(kind=source.kind):
                corrected = dp.apply_mineos_physical_dispersion(source, [600., 800.],
                    [300., 400.], reference_period=1., period=5.)
                reference_materials = [self.raw_layer, self.raw_halfspace]
                expected = raw.apply_mineos_physical_dispersion_model(reference_materials,
                    [600., 800.], [300., 400.], 1., 5.)
                np.testing.assert_allclose(fields(corrected.materials[0]), fields(expected[0], True), rtol=1e-12)
                np.testing.assert_allclose(fields(corrected.halfspace), fields(expected[-1], True), rtol=1e-12)
                self.assertEqual(corrected.q_reference_period, 1.)
                self.assertEqual(corrected.q_period, 5.)
                self.assertIsNone(source.q_period)
                np.testing.assert_array_equal(fields(source.halfspace), fields(self.halfspace))
                with self.assertRaises(dp.Error):
                    dp.apply_mineos_physical_dispersion(corrected, [600., 800.], [300., 400.],
                        reference_period=1., period=10.)
                with self.assertRaises(dp.Error):
                    dp.compute_dispersion(corrected, period=10.)
        with self.assertRaises(dp.Error):
            dp.apply_mineos_physical_dispersion(self.model, [600.], [300.],
                reference_period=1., period=5.)

    def test_q_correction_rejects_unrepresentable_velocity_grid(self):
        # Choose Qk so corrected A equals 2L while F remains nonzero.
        node = dp.Material(1., 2.25, 4., .25, 1., 1.)
        mu = (node.A + node.C - 2*node.F + 5*node.N + 6*node.L) / 15
        lam = (4*(node.A + node.F - node.N) + node.C) / 9 - 2*mu/3
        ratio = 4*mu / (3*(lam + 2*mu))
        factor = raw.mineos_physical_dispersion_factor(1., 2.)
        q = factor*(1 - ratio) / (2*node.L/node.A - 1)
        for kind in ("stiffness", "velocity"):
            model = dp.create_grid_model([0., 20.], [node, node], interpolation=kind)
            if kind == "stiffness":
                result = dp.apply_mineos_physical_dispersion(model, [q, q], [0., 0.],
                    reference_period=1., period=2.).materials[0]
                self.assertEqual(result.A, 2*result.L)
                self.assertNotEqual(result.F, 0.)
            else:
                with self.assertRaises(dp.Error) as raised:
                    dp.apply_mineos_physical_dispersion(model, [q, q], [0., 0.],
                        reference_period=1., period=2.)
                self.assertEqual(raised.exception.code, "invalid_argument")

    def test_capabilities_distinguish_ad_and_variational_model_constraints(self):
        def capabilities(model):
            return {(entry.operation, entry.basis, entry.method): entry
                    for entry in dp.get_capabilities(model)}

        transition = dp.create_grid_model([0., 4., 20.],
            [self.fluid, self.layer, self.halfspace], interpolation="stiffness")
        rows = capabilities(transition)
        # Meshing already rejects this transition, before either derivative
        # path can run. AD cannot make the unsupported model valid.
        self.assertFalse(rows["kernels", "stiffness", "ad"].supported)
        with self.assertRaisesRegex((dp.Error, ValueError), "duplicate-depth"):
            dp.compute_dispersion(transition, period=5.)
        for operation, method in (("kernels", "standard"), ("kernels", "variational"),
                                  ("hessian", "variational_ad")):
            row = rows[operation, "stiffness", method]
            self.assertFalse(row.supported)
            self.assertIn("duplicate-depth", row.reason)
        velocity = dp.create_grid_model([0., 20.], [self.layer, self.halfspace],
            interpolation="velocity")
        rows = capabilities(velocity)
        self.assertTrue(rows["kernels", "stiffness", "ad"].supported)
        self.assertFalse(rows["kernels", "stiffness", "variational"].supported)
        self.assertTrue(rows["frequency_curvature", "", "ad"].supported)
        rows = capabilities(self.model)
        self.assertTrue(rows["frequency_curvature", "", "ad"].supported)
        self.assertTrue(rows["hessian", "stiffness", "ad"].supported)
        large = dp.create_layer_cake_model([self.layer] * 13, [1.] * 13, self.halfspace)
        self.assertFalse(capabilities(large)["hessian", "stiffness", "ad"].supported)
        liquid_halfspace = dp.create_layer_cake_model([self.fluid], [4.], self.fluid)
        rows = capabilities(liquid_halfspace)
        self.assertFalse(rows["dispersion", "", "mode_order"].supported)
        self.assertIn("halfspace", rows["dispersion", "", "mode_order"].reason)

    def test_gravity_matches_adopted_python_implementation(self):
        # Load this optional reference module without importing the inversion
        # package and its unrelated dependencies.
        path = Path(__file__).resolve().parents[2] / "inversion/disper26_inv/gravity.py"
        spec = importlib.util.spec_from_file_location("_public_api_gravity_reference", path)
        gravity = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gravity
        try:
            spec.loader.exec_module(gravity)
            solution = dp.compute_dispersion(self.ocean, period=5.)
            layers = [legacy_material(self.fluid), self.raw_layer]
            for region in ("fluid", "solid"):
                with self.subTest(region=region):
                    expected = gravity.rayleigh_gravity_correction(layers, [4., 20.],
                        self.raw_halfspace, [5.], phase_velocity=[solution.phase_velocity], region=region)
                    actual = dp.compute_gravity_correction(solution, region=region)
                    np.testing.assert_allclose([actual.delta_c, actual.delta_u,
                            actual.omega_delta_c_omega_derivative],
                        [expected.delta_c[0], expected.delta_u[0], expected.omega_dc_domega[0]],
                        rtol=1e-9, atol=1e-12)
                    self.assertEqual(actual.region, region)
                    self.assertEqual(solution.period, 5.)
        finally:
            sys.modules.pop(spec.name, None)
        love = dp.compute_dispersion(self.model, period=5., wave="Love")
        with self.assertRaises(dp.Error):
            dp.compute_gravity_correction(love)


if __name__ == "__main__":
    unittest.main()
