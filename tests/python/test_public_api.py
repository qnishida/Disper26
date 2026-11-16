"""Public API ownership, axis contracts, and independent legacy comparisons.

Run in the project environment: python -m unittest discover -s tests/python -p test_public_api.py
The 1e-9 relative / 1e-11 absolute comparisons allow root-solver roundoff when
comparing the integer-count mode selector with the legacy sign-change scan.
"""
import math
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import unittest
import warnings

import numpy as np
import disper26 as dp
import disper26py as raw


class PublicAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layer = dp.create_isotropic_parameters(3.0, 5.2, 2.5)
        cls.halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
        cls.model = dp.create_layer_cake_model([cls.layer], [10.0], cls.halfspace)
        cls.raw_layers = [raw.create_isotropic_parameters(3.0, 5.2, 2.5)]
        cls.raw_halfspace = raw.create_isotropic_parameters(4.5, 8.0, 3.3)
        cls.solution = dp.compute_dispersion(cls.model, period=5.0)

    def test_removed_seabed_option_and_remaining_gravity_fields(self):
        gravity = 0.00980665
        water = raw.create_fluid_parameters(1.5, 1.03, gravity, 0.5, 0.25)
        self.assertEqual(water.gravity, gravity)
        self.assertEqual(water.gravity_internal_scale, 0.5)
        self.assertEqual(water.gravity_surface_scale, 0.25)
        material_args = (water.rho, water.c11, water.c33, water.c13, water.c44, water.c66)
        material = dp.Material(*material_args, gravity=gravity,
            gravity_internal_scale=0.5, gravity_surface_scale=0.25)
        for obj in (water, material):
            self.assertFalse(hasattr(obj, "gravity_seabed_scale"))
            with self.assertRaises(AttributeError):
                setattr(obj, "gravity_seabed_scale", 0.0)
        for value in (0.0, 1.0):
            with self.assertRaises(TypeError):
                raw.create_fluid_parameters(1.5, 1.03, seabed_scale=value)
            with self.assertRaises(TypeError):
                dp.Material(*material_args, gravity_seabed_scale=value)
        with self.assertRaises(TypeError):
            raw.create_fluid_parameters(1.5, 1.03, gravity, 1.0, 0.0, 0.0)
        with self.assertRaises(TypeError):
            dp.Material(*material_args, gravity, 1.0, 0.0, 0.0)
        model = dp.create_layer_cake_model([material], [3.0], self.halfspace)
        self.assertEqual(model.materials[0].gravity, gravity)
        self.assertEqual(model.materials[0].gravity_internal_scale, 0.5)
        self.assertEqual(model.materials[0].gravity_surface_scale, 0.25)

    def test_input_and_result_ownership(self):
        depths = np.array([0.0, 10.0, 30.0])
        vs = np.array([3.0, 3.4, 4.5])
        model = dp.create_grid_model(depths, interpolation="stiffness", vs=vs,
                                     vp=[5.2, 6.0, 8.0], rho=[2.5, 2.8, 3.3])
        depths[1] = 500.0
        vs[0] = 1.0
        np.testing.assert_array_equal(model.depths, [0., 10., 30.])
        self.assertEqual(model.materials[0].L, 2.5 * 3.0**2)
        model.depths[1] = 900.0
        self.assertEqual(model.depths[1], 10.)
        with self.assertRaises(AttributeError):
            model.materials[0].L = 1.0
        with self.assertRaises(AttributeError):
            self.solution.phase_velocity = 0.0
        copy = self.solution.options
        copy.max_steps = 1
        self.assertEqual(self.solution.options.max_steps, 5)
        self.assertEqual(self.solution.model.halfspace.L, self.halfspace.L)

    def test_roots_match_legacy_scan(self):
        for wave in ("Rayleigh", "Love"):
            solution = dp.compute_dispersion(self.model, period=5.0, wave=wave)
            legacy = raw.search_roots(self.raw_layers, [10.0], self.raw_halfspace,
                                      2 * math.pi / 5.0, 2.0, 4.49, 1000, wave)
            self.assertTrue(legacy)
            np.testing.assert_allclose(solution.phase_velocity, legacy[0], rtol=1e-9, atol=1e-11)
            self.assertEqual((solution.period, solution.mode, solution.wave), (5.0, 0, wave))

    def test_requested_and_inapplicable_outputs(self):
        self.assertIsNone(self.solution.group_velocity)
        self.assertIsNone(self.solution.hv_ratio)
        self.assertEqual(self.solution.diagnostics.availability["group_velocity"], "not_requested")
        rayleigh = dp.compute_dispersion(self.model, period=5.0,
                                         outputs=("phase_velocity", "group_velocity", "hv_ratio"))
        self.assertTrue(np.isfinite(rayleigh.group_velocity))
        self.assertTrue(np.isfinite(rayleigh.hv_ratio))
        love = dp.compute_dispersion(self.model, period=5.0, wave="Love", outputs=("hv_ratio",))
        self.assertIsNone(love.hv_ratio)
        self.assertEqual(love.diagnostics.availability["hv_ratio"], "not_applicable")

    def test_convergence_is_reported_per_requested_quantity(self):
        # A root error estimate establishes phase convergence only. Neither
        # optional value may borrow that accuracy claim, even on a Layer-cake.
        phase = dp.compute_dispersion(self.model, period=5., options=dp.Options(strict=True))
        self.assertEqual(phase.diagnostics.convergence_by_quantity["phase_velocity"], "converged")
        models = [self.model, dp.create_grid_model(
            [0., 10., 30.], interpolation="stiffness", vs=[3., 3.4, 4.5],
            vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])]
        for model in models:
            with self.subTest(kind=model.kind), warnings.catch_warnings():
                warnings.simplefilter("ignore", dp.ConvergenceWarning)
                # A loose phase threshold makes this check independent of the
                # model's convergence rate; U/HV still have no own estimate.
                options = dp.Options(max_steps=3, tolerance=1e-2)
                result = dp.compute_dispersion(model, period=5.,
                    outputs=("phase_velocity", "group_velocity", "hv_ratio"), options=options)
                convergence = result.diagnostics.convergence_by_quantity
                self.assertEqual(convergence["phase_velocity"], "converged")
                self.assertEqual(convergence["group_velocity"], "not_evaluated")
                self.assertEqual(convergence["hv_ratio"], "not_evaluated")
                self.assertEqual(result.diagnostics.convergence, "not_evaluated")
                options.strict = True
                with self.assertRaises(dp.Error) as raised:
                    dp.compute_dispersion(model, period=5., outputs=("group_velocity",), options=options)
                self.assertEqual(raised.exception.code, "not_converged")

    def test_curve_keeps_order_and_repeated_periods(self):
        periods = [8., 5., 8., 3.]
        curve = dp.compute_dispersion_curve(self.model, periods=periods)
        np.testing.assert_array_equal(curve.periods, periods)
        self.assertIsNone(curve.group_velocity)
        self.assertEqual(curve.phase_velocity[0], curve.phase_velocity[2])
        singles = [dp.compute_dispersion(self.model, period=p).phase_velocity for p in periods]
        np.testing.assert_allclose(curve.phase_velocity, singles, rtol=1e-9, atol=1e-11)

    def test_gradient_matches_legacy_with_named_axes(self):
        kernels = dp.compute_kernels(self.solution, basis="stiffness")
        legacy = raw.compute_kernels_variational_layer_cake(
            self.raw_layers, [10.0], self.raw_halfspace, 2 * math.pi / 5.0,
            self.solution.phase_velocity, sensitivity_mode="VTI", use_analytical=True,
            want_kernel_u=False, want_kernel_hv=False)
        # Legacy uses -1 for halfspace; the public API names the entity.
        lookup = {(k.layer_index, k.parameter_name): k.d_c for k in legacy}
        reference = [lookup[(-1 if p.entity == "halfspace" else p.index, p.name)]
                     for p in kernels.parameters]
        np.testing.assert_allclose(kernels.gradient, reference, rtol=1e-9, atol=1e-11)
        self.assertEqual(kernels.gradient.shape, (len(kernels.parameters),))
        self.assertIsNone(kernels.hessian)
        self.assertNotIn("group_velocity", kernels.gradients)
        self.assertEqual(kernels.parameters[-1].unit, "g/cm^3")
        changed = kernels.gradient
        changed[:] = 0
        np.testing.assert_allclose(kernels.gradient, reference, rtol=1e-9, atol=1e-11)

    def test_hessian_layout_matches_legacy(self):
        result = dp.compute_hessian(self.solution, basis="stiffness")
        legacy = raw.compute_hessian_variational_ad_rayleigh_layer_cake(
            self.raw_layers, [10.0], self.raw_halfspace, 2 * math.pi / 5.0,
            self.solution.phase_velocity, sensitivity_mode="VTI", use_analytical=True)
        indices = {(int(i), n): k for k, (i, n) in
                   enumerate(zip(legacy["layer_indices"], legacy["parameter_names"]))}
        order = [indices[(-1 if p.entity == "halfspace" else p.index, p.name)]
                 for p in result.parameters]
        reference = np.asarray(legacy["hessian"])[np.ix_(order, order)]
        self.assertEqual(result.hessian.shape, (len(order), len(order)))
        np.testing.assert_allclose(result.hessian, reference, rtol=1e-9, atol=1e-11)

    def test_reject_ambiguous_and_invalid_requests(self):
        with self.assertRaises(TypeError):
            dp.create_grid_model([0., 10.], [self.layer, self.halfspace])
        with self.assertRaises(ValueError):
            dp.create_grid_model([0., 10.], [self.layer, self.halfspace], interpolation="stiffness",
                                 vs=[3., 4.5])
        with self.assertRaises(ValueError):
            dp.create_grid_model([0., 10.], interpolation="stiffness", vs=[3.], vp=[5.2], rho=[2.5])
        with self.assertRaises(dp.Error):
            dp.create_grid_model([0., 0., 0.], [self.layer] * 3, interpolation="stiffness")
        for kwargs in ({"period": -1.}, {"period": 5., "mode": -1},
                       {"period": 5., "outputs": ("unknown",)}):
            with self.assertRaises(dp.Error) as raised:
                dp.compute_dispersion(self.model, **kwargs)
            self.assertEqual(raised.exception.code, "invalid_argument")
        with self.assertRaises(TypeError):
            dp.compute_kernels(self.solution)
        with self.assertRaises(TypeError):
            dp.Options(max_level=2)
        with self.assertRaises(ValueError):
            dp.compute_hessian_reduced(self.solution, np.ones(12))

    def test_eigenfunctions_and_integrals_named_results(self):
        eigenfunctions = dp.compute_eigenfunctions(self.solution, num_sub_layers=8)
        self.assertTrue(len(eigenfunctions.depth) > 0)
        for values in eigenfunctions.components.values():
            self.assertEqual(values.shape, eigenfunctions.depth.shape)
            self.assertTrue(np.iscomplexobj(values))
        energy = dp.compute_energy_integrals(self.solution)
        self.assertTrue(np.isfinite([energy.I1, energy.I2, energy.I3]).all())
        self.assertAlmostEqual(energy.phase_velocity, self.solution.phase_velocity, places=8)

    def test_grid_warning_and_strict_policy(self):
        model = dp.create_grid_model([0., 10., 30.], interpolation="stiffness",
                                     vs=[3., 3.4, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
        options = dp.Options(max_steps=1, tolerance=1e-30)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", dp.ConvergenceWarning)
            result = dp.compute_dispersion(model, period=5., options=options)
        self.assertEqual(result.diagnostics.status, "warning")
        self.assertTrue(any(issubclass(w.category, dp.ConvergenceWarning) for w in caught))
        options.strict = True
        with self.assertRaises(dp.Error):
            dp.compute_dispersion(model, period=5., options=options)
        velocity_model = dp.create_grid_model([0., 10., 30.], interpolation="velocity",
                                              vs=[3., 3.4, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", dp.ConvergenceWarning)
            solution = dp.compute_dispersion(velocity_model, period=5.)
        with self.assertRaises(dp.Error):
            dp.compute_hessian(solution, basis="stiffness")
        self.assertEqual(solution.model.interpolation, "velocity")

    def test_grid_gradient_axes_in_both_interpolations(self):
        depths = [0., 10., 30.]
        vs, vp, rho = np.array([3., 3.4, 4.5]), np.array([5.2, 6., 8.]), np.array([2.5, 2.8, 3.3])
        options = dp.Options(max_steps=2, max_levels=2)
        raw_materials = [raw.create_isotropic_parameters(s, p, r) for s, p, r in zip(vs, vp, rho)]
        for interpolation in ("stiffness", "velocity"):
            with self.subTest(interpolation=interpolation), warnings.catch_warnings():
                warnings.simplefilter("ignore", dp.ConvergenceWarning)
                model = dp.create_grid_model(depths, interpolation=interpolation, vs=vs, vp=vp, rho=rho)
                solution = dp.compute_dispersion(model, period=5., options=options)
                stiffness = dp.compute_kernels(solution, basis="stiffness", options=options)
                velocity = dp.compute_kernels(solution, basis="velocity_vti", options=options)
                self.assertEqual([p.name for p in stiffness.parameters[:6]], ["A", "C", "F", "L", "N", "Rho"])
                self.assertEqual([p.name for p in velocity.parameters[:6]], ["rho", "vph", "vpv", "vsh", "vsv", "eta"])
                self.assertEqual([p.index for p in stiffness.parameters], [i for i in range(3) for _ in range(6)])
                if interpolation == "stiffness":
                    legacy = raw.compute_kernels_variational_grid(
                        depths, raw_materials, [False] * 3, freq=0.2,
                        c_guess=solution.phase_velocity, max_levels=2)
                    reference = np.asarray(legacy["gradient_node"]).reshape(-1, 6)[:, [1, 2, 3, 4, 5, 0]].ravel()
                    np.testing.assert_allclose(stiffness.gradient, reference, rtol=1e-8, atol=1e-10)
                else:
                    legacy = raw.compute_kernels_ad_grid_velocity_vti(
                        depths, rho, vp, vp, vs, vs, np.ones(3), [False] * 3, period=5.,
                        c_guess=solution.phase_velocity, max_steps=2, want_kernel_u=False,
                        want_kernel_hv=False, phase_only=True, want_value_u=False, want_value_hv=False)[5]
                    lookup = {(k.layer_index, k.parameter_name.lower()): k.d_c for k in legacy}
                    reference = [lookup[(p.index, p.name)] for p in velocity.parameters]
                    np.testing.assert_allclose(velocity.gradient, reference, rtol=1e-8, atol=1e-10)
                # Independent coordinate differential in canonical axis order.
                gs = stiffness.gradient.reshape(-1, 6)
                converted = np.column_stack((
                    vp**2 * (gs[:, 0] + gs[:, 1]) + (vp**2 - 2 * vs**2) * gs[:, 2]
                    + vs**2 * (gs[:, 3] + gs[:, 4]) + gs[:, 5],
                    2 * rho * vp * (gs[:, 0] + gs[:, 2]),
                    2 * rho * vp * gs[:, 1], 2 * rho * vs * gs[:, 4],
                    2 * rho * vs * (gs[:, 3] - 2 * gs[:, 2]),
                    rho * (vp**2 - 2 * vs**2) * gs[:, 2]))
                np.testing.assert_allclose(velocity.gradient, converted.ravel(), rtol=1e-10, atol=1e-11)
                self.assertEqual(solution.model.interpolation, interpolation)

    def test_isotropic_and_log_hessian_curvature_by_gradient_differences(self):
        # Central differences of analytical gradients: h=1e-4 keeps truncation
        # O(h²); 3e-5 relative / 1e-7 absolute permits root-roundoff amplification.
        def model_from_coordinates(coordinates, basis):
            if basis == "isotropic":
                materials = [dp.create_isotropic_parameters(*row) for row in coordinates.reshape(-1, 3)]
            else:
                materials = [dp.create_velocity_vti_parameters(*row) for row in np.exp(coordinates).reshape(-1, 6)]
            return dp.create_layer_cake_model(materials[:-1], [10.], materials[-1])

        coordinates_by_basis = {
            "isotropic": (np.array([3., 5.2, 2.5, 4.5, 8., 3.3]), (0, 2)),
            "log_velocity_vti": (np.log([2.5, 5.2, 5.2, 3., 3., 1., 3.3, 8., 8., 4.5, 4.5, 1.]), (0, 4, 5)),
        }
        for basis, (coordinates, columns) in coordinates_by_basis.items():
            with self.subTest(basis=basis):
                model = model_from_coordinates(coordinates, basis)
                solution = dp.compute_dispersion(model, period=5.)
                hessian = dp.compute_hessian(solution, basis=basis).hessian
                for column in columns:
                    step = 1e-4
                    plus, minus = coordinates.copy(), coordinates.copy()
                    plus[column] += step
                    minus[column] -= step
                    sp = dp.compute_dispersion(model_from_coordinates(plus, basis), period=5.)
                    sm = dp.compute_dispersion(model_from_coordinates(minus, basis), period=5.)
                    finite_difference = (dp.compute_kernels(sp, basis=basis).gradient
                                         - dp.compute_kernels(sm, basis=basis).gradient) / (2 * step)
                    np.testing.assert_allclose(hessian[:, column], finite_difference, rtol=3e-5, atol=1e-7)

    def test_reduced_hessian_accepts_noncontiguous_nonsquare_projection(self):
        model = dp.create_grid_model([0., 10., 30.], interpolation="stiffness",
                                     vs=[3., 3.4, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
        options = dp.Options(max_steps=2, max_levels=2, derivative_tolerance=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", dp.ConvergenceWarning)
            solution = dp.compute_dispersion(model, period=5., options=options)
            full = dp.compute_hessian(solution, basis="stiffness", options=options)
            storage = np.zeros((18, 4))
            storage[3, 0] = 1.0
            storage[9, 0] = 0.4
            storage[5, 2] = 0.2
            storage[15, 2] = -0.5
            projection = storage[:, ::2]
            self.assertFalse(projection.flags.c_contiguous or projection.flags.f_contiguous)
            reduced = dp.compute_hessian_reduced(solution, projection, options=options)
        self.assertEqual(reduced.hessian.shape, (2, 2))
        self.assertEqual(len(reduced.parameters), 2)
        # Both use the same finite levels; this tests transport and canonical
        # stiffness-axis ordering independently of convergence to a continuum.
        np.testing.assert_allclose(reduced.hessian, projection.T @ full.hessian @ projection,
                                   rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(reduced.gradient, projection.T @ full.gradient,
                                   rtol=1e-8, atol=1e-10)

    def test_grid_fixed_fluid_parameters_and_capability_restrictions(self):
        water = dp.create_fluid_parameters(vp=1.5, rho=1.03)
        model = dp.create_grid_model([0., 1., 1., 10.],
                                     [water, water, self.layer, self.halfspace],
                                     interpolation="stiffness")
        options = dp.Options(max_steps=2, max_levels=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", dp.ConvergenceWarning)
            solution = dp.compute_dispersion(model, period=5., options=options)
            for basis, block_size in (("stiffness", 6), ("isotropic", 3), ("log_velocity_vti", 6)):
                with self.subTest(basis=basis):
                    result = dp.compute_hessian(solution, basis=basis, options=options)
                    fixed = 2 * block_size
                    self.assertEqual([p.active for p in result.parameters],
                                     [False] * fixed + [True] * fixed)
                    np.testing.assert_array_equal(result.gradient[:fixed], 0.)
                    np.testing.assert_array_equal(result.hessian[:fixed, :], 0.)
                    np.testing.assert_array_equal(result.hessian[:, :fixed], 0.)
                    self.assertTrue(np.isfinite(result.hessian).all())
            projection = np.zeros((24, 2))
            projection[0, 0] = 1.0  # Entirely inside a fixed fluid node.
            projection[15, 1] = 1.0
            reduced = dp.compute_hessian_reduced(solution, projection, options=options)
            self.assertEqual([p.active for p in reduced.parameters], [False, True])
            np.testing.assert_array_equal(reduced.hessian[0, :], 0.)
        love = {entry.operation: entry for entry in dp.get_capabilities(model, wave="Love")}
        self.assertFalse(love["dispersion"].supported)
        self.assertTrue(love["dispersion"].reason)
        with self.assertRaises(dp.Error) as raised:
            dp.compute_dispersion(model, period=5., wave="Love")
        self.assertEqual(raised.exception.code, "unsupported")
        velocity_model = dp.create_grid_model([0., 10.], [self.layer, self.halfspace], interpolation="velocity")
        hessians = [entry for entry in dp.get_capabilities(velocity_model) if entry.operation == "hessian"]
        self.assertTrue(hessians)
        self.assertTrue(all(not entry.supported and entry.reason for entry in hessians))

    def test_explicit_ad_group_kernels_match_legacy_both_waves(self):
        depths = [0., 10., 30.]
        vs, vp, rho = np.array([3., 3.4, 4.5]), np.array([5.2, 6., 8.]), np.array([2.5, 2.8, 3.3])
        root_options = dp.Options(max_steps=2)
        ad_options = dp.Options(method="ad", max_steps=2, max_levels=2)
        raw_materials = [raw.create_isotropic_parameters(s, p, r) for s, p, r in zip(vs, vp, rho)]
        for wave in ("Rayleigh", "Love"):
            for interpolation in ("stiffness", "velocity"):
                with self.subTest(wave=wave, interpolation=interpolation), warnings.catch_warnings():
                    warnings.simplefilter("ignore", dp.ConvergenceWarning)
                    model = dp.create_grid_model(depths, interpolation=interpolation, vs=vs, vp=vp, rho=rho)
                    solution = dp.compute_dispersion(model, period=5., wave=wave, options=root_options)
                    basis = "stiffness" if interpolation == "stiffness" else "velocity_vti"
                    result = dp.compute_kernels(solution, basis=basis, outputs=("group_velocity",), options=ad_options)
                    kwargs = dict(period=5., wave_type=wave, c_guess=solution.phase_velocity,
                                  max_steps=2, compute_c=False, want_kernel_u=True,
                                  want_kernel_hv=False, phase_only=False,
                                  want_value_u=True, want_value_hv=True, u_kernel_mode="full_ad")
                    if interpolation == "stiffness":
                        legacy = raw.compute_kernels_ad_grid(
                            depths, raw_materials, [False] * 3, sensitivity_mode="VTI", **kwargs)[5]
                    else:
                        legacy = raw.compute_kernels_ad_grid_velocity_vti(
                            depths, rho, vp, vp, vs, vs, np.ones(3), [False] * 3, **kwargs)[5]
                    lookup = {(k.layer_index, k.parameter_name.lower()): k.d_U for k in legacy}
                    reference = [lookup[(p.index, p.name.lower())] for p in result.parameters]
                    np.testing.assert_allclose(result.gradients["group_velocity"], reference,
                                               rtol=1e-8, atol=1e-10)
                    self.assertEqual(result.diagnostics.method, "ad")
                    self.assertIsNone(result.gradient)
                    self.assertIsNone(solution.group_velocity)

    def test_shared_solution_concurrent_derivatives_preserve_snapshot(self):
        model = dp.create_grid_model([0., 10., 30.], interpolation="stiffness",
                                     vs=[3., 3.4, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
        # The solution initially has only one mesh root. Distinct requests must
        # share trace extensions while leaving the published solution intact.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", dp.ConvergenceWarning)
            solution = dp.compute_dispersion(model, period=5., options=dp.Options(max_steps=0))
        before = (solution.phase_velocity, solution.diagnostics.accepted_level,
                  solution.diagnostics.generated_level, solution.options.max_steps)
        def compute(level):
            return dp.compute_hessian(solution, basis="stiffness", options=dp.Options(max_levels=level))
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(compute, [2, 3, 2, 3]))
        for level, result in zip([2, 3, 2, 3], concurrent):
            reference = compute(level)
            np.testing.assert_allclose(result.gradient, reference.gradient, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(result.hessian, reference.hessian, rtol=1e-10, atol=1e-12)
        self.assertEqual(before, (solution.phase_velocity, solution.diagnostics.accepted_level,
                                 solution.diagnostics.generated_level, solution.options.max_steps))

    def test_packaged_prem_and_transformation_metadata(self):
        spherical = dp.load_prem_spherical(max_depth_km=40., max_spacing_km=None)
        flat = dp.load_prem_flat(max_depth_km=40., max_spacing_km=None)
        self.assertEqual(spherical.depths[0], 0.)
        self.assertEqual(spherical.geometry.kind, "spherical_experimental_density")
        self.assertTrue(flat.flattened)
        self.assertEqual(flat.flattening_policy, spherical.geometry.kind)
        np.testing.assert_allclose(flat.depths, 6371.0 * np.log(6371.0 / (6371.0 - spherical.depths)),
                                   rtol=1e-12, atol=1e-12)
        with self.assertRaises(dp.Error):
            dp.earth_flattening_transformation(flat)
        self.assertTrue(np.any(spherical.discontinuities))

    def test_no_inversion_or_scipy_dependency_on_import(self):
        subprocess.run([sys.executable, "-c", "import sys, disper26; "
                        "assert 'scipy' not in sys.modules; "
                        "assert 'disper26_inv' not in sys.modules; "
                        "assert len(disper26.load_prem_flat(max_depth_km=40).materials) > 0"], check=True)


if __name__ == "__main__":
    unittest.main()
