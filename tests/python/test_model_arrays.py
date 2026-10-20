"""Array constructors preserve legacy models and reject ambiguous inputs."""
import unittest
import warnings
import numpy as np
import disper26 as dp


class ModelArrays(unittest.TestCase):
    def setUp(self):
        self.depths = [0., 10., 30.]
        self.iso = dict(vs=[3., 3.5, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
        self.materials = [dp.create_isotropic_parameters(s, p, r)
                          for s, p, r in zip(self.iso['vs'], self.iso['vp'], self.iso['rho'])]

    def assert_models_equal(self, a, b):
        self.assertEqual((a.kind, a.interpolation, a.name), (b.kind, b.interpolation, b.name))
        np.testing.assert_array_equal(a.depths, b.depths)
        np.testing.assert_array_equal(a.thicknesses, b.thicknesses)
        np.testing.assert_array_equal(a.discontinuities, b.discontinuities)
        for x, y in zip([*a.materials, a.halfspace], [*b.materials, b.halfspace]):
            for field in ('rho', 'A', 'C', 'F', 'L', 'N'):
                self.assertEqual(getattr(x, field), getattr(y, field))

    def test_all_representations_and_numerical_parity(self):
        velocity = dict(rho=self.iso['rho'], vph=[5.3, 6.2, 8.2], vpv=self.iso['vp'],
                        vsh=[3.1, 3.6, 4.6], vsv=self.iso['vs'], eta=[.9, 1.1, .95])
        vti = [dp.create_velocity_vti_parameters(*x) for x in zip(*velocity.values())]
        stiffness = {f: [getattr(m, f) for m in vti] for f in ('rho', 'A', 'C', 'F', 'L', 'N')}
        options = dp.Options(max_steps=2, max_levels=2)
        for fields, materials in ((self.iso, self.materials), (velocity, vti), (stiffness, vti)):
            for kind, interpolation in (('pancake', None), ('grid', 'stiffness'), ('grid', 'velocity')):
                with self.subTest(fields=tuple(fields), kind=kind, interpolation=interpolation):
                    new = dp.create_model(self.depths, kind=kind, interpolation=interpolation, **fields)
                    old = (dp.create_pancake_model(materials[:-1], [10., 20.], materials[-1])
                           if kind == 'pancake' else dp.create_grid_model(self.depths, materials, interpolation=interpolation))
                    self.assert_models_equal(new, old)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', dp.ConvergenceWarning)
                        for wave in ('Rayleigh', 'Love'):
                            a = dp.compute_dispersion(new, period=10., wave=wave, options=options)
                            b = dp.compute_dispersion(old, period=10., wave=wave, options=options)
                            self.assertEqual(a.phase_velocity, b.phase_velocity)
                            ga = dp.compute_kernels(a, basis='stiffness', options=options)
                            gb = dp.compute_kernels(b, basis='stiffness', options=options)
                            np.testing.assert_array_equal(ga.gradient, gb.gradient)
                            if interpolation != 'velocity':
                                ha = dp.compute_hessian(a, basis='stiffness', options=options)
                                hb = dp.compute_hessian(b, basis='stiffness', options=options)
                                np.testing.assert_array_equal(ha.hessian, hb.hessian)

    def test_halfspace_fluid_and_interfaces(self):
        model = dp.create_model([0.], kind='pancake', vs=[4.5], vp=[8.], rho=[3.3])
        self.assertEqual(len(model.materials), 0)
        # Construction and mode-order search both support a halfspace alone.
        self.assert_models_equal(model, dp.create_pancake_model([], [], self.materials[-1]))
        self.assertAlmostEqual(dp.compute_dispersion(model, period=10.).phase_velocity,
                               4.150909460643234, places=8)
        fields = dict(vs=[0., 0., 3.5, 4.5], vp=[1.5, 1.5, 6., 8.], rho=[1., 1., 2.8, 3.3])
        grid = dp.create_model([0., 4., 4., 30.], kind='grid', interpolation='stiffness', **fields)
        np.testing.assert_array_equal(grid.discontinuities, [False, True, False, False])
        pancake = dp.create_model([0., 4., 30.], kind='pancake',
                                  vs=[0., 3.5, 4.5], vp=[1.5, 6., 8.], rho=[1., 2.8, 3.3])
        for m in (grid, pancake):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', dp.ConvergenceWarning)
                self.assertTrue(np.isfinite(dp.compute_dispersion(m, period=10.).phase_velocity))

    def test_owned_strided_arrays_and_geometry(self):
        backing = np.column_stack([self.depths, [-1., -1., -1.]])
        depths = backing[:, 0]
        fields = {name: np.column_stack([values, [0., 0., 0.]])[:, 0]
                  for name, values in self.iso.items()}
        self.assertFalse(depths.flags.c_contiguous)
        self.assertFalse(fields['vs'].flags.c_contiguous)
        geometry = dp.Geometry('spherical_ba97')
        model = dp.create_model(depths, kind='pancake', geometry=geometry, name='arrays', **fields)
        depths[:] = 99.
        fields['vs'][:] = 99.
        np.testing.assert_array_equal(model.thicknesses, [10., 20.])
        self.assertEqual(model.materials[0].L, self.materials[0].L)
        self.assertEqual(model.geometry.kind, geometry.kind)
        self.assertEqual(model.name, 'arrays')

    def test_invalid_inputs(self):
        for depth in ([0., 10., 10.], [0., 30., 10.], [1., 10., 30.], [0., np.nan, 30.], [], [[0., 10., 30.]]):
            with self.subTest(depth=depth), self.assertRaises((dp.Error, ValueError)):
                dp.create_model(depth, kind='pancake', **self.iso)
        variants = [dict(self.iso, vs=[3.]), dict(self.iso, vs=3.), dict(self.iso, vp=[5., np.inf, 8.]),
                    dict(self.iso, A=[1., 2., 3.]), dict(rho=[2.5, 2.8, 3.3]),
                    dict(vs=[3., 3.5, 4.5], rho=[2.5, 2.8, 3.3])]
        for fields in variants:
            with self.subTest(fields=fields), self.assertRaises((dp.Error, ValueError)):
                dp.create_model(self.depths, kind='pancake', **fields)
        for kind, interpolation in [('unknown', None), ('pancake', 'constant'), ('pancake', ''),
                                    ('grid', None), ('grid', 'linear')]:
            with self.subTest(kind=kind, interpolation=interpolation), self.assertRaises(dp.Error):
                dp.create_model(self.depths, kind=kind, interpolation=interpolation, **self.iso)
        with self.assertRaises(TypeError):
            dp.create_model(self.depths, **self.iso)
        with self.assertRaises(dp.Error):
            dp.create_model([0., 0., 0.], kind='grid', interpolation='stiffness', **self.iso)
        with self.assertRaises(dp.Error):
            dp.create_model([0.], kind='grid', interpolation='stiffness', vs=[3.], vp=[5.], rho=[2.5])


if __name__ == '__main__':
    unittest.main()
