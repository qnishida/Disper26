"""Independent regressions for the public Love analytical energy integrals.

AD and energy balance allow 1e-10 for root and state-propagation roundoff.
Frequency differences use the fluid-group suite's 1e-6 relative bound.
The 200-sublayer trapezoidal reference allows 2e-6 for its O(dz**2) error.
"""
import math
import unittest
import warnings

import numpy as np

import disper26 as d26
import disper26py as dp


def multilayer_model(anisotropic=False):
    depths = np.linspace(0., 100., 4)
    vs = np.linspace(3., 4.5, 4)
    vp = np.linspace(5.2, 8., 4)
    rho = np.linspace(2.5, 3.3, 4)
    if anisotropic:
        return d26.create_model(depths, kind="layer_cake", rho=rho,
                                vsv=vs, vsh=1.07 * vs, vpv=vp,
                                vph=1.03 * vp, eta=np.full(4, 0.95))
    return d26.create_model(depths, kind="layer_cake", vs=vs, vp=vp, rho=rho)


class LoveEnergyIntegrals(unittest.TestCase):
    def test_multilayer_energy_matches_ad_and_energy_balance(self):
        # A single finite layer has zero traction at its top and cannot expose
        # a sign error when decomposing nonzero interface tractions into modes.
        for anisotropic in (False, True):
            model = multilayer_model(anisotropic)
            for mode in (0, 1):
                with self.subTest(anisotropic=anisotropic, mode=mode):
                    solution = d26.compute_dispersion(
                        model, period=20., wave="Love", mode=mode,
                        outputs=("phase_velocity", "group_velocity"))
                    for sublayers in (1, 10):
                        energy = d26.compute_energy_integrals(solution,
                            options=d26.Options(integration="analytical",
                                                num_sub_layers=sublayers))
                        np.testing.assert_allclose(energy.group_velocity,
                            solution.group_velocity, rtol=1e-10, atol=0.)
                        np.testing.assert_allclose(energy.I2,
                            (2 * math.pi / solution.period)**2 * energy.I1,
                            rtol=1e-10, atol=0.)

    def test_energy_matches_frequency_difference_and_trapezoidal_limit(self):
        model = multilayer_model()
        solution = d26.compute_dispersion(model, period=20., wave="Love")
        energy = d26.compute_energy_integrals(solution)
        omega = 2 * math.pi / solution.period
        # Differentiate roots of the dispersion equation, independently of AD
        # and of eigenfunction reconstruction / energy integration.
        for relative_step in (1e-4, 3e-5):
            dw = omega * relative_step
            cm = d26.compute_dispersion(model,
                period=2 * math.pi / (omega - dw), wave="Love").phase_velocity
            cp = d26.compute_dispersion(model,
                period=2 * math.pi / (omega + dw), wave="Love").phase_velocity
            u_fd = 2 * dw / ((omega + dw) / cp - (omega - dw) / cm)
            np.testing.assert_allclose(energy.group_velocity, u_fd,
                                       rtol=1e-6, atol=0.)
        exact = np.array([energy.I1, energy.I2, energy.I3])
        errors = []
        for sublayers in (100, 200):
            numeric = d26.compute_energy_integrals(solution,
                options=d26.Options(integration="trapezoidal",
                                    num_sub_layers=sublayers))
            values = np.array([numeric.I1, numeric.I2, numeric.I3])
            errors.append(np.max(np.abs(values / exact - 1)))
        self.assertLess(errors[1], errors[0])
        np.testing.assert_allclose(values, exact, rtol=2e-6, atol=0.)

    def test_lowlevel_analytical_integrals_match_ad(self):
        # Exercise the low-level entry with eigenfunctions in the public
        # traction convention, without adjusting their sign in the caller.
        vs = np.linspace(3., 4.5, 4)
        vp = np.linspace(5.2, 8., 4)
        rho = np.linspace(2.5, 3.3, 4)
        materials = [dp.create_isotropic_parameters(s, p, r)
                     for s, p, r in zip(vs, vp, rho)]
        thicknesses = np.diff(np.linspace(0., 100., 4)).tolist()
        solution = d26.compute_dispersion(multilayer_model(),
                                          period=20., wave="Love")
        omega = 2 * math.pi / solution.period
        args = (materials[:-1], thicknesses, materials[-1],
                omega, solution.phase_velocity)
        fields = dp.compute_eigenfunctions_love(*args, 10)
        integrals = dp.compute_energy_integrals_analytical_love(fields, *args)
        u_energy = integrals[2] / (solution.phase_velocity * integrals[0])
        u_ad = dp.compute_group_velocity_ad(*args, "Love")
        np.testing.assert_allclose(u_energy, u_ad, rtol=1e-10, atol=0.)

    def test_grid_energy_matches_ad_on_its_accepted_mesh(self):
        depths = np.linspace(0., 100., 4)
        vs = np.linspace(3., 4.5, 4)
        vp = np.linspace(5.2, 8., 4)
        rho = np.linspace(2.5, 3.3, 4)
        model = d26.create_model(depths, kind="grid", interpolation="stiffness",
                                  vs=vs, vp=vp, rho=rho)
        materials = [dp.create_isotropic_parameters(s, p, r)
                     for s, p, r in zip(vs, vp, rho)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", d26.ConvergenceWarning)
            solution = d26.compute_dispersion(model, period=20., wave="Love",
                                               options=d26.Options(max_steps=2))
        energy = d26.compute_energy_integrals(solution)
        mesh = dp.debug_refined_grid_layers(depths, materials, [False] * 4,
                                             energy.diagnostics.accepted_level)
        # Energy integrals use one mesh's root, not the extrapolated curve.
        u_ad = dp.compute_group_velocity_ad(mesh["layers"], mesh["thicknesses"],
            mesh["halfspace"], 2 * math.pi / solution.period,
            energy.phase_velocity, "Love")
        np.testing.assert_allclose(energy.group_velocity, u_ad,
                                   rtol=1e-10, atol=0.)


if __name__ == "__main__":
    unittest.main()
