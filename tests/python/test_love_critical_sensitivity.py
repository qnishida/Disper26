"""Love-wave derivatives at a finite-layer turning point and in a weakly decaying half-space."""
import warnings

import mpmath as mp
import numpy as np
import pytest

import disper26 as dp
import disper26py as raw


def critical_model():
    return [dp.create_isotropic_parameters(vs, vp, rho)
            for vs, vp, rho in zip(np.linspace(3, 4.5, 4), np.linspace(5.2, 8, 4),
                                   np.linspace(2.5, 3.3, 4))]


def reference(materials, depths, period):
    """Differentiate the free-surface traction with 60 digits and no solver calls."""
    with mp.workdps(60):
        w = 2 * mp.pi / period
        h = list(map(mp.mpf, depths))
        params = [mp.mpf(x) for p in materials for x in (p.L, p.N, p.rho)]

        def traction(c, omega, *values):
            k2 = (omega / c) ** 2
            L, N, rho = values[-3:]
            u, t = 1, mp.sqrt(L * (N * k2 - rho * omega**2))
            for i in reversed(range(len(h))):
                L, N, rho = values[3*i:3*i+3]
                n = (N*k2 - rho*omega**2) / L
                q = n*h[i]**2 / 4
                C = mp.hyp0f1(mp.mpf("0.5"), q)
                S = h[i] * mp.hyp0f1(mp.mpf("1.5"), q)
                u, t = C*u + S*t/L, L*n*S*u + C*t
            return t

        c = mp.findroot(lambda speed: traction(speed, w, *params), 3.5)
        point = [c, w] + params
        # Partial derivatives of F(c, omega, material parameters) at the root.
        size = len(point)
        first, second = mp.matrix(1, size), mp.matrix(size)
        for i in range(size):
            order = [0]*size
            order[i] = 1
            first[i] = mp.diff(traction, point, order)
            for j in range(i+1):
                order[j] += 1
                second[i,j] = second[j,i] = mp.diff(traction, point, order)
                order[j] -= 1
        gradient = -first[:, 1:].T / first[0]
        cross = second[1:, :1]
        H = -(second[1:, 1:] + cross*gradient.T + gradient*cross.T
              + second[0, 0]*gradient*gradient.T) / first[0]
        # Public stiffness coordinates are (A, C, F, L, N, Rho).
        indices = [6*i+j for i in range(len(materials)) for j in (3, 4, 5)]
        n = 6*len(materials)
        g, hessian = np.zeros(n), np.zeros((n, n))
        g[indices] = list(gradient[1:, :])
        denominator = c - w*gradient[0]
        U = c*c / denominator
        dU = np.zeros(n)
        dU[indices] = list((2*c*gradient[1:, :] - U*(gradient[1:, :] - w*H[1:, :1])) / denominator)
        hessian[np.ix_(indices, indices)] = np.asarray(H[1:, 1:].tolist(), dtype=float)
        return float(c), g, hessian, float(U), dU


@pytest.mark.parametrize("period", [46.033865, 46.0338660059098, 46.033867])
def test_critical_layer_hessian(period):
    materials = critical_model()
    h = [100./3]*3
    model = dp.create_layer_cake_model(materials[:-1], h, materials[-1])
    c, gradient, hessian, U, dU = reference(materials, h, period)
    sol = dp.compute_dispersion(model, period=period, wave="Love")
    assert sol.phase_velocity == pytest.approx(c, abs=1e-10, rel=0)
    kernels = dp.compute_kernels(sol, basis="stiffness", outputs=["phase_velocity", "group_velocity"])
    np.testing.assert_allclose(kernels.gradients["group_velocity"], dU, atol=1e-10, rtol=1e-8)
    energy = dp.compute_energy_integrals(sol)
    assert energy.group_velocity == pytest.approx(U, rel=1e-9)
    for backend in ("dual", "vector", "vector_dyn"):
        result = dp.compute_hessian(sol, basis="stiffness", options=dp.Options(backend=backend))
        np.testing.assert_allclose(result.gradient, gradient, atol=1e-11, rtol=1e-9)
        np.testing.assert_allclose(result.hessian, hessian, atol=1e-11, rtol=1e-9)

    if period == 46.0338660059098:
        # Exercise the per-parameter energy derivative as well as the public
        # frequency-derivative path for the group-velocity kernels.
        layers = [raw.create_isotropic_parameters(vs, vp, rho)
                  for vs, vp, rho in zip(np.linspace(3, 4.5, 4), np.linspace(5.2, 8, 4),
                                         np.linspace(2.5, 3.3, 4))]
        energy_kernels = raw.compute_kernels_variational_layer_cake(
            layers[:-1], h, layers[-1], 2*np.pi/period, sol.phase_velocity, "Love", "VTI",
            u_kernel_mode="variational", want_kernel_u=True, want_kernel_hv=False)
        by_axis = {(v.layer_index, v.parameter_name): v.d_U for v in energy_kernels}
        values = [by_axis[(-1 if param.entity == "halfspace" else param.index, param.name)]
                  for param in kernels.parameters]
        np.testing.assert_allclose(values, dU, atol=1e-10, rtol=1e-8)

        # The sampled variational path must converge to the same Hessian at the turning point.
        errors = []
        for subdivisions in (16, 128):
            opts = dp.Options(integration="trapezoidal", num_sub_layers=subdivisions)
            sampled = dp.compute_hessian(sol, basis="stiffness", options=opts)
            errors.append(np.linalg.norm(sampled.hessian - hessian))
        assert errors[1] < errors[0]
        np.testing.assert_allclose(sampled.hessian, hessian, atol=1e-9, rtol=1e-8)


def test_critical_grid_midpoint():
    p = critical_model()
    # Opposite N offsets leave the midpoint on the same critical layer as the
    # layered reference, while giving each end of the GRID segment L != N.
    def node(delta):
        m = p[1]
        return dp.Material(m.rho, m.A, m.C, m.F, m.L, m.N + delta)
    h = 100./3
    materials = [p[0], p[0], node(-1.), node(1.), p[2], p[2], p[3]]
    grid = dp.create_grid_model([0, h, h, 2*h, 2*h, 3*h, 3*h], materials,
                                interpolation="stiffness")
    period = 46.0338660059098
    c, g, H, _, _ = reference(p, [h]*3, period)
    options = dp.Options(max_steps=0, max_levels=0)
    # A single prescribed mesh has no inter-level convergence estimate.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Refinement limit reached", category=dp.ConvergenceWarning)
        sol = dp.compute_dispersion(grid, period=period, wave="Love", options=options)
    assert sol.phase_velocity == pytest.approx(c, abs=1e-10, rel=0)
    # At refinement level zero each finite layer is the average of its nodes.
    weights = np.zeros((4, 7))
    for i in range(3):
        weights[i, 2*i:2*i+2] = 0.5
    weights[3, 6] = 1
    mapping = np.kron(weights, np.eye(6))
    J = np.eye(42)[:, [10, 16, 21, 40]]
    for backend in ("dual", "vector", "vector_dyn"):
        options.backend = backend
        full = dp.compute_hessian(sol, basis="stiffness", options=options)
        reduced = dp.compute_hessian_reduced(sol, J, options=options)
        np.testing.assert_allclose(full.gradient, mapping.T @ g, atol=1e-11, rtol=1e-9)
        np.testing.assert_allclose(full.hessian, mapping.T @ H @ mapping, atol=1e-11, rtol=1e-9)
        np.testing.assert_allclose(reduced.gradient, (mapping @ J).T @ g, atol=1e-11, rtol=1e-9)
        np.testing.assert_allclose(reduced.hessian, (mapping @ J).T @ H @ (mapping @ J),
                                   atol=1e-11, rtol=1e-9)


@pytest.mark.parametrize("h,period,expected", [(10., 140., .9956594687),
                                               (500., 7000., .9956594687),
                                               (500., 6000., .9940798726)])
def test_halfspace_velocity_sensitivity(h, period, expected):
    layer = dp.create_isotropic_parameters(3., 5.2, 2.7)
    hs = dp.create_isotropic_parameters(4.5, 8., 3.3)
    sol = dp.compute_dispersion(dp.create_layer_cake_model([layer], [h], hs),
                                wave="Love", period=period)
    result = dp.compute_kernels(sol, basis="stiffness")
    index = next(i for i, p in enumerate(result.parameters)
                 if p.entity == "halfspace" and p.name == "N")
    # Implicit derivative of L1*r*tan(r*h) - Lh*nu_h at fixed Lh, rho_h.
    assert result.gradient[index]*2*hs.rho*4.5 == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("c", [1., np.nextafter(1., 2.)])
def test_energy_rejects_nondecaying_halfspace(c):
    layer = raw.create_isotropic_parameters(.5, 1., 2.)
    hs = raw.create_isotropic_parameters(1., 2., 2.5)
    points = {"depth": np.array([0., .3]), "uy": np.ones(2, dtype=complex),
              "tyz": np.zeros(2, dtype=complex)}
    args = (points, [layer], [.3], hs, .7, c)
    with pytest.raises(ValueError, match="decaying half-space"):
        raw.compute_energy_integrals_analytical_love(*args)
    with pytest.raises(ValueError, match="decaying half-space"):
        raw.compute_energy_integrals(*args, "Love")
