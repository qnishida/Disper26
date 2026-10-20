#!/usr/bin/env python3
"""Regression for a derivative-only Love fundamental branch jump."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))

import disper26py as dp  # noqa: E402
from disper26_inv.derivatives import (  # noqa: E402
    basis_node_jacobian,
    gradient_log,
    kernels_to_stiffness_gradient,
    project_gradient,
)
from disper26_inv.engine import Datum, ForwardEngine  # noqa: E402
from disper26_inv.model import VELOCITY_FIELDS  # noqa: E402
from disper26_inv.priors import PriorCovariance  # noqa: E402
from disper26_inv.shallow import shallow_basis, shallow_reference  # noqa: E402


def _maximum_shear_anomaly(basis, delta_m):
    operator = np.asarray(basis.full_log_operator(), dtype=float)
    n_nodes = basis.reference.n_nodes
    rows = []
    for field in ("vsv", "vsh"):
        index = VELOCITY_FIELDS.index(field)
        rows.append(operator[index * n_nodes : (index + 1) * n_nodes])
    anomaly = np.expm1(np.vstack(rows) @ np.asarray(delta_m, dtype=float))
    return float(np.max(np.abs(anomaly)))


def _scale_to_anomaly(basis, direction_m, target=0.08):
    low, high = 0.0, 1.0
    while _maximum_shear_anomaly(basis, high * direction_m) < target:
        high *= 2.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if _maximum_shear_anomaly(basis, middle * direction_m) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high) * direction_m


def _radial_prior_model(seed, direction, sign, target, name):
    reference = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
    basis = shallow_basis(reference)
    prior = PriorCovariance(
        basis,
        sigma={"vsv": 0.035, "vsh": 0.035},
        param_corr="birch_tight_aniso",
    )
    rng = np.random.default_rng(seed)
    directions_z = rng.standard_normal((32, basis.n_params))
    directions_z /= np.linalg.norm(directions_z, axis=1)[:, None]
    direction_m = float(sign) * (prior.chol() @ directions_z[int(direction)])
    delta_m = _scale_to_anomaly(basis, direction_m, target=float(target))
    return basis, basis.m_to_forward_model(
        delta_m, name=name
    ), delta_m


def love_fundamental_holdout_model():
    """Reconstruct direction 26, positive 8% ray from seed 20261109."""

    return _radial_prior_model(
        20261109,
        26,
        1.0,
        0.08,
        "love0_derivative_root_trace_regression",
    )


def love_overtone_residual_model():
    """Reconstruct direction 9, positive 6% ray from seed 20270108."""

    return _radial_prior_model(
        20270108,
        9,
        1.0,
        0.06,
        "love1_mode_order_residual_polish_regression",
    )


def _standalone_to_stiffness(gradient_node, n_nodes):
    # C++ node-VTI order is [Rho,A,C,F,L,N]; inversion order is
    # [A,C,F,L,N,Rho].
    raw = np.asarray(gradient_node, dtype=float).reshape(n_nodes, 6)
    return raw[:, [1, 2, 3, 4, 5, 0]]


def test_trace_driven_variational_gradient_stays_on_love_fundamental():
    basis, model, delta_m = love_fundamental_holdout_model()
    datum = Datum("Love", 0, 1.0 / 0.055)
    engine = ForwardEngine(max_steps=6, tolerance=1.0e-10)
    solution = engine.phase_velocity(model, datum, minimum_trace_level=5)
    assert abs(solution.c - 3.841299598179) < 5.0e-7
    assert solution.romberg_roots is not None
    assert len(solution.romberg_roots) >= 6
    assert max(solution.romberg_roots) < 4.0

    depths, parameters, is_discontinuity = model.to_params()
    variational = dp.compute_kernels_variational_grid(
        depths,
        parameters,
        is_discontinuity,
        0.055,
        solution.c,
        "Love",
        5,
        1.0e-8,
        phase_velocity_roots=solution.romberg_roots,
        phase_accepted_level=solution.level,
        **model.refinement_kwargs(),
    )
    g_variational_a = _standalone_to_stiffness(
        variational["gradient_node"], model.n_nodes
    )

    ad_solution, ad_kernels = engine.phase_velocity_with_kernels(
        model, datum, c_hint=solution.c
    )
    assert abs(ad_solution.c - solution.c) < 5.0e-10
    g_ad_a = kernels_to_stiffness_gradient(ad_kernels, model.n_nodes)

    relative_l2 = np.linalg.norm(g_variational_a - g_ad_a) / np.linalg.norm(g_ad_a)
    cosine = abs(
        np.vdot(g_variational_a.ravel(), g_ad_a.ravel())
        / (np.linalg.norm(g_variational_a) * np.linalg.norm(g_ad_a))
    )
    assert relative_l2 < 5.0e-7
    assert cosine > 0.999999
    norm_ratio = np.linalg.norm(g_variational_a) / np.linalg.norm(g_ad_a)
    assert abs(norm_ratio - 1.0) < 5.0e-7

    g_variational = project_gradient(
        gradient_log(model, g_variational_a), basis
    )
    directions = (0, 3, 6)
    step = 1.0e-4
    for index in directions:
        plus = delta_m.copy()
        minus = delta_m.copy()
        plus[index] += step
        minus[index] -= step
        c_plus = engine.phase_velocity(
            basis.m_to_forward_model(plus), datum, c_hint=solution.c
        ).c
        c_minus = engine.phase_velocity(
            basis.m_to_forward_model(minus), datum, c_hint=solution.c
        ).c
        finite_difference = (c_plus - c_minus) / (2.0 * step)
        scale = max(abs(finite_difference), 1.0e-10)
        assert abs(g_variational[index] - finite_difference) / scale < 1.0e-6


def test_mode_order_root_is_polished_to_the_derivative_residual_contract():
    """A steep Love-1 level-5 root must remain usable by derivatives."""

    _, model, _ = love_overtone_residual_model()
    datum = Datum("Love", 1, 1.0 / 0.06)
    engine = ForwardEngine(
        max_steps=6,
        tolerance=1.0e-9,
        hessian_max_levels=5,
        hessian_tolerance=1.0e-6,
    )
    solution = engine.phase_velocity(model, datum, minimum_trace_level=5)
    assert abs(solution.c - 4.571119394863) < 5.0e-10
    assert solution.level == 4
    assert len(solution.romberg_roots) == 6

    depths, parameters, is_discontinuity = model.to_params()
    variational = dp.compute_kernels_variational_grid(
        depths,
        parameters,
        is_discontinuity,
        0.06,
        solution.c,
        "Love",
        5,
        1.0e-6,
        phase_velocity_roots=solution.romberg_roots,
        phase_accepted_level=solution.level,
        **model.refinement_kwargs(),
    )
    assert np.all(np.isfinite(variational["gradient_node"]))

    tighter = ForwardEngine(
        max_steps=6,
        tolerance=1.0e-9,
        root_tolerance=1.0e-10,
        hessian_max_levels=5,
        hessian_tolerance=1.0e-6,
    ).phase_velocity(model, datum, minimum_trace_level=5)
    assert abs(tighter.c - solution.c) < 5.0e-10
    np.testing.assert_allclose(
        solution.romberg_roots,
        tighter.romberg_roots,
        rtol=0.0,
        atol=5.0e-11,
    )


def test_plain_forward_does_not_force_hessian_depth_trace():
    _, model, _ = love_fundamental_holdout_model()
    datum = Datum("Love", 0, 1.0 / 0.055)
    engine = ForwardEngine(max_steps=6, tolerance=1.0e-10)
    plain = engine.phase_velocity(model, datum)
    derivative_ready = engine.phase_velocity(
        model, datum, c_hint=plain.c, minimum_trace_level=5
    )
    assert len(plain.romberg_roots) == plain.level + 1
    assert len(derivative_ready.romberg_roots) >= 6
    assert abs(derivative_ready.c - plain.c) < 5.0e-10


def test_mismatched_trace_fails_closed():
    _, model, _ = love_fundamental_holdout_model()
    datum = Datum("Love", 0, 1.0 / 0.055)
    solution = ForwardEngine(max_steps=6, tolerance=1.0e-10).phase_velocity(
        model, datum, minimum_trace_level=5
    )
    bad_roots = list(solution.romberg_roots)
    bad_roots[0] += 0.01
    depths, parameters, is_discontinuity = model.to_params()
    try:
        dp.compute_kernels_variational_grid(
            depths,
            parameters,
            is_discontinuity,
            0.055,
            solution.c,
            "Love",
            5,
            1.0e-8,
            phase_velocity_roots=bad_roots,
            phase_accepted_level=solution.level,
            **model.refinement_kwargs(),
        )
    except (RuntimeError, ValueError) as exc:
        assert (
            "root trace" in str(exc)
            or "phase_velocity_roots" in str(exc)
        )
    else:
        raise AssertionError("a mismatched Romberg root trace was accepted")


def test_reduced_hessian_consumes_the_same_root_trace():
    basis, model, _ = love_fundamental_holdout_model()
    datum = Datum("Love", 0, 1.0 / 0.055)
    engine = ForwardEngine(
        max_steps=6,
        tolerance=1.0e-10,
        hessian_backend="vector",
    )
    solution = engine.phase_velocity(model, datum, minimum_trace_level=5)
    depths, parameters, is_discontinuity = model.to_params()
    gradient = dp.compute_kernels_variational_grid(
        depths,
        parameters,
        is_discontinuity,
        0.055,
        solution.c,
        "Love",
        5,
        1.0e-8,
        phase_velocity_roots=solution.romberg_roots,
        phase_accepted_level=solution.level,
        **model.refinement_kwargs(),
    )
    reduced = engine.datum_hessian_reduced(
        model,
        datum,
        solution.c,
        basis_node_jacobian(model, basis),
        phase_velocity_roots=solution.romberg_roots,
        phase_accepted_level=solution.level,
    )
    np.testing.assert_allclose(
        reduced["gradient_node"],
        gradient["gradient_node"],
        rtol=5.0e-7,
        atol=1.0e-12,
    )
    assert np.all(np.isfinite(reduced["hessian_reduced"]))
    assert reduced["asymmetry_rel"] < 1.0e-12


def test_trace_reuse_reduces_dispersion_evaluations():
    _, model, _ = love_fundamental_holdout_model()
    datum = Datum("Love", 0, 1.0 / 0.055)
    engine = ForwardEngine(max_steps=6, tolerance=1.0e-10)

    dp.reset_dispersion_eval_counts()
    solution = engine.phase_velocity(model, datum, minimum_trace_level=5)
    direct_count = int(dp.get_dispersion_eval_counts()["total"])

    depths, parameters, is_discontinuity = model.to_params()
    common = dict(
        depths=depths,
        params=parameters,
        is_discontinuity=is_discontinuity,
        freq=0.055,
        c_guess=solution.c,
        wave_type="Love",
        max_levels=5,
        tolerance=1.0e-8,
        **model.refinement_kwargs(),
    )
    dp.reset_dispersion_eval_counts()
    dp.compute_kernels_variational_grid(**common)
    independent_count = int(dp.get_dispersion_eval_counts()["total"])

    dp.reset_dispersion_eval_counts()
    dp.compute_kernels_variational_grid(
        **common,
        phase_velocity_roots=solution.romberg_roots,
        phase_accepted_level=solution.level,
    )
    trace_count = int(dp.get_dispersion_eval_counts()["total"])

    baseline = direct_count + independent_count
    traced = direct_count + trace_count
    assert trace_count <= len(solution.romberg_roots)
    assert 1.0 - traced / baseline >= 0.25


if __name__ == "__main__":
    test_trace_driven_variational_gradient_stays_on_love_fundamental()
    test_mode_order_root_is_polished_to_the_derivative_residual_contract()
    test_plain_forward_does_not_force_hessian_depth_trace()
    test_mismatched_trace_fails_closed()
    test_reduced_hessian_consumes_the_same_root_trace()
    test_trace_reuse_reduces_dispersion_evaluations()
    print("test_variational_root_trace_consistency: PASS")
