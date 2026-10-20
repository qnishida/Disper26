#!/usr/bin/env python3
"""Regression for Love mode-order retention through Romberg refinement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))

from disper26_inv.engine import Datum, ForwardEngine  # noqa: E402
from disper26_inv.model import VELOCITY_FIELDS  # noqa: E402
from disper26_inv.priors import PriorCovariance  # noqa: E402
from disper26_inv.shallow import shallow_basis, shallow_reference  # noqa: E402
import disper26py as dp  # noqa: E402


def _maximum_shear_anomaly(basis, delta_m):
    depth = np.linspace(0.0, 60.0, 1201)
    vsv_log = np.interp(depth, basis.inv_depths, delta_m[: basis.n_inv])
    vsh_log = np.interp(depth, basis.inv_depths, delta_m[basis.n_inv :])
    return float(
        max(np.max(np.abs(np.expm1(vsv_log))), np.max(np.abs(np.expm1(vsh_log))))
    )


def _scale_to_maximum(basis, direction, target):
    low = 0.0
    high = 1.0
    while _maximum_shear_anomaly(basis, high * direction) < target:
        high *= 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if _maximum_shear_anomaly(basis, middle * direction) < target:
            low = middle
        else:
            high = middle
    return high * direction


def _maximum_operator_shear_anomaly(basis, delta_m):
    operator = np.asarray(basis.full_log_operator(), dtype=float)
    n_nodes = basis.reference.n_nodes
    rows = []
    for field in ("vsv", "vsh"):
        index = VELOCITY_FIELDS.index(field)
        rows.append(operator[index * n_nodes : (index + 1) * n_nodes])
    anomaly = np.expm1(np.vstack(rows) @ np.asarray(delta_m, dtype=float))
    return float(np.max(np.abs(anomaly)))


def _scale_operator_to_maximum(basis, direction, target):
    low = 0.0
    high = 1.0
    while _maximum_operator_shear_anomaly(basis, high * direction) < target:
        high *= 2.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if _maximum_operator_shear_anomaly(basis, middle * direction) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high) * direction


def validation_failure_model(direction_index=1, target=0.02):
    """Reconstruct a positive validation ray from seed 20260822 exactly."""

    reference = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
    basis = shallow_basis(reference)
    prior = PriorCovariance(
        basis,
        sigma={"vsv": 0.035, "vsh": 0.035},
        param_corr="birch_tight_aniso",
    )
    direction = prior.sample(16, np.random.default_rng(20260822))[direction_index]
    direction /= np.linalg.norm(direction)
    delta_m = _scale_to_maximum(basis, direction, target)
    return basis.m_to_forward_model(delta_m, name="love_mode_order_regression")


def close_love_overtone_model():
    """Reconstruct validation direction 13, negative 8% from seed 20270208."""

    reference = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
    basis = shallow_basis(reference)
    prior = PriorCovariance(
        basis,
        sigma={"vsv": 0.035, "vsh": 0.035},
        param_corr="birch_tight_aniso",
    )
    directions = np.random.default_rng(20270208).standard_normal(
        (32, basis.n_params)
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    direction = -(prior.chol() @ directions[13])
    delta_m = _scale_operator_to_maximum(basis, direction, 0.08)
    return basis.m_to_forward_model(
        delta_m, name="love_close_overtone_confirmation_regression"
    )


def test_love_first_overtone_retains_mode_order():
    frequencies = np.arange(0.1175, 0.1275 + 1.0e-12, 0.00125)
    cases = ((1, 0.02), (10, 0.005), (12, 0.005))
    for direction_index, target in cases:
        model = validation_failure_model(direction_index, target)
        engine = ForwardEngine(max_steps=6, tolerance=1.0e-9)
        solutions = engine.trace_dispersion_curve(model, "Love", 1, frequencies)

        assert len(solutions) == frequencies.size
        assert all(solution.confirmation["accepted"] for solution in solutions)
        assert all(solution.branch == 1 for solution in solutions)
        assert np.all(np.isfinite([solution.c for solution in solutions]))
        assert all(
            len(solution.confirmation["records"][-1]["window_attempts"]) == 1
            for solution in solutions
        )
        assert all(
            solution.confirmation["records"][-1]["window_rel"]
            == engine.confirm_delta_rel
            for solution in solutions
        )
        if direction_index == 1:
            assert abs(solutions[7].c - 4.53107673) < 5.0e-7


def test_raw_romberg_branch_keyword_selects_mode_order():
    model = validation_failure_model()
    depths, parameters, is_discontinuity = model.to_params()
    result = dp.compute_dispersion_romberg(
        depths,
        parameters,
        is_discontinuity,
        1.0 / 0.12625,
        "Love",
        4.53108,
        6,
        1.0e-9,
        branch=1,
        **model.refinement_kwargs(),
    )
    assert abs(result[0] - 4.53107673) < 5.0e-7


def test_close_love_overtone_adapts_confirmation_without_switching_branch():
    """Resolve two valid close roots without weakening mode-order identity."""

    model = close_love_overtone_model()
    datum = Datum("Love", 1, 1.0 / 0.17)
    engine = ForwardEngine(
        max_steps=6,
        tolerance=1.0e-9,
        hessian_max_levels=5,
        hessian_tolerance=1.0e-6,
    )
    solution = engine.phase_velocity(model, datum, minimum_trace_level=5)

    assert engine.confirm_delta_rel == 3.0e-4
    assert abs(solution.c - 4.522236049761) < 5.0e-10
    assert abs(solution.U - 3.728208291523) < 5.0e-9
    assert solution.confirmation["accepted"]
    accepted_record = solution.confirmation["records"][-1]
    assert accepted_record["window_rel"] < engine.confirm_delta_rel
    assert accepted_record["count_below"] == 1
    assert accepted_record["count_above"] == 2
    assert accepted_record["window_attempts"] == [
        {
            "count_below": 1,
            "count_above": 3,
            "window_rel": 3.0e-4,
            "accepted": False,
        },
        {
            "count_below": 1,
            "count_above": 2,
            "window_rel": 1.5e-4,
            "accepted": True,
        },
    ]
    assert engine.counters["forward"] == 1

    depths, parameters, is_discontinuity = model.to_params()
    variational = dp.compute_kernels_variational_grid(
        depths,
        parameters,
        is_discontinuity,
        0.17,
        solution.c,
        "Love",
        5,
        1.0e-6,
        phase_velocity_roots=solution.romberg_roots,
        phase_accepted_level=solution.level,
        **model.refinement_kwargs(),
    )
    assert np.all(np.isfinite(variational["gradient_node"]))

    curve_engine = ForwardEngine(max_steps=6, tolerance=1.0e-9)
    curve = curve_engine.trace_dispersion_curve(
        model,
        "Love",
        1,
        (0.165, 0.17, 0.175),
    )
    np.testing.assert_allclose(
        [value.c for value in curve],
        [4.523712553987, 4.522236049761, 4.483173378136],
        rtol=0.0,
        atol=5.0e-10,
    )
    assert all(value.confirmation["accepted"] for value in curve)
    # At most one branch-indexed fallback is allowed for each warm start.
    assert 3 <= curve_engine.counters["forward"] <= 5

    # The neighboring Love-2 root is only 2.49e-4 away.  Adaptive shrinking
    # must reveal that its local count jump is 2 -> 3, not accept it as Love-1.
    wrong = engine._confirm_branch(model, datum, 4.523364051277)
    assert not wrong["accepted"]


if __name__ == "__main__":
    test_love_first_overtone_retains_mode_order()
    test_raw_romberg_branch_keyword_selects_mode_order()
    test_close_love_overtone_adapts_confirmation_without_switching_branch()
    print("test_romberg_mode_order_tracking: PASS")
