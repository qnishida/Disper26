#!/usr/bin/env python3
"""Compatibility and shape checks for the trace-bearing Romberg API."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))

import disper26py as dp  # noqa: E402
from disper26_inv.engine import Datum, ForwardEngine  # noqa: E402
from disper26_inv.shallow import shallow_reference  # noqa: E402
from test_romberg_mode_order_tracking import validation_failure_model  # noqa: E402


def test_trace_api_preserves_legacy_values_and_extends_roots():
    model = validation_failure_model()
    depths, parameters, is_discontinuity = model.to_params()
    kwargs = dict(
        depths=depths,
        params=parameters,
        is_discontinuity=is_discontinuity,
        period=1.0 / 0.12625,
        wave_type="Love",
        c_guess=4.53108,
        max_steps=6,
        tolerance=1.0e-9,
        branch=1,
        **model.refinement_kwargs(),
    )
    legacy = dp.compute_dispersion_romberg(**kwargs)
    traced = dp.compute_dispersion_romberg_levels(
        **kwargs, minimum_trace_level=6
    )

    assert traced["phase_velocity"] == legacy[0]
    assert traced["group_velocity"] == legacy[1]
    if np.isnan(legacy[2]):
        assert np.isnan(traced["hv_ratio"])
    else:
        assert traced["hv_ratio"] == legacy[2]
    assert traced["converged_level"] == legacy[3]
    assert traced["phase_accepted_level"] == legacy[3]
    assert traced["relative_error"] == legacy[4]
    roots = np.asarray(traced["phase_velocity_roots"], dtype=float)
    assert traced["generated_level"] == 6
    assert roots.shape == (7,)
    assert np.all(np.isfinite(roots))
    assert np.all(roots > 0.0)

    # This Love-overtone trace contains a producer-accepted level whose raw
    # secular residual is about 2.9e-4.  Consumers must use the same 0.1
    # acceptance contract as the direct adaptive root search, rather than a
    # stricter normalization-dependent threshold.
    variational = dp.compute_kernels_variational_grid(
        depths,
        parameters,
        is_discontinuity,
        0.12625,
        traced["phase_velocity"],
        "Love",
        6,
        1.0e-8,
        phase_velocity_roots=roots,
        phase_accepted_level=traced["phase_accepted_level"],
        **model.refinement_kwargs(),
    )
    assert np.all(np.isfinite(variational["gradient_node"]))
    try:
        dp.compute_kernels_variational_grid(
            depths,
            parameters,
            is_discontinuity,
            0.12625,
            traced["phase_velocity"],
            "Love",
            6,
            1.0e-8,
            phase_velocity_roots=roots,
            **model.refinement_kwargs(),
        )
    except (RuntimeError, ValueError) as exc:
        assert "phase_accepted_level" in str(exc)
    else:
        raise AssertionError("a root trace without its accepted level was accepted")
    ad = dp.compute_kernels_ad_grid(
        depths,
        parameters,
        is_discontinuity,
        1.0 / 0.12625,
        "Love",
        traced["phase_velocity"],
        "VTI",
        6,
        1.0e-9,
        want_kernel_u=False,
        want_kernel_hv=False,
        phase_velocity_roots=roots,
        phase_accepted_level=traced["phase_accepted_level"],
        **model.refinement_kwargs(),
    )
    assert ad[0] == traced["phase_velocity"]

    love0 = dp.compute_dispersion_romberg_levels(
        depths,
        parameters,
        is_discontinuity,
        1.0 / 0.12625,
        "Love",
        3.4,
        6,
        1.0e-9,
        branch=0,
        **model.refinement_kwargs(),
    )
    try:
        dp.compute_kernels_variational_grid(
            depths,
            parameters,
            is_discontinuity,
            0.12625,
            love0["phase_velocity"],
            "Love",
            6,
            1.0e-8,
            phase_velocity_roots=roots,
            phase_accepted_level=traced["phase_accepted_level"],
            **model.refinement_kwargs(),
        )
    except (RuntimeError, ValueError) as exc:
        assert "different direct phase solution or branch" in str(exc)
    else:
        raise AssertionError("a valid trace from a different branch was accepted")


def test_trace_can_extend_beyond_phase_max_steps():
    model = validation_failure_model()
    depths, parameters, is_discontinuity = model.to_params()
    legacy = dp.compute_dispersion_romberg(
        depths,
        parameters,
        is_discontinuity,
        1.0 / 0.12625,
        "Love",
        4.53108,
        4,
        -1.0,
        branch=1,
        **model.refinement_kwargs(),
    )
    traced = dp.compute_dispersion_romberg_levels(
        depths,
        parameters,
        is_discontinuity,
        1.0 / 0.12625,
        "Love",
        4.53108,
        4,
        -1.0,
        branch=1,
        minimum_trace_level=6,
        **model.refinement_kwargs(),
    )
    assert traced["phase_velocity"] == legacy[0]
    assert traced["converged_level"] == legacy[3] == 4
    assert traced["generated_level"] == 6
    assert len(traced["phase_velocity_roots"]) == 7

    try:
        dp.compute_dispersion_romberg_levels(
            depths,
            parameters,
            is_discontinuity,
            1.0 / 0.12625,
            "Love",
            4.53108,
            4,
            -1.0,
            minimum_trace_level=6,
            **model.refinement_kwargs(),
        )
    except (RuntimeError, ValueError) as exc:
        assert "requires branch" in str(exc)
    else:
        raise AssertionError("trace-only refinement without branch was accepted")


def test_root_tolerance_is_independent_and_validated():
    model = validation_failure_model()
    depths, parameters, is_discontinuity = model.to_params()
    common = dict(
        depths=depths,
        params=parameters,
        is_discontinuity=is_discontinuity,
        period=1.0 / 0.12625,
        wave_type="Love",
        c_guess=4.53108,
        max_steps=6,
        tolerance=1.0e-9,
        branch=1,
        **model.refinement_kwargs(),
    )
    default = dp.compute_dispersion_romberg_levels(**common)
    explicit = dp.compute_dispersion_romberg_levels(
        **common, root_tolerance=1.0e-9
    )
    assert default["phase_velocity_roots"] == explicit["phase_velocity_roots"]
    assert default["phase_velocity"] == explicit["phase_velocity"]

    try:
        dp.compute_dispersion_romberg_levels(**common, root_tolerance=0.0)
    except (RuntimeError, ValueError) as exc:
        assert "root_tolerance" in str(exc)
    else:
        raise AssertionError("non-positive root_tolerance was accepted")

    # Trace consumers currently extrapolate from level zero.  Reject a
    # nonzero start at trace production rather than silently differentiating
    # a different Romberg table.  The legacy value-only API remains callable.
    legacy_start_one = dp.compute_dispersion_romberg(
        **common, extrapolate_start_level=1
    )
    assert np.isfinite(legacy_start_one[0])
    try:
        dp.compute_dispersion_romberg_levels(
            **common, extrapolate_start_level=1
        )
    except (RuntimeError, ValueError) as exc:
        assert "extrapolate_start_level=0" in str(exc)
    else:
        raise AssertionError("trace API accepted a nonzero extrapolation start")


def test_mode_order_trace_accepts_a_small_count_root_offset():
    model = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
    engine = ForwardEngine(max_steps=6, tolerance=1.0e-10)
    solution = engine.phase_velocity(
        model,
        Datum("Love", 0, 20.0),
        minimum_trace_level=5,
    )
    assert len(solution.romberg_roots) >= 6
    assert np.all(np.isfinite(solution.romberg_roots))


if __name__ == "__main__":
    test_trace_api_preserves_legacy_values_and_extends_roots()
    test_trace_can_extend_beyond_phase_max_steps()
    test_root_tolerance_is_independent_and_validated()
    test_mode_order_trace_accepts_a_small_count_root_offset()
    print("test_romberg_level_root_trace: PASS")
