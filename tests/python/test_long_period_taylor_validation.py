#!/usr/bin/env python3
"""Deterministic synthetic tests for the frozen Plan-L accuracy gates."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "inversion" / "disper26_inv"
sys.path.insert(0, str(ROOT))
package = types.ModuleType("disper26_inv")
package.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("disper26_inv", package)

from disper26_inv.long_period_taylor import (  # noqa: E402
    PRIMARY_FREQUENCY_HZ,
    LongPeriodTaylorArtifact,
    LongPeriodTaylorProvenance,
    build_long_period_taylor_curve,
)
from disper26_inv.long_period_taylor_manifest import (  # noqa: E402
    AMPLITUDE_LEVELS,
    SEALED_HOLDOUT_SEED,
    build_radial_prior_cohort,
)
from disper26_inv.long_period_taylor_mode import (  # noqa: E402
    ModeCharacterDiagnostics,
)
from disper26_inv.long_period_taylor_validation import (  # noqa: E402
    classify_mode_identity,
    evaluate_plan_l_gate,
    validate_long_period_taylor_curve,
)


def synthetic_problem():
    frequency = np.asarray(PRIMARY_FREQUENCY_HZ)
    prior = np.eye(2)
    curves = []
    for curve_index, (wave, root) in enumerate(
        (("rayleigh", 0), ("rayleigh", 1), ("love", 0), ("love", 1))
    ):
        c0 = 3.0 + 0.25 * curve_index + 0.01 * np.arange(frequency.size)
        gradient = np.tile(np.asarray([0.08, -0.03]), (frequency.size, 1))
        hessian = np.zeros((frequency.size, 2, 2))
        hessian[:, 0, 0] = 0.30
        hessian[:, 0, 1] = 0.04
        hessian[:, 1, 0] = 0.04
        hessian[:, 1, 1] = -0.10
        curves.append(
            build_long_period_taylor_curve(
                wave=wave,
                root_order=root,
                frequency_hz=frequency,
                reference_phase_velocity=c0,
                phase_gradient_m=gradient,
                phase_hessian_m=hessian,
                prior_cholesky=prior,
            )
        )
    artifact = LongPeriodTaylorArtifact(
        prior_cholesky=prior,
        shear_log_operator=np.eye(2),
        curves=tuple(curves),
        provenance=LongPeriodTaylorProvenance(
            code_commit="a" * 40,
            plan_sha256="b" * 64,
            reference_manifest_sha256="c" * 64,
            depth_convergence_report_sha256="d" * 64,
            reference_fingerprint="e" * 64,
            basis_fingerprint="f" * 64,
            prior_fingerprint="1" * 64,
            dataset_fingerprint="2" * 64,
            calibration_manifest_sha256="3" * 64,
        ),
        metadata_json=json.dumps({"purpose": "validation-unit-test"}),
    )

    models = []
    amplitudes = []
    for _ in range(8):
        for amplitude in AMPLITUDE_LEVELS:
            for sign in (-1.0, 1.0):
                coordinate = (
                    np.log1p(amplitude)
                    if sign > 0.0
                    else np.log1p(-amplitude)
                )
                models.append([coordinate, 0.0])
                amplitudes.append(amplitude)
    models = np.asarray(models)
    amplitudes = np.asarray(amplitudes)
    exact_by_curve = {}
    diagnostics = {}
    sign = np.sign(models[:, 0])[:, None]
    for curve in artifact.curves:
        taylor2, _, _, _ = artifact.evaluate_unchecked_batch(
            curve.wave, curve.root_order, models, order=2
        )
        frequency_scale = 1.0 + 0.02 * np.arange(frequency.size)[None, :]
        remainder = (
            2.0e-3
            * sign
            * amplitudes[:, None] ** 3
            * frequency_scale
        )
        exact_by_curve[curve.key] = taylor2 + remainder
        shape = exact_by_curve[curve.key].shape
        diagnostics[curve.key] = ModeCharacterDiagnostics(
            adjacent_relative_gap=np.full(shape, 0.08),
            eigenfunction_overlap=np.full(shape, 0.99),
            kernel_overlap=np.full(shape, 0.98),
            crustal_energy_fraction=np.full(shape, 0.45),
            energy_centroid_jump_km=np.full(shape, 1.0),
            group_velocity_relative_jump=np.full(shape, 0.01),
            trace_agreement=np.ones(shape, dtype=bool),
            exact_success=np.ones(shape, dtype=bool),
        )
    return artifact, models, amplitudes, exact_by_curve, diagnostics


def test_all_four_curves_pass_cubic_remainder_gate():
    artifact, models, amplitudes, exact, diagnostics = synthetic_problem()
    reports = []
    for curve in artifact.curves:
        values = exact[curve.key]
        report = validate_long_period_taylor_curve(
            artifact,
            curve.wave,
            curve.root_order,
            models,
            values,
            np.ones(values.shape, dtype=bool),
            diagnostics[curve.key],
            amplitudes,
        )
        assert report.passed
        assert report.supported_fraction == 1.0
        assert report.minimum_primary_model_support_fraction == 1.0
        assert report.minimum_primary_contiguous_support_count == 10
        assert report.low_frequency_contiguous_pass_count == 10
        assert report.buffered_candidate_frequency_count == 10
        assert report.stress_10_percent.model_count == 16
        assert all(item.taylor2_residual_order > 2.99 for item in report.frequency_results)
        assert all(item.taylor1_residual_order > 1.9 for item in report.frequency_results)
        reports.append(report)
    promotion = evaluate_plan_l_gate(reports, split_kind="development")
    assert promotion.passed


def test_one_frequency_failure_blocks_whole_curve():
    artifact, models, amplitudes, exact, diagnostics = synthetic_problem()
    values = exact[("love", 0)].copy()
    primary = amplitudes <= 0.08
    values[primary, 4] += 0.01
    report = validate_long_period_taylor_curve(
        artifact,
        "Love",
        0,
        models,
        values,
        np.ones(values.shape, dtype=bool),
        diagnostics[("love", 0)],
        amplitudes,
    )
    assert not report.passed
    assert not report.frequency_results[4].passed
    assert report.low_frequency_contiguous_pass_count == 4
    assert report.buffered_candidate_frequency_count == 3


def test_mode_identity_rule_and_sealed_seed_guard():
    shape = (2, 3)
    diagnostics = ModeCharacterDiagnostics(
        adjacent_relative_gap=np.asarray([[0.01] * 3, [0.05] * 3]),
        eigenfunction_overlap=np.full(shape, 0.5),
        kernel_overlap=np.full(shape, 0.5),
        crustal_energy_fraction=np.full(shape, 0.5),
        energy_centroid_jump_km=np.full(shape, 2.0),
        group_velocity_relative_jump=np.full(shape, 0.4),
        trace_agreement=np.ones(shape, dtype=bool),
        exact_success=np.ones(shape, dtype=bool),
    )
    safe = classify_mode_identity(diagnostics)
    assert not np.any(safe[0])
    assert np.all(safe[1])

    try:
        build_radial_prior_cohort(
            None,
            np.eye(1),
            split_kind="holdout",
            seed=SEALED_HOLDOUT_SEED,
        )
    except PermissionError as error:
        assert "frozen gate lock" in str(error)
    else:
        raise AssertionError("sealed holdout seed was usable without a gate lock")


def main():
    test_all_four_curves_pass_cubic_remainder_gate()
    test_one_frequency_failure_blocks_whole_curve()
    test_mode_identity_rule_and_sealed_seed_guard()
    print("long-period Taylor validation checks passed")


if __name__ == "__main__":
    main()
