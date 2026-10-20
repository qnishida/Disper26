#!/usr/bin/env python3
"""Synthetic gate and split-policy tests for Plan L v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))

from disper26_inv.long_period_taylor import (  # noqa: E402
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
    PHASE_VELOCITY_RESPONSE_KIND,
    PRIMARY_FREQUENCY_HZ,
    LongPeriodTaylorArtifact,
    LongPeriodTaylorProvenance,
    build_long_period_taylor_curve,
)
from disper26_inv.long_period_taylor_manifest import (  # noqa: E402
    AMPLITUDE_LEVELS,
    MIXED_RESPONSE_DEVELOPMENT_SEED,
    MIXED_RESPONSE_SEALED_HOLDOUT_SEED,
    LongPeriodMixedResponseDatasetManifest,
    LongPeriodMixedResponseSplitManifest,
    build_mixed_response_radial_prior_cohort,
)
from disper26_inv.long_period_taylor_mode import (  # noqa: E402
    ModeCharacterDiagnostics,
)
from disper26_inv.long_period_taylor_validation import (  # noqa: E402
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
                response_kind=(
                    PHASE_VELOCITY_RESPONSE_KIND
                    if wave == "rayleigh"
                    else NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND
                ),
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
        metadata_json=json.dumps({"purpose": "mixed-validation-unit-test"}),
        schema_name=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
        schema_version=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
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
        response_order2, _, _, _ = artifact.evaluate_unchecked_batch(
            curve.wave, curve.root_order, models, order=2
        )
        frequency_scale = 1.0 + 0.02 * np.arange(frequency.size)[None, :]
        remainder = (
            2.0e-3
            * sign
            * amplitudes[:, None] ** 3
            * frequency_scale
        )
        exact_by_curve[curve.key] = response_order2 + remainder
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


def test_all_four_mixed_response_curves_pass_cubic_remainder_gate():
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
        assert report.response_kind == curve.response_kind
        assert report.nonpositive_response_count == 0
        assert report.minimum_response_order1 > 0.0
        assert report.minimum_response_order2 > 0.0
        assert all(
            item.response_kind == curve.response_kind
            and item.nonpositive_response_count == 0
            for item in report.frequency_results
        )
        reports.append(report)
    promotion = evaluate_plan_l_gate(reports, split_kind="development")
    assert promotion.passed


def test_v2_manifest_and_sealed_seed_guard():
    dataset = LongPeriodMixedResponseDatasetManifest()
    splits = LongPeriodMixedResponseSplitManifest()
    assert dataset.response_kinds == {
        "rayleigh": PHASE_VELOCITY_RESPONSE_KIND,
        "love": NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
    }
    assert splits.development_seed == MIXED_RESPONSE_DEVELOPMENT_SEED
    assert splits.sealed_holdout_seed == MIXED_RESPONSE_SEALED_HOLDOUT_SEED

    try:
        build_mixed_response_radial_prior_cohort(
            None,
            np.eye(1),
            split_kind="holdout",
            seed=MIXED_RESPONSE_SEALED_HOLDOUT_SEED,
        )
    except PermissionError as error:
        assert "frozen gate lock" in str(error)
    else:
        raise AssertionError("v2 sealed holdout was usable without a gate lock")


def main():
    test_all_four_mixed_response_curves_pass_cubic_remainder_gate()
    test_v2_manifest_and_sealed_seed_guard()
    print("long-period mixed-response validation checks passed")


if __name__ == "__main__":
    main()
