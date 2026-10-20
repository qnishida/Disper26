#!/usr/bin/env python3
"""Unit tests for the Plan L v2 mixed-response Taylor artifact."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "inversion" / "disper26_inv"
package = types.ModuleType("disper26_inv")
package.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("disper26_inv", package)

from disper26_inv.long_period_taylor import (  # noqa: E402
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
    PHASE_VELOCITY_RESPONSE_KIND,
    PRIMARY_FREQUENCY_HZ,
    LongPeriodTaylorArtifact,
    LongPeriodTaylorCurve,
    LongPeriodTaylorProvenance,
    build_long_period_taylor_curve,
)
from disper26_inv.long_period_taylor_io import (  # noqa: E402
    compute_long_period_mixed_response_artifact_hash,
    load_long_period_mixed_response_artifact,
    load_long_period_taylor_artifact,
    save_long_period_mixed_response_artifact,
    save_long_period_taylor_artifact,
)


def _phase_jet(curve_index: int):
    frequency = np.asarray(PRIMARY_FREQUENCY_HZ)
    c0 = 3.0 + 0.2 * curve_index + 0.01 * np.arange(frequency.size)
    gradient_m = np.column_stack(
        (
            0.2 + 0.01 * np.arange(frequency.size),
            -0.1 + 0.005 * np.arange(frequency.size),
        )
    )
    hessian_m = np.empty((frequency.size, 2, 2))
    hessian_m[:, 0, 0] = 0.4
    hessian_m[:, 0, 1] = 0.05
    hessian_m[:, 1, 0] = 0.05
    hessian_m[:, 1, 1] = -0.2
    return c0, gradient_m, hessian_m


def _provenance():
    return LongPeriodTaylorProvenance(
        code_commit="a" * 40,
        plan_sha256="b" * 64,
        reference_manifest_sha256="c" * 64,
        depth_convergence_report_sha256="d" * 64,
        reference_fingerprint="e" * 64,
        basis_fingerprint="f" * 64,
        prior_fingerprint="1" * 64,
        dataset_fingerprint="2" * 64,
        calibration_manifest_sha256="3" * 64,
    )


def synthetic_artifact(*, love_gradient_scale: float = 1.0):
    frequency = np.asarray(PRIMARY_FREQUENCY_HZ)
    prior = np.asarray([[0.04, 0.0], [0.01, 0.03]])
    curves = []
    for curve_index, (wave, root) in enumerate(
        (("rayleigh", 0), ("rayleigh", 1), ("love", 0), ("love", 1))
    ):
        c0, gradient_m, hessian_m = _phase_jet(curve_index)
        if wave == "love":
            gradient_m = love_gradient_scale * gradient_m
            response_kind = NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND
        else:
            response_kind = PHASE_VELOCITY_RESPONSE_KIND
        curves.append(
            build_long_period_taylor_curve(
                wave=wave,
                root_order=root,
                frequency_hz=frequency,
                reference_phase_velocity=c0,
                phase_gradient_m=gradient_m,
                phase_hessian_m=hessian_m,
                prior_cholesky=prior,
                response_kind=response_kind,
            )
        )
    return LongPeriodTaylorArtifact(
        prior_cholesky=prior,
        shear_log_operator=np.eye(2),
        curves=tuple(curves),
        provenance=_provenance(),
        metadata_json=json.dumps(
            {
                "purpose": "mixed-response-unit-test",
                "response_kind_by_wave": {
                    "rayleigh": PHASE_VELOCITY_RESPONSE_KIND,
                    "love": NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
                },
            }
        ),
        schema_name=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
        schema_version=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    )


def synthetic_v1_artifact():
    frequency = np.asarray(PRIMARY_FREQUENCY_HZ)
    prior = np.asarray([[0.04, 0.0], [0.01, 0.03]])
    curves = []
    for curve_index, (wave, root) in enumerate(
        (("rayleigh", 0), ("rayleigh", 1), ("love", 0), ("love", 1))
    ):
        c0, gradient_m, hessian_m = _phase_jet(curve_index)
        curves.append(
            build_long_period_taylor_curve(
                wave=wave,
                root_order=root,
                frequency_hz=frequency,
                reference_phase_velocity=c0,
                phase_gradient_m=gradient_m,
                phase_hessian_m=hessian_m,
                prior_cholesky=prior,
            )
        )
    return LongPeriodTaylorArtifact(
        prior_cholesky=prior,
        shear_log_operator=np.eye(2),
        curves=tuple(curves),
        provenance=_provenance(),
    )


def test_love_squared_slowness_chain_rule_and_normalization():
    artifact = synthetic_artifact()
    curve = artifact.get_curve("Love", 0)
    c0, gradient_m, hessian_m = _phase_jet(2)
    prior = artifact.prior_cholesky
    gradient_c_z = gradient_m @ prior
    hessian_c_z = np.einsum(
        "ai,fab,bj->fij", prior, hessian_m, prior, optimize=True
    )
    expected_gradient = -2.0 * gradient_c_z / c0[:, None]
    expected_hessian = (
        6.0
        * gradient_c_z[:, :, None]
        * gradient_c_z[:, None, :]
        / np.square(c0)[:, None, None]
        - 2.0 * hessian_c_z / c0[:, None, None]
    )
    np.testing.assert_allclose(
        curve.reference_response, np.ones_like(c0), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        curve.response_gradient_z, expected_gradient, rtol=2.0e-15, atol=2.0e-15
    )
    np.testing.assert_allclose(
        curve.response_hessian_z, expected_hessian, rtol=3.0e-15, atol=3.0e-15
    )

    z = np.asarray([0.07, -0.04])
    normalized, normalized_linear, normalized_quadratic = (
        curve.evaluate_response_z(z, order=2)
    )
    unnormalized_reference = 1.0 / np.square(c0)
    unnormalized_gradient = expected_gradient / np.square(c0)[:, None]
    unnormalized_hessian = expected_hessian / np.square(c0)[:, None, None]
    unnormalized = (
        unnormalized_reference
        + unnormalized_gradient @ z
        + 0.5 * ((unnormalized_hessian @ z) @ z)
    )
    np.testing.assert_allclose(
        c0 / np.sqrt(normalized),
        1.0 / np.sqrt(unnormalized),
        rtol=3.0e-15,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        normalized,
        normalized_linear + normalized_quadratic,
        rtol=0.0,
        atol=2.0e-16,
    )


def test_mapped_second_order_has_the_original_phase_jet():
    artifact = synthetic_artifact()
    curve = artifact.get_curve("Love", 0)
    c0, gradient_m, hessian_m = _phase_jet(2)
    prior = artifact.prior_cholesky
    gradient_c_z = gradient_m @ prior
    hessian_c_z = np.einsum(
        "ai,fab,bj->fij", prior, hessian_m, prior, optimize=True
    )
    direction = np.asarray([0.6, -0.8])

    errors = []
    for scale in (0.04, 0.02, 0.01):
        z = scale * direction
        mapped, _, _ = curve.evaluate_z(z, order=2)
        direct_phase_taylor2 = (
            c0
            + gradient_c_z @ z
            + 0.5 * ((hessian_c_z @ z) @ z)
        )
        errors.append(float(np.max(np.abs(mapped - direct_phase_taylor2))))
    assert errors[0] > errors[1] > errors[2] > 0.0
    assert 7.0 < errors[0] / errors[1] < 9.0
    assert 7.0 < errors[1] / errors[2] < 9.0


def test_rayleigh_v2_is_identical_to_phase_velocity_taylor():
    mixed = synthetic_artifact()
    mixed_curve = mixed.get_curve("Rayleigh", 1)
    c0, gradient_m, hessian_m = _phase_jet(1)
    phase_curve = build_long_period_taylor_curve(
        wave="rayleigh",
        root_order=1,
        frequency_hz=PRIMARY_FREQUENCY_HZ,
        reference_phase_velocity=c0,
        phase_gradient_m=gradient_m,
        phase_hessian_m=hessian_m,
        prior_cholesky=mixed.prior_cholesky,
    )
    z = np.asarray([0.08, -0.03])
    for order in (1, 2):
        mixed_values = mixed_curve.evaluate_z(z, order=order)
        phase_values = phase_curve.evaluate_z(z, order=order)
        for actual, expected in zip(mixed_values, phase_values):
            np.testing.assert_array_equal(actual, expected)


def test_prediction_reports_response_kind_and_fails_closed():
    artifact = synthetic_artifact()
    delta = np.asarray([0.025, -0.015])
    result = artifact.predict("Love", 1, delta, order=2)
    curve = artifact.get_curve("Love", 1)
    z = np.linalg.solve(artifact.prior_cholesky, delta)
    response2, response1, response_quadratic = curve.evaluate_response_z(
        z, order=2
    )
    expected1 = curve.reference_phase_velocity / np.sqrt(response1)
    expected2 = curve.reference_phase_velocity / np.sqrt(response2)
    assert result.supported
    assert result.response_kind == NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND
    np.testing.assert_allclose(
        result.linear_phase_velocity, expected1, rtol=2.0e-15, atol=2.0e-15
    )
    np.testing.assert_allclose(
        result.phase_velocity, expected2, rtol=2.0e-15, atol=2.0e-15
    )
    np.testing.assert_allclose(
        result.phase_velocity_order1, expected1, rtol=2.0e-15, atol=2.0e-15
    )
    np.testing.assert_allclose(
        result.phase_velocity_order2, expected2, rtol=2.0e-15, atol=2.0e-15
    )
    np.testing.assert_allclose(
        result.quadratic_correction,
        expected2 - expected1,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert np.all(response1 + response_quadratic > 0.0)

    unsafe_base = synthetic_artifact()
    unsafe_curves = list(unsafe_base.curves)
    unsafe_source = unsafe_curves[2]
    unsafe_curves[2] = LongPeriodTaylorCurve(
        wave="love",
        root_order=0,
        frequency_hz=unsafe_source.frequency_hz,
        reference_phase_velocity=unsafe_source.reference_phase_velocity,
        gradient_z=np.full_like(unsafe_source.gradient_z, -10.0),
        hessian_z=np.zeros_like(unsafe_source.hessian_z),
        response_kind=NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
    )
    unsafe = LongPeriodTaylorArtifact(
        prior_cholesky=unsafe_base.prior_cholesky,
        shear_log_operator=unsafe_base.shear_log_operator,
        curves=tuple(unsafe_curves),
        provenance=unsafe_base.provenance,
        schema_name=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
        schema_version=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    )
    rejected = unsafe.predict("Love", 0, np.asarray([0.01, 0.0]), order=2)
    assert not rejected.supported
    assert rejected.reason == "nonfinite"
    assert np.all(np.isnan(rejected.phase_velocity))


def test_deterministic_v2_round_trip_and_cross_schema_rejection():
    artifact = synthetic_artifact()
    v1_artifact = synthetic_v1_artifact()
    with tempfile.TemporaryDirectory() as directory:
        first_path = Path(directory) / "first.npz"
        second_path = Path(directory) / "second.npz"
        v1_path = Path(directory) / "v1.npz"
        save_long_period_mixed_response_artifact(artifact, first_path)
        save_long_period_mixed_response_artifact(artifact, second_path)
        save_long_period_taylor_artifact(v1_artifact, v1_path)
        assert compute_long_period_mixed_response_artifact_hash(
            first_path
        ) == compute_long_period_mixed_response_artifact_hash(second_path)

        loaded = load_long_period_mixed_response_artifact(first_path)
        assert loaded.schema_name == LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME
        assert tuple(curve.response_kind for curve in loaded.curves) == (
            PHASE_VELOCITY_RESPONSE_KIND,
            PHASE_VELOCITY_RESPONSE_KIND,
            NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
            NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
        )
        for left, right in zip(loaded.curves, artifact.curves):
            np.testing.assert_array_equal(
                left.reference_phase_velocity, right.reference_phase_velocity
            )
            np.testing.assert_array_equal(
                left.response_gradient_z, right.response_gradient_z
            )
            np.testing.assert_array_equal(
                left.response_hessian_z, right.response_hessian_z
            )

        try:
            load_long_period_taylor_artifact(first_path)
        except ValueError as error:
            assert "unsupported" in str(error)
        else:
            raise AssertionError("schema-v1 loader accepted a schema-v2 artifact")

        try:
            save_long_period_taylor_artifact(artifact, Path(directory) / "v1.npz")
        except ValueError as error:
            assert "schema-v2" in str(error) or "v1" in str(error)
        else:
            raise AssertionError("schema-v1 writer accepted a schema-v2 artifact")

        try:
            load_long_period_mixed_response_artifact(v1_path)
        except ValueError as error:
            assert "unsupported" in str(error)
        else:
            raise AssertionError("schema-v2 loader accepted a schema-v1 artifact")

        try:
            save_long_period_mixed_response_artifact(
                v1_artifact, Path(directory) / "not-v2.npz"
            )
        except ValueError as error:
            assert "schema-v2" in str(error)
        else:
            raise AssertionError("schema-v2 writer accepted a schema-v1 artifact")


def test_schema_rejects_response_kind_mismatch():
    artifact = synthetic_artifact()
    curves = list(artifact.curves)
    c0, gradient_m, hessian_m = _phase_jet(2)
    curves[2] = build_long_period_taylor_curve(
        wave="love",
        root_order=0,
        frequency_hz=PRIMARY_FREQUENCY_HZ,
        reference_phase_velocity=c0,
        phase_gradient_m=gradient_m,
        phase_hessian_m=hessian_m,
        prior_cholesky=artifact.prior_cholesky,
        response_kind=PHASE_VELOCITY_RESPONSE_KIND,
    )
    try:
        LongPeriodTaylorArtifact(
            prior_cholesky=artifact.prior_cholesky,
            shear_log_operator=artifact.shear_log_operator,
            curves=tuple(curves),
            provenance=artifact.provenance,
            schema_name=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
            schema_version=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
        )
    except ValueError as error:
        assert "response kinds" in str(error)
    else:
        raise AssertionError("schema v2 accepted a phase-velocity Love jet")


def main():
    test_love_squared_slowness_chain_rule_and_normalization()
    test_mapped_second_order_has_the_original_phase_jet()
    test_rayleigh_v2_is_identical_to_phase_velocity_taylor()
    test_prediction_reports_response_kind_and_fails_closed()
    test_deterministic_v2_round_trip_and_cross_schema_rejection()
    test_schema_rejects_response_kind_mismatch()
    print("long-period mixed-response Taylor checks passed")


if __name__ == "__main__":
    main()
