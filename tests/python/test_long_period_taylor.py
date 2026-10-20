#!/usr/bin/env python3
"""Unit tests for the fixed-contract long-period Taylor artifact."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "inversion" / "disper26_inv"
package = types.ModuleType("disper26_inv")
package.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("disper26_inv", package)

from disper26_inv.long_period_taylor import (  # noqa: E402
    EXACT_REQUIRED,
    PRIMARY_FREQUENCY_HZ,
    ExactRequiredError,
    LongPeriodTaylorArtifact,
    LongPeriodTaylorProvenance,
    build_long_period_taylor_curve,
)
from disper26_inv.long_period_taylor_io import (  # noqa: E402
    compute_long_period_taylor_artifact_hash,
    load_long_period_taylor_artifact,
    save_long_period_taylor_artifact,
)


def synthetic_artifact():
    frequency = np.asarray(PRIMARY_FREQUENCY_HZ)
    prior = np.asarray([[0.04, 0.0], [0.01, 0.03]])
    curves = []
    for curve_index, (wave, root) in enumerate(
        (("rayleigh", 0), ("rayleigh", 1), ("love", 0), ("love", 1))
    ):
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
    provenance = LongPeriodTaylorProvenance(
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
    return LongPeriodTaylorArtifact(
        prior_cholesky=prior,
        shear_log_operator=np.eye(2),
        curves=tuple(curves),
        provenance=provenance,
        metadata_json=json.dumps({"purpose": "unit-test"}),
    )


def test_coordinate_transform_and_taylor_orders():
    artifact = synthetic_artifact()
    delta = np.asarray([0.025, -0.015])
    curve = artifact.get_curve("Rayleigh", 0)
    z = np.linalg.solve(artifact.prior_cholesky, delta)
    expected_linear = curve.reference_phase_velocity + curve.gradient_z @ z
    expected_quadratic = 0.5 * ((curve.hessian_z @ z) @ z)

    first = artifact.predict("Rayleigh", 0, delta, order=1)
    second = artifact.predict("Rayleigh", 0, delta, order=2)
    assert first.supported and second.supported
    assert np.array_equal(first.phase_velocity, expected_linear)
    assert np.allclose(
        second.phase_velocity,
        expected_linear + expected_quadratic,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert np.array_equal(second.linear_phase_velocity, expected_linear)
    assert np.allclose(
        second.quadratic_correction,
        expected_quadratic,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_batch_and_fail_closed_contract():
    artifact = synthetic_artifact()
    models = np.asarray([[0.01, -0.02], [0.09, 0.0], [np.nan, 0.0]])
    result = artifact.predict_batch("Love", 1, models)
    assert result.supported.tolist() == [True, False, False]
    assert result.status == ("SUPPORTED", EXACT_REQUIRED, EXACT_REQUIRED)
    assert np.all(np.isfinite(result.phase_velocity[0]))
    assert np.all(np.isnan(result.phase_velocity[1:]))
    assert result.maximum_physical_anomaly[0] < 0.08
    assert result.maximum_physical_anomaly[1] > 0.08
    assert np.isinf(result.maximum_physical_anomaly[2])

    boundary = np.asarray([np.log1p(0.08), 0.0])
    scalar_boundary = artifact.predict("Love", 1, boundary)
    batch_boundary = artifact.predict_batch("Love", 1, boundary[None, :])
    assert scalar_boundary.supported
    assert batch_boundary.supported.tolist() == [True]
    np.testing.assert_allclose(
        scalar_boundary.phase_velocity,
        batch_boundary.phase_velocity[0],
        rtol=0.0,
        atol=2.0e-15,
    )

    raw, _, _, anomaly = artifact.evaluate_unchecked_batch(
        "Love", 1, models[1:2]
    )
    assert np.all(np.isfinite(raw))
    assert anomaly[0] > 0.08

    for call in (
        lambda: artifact.predict("Love", 2, models[0]),
        lambda: artifact.predict("Love", 0.5, models[0]),
        lambda: artifact.predict("Scholte", 0, models[0]),
        lambda: artifact.predict(
            "Love", 0, models[0], frequency_hz=np.asarray([0.01, 0.02])
        ),
        lambda: artifact.predict(
            "Love", 0, models[0], reference_fingerprint="wrong"
        ),
    ):
        try:
            call()
        except ExactRequiredError as error:
            assert str(error).startswith(EXACT_REQUIRED)
        else:
            raise AssertionError("outside-contract request did not fail closed")

    try:
        artifact.predict("Love", 0, models[0], order=1.5)
    except ValueError as error:
        assert "integer 1 or 2" in str(error)
    else:
        raise AssertionError("non-integral Taylor order was accepted")


def test_deterministic_pickle_free_round_trip():
    artifact = synthetic_artifact()
    with tempfile.TemporaryDirectory() as directory:
        first_path = Path(directory) / "first.npz"
        second_path = Path(directory) / "second.npz"
        save_long_period_taylor_artifact(artifact, first_path)
        save_long_period_taylor_artifact(artifact, second_path)
        assert compute_long_period_taylor_artifact_hash(
            first_path
        ) == compute_long_period_taylor_artifact_hash(second_path)

        loaded = load_long_period_taylor_artifact(first_path)
        assert loaded.provenance == artifact.provenance
        assert loaded.metadata_json == artifact.metadata_json
        assert np.array_equal(loaded.prior_cholesky, artifact.prior_cholesky)
        for left, right in zip(loaded.curves, artifact.curves):
            assert left.key == right.key
            assert np.array_equal(left.reference_phase_velocity, right.reference_phase_velocity)
            assert np.array_equal(left.gradient_z, right.gradient_z)
            assert np.array_equal(left.hessian_z, right.hessian_z)
            assert not left.hessian_z.flags.writeable


def test_archive_schema_rejects_unknown_members():
    artifact = synthetic_artifact()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.npz"
        save_long_period_taylor_artifact(artifact, path)
        with zipfile.ZipFile(path, mode="a") as archive:
            archive.writestr("unexpected.npy", b"not-a-valid-npy")
        try:
            load_long_period_taylor_artifact(path)
        except ValueError as error:
            assert "archive keys differ" in str(error)
        else:
            raise AssertionError("unknown artifact member was accepted")


def main():
    test_coordinate_transform_and_taylor_orders()
    test_batch_and_fail_closed_contract()
    test_deterministic_pickle_free_round_trip()
    test_archive_schema_rejects_unknown_members()
    print("long-period Taylor artifact checks passed")


if __name__ == "__main__":
    main()
