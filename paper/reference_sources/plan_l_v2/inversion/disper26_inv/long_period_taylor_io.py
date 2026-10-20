"""Deterministic, pickle-free, atomic I/O for long-period Taylor artifacts."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .long_period_taylor import (
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    LONG_PERIOD_TAYLOR_SCHEMA_NAME,
    LONG_PERIOD_TAYLOR_SCHEMA_VERSION,
    LongPeriodTaylorArtifact,
    LongPeriodTaylorCurve,
    LongPeriodTaylorProvenance,
)


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def canonical_json(value) -> str:
    """Return the byte-stable JSON representation used by all manifests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_fingerprint(*arrays) -> str:
    """Hash dtype, shape, and C-order bytes for a sequence of arrays."""

    digest = hashlib.sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        if array.dtype.hasobject:
            raise ValueError("object arrays are forbidden in deterministic artifacts")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _npy_bytes(values) -> bytes:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("object arrays are forbidden in deterministic archives")
    output = BytesIO()
    np.lib.format.write_array(output, array, allow_pickle=False)
    return output.getvalue()


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def save_deterministic_npz_arrays(arrays: dict[str, np.ndarray], path) -> None:
    """Write sorted ``.npy`` members with fixed zip metadata, then replace atomically."""

    if not arrays:
        raise ValueError("deterministic archive requires at least one array")
    output = BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ValueError("archive array names must be simple non-empty strings")
            member = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            member.create_system = 3
            archive.writestr(member, _npy_bytes(arrays[name]), compresslevel=9)
    _atomic_replace_bytes(Path(path), output.getvalue())


def write_json_atomic(path, value) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    _atomic_replace_bytes(Path(path), payload)


def _artifact_arrays(artifact: LongPeriodTaylorArtifact):
    if (
        artifact.schema_name != LONG_PERIOD_TAYLOR_SCHEMA_NAME
        or artifact.schema_version != LONG_PERIOD_TAYLOR_SCHEMA_VERSION
    ):
        raise ValueError("schema-v1 writer requires a phase-velocity v1 artifact")
    metadata = {
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
        "maximum_physical_anomaly": artifact.maximum_physical_anomaly,
        "provenance": asdict(artifact.provenance),
        "artifact_metadata": json.loads(artifact.metadata_json),
        "curves": [
            {"wave": curve.wave, "root_order": curve.root_order}
            for curve in artifact.curves
        ],
    }
    arrays = {
        "metadata_json_utf8": np.frombuffer(
            canonical_json(metadata).encode("utf-8"), dtype=np.uint8
        ).copy(),
        "prior_cholesky": artifact.prior_cholesky,
        "shear_log_operator": artifact.shear_log_operator,
    }
    for index, curve in enumerate(artifact.curves):
        prefix = f"curve_{index:02d}"
        arrays[f"{prefix}_frequency_hz"] = curve.frequency_hz
        arrays[f"{prefix}_reference_phase_velocity"] = (
            curve.reference_phase_velocity
        )
        arrays[f"{prefix}_gradient_z"] = curve.gradient_z
        arrays[f"{prefix}_hessian_z"] = curve.hessian_z
    return arrays


def save_long_period_taylor_artifact(
    artifact: LongPeriodTaylorArtifact, path
) -> None:
    save_deterministic_npz_arrays(_artifact_arrays(artifact), path)


def load_long_period_taylor_artifact(path) -> LongPeriodTaylorArtifact:
    with np.load(Path(path), allow_pickle=False) as archive:
        if "metadata_json_utf8" not in archive.files:
            raise ValueError("artifact metadata is absent")
        metadata_bytes = np.asarray(archive["metadata_json_utf8"], dtype=np.uint8)
        if metadata_bytes.ndim != 1:
            raise ValueError("artifact metadata byte array must be one-dimensional")
        try:
            metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("artifact metadata is malformed") from error
        if (
            metadata.get("schema_name") != LONG_PERIOD_TAYLOR_SCHEMA_NAME
            or int(metadata.get("schema_version", -1))
            != LONG_PERIOD_TAYLOR_SCHEMA_VERSION
        ):
            raise ValueError("unsupported long-period Taylor artifact schema")
        expected_metadata_keys = {
            "schema_name",
            "schema_version",
            "maximum_physical_anomaly",
            "provenance",
            "artifact_metadata",
            "curves",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
            raise ValueError("artifact scalar metadata keys differ from schema v1")
        expected = {
            "metadata_json_utf8",
            "prior_cholesky",
            "shear_log_operator",
        }
        curves = []
        curve_metadata = metadata.get("curves")
        if not isinstance(curve_metadata, list):
            raise ValueError("artifact curve metadata must be a list")
        for index, values in enumerate(curve_metadata):
            if not isinstance(values, dict) or set(values) != {
                "wave",
                "root_order",
            }:
                raise ValueError("artifact curve metadata keys differ from schema v1")
            prefix = f"curve_{index:02d}"
            keys = {
                "frequency_hz": f"{prefix}_frequency_hz",
                "reference_phase_velocity": (
                    f"{prefix}_reference_phase_velocity"
                ),
                "gradient_z": f"{prefix}_gradient_z",
                "hessian_z": f"{prefix}_hessian_z",
            }
            expected.update(keys.values())
            curves.append(
                LongPeriodTaylorCurve(
                    wave=values["wave"],
                    root_order=values["root_order"],
                    frequency_hz=archive[keys["frequency_hz"]],
                    reference_phase_velocity=archive[
                        keys["reference_phase_velocity"]
                    ],
                    gradient_z=archive[keys["gradient_z"]],
                    hessian_z=archive[keys["hessian_z"]],
                )
            )
        actual = set(archive.files)
        if actual != expected:
            raise ValueError(
                "artifact archive keys differ: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        try:
            provenance = LongPeriodTaylorProvenance(**metadata["provenance"])
            artifact_metadata = canonical_json(metadata["artifact_metadata"])
            maximum = float(metadata["maximum_physical_anomaly"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("artifact scalar metadata is malformed") from error
        return LongPeriodTaylorArtifact(
            prior_cholesky=archive["prior_cholesky"],
            shear_log_operator=archive["shear_log_operator"],
            curves=tuple(curves),
            provenance=provenance,
            metadata_json=artifact_metadata,
            maximum_physical_anomaly=maximum,
            schema_name=metadata["schema_name"],
            schema_version=metadata["schema_version"],
        )


def compute_long_period_taylor_artifact_hash(path) -> str:
    return file_sha256(path)


def _mixed_response_artifact_arrays(artifact: LongPeriodTaylorArtifact):
    if (
        artifact.schema_name != LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME
        or artifact.schema_version != LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION
    ):
        raise ValueError("mixed-response writer requires a schema-v2 artifact")
    metadata = {
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
        "maximum_physical_anomaly": artifact.maximum_physical_anomaly,
        "provenance": asdict(artifact.provenance),
        "artifact_metadata": json.loads(artifact.metadata_json),
        "curves": [
            {
                "wave": curve.wave,
                "root_order": curve.root_order,
                "response_kind": curve.response_kind,
            }
            for curve in artifact.curves
        ],
    }
    arrays = {
        "metadata_json_utf8": np.frombuffer(
            canonical_json(metadata).encode("utf-8"), dtype=np.uint8
        ).copy(),
        "prior_cholesky": artifact.prior_cholesky,
        "shear_log_operator": artifact.shear_log_operator,
    }
    for index, curve in enumerate(artifact.curves):
        prefix = f"curve_{index:02d}"
        arrays[f"{prefix}_frequency_hz"] = curve.frequency_hz
        arrays[f"{prefix}_reference_phase_velocity"] = (
            curve.reference_phase_velocity
        )
        arrays[f"{prefix}_reference_response"] = curve.reference_response
        arrays[f"{prefix}_response_gradient_z"] = curve.response_gradient_z
        arrays[f"{prefix}_response_hessian_z"] = curve.response_hessian_z
    return arrays


def save_long_period_mixed_response_artifact(
    artifact: LongPeriodTaylorArtifact, path
) -> None:
    """Save a deterministic schema-v2 mixed-response artifact."""

    save_deterministic_npz_arrays(_mixed_response_artifact_arrays(artifact), path)


def load_long_period_mixed_response_artifact(path) -> LongPeriodTaylorArtifact:
    """Load only schema-v2 artifacts; schema v1 is intentionally rejected."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if "metadata_json_utf8" not in archive.files:
            raise ValueError("artifact metadata is absent")
        metadata_bytes = np.asarray(archive["metadata_json_utf8"], dtype=np.uint8)
        if metadata_bytes.ndim != 1:
            raise ValueError("artifact metadata byte array must be one-dimensional")
        try:
            metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("artifact metadata is malformed") from error
        if (
            metadata.get("schema_name") != LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME
            or int(metadata.get("schema_version", -1))
            != LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported long-period mixed-response artifact schema")
        expected_metadata_keys = {
            "schema_name",
            "schema_version",
            "maximum_physical_anomaly",
            "provenance",
            "artifact_metadata",
            "curves",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
            raise ValueError("artifact scalar metadata keys differ from schema v2")
        expected = {
            "metadata_json_utf8",
            "prior_cholesky",
            "shear_log_operator",
        }
        curves = []
        curve_metadata = metadata.get("curves")
        if not isinstance(curve_metadata, list):
            raise ValueError("artifact curve metadata must be a list")
        for index, values in enumerate(curve_metadata):
            if not isinstance(values, dict) or set(values) != {
                "wave",
                "root_order",
                "response_kind",
            }:
                raise ValueError("artifact curve metadata keys differ from schema v2")
            prefix = f"curve_{index:02d}"
            keys = {
                "frequency_hz": f"{prefix}_frequency_hz",
                "reference_phase_velocity": (
                    f"{prefix}_reference_phase_velocity"
                ),
                "reference_response": f"{prefix}_reference_response",
                "response_gradient_z": f"{prefix}_response_gradient_z",
                "response_hessian_z": f"{prefix}_response_hessian_z",
            }
            expected.update(keys.values())
            curve = LongPeriodTaylorCurve(
                wave=values["wave"],
                root_order=values["root_order"],
                frequency_hz=archive[keys["frequency_hz"]],
                reference_phase_velocity=archive[
                    keys["reference_phase_velocity"]
                ],
                gradient_z=archive[keys["response_gradient_z"]],
                hessian_z=archive[keys["response_hessian_z"]],
                response_kind=values["response_kind"],
            )
            if not np.array_equal(
                np.asarray(archive[keys["reference_response"]]),
                curve.reference_response,
            ):
                raise ValueError(
                    "stored reference response disagrees with response kind"
                )
            curves.append(curve)
        actual = set(archive.files)
        if actual != expected:
            raise ValueError(
                "artifact archive keys differ: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        try:
            provenance = LongPeriodTaylorProvenance(**metadata["provenance"])
            artifact_metadata = canonical_json(metadata["artifact_metadata"])
            maximum = float(metadata["maximum_physical_anomaly"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("artifact scalar metadata is malformed") from error
        return LongPeriodTaylorArtifact(
            prior_cholesky=archive["prior_cholesky"],
            shear_log_operator=archive["shear_log_operator"],
            curves=tuple(curves),
            provenance=provenance,
            metadata_json=artifact_metadata,
            maximum_physical_anomaly=maximum,
            schema_name=metadata["schema_name"],
            schema_version=metadata["schema_version"],
        )


def compute_long_period_mixed_response_artifact_hash(path) -> str:
    return file_sha256(path)


__all__ = [
    "array_fingerprint",
    "canonical_json",
    "compute_long_period_taylor_artifact_hash",
    "compute_long_period_mixed_response_artifact_hash",
    "file_sha256",
    "json_sha256",
    "load_long_period_taylor_artifact",
    "load_long_period_mixed_response_artifact",
    "save_deterministic_npz_arrays",
    "save_long_period_taylor_artifact",
    "save_long_period_mixed_response_artifact",
    "write_json_atomic",
]
