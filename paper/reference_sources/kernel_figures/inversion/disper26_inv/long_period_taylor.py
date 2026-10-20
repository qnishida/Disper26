"""Fixed-contract long-period Rayleigh/Love Taylor-2 artifacts.

The online path in this module is deliberately solver-free.  It evaluates an
immutable reference jet for Rayleigh/Love root orders 0 and 1 on the ten
frequencies in :data:`PRIMARY_FREQUENCY_HZ`.  Inputs outside that contract are
masked with ``EXACT_REQUIRED``; raw extrapolation is available only through an
explicitly named offline diagnostic method.

``SUPPORTED`` means only that an input satisfies this structural/amplitude
contract and that the polynomial evaluation is finite and positive.  It is
not an accuracy estimate or certification.  The exact solver remains
authoritative; users must consult artifact-matched validation evidence before
deciding whether the local approximation is adequate for an application.

Model coordinates ``delta_m`` are log-parameter perturbations in the standard
18-parameter shallow Vsv/Vsh basis.  A stored Cholesky factor ``L`` defines the
prior-whitened coordinate ``z`` by ``delta_m = L z``.  Each curve stores

    c(z) = c0 + G_z z + 0.5 z.T H_z z.

Schema v2 keeps that response for Rayleigh and stores normalized squared
slowness y=(c0/c)**2 for Love.  Both schemas return phase velocity; the v2
Love back-transform is c=c0/sqrt(y).  The schema and per-curve response kind
prevent a v1 phase-velocity jet from being silently reinterpreted.

No frequency interpolation, root relabeling, exact fallback, or model-dependent
frequency mask occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np


LONG_PERIOD_TAYLOR_SCHEMA_NAME = "rayleigh-love-long-period-taylor-v1"
LONG_PERIOD_TAYLOR_SCHEMA_VERSION = 1
LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME = (
    "rayleigh-love-long-period-mixed-response-taylor-v2"
)
LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION = 2
PHASE_VELOCITY_RESPONSE_KIND = "phase_velocity"
NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND = "normalized_squared_slowness"
PRIMARY_FREQUENCY_HZ = (
    0.0075,
    0.0100,
    0.0125,
    0.0150,
    0.0175,
    0.0200,
    0.0225,
    0.0250,
    0.0275,
    0.0300,
)
PRIMARY_TARGETS = (
    ("rayleigh", 0),
    ("rayleigh", 1),
    ("love", 0),
    ("love", 1),
)
PRIMARY_MAXIMUM_PHYSICAL_ANOMALY = 0.08
_AMPLITUDE_TOLERANCE = 64.0 * np.finfo(float).eps

SUPPORTED_REASON_CODE = np.uint8(0)
OUTSIDE_AMPLITUDE_REASON_CODE = np.uint8(1)
NONFINITE_REASON_CODE = np.uint8(2)

SUPPORTED = "SUPPORTED"
EXACT_REQUIRED = "EXACT_REQUIRED"
OUTSIDE_AMPLITUDE_REASON = "outside_amplitude"
NONFINITE_REASON = "nonfinite"

_SUPPORTED_SCHEMAS = {
    (LONG_PERIOD_TAYLOR_SCHEMA_NAME, LONG_PERIOD_TAYLOR_SCHEMA_VERSION),
    (
        LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
        LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    ),
}
_MIXED_RESPONSE_KIND_BY_WAVE = {
    "rayleigh": PHASE_VELOCITY_RESPONSE_KIND,
    "love": NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
}


class ExactRequiredError(ValueError):
    """Raised when a requested curve is outside the immutable artifact contract."""


def _readonly(values, dtype=float) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _canonical_wave(value: str) -> str:
    wave = str(value).lower()
    if wave not in {"rayleigh", "love"}:
        raise ExactRequiredError(f"{EXACT_REQUIRED}: unsupported wave {value!r}")
    return wave


def _canonical_root_order(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ExactRequiredError(
            f"{EXACT_REQUIRED}: root order must be the integer 0 or 1"
        )
    root = int(value)
    if root not in (0, 1):
        raise ExactRequiredError(
            f"{EXACT_REQUIRED}: unsupported root order {root}"
        )
    return root


def _canonical_taylor_order(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("Taylor order must be the integer 1 or 2")
    order = int(value)
    if order not in (1, 2):
        raise ValueError("Taylor order must be 1 or 2")
    return order


def _canonical_response_kind(value: str) -> str:
    response_kind = str(value).lower()
    supported = {
        PHASE_VELOCITY_RESPONSE_KIND,
        NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
    }
    if response_kind not in supported:
        raise ValueError(f"unsupported Taylor response kind {value!r}")
    return response_kind


def _canonical_json(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("metadata_json must be a JSON string")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("metadata_json is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("metadata_json must encode an object")
    return json.dumps(
        decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


@dataclass(frozen=True)
class LongPeriodTaylorProvenance:
    """Hashes binding the artifact to its plan, code, model, and opened inputs."""

    code_commit: str
    plan_sha256: str
    reference_manifest_sha256: str
    depth_convergence_report_sha256: str
    reference_fingerprint: str
    basis_fingerprint: str
    prior_fingerprint: str
    dataset_fingerprint: str
    calibration_manifest_sha256: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class LongPeriodTaylorCurve:
    """One fixed wave/root response jet in prior-whitened coordinates."""

    wave: str
    root_order: int
    frequency_hz: np.ndarray
    reference_phase_velocity: np.ndarray
    gradient_z: np.ndarray
    hessian_z: np.ndarray
    response_kind: str = PHASE_VELOCITY_RESPONSE_KIND

    def __post_init__(self):
        wave = _canonical_wave(self.wave)
        response_kind = _canonical_response_kind(self.response_kind)
        if (
            response_kind == NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND
            and wave != "love"
        ):
            raise ValueError("normalized squared slowness is a Love-only response")
        root = int(self.root_order)
        if root != self.root_order or root not in (0, 1):
            raise ValueError("long-period Taylor stores root orders 0 and 1 only")
        frequency = np.asarray(self.frequency_hz, dtype=float)
        expected_frequency = np.asarray(PRIMARY_FREQUENCY_HZ, dtype=float)
        if not np.array_equal(frequency, expected_frequency):
            raise ValueError("curve frequency_hz differs from the fixed primary grid")
        reference = np.asarray(self.reference_phase_velocity, dtype=float)
        gradient = np.asarray(self.gradient_z, dtype=float)
        hessian = np.asarray(self.hessian_z, dtype=float)
        n_frequency = expected_frequency.size
        if reference.shape != (n_frequency,):
            raise ValueError("reference_phase_velocity must have one value per frequency")
        if gradient.ndim != 2 or gradient.shape[0] != n_frequency:
            raise ValueError("gradient_z must have shape (frequency, parameter)")
        n_params = gradient.shape[1]
        if hessian.shape != (n_frequency, n_params, n_params):
            raise ValueError("hessian_z has an inconsistent shape")
        if any(np.any(~np.isfinite(a)) for a in (reference, gradient, hessian)):
            raise ValueError("Taylor curve arrays must be finite")
        if np.any(reference <= 0.0):
            raise ValueError("reference phase velocities must be positive")
        scale = max(float(np.max(np.abs(hessian))), 1.0)
        asymmetry = float(np.max(np.abs(hessian - np.swapaxes(hessian, -1, -2))))
        if asymmetry > 1.0e-12 * scale:
            raise ValueError("hessian_z must be symmetric")
        object.__setattr__(self, "wave", wave)
        object.__setattr__(self, "response_kind", response_kind)
        object.__setattr__(self, "root_order", root)
        object.__setattr__(self, "frequency_hz", _readonly(frequency))
        object.__setattr__(self, "reference_phase_velocity", _readonly(reference))
        object.__setattr__(self, "gradient_z", _readonly(gradient))
        object.__setattr__(
            self,
            "hessian_z",
            _readonly(0.5 * (hessian + np.swapaxes(hessian, -1, -2))),
        )

    @property
    def key(self) -> tuple[str, int]:
        return (self.wave, self.root_order)

    @property
    def n_params(self) -> int:
        return int(self.gradient_z.shape[1])

    @property
    def reference_response(self) -> np.ndarray:
        if self.response_kind == PHASE_VELOCITY_RESPONSE_KIND:
            return self.reference_phase_velocity
        return _readonly(np.ones_like(self.reference_phase_velocity))

    @property
    def response_gradient_z(self) -> np.ndarray:
        return self.gradient_z

    @property
    def response_hessian_z(self) -> np.ndarray:
        return self.hessian_z

    def _response_to_phase_velocity(self, response) -> np.ndarray:
        if self.response_kind == PHASE_VELOCITY_RESPONSE_KIND:
            return np.asarray(response, dtype=float).copy()
        return self.reference_phase_velocity / np.sqrt(response)

    def evaluate_response_z_batch(self, z, *, order: int = 2):
        """Return selected, first-order, and quadratic response terms."""

        models = np.asarray(z, dtype=float)
        if models.ndim == 1:
            models = models[None, :]
        if models.ndim != 2 or models.shape[1] != self.n_params:
            raise ValueError("z must have shape (model, parameter)")
        taylor_order = _canonical_taylor_order(order)
        reference = (
            self.reference_phase_velocity[None, :]
            if self.response_kind == PHASE_VELOCITY_RESPONSE_KIND
            else 1.0
        )
        linear = reference + models @ self.gradient_z.T
        quadratic = 0.5 * np.einsum(
            "mp,fpq,mq->mf", models, self.hessian_z, models, optimize=True
        )
        value = linear if taylor_order == 1 else linear + quadratic
        return value, linear, quadratic

    def evaluate_response_z(self, z, *, order: int = 2):
        """Scalar counterpart of evaluate_response_z_batch."""

        model = np.asarray(z, dtype=float)
        if model.shape != (self.n_params,):
            raise ValueError(f"z must have shape ({self.n_params},)")
        taylor_order = _canonical_taylor_order(order)
        reference = (
            self.reference_phase_velocity
            if self.response_kind == PHASE_VELOCITY_RESPONSE_KIND
            else 1.0
        )
        linear = reference + self.gradient_z @ model
        quadratic = 0.5 * ((self.hessian_z @ model) @ model)
        value = linear if taylor_order == 1 else linear + quadratic
        return value, linear, quadratic

    def evaluate_z_batch(self, z, *, order: int = 2):
        """Return phase velocity, mapped order-1, and order-2 increment."""

        response_value, response_order1, response_quadratic = (
            self.evaluate_response_z_batch(z, order=order)
        )
        taylor_order = int(order)
        if self.response_kind == PHASE_VELOCITY_RESPONSE_KIND:
            return response_value, response_order1, response_quadratic
        response_order2 = response_order1 + response_quadratic
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            phase_order1 = self._response_to_phase_velocity(response_order1)
            phase_order2 = self._response_to_phase_velocity(response_order2)
        value = phase_order1 if taylor_order == 1 else phase_order2
        return value, phase_order1, phase_order2 - phase_order1

    def evaluate_z(self, z, *, order: int = 2):
        """Return phase velocity, mapped order-1, and order-2 increment."""

        response_value, response_order1, response_quadratic = self.evaluate_response_z(
            z, order=order
        )
        taylor_order = int(order)
        if self.response_kind == PHASE_VELOCITY_RESPONSE_KIND:
            return response_value, response_order1, response_quadratic
        response_order2 = response_order1 + response_quadratic
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            phase_order1 = self._response_to_phase_velocity(response_order1)
            phase_order2 = self._response_to_phase_velocity(response_order2)
        value = phase_order1 if taylor_order == 1 else phase_order2
        return value, phase_order1, phase_order2 - phase_order1


@dataclass(frozen=True)
class LongPeriodTaylorBatchPrediction:
    """Batch result whose order is defined in the reported response kind."""

    phase_velocity: np.ndarray
    linear_phase_velocity: np.ndarray
    quadratic_correction: np.ndarray
    supported: np.ndarray
    point_supported: np.ndarray
    reason_code: np.ndarray
    maximum_physical_anomaly: np.ndarray
    status: tuple[str, ...]
    taylor_order: int
    response_kind: str = PHASE_VELOCITY_RESPONSE_KIND

    @property
    def phase_velocity_order1(self) -> np.ndarray:
        return self.linear_phase_velocity

    @property
    def phase_velocity_order2(self) -> np.ndarray:
        return self.linear_phase_velocity + self.quadratic_correction


@dataclass(frozen=True)
class LongPeriodTaylorPrediction:
    """Scalar result whose order is defined in the reported response kind."""

    phase_velocity: np.ndarray
    linear_phase_velocity: np.ndarray
    quadratic_correction: np.ndarray
    supported: bool
    point_supported: np.ndarray
    reason_code: int
    maximum_physical_anomaly: float
    status: str
    taylor_order: int
    response_kind: str = PHASE_VELOCITY_RESPONSE_KIND

    @property
    def phase_velocity_order1(self) -> np.ndarray:
        return self.linear_phase_velocity

    @property
    def phase_velocity_order2(self) -> np.ndarray:
        return self.linear_phase_velocity + self.quadratic_correction

    @property
    def reason(self) -> str | None:
        if self.reason_code == int(SUPPORTED_REASON_CODE):
            return None
        if self.reason_code == int(OUTSIDE_AMPLITUDE_REASON_CODE):
            return OUTSIDE_AMPLITUDE_REASON
        if self.reason_code == int(NONFINITE_REASON_CODE):
            return NONFINITE_REASON
        raise ValueError("unknown long-period Taylor reason code")


@dataclass(frozen=True)
class LongPeriodTaylorArtifact:
    """Immutable four-curve local Taylor evaluator with fail-closed inputs.

    Acceptance by the input contract is deliberately separate from empirical
    accuracy assessment.  In particular, ``SUPPORTED`` must not be interpreted
    as a forward-grade error guarantee.
    """

    prior_cholesky: np.ndarray
    shear_log_operator: np.ndarray
    curves: tuple[LongPeriodTaylorCurve, ...]
    provenance: LongPeriodTaylorProvenance
    metadata_json: str = "{}"
    maximum_physical_anomaly: float = PRIMARY_MAXIMUM_PHYSICAL_ANOMALY
    schema_name: str = LONG_PERIOD_TAYLOR_SCHEMA_NAME
    schema_version: int = LONG_PERIOD_TAYLOR_SCHEMA_VERSION
    _prior_inverse: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        schema = (str(self.schema_name), int(self.schema_version))
        if schema not in _SUPPORTED_SCHEMAS:
            raise ValueError("unsupported long-period Taylor artifact schema")
        prior = np.asarray(self.prior_cholesky, dtype=float)
        if (
            prior.ndim != 2
            or prior.shape[0] != prior.shape[1]
            or prior.shape[0] < 1
            or np.any(~np.isfinite(prior))
            or np.any(np.diag(prior) <= 0.0)
            or not np.allclose(prior, np.tril(prior), rtol=0.0, atol=0.0)
        ):
            raise ValueError("prior_cholesky must be finite, lower triangular, and positive")
        shear = np.asarray(self.shear_log_operator, dtype=float)
        if (
            shear.ndim != 2
            or shear.shape[0] < 1
            or shear.shape[1] != prior.shape[0]
            or np.any(~np.isfinite(shear))
        ):
            raise ValueError("shear_log_operator has an inconsistent shape")
        curves = tuple(self.curves)
        if tuple(curve.key for curve in curves) != PRIMARY_TARGETS:
            raise ValueError("artifact must contain the four primary curves in fixed order")
        if any(curve.n_params != prior.shape[0] for curve in curves):
            raise ValueError("all curves must use the artifact parameter dimension")
        if schema == (
            LONG_PERIOD_TAYLOR_SCHEMA_NAME,
            LONG_PERIOD_TAYLOR_SCHEMA_VERSION,
        ):
            expected_response_kinds = (
                PHASE_VELOCITY_RESPONSE_KIND,
            ) * len(curves)
        else:
            expected_response_kinds = tuple(
                _MIXED_RESPONSE_KIND_BY_WAVE[curve.wave] for curve in curves
            )
        actual_response_kinds = tuple(curve.response_kind for curve in curves)
        if actual_response_kinds != expected_response_kinds:
            raise ValueError(
                "curve response kinds do not match the artifact schema"
            )
        maximum = float(self.maximum_physical_anomaly)
        if not np.isclose(
            maximum,
            PRIMARY_MAXIMUM_PHYSICAL_ANOMALY,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("long-period physical anomaly limit is fixed at 8 percent")
        object.__setattr__(self, "prior_cholesky", _readonly(prior))
        object.__setattr__(self, "shear_log_operator", _readonly(shear))
        object.__setattr__(self, "curves", curves)
        object.__setattr__(self, "schema_name", schema[0])
        object.__setattr__(self, "schema_version", schema[1])
        object.__setattr__(self, "metadata_json", _canonical_json(self.metadata_json))
        object.__setattr__(self, "maximum_physical_anomaly", maximum)
        object.__setattr__(self, "_prior_inverse", _readonly(np.linalg.inv(prior)))

    @property
    def n_params(self) -> int:
        return int(self.prior_cholesky.shape[0])

    @property
    def frequency_hz(self) -> np.ndarray:
        return _readonly(PRIMARY_FREQUENCY_HZ)

    def get_curve(self, wave: str, root_order: int) -> LongPeriodTaylorCurve:
        key = (_canonical_wave(wave), _canonical_root_order(root_order))
        for curve in self.curves:
            if curve.key == key:
                return curve
        raise ExactRequiredError(f"{EXACT_REQUIRED}: unsupported curve {key!r}")

    def assert_compatible(
        self,
        *,
        reference_fingerprint: str | None = None,
        basis_fingerprint: str | None = None,
        prior_fingerprint: str | None = None,
    ) -> None:
        """Reject an explicitly supplied model/basis/prior identity mismatch."""

        if (
            reference_fingerprint is None
            and basis_fingerprint is None
            and prior_fingerprint is None
        ):
            return

        supplied = {
            "reference": (
                reference_fingerprint,
                self.provenance.reference_fingerprint,
            ),
            "basis": (basis_fingerprint, self.provenance.basis_fingerprint),
            "prior": (prior_fingerprint, self.provenance.prior_fingerprint),
        }
        for label, (actual, expected) in supplied.items():
            if actual is not None and str(actual) != expected:
                raise ExactRequiredError(
                    f"{EXACT_REQUIRED}: {label} fingerprint differs from the artifact"
                )

    def _validate_frequency(self, frequency_hz) -> None:
        if frequency_hz is None:
            return
        requested = np.asarray(frequency_hz, dtype=float)
        if not np.array_equal(requested, np.asarray(PRIMARY_FREQUENCY_HZ)):
            raise ExactRequiredError(
                f"{EXACT_REQUIRED}: frequency grid differs from the ten-point primary grid"
            )

    def _coordinates_and_anomaly(self, delta_m):
        models = np.asarray(delta_m, dtype=float)
        scalar = models.ndim == 1
        if scalar:
            models = models[None, :]
        if models.ndim != 2 or models.shape[1] != self.n_params:
            raise ValueError("delta_m must have shape (model, parameter)")
        finite = np.isfinite(models).all(axis=1)
        safe = models if np.all(finite) else np.where(np.isfinite(models), models, 0.0)
        z = safe @ self._prior_inverse.T
        with np.errstate(over="ignore", invalid="ignore"):
            shear = np.expm1(safe @ self.shear_log_operator.T)
        anomaly = np.abs(shear).max(axis=1)
        finite &= np.isfinite(shear).all(axis=1)
        anomaly[~finite] = np.inf
        z[~finite] = np.nan
        return models, z, anomaly, finite, scalar

    def evaluate_unchecked_batch(
        self, wave: str, root_order: int, delta_m, *, order: int = 2
    ):
        """Offline-only raw Taylor evaluation, including stress extrapolation.

        This method intentionally performs no 8 percent support decision and
        must not be used as the online forward contract.
        """

        curve = self.get_curve(wave, root_order)
        _, z, anomaly, finite, _ = self._coordinates_and_anomaly(delta_m)
        safe_z = np.where(np.isfinite(z), z, 0.0)
        value, linear, quadratic = curve.evaluate_z_batch(safe_z, order=order)
        value[~finite] = np.nan
        linear[~finite] = np.nan
        quadratic[~finite] = np.nan
        return value, linear, quadratic, anomaly

    def evaluate_response_unchecked_batch(
        self, wave: str, root_order: int, delta_m, *, order: int = 2
    ):
        """Offline response-space evaluation for validation diagnostics."""

        curve = self.get_curve(wave, root_order)
        _, z, anomaly, finite, _ = self._coordinates_and_anomaly(delta_m)
        safe_z = np.where(np.isfinite(z), z, 0.0)
        value, linear, quadratic = curve.evaluate_response_z_batch(
            safe_z, order=order
        )
        value[~finite] = np.nan
        linear[~finite] = np.nan
        quadratic[~finite] = np.nan
        return value, linear, quadratic, anomaly

    def predict_batch(
        self,
        wave: str,
        root_order: int,
        delta_m,
        *,
        order: int = 2,
        frequency_hz=None,
        reference_fingerprint: str | None = None,
        basis_fingerprint: str | None = None,
        prior_fingerprint: str | None = None,
    ) -> LongPeriodTaylorBatchPrediction:
        """Evaluate accepted inputs without estimating Taylor truncation error.

        ``supported`` and ``point_supported`` report only input-contract and
        finite-output validity.  They do not certify numerical accuracy.
        """

        self._validate_frequency(frequency_hz)
        self.assert_compatible(
            reference_fingerprint=reference_fingerprint,
            basis_fingerprint=basis_fingerprint,
            prior_fingerprint=prior_fingerprint,
        )
        curve = self.get_curve(wave, root_order)
        _, z, anomaly, finite, _ = self._coordinates_and_anomaly(delta_m)
        safe_z = np.where(np.isfinite(z), z, 0.0)
        value, linear, quadratic = curve.evaluate_z_batch(safe_z, order=order)
        outside = anomaly > self.maximum_physical_anomaly + _AMPLITUDE_TOLERANCE
        reason = np.full(anomaly.shape, SUPPORTED_REASON_CODE, dtype=np.uint8)
        reason[outside] = OUTSIDE_AMPLITUDE_REASON_CODE
        reason[~finite] = NONFINITE_REASON_CODE
        supported = reason == SUPPORTED_REASON_CODE
        prediction_finite = (
            np.isfinite(value).all(axis=1)
            & np.isfinite(linear).all(axis=1)
            & np.isfinite(quadratic).all(axis=1)
            & (value > 0.0).all(axis=1)
        )
        reason[~prediction_finite] = NONFINITE_REASON_CODE
        supported = reason == SUPPORTED_REASON_CODE
        masked_value = value.copy()
        masked_linear = linear.copy()
        masked_quadratic = quadratic.copy()
        masked_value[~supported] = np.nan
        masked_linear[~supported] = np.nan
        masked_quadratic[~supported] = np.nan
        point_supported = np.broadcast_to(
            supported[:, None], masked_value.shape
        ).copy()
        status = tuple(SUPPORTED if flag else EXACT_REQUIRED for flag in supported)
        return LongPeriodTaylorBatchPrediction(
            phase_velocity=masked_value,
            linear_phase_velocity=masked_linear,
            quadratic_correction=masked_quadratic,
            supported=supported.copy(),
            point_supported=point_supported,
            reason_code=reason.copy(),
            maximum_physical_anomaly=anomaly.copy(),
            status=status,
            taylor_order=int(order),
            response_kind=curve.response_kind,
        )

    def predict(
        self,
        wave: str,
        root_order: int,
        delta_m,
        *,
        order: int = 2,
        frequency_hz=None,
        reference_fingerprint: str | None = None,
        basis_fingerprint: str | None = None,
        prior_fingerprint: str | None = None,
    ) -> LongPeriodTaylorPrediction:
        """Evaluate one accepted input without an approximation-error bound."""

        self._validate_frequency(frequency_hz)
        self.assert_compatible(
            reference_fingerprint=reference_fingerprint,
            basis_fingerprint=basis_fingerprint,
            prior_fingerprint=prior_fingerprint,
        )
        order = _canonical_taylor_order(order)
        curve = self.get_curve(wave, root_order)
        model = np.asarray(delta_m, dtype=float)
        if model.shape != (self.n_params,):
            raise ValueError(f"delta_m must have shape ({self.n_params},)")
        finite = bool(np.isfinite(model).all())
        if finite:
            with np.errstate(over="ignore", invalid="ignore"):
                shear = np.expm1(model @ self.shear_log_operator.T)
            finite = bool(np.isfinite(shear).all())
            anomaly = float(np.abs(shear).max()) if finite else float("inf")
        else:
            anomaly = float("inf")
        if not finite:
            reason_code = int(NONFINITE_REASON_CODE)
        elif anomaly > self.maximum_physical_anomaly + _AMPLITUDE_TOLERANCE:
            reason_code = int(OUTSIDE_AMPLITUDE_REASON_CODE)
        else:
            z = model @ self._prior_inverse.T
            value, linear, quadratic = curve.evaluate_z(z, order=order)
            prediction_finite = bool(
                np.isfinite(value).all()
                and np.isfinite(linear).all()
                and np.isfinite(quadratic).all()
                and (value > 0.0).all()
            )
            reason_code = (
                int(SUPPORTED_REASON_CODE)
                if prediction_finite
                else int(NONFINITE_REASON_CODE)
            )
        supported = reason_code == int(SUPPORTED_REASON_CODE)
        if not supported:
            value = np.full(curve.frequency_hz.shape, np.nan)
            linear = value.copy()
            quadratic = value.copy()
        return LongPeriodTaylorPrediction(
            phase_velocity=value,
            linear_phase_velocity=linear,
            quadratic_correction=quadratic,
            supported=supported,
            point_supported=np.full(curve.frequency_hz.shape, supported, dtype=bool),
            reason_code=reason_code,
            maximum_physical_anomaly=anomaly,
            status=SUPPORTED if supported else EXACT_REQUIRED,
            taylor_order=order,
            response_kind=curve.response_kind,
        )


def build_long_period_taylor_curve(
    *,
    wave: str,
    root_order: int,
    frequency_hz,
    reference_phase_velocity,
    phase_gradient_m,
    phase_hessian_m,
    prior_cholesky,
    response_kind: str = PHASE_VELOCITY_RESPONSE_KIND,
) -> LongPeriodTaylorCurve:
    """Transform a phase-velocity jet into whitened response coordinates."""

    wave_name = _canonical_wave(wave)
    response_name = _canonical_response_kind(response_kind)
    reference = np.asarray(reference_phase_velocity, dtype=float)
    gradient_m = np.asarray(phase_gradient_m, dtype=float)
    hessian_m = np.asarray(phase_hessian_m, dtype=float)
    cholesky = np.asarray(prior_cholesky, dtype=float)
    if gradient_m.ndim != 2 or gradient_m.shape[1] != cholesky.shape[0]:
        raise ValueError("phase_gradient_m and prior_cholesky do not align")
    if hessian_m.shape != (
        gradient_m.shape[0],
        cholesky.shape[0],
        cholesky.shape[0],
    ):
        raise ValueError("phase_hessian_m has an inconsistent shape")
    if (
        reference.shape != (gradient_m.shape[0],)
        or np.any(~np.isfinite(reference))
        or np.any(reference <= 0.0)
    ):
        raise ValueError(
            "reference_phase_velocity must be finite, positive, and align with frequency"
        )
    gradient_z = gradient_m @ cholesky
    hessian_z = np.einsum(
        "ai,fab,bj->fij", cholesky, hessian_m, cholesky, optimize=True
    )
    hessian_z = 0.5 * (hessian_z + np.swapaxes(hessian_z, -1, -2))
    if response_name == NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND:
        if wave_name != "love":
            raise ValueError("normalized squared slowness is a Love-only response")
        response_gradient_z = -2.0 * gradient_z / reference[:, None]
        response_hessian_z = (
            6.0
            * gradient_z[:, :, None]
            * gradient_z[:, None, :]
            / np.square(reference)[:, None, None]
            - 2.0 * hessian_z / reference[:, None, None]
        )
    else:
        response_gradient_z = gradient_z
        response_hessian_z = hessian_z
    return LongPeriodTaylorCurve(
        wave=wave_name,
        root_order=root_order,
        frequency_hz=frequency_hz,
        reference_phase_velocity=reference,
        gradient_z=response_gradient_z,
        hessian_z=0.5
        * (response_hessian_z + np.swapaxes(response_hessian_z, -1, -2)),
        response_kind=response_name,
    )


__all__ = [
    "EXACT_REQUIRED",
    "LONG_PERIOD_TAYLOR_SCHEMA_NAME",
    "LONG_PERIOD_TAYLOR_SCHEMA_VERSION",
    "LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME",
    "LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION",
    "NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND",
    "NONFINITE_REASON",
    "NONFINITE_REASON_CODE",
    "OUTSIDE_AMPLITUDE_REASON",
    "OUTSIDE_AMPLITUDE_REASON_CODE",
    "PRIMARY_FREQUENCY_HZ",
    "PRIMARY_MAXIMUM_PHYSICAL_ANOMALY",
    "PRIMARY_TARGETS",
    "PHASE_VELOCITY_RESPONSE_KIND",
    "SUPPORTED",
    "SUPPORTED_REASON_CODE",
    "ExactRequiredError",
    "LongPeriodTaylorArtifact",
    "LongPeriodTaylorBatchPrediction",
    "LongPeriodTaylorCurve",
    "LongPeriodTaylorPrediction",
    "LongPeriodTaylorProvenance",
    "build_long_period_taylor_curve",
]
