"""Immutable Plan-L domain, split, prior, and gate definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .long_period_taylor import (
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    LONG_PERIOD_TAYLOR_SCHEMA_NAME,
    LONG_PERIOD_TAYLOR_SCHEMA_VERSION,
    NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
    PHASE_VELOCITY_RESPONSE_KIND,
    PRIMARY_FREQUENCY_HZ,
    PRIMARY_MAXIMUM_PHYSICAL_ANOMALY,
    PRIMARY_TARGETS,
)
from .long_period_taylor_io import array_fingerprint, json_sha256
from .model import VELOCITY_FIELDS
from .physics import check_stiffness_posdef


CALIBRATION_SEED = 20270507
DEVELOPMENT_SEED = 20270508
SEALED_HOLDOUT_SEED = 20270509
MIXED_RESPONSE_CALIBRATION_SEED = CALIBRATION_SEED
MIXED_RESPONSE_DEVELOPMENT_SEED = 20270510
MIXED_RESPONSE_SEALED_HOLDOUT_SEED = 20270511
AMPLITUDE_LEVELS = (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10)
PRIMARY_AMPLITUDE_LEVELS = (0.005, 0.01, 0.02, 0.04, 0.06, 0.08)
ORDER_FIT_AMPLITUDE_LEVELS = (0.01, 0.02, 0.04, 0.06)
REFERENCE_DEPTHS_KM = (2200.0, 2400.0, 2600.0)
ARTIFACT_REFERENCE_DEPTH_KM = 2400.0
REFERENCE_MAX_SPACING_KM = 15.0
INVERSION_MAXIMUM_DEPTH_KM = 60.0
DEFAULT_DIRECTION_COUNT = 32


@dataclass(frozen=True)
class LongPeriodTaylorDatasetManifest:
    """The exact, non-adaptive runtime scope for Plan L v1."""

    schema_name: str = LONG_PERIOD_TAYLOR_SCHEMA_NAME
    schema_version: int = LONG_PERIOD_TAYLOR_SCHEMA_VERSION
    medium: str = "ocean_free_sediment_free_solid_vti"
    reference_model: str = "prem_noocean_flat"
    reference_bottom_depth_km: float = ARTIFACT_REFERENCE_DEPTH_KM
    reference_max_spacing_km: float = REFERENCE_MAX_SPACING_KM
    inversion_maximum_depth_km: float = INVERSION_MAXIMUM_DEPTH_KM
    inversion_node_depths_km: tuple[float, ...] = (
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    )
    parameter_names: tuple[str, ...] = ("vsv", "vsh")
    parameter_count: int = 18
    waves: tuple[str, ...] = ("rayleigh", "love")
    target_root_orders: tuple[int, ...] = (0, 1)
    diagnostic_root_orders: tuple[int, ...] = (0, 1, 2)
    frequency_hz: tuple[float, ...] = PRIMARY_FREQUENCY_HZ
    fixed_moho: bool = True
    maximum_physical_shear_anomaly: float = (
        PRIMARY_MAXIMUM_PHYSICAL_ANOMALY
    )
    frequency_interpolation: bool = False

    def __post_init__(self):
        if self.schema_name != LONG_PERIOD_TAYLOR_SCHEMA_NAME:
            raise ValueError("dataset schema_name is fixed")
        if int(self.schema_version) != LONG_PERIOD_TAYLOR_SCHEMA_VERSION:
            raise ValueError("dataset schema_version is fixed")
        if tuple(self.frequency_hz) != PRIMARY_FREQUENCY_HZ:
            raise ValueError("Plan L v1 frequency grid is fixed")
        if tuple(self.waves) != ("rayleigh", "love"):
            raise ValueError("Plan L v1 requires Rayleigh and Love")
        if tuple(self.target_root_orders) != (0, 1):
            raise ValueError("Plan L v1 targets root orders 0 and 1")
        if tuple(self.diagnostic_root_orders) != (0, 1, 2):
            raise ValueError("Plan L v1 diagnoses root orders 0 through 2")
        if tuple(self.parameter_names) != ("vsv", "vsh"):
            raise ValueError("Plan L v1 uses independent Vsv/Vsh parameters")
        if int(self.parameter_count) != 18:
            raise ValueError("Plan L v1 uses 18 model parameters")
        if not self.fixed_moho or self.frequency_interpolation:
            raise ValueError("Plan L v1 requires fixed Moho and no frequency interpolation")
        fixed_scalars = (
            (self.reference_bottom_depth_km, ARTIFACT_REFERENCE_DEPTH_KM),
            (self.reference_max_spacing_km, REFERENCE_MAX_SPACING_KM),
            (self.inversion_maximum_depth_km, INVERSION_MAXIMUM_DEPTH_KM),
            (
                self.maximum_physical_shear_anomaly,
                PRIMARY_MAXIMUM_PHYSICAL_ANOMALY,
            ),
        )
        if any(
            not np.isclose(actual, expected, rtol=0.0, atol=0.0)
            for actual, expected in fixed_scalars
        ):
            raise ValueError("Plan L v1 reference/domain scalars are fixed")

    @property
    def target_keys(self) -> tuple[tuple[str, int], ...]:
        return PRIMARY_TARGETS


@dataclass(frozen=True)
class LongPeriodMixedResponseDatasetManifest:
    """The exact, non-adaptive runtime scope for Plan L v2."""

    schema_name: str = LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME
    schema_version: int = LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION
    medium: str = "ocean_free_sediment_free_solid_vti"
    reference_model: str = "prem_noocean_flat"
    reference_bottom_depth_km: float = ARTIFACT_REFERENCE_DEPTH_KM
    reference_max_spacing_km: float = REFERENCE_MAX_SPACING_KM
    inversion_maximum_depth_km: float = INVERSION_MAXIMUM_DEPTH_KM
    inversion_node_depths_km: tuple[float, ...] = (
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    )
    parameter_names: tuple[str, ...] = ("vsv", "vsh")
    parameter_count: int = 18
    waves: tuple[str, ...] = ("rayleigh", "love")
    target_root_orders: tuple[int, ...] = (0, 1)
    diagnostic_root_orders: tuple[int, ...] = (0, 1, 2)
    frequency_hz: tuple[float, ...] = PRIMARY_FREQUENCY_HZ
    response_kind_by_wave: tuple[tuple[str, str], ...] = (
        ("rayleigh", PHASE_VELOCITY_RESPONSE_KIND),
        ("love", NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND),
    )
    fixed_moho: bool = True
    maximum_physical_shear_anomaly: float = (
        PRIMARY_MAXIMUM_PHYSICAL_ANOMALY
    )
    frequency_interpolation: bool = False

    def __post_init__(self):
        if self.schema_name != LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME:
            raise ValueError("mixed-response dataset schema_name is fixed")
        if int(self.schema_version) != LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION:
            raise ValueError("mixed-response dataset schema_version is fixed")
        if tuple(self.frequency_hz) != PRIMARY_FREQUENCY_HZ:
            raise ValueError("Plan L v2 frequency grid is fixed")
        if tuple(self.waves) != ("rayleigh", "love"):
            raise ValueError("Plan L v2 requires Rayleigh and Love")
        if tuple(self.target_root_orders) != (0, 1):
            raise ValueError("Plan L v2 targets root orders 0 and 1")
        if tuple(self.diagnostic_root_orders) != (0, 1, 2):
            raise ValueError("Plan L v2 diagnoses root orders 0 through 2")
        if tuple(self.parameter_names) != ("vsv", "vsh"):
            raise ValueError("Plan L v2 uses independent Vsv/Vsh parameters")
        if int(self.parameter_count) != 18:
            raise ValueError("Plan L v2 uses 18 model parameters")
        if tuple(self.response_kind_by_wave) != (
            ("rayleigh", PHASE_VELOCITY_RESPONSE_KIND),
            ("love", NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND),
        ):
            raise ValueError("Plan L v2 response kinds are fixed by wave")
        if not self.fixed_moho or self.frequency_interpolation:
            raise ValueError("Plan L v2 requires fixed Moho and no interpolation")
        fixed_scalars = (
            (self.reference_bottom_depth_km, ARTIFACT_REFERENCE_DEPTH_KM),
            (self.reference_max_spacing_km, REFERENCE_MAX_SPACING_KM),
            (self.inversion_maximum_depth_km, INVERSION_MAXIMUM_DEPTH_KM),
            (
                self.maximum_physical_shear_anomaly,
                PRIMARY_MAXIMUM_PHYSICAL_ANOMALY,
            ),
        )
        if any(
            not np.isclose(actual, expected, rtol=0.0, atol=0.0)
            for actual, expected in fixed_scalars
        ):
            raise ValueError("Plan L v2 reference/domain scalars are fixed")

    @property
    def target_keys(self) -> tuple[tuple[str, int], ...]:
        return PRIMARY_TARGETS

    @property
    def response_kinds(self) -> dict[str, str]:
        return dict(self.response_kind_by_wave)


@dataclass(frozen=True)
class LongPeriodTaylorPriorManifest:
    """The prior coordinate used for the opened evidence and Plan-L jets."""

    vsv_sigma: float = 0.035
    vsh_sigma: float = 0.035
    correlation_length_logdepth: float = 0.4
    depth_kernel: str = "gaussian"
    parameter_correlation: str = "birch_tight_aniso"
    z0_km: float = 30.0
    jitter: float = 1.0e-10

    def __post_init__(self):
        if not np.isclose(self.vsv_sigma, 0.035) or not np.isclose(
            self.vsh_sigma, 0.035
        ):
            raise ValueError("Plan L v1 Vsv/Vsh prior sigma is fixed at 0.035")
        if not np.isclose(self.correlation_length_logdepth, 0.4):
            raise ValueError("Plan L v1 log-depth correlation length is fixed")
        if self.depth_kernel != "gaussian":
            raise ValueError("Plan L v1 depth kernel is gaussian")
        if self.parameter_correlation != "birch_tight_aniso":
            raise ValueError("Plan L v1 parameter-correlation preset is fixed")
        if not np.isclose(self.z0_km, 30.0) or not np.isclose(
            self.jitter, 1.0e-10
        ):
            raise ValueError("Plan L v1 prior z0/jitter are fixed")

    def prior_kwargs(self) -> dict:
        return {
            "sigma": {"vsv": self.vsv_sigma, "vsh": self.vsh_sigma},
            "corr_length_logdepth": self.correlation_length_logdepth,
            "depth_kernel": self.depth_kernel,
            "param_corr": self.parameter_correlation,
            "z0_km": self.z0_km,
            "jitter": self.jitter,
        }


@dataclass(frozen=True)
class LongPeriodTaylorSplitManifest:
    calibration_seed: int = CALIBRATION_SEED
    development_seed: int = DEVELOPMENT_SEED
    sealed_holdout_seed: int = SEALED_HOLDOUT_SEED
    direction_count: int = DEFAULT_DIRECTION_COUNT
    amplitude_levels: tuple[float, ...] = AMPLITUDE_LEVELS

    def __post_init__(self):
        seeds = (
            int(self.calibration_seed),
            int(self.development_seed),
            int(self.sealed_holdout_seed),
        )
        if seeds != (CALIBRATION_SEED, DEVELOPMENT_SEED, SEALED_HOLDOUT_SEED):
            raise ValueError("Plan L v1 split seeds are fixed")
        if len(set(seeds)) != 3:
            raise ValueError("split seeds must be distinct")
        if int(self.direction_count) != self.direction_count or self.direction_count < 1:
            raise ValueError("direction_count must be a positive integer")
        if tuple(self.amplitude_levels) != AMPLITUDE_LEVELS:
            raise ValueError("Plan L v1 amplitude levels are fixed")

    @property
    def model_count(self) -> int:
        return int(self.direction_count * 2 * len(self.amplitude_levels))


@dataclass(frozen=True)
class LongPeriodMixedResponseSplitManifest:
    calibration_seed: int = MIXED_RESPONSE_CALIBRATION_SEED
    development_seed: int = MIXED_RESPONSE_DEVELOPMENT_SEED
    sealed_holdout_seed: int = MIXED_RESPONSE_SEALED_HOLDOUT_SEED
    direction_count: int = DEFAULT_DIRECTION_COUNT
    amplitude_levels: tuple[float, ...] = AMPLITUDE_LEVELS

    def __post_init__(self):
        seeds = (
            int(self.calibration_seed),
            int(self.development_seed),
            int(self.sealed_holdout_seed),
        )
        expected = (
            MIXED_RESPONSE_CALIBRATION_SEED,
            MIXED_RESPONSE_DEVELOPMENT_SEED,
            MIXED_RESPONSE_SEALED_HOLDOUT_SEED,
        )
        if seeds != expected:
            raise ValueError("Plan L v2 split seeds are fixed")
        if len(set(seeds)) != 3:
            raise ValueError("split seeds must be distinct")
        if int(self.direction_count) != self.direction_count or self.direction_count < 1:
            raise ValueError("direction_count must be a positive integer")
        if tuple(self.amplitude_levels) != AMPLITUDE_LEVELS:
            raise ValueError("Plan L v2 amplitude levels are fixed")

    @property
    def model_count(self) -> int:
        return int(self.direction_count * 2 * len(self.amplitude_levels))


@dataclass(frozen=True)
class ModeIdentityThresholds:
    minimum_adjacent_relative_gap: float = 0.02
    minimum_eigenfunction_overlap: float = 0.80
    minimum_kernel_overlap: float = 0.80
    maximum_group_velocity_relative_jump: float = 0.20

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class LongPeriodTaylorGate:
    """Numerical thresholds frozen before the sealed holdout is authorized."""

    relative_observation_sigma: float = 0.001
    pointwise_error_over_sigma_p90_maximum: float = 1.0 / 3.0
    pointwise_error_over_sigma_p99_maximum: float = 1.0
    curve_rms_error_over_sigma_p90_maximum: float = 1.0 / 3.0
    taylor2_to_taylor1_p90_maximum: float = 0.5
    taylor2_residual_order_minimum: float = 2.7
    order_fit_amplitude_levels: tuple[float, ...] = ORDER_FIT_AMPLITUDE_LEVELS
    reference_truncation_error_over_sigma_p99_maximum: float = 1.0 / 30.0
    minimum_scalar_speedup: float = 1000.0
    minimum_batch_speedup: float = 1000.0
    primary_point_support: float = 1.0
    frequency_buffer_hz: float = 0.0025
    mode_identity: ModeIdentityThresholds = ModeIdentityThresholds()

    def __post_init__(self):
        positive = (
            "relative_observation_sigma",
            "pointwise_error_over_sigma_p90_maximum",
            "pointwise_error_over_sigma_p99_maximum",
            "curve_rms_error_over_sigma_p90_maximum",
            "taylor2_to_taylor1_p90_maximum",
            "taylor2_residual_order_minimum",
            "reference_truncation_error_over_sigma_p99_maximum",
            "minimum_scalar_speedup",
            "minimum_batch_speedup",
            "frequency_buffer_hz",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if tuple(self.order_fit_amplitude_levels) != ORDER_FIT_AMPLITUDE_LEVELS:
            raise ValueError("Taylor residual fit amplitudes are fixed")
        if not np.isclose(self.primary_point_support, 1.0):
            raise ValueError("Plan L v1 requires 100 percent primary support")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RadialPriorCohort:
    delta_m: np.ndarray
    z: np.ndarray
    direction_index: np.ndarray
    sign: np.ndarray
    target_maximum_anomaly: np.ndarray
    achieved_maximum_anomaly: np.ndarray
    signed_mean_shear_anomaly: np.ndarray
    seed: int
    split_kind: str

    @property
    def count(self) -> int:
        return int(self.delta_m.shape[0])


def compute_shear_log_operator(basis) -> np.ndarray:
    """Extract forward-node Vsv/Vsh log rows from the inversion basis."""

    operator = np.asarray(basis.full_log_operator(), dtype=float)
    n_nodes = basis.reference.n_nodes
    expected_rows = len(VELOCITY_FIELDS) * n_nodes
    if operator.shape != (expected_rows, basis.n_params):
        raise ValueError("basis full_log_operator has an inconsistent shape")
    rows = []
    for field_name in ("vsv", "vsh"):
        field_index = VELOCITY_FIELDS.index(field_name)
        rows.append(
            operator[field_index * n_nodes : (field_index + 1) * n_nodes]
        )
    return np.vstack(rows)


def physical_shear_anomaly(shear_log_operator, delta_m) -> tuple[float, float]:
    operator = np.asarray(shear_log_operator, dtype=float)
    model = np.asarray(delta_m, dtype=float)
    if operator.ndim != 2 or model.shape != (operator.shape[1],):
        raise ValueError("shear operator and delta_m dimensions do not agree")
    with np.errstate(over="raise", invalid="raise"):
        anomaly = np.expm1(operator @ model)
    if np.any(~np.isfinite(anomaly)):
        raise ValueError("physical shear anomaly is non-finite")
    return float(np.max(np.abs(anomaly))), float(np.mean(anomaly))


def _scale_ray_to_physical_anomaly(operator, direction_m, target: float) -> float:
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target physical anomaly must lie in (0, 1)")
    high = 1.0
    for _ in range(80):
        maximum, _ = physical_shear_anomaly(operator, high * direction_m)
        if maximum >= target:
            break
        high *= 2.0
    else:
        raise RuntimeError("could not bracket the target physical anomaly")
    low = 0.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        maximum, _ = physical_shear_anomaly(operator, middle * direction_m)
        if maximum < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _build_radial_prior_cohort(
    basis,
    prior_cholesky,
    *,
    expected_seed_by_kind,
    policy_name: str,
    split_kind: str,
    seed: int,
    direction_count: int = DEFAULT_DIRECTION_COUNT,
    amplitude_levels=AMPLITUDE_LEVELS,
    holdout_authorized: bool = False,
) -> RadialPriorCohort:
    """Build paired prior rays; the sealed seed needs explicit authorization."""

    expected_seed = dict(expected_seed_by_kind)
    if split_kind not in expected_seed:
        raise ValueError("split_kind must be calibration, development, or holdout")
    if int(seed) != expected_seed[split_kind]:
        raise ValueError("split seed does not match the immutable split policy")
    if split_kind == "holdout" and not holdout_authorized:
        raise PermissionError(
            "sealed holdout generation requires a verified frozen gate lock"
        )
    cholesky = np.asarray(prior_cholesky, dtype=float)
    if cholesky.shape != (basis.n_params, basis.n_params):
        raise ValueError("prior_cholesky and basis dimensions do not agree")
    if np.any(~np.isfinite(cholesky)) or np.linalg.matrix_rank(cholesky) != basis.n_params:
        raise ValueError("prior_cholesky must be finite and nonsingular")
    if int(direction_count) != direction_count or direction_count < 1:
        raise ValueError("direction_count must be a positive integer")
    levels = np.asarray(tuple(amplitude_levels), dtype=float)
    if tuple(levels.tolist()) != AMPLITUDE_LEVELS:
        raise ValueError(f"{policy_name} amplitude levels are fixed")

    shear_operator = compute_shear_log_operator(basis)
    rng = np.random.default_rng(int(seed))
    directions_z = rng.standard_normal((int(direction_count), basis.n_params))
    directions_z /= np.linalg.norm(directions_z, axis=1)[:, None]
    delta_rows = []
    z_rows = []
    direction_rows = []
    sign_rows = []
    target_rows = []
    achieved_rows = []
    mean_rows = []
    for direction_index, direction_z in enumerate(directions_z):
        for sign_value, sign_label in ((-1.0, "negative"), (1.0, "positive")):
            signed_z = sign_value * direction_z
            signed_m = cholesky @ signed_z
            for target in levels:
                scale = _scale_ray_to_physical_anomaly(
                    shear_operator, signed_m, float(target)
                )
                delta_m = scale * signed_m
                z = scale * signed_z
                maximum, signed_mean = physical_shear_anomaly(
                    shear_operator, delta_m
                )
                if not check_stiffness_posdef(basis.m_to_forward_model(delta_m)).ok:
                    raise RuntimeError(
                        "nonphysical radial model: "
                        f"direction={direction_index}, sign={sign_label}, "
                        f"target={target:.3f}"
                    )
                if not np.isclose(maximum, target, rtol=0.0, atol=2.0e-12):
                    raise RuntimeError("physical anomaly ray scaling did not converge")
                delta_rows.append(delta_m)
                z_rows.append(z)
                direction_rows.append(direction_index)
                sign_rows.append(sign_label)
                target_rows.append(target)
                achieved_rows.append(maximum)
                mean_rows.append(signed_mean)
    return RadialPriorCohort(
        delta_m=np.asarray(delta_rows),
        z=np.asarray(z_rows),
        direction_index=np.asarray(direction_rows, dtype=int),
        sign=np.asarray(sign_rows),
        target_maximum_anomaly=np.asarray(target_rows),
        achieved_maximum_anomaly=np.asarray(achieved_rows),
        signed_mean_shear_anomaly=np.asarray(mean_rows),
        seed=int(seed),
        split_kind=str(split_kind),
    )


def build_radial_prior_cohort(
    basis,
    prior_cholesky,
    *,
    split_kind: str,
    seed: int,
    direction_count: int = DEFAULT_DIRECTION_COUNT,
    amplitude_levels=AMPLITUDE_LEVELS,
    holdout_authorized: bool = False,
) -> RadialPriorCohort:
    """Build a Plan L v1 radial cohort under its immutable seed policy."""

    return _build_radial_prior_cohort(
        basis,
        prior_cholesky,
        expected_seed_by_kind={
            "calibration": CALIBRATION_SEED,
            "development": DEVELOPMENT_SEED,
            "holdout": SEALED_HOLDOUT_SEED,
        },
        policy_name="Plan L v1",
        split_kind=split_kind,
        seed=seed,
        direction_count=direction_count,
        amplitude_levels=amplitude_levels,
        holdout_authorized=holdout_authorized,
    )


def build_mixed_response_radial_prior_cohort(
    basis,
    prior_cholesky,
    *,
    split_kind: str,
    seed: int,
    direction_count: int = DEFAULT_DIRECTION_COUNT,
    amplitude_levels=AMPLITUDE_LEVELS,
    holdout_authorized: bool = False,
) -> RadialPriorCohort:
    """Build a Plan L v2 cohort with fresh development and holdout seeds."""

    return _build_radial_prior_cohort(
        basis,
        prior_cholesky,
        expected_seed_by_kind={
            "calibration": MIXED_RESPONSE_CALIBRATION_SEED,
            "development": MIXED_RESPONSE_DEVELOPMENT_SEED,
            "holdout": MIXED_RESPONSE_SEALED_HOLDOUT_SEED,
        },
        policy_name="Plan L v2",
        split_kind=split_kind,
        seed=seed,
        direction_count=direction_count,
        amplitude_levels=amplitude_levels,
        holdout_authorized=holdout_authorized,
    )


def radial_neighbor_pairs(cohort: RadialPriorCohort) -> tuple[tuple[int, int], ...]:
    """Connect adjacent amplitudes and the two signs along every random ray."""

    pairs = []
    for direction in np.unique(cohort.direction_index):
        closest = []
        for sign in ("negative", "positive"):
            indices = np.flatnonzero(
                (cohort.direction_index == direction) & (cohort.sign == sign)
            )
            order = indices[np.argsort(cohort.target_maximum_anomaly[indices])]
            pairs.extend(zip(order[:-1].tolist(), order[1:].tolist()))
            if order.size:
                closest.append(int(order[0]))
        if len(closest) == 2:
            pairs.append((closest[0], closest[1]))
    return tuple((int(left), int(right)) for left, right in pairs)


def reference_fingerprint(model) -> str:
    return array_fingerprint(
        model.depths,
        model.rho,
        model.vph,
        model.vpv,
        model.vsh,
        model.vsv,
        model.eta,
        model.is_discontinuity,
    )


def basis_fingerprint(basis) -> str:
    return json_sha256(basis.describe())


def prior_fingerprint(prior) -> str:
    return json_sha256(prior.describe())


def dataset_fingerprint(
    manifest: LongPeriodTaylorDatasetManifest = LongPeriodTaylorDatasetManifest(),
) -> str:
    return json_sha256(asdict(manifest))


__all__ = [
    "AMPLITUDE_LEVELS",
    "ARTIFACT_REFERENCE_DEPTH_KM",
    "CALIBRATION_SEED",
    "DEFAULT_DIRECTION_COUNT",
    "DEVELOPMENT_SEED",
    "LongPeriodTaylorDatasetManifest",
    "LongPeriodTaylorGate",
    "LongPeriodTaylorPriorManifest",
    "LongPeriodTaylorSplitManifest",
    "LongPeriodMixedResponseDatasetManifest",
    "LongPeriodMixedResponseSplitManifest",
    "MIXED_RESPONSE_CALIBRATION_SEED",
    "MIXED_RESPONSE_DEVELOPMENT_SEED",
    "MIXED_RESPONSE_SEALED_HOLDOUT_SEED",
    "ModeIdentityThresholds",
    "ORDER_FIT_AMPLITUDE_LEVELS",
    "PRIMARY_AMPLITUDE_LEVELS",
    "REFERENCE_DEPTHS_KM",
    "REFERENCE_MAX_SPACING_KM",
    "RadialPriorCohort",
    "SEALED_HOLDOUT_SEED",
    "basis_fingerprint",
    "build_radial_prior_cohort",
    "build_mixed_response_radial_prior_cohort",
    "compute_shear_log_operator",
    "dataset_fingerprint",
    "physical_shear_anomaly",
    "prior_fingerprint",
    "radial_neighbor_pairs",
    "reference_fingerprint",
]
