"""Accuracy, mode-safety, convergence-order, and runtime gates for Plan L."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .long_period_taylor import (
    NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
    LongPeriodTaylorArtifact,
    PRIMARY_TARGETS,
)
from .long_period_taylor_manifest import (
    LongPeriodTaylorGate,
    ModeIdentityThresholds,
)
from .long_period_taylor_mode import ModeCharacterDiagnostics


@dataclass(frozen=True)
class ErrorQuantiles:
    p50: float
    p90: float
    p99: float
    maximum: float


@dataclass(frozen=True)
class FrequencyGateResult:
    frequency_hz: float
    response_kind: str
    passed: bool
    assessed_point_count: int
    exact_branch_failure_count: int
    mode_identity_failure_count: int
    prediction_nonfinite_count: int
    nonpositive_response_count: int
    unsupported_point_count: int
    minimum_response_order1: float
    minimum_response_order2: float
    taylor1_absolute_relative_error: ErrorQuantiles
    taylor2_absolute_relative_error: ErrorQuantiles
    taylor2_absolute_error_over_sigma: ErrorQuantiles
    taylor2_signed_relative_error: ErrorQuantiles
    taylor2_to_taylor1_p90: float
    taylor1_residual_order: float
    taylor2_residual_order: float


@dataclass(frozen=True)
class StressResult:
    model_count: int
    exact_branch_failure_count: int
    mode_identity_failure_count: int
    taylor1_absolute_relative_error: ErrorQuantiles
    taylor2_absolute_relative_error: ErrorQuantiles
    taylor2_absolute_error_over_sigma: ErrorQuantiles


@dataclass(frozen=True)
class CurveGateResult:
    wave: str
    root_order: int
    response_kind: str
    passed: bool
    primary_model_count: int
    primary_frequency_count: int
    supported_point_count: int
    supported_fraction: float
    minimum_primary_model_support_fraction: float
    minimum_primary_contiguous_support_count: int
    low_frequency_contiguous_pass_count: int
    buffered_candidate_frequency_count: int
    exact_branch_failure_count: int
    mode_identity_failure_count: int
    prediction_nonfinite_count: int
    nonpositive_response_count: int
    minimum_response_order1: float
    minimum_response_order2: float
    taylor1_absolute_relative_error: ErrorQuantiles
    taylor2_absolute_relative_error: ErrorQuantiles
    taylor2_absolute_error_over_sigma: ErrorQuantiles
    taylor2_signed_relative_error: ErrorQuantiles
    curve_rms_error_over_sigma: ErrorQuantiles
    taylor2_to_taylor1_p90: float
    pointwise_taylor1_to_taylor2_improvement_median: float
    surrogate_to_exact_anomaly_error_ratio_median: float
    zero_mode_failure_probability_upper_95: float
    p90_error_over_sigma_by_amplitude: dict[str, float]
    frequency_results: tuple[FrequencyGateResult, ...]
    stress_10_percent: StressResult


@dataclass(frozen=True)
class PlanLGateResult:
    passed: bool
    split_kind: str
    curve_results: tuple[CurveGateResult, ...]


@dataclass(frozen=True)
class CurveSpeedResult:
    wave: str
    root_order: int
    exact_scalar_seconds: float
    surrogate_scalar_seconds: float
    exact_batch_seconds: float
    surrogate_batch_seconds: float
    scalar_speedup: float
    batch_speedup: float
    passed: bool


def _finite_quantile(values, quantile: float, *, empty=float("inf")) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return float(empty)
    return float(np.quantile(finite, quantile))


def _finite_minimum(values, *, empty=float("nan")) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return float(empty)
    return float(np.min(finite))


def error_quantiles(values, *, empty=float("inf")) -> ErrorQuantiles:
    return ErrorQuantiles(
        p50=_finite_quantile(values, 0.50, empty=empty),
        p90=_finite_quantile(values, 0.90, empty=empty),
        p99=_finite_quantile(values, 0.99, empty=empty),
        maximum=_finite_quantile(values, 1.00, empty=empty),
    )


def zero_failure_upper_probability(
    sample_count: int, confidence: float = 0.95
) -> float:
    """One-sided Clopper--Pearson upper bound after zero observed failures."""

    count = int(sample_count)
    if count < 0 or not 0.0 < float(confidence) < 1.0:
        raise ValueError("invalid sample count or confidence")
    if count == 0:
        return 1.0
    return float(1.0 - (1.0 - float(confidence)) ** (1.0 / count))


def classify_mode_identity(
    diagnostics: ModeCharacterDiagnostics,
    thresholds: ModeIdentityThresholds = ModeIdentityThresholds(),
) -> np.ndarray:
    """Fail a small-gap point when continuation evidence is also abnormal.

    Root/trace failure and a Hungarian mode-shape assignment disagreement are
    always unsafe.  No individual soft diagnostic is otherwise promoted to a
    mode-identity claim by itself.
    """

    gap_finite = np.isfinite(diagnostics.adjacent_relative_gap)
    continuation_finite = (
        np.isfinite(diagnostics.eigenfunction_overlap)
        & np.isfinite(diagnostics.kernel_overlap)
        & np.isfinite(diagnostics.group_velocity_relative_jump)
        & np.isfinite(diagnostics.crustal_energy_fraction)
        & np.isfinite(diagnostics.energy_centroid_jump_km)
    )
    hard_safe = (
        diagnostics.exact_success
        & diagnostics.trace_agreement
        & gap_finite
        & continuation_finite
    )
    small_gap = (
        diagnostics.adjacent_relative_gap
        < thresholds.minimum_adjacent_relative_gap
    )
    abnormal_continuation = (
        diagnostics.eigenfunction_overlap
        < thresholds.minimum_eigenfunction_overlap
    ) | (
        diagnostics.kernel_overlap < thresholds.minimum_kernel_overlap
    ) | (
        diagnostics.group_velocity_relative_jump
        > thresholds.maximum_group_velocity_relative_jump
    )
    return hard_safe & ~(small_gap & abnormal_continuation)


def _residual_order(
    prediction,
    exact,
    valid,
    maximum_anomaly,
    amplitude_levels,
) -> float:
    residual_p90 = []
    fitted_amplitude = []
    for level in amplitude_levels:
        rows = np.isclose(maximum_anomaly, level, rtol=0.0, atol=2.0e-12)
        selected = np.abs(prediction - exact)[rows & valid]
        selected = selected[np.isfinite(selected) & (selected > 0.0)]
        if selected.size:
            residual_p90.append(float(np.quantile(selected, 0.90)))
            fitted_amplitude.append(float(level))
    residual_p90 = np.asarray(residual_p90, dtype=float)
    fitted_amplitude = np.asarray(fitted_amplitude, dtype=float)
    usable = np.isfinite(residual_p90) & (residual_p90 > 0.0)
    if np.count_nonzero(usable) < 3:
        return float("nan")
    return float(
        np.polyfit(
            np.log(fitted_amplitude[usable]),
            np.log(residual_p90[usable]),
            1,
        )[0]
    )


def _stress_result(
    taylor1,
    taylor2,
    exact,
    exact_success,
    mode_safe,
    maximum_anomaly,
    relative_sigma,
) -> StressResult:
    stress_rows = np.isclose(maximum_anomaly, 0.10, rtol=0.0, atol=2.0e-12)
    row_mask = stress_rows[:, None]
    valid = (
        row_mask
        & exact_success
        & mode_safe
        & np.isfinite(exact)
        & (exact > 0.0)
        & np.isfinite(taylor1)
        & np.isfinite(taylor2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        relative1 = taylor1 / exact - 1.0
        relative2 = taylor2 / exact - 1.0
        normalized2 = np.abs(taylor2 - exact) / (relative_sigma * exact)
    return StressResult(
        model_count=int(np.count_nonzero(stress_rows)),
        exact_branch_failure_count=int(
            np.count_nonzero(row_mask & ~exact_success)
        ),
        mode_identity_failure_count=int(
            np.count_nonzero(row_mask & exact_success & ~mode_safe)
        ),
        taylor1_absolute_relative_error=error_quantiles(
            np.abs(relative1[valid])
        ),
        taylor2_absolute_relative_error=error_quantiles(
            np.abs(relative2[valid])
        ),
        taylor2_absolute_error_over_sigma=error_quantiles(
            normalized2[valid]
        ),
    )


def validate_long_period_taylor_curve(
    artifact: LongPeriodTaylorArtifact,
    wave: str,
    root_order: int,
    delta_m,
    exact_phase_velocity,
    exact_success,
    diagnostics: ModeCharacterDiagnostics,
    maximum_physical_anomaly,
    *,
    gate: LongPeriodTaylorGate = LongPeriodTaylorGate(),
) -> CurveGateResult:
    """Apply every primary accuracy and branch-safety gate to one curve."""

    models = np.asarray(delta_m, dtype=float)
    exact = np.asarray(exact_phase_velocity, dtype=float)
    success = np.asarray(exact_success, dtype=bool)
    anomaly = np.asarray(maximum_physical_anomaly, dtype=float)
    if models.ndim != 2 or models.shape[1] != artifact.n_params:
        raise ValueError("delta_m must have shape (model, parameter)")
    expected = (models.shape[0], artifact.frequency_hz.size)
    if exact.shape != expected or success.shape != expected:
        raise ValueError("exact phase/success arrays do not match the artifact")
    if diagnostics.shape != expected or anomaly.shape != (models.shape[0],):
        raise ValueError("diagnostics/anomaly arrays do not match the models")

    curve = artifact.get_curve(wave, root_order)
    taylor1, _, _, evaluated_anomaly = artifact.evaluate_unchecked_batch(
        wave, root_order, models, order=1
    )
    taylor2, _, _, _ = artifact.evaluate_unchecked_batch(
        wave, root_order, models, order=2
    )
    response2, response1, _, response_anomaly = (
        artifact.evaluate_response_unchecked_batch(
            wave, root_order, models, order=2
        )
    )
    if not np.allclose(
        evaluated_anomaly, anomaly, rtol=0.0, atol=3.0e-12, equal_nan=False
    ):
        raise ValueError("stored and artifact-computed physical anomalies differ")
    if not np.array_equal(response_anomaly, evaluated_anomaly):
        raise ValueError("phase and response evaluations used different models")
    online = artifact.predict_batch(wave, root_order, models, order=2)
    mode_safe = classify_mode_identity(diagnostics, gate.mode_identity)
    primary_rows = anomaly <= artifact.maximum_physical_anomaly + 2.0e-12
    primary = primary_rows[:, None]
    exact_finite = np.isfinite(exact) & (exact > 0.0)
    prediction_finite = np.isfinite(taylor1) & np.isfinite(taylor2) & (taylor2 > 0.0)
    scored = primary & success & mode_safe & exact_finite & prediction_finite
    with np.errstate(divide="ignore", invalid="ignore"):
        relative1 = taylor1 / exact - 1.0
        relative2 = taylor2 / exact - 1.0
        normalized2 = np.abs(taylor2 - exact) / (
            gate.relative_observation_sigma * exact
        )
    absolute1 = np.abs(relative1)
    absolute2 = np.abs(relative2)
    exact_failure = primary & (~success | ~exact_finite)
    mode_failure = primary & success & exact_finite & ~mode_safe
    nonfinite = primary & ~prediction_finite
    if curve.response_kind == NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND:
        nonpositive_response = primary & (
            (np.isfinite(response1) & (response1 <= 0.0))
            | (np.isfinite(response2) & (response2 <= 0.0))
        )
    else:
        nonpositive_response = np.zeros_like(response1, dtype=bool)
    unsupported = primary & ~online.point_supported

    frequency_results = []
    for frequency_index, frequency in enumerate(artifact.frequency_hz):
        valid = scored[:, frequency_index]
        t1_quantiles = error_quantiles(absolute1[:, frequency_index][valid])
        t2_quantiles = error_quantiles(absolute2[:, frequency_index][valid])
        normalized_quantiles = error_quantiles(
            normalized2[:, frequency_index][valid]
        )
        signed_quantiles = error_quantiles(
            relative2[:, frequency_index][valid]
        )
        ratio = t2_quantiles.p90 / max(
            t1_quantiles.p90, np.finfo(float).tiny
        )
        order1 = _residual_order(
            taylor1[:, frequency_index],
            exact[:, frequency_index],
            success[:, frequency_index]
            & mode_safe[:, frequency_index]
            & exact_finite[:, frequency_index]
            & prediction_finite[:, frequency_index],
            anomaly,
            gate.order_fit_amplitude_levels,
        )
        order2 = _residual_order(
            taylor2[:, frequency_index],
            exact[:, frequency_index],
            success[:, frequency_index]
            & mode_safe[:, frequency_index]
            & exact_finite[:, frequency_index]
            & prediction_finite[:, frequency_index],
            anomaly,
            gate.order_fit_amplitude_levels,
        )
        exact_failure_count = int(np.count_nonzero(exact_failure[:, frequency_index]))
        mode_failure_count = int(np.count_nonzero(mode_failure[:, frequency_index]))
        nonfinite_count = int(np.count_nonzero(nonfinite[:, frequency_index]))
        nonpositive_response_count = int(
            np.count_nonzero(nonpositive_response[:, frequency_index])
        )
        unsupported_count = int(np.count_nonzero(unsupported[:, frequency_index]))
        minimum_response_order1 = _finite_minimum(
            response1[:, frequency_index][primary_rows]
        )
        minimum_response_order2 = _finite_minimum(
            response2[:, frequency_index][primary_rows]
        )
        frequency_passed = (
            normalized_quantiles.p90
            <= gate.pointwise_error_over_sigma_p90_maximum
            and normalized_quantiles.p99
            <= gate.pointwise_error_over_sigma_p99_maximum
            and ratio <= gate.taylor2_to_taylor1_p90_maximum
            and np.isfinite(order2)
            and order2 >= gate.taylor2_residual_order_minimum
            and exact_failure_count == 0
            and mode_failure_count == 0
            and nonfinite_count == 0
            and nonpositive_response_count == 0
            and unsupported_count == 0
            and int(np.count_nonzero(valid))
            == int(np.count_nonzero(primary_rows))
        )
        frequency_results.append(
            FrequencyGateResult(
                frequency_hz=float(frequency),
                response_kind=curve.response_kind,
                passed=bool(frequency_passed),
                assessed_point_count=int(np.count_nonzero(valid)),
                exact_branch_failure_count=exact_failure_count,
                mode_identity_failure_count=mode_failure_count,
                prediction_nonfinite_count=nonfinite_count,
                nonpositive_response_count=nonpositive_response_count,
                unsupported_point_count=unsupported_count,
                minimum_response_order1=minimum_response_order1,
                minimum_response_order2=minimum_response_order2,
                taylor1_absolute_relative_error=t1_quantiles,
                taylor2_absolute_relative_error=t2_quantiles,
                taylor2_absolute_error_over_sigma=normalized_quantiles,
                taylor2_signed_relative_error=signed_quantiles,
                taylor2_to_taylor1_p90=float(ratio),
                taylor1_residual_order=order1,
                taylor2_residual_order=order2,
            )
        )

    complete_model = primary_rows & np.all(scored, axis=1)
    curve_rms = np.sqrt(np.mean(normalized2[complete_model] ** 2, axis=1))
    curve_rms_quantiles = error_quantiles(curve_rms)
    scored_values = scored
    t1_all = error_quantiles(absolute1[scored_values])
    t2_all = error_quantiles(absolute2[scored_values])
    normalized_all = error_quantiles(normalized2[scored_values])
    signed_all = error_quantiles(relative2[scored_values])
    overall_ratio = t2_all.p90 / max(t1_all.p90, np.finfo(float).tiny)
    improvement_values = absolute1[scored_values] / np.maximum(
        absolute2[scored_values], np.finfo(float).eps
    )
    improvement_median = _finite_quantile(improvement_values, 0.50)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact_anomaly = exact / curve.reference_phase_velocity[None, :] - 1.0
    anomaly_ratio_values = absolute2[scored_values] / np.maximum(
        np.abs(exact_anomaly[scored_values]), np.finfo(float).eps
    )
    anomaly_ratio = _finite_quantile(anomaly_ratio_values, 0.50)
    p90_by_amplitude = {}
    for level in np.unique(anomaly[primary_rows]):
        rows = np.isclose(anomaly, level, rtol=0.0, atol=2.0e-12)
        values = normalized2[rows[:, None] & scored_values]
        p90_by_amplitude[f"{float(level):.6g}"] = _finite_quantile(values, 0.90)

    primary_point_count = int(np.count_nonzero(primary)) * exact.shape[1]
    supported_count = int(np.count_nonzero(primary & online.point_supported))
    primary_support = online.point_supported[primary_rows]
    if primary_support.shape[0]:
        model_support_fraction = np.mean(primary_support, axis=1)

        def longest_run(row):
            longest = current = 0
            for supported in row:
                current = current + 1 if supported else 0
                longest = max(longest, current)
            return longest

        minimum_model_support = float(np.min(model_support_fraction))
        minimum_contiguous_support = min(
            longest_run(row) for row in primary_support
        )
    else:
        minimum_model_support = 0.0
        minimum_contiguous_support = 0
    contiguous_pass_count = 0
    for result in frequency_results:
        if not result.passed:
            break
        contiguous_pass_count += 1
    buffered_candidate_count = (
        contiguous_pass_count
        if contiguous_pass_count == len(frequency_results)
        else max(0, contiguous_pass_count - 1)
    )
    exact_failure_count = int(np.count_nonzero(exact_failure))
    mode_failure_count = int(np.count_nonzero(mode_failure))
    nonfinite_count = int(np.count_nonzero(nonfinite))
    nonpositive_response_count = int(np.count_nonzero(nonpositive_response))
    minimum_response_order1 = _finite_minimum(response1[primary_rows])
    minimum_response_order2 = _finite_minimum(response2[primary_rows])
    passed = (
        all(result.passed for result in frequency_results)
        and curve_rms_quantiles.p90
        <= gate.curve_rms_error_over_sigma_p90_maximum
        and overall_ratio <= gate.taylor2_to_taylor1_p90_maximum
        and supported_count == primary_point_count
        and exact_failure_count == 0
        and mode_failure_count == 0
        and nonfinite_count == 0
        and nonpositive_response_count == 0
    )
    return CurveGateResult(
        wave=curve.wave,
        root_order=curve.root_order,
        response_kind=curve.response_kind,
        passed=bool(passed),
        primary_model_count=int(np.count_nonzero(primary_rows)),
        primary_frequency_count=int(exact.shape[1]),
        supported_point_count=supported_count,
        supported_fraction=(
            float(supported_count / primary_point_count)
            if primary_point_count
            else 0.0
        ),
        minimum_primary_model_support_fraction=minimum_model_support,
        minimum_primary_contiguous_support_count=int(
            minimum_contiguous_support
        ),
        low_frequency_contiguous_pass_count=int(contiguous_pass_count),
        buffered_candidate_frequency_count=int(buffered_candidate_count),
        exact_branch_failure_count=exact_failure_count,
        mode_identity_failure_count=mode_failure_count,
        prediction_nonfinite_count=nonfinite_count,
        nonpositive_response_count=nonpositive_response_count,
        minimum_response_order1=minimum_response_order1,
        minimum_response_order2=minimum_response_order2,
        taylor1_absolute_relative_error=t1_all,
        taylor2_absolute_relative_error=t2_all,
        taylor2_absolute_error_over_sigma=normalized_all,
        taylor2_signed_relative_error=signed_all,
        curve_rms_error_over_sigma=curve_rms_quantiles,
        taylor2_to_taylor1_p90=float(overall_ratio),
        pointwise_taylor1_to_taylor2_improvement_median=improvement_median,
        surrogate_to_exact_anomaly_error_ratio_median=anomaly_ratio,
        zero_mode_failure_probability_upper_95=(
            zero_failure_upper_probability(primary_point_count)
            if mode_failure_count == 0
            else 1.0
        ),
        p90_error_over_sigma_by_amplitude=p90_by_amplitude,
        frequency_results=tuple(frequency_results),
        stress_10_percent=_stress_result(
            taylor1,
            taylor2,
            exact,
            success,
            mode_safe,
            anomaly,
            gate.relative_observation_sigma,
        ),
    )


def evaluate_plan_l_gate(
    curve_results, *, split_kind: str
) -> PlanLGateResult:
    results = tuple(curve_results)
    if tuple((result.wave, result.root_order) for result in results) != PRIMARY_TARGETS:
        raise ValueError("Plan L gate requires exactly the four primary curves")
    if split_kind not in {"calibration", "development", "holdout"}:
        raise ValueError("unrecognized Plan L split kind")
    return PlanLGateResult(
        passed=bool(all(result.passed for result in results)),
        split_kind=str(split_kind),
        curve_results=results,
    )


def benchmark_long_period_taylor_curve(
    *,
    wave: str,
    root_order: int,
    exact_scalar,
    surrogate_scalar,
    exact_batch,
    surrogate_batch,
    repeats: int = 5,
    gate: LongPeriodTaylorGate = LongPeriodTaylorGate(),
) -> CurveSpeedResult:
    if int(repeats) < 1:
        raise ValueError("repeats must be positive")

    def minimum_time(function):
        function()
        samples = []
        for _ in range(int(repeats)):
            start = time.perf_counter()
            function()
            samples.append(time.perf_counter() - start)
        return float(min(samples))

    exact_scalar_seconds = minimum_time(exact_scalar)
    surrogate_scalar_seconds = minimum_time(surrogate_scalar)
    exact_batch_seconds = minimum_time(exact_batch)
    surrogate_batch_seconds = minimum_time(surrogate_batch)
    scalar_speedup = exact_scalar_seconds / max(
        surrogate_scalar_seconds, np.finfo(float).tiny
    )
    batch_speedup = exact_batch_seconds / max(
        surrogate_batch_seconds, np.finfo(float).tiny
    )
    return CurveSpeedResult(
        wave=str(wave).lower(),
        root_order=int(root_order),
        exact_scalar_seconds=exact_scalar_seconds,
        surrogate_scalar_seconds=surrogate_scalar_seconds,
        exact_batch_seconds=exact_batch_seconds,
        surrogate_batch_seconds=surrogate_batch_seconds,
        scalar_speedup=scalar_speedup,
        batch_speedup=batch_speedup,
        passed=bool(
            scalar_speedup >= gate.minimum_scalar_speedup
            and batch_speedup >= gate.minimum_batch_speedup
        ),
    )


__all__ = [
    "CurveGateResult",
    "CurveSpeedResult",
    "ErrorQuantiles",
    "FrequencyGateResult",
    "PlanLGateResult",
    "StressResult",
    "benchmark_long_period_taylor_curve",
    "classify_mode_identity",
    "error_quantiles",
    "evaluate_plan_l_gate",
    "validate_long_period_taylor_curve",
    "zero_failure_upper_probability",
]
