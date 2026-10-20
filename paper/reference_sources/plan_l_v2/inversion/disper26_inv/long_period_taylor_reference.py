"""Reference-jet construction and 2200/2400/2600 km convergence checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DatasetSpec
from .engine import Datum
from .long_period_taylor import (
    PRIMARY_FREQUENCY_HZ,
    PRIMARY_TARGETS,
    build_long_period_taylor_curve,
)
from .long_period_taylor_exact import (
    ExactModeTraceEnsemble,
    collect_exact_mode_trace_ensemble,
)
from .long_period_taylor_manifest import (
    ARTIFACT_REFERENCE_DEPTH_KM,
    CALIBRATION_SEED,
    REFERENCE_DEPTHS_KM,
    REFERENCE_MAX_SPACING_KM,
    LongPeriodTaylorGate,
    LongPeriodTaylorPriorManifest,
    basis_fingerprint,
    build_radial_prior_cohort,
    prior_fingerprint,
    reference_fingerprint,
)
from .long_period_taylor_mode import (
    ModeCharacterDiagnostics,
    compute_character_assignment,
    compute_mode_shape_overlap,
)
from .pipeline import run_phase1
from .priors import PriorCovariance
from .shallow import shallow_basis, shallow_reference


REFERENCE_PHASE_TRACE_RELATIVE_TOLERANCE = 2.0e-8


@dataclass(frozen=True)
class LongPeriodReferenceDepthResult:
    bottom_depth_km: float
    frequency_hz: np.ndarray
    curve_keys: tuple[tuple[str, int], ...]
    phase_velocity: np.ndarray
    phase_gradient_m: np.ndarray
    phase_hessian_m: np.ndarray
    phase1_relative_error: np.ndarray
    phase1_hessian_asymmetry: np.ndarray
    exact_by_wave: tuple[ExactModeTraceEnsemble, ...]
    diagnostics_by_curve: tuple[ModeCharacterDiagnostics, ...]
    reference_fingerprint: str
    basis_fingerprint: str
    prior_fingerprint: str
    basis_description: dict
    prior_description: dict

    def __post_init__(self):
        frequency = np.asarray(self.frequency_hz, dtype=float)
        if not np.array_equal(frequency, np.asarray(PRIMARY_FREQUENCY_HZ)):
            raise ValueError("reference result does not use the primary frequency grid")
        if tuple(self.curve_keys) != PRIMARY_TARGETS:
            raise ValueError("reference result curve order differs from Plan L")
        shape = (len(PRIMARY_TARGETS), frequency.size)
        phase = np.asarray(self.phase_velocity, dtype=float)
        gradient = np.asarray(self.phase_gradient_m, dtype=float)
        hessian = np.asarray(self.phase_hessian_m, dtype=float)
        if phase.shape != shape:
            raise ValueError("reference phase array has an inconsistent shape")
        if gradient.ndim != 3 or gradient.shape[:2] != shape:
            raise ValueError("reference gradient array has an inconsistent shape")
        n_params = gradient.shape[2]
        if hessian.shape != shape + (n_params, n_params):
            raise ValueError("reference Hessian array has an inconsistent shape")
        if len(self.exact_by_wave) != 2 or len(self.diagnostics_by_curve) != 4:
            raise ValueError("reference result requires two waves and four diagnostics")

    @property
    def n_params(self) -> int:
        return int(self.phase_gradient_m.shape[-1])


@dataclass(frozen=True)
class DepthConvergenceResult:
    passed: bool
    reference_depths_km: tuple[float, ...]
    exact_root_failure_count: int
    branch_identity_disagreement_count: int
    nonfinite_derivative_count: int
    phase1_trace_mismatch_count: int
    taylor2_prediction_difference_over_sigma_p50: float
    taylor2_prediction_difference_over_sigma_p90: float
    taylor2_prediction_difference_over_sigma_p99: float
    taylor2_prediction_difference_over_sigma_maximum: float
    maximum_phase_velocity_relative_difference_2400_2600: float
    maximum_adjacent_gap_absolute_difference_2400_2600: float
    gradient_relative_difference_by_curve: dict[str, float]
    hessian_relative_difference_by_curve: dict[str, float]
    phase1_trace_maximum_relative_difference_by_depth: dict[str, float]


def build_plan_l_objects(
    bottom_depth_km: float = ARTIFACT_REFERENCE_DEPTH_KM,
    prior_manifest: LongPeriodTaylorPriorManifest = LongPeriodTaylorPriorManifest(),
):
    reference = shallow_reference(
        max_spacing_km=REFERENCE_MAX_SPACING_KM,
        max_depth_km=float(bottom_depth_km),
    )
    basis = shallow_basis(reference)
    prior = PriorCovariance(basis, **prior_manifest.prior_kwargs())
    return reference, basis, prior


def plan_l_dataset() -> DatasetSpec:
    data = []
    for wave, root_order in PRIMARY_TARGETS:
        display_wave = wave.capitalize()
        data.extend(
            Datum(display_wave, root_order, 1.0 / frequency)
            for frequency in PRIMARY_FREQUENCY_HZ
        )
    return DatasetSpec(data)


def _moho_depth(reference) -> float:
    discontinuities = np.flatnonzero(reference.is_discontinuity)
    if discontinuities.size < 2:
        raise RuntimeError("ocean-free PREM reference has no crust-mantle interface")
    return float(reference.depths[int(discontinuities[1])])


def collect_reference_depth(
    bottom_depth_km: float,
    *,
    processes: int = 1,
    cache_dir=None,
    engine_kwargs=None,
) -> LongPeriodReferenceDepthResult:
    """Collect target Hessian jets and root-0--2 identity diagnostics."""

    if float(bottom_depth_km) not in REFERENCE_DEPTHS_KM:
        raise ValueError("bottom depth must be one of 2200/2400/2600 km")
    reference, basis, prior = build_plan_l_objects(bottom_depth_km)
    kwargs = {
        "max_steps": 6,
        "tolerance": 1.0e-9,
        "hessian_max_levels": 5,
        "hessian_tolerance": 1.0e-6,
    }
    kwargs.update(engine_kwargs or {})
    phase1 = run_phase1(
        reference,
        basis,
        plan_l_dataset(),
        processes=int(processes),
        engine_kwargs=kwargs,
        cache_dir=cache_dir,
        reduced_hessian=True,
        ad_kernel_every=0,
    )
    n_curve = len(PRIMARY_TARGETS)
    n_frequency = len(PRIMARY_FREQUENCY_HZ)
    phase = phase1.c0.reshape(n_curve, n_frequency)
    gradient = phase1.J.reshape(n_curve, n_frequency, basis.n_params)
    hessian = phase1.H.reshape(
        n_curve, n_frequency, basis.n_params, basis.n_params
    )
    exact_by_wave = []
    diagnostic_map = {}
    zero = np.zeros((1, basis.n_params))
    for wave in ("rayleigh", "love"):
        exact = collect_exact_mode_trace_ensemble(
            basis,
            zero,
            wave.capitalize(),
            np.asarray(PRIMARY_FREQUENCY_HZ),
            root_orders=(0, 1, 2),
            processes=1,
            engine_kwargs=kwargs,
        )
        exact_by_wave.append(exact)
        for target_root in (0, 1):
            diagnostic_map[(wave, target_root)] = exact.diagnostics(
                target_root, _moho_depth(reference)
            )
    for curve_index, (wave, root_order) in enumerate(PRIMARY_TARGETS):
        ensemble = exact_by_wave[0 if wave == "rayleigh" else 1]
        root_index = ensemble.root_orders.index(root_order)
        exact_phase = ensemble.phase_velocity[0, root_index]
        with np.errstate(divide="ignore", invalid="ignore"):
            difference = np.abs(phase[curve_index] / exact_phase - 1.0)
        if np.any(~np.isfinite(difference)) or np.max(difference) > (
            REFERENCE_PHASE_TRACE_RELATIVE_TOLERANCE
        ):
            raise RuntimeError(
                f"{wave}{root_order} Phase-1/reference trace mismatch: "
                f"{np.nanmax(difference):.3e}"
            )
    return LongPeriodReferenceDepthResult(
        bottom_depth_km=float(bottom_depth_km),
        frequency_hz=np.asarray(PRIMARY_FREQUENCY_HZ),
        curve_keys=PRIMARY_TARGETS,
        phase_velocity=phase,
        phase_gradient_m=gradient,
        phase_hessian_m=hessian,
        phase1_relative_error=phase1.rel_err.reshape(n_curve, n_frequency),
        phase1_hessian_asymmetry=phase1.asymmetry.reshape(n_curve, n_frequency),
        exact_by_wave=tuple(exact_by_wave),
        diagnostics_by_curve=tuple(
            diagnostic_map[key] for key in PRIMARY_TARGETS
        ),
        reference_fingerprint=reference_fingerprint(reference),
        basis_fingerprint=basis_fingerprint(basis),
        prior_fingerprint=prior_fingerprint(prior),
        basis_description=basis.describe(),
        prior_description=prior.describe(),
    )


def _relative_array_difference(left, right) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    denominator = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / denominator)


def _ordered_gap(phase, target_root: int, neighbor_root: int):
    target = np.asarray(phase[target_root], dtype=float)
    neighbor = np.asarray(phase[neighbor_root], dtype=float)
    numerator = (
        target - neighbor if target_root > neighbor_root else neighbor - target
    )
    denominator = 0.5 * (target + neighbor)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def _branch_identity_disagreements(left, right) -> int:
    disagreements = 0
    for left_wave, right_wave in zip(left.exact_by_wave, right.exact_by_wave):
        for frequency_index in range(len(PRIMARY_FREQUENCY_HZ)):
            overlap = np.empty((3, 3))
            for left_root in range(3):
                for right_root in range(3):
                    left_profile = left_wave.profiles[0, left_root, frequency_index]
                    right_profile = right_wave.profiles[0, right_root, frequency_index]
                    try:
                        overlap[left_root, right_root] = compute_mode_shape_overlap(
                            left_profile, right_profile
                        )
                    except (TypeError, ValueError):
                        overlap[left_root, right_root] = np.nan
            if np.any(~np.isfinite(overlap)):
                disagreements += 3
                continue
            assignment = compute_character_assignment(overlap)
            disagreements += int(np.count_nonzero(assignment != np.arange(3)))
    return disagreements


def evaluate_depth_convergence(
    results,
    *,
    gate: LongPeriodTaylorGate = LongPeriodTaylorGate(),
    response_kind_by_wave=None,
    cohort_builder=build_radial_prior_cohort,
    calibration_seed: int = CALIBRATION_SEED,
) -> DepthConvergenceResult:
    """Apply the pre-artifact 2400-vs-2600 km prediction budget."""

    values = tuple(results)
    if tuple(value.bottom_depth_km for value in values) != REFERENCE_DEPTHS_KM:
        raise ValueError("depth results must be ordered as 2200/2400/2600 km")
    shallow, reference, deep = values
    root_failures = sum(
        int(
            np.count_nonzero(
                ~(ensemble.success & ensemble.trace_continuity)
            )
        )
        for value in values
        for ensemble in value.exact_by_wave
    )
    branch_disagreements = _branch_identity_disagreements(
        shallow, reference
    ) + _branch_identity_disagreements(reference, deep)
    nonfinite_derivatives = sum(
        int(np.count_nonzero(~np.isfinite(value.phase_gradient_m)))
        + int(np.count_nonzero(~np.isfinite(value.phase_hessian_m)))
        for value in values
    )
    trace_differences = {}
    trace_mismatch_count = 0
    for value in values:
        differences = []
        for curve_index, (wave, root_order) in enumerate(PRIMARY_TARGETS):
            ensemble = value.exact_by_wave[0 if wave == "rayleigh" else 1]
            root_index = ensemble.root_orders.index(root_order)
            exact = ensemble.phase_velocity[0, root_index]
            with np.errstate(divide="ignore", invalid="ignore"):
                relative = np.abs(value.phase_velocity[curve_index] / exact - 1.0)
            differences.extend(relative.tolist())
            trace_mismatch_count += int(
                np.count_nonzero(
                    ~np.isfinite(relative)
                    | (relative > REFERENCE_PHASE_TRACE_RELATIVE_TOLERANCE)
                )
            )
        trace_differences[f"{value.bottom_depth_km:g}"] = float(
            np.nanmax(differences)
        )

    _, basis, prior = build_plan_l_objects(ARTIFACT_REFERENCE_DEPTH_KM)
    cohort = cohort_builder(
        basis,
        prior.chol(),
        split_kind="calibration",
        seed=calibration_seed,
    )
    selected = np.isclose(
        cohort.target_maximum_anomaly, 0.08, rtol=0.0, atol=2.0e-12
    )
    models = cohort.delta_m[selected]

    def predict_phase_velocity(value):
        linear = value.phase_velocity[None, :, :] + np.einsum(
            "mP,cfP->mcf", models, value.phase_gradient_m, optimize=True
        )
        quadratic = 0.5 * np.einsum(
            "mP,cfPQ,mQ->mcf",
            models,
            value.phase_hessian_m,
            models,
            optimize=True,
        )
        return linear + quadratic

    def predict_mixed_response(value):
        predictions = []
        identity = np.eye(value.n_params)
        for curve_index, (wave, root_order) in enumerate(PRIMARY_TARGETS):
            curve = build_long_period_taylor_curve(
                wave=wave,
                root_order=root_order,
                frequency_hz=value.frequency_hz,
                reference_phase_velocity=value.phase_velocity[curve_index],
                phase_gradient_m=value.phase_gradient_m[curve_index],
                phase_hessian_m=value.phase_hessian_m[curve_index],
                prior_cholesky=identity,
                response_kind=response_kind_by_wave[wave],
            )
            prediction, _, _ = curve.evaluate_z_batch(models, order=2)
            predictions.append(prediction)
        return np.stack(predictions, axis=1)

    if response_kind_by_wave is None:
        prediction_reference = predict_phase_velocity(reference)
        prediction_deep = predict_phase_velocity(deep)
    else:
        required_waves = {"rayleigh", "love"}
        if set(response_kind_by_wave) != required_waves:
            raise ValueError("response_kind_by_wave must define Rayleigh and Love")
        prediction_reference = predict_mixed_response(reference)
        prediction_deep = predict_mixed_response(deep)
    sigma = gate.relative_observation_sigma * prediction_reference
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_difference = np.abs(
            prediction_reference - prediction_deep
        ) / sigma
    finite_difference = normalized_difference[np.isfinite(normalized_difference)]
    if finite_difference.size:
        q50, q90, q99, qmax = (
            float(np.quantile(finite_difference, value))
            for value in (0.50, 0.90, 0.99, 1.0)
        )
    else:
        q50 = q90 = q99 = qmax = float("inf")

    exact_reference = np.stack(
        [ensemble.phase_velocity[0] for ensemble in reference.exact_by_wave]
    )
    exact_deep = np.stack(
        [ensemble.phase_velocity[0] for ensemble in deep.exact_by_wave]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_relative = np.abs(exact_reference / exact_deep - 1.0)
    maximum_phase_relative = float(np.nanmax(phase_relative))
    gaps_reference = []
    gaps_deep = []
    for wave_index in range(2):
        phase_ref = exact_reference[wave_index]
        phase_deep = exact_deep[wave_index]
        for target, neighbor in ((0, 1), (1, 0), (1, 2)):
            gaps_reference.append(_ordered_gap(phase_ref, target, neighbor))
            gaps_deep.append(_ordered_gap(phase_deep, target, neighbor))
    maximum_gap_difference = float(
        np.nanmax(np.abs(np.asarray(gaps_reference) - np.asarray(gaps_deep)))
    )
    gradient_difference = {}
    hessian_difference = {}
    for curve_index, key in enumerate(PRIMARY_TARGETS):
        label = f"{key[0]}{key[1]}"
        gradient_difference[label] = _relative_array_difference(
            reference.phase_gradient_m[curve_index],
            deep.phase_gradient_m[curve_index],
        )
        hessian_difference[label] = _relative_array_difference(
            reference.phase_hessian_m[curve_index],
            deep.phase_hessian_m[curve_index],
        )
    passed = (
        root_failures == 0
        and branch_disagreements == 0
        and nonfinite_derivatives == 0
        and trace_mismatch_count == 0
        and np.isfinite(q99)
        and q99
        <= gate.reference_truncation_error_over_sigma_p99_maximum
    )
    return DepthConvergenceResult(
        passed=bool(passed),
        reference_depths_km=REFERENCE_DEPTHS_KM,
        exact_root_failure_count=root_failures,
        branch_identity_disagreement_count=branch_disagreements,
        nonfinite_derivative_count=nonfinite_derivatives,
        phase1_trace_mismatch_count=trace_mismatch_count,
        taylor2_prediction_difference_over_sigma_p50=q50,
        taylor2_prediction_difference_over_sigma_p90=q90,
        taylor2_prediction_difference_over_sigma_p99=q99,
        taylor2_prediction_difference_over_sigma_maximum=qmax,
        maximum_phase_velocity_relative_difference_2400_2600=(
            maximum_phase_relative
        ),
        maximum_adjacent_gap_absolute_difference_2400_2600=(
            maximum_gap_difference
        ),
        gradient_relative_difference_by_curve=gradient_difference,
        hessian_relative_difference_by_curve=hessian_difference,
        phase1_trace_maximum_relative_difference_by_depth=trace_differences,
    )


def reference_result_arrays(value: LongPeriodReferenceDepthResult) -> dict:
    """Return the pickle-free arrays persisted by the offline driver."""

    arrays = {
        "bottom_depth_km": np.asarray(value.bottom_depth_km),
        "frequency_hz": value.frequency_hz,
        "phase_velocity": value.phase_velocity,
        "phase_gradient_m": value.phase_gradient_m,
        "phase_hessian_m": value.phase_hessian_m,
        "phase1_relative_error": value.phase1_relative_error,
        "phase1_hessian_asymmetry": value.phase1_hessian_asymmetry,
    }
    for wave_index, ensemble in enumerate(value.exact_by_wave):
        prefix = ("rayleigh", "love")[wave_index]
        arrays[f"{prefix}_root_orders"] = np.asarray(ensemble.root_orders)
        arrays[f"{prefix}_exact_phase_velocity"] = ensemble.phase_velocity
        arrays[f"{prefix}_exact_group_velocity"] = ensemble.group_velocity
        arrays[f"{prefix}_exact_kernels"] = ensemble.kernels
        arrays[f"{prefix}_exact_success"] = ensemble.success
        arrays[f"{prefix}_trace_continuity"] = ensemble.trace_continuity
    diagnostic_names = (
        "adjacent_relative_gap",
        "eigenfunction_overlap",
        "kernel_overlap",
        "crustal_energy_fraction",
        "energy_centroid_jump_km",
        "group_velocity_relative_jump",
        "trace_agreement",
        "exact_success",
    )
    for key, diagnostics in zip(PRIMARY_TARGETS, value.diagnostics_by_curve):
        prefix = f"{key[0]}{key[1]}"
        for name in diagnostic_names:
            arrays[f"{prefix}_{name}"] = getattr(diagnostics, name)
    return arrays


__all__ = [
    "DepthConvergenceResult",
    "LongPeriodReferenceDepthResult",
    "REFERENCE_PHASE_TRACE_RELATIVE_TOLERANCE",
    "build_plan_l_objects",
    "collect_reference_depth",
    "evaluate_depth_convergence",
    "plan_l_dataset",
    "reference_result_arrays",
]
