"""Offline exact traces and physical mode diagnostics for Plan L."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

import disper26py as dp

from .derivatives import gradient_log, project_gradient, reduced_result_to_node_gradient
from .engine import Datum, ForwardEngine
from .long_period_taylor_mode import (
    ModeShapeProfile,
    compute_mode_character_diagnostics,
    mode_shape_from_solver_output,
)


def _trace_root_with_partial_recovery(
    engine: ForwardEngine,
    model,
    wave: str,
    root_order: int,
    frequency: np.ndarray,
    *,
    minimum_trace_level: int = 0,
):
    """Trace one root, retaining frequencies around a local branch failure.

    ``ForwardEngine.trace_dispersion_curve`` deliberately raises when any
    frequency cannot be branch-confirmed.  That all-or-nothing contract is
    appropriate for an exact likelihood curve, but not for an offline mask:
    discarding the whole adjacent-mode curve turns one local osculation into
    missing character evidence at every frequency.  On a trace exception we
    therefore repeat the offline collection point by point, retaining every
    independently confirmed solution and restarting continuation after a
    failed point.
    """

    frequencies = np.asarray(frequency, dtype=float)
    try:
        solutions = list(
            engine.trace_dispersion_curve(
                model,
                str(wave),
                int(root_order),
                frequencies,
                minimum_trace_level=int(minimum_trace_level),
            )
        )
        if len(solutions) != frequencies.size:
            raise RuntimeError(
                "trace_dispersion_curve returned an incomplete solution list"
            )
        return solutions, ()
    except Exception as error:
        errors = [
            f"root {root_order} trace recovery: {type(error).__name__}: {error}"
        ]

    solutions = [None] * frequencies.size
    previous = None
    previous_frequency = None
    for frequency_index, frequency_hz in enumerate(frequencies):
        hint = None
        if previous is not None:
            hint = ForwardEngine._group_predict(
                previous_frequency,
                previous.c,
                previous.U,
                float(frequency_hz),
            )
            if not np.isfinite(hint) or hint <= 0.0:
                hint = previous.c
        datum = Datum(str(wave), int(root_order), 1.0 / float(frequency_hz))
        try:
            solution = engine.phase_velocity(
                model,
                datum,
                c_hint=hint,
                minimum_trace_level=int(minimum_trace_level),
            )
        except Exception as error:
            errors.append(
                f"root {root_order} f={frequency_hz:.6f}: "
                f"{type(error).__name__}: {error}"
            )
            # Do not carry a predictor across an unresolved crossing.  The
            # next frequency restarts from mode-count bisection.
            previous = None
            previous_frequency = None
            continue
        solutions[frequency_index] = solution
        previous = solution
        previous_frequency = float(frequency_hz)
    return solutions, tuple(errors)


def _density_at_eigenfunction_depth(model, engine, depth_km):
    layers, thicknesses, halfspace = model.refined_layers(engine.scan_level)
    boundaries = np.cumsum(np.asarray(thicknesses, dtype=float))
    depth = np.asarray(depth_km, dtype=float)
    indices = np.searchsorted(boundaries, depth, side="right")
    density = np.empty(depth.size)
    for index, layer_index in enumerate(indices):
        density[index] = (
            float(layers[layer_index].rho)
            if layer_index < len(layers)
            else float(halfspace.rho)
        )
    return density


@dataclass(frozen=True)
class ExactModeTraceModel:
    phase_velocity: np.ndarray
    group_velocity: np.ndarray
    kernels: np.ndarray
    profiles: np.ndarray
    success: np.ndarray
    trace_continuity: np.ndarray
    errors: tuple[str, ...]


def _collect_one_exact_model(task) -> ExactModeTraceModel:
    basis, delta_m, wave, frequency, root_orders, engine_kwargs = task
    model = basis.m_to_forward_model(np.asarray(delta_m, dtype=float))
    engine = ForwardEngine(**engine_kwargs)
    n_root = len(root_orders)
    n_frequency = frequency.size
    phase = np.full((n_root, n_frequency), np.nan)
    group = np.full((n_root, n_frequency), np.nan)
    kernels = np.full((n_root, n_frequency, basis.n_params), np.nan)
    profiles = np.empty((n_root, n_frequency), dtype=object)
    profiles[:] = None
    success = np.zeros((n_root, n_frequency), dtype=bool)
    trace_continuity = np.zeros((n_root, n_frequency), dtype=bool)
    errors = []
    model_args = model.to_params()
    for root_index, root_order in enumerate(root_orders):
        data = [Datum(wave, int(root_order), 1.0 / value) for value in frequency]
        solutions, trace_errors = _trace_root_with_partial_recovery(
            engine,
            model,
            wave,
            int(root_order),
            frequency,
            minimum_trace_level=int(engine.hessian_max_levels),
        )
        errors.extend(trace_errors)
        if not trace_errors:
            trace_continuity[root_index, :] = True
        for frequency_index, (datum, solution) in enumerate(zip(data, solutions)):
            if solution is None:
                continue
            try:
                result = dp.compute_kernels_variational_grid(
                    *model_args,
                    float(frequency[frequency_index]),
                    float(solution.c),
                    datum.wave,
                    int(engine.hessian_max_levels),
                    float(engine.hessian_tolerance),
                    float(engine.initial_root_window_frac),
                    str(engine.hessian_backend),
                    phase_velocity_roots=solution.romberg_roots,
                    phase_accepted_level=int(solution.level),
                    **model.refinement_kwargs(),
                )
                gradient_stiffness = reduced_result_to_node_gradient(
                    model, result["gradient_node"]
                )
                kernel = project_gradient(
                    gradient_log(model, gradient_stiffness),
                    basis,
                    model=model,
                )
                eigenfunction = engine.eigenfunctions(model, datum, solution.c)
                density = _density_at_eigenfunction_depth(
                    model, engine, eigenfunction["depth"]
                )
                profile = mode_shape_from_solver_output(
                    eigenfunction, datum.wave, density
                )
            except Exception as error:
                errors.append(
                    f"root {root_order} f={frequency[frequency_index]:.6f}: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            phase[root_index, frequency_index] = solution.c
            group[root_index, frequency_index] = solution.U
            kernels[root_index, frequency_index] = kernel
            profiles[root_index, frequency_index] = profile
            success[root_index, frequency_index] = True
    return ExactModeTraceModel(
        phase_velocity=phase,
        group_velocity=group,
        kernels=kernels,
        profiles=profiles,
        success=success,
        trace_continuity=trace_continuity,
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class ExactModeTraceEnsemble:
    """Modes 0--2 for one wave over a fixed model/frequency ensemble."""

    frequency_hz: np.ndarray
    root_orders: tuple[int, ...]
    phase_velocity: np.ndarray
    group_velocity: np.ndarray
    kernels: np.ndarray
    profiles: np.ndarray
    success: np.ndarray
    trace_continuity: np.ndarray
    errors: tuple[tuple[str, ...], ...]

    def diagnostics(
        self,
        root_order: int,
        moho_depth_km: float,
        *,
        model_neighbor_pairs=(),
    ):
        if int(root_order) not in self.root_orders:
            raise KeyError("requested root order was not collected")
        root_index = self.root_orders.index(int(root_order))
        return compute_mode_character_diagnostics(
            self.phase_velocity,
            self.group_velocity,
            self.profiles,
            self.kernels,
            self.success & self.trace_continuity,
            target_root_order=root_index,
            moho_depth_km=moho_depth_km,
            model_neighbor_pairs=model_neighbor_pairs,
        )


def collect_exact_mode_trace_ensemble(
    basis,
    delta_m,
    wave: str,
    frequency_hz,
    *,
    root_orders=(0, 1, 2),
    processes: int = 1,
    engine_kwargs=None,
) -> ExactModeTraceEnsemble:
    """Collect confirmed roots, U, cheap variational kernels, and eigenfunctions."""

    models = np.asarray(delta_m, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    roots = tuple(int(value) for value in root_orders)
    if models.ndim != 2 or models.shape[1] != basis.n_params:
        raise ValueError("delta_m must have shape (model, parameter)")
    if (
        frequency.ndim != 1
        or frequency.size < 2
        or np.any(~np.isfinite(frequency))
        or np.any(frequency <= 0.0)
        or np.any(np.diff(frequency) <= 0.0)
    ):
        raise ValueError("frequency_hz must be finite, positive, and increasing")
    if len(roots) < 2 or len(set(roots)) != len(roots) or min(roots) < 0:
        raise ValueError("root_orders must contain distinct non-negative modes")
    if int(processes) != processes or processes < 1:
        raise ValueError("processes must be a positive integer")
    kwargs = {
        "max_steps": 6,
        "tolerance": 1.0e-9,
        "hessian_max_levels": 5,
        "hessian_tolerance": 1.0e-6,
    }
    kwargs.update(engine_kwargs or {})
    tasks = [
        (basis, model.copy(), str(wave), frequency.copy(), roots, kwargs)
        for model in models
    ]
    if int(processes) == 1:
        results = [_collect_one_exact_model(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(processes)) as executor:
            results = list(executor.map(_collect_one_exact_model, tasks))
    return ExactModeTraceEnsemble(
        frequency_hz=frequency.copy(),
        root_orders=roots,
        phase_velocity=np.stack([value.phase_velocity for value in results]),
        group_velocity=np.stack([value.group_velocity for value in results]),
        kernels=np.stack([value.kernels for value in results]),
        profiles=np.stack([value.profiles for value in results]),
        success=np.stack([value.success for value in results]),
        trace_continuity=np.stack(
            [value.trace_continuity for value in results]
        ),
        errors=tuple(value.errors for value in results),
    )


__all__ = [
    "ExactModeTraceEnsemble",
    "ExactModeTraceModel",
    "collect_exact_mode_trace_ensemble",
]
