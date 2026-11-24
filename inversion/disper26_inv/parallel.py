"""Process-pool evaluation of per-datum forward/kernel/Hessian tasks.

The bindings hold the GIL (probe 2026-07-12: 4-thread speedup x0.6), so
parallelism uses multiprocessing. Payloads are plain arrays/tuples; pybind
objects (PhysicalParameters, SensitivityKernel) never cross the pool boundary
— workers convert results to numpy via derivatives helpers before returning.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from .derivatives import (
    basis_node_jacobian,
    hessian_result_to_node_arrays,
    kernels_to_stiffness_gradient,
    reduced_result_to_node_gradient,
)
from .depth_hessian import make_depth_backend
from .engine import BranchError, Datum, ForwardEngine
from .model import ForwardGridModel

_WORKER_STATE = {}


def _model_payload(model: ForwardGridModel) -> dict:
    return {
        "depths": model.depths,
        "rho": model.rho,
        "vph": model.vph,
        "vpv": model.vpv,
        "vsh": model.vsh,
        "vsv": model.vsv,
        "eta": model.eta,
        "is_discontinuity": model.is_discontinuity,
        "name": model.name,
        "refinement_policy": model.refinement_policy.describe(),
    }


def _init_worker(model_payload: dict, engine_kwargs: dict, depth_config: dict = None, basis=None):
    _WORKER_STATE.clear()
    _WORKER_STATE["model"] = ForwardGridModel(**model_payload)
    _WORKER_STATE["engine"] = ForwardEngine(**engine_kwargs)
    _WORKER_STATE["depth_backend"] = make_depth_backend(
        _WORKER_STATE["engine"], depth_config, basis=basis
    )
    _WORKER_STATE["basis"] = basis


def _eval_datum(task):
    """(wave, branch, period, c_hint, want_hessian, want_kernels, reduced_hessian,
    want_ad_kernels) -> plain arrays.

    Branch failures are returned as {"failed": label} instead of raising, so a
    single osculation event does not abort a whole pool map; callers decide.

    The optional kernel pass returns ``g_a_ad`` and, for a depth basis,
    ``g_inv_kernel``. A flat interface supplies ``g_variational_depth`` from
    its boundary kernel. A spherical interface supplies ``g_interface_depth``
    from AD through EFT, reusing the depth kernel when it is available.
    """
    datum = Datum(*task[:3])
    try:
        return _compute_datum(datum, *task[3:])
    except BranchError as exc:
        return {"failed": datum.label(), "error": f"{type(exc).__name__}: {exc}"}


def _compute_datum(datum, c_hint, want_hessian, want_kernels, reduced_hessian,
                   want_ad_kernels):
    model = _WORKER_STATE["model"]
    engine = _WORKER_STATE["engine"]
    depth_backend = _WORKER_STATE.get("depth_backend")
    minimum_trace_level = engine.hessian_max_levels if want_hessian else 0
    if (want_kernels or want_hessian) and want_ad_kernels:
        solution, kernels = engine.phase_velocity_with_kernels(
            model,
            datum,
            c_hint=c_hint,
            minimum_trace_level=minimum_trace_level,
            include_interface_depths=depth_backend is not None or model.refinement_policy.kind == "flat",
        )
    else:
        solution = engine.phase_velocity(
            model,
            datum,
            c_hint=c_hint,
            minimum_trace_level=minimum_trace_level,
        )
        kernels = None
    out = {
        "c": solution.c,
        "U": solution.U,
        "rel_err": solution.rel_err,
        "branch_confirmation": solution.confirmation,
    }
    if kernels is not None:
        out["g_a_ad"] = kernels_to_stiffness_gradient(kernels, model.n_nodes)
        if depth_backend is not None:
            out["g_inv_kernel"] = depth_backend.gradient_row(model, kernels, solution)
    if (want_kernels or want_hessian) and depth_backend is not None and hasattr(
        depth_backend, "depth_gradient_variational"
    ):
        # The AD-free route (B3): 2-6 ms, computed whether or not this datum
        # also ran the AD pass, since it is what lets the Jacobian's depth
        # component stop needing ad_kernel_every at all. Gated the same as the
        # AD kernel pass above (want_kernels or want_hessian) rather than
        # unconditionally: landing.py calls this with want_hessian=False and
        # want_kernels=False for plain forward evaluations during a line
        # search, and those must not pay for a gradient nobody asked for.
        out["g_variational_depth"] = depth_backend.depth_gradient_variational(
            model,
            datum,
            solution.c,
            solution.level,
            phase_velocity_roots=solution.romberg_roots,
        )
    elif (want_kernels or want_hessian) and hasattr(depth_backend, "depth_gradient"):
        if kernels is None:
            out["g_interface_depth"] = depth_backend.depth_gradient(
                model, datum, solution.c, solution.level, solution.romberg_roots
            )
        else:
            out["g_interface_depth"] = out["g_inv_kernel"][depth_backend.depth_parameter_index]
    if want_hessian:
        if reduced_hessian:
            J = _WORKER_STATE.get("J_reduced")
            if J is None:
                # J only depends on the model's current node parameters (not
                # on any per-datum quantity), so build it lazily once per
                # worker when the reduced route is actually requested.
                J = basis_node_jacobian(model, _WORKER_STATE["basis"])
                _WORKER_STATE["J_reduced"] = J
            reduced = engine.datum_hessian_reduced(
                model,
                datum,
                solution.c,
                J,
                phase_velocity_roots=solution.romberg_roots,
                phase_accepted_level=solution.level,
            )
            out["g_a_variational"] = reduced_result_to_node_gradient(model, reduced["gradient_node"])
            out["hessian_reduced"] = reduced["hessian_reduced"]
            out["asymmetry_rel"] = reduced["asymmetry_rel"]
        else:
            hess = engine.datum_hessian(
                model,
                datum,
                solution.c,
                phase_velocity_roots=solution.romberg_roots,
                phase_accepted_level=solution.level,
            )
            g_a, H_a = hessian_result_to_node_arrays(model, hess)
            out["g_a_variational"] = g_a
            out["H_a"] = H_a
            out["asymmetry_rel"] = hess["asymmetry_rel"]
        if depth_backend is not None:
            depth_result = depth_backend.interface_hessian_row(
                model, datum, solution.c, solution=solution
            )
            out["depth_hessian_column"] = depth_result.column
            out["depth_hessian_meta"] = depth_result.as_dict()
    return out


def map_over_data(
    model: ForwardGridModel,
    data,
    c_hints=None,
    want_hessian: bool = True,
    want_kernels: bool = True,
    processes: int = 1,
    engine_kwargs: dict = None,
    depth_config: dict = None,
    reduced_hessian: bool = False,
    basis=None,
    ad_kernel_every: int = 1,
):
    """Evaluate (c[, kernels][, Hessian]) for every datum; preserves order.

    Branch failures appear as {"failed": label, "error": ...} entries.
    Other exceptions propagate in both serial and pool execution.

    ``ad_kernel_every`` thins the AD Romberg kernel pass: 1 (default) runs it
    for every datum as before, N runs it for every Nth, 0 skips it entirely.
    Skipped data omit ``g_a_ad`` and ``g_inv_kernel``. For a depth basis the
    default backend still returns the interface gradient for its geometry.

    ``reduced_hessian`` (opt-in, default off): when ``want_hessian`` and this
    are both true, the C++ side seeds ``basis``'s directions directly
    (``ForwardEngine.datum_hessian_reduced``) instead of every node-VTI unit
    vector, requiring ``basis`` to be supplied. Results carry
    ``hessian_reduced`` (P x P, still needs
    ``derivatives.hessian_curvature_only`` added and symmetrizing -- see
    ``derivatives.datum_hessian_inv_reduced``) instead of the full ``H_a``
    (n_nodes*6 x n_nodes*6). ``run_phase1`` enables this by default; direct
    ``map_over_data`` calls keep the opt-in default. Historical numerical
    parity and benchmark comparisons are retained in the research archive.
    """
    if reduced_hessian and basis is None:
        raise ValueError("reduced_hessian=True requires a basis")
    if ad_kernel_every < 0:
        raise ValueError("ad_kernel_every must be >= 0")
    engine_kwargs = engine_kwargs or {}
    tasks = [
        (
            d.wave,
            d.branch,
            d.period,
            None if c_hints is None else float(c_hints[i]),
            want_hessian,
            want_kernels,
            reduced_hessian,
            ad_kernel_every > 0 and i % ad_kernel_every == 0,
        )
        for i, d in enumerate(data)
    ]
    if processes <= 1:
        try:
            _init_worker(_model_payload(model), engine_kwargs, depth_config, basis)
            return [_eval_datum(t) for t in tasks]
        finally:
            _WORKER_STATE.clear()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(_model_payload(model), engine_kwargs, depth_config, basis),
    ) as pool:
        return pool.map(_eval_datum, tasks)


# -- forward-only sweeps: many perturbed models x fixed dataset ---------------

def _init_forward_worker(data_tuples, c_hints, engine_kwargs, curve_segments=None):
    _WORKER_STATE.clear()
    _WORKER_STATE["data"] = [Datum(w, b, p) for (w, b, p) in data_tuples]
    _WORKER_STATE["hints"] = c_hints
    _WORKER_STATE["engine"] = ForwardEngine(**engine_kwargs)
    _WORKER_STATE["curve_segments"] = curve_segments


def _eval_model(model_payload: dict) -> np.ndarray:
    """c for every datum on one perturbed model; NaN where the branch fails."""
    model = ForwardGridModel(**model_payload)
    engine = _WORKER_STATE["engine"]
    hints = _WORKER_STATE["hints"]
    out = np.full(len(_WORKER_STATE["data"]), np.nan)
    for i, datum in enumerate(_WORKER_STATE["data"]):
        hint = None if hints is None else float(hints[i])
        try:
            out[i] = engine.phase_velocity(model, datum, c_hint=hint).c
        except Exception:
            pass  # NaN marks branch failures; callers count them
    return out


def _eval_model_traced(model_payload: dict) -> np.ndarray:
    """Frequency-continuation trace of every curve on one perturbed model.

    Independent per-datum evaluation with fixed hints picks the wrong root at
    known osculation-adjacent points (L1 at 0.125 Hz) once the model moves;
    tracing each (wave, branch) curve in ascending frequency is the method
    plan's root policy and is used wherever a perturbed model is evaluated
    without a model-specific hint table.  Any curve failure marks the whole
    model as NaN, since a joint likelihood needs every datum.
    """
    model = ForwardGridModel(**model_payload)
    engine = _WORKER_STATE["engine"]
    try:
        return engine.trace_dataset(model, _WORKER_STATE["data"])
    except Exception:
        return np.full(len(_WORKER_STATE["data"]), np.nan)


def _eval_model_traced_partial(model_payload: dict) -> np.ndarray:
    """Trace curves independently and preserve successful curves.

    This is intended for offline construction of a fixed observation mask.
    A failure in a curve that will later be excluded must not erase exact
    values from the other curves.  Runtime joint-likelihood evaluation keeps
    using ``_eval_model_traced`` and its all-or-nothing semantics.
    """

    model = ForwardGridModel(**model_payload)
    engine = _WORKER_STATE["engine"]
    data = _WORKER_STATE["data"]
    groups = {}
    for index, datum in enumerate(data):
        groups.setdefault((datum.wave, int(datum.branch)), []).append(
            (1.0 / float(datum.period), index)
        )
    out = np.full(len(data), np.nan)
    for (wave, branch), entries in groups.items():
        entries.sort(key=lambda item: item[0])
        frequencies = [item[0] for item in entries]
        try:
            solutions = engine.trace_dispersion_curve(model, wave, branch, frequencies)
        except Exception:
            continue
        for (_, index), solution in zip(entries, solutions):
            out[index] = solution.c
    return out


def split_curve_segments(data, original_indices):
    """Split a masked dataset at curve changes and removed-index gaps.

    ``data`` contains the retained observations and ``original_indices`` maps
    them back to the unmasked dataset.  A removed frequency band terminates a
    continuation segment even when the observations on both sides have the
    same wave and branch.  This prevents an intentionally excluded
    osculation/escape band from erasing exact values in either safe segment.
    """

    data = list(data)
    original = np.asarray(original_indices, dtype=int)
    if original.shape != (len(data),):
        raise ValueError("original_indices must contain one index per datum")
    if original.size and (
        np.any(original < 0)
        or np.unique(original).size != original.size
        or np.any(np.diff(original) <= 0)
    ):
        raise ValueError("original_indices must be unique, non-negative, and increasing")
    if not data:
        return ()
    segments = []
    start = 0
    for index in range(1, len(data)):
        previous = data[index - 1]
        current = data[index]
        contiguous = original[index] == original[index - 1] + 1
        same_curve = (
            current.wave == previous.wave
            and int(current.branch) == int(previous.branch)
        )
        if not contiguous or not same_curve:
            segments.append(tuple(range(start, index)))
            start = index
    segments.append(tuple(range(start, len(data))))
    return tuple(segments)


def _eval_model_traced_segments(model_payload: dict) -> np.ndarray:
    """Trace precomputed contiguous curve segments independently."""

    model = ForwardGridModel(**model_payload)
    engine = _WORKER_STATE["engine"]
    data = _WORKER_STATE["data"]
    segments = _WORKER_STATE.get("curve_segments")
    if segments is None:
        raise RuntimeError("curve segments were not initialized")
    out = np.full(len(data), np.nan)
    for segment in segments:
        entries = sorted(
            ((1.0 / float(data[index].period), int(index)) for index in segment),
            key=lambda item: (item[0], item[1]),
        )
        first = data[entries[0][1]]
        frequencies = [entry[0] for entry in entries]
        try:
            solutions = engine.trace_dispersion_curve(
                model, first.wave, int(first.branch), frequencies
            )
        except Exception:
            continue
        for (_, index), solution in zip(entries, solutions):
            out[index] = solution.c
    return out


def map_over_models_traced(models, data, processes: int = 1, engine_kwargs: dict = None):
    """Trace the whole dataset on each perturbed model; returns (M, N) array."""
    engine_kwargs = engine_kwargs or {}
    data_tuples = [(d.wave, d.branch, d.period) for d in data]
    payloads = [_model_payload(m) for m in models]
    if processes <= 1:
        try:
            _init_forward_worker(data_tuples, None, engine_kwargs)
            rows = [_eval_model_traced(p) for p in payloads]
            return np.asarray(rows)
        finally:
            _WORKER_STATE.clear()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=processes,
        initializer=_init_forward_worker,
        initargs=(data_tuples, None, engine_kwargs),
    ) as pool:
        return np.asarray(pool.map(_eval_model_traced, payloads))


def map_over_models_traced_partial(
    models, data, processes: int = 1, engine_kwargs: dict = None
):
    """Trace each curve independently; failed curves are NaN, not whole rows.

    Use this only for offline mask/support calibration.  Production objectives
    require every retained datum and therefore use the ordinary traced path.
    """

    engine_kwargs = engine_kwargs or {}
    data_tuples = [(d.wave, d.branch, d.period) for d in data]
    payloads = [_model_payload(m) for m in models]
    if processes <= 1:
        try:
            _init_forward_worker(data_tuples, None, engine_kwargs)
            rows = [_eval_model_traced_partial(payload) for payload in payloads]
            return np.asarray(rows)
        finally:
            _WORKER_STATE.clear()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=processes,
        initializer=_init_forward_worker,
        initargs=(data_tuples, None, engine_kwargs),
    ) as pool:
        return np.asarray(pool.map(_eval_model_traced_partial, payloads))


def map_over_models_traced_segments(
    models,
    data,
    original_indices,
    processes: int = 1,
    engine_kwargs: dict = None,
):
    """Trace only contiguous retained curve segments for offline validation.

    Unlike :func:`map_over_models_traced_partial`, gaps in ``original_indices``
    split one wave/branch curve into independent continuations.  The function
    is deliberately offline-only; production likelihood evaluation still
    requires every observation in its fixed dataset.
    """

    engine_kwargs = engine_kwargs or {}
    data = list(data)
    segments = split_curve_segments(data, original_indices)
    data_tuples = [(datum.wave, datum.branch, datum.period) for datum in data]
    payloads = [_model_payload(model) for model in models]
    if processes <= 1:
        try:
            _init_forward_worker(data_tuples, None, engine_kwargs, segments)
            rows = [_eval_model_traced_segments(payload) for payload in payloads]
            return np.asarray(rows)
        finally:
            _WORKER_STATE.clear()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=processes,
        initializer=_init_forward_worker,
        initargs=(data_tuples, None, engine_kwargs, segments),
    ) as pool:
        return np.asarray(pool.map(_eval_model_traced_segments, payloads))


def map_over_models(models, data, c_hints=None, processes: int = 1, engine_kwargs: dict = None):
    """Evaluate the whole dataset on each perturbed model; returns (M, N) array."""
    engine_kwargs = engine_kwargs or {}
    data_tuples = [(d.wave, d.branch, d.period) for d in data]
    payloads = [_model_payload(m) for m in models]
    if processes <= 1:
        try:
            _init_forward_worker(data_tuples, c_hints, engine_kwargs)
            rows = [_eval_model(p) for p in payloads]
            return np.asarray(rows)
        finally:
            _WORKER_STATE.clear()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=processes,
        initializer=_init_forward_worker,
        initargs=(data_tuples, c_hints, engine_kwargs),
    ) as pool:
        return np.asarray(pool.map(_eval_model, payloads))
