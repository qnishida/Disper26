"""Phase orchestration and caching.

Phase 1 evaluates everything that is independent of the noise realization
(baseline c0, Jacobian, per-datum Hessians, surrogate, consistency stats) and
caches it to phase1.npz keyed by a configuration hash — the cache makes the
Stage A coverage test (30 noise realizations) affordable, since each
realization then only needs a MAP solve.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import DatasetSpec, ObservationCovariance
from .derivatives import (
    gradient_log,
    hessian_curvature_only,
    hessian_log,
    project_gradient,
    project_hessian,
)
from .depth_hessian import depth_backend_config
from .engine import ForwardEngine
from .linearized import eigen_analysis, gauss_newton, linearized_map
from .model import ForwardGridModel, InversionBasis
from .parallel import map_over_data
from .physics import check_stiffness_posdef
from .priors import PriorCovariance
from .surrogate import QuadraticSurrogate


def _config_hash(model: ForwardGridModel, basis: InversionBasis, dataset: DatasetSpec,
                 engine_kwargs: dict, depth_validation_step_km: float | None = None,
                 ad_kernel_every: int = 1) -> str:
    h = hashlib.sha256()
    for arr in (model.depths, model.rho, model.vph, model.vpv, model.vsh, model.vsv, model.eta):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps(model.refinement_policy.describe(), sort_keys=True).encode())
    h.update(np.ascontiguousarray(basis.inv_depths).tobytes())
    h.update(",".join(basis.param_names).encode())
    basis_description = basis.describe() if hasattr(basis, "describe") else {
        "type": type(basis).__name__,
        "param_names": list(basis.param_names),
    }
    h.update(json.dumps(basis_description, sort_keys=True).encode())
    if hasattr(basis, "depth_parameter_index"):
        # The depth backend is part of the Phase-1 cache contract.  Keep the
        # production step explicit so changing it cannot reuse stale columns.
        h.update(json.dumps({
            "depth_backend": "variational_ad",
            "step_km": 0.05,
            "validation_step_km": depth_validation_step_km,
            "convergence_tolerance": 2.0e-3,
        }, sort_keys=True).encode())
    h.update(json.dumps(dataset.to_json(), sort_keys=True).encode())
    # Root-traced Phase 1 is a different cache contract from the historical
    # independent-point evaluation.  Include the policy explicitly so stale
    # 2026-07-12 caches cannot silently bypass branch verification.
    cache_engine = dict(engine_kwargs)
    cache_engine.setdefault("initial_root_window_frac", 1e-3)
    cache_engine.setdefault("confirmation_max_scan_level", 5)
    cache_engine["phase1_root_trace_policy"] = "group_velocity_v1"
    h.update(json.dumps(cache_engine, sort_keys=True).encode())
    # Only perturbed away from the default so that caches written before this
    # knob existed stay valid. A thinned run has NaN in `consistency` where a
    # full one has numbers, so it must not reuse a full run's cache.
    if ad_kernel_every != 1:
        h.update(json.dumps({"ad_kernel_every": ad_kernel_every}).encode())
    return h.hexdigest()[:16]


@dataclass
class Phase1Result:
    c0: np.ndarray                 # (N,) baseline phase velocities
    J: np.ndarray                  # (N, n) Jacobian in inversion coordinates
    H: np.ndarray                  # (N, n, n) per-datum Hessians in inversion coords
    consistency: np.ndarray        # (N,) AD-vs-production (weak-form) gradient rel
                                    # diff; NaN where ad_kernel_every skipped the AD
                                    # cross-check for that datum
    asymmetry: np.ndarray          # (N,) pre-symmetrization residuals
    rel_err: np.ndarray            # (N,) Romberg extrapolation residuals
    config_hash: str = ""
    meta: dict = field(default_factory=dict)

    def surrogate(self) -> QuadraticSurrogate:
        return QuadraticSurrogate(self.c0, self.J, self.H)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            c0=self.c0,
            J=self.J,
            H=self.H,
            consistency=self.consistency,
            asymmetry=self.asymmetry,
            rel_err=self.rel_err,
            config_hash=np.array(self.config_hash),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: Path) -> "Phase1Result":
        arrays = np.load(path, allow_pickle=False)
        return cls(
            c0=arrays["c0"],
            J=arrays["J"],
            H=arrays["H"],
            consistency=arrays["consistency"],
            asymmetry=arrays["asymmetry"],
            rel_err=arrays["rel_err"],
            config_hash=str(arrays["config_hash"]),
            meta=json.loads(str(arrays["meta"])),
        )


def run_phase1(
    model0: ForwardGridModel,
    basis: InversionBasis,
    dataset: DatasetSpec,
    processes: int = 1,
    engine_kwargs: dict = None,
    cache_dir: Path = None,
    depth_validation_step_km: float | None = None,
    reduced_hessian: bool = True,
    ad_kernel_every: int = 1,
) -> Phase1Result:
    """Baseline evaluation: c0, J, per-datum H (parallel), with caching.

    ``reduced_hessian`` (default True) seeds the basis Jacobian's columns
    directly in the C++ grid Hessian instead of every node-VTI unit
    direction (16x-24x faster at production scale, machine-precision
    agreement with the non-reduced path -- see
    tests/REPORT_GRID_HESSIAN_REDUCED_BASIS_20260805.md). Set False to fall
    back to the original node-space path.

    ``ad_kernel_every`` (default 1) controls how often the AD Romberg kernel
    pass runs. **The Jacobian no longer comes from it.** Since the derivative
    fix in 986b360 put the AD and weak-form dc/dp paths within 1e-08 of each
    other, ``J`` is built from the weak-form gradient (the reduced Hessian
    call's ``gradient_node``) plus, where a depth column is configured, the
    variational K_d interface gradient (``VariationalAdDepthHessianBackend.
    depth_gradient_variational``, needs no AD at all). AD now exists only to
    populate ``consistency``: 1 runs it on every datum (full cross-check,
    matching behaviour before this change numerically to ~1e-08), N thins it
    to every Nth, 0 skips it entirely. ``consistency`` is NaN where AD was not
    run for a datum -- an unmeasured check must not read as a passing one.

    The default stays 1 deliberately: the AD pass remains the intended safety
    net while confidence in the switch builds, not something turned off by
    this change. Whoever calls run_phase1 dials ``ad_kernel_every`` down once
    ready to spend less of Phase 1's time on it -- the AD pass is 733 ms per
    datum against about 30-77 ms for weak-form + K_d, so an unthinned run no
    longer saves anything; the saving is realized only once thinning is turned
    on. The disagreements found across 2805 saved data points came in blocks
    contiguous in period, never as isolated points, so thinning keeps most of
    the detection.

    If a depth column is configured with a backend that lacks
    ``depth_gradient_variational`` (only the default 'variational_ad' backend has
    it; 'fd' and the plain nested-AD backend do not), the depth
    component falls back to the AD kernel's ``Depth`` entry, and thinning that
    away leaves it undefined -- ``RuntimeError`` rather than a silently wrong
    Jacobian row.
    """
    engine_kwargs = engine_kwargs or {}
    cfg_hash = _config_hash(
        model0, basis, dataset, engine_kwargs, depth_validation_step_km,
        ad_kernel_every,
    )
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"phase1_{cfg_hash}.npz"
        if cache_path.exists():
            result = Phase1Result.load(cache_path)
            if result.config_hash == cfg_hash and result.meta.get("root_trace"):
                return result

    # Build one branch-verified reference trace before the per-datum process
    # pool.  The pool keeps Phase 1 parallel while every kernels/Hessian call
    # starts from a root on the intended branch.
    trace_engine = ForwardEngine(**engine_kwargs)
    root_hints = trace_engine.trace_dataset(model0, dataset.data)
    depth_config = depth_backend_config(
        basis, validation_step_km=depth_validation_step_km
    )
    results = map_over_data(
        model0, dataset.data, c_hints=root_hints, want_hessian=True,
        processes=processes, engine_kwargs=engine_kwargs,
        depth_config=depth_config,
        reduced_hessian=reduced_hessian, basis=basis,
        ad_kernel_every=ad_kernel_every,
    )
    failed = [res["failed"] for res in results if "failed" in res]
    if failed:
        raise RuntimeError(f"phase 1 failed for data: {failed}")

    n_data = len(dataset)
    n_inv = basis.n_params
    c0 = np.empty(n_data)
    J = np.empty((n_data, n_inv))
    H = np.empty((n_data, n_inv, n_inv))
    consistency = np.empty(n_data)
    asymmetry = np.empty(n_data)
    rel_err = np.empty(n_data)
    for i, res in enumerate(results):
        c0[i] = res["c"]
        rel_err[i] = res["rel_err"]
        asymmetry[i] = res["asymmetry_rel"]
        # Production Jacobian: velocity components from the weak-form gradient
        # (the reduced Hessian call's gradient_node, computed unconditionally
        # below), depth component from the variational K_d route when the
        # depth backend supports it. AD supplies only the cross-check, in
        # "g_a_ad"/"g_inv_kernel", present when ad_kernel_every kept the
        # AD pass on for this datum.
        G_hess = gradient_log(model0, res["g_a_variational"])
        J[i] = project_gradient(G_hess, basis, model=model0)
        if depth_config is not None:
            if "g_variational_depth" in res:
                J[i][basis.depth_parameter_index] = res["g_variational_depth"]
            elif "g_inv_kernel" in res:
                # The configured depth backend has no depth_gradient_variational
                # route (only the default 'variational_ad' backend does), so the
                # depth component still needs the AD pass. Fine as long as it
                # ran for this datum.
                J[i][basis.depth_parameter_index] = res["g_inv_kernel"][
                    basis.depth_parameter_index]
            else:
                raise RuntimeError(
                    f"{dataset.data[i].label()}: no depth-gradient source -- "
                    "ad_kernel_every thinned the AD pass and the configured "
                    "depth Hessian backend has no depth_gradient_variational "
                    "route (finite_difference and the plain analytic backend "
                    "do not; the default 'variational_ad' backend does). Use "
                    "ad_kernel_every=1, or the 'variational_ad' depth backend, "
                    "or both."
                )
        if "g_a_ad" in res:
            G_kernel = gradient_log(model0, res["g_a_ad"])
            ad_row = res.get("g_inv_kernel", project_gradient(G_kernel, basis, model=model0))
            scale = float(np.max(np.abs(J[i]))) or 1.0
            consistency[i] = float(np.max(np.abs(J[i] - ad_row)) / scale)
        else:
            # ad_kernel_every thinned the AD pass out here: nothing to compare
            # J against. NaN, not 0 -- a check that was never run must not
            # read as one that passed.
            consistency[i] = np.nan
        if reduced_hessian:
            H_i = np.asarray(res["hessian_reduced"], dtype=float) + hessian_curvature_only(
                model0, res["g_a_variational"], basis
            )
        else:
            G_hess = gradient_log(model0, res["g_a_variational"])
            H_i = project_hessian(
                hessian_log(model0, res["g_a_variational"], res["H_a"]),
                basis, model=model0, g_log=G_hess
            )
        if depth_config is not None:
            depth_column = res.get("depth_hessian_column")
            depth_meta = res.get("depth_hessian_meta", {})
            if (
                depth_column is None
                or not depth_meta.get("plus_success")
                or not depth_meta.get("minus_success")
                or (
                    depth_meta.get("convergence_checked")
                    and depth_meta.get("converged") is not True
                )
            ):
                raise RuntimeError(
                    f"depth Hessian column failed for {dataset.data[i].label()}: {depth_meta}"
                )
            depth_column = np.asarray(depth_column, dtype=float)
            depth_idx = basis.depth_parameter_index
            H_i[:, depth_idx] = depth_column
            H_i[depth_idx, :] = depth_column
        H[i] = H_i

    result = Phase1Result(
        c0=c0,
        J=J,
        H=H,
        consistency=consistency,
        asymmetry=asymmetry,
        rel_err=rel_err,
        config_hash=cfg_hash,
        meta={
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_data": n_data,
            "n_params": n_inv,
            "processes": processes,
            "reduced_hessian": reduced_hessian,
            "ad_kernel_every": ad_kernel_every,
            "dataset": dataset.to_json(),
            "root_trace": {
                "n_data": int(root_hints.size),
                "mode_count_calls": int(trace_engine.counters["count"]),
                "forward_calls": int(trace_engine.counters["forward"]),
            },
            "branch_confirmation": [
                {
                    "datum": dataset.data[i].label(),
                    **(
                        res.get("branch_confirmation")
                        if res.get("branch_confirmation") is not None
                        else {"status": "unavailable"}
                    ),
                }
                for i, res in enumerate(results)
            ],
            "depth_backend": (
                None if depth_config is None else {
                    **{k: v for k, v in depth_config.items() if k != "velocity_operator"},
                    "convergence_checked": all(
                        bool(res.get("depth_hessian_meta", {}).get("convergence_checked"))
                        for res in results
                    ),
                    "all_converged": (
                        None
                        if not any(
                            res.get("depth_hessian_meta", {}).get("convergence_checked")
                            for res in results
                        )
                        else all(
                            res.get("depth_hessian_meta", {}).get("converged") is True
                            for res in results
                        )
                    ),
                    "records": [
                        {
                            **res.get("depth_hessian_meta", {}),
                            "datum": dataset.data[i].label(),
                        }
                        for i, res in enumerate(results)
                    ],
                }
            ),
        },
    )
    if cache_path is not None:
        result.save(cache_path)
    return result


def solve_linearized(phase1: Phase1Result, c_obs: np.ndarray, sigma: np.ndarray,
                     prior: PriorCovariance, m0_minus_mprior: np.ndarray = None):
    """(dm_MAP, Sigma_lin, H_GN, eigen) for one noise realization."""
    r0 = np.asarray(c_obs) - phase1.c0
    H_GN, g0 = gauss_newton(phase1.J, r0, sigma, prior, m0_minus_mprior)
    dm_map, Sigma_lin = linearized_map(H_GN, g0)
    return dm_map, Sigma_lin, H_GN, eigen_analysis(H_GN, prior)


# ---------------------------------------------------------------------------
# Phases 2-4 + full pipeline
# ---------------------------------------------------------------------------


DEFAULT_PROCESSES = min(24, os.cpu_count() or 8)


@dataclass
class PipelineConfig:
    n_seeds: int = 200
    delta_safe: float = 1.0
    merge_radius: float = 1.0            # linearized-posterior sigmas (H_GN metric)
    seed_temper: float = 1.0
    # Boundary candidates are landed by default with their label kept (the
    # method plan tracks inside/outside survival separately in Stage E). In
    # high dimension the boundary case is common even for the global basin:
    # the linearized MAP itself can sit just outside Delta_safe.
    land_boundary_candidates: bool = True
    max_landings: int = 20               # cap on LM landings (best surrogate values first)
    # Certification runs must visit every eligible surrogate cluster. This
    # disables both the cap and the consecutive-merge early stop while
    # retaining the normal cap/patience defaults for routine inversions.
    exhaustive_landings: bool = False
    # Stop landing further clusters once this many consecutive landings merged
    # into already-known minima (same-basin scatter produces many clusters that
    # all land on one minimum; distinct valleys differ in surrogate value and
    # surface early in the value-ordered sweep). 0 disables the early stop.
    landing_patience: int = 3
    min_component_weight: float = 1e-3   # reported but dropped from the headline GMM
    # Phase-4 Hessian reuse is certified per landing by the surrogate itself:
    # if the phase-1 quadratic prediction c-hat(dm_k) matches the true landed
    # c(m_k) to sigma/3 at the p90 level (the Delta_safe calibration target),
    # the same per-datum H_i are valid at m_k and re-evaluation is skipped.
    # Set False to always re-evaluate.
    hessian_reuse: bool = True
    hessian_reuse_p90: float = 1.0 / 3.0
    # Seed the basis Jacobian's columns directly in the C++ grid Hessian
    # instead of every node-VTI unit direction (16x-24x faster at production
    # scale, machine-precision agreement with the non-reduced path -- see
    # tests/REPORT_GRID_HESSIAN_REDUCED_BASIS_20260805.md). Set False to fall
    # back to the original node-space path.
    reduced_hessian: bool = True
    processes: int = DEFAULT_PROCESSES
    # Optional h/2 validation for the interface-depth Hessian backend.  The
    # production default uses the two primary z±h evaluations; set to 0.025
    # km for the full convergence audit.
    depth_validation_step_km: float | None = None
    rng_seed: int = 0
    # Optional deterministic starts appended to the Gaussian linearized-
    # posterior seeds.  Application tests with a large, physically motivated
    # anomaly can place the linearized MAP well outside the feasible/positive-
    # definite region; supplying a small set of admissible hypotheses keeps
    # that stress test from being conflated with random-seed rejection.
    initial_seeds: np.ndarray | None = None
    engine_kwargs: dict = field(default_factory=dict)
    lm_kwargs: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    phase1: "Phase1Result"
    dm_map: np.ndarray
    Sigma_lin: np.ndarray
    eigen: dict
    candidates: list
    clusters: list
    landings: list
    groups: list
    gmm: object
    dropped_components: list
    survival: dict
    posdef_reports: list


def run_phase2(phase1: Phase1Result, c_obs, sigma, prior: PriorCovariance,
               basis: InversionBasis, config: PipelineConfig):
    """Multistart on the surrogate (zero forward calls). Returns (candidates, clusters,
    dm_map, Sigma_lin, eigen)."""
    from .search import cluster_candidates, run_multistart, sample_seeds
    from .surrogate import SurrogateMisfit

    dm_map, Sigma_lin, H_GN, eigen = solve_linearized(phase1, c_obs, sigma, prior)
    rng = np.random.default_rng(config.rng_seed)
    seed_blocks = []
    if config.n_seeds > 0:
        try:
            seed_blocks.append(
                sample_seeds(dm_map, Sigma_lin, config.n_seeds, rng, basis=basis,
                             temper=config.seed_temper)
            )
        except RuntimeError:
            # A broad application anomaly can make essentially all Gaussian
            # posterior draws infeasible.  If deterministic starts were
            # supplied, retain those starts and let the caller report the
            # rejection as a search diagnostic rather than aborting Phase 2.
            if config.initial_seeds is None:
                raise
    if config.initial_seeds is not None:
        initial = np.asarray(config.initial_seeds, dtype=float)
        if initial.ndim == 1:
            initial = initial[None, :]
        if initial.ndim != 2 or initial.shape[1] != basis.n_params:
            raise ValueError(
                "initial_seeds must have shape (n, basis.n_params), got "
                f"{initial.shape} for n_params={basis.n_params}"
            )
        checked = []
        for seed in initial:
            candidate = basis.project_feasible(seed) if hasattr(basis, "project_feasible") else seed.copy()
            try:
                model = basis.m_to_forward_model(candidate)
            except (IndexError, ValueError, FloatingPointError):
                continue
            if check_stiffness_posdef(model).ok:
                checked.append(candidate)
        if checked:
            seed_blocks.append(np.asarray(checked))
    if not seed_blocks:
        raise RuntimeError("no feasible phase-2 seeds; provide n_seeds or initial_seeds")
    seeds = np.vstack(seed_blocks)
    misfit = SurrogateMisfit(phase1.surrogate(), c_obs, sigma, prior)
    candidates = run_multistart(misfit, seeds, config.delta_safe, prior)
    clusters = cluster_candidates(candidates, metric=H_GN, merge_radius=config.merge_radius)
    return candidates, clusters, dm_map, Sigma_lin, eigen


def run_phase3(engine, basis: InversionBasis, dataset: DatasetSpec, c_obs, sigma,
               prior: PriorCovariance, clusters, config: PipelineConfig):
    """LM landing from each unique candidate cluster; merge converged minima."""
    from .landing import LMOptimizer, merge_minima

    lm = LMOptimizer(
        engine, basis, dataset, c_obs, sigma, prior,
        processes=config.processes, engine_kwargs=config.engine_kwargs, **config.lm_kwargs,
    )
    landings = []
    failed = []
    eligible = [
        c for c in clusters if config.land_boundary_candidates or not c.at_boundary
    ]
    eligible = sorted(eligible, key=lambda c: c.value)
    if not config.exhaustive_landings:
        eligible = eligible[: config.max_landings]
    groups = []
    consecutive_merges = 0
    n_skipped = 0
    for index, cluster in enumerate(eligible):
        if (not config.exhaustive_landings and config.landing_patience
                and consecutive_merges >= config.landing_patience):
            n_skipped = len(eligible) - index
            break
        try:
            result = lm.run(cluster.center)
            result.from_boundary = cluster.at_boundary
            landings.append(result)
        except Exception as exc:
            failed.append((cluster, f"{type(exc).__name__}: {exc}"))
            continue
        if not result.converged:
            continue
        n_groups_before = len(groups)
        groups = merge_minima([l for l in landings if l.converged], prior, sigma)
        if len(groups) == n_groups_before and n_groups_before > 0:
            consecutive_merges += 1
        else:
            consecutive_merges = 0
    return landings, groups, failed, n_skipped


def run_phase4(model0, basis: InversionBasis, dataset: DatasetSpec, c_obs, sigma,
               prior: PriorCovariance, groups, config: PipelineConfig,
               phase1: Phase1Result = None):
    """Full Hessians at the minima, posdef handling, evidence-weighted GMM.

    In the linear regime (scaled ||dm|| <= config.hessian_reuse_threshold) the
    phase-1 per-datum Hessians are reused instead of re-evaluated at the
    minimum; the Jacobian is always the landed one (exact at the minimum).
    """
    from .derivatives import hessian_curvature_only, hessian_log, project_hessian
    from .laplace import (
        GaussianMixturePosterior,
        evidence_log_weight,
        full_hessian,
        repair_posdef,
    )

    if not groups:
        raise RuntimeError(
            "phase 4 received no converged minima; check phase-3 landing "
            "eligibility (land_boundary_candidates) and LM failures"
        )
    observation_covariance = ObservationCovariance(sigma, n_data=len(dataset))
    surrogate = phase1.surrogate() if phase1 is not None else None
    means, covs, raw_log_w, metas, reports = [], [], [], [], []

    def require_finite(label, value):
        arr = np.asarray(value)
        if np.all(np.isfinite(arr)):
            return
        bad = np.argwhere(~np.isfinite(arr))
        first = tuple(int(i) for i in bad[0]) if bad.size else ()
        raise RuntimeError(
            f"phase 4 {label} contains non-finite values at {first} "
            f"(shape={arr.shape})"
        )

    def max_abs(value):
        arr = np.asarray(value, dtype=float)
        finite = np.abs(arr[np.isfinite(arr)])
        return "none" if finite.size == 0 else f"{float(np.max(finite)):.6g}"

    for group in groups:
        best = group[0]
        step_norm = prior.scaled_norm(best.dm)
        # direct certification: the phase-1 quadratic must predict the landed
        # (true-forward) phase velocities to the Delta_safe calibration target
        reuse_p90 = math.inf
        if config.hessian_reuse and surrogate is not None:
            c_hat = surrogate.predict(best.dm)
            reuse_p90 = float(np.quantile(
                np.abs(observation_covariance.whiten(c_hat - best.c_pred)), 0.9
            ))
        reused = reuse_p90 <= config.hessian_reuse_p90
        if reused:
            H_data = phase1.H
            depth_meta_for_component = (
                None if phase1 is None else phase1.meta.get("depth_backend")
            )
        else:
            model_k = basis.m_to_forward_model(best.dm)
            depth_config = depth_backend_config(
                basis, validation_step_km=config.depth_validation_step_km
            )
            results = map_over_data(
                model_k, dataset.data, c_hints=best.c_pred, want_hessian=True,
                processes=config.processes, engine_kwargs=config.engine_kwargs,
                depth_config=depth_config,
                reduced_hessian=config.reduced_hessian,
                basis=basis,
            )
            failed = [res["failed"] for res in results if "failed" in res]
            if failed:
                raise RuntimeError(f"phase 4 Hessian evaluation failed for data: {failed}")
            n = basis.n_params
            H_data = np.empty((len(dataset), n, n))
            depth_meta_for_component = (
                None if depth_config is None else [
                    {
                        **res.get("depth_hessian_meta", {}),
                        "datum": dataset.data[i].label(),
                    }
                    for i, res in enumerate(results)
                ]
            )
            for i, res in enumerate(results):
                if config.reduced_hessian:
                    H_i = np.asarray(res["hessian_reduced"], dtype=float) + hessian_curvature_only(
                        model_k, res["g_a_variational"], basis
                    )
                else:
                    G_hess = gradient_log(model_k, res["g_a_variational"])
                    H_i = project_hessian(
                        hessian_log(model_k, res["g_a_variational"], res["H_a"]),
                        basis, model=model_k, g_log=G_hess
                    )
                if depth_config is not None:
                    depth_column = res.get("depth_hessian_column")
                    depth_meta = res.get("depth_hessian_meta", {})
                    if (
                        depth_column is None
                        or not depth_meta.get("plus_success")
                        or not depth_meta.get("minus_success")
                        or (
                            depth_meta.get("convergence_checked")
                            and depth_meta.get("converged") is not True
                        )
                    ):
                        raise RuntimeError(
                            f"depth Hessian column failed for {dataset.data[i].label()}: {depth_meta}"
                        )
                    depth_idx = basis.depth_parameter_index
                    depth_column = np.asarray(depth_column, dtype=float)
                    H_i[:, depth_idx] = depth_column
                    H_i[depth_idx, :] = depth_column
                require_finite(
                    f"per-datum Hessian for {dataset.data[i].label()}", H_i
                )
                H_data[i] = H_i
        require_finite("landed Jacobian", best.J)
        r = np.asarray(c_obs) - best.c_pred
        require_finite("landed prediction", best.c_pred)
        require_finite("residual", r)
        require_finite("data Hessian", H_data)
        H_full = full_hessian(best.J, r, H_data, sigma, prior)
        if not np.all(np.isfinite(H_full)):
            raise RuntimeError(
                "phase 4 full Hessian contains non-finite values "
                f"(component={len(means)}, max_finite="
                f"{max_abs(H_full)}; max_inputs="
                f"J:{max_abs(best.J)},r:{max_abs(r)},H_data:{max_abs(H_data)})"
            )
        H_spd, report = repair_posdef(H_full)
        reports.append(report)
        Sigma_k = np.linalg.inv(H_spd)
        means.append(best.dm)
        covs.append(0.5 * (Sigma_k + Sigma_k.T))
        raw_log_w.append(evidence_log_weight(best.misfit, H_spd))
        metas.append(
            {
                "misfit": best.misfit,
                "chi2": best.chi2,
                "n_iter": best.n_iter,
                "n_members": len(group),
                "hessian_reused": bool(reused),
                "hessian_reuse_p90": None if not np.isfinite(reuse_p90) else reuse_p90,
                "step_scaled_norm": float(step_norm),
                "depth_hessian_backend": depth_meta_for_component,
                "posdef": report,
            }
        )
    gmm_all = GaussianMixturePosterior.from_components(means, covs, raw_log_w, metas)
    keep = gmm_all.weights >= config.min_component_weight
    dropped = [
        {"index": int(k), "weight": float(gmm_all.weights[k]), **metas[k]}
        for k in np.nonzero(~keep)[0]
    ]
    if not np.all(keep):
        gmm = GaussianMixturePosterior.from_components(
            gmm_all.means[keep], gmm_all.covs[keep],
            gmm_all.log_weights[keep], [m for m, k in zip(metas, keep) if k],
        )
    else:
        gmm = gmm_all
    return gmm, dropped, reports


def run_pipeline(engine, model0, basis: InversionBasis, dataset: DatasetSpec, c_obs,
                 sigma, prior: PriorCovariance, config: PipelineConfig,
                 cache_dir: Path = None) -> PipelineResult:
    phase1 = run_phase1(model0, basis, dataset, processes=config.processes,
                        engine_kwargs=config.engine_kwargs, cache_dir=cache_dir,
                        depth_validation_step_km=config.depth_validation_step_km,
                        reduced_hessian=config.reduced_hessian)
    forward_calls_before = engine.counters["forward"] + engine.counters["kernels"]
    candidates, clusters, dm_map, Sigma_lin, eigen = run_phase2(
        phase1, c_obs, sigma, prior, basis, config
    )
    forward_calls_phase2 = (
        engine.counters["forward"] + engine.counters["kernels"] - forward_calls_before
    )
    if forward_calls_phase2 != 0:
        raise RuntimeError(f"phase 2 must be forward-free, saw {forward_calls_phase2} calls")

    landings, groups, failed, n_landings_skipped = run_phase3(
        engine, basis, dataset, c_obs, sigma, prior, clusters, config
    )
    try:
        gmm, dropped, posdef_reports = run_phase4(
            model0, basis, dataset, c_obs, sigma, prior, groups, config, phase1=phase1
        )
    except Exception as exc:
        # Preserve the Phase-2/3 funnel when an application stress test has
        # no converged landing or Phase 4 encounters a nonfinite Hessian. The
        # caller can serialize this diagnostic without rerunning expensive
        # forward evaluations.
        exc.phase3_diagnostics = {
            "phase4_error": f"{type(exc).__name__}: {exc}",
            "n_seeds": config.n_seeds,
            "n_initial_seeds": (
                0 if config.initial_seeds is None
                else int(np.asarray(config.initial_seeds).reshape(-1, basis.n_params).shape[0])
            ),
            "n_candidates": len(candidates),
            "n_clusters": len(clusters),
            "n_clusters_interior": sum(1 for c in clusters if not c.at_boundary),
            "n_clusters_boundary": sum(1 for c in clusters if c.at_boundary),
            "n_landed": len(landings),
            "n_landings_skipped_early_stop": n_landings_skipped,
            "n_converged": sum(1 for l in landings if l.converged),
            "n_landing_stalled": sum(1 for l in landings if not l.converged),
            "n_landing_failed": len(failed),
            "landing_failures": [str(message) for _cluster, message in failed],
            "mode_failure_events": sum(len(l.mode_failures) for l in landings),
            "landing_termination_reasons": [l.termination_reason for l in landings],
        }
        raise
    survival = {
        "n_seeds": config.n_seeds,
        "n_initial_seeds": (
            0 if config.initial_seeds is None
            else int(np.asarray(config.initial_seeds).reshape(-1, basis.n_params).shape[0])
        ),
        "n_total_seeds": len(candidates),
        "n_candidates": len(candidates),
        "n_clusters": len(clusters),
        "n_clusters_interior": sum(1 for c in clusters if not c.at_boundary),
        "n_clusters_boundary": sum(1 for c in clusters if c.at_boundary),
        "n_landed": len(landings),
        "n_landings_skipped_early_stop": n_landings_skipped,
        "landing_exhaustive": bool(config.exhaustive_landings),
        "n_converged": sum(1 for l in landings if l.converged),
        "n_landing_stalled": sum(1 for l in landings if not l.converged),
        "n_landing_failed": len(failed),
        "n_unique_minima": len(groups),
        "n_gmm_components": gmm.n_components,
        "n_dropped_components": len(dropped),
        "n_hessian_reused": sum(1 for m in gmm.component_meta if m.get("hessian_reused")),
        "mode_failure_events": sum(len(l.mode_failures) for l in landings),
    }
    return PipelineResult(
        phase1=phase1,
        dm_map=dm_map,
        Sigma_lin=Sigma_lin,
        eigen=eigen,
        candidates=candidates,
        clusters=clusters,
        landings=landings,
        groups=groups,
        gmm=gmm,
        dropped_components=dropped,
        survival=survival,
        posdef_reports=posdef_reports,
    )
