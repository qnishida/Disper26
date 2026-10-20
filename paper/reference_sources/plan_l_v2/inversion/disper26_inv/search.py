"""Phase 2: multistart search on the quadratic surrogate (zero forward calls).

Seeds are drawn from the linearized posterior N(m_MAP, Sigma_lin) — they
spread naturally along weakly-constrained (trade-off) directions. Each seed is
locally minimized on the quartic surrogate misfit inside the trust region
||dm||_{C_m-scaled} <= Delta_safe (surrogate divergence outside the region is
harmless by construction). Converged points are merged by whitened distance;
candidates that finish on the trust-region boundary are labeled separately
(likely surrogate artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh, solve_triangular

from .physics import check_stiffness_posdef
from .priors import PriorCovariance
from .surrogate import SurrogateMisfit


@dataclass
class SurrogateCandidate:
    dm: np.ndarray
    value: float
    scaled_radius: float
    at_boundary: bool
    n_iter: int
    seed_index: int


@dataclass
class CandidateCluster:
    center: np.ndarray
    value: float
    members: list = field(default_factory=list)
    at_boundary: bool = False

    @property
    def n_members(self) -> int:
        return len(self.members)


def sample_seeds(
    dm_map: np.ndarray,
    Sigma_lin: np.ndarray,
    n_seeds: int,
    rng: np.random.Generator,
    basis=None,
    temper: float = 1.0,
    max_attempts_factor: int = 20,
) -> np.ndarray:
    """Draw seeds from N(dm_map, temper^2 Sigma_lin); reject posdef violations."""
    L = np.linalg.cholesky(Sigma_lin)
    seeds = []
    attempts = 0
    while len(seeds) < n_seeds and attempts < n_seeds * max_attempts_factor:
        attempts += 1
        dm = dm_map + temper * (L @ rng.standard_normal(dm_map.shape[0]))
        if basis is not None:
            # Interface-depth coordinates have hard neighboring-discontinuity
            # bounds.  An out-of-range Gaussian draw is an infeasible seed,
            # not a fatal search error; treat it like a positive-definiteness
            # rejection and continue sampling.
            if hasattr(basis, "project_feasible"):
                dm = basis.project_feasible(dm)
            try:
                candidate_model = basis.m_to_forward_model(dm)
            except (IndexError, ValueError, FloatingPointError):
                continue
            if not check_stiffness_posdef(candidate_model).ok:
                continue
        seeds.append(dm)
    if len(seeds) < n_seeds:
        raise RuntimeError(f"only {len(seeds)}/{n_seeds} feasible seeds after {attempts} attempts")
    return np.asarray(seeds)


def _ball_constrained_argmin(H_y: np.ndarray, b: np.ndarray, delta: float):
    """argmin_y { -b.y + 0.5 y^T H_y y } subject to ||y|| <= delta, H_y SPD.

    Exact via eigendecomposition + a secular bisection on the boundary
    multiplier (More-Sorensen structure with the ball centered at the origin,
    i.e. the C_m-whitened trust region of the method plan). O(n^3), ~1 ms at
    n ~ 100 — replaces scipy trust-constr entirely.
    """
    lam, V = eigh(H_y)
    lam = np.clip(lam, 1e-12 * max(lam[-1], 1e-300), None)
    bt = V.T @ b
    y = bt / lam
    if np.linalg.norm(y) <= delta:
        return V @ y, False
    lo = 0.0
    hi = float(np.linalg.norm(bt)) / delta  # ||bt/(lam+hi)|| <= ||bt||/hi = delta
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(bt / (lam + mid)) > delta:
            lo = mid
        else:
            hi = mid
    return V @ (bt / (lam + hi)), True


def minimize_surrogate(
    misfit: SurrogateMisfit,
    seed_dm: np.ndarray,
    delta_safe: float,
    prior: PriorCovariance,
    seed_index: int = -1,
    boundary_tol: float = 0.98,
    max_iter: int = 200,
) -> SurrogateCandidate:
    """Ball-constrained Gauss-Newton minimization of the quartic surrogate.

    Iterates exact solutions of the local GN model over the whitened ball
    ||L^{-1} dm|| <= Delta_safe with backtracking on the true quartic value.
    The GN Hessian (hess_gn) is SPD by construction (the full quartic Hessian
    is indefinite wherever residuals sit at noise level), each subproblem is
    solved exactly, and fixed points satisfy the true KKT conditions
    (grad + mu C_m^{-1} dm = 0). High-dimensional note: linearized-posterior
    seeds concentrate at radius ~ sqrt(n_null) >> Delta_safe and even the MAP
    can sit outside the ball, so the boundary case is the NORM, not the
    exception — it must be as cheap as the interior case.
    """
    L = prior.chol()

    def to_y(dm):
        return solve_triangular(L, dm, lower=True)

    y = to_y(np.asarray(seed_dm, dtype=float))
    r0 = float(np.linalg.norm(y))
    if r0 > delta_safe:
        y = y * (0.95 * delta_safe / r0)
    dm = L @ y
    value = misfit.value(dm)

    n_iter = 0
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        g = misfit.grad(dm)
        H = misfit.hess_gn(dm)
        g_y = L.T @ g
        H_y = L.T @ H @ L
        b = H_y @ y - g_y
        y_new, _hit = _ball_constrained_argmin(H_y, b, delta_safe)
        step = y_new - y
        step_norm = float(np.linalg.norm(step))
        if step_norm < 1e-13 * max(1.0, float(np.linalg.norm(y))):
            break
        alpha = 1.0
        accepted = False
        for _bt in range(14):
            y_trial = y + alpha * step
            dm_trial = L @ y_trial
            v_trial = misfit.value(dm_trial)
            if np.isfinite(v_trial) and v_trial < value:
                y, dm, value = y_trial, dm_trial, v_trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break  # no descent along the exact subproblem step: converged

    radius = float(np.linalg.norm(y))
    return SurrogateCandidate(
        dm=dm,
        value=float(value),
        scaled_radius=radius,
        at_boundary=radius >= boundary_tol * delta_safe,
        n_iter=n_iter,
        seed_index=seed_index,
    )


def run_multistart(
    misfit: SurrogateMisfit,
    seeds: np.ndarray,
    delta_safe: float,
    prior: PriorCovariance,
) -> list:
    return [
        minimize_surrogate(misfit, seed, delta_safe, prior, seed_index=i)
        for i, seed in enumerate(seeds)
    ]


def cluster_candidates(
    candidates,
    prior: PriorCovariance = None,
    metric: np.ndarray = None,
    merge_radius: float = 1.0,
) -> list:
    """Greedy merge; boundary and interior kept separate.

    Distance metric: Mahalanobis under `metric` (pass H_GN — two candidates
    within ~1 linearized-posterior sigma belong to the same basin, which is
    what distinguishes distinct valleys from noise-induced scatter around one
    minimum). Falls back to prior-whitened distance when metric is None (then
    a much smaller merge_radius is appropriate).
    """
    if metric is not None:
        def distance(delta):
            return float(np.sqrt(delta @ metric @ delta))
    elif prior is not None:
        L = prior.chol()

        def distance(delta):
            return float(np.linalg.norm(solve_triangular(L, delta, lower=True)))
    else:
        raise ValueError("provide either metric or prior")

    clusters = []
    for cand in sorted(candidates, key=lambda c: c.value):
        placed = False
        for cluster in clusters:
            if cluster.at_boundary != cand.at_boundary:
                continue
            if distance(cand.dm - cluster.center) < merge_radius:
                cluster.members.append(cand)
                placed = True
                break
        if not placed:
            clusters.append(
                CandidateCluster(
                    center=cand.dm.copy(),
                    value=cand.value,
                    members=[cand],
                    at_boundary=cand.at_boundary,
                )
            )
    return clusters
