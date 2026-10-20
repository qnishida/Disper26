"""Phase 4: full Hessians at the minima, posdef handling, evidence-weighted GMM.

Full misfit Hessian at a landed minimum m_k:

    H_full = J^T C_d^{-1} J + C_m^{-1} - sum_i (r_i / sigma_i^2) H_i

with r_i = c_obs_i - c_i(m_k) and H_i the per-datum phase-velocity Hessians
re-evaluated at m_k. Negative eigenvalues trigger (a) a needs_more_lm flag and
then (b) eigenvalue flooring, with the intervention recorded.

Mixture weights use the Laplace-approximation evidence,
w_k ~ exp(-J_k) |Sigma_k|^{1/2}, evaluated in the log domain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2 as chi2_dist

from .priors import PriorCovariance
from .data import ObservationCovariance


def full_hessian(J: np.ndarray, r: np.ndarray, H_data: np.ndarray, sigma: np.ndarray,
                 prior: PriorCovariance) -> np.ndarray:
    metric = ObservationCovariance(sigma, n_data=np.asarray(J).shape[0])
    H = J.T @ metric.solve(np.asarray(J, dtype=float)) + prior.inv()
    H -= np.einsum("n,nij->ij", metric.solve(np.asarray(r, dtype=float)), H_data)
    return 0.5 * (H + H.T)


def repair_posdef(H: np.ndarray, floor_rel: float = 1e-12):
    """(H_spd, report). Floors negative/small eigenvalues at floor_rel * max_eig.

    floor_rel must sit well below the legitimate spectral range: the weakest
    H_full eigenvalues are prior-floor curvature (~1e2 in scaled units) while
    strong data directions reach ~1e10, i.e. a genuine condition number of
    ~1e8 — harmless in float64. A floor at 1e-8 * max clipped healthy
    prior-dominated eigenvalues once Stage-C data curvature grew.
    n_negative (eigenvalues <= 0) remains the meaningful repair indicator.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    max_eig = float(eigvals.max())
    n_negative = int(np.sum(eigvals <= 0.0))
    floor = floor_rel * max_eig
    floored = eigvals < floor
    report = {
        "n_negative": n_negative,
        "min_eig": float(eigvals.min()),
        "max_eig": max_eig,
        "n_floored": int(np.sum(floored)),
        "needs_more_lm": n_negative > 0,
    }
    if report["n_floored"] == 0:
        return H, report
    fixed = eigvecs @ np.diag(np.maximum(eigvals, floor)) @ eigvecs.T
    return 0.5 * (fixed + fixed.T), report


def evidence_log_weight(misfit_value: float, H_spd: np.ndarray) -> float:
    """log w_k (unnormalized) = -J_k + 0.5 log|Sigma_k| = -J_k - 0.5 log|H_k|."""
    sign, logdet = np.linalg.slogdet(H_spd)
    if sign <= 0:
        raise ValueError("H must be SPD for the evidence weight")
    return float(-misfit_value - 0.5 * logdet)


@dataclass
class GaussianMixturePosterior:
    means: np.ndarray          # (K, n) dm relative to the reference model
    covs: np.ndarray           # (K, n, n)
    log_weights: np.ndarray    # (K,), normalized: logsumexp == 0
    component_meta: list = field(default_factory=list)

    @classmethod
    def from_components(cls, means, covs, raw_log_weights, component_meta=None):
        lw = np.asarray(raw_log_weights, dtype=float)
        lw = lw - logsumexp(lw)
        return cls(
            means=np.asarray(means, dtype=float),
            covs=np.asarray(covs, dtype=float),
            log_weights=lw,
            component_meta=component_meta or [],
        )

    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.log_weights)

    def mahalanobis2(self, m: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of m to every component."""
        out = np.empty(self.n_components)
        for k in range(self.n_components):
            delta = np.asarray(m) - self.means[k]
            out[k] = float(delta @ np.linalg.solve(self.covs[k], delta))
        return out

    def mahalanobis_nearest(self, m: np.ndarray) -> tuple:
        d2 = self.mahalanobis2(m)
        k = int(np.argmin(d2))
        return float(d2[k]), k

    def credible_interval_contains(self, m: np.ndarray, level: float) -> bool:
        """m inside the `level` HDR of the nearest component (chi2_n threshold)."""
        d2, _k = self.mahalanobis_nearest(m)
        n = self.means.shape[1]
        return d2 <= chi2_dist.ppf(level, df=n)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            means=self.means,
            covs=self.covs,
            log_weights=self.log_weights,
            component_meta=np.array(json.dumps(self.component_meta)),
        )

    @classmethod
    def load(cls, path: Path) -> "GaussianMixturePosterior":
        arrays = np.load(path, allow_pickle=False)
        return cls(
            means=arrays["means"],
            covs=arrays["covs"],
            log_weights=arrays["log_weights"],
            component_meta=json.loads(str(arrays["component_meta"])),
        )
