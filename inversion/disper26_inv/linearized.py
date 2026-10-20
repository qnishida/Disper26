"""Gauss-Newton matrix, linearized MAP / posterior, eigen-spectrum analysis.

Sign conventions: residual r = c_obs - c_pred(m0); the misfit gradient at m0 is
g0 = -J^T C_d^{-1} r + C_m^{-1}(m0 - m_prior) and the linearized MAP step is
dm_MAP = -H_GN^{-1} g0.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

from .data import ObservationCovariance
from .priors import PriorCovariance


def gauss_newton(J: np.ndarray, r: np.ndarray, sigma: np.ndarray, prior: PriorCovariance,
                 m0_minus_mprior: np.ndarray = None):
    """(H_GN, g0) for the current linearization point."""
    metric = ObservationCovariance(sigma, n_data=np.asarray(J).shape[0])
    WJ = metric.solve(np.asarray(J, dtype=float))
    H_GN = J.T @ WJ + prior.inv()
    g0 = -J.T @ metric.solve(np.asarray(r, dtype=float))
    if m0_minus_mprior is not None:
        g0 = g0 + prior.solve(m0_minus_mprior)
    return H_GN, g0


def linearized_map(H_GN: np.ndarray, g0: np.ndarray):
    """(dm_MAP, Sigma_lin) via Cholesky; raises if H_GN is not SPD."""
    factor = cho_factor(H_GN)
    dm_map = -cho_solve(factor, g0)
    Sigma_lin = cho_solve(factor, np.eye(H_GN.shape[0]))
    return dm_map, Sigma_lin


def eigen_analysis(H_GN: np.ndarray, prior: PriorCovariance) -> dict:
    """Prior-whitened spectrum of H_GN (the phase-1 sensitivity analysis).

    In whitened coordinates u = L^{-1} dm the prior is the identity and
    L^T H_GN L has eigenvalues >= 1; directions with lambda >> 1 are data-
    resolved, lambda ~ 1 is the null space. The effective number of resolved
    parameters is sum(1 - 1/lambda).
    """
    L = prior.chol()
    A = L.T @ H_GN @ L
    eigvals, eigvecs = eigh(0.5 * (A + A.T))
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    directions = L @ eigvecs  # columns: directions in m space
    effective = float(np.sum(1.0 - 1.0 / np.clip(eigvals, 1.0, None)))
    return {
        "eigvals": eigvals,
        "directions": directions,
        "effective_n_params": effective,
        "n_resolved_10x": int(np.sum(eigvals > 10.0)),
    }
