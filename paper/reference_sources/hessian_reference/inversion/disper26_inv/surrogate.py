"""Per-datum quadratic surrogates and the phase-2 surrogate misfit.

The surrogate misfit J-hat is a quartic in dm with closed-form gradient and
Hessian — phase 2 optimizes it with zero forward-solver calls.
"""

from __future__ import annotations

import numpy as np

from .data import ObservationCovariance
from .priors import PriorCovariance


class QuadraticSurrogate:
    """c-hat_i(dm) = c0_i + g_i . dm + 0.5 dm^T H_i dm for every datum."""

    def __init__(self, c0: np.ndarray, G: np.ndarray, H: np.ndarray):
        self.c0 = np.asarray(c0, dtype=float)          # (N,)
        self.G = np.asarray(G, dtype=float)            # (N, n)
        self.H = np.asarray(H, dtype=float)            # (N, n, n)
        n_data = self.c0.shape[0]
        if self.G.shape[0] != n_data or self.H.shape[0] != n_data:
            raise ValueError("inconsistent surrogate shapes")

    @property
    def n_data(self) -> int:
        return self.c0.shape[0]

    @property
    def n_params(self) -> int:
        return self.G.shape[1]

    def predict_linear(self, dm: np.ndarray) -> np.ndarray:
        return self.c0 + self.G @ dm

    def predict(self, dm: np.ndarray) -> np.ndarray:
        return self.predict_linear(dm) + 0.5 * np.einsum("i,nij,j->n", dm, self.H, dm)

    def gradients_at(self, dm: np.ndarray) -> np.ndarray:
        """dc-hat_i/dm at dm: rows g_i + H_i dm, shape (N, n)."""
        return self.G + np.einsum("nij,j->ni", self.H, dm)


class SurrogateMisfit:
    """J-hat(dm) = 0.5 sum_i (c_obs_i - c-hat_i(dm))^2 / sigma_i^2
                 + 0.5 (m0 + dm - m_prior)^T C_m^{-1} (m0 + dm - m_prior)."""

    def __init__(
        self,
        surrogate: QuadraticSurrogate,
        c_obs: np.ndarray,
        sigma: np.ndarray,
        prior: PriorCovariance,
        m0_minus_mprior: np.ndarray = None,
    ):
        self.surrogate = surrogate
        self.c_obs = np.asarray(c_obs, dtype=float)
        self.covariance = ObservationCovariance(sigma, n_data=surrogate.n_data)
        # Kept for callers that inspect the historical diagonal weight field.
        self.w = 1.0 / self.covariance.sigma**2
        self.prior = prior
        self.m0_minus_mprior = (
            np.zeros(surrogate.n_params)
            if m0_minus_mprior is None
            else np.asarray(m0_minus_mprior, dtype=float)
        )

    def _prior_vec(self, dm: np.ndarray) -> np.ndarray:
        return self.prior.solve(dm + self.m0_minus_mprior)

    def value(self, dm: np.ndarray) -> float:
        r = self.c_obs - self.surrogate.predict(dm)
        prior_term = 0.5 * float((dm + self.m0_minus_mprior) @ self._prior_vec(dm))
        return 0.5 * self.covariance.quadratic(r) + prior_term

    def grad(self, dm: np.ndarray) -> np.ndarray:
        r = self.c_obs - self.surrogate.predict(dm)
        grads = self.surrogate.gradients_at(dm)
        return -grads.T @ self.covariance.solve(r) + self._prior_vec(dm)

    def hess(self, dm: np.ndarray) -> np.ndarray:
        r = self.c_obs - self.surrogate.predict(dm)
        grads = self.surrogate.gradients_at(dm)
        H = grads.T @ self.covariance.solve(grads)
        H -= np.einsum("n,nij->ij", self.covariance.solve(r), self.surrogate.H)
        return H + self.prior.inv()

    def hess_gn(self, dm: np.ndarray) -> np.ndarray:
        """Gauss-Newton approximation: drops the residual-curvature term.

        SPD by construction (grads^T W grads + C_m^{-1}), so descent steps
        never hit the indefinite-Hessian case that noise-level residuals
        cause in the full `hess` (the -r_i H_i term flips weak directions).
        Fixed points still satisfy grad = 0 exactly.
        """
        grads = self.surrogate.gradients_at(dm)
        return grads.T @ self.covariance.solve(grads) + self.prior.inv()
