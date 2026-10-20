"""Prior covariance C_m on the inversion basis.

Structure (see the implementation plan):

    C_m[(p,i),(q,j)] = sigma_p(z_i) * sigma_q(z_j) * R[p,q] * k(x_i, x_j)

with depth coordinate x = log(1 + z/z0) so that correlation lengths scale with
depth (fine shallow structure, smooth deep structure). R is the parameter
cross-correlation (Birch-like rho-velocity coupling etc.), registered as named
presets so Stage D can run C_m sensitivity sweeps. Velocity sigmas are
fractional (dimensionless log-space); an optional interface-depth coordinate is
expressed in km and receives its own physical-unit prior.

The weakly-sensitive directions (eta, rho) are deliberately given tight sigmas:
their posterior behaviour is prior-dominated by design (documented in the
method plan).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky

from .model import InversionBasis

DEFAULT_SIGMA = {
    "vph": 0.05,
    "vpv": 0.05,
    "vsh": 0.05,
    "vsv": 0.05,
    "eta": 0.02,
    "rho": 0.02,
}

# Parameter cross-correlation presets over (rho, vph, vpv, vsh, vsv, eta).
_FIELDS = ("rho", "vph", "vpv", "vsh", "vsv", "eta")


def _preset_matrix(pairs: dict) -> np.ndarray:
    R = np.eye(len(_FIELDS))
    index = {f: i for i, f in enumerate(_FIELDS)}
    for (a, b), value in pairs.items():
        R[index[a], index[b]] = R[index[b], index[a]] = value
    return R


PARAM_CORR_PRESETS = {
    "diagonal": _preset_matrix({}),
    "birch": _preset_matrix(
        {
            ("rho", "vph"): 0.5, ("rho", "vpv"): 0.5, ("rho", "vsh"): 0.5, ("rho", "vsv"): 0.5,
            ("vph", "vpv"): 0.7, ("vsh", "vsv"): 0.7,
            ("vph", "vsh"): 0.3, ("vph", "vsv"): 0.3, ("vpv", "vsh"): 0.3, ("vpv", "vsv"): 0.3,
            ("eta", "vph"): 0.3, ("eta", "vpv"): 0.3,
        }
    ),
    "birch_tight_aniso": _preset_matrix(
        {
            ("rho", "vph"): 0.5, ("rho", "vpv"): 0.5, ("rho", "vsh"): 0.5, ("rho", "vsv"): 0.5,
            ("vph", "vpv"): 0.9, ("vsh", "vsv"): 0.9,
            ("vph", "vsh"): 0.3, ("vph", "vsv"): 0.3, ("vpv", "vsh"): 0.3, ("vpv", "vsv"): 0.3,
            ("eta", "vph"): 0.3, ("eta", "vpv"): 0.3,
        }
    ),
}


class PriorCovariance:
    def __init__(
        self,
        basis: InversionBasis,
        sigma: dict = None,
        corr_length_logdepth: float = 0.4,
        depth_kernel: str = "gaussian",
        param_corr: str = "birch",
        z0_km: float = 30.0,
        jitter: float = 1e-10,
        depth_sigma_km: float = 5.0,
    ):
        self.basis = basis
        self.sigma_spec = dict(DEFAULT_SIGMA, **(sigma or {}))
        self.depth_sigma_km = float(depth_sigma_km)
        if self.depth_sigma_km <= 0.0 or not np.isfinite(self.depth_sigma_km):
            raise ValueError("depth_sigma_km must be finite and positive")
        self.corr_length = float(corr_length_logdepth)
        self.depth_kernel = depth_kernel
        self.param_corr_name = param_corr
        self.z0_km = float(z0_km)
        self.jitter = float(jitter)

        if param_corr not in PARAM_CORR_PRESETS:
            raise ValueError(f"unknown param_corr preset {param_corr!r}")
        R_full = PARAM_CORR_PRESETS[param_corr]
        sel = [_FIELDS.index(name) for name in basis.param_names]
        self.R = R_full[np.ix_(sel, sel)]

        x = np.log1p(basis.inv_depths / self.z0_km)
        dx = x[:, None] - x[None, :]
        if depth_kernel == "gaussian":
            K = np.exp(-0.5 * (dx / self.corr_length) ** 2)
        elif depth_kernel == "exponential":
            K = np.exp(-np.abs(dx) / self.corr_length)
        else:
            raise ValueError("depth_kernel must be 'gaussian' or 'exponential'")

        sig = np.concatenate(
            [np.full(basis.n_inv, self.sigma_spec[name]) for name in basis.param_names]
        )
        C_velocity = np.kron(self.R, K) * np.outer(sig, sig)
        # InterfaceShiftBasis appends one coordinate in physical km.  Keep it
        # independent of the velocity field by default; the coordinate can be
        # correlated later without changing the backend or inversion API.
        if getattr(basis, "n_params", C_velocity.shape[0]) > C_velocity.shape[0]:
            if int(basis.n_params) != C_velocity.shape[0] + 1:
                raise ValueError("unsupported non-velocity basis size for PriorCovariance")
            C = np.zeros((C_velocity.shape[0] + 1, C_velocity.shape[1] + 1))
            C[:-1, :-1] = C_velocity
            C[-1, -1] = self.depth_sigma_km**2
        else:
            C = C_velocity
        C[np.diag_indices_from(C)] += self.jitter
        self._C = C
        # Build-time SPD verification; raises LinAlgError if the preset is invalid.
        self._L = cholesky(C, lower=True)

    @property
    def n(self) -> int:
        return self._C.shape[0]

    def matrix(self) -> np.ndarray:
        return self._C

    def chol(self) -> np.ndarray:
        return self._L

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """C_m^{-1} @ rhs via the cached Cholesky factor."""
        return cho_solve((self._L, True), rhs)

    def inv(self) -> np.ndarray:
        return self.solve(np.eye(self.n))

    def logdet(self) -> float:
        return float(2.0 * np.sum(np.log(np.diag(self._L))))

    def scaled_norm(self, dm: np.ndarray) -> float:
        """||L^{-1} dm|| — the trust-region norm used for Delta_safe."""
        from scipy.linalg import solve_triangular

        return float(np.linalg.norm(solve_triangular(self._L, dm, lower=True)))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, n_params) prior draws centered at zero."""
        z = rng.standard_normal((self.n, n))
        return (self._L @ z).T

    def describe(self) -> dict:
        return {
            "sigma": self.sigma_spec,
            "corr_length_logdepth": self.corr_length,
            "depth_kernel": self.depth_kernel,
            "param_corr": self.param_corr_name,
            "z0_km": self.z0_km,
            "jitter": self.jitter,
            "depth_sigma_km": self.depth_sigma_km,
            "n": self.n,
        }


class ConditionalPriorCovariance:
    """Prior for active coordinates conditioned on fixed full-basis coordinates.

    If a full prior has precision ``Q = C^{-1}`` and the inactive coordinates
    are fixed at their prior mean, the active conditional precision is
    ``Q_aa``.  This differs from simply taking the active covariance block
    ``C_aa`` and is required for an apples-to-apples depth-fixed comparison.
    """

    def __init__(self, full_prior: PriorCovariance, active_indices, basis=None):
        active = np.asarray(active_indices, dtype=int)
        if active.ndim != 1 or active.size == 0:
            raise ValueError("active_indices must be a non-empty one-dimensional array")
        if np.any(active < 0) or np.any(active >= full_prior.n):
            raise IndexError("active_indices contains an out-of-range parameter")
        if np.unique(active).size != active.size:
            raise ValueError("active_indices must be unique")
        self.full_prior = full_prior
        self.active_indices = active.copy()
        self.basis = basis
        precision = full_prior.inv()
        precision_active = precision[np.ix_(active, active)]
        precision_active = 0.5 * (precision_active + precision_active.T)
        self._C = np.linalg.inv(precision_active)
        self._C = 0.5 * (self._C + self._C.T)
        self._L = cholesky(self._C, lower=True)

    @property
    def n(self) -> int:
        return self._C.shape[0]

    def matrix(self) -> np.ndarray:
        return self._C

    def chol(self) -> np.ndarray:
        return self._L

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve((self._L, True), rhs)

    def inv(self) -> np.ndarray:
        return self.solve(np.eye(self.n))

    def logdet(self) -> float:
        return float(2.0 * np.sum(np.log(np.diag(self._L))))

    def scaled_norm(self, dm: np.ndarray) -> float:
        from scipy.linalg import solve_triangular

        return float(np.linalg.norm(solve_triangular(self._L, dm, lower=True)))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((self.n, n))
        return (self._L @ z).T

    def describe(self) -> dict:
        return {
            "type": type(self).__name__,
            "full_prior": self.full_prior.describe(),
            "active_indices": self.active_indices.tolist(),
            "n": self.n,
        }
