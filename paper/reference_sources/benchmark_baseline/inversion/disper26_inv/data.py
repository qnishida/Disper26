"""Dataset specification, noise model (C_d) and synthetic data generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .engine import Datum, ForwardEngine
from .model import ForwardGridModel


class ObservationCovariance:
    """Validated observation covariance with a diagonal fast path.

    Public inversion APIs historically accepted a vector of standard
    deviations.  This wrapper keeps that interface valid while also allowing
    a full positive-definite ``C_d`` matrix for the correlated-noise tests.
    ``solve`` accepts either a vector or a matrix right-hand side.
    """

    def __init__(self, sigma_or_cov, n_data: int = None):
        arr = np.asarray(sigma_or_cov, dtype=float)
        if arr.ndim == 1:
            if n_data is not None and arr.size != n_data:
                raise ValueError("sigma length does not match n_data")
            if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ValueError("sigma must contain finite positive values")
            self._sigma = arr.copy()
            self._cov = None
            self._factor = None
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            if n_data is not None and arr.shape[0] != n_data:
                raise ValueError("covariance shape does not match n_data")
            if arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
                raise ValueError("covariance must be finite and non-empty")
            cov = 0.5 * (arr + arr.T)
            self._sigma = np.sqrt(np.diag(cov))
            if not np.all(np.isfinite(self._sigma)) or np.any(self._sigma <= 0.0):
                raise ValueError("covariance diagonal must contain finite positive variances")
            if not np.allclose(
                cov, np.diag(np.diag(cov)), rtol=0.0, atol=1e-14 * np.max(np.diag(cov))
            ):
                self._factor = cho_factor(cov, lower=True, check_finite=True)
                self._cov = cov
            else:
                self._factor = None
                self._cov = None
        else:
            raise ValueError("observation covariance must be a sigma vector or square matrix")

    @property
    def n(self) -> int:
        return self._sigma.size

    @property
    def sigma(self) -> np.ndarray:
        return self._sigma.copy()

    @property
    def is_diagonal(self) -> bool:
        return self._cov is None

    def matrix(self) -> np.ndarray:
        if self._cov is None:
            return np.diag(self._sigma ** 2)
        return self._cov.copy()

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape[0] != self.n:
            raise ValueError("right-hand side length does not match covariance")
        if self._cov is None:
            return rhs / self._sigma**2 if rhs.ndim == 1 else rhs / self._sigma[:, None]**2
        return cho_solve(self._factor, rhs)

    def whiten(self, rhs: np.ndarray) -> np.ndarray:
        """Apply ``C_d^{-1/2}`` using the Cholesky factor."""
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape[0] != self.n:
            raise ValueError("right-hand side length does not match covariance")
        if self._cov is None:
            return rhs / self._sigma if rhs.ndim == 1 else rhs / self._sigma[:, None]
        return solve_triangular(self._factor[0], rhs, lower=True)

    def quadratic(self, residual: np.ndarray) -> float:
        residual = np.asarray(residual, dtype=float)
        return float(residual @ self.solve(residual))


@dataclass
class DatasetSpec:
    data: list

    def __len__(self) -> int:
        return len(self.data)

    def labels(self) -> list:
        return [d.label() for d in self.data]

    @classmethod
    def standard(
        cls,
        rayleigh_fundamental=(),
        love_fundamental=(),
        overtone_branches=(),
        overtone_periods=(),
    ) -> "DatasetSpec":
        """Fundamental periods per wave plus shared overtone (branch, period) grid."""
        data = [Datum("Rayleigh", 0, T) for T in rayleigh_fundamental]
        data += [Datum("Love", 0, T) for T in love_fundamental]
        for n in overtone_branches:
            for T in overtone_periods:
                data.append(Datum("Rayleigh", n, T))
                data.append(Datum("Love", n, T))
        return cls(data)

    @classmethod
    def shallow_frequency_grid(
        cls,
        f_min_hz: float = 0.03,
        f_max_hz: float = 0.2,
        step_hz: float = 0.005,
        branches=(0, 1),
    ) -> "DatasetSpec":
        """Rayleigh/Love fundamental + first-overtone shallow-data grid.

        Frequencies are ascending within each curve so the forward engine can
        use group-velocity continuation.  Datum itself remains period based.
        """
        if f_min_hz <= 0.0 or f_max_hz < f_min_hz or step_hz <= 0.0:
            raise ValueError("invalid frequency-grid bounds")
        n = int(round((f_max_hz - f_min_hz) / step_hz))
        frequencies = f_min_hz + step_hz * np.arange(n + 1, dtype=float)
        if not np.isclose(frequencies[-1], f_max_hz, rtol=0.0, atol=1e-12):
            raise ValueError("frequency range is not an integer number of steps")
        data = []
        for wave in ("Rayleigh", "Love"):
            for branch in tuple(branches):
                data.extend(Datum(wave, int(branch), 1.0 / f) for f in frequencies)
        return cls(data)

    def to_json(self) -> list:
        return [{"wave": d.wave, "branch": d.branch, "period": d.period} for d in self.data]

    @classmethod
    def from_json(cls, items) -> "DatasetSpec":
        return cls([Datum(i["wave"], i["branch"], i["period"]) for i in items])


class NoiseModel:
    """sigma_i per datum; C_d is diagonal. Options combine as
    sigma_i = max(relative * c_i, absolute_kms, floor per (wave, branch))."""

    def __init__(self, relative: float = 0.0, absolute_kms: float = 0.0, floors: dict = None):
        self.relative = float(relative)
        self.absolute_kms = float(absolute_kms)
        self.floors = dict(floors or {})

    @classmethod
    def relative_frac(cls, frac: float) -> "NoiseModel":
        return cls(relative=frac)

    def sigma(self, datum: Datum, c: float) -> float:
        floor = self.floors.get((datum.wave, datum.branch), 0.0)
        return max(self.relative * c, self.absolute_kms, floor)

    def covariance_diag(self, dataset: DatasetSpec, c_pred: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.sigma(d, c) ** 2 for d, c in zip(dataset.data, c_pred)], dtype=float
        )

    def covariance(self, dataset: DatasetSpec, c_pred: np.ndarray) -> np.ndarray:
        """Return the observation covariance; diagonal by default."""
        return np.diag(self.covariance_diag(dataset, c_pred))

    def describe(self) -> dict:
        return {
            "relative": self.relative,
            "absolute_kms": self.absolute_kms,
            "floors": {f"{k[0]}_{k[1]}": v for k, v in self.floors.items()},
        }


class CorrelatedNoiseModel(NoiseModel):
    """Mode/frequency-correlated velocity errors used by Stage A3/A4.

    Correlation is applied only within one wave and mode.  The diagonal is
    unchanged from :class:`NoiseModel`, so diagonal and correlated runs share
    exactly the same per-datum error scale.
    """

    def __init__(self, relative: float = 0.0, absolute_kms: float = 0.0,
                 floors: dict = None, correlation_strength: float = 0.5,
                 correlation_length_hz: float = 0.01):
        super().__init__(relative=relative, absolute_kms=absolute_kms, floors=floors)
        self.correlation_strength = float(correlation_strength)
        self.correlation_length_hz = float(correlation_length_hz)
        if not 0.0 <= self.correlation_strength < 1.0:
            raise ValueError("correlation_strength must be in [0, 1)")
        if self.correlation_length_hz <= 0.0:
            raise ValueError("correlation_length_hz must be positive")

    def covariance(self, dataset: DatasetSpec, c_pred: np.ndarray) -> np.ndarray:
        sigma = np.asarray([self.sigma(d, c) for d, c in zip(dataset.data, c_pred)])
        n = len(dataset)
        cov = np.diag(sigma**2)
        frequencies = np.asarray([1.0 / d.period for d in dataset.data])
        for i, di in enumerate(dataset.data):
            for j in range(i):
                dj = dataset.data[j]
                if (di.wave, di.branch) != (dj.wave, dj.branch):
                    continue
                corr = self.correlation_strength * np.exp(
                    -abs(frequencies[i] - frequencies[j]) / self.correlation_length_hz
                )
                cov[i, j] = cov[j, i] = corr * sigma[i] * sigma[j]
        return cov

    def describe(self) -> dict:
        return {
            **super().describe(),
            "correlation_strength": self.correlation_strength,
            "correlation_length_hz": self.correlation_length_hz,
        }


@dataclass
class SyntheticData:
    dataset: DatasetSpec
    c_true: np.ndarray                 # noiseless phase velocities (N,)
    sigma: np.ndarray                  # per-datum noise std (N,)
    realizations: np.ndarray           # (n_realizations, N) noisy observations
    seed: int
    truth_name: str = ""
    extras: dict = field(default_factory=dict)
    covariance: np.ndarray = None

    @property
    def n_data(self) -> int:
        return self.c_true.shape[0]

    def save(self, run_dir: Path):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            run_dir / "synthetic_data.npz",
            c_true=self.c_true,
            sigma=self.sigma,
            realizations=self.realizations,
            covariance=(self.covariance if self.covariance is not None else np.diag(self.sigma**2)),
        )
        meta = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "truth_name": self.truth_name,
            "dataset": self.dataset.to_json(),
            "n_realizations": int(self.realizations.shape[0]),
            **self.extras,
        }
        (run_dir / "synthetic_data.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path) -> "SyntheticData":
        run_dir = Path(run_dir)
        arrays = np.load(run_dir / "synthetic_data.npz")
        meta = json.loads((run_dir / "synthetic_data.json").read_text(encoding="utf-8"))
        return cls(
            dataset=DatasetSpec.from_json(meta["dataset"]),
            c_true=arrays["c_true"],
            sigma=arrays["sigma"],
            realizations=arrays["realizations"],
            seed=int(meta["seed"]),
            truth_name=meta.get("truth_name", ""),
            covariance=arrays["covariance"] if "covariance" in arrays else np.diag(arrays["sigma"]**2),
        )


def generate_synthetic_data(
    engine: ForwardEngine,
    truth_model: ForwardGridModel,
    dataset: DatasetSpec,
    noise_model: NoiseModel,
    n_realizations: int,
    seed: int,
    c_hints: np.ndarray = None,
) -> SyntheticData:
    """Noiseless c once per datum, then seeded Gaussian realizations."""
    if c_hints is None and hasattr(engine, "trace_dataset"):
        c_true = engine.trace_dataset(truth_model, dataset.data)
    else:
        c_true = np.empty(len(dataset))
        for i, datum in enumerate(dataset.data):
            hint = None if c_hints is None else float(c_hints[i])
            c_true[i] = engine.phase_velocity(truth_model, datum, c_hint=hint).c
    covariance = np.asarray(noise_model.covariance(dataset, c_true), dtype=float)
    observation_covariance = ObservationCovariance(covariance)
    sigma = observation_covariance.sigma
    rng = np.random.default_rng(seed)
    if observation_covariance.is_diagonal:
        # Preserve the historical seeded sequence for all existing diagonal
        # validation fixtures; correlated runs use the Cholesky draw below.
        noise = rng.standard_normal((n_realizations, len(dataset))) * sigma[None, :]
    else:
        noise = rng.multivariate_normal(np.zeros(len(dataset)), covariance, size=n_realizations)
    return SyntheticData(
        dataset=dataset,
        c_true=c_true,
        sigma=sigma,
        realizations=c_true[None, :] + noise,
        seed=seed,
        truth_name=truth_model.name,
        extras={"noise_model": noise_model.describe()},
        covariance=covariance,
    )
