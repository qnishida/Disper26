"""Python-side Romberg frequency-sweep helpers.

This module is intentionally small and depends only on the existing
``disper26py.compute_dispersion_romberg`` scalar API.  It is a staging point for
testing sweep continuation before adding a C++/pybind batch binding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class SweepPoint:
    frequency_hz: float
    period_s: float
    c_guess: float
    phase_velocity: float
    group_velocity: float
    hv_ratio: float
    level: int
    relative_error: float
    double_evals: int | None = None


def _secant_predict(frequencies: Sequence[float], velocities: Sequence[float], next_frequency: float) -> float:
    if len(velocities) == 1:
        return velocities[-1]
    f0 = frequencies[-2]
    f1 = frequencies[-1]
    c0 = velocities[-2]
    c1 = velocities[-1]
    df = f1 - f0
    if abs(df) < 1e-30:
        return c1
    return c1 + (c1 - c0) * (next_frequency - f1) / df


def _group_predict(frequency: float, c: float, U: float, next_frequency: float) -> float:
    if frequency <= 0.0 or not (math.isfinite(c) and math.isfinite(U)) or abs(U) < 1e-30:
        return float("nan")
    return c + c * (1.0 - c / U) * ((next_frequency - frequency) / frequency)


def predict_next_phase_velocity(
    frequencies_hz: Sequence[float],
    phase_velocities: Sequence[float],
    group_velocities: Sequence[float] | None,
    next_frequency_hz: float,
) -> float:
    """Predict the next phase velocity from previous sweep points.

    If a finite group velocity is available, use the local relation
    dc/domega = c/omega * (1 - c/U).  Otherwise use a two-point secant in
    frequency, falling back to the previous phase velocity for the second point.
    """
    if not phase_velocities:
        raise ValueError("at least one previous phase velocity is required")
    if group_velocities:
        pred = _group_predict(
            frequencies_hz[-1], phase_velocities[-1], group_velocities[-1], next_frequency_hz
        )
        if math.isfinite(pred) and pred > 0.0:
            return pred
    pred = _secant_predict(frequencies_hz, phase_velocities, next_frequency_hz)
    if math.isfinite(pred) and pred > 0.0:
        return pred
    return phase_velocities[-1]


def compute_dispersion_romberg_frequency_sweep(
    dp,
    depths,
    params,
    is_discontinuity,
    frequencies_hz: Iterable[float],
    wave_type: str = "Rayleigh",
    c_guess: float = 3.0,
    max_steps: int = 5,
    tolerance: float = 1e-10,
    *,
    phase_only: bool = True,
    adaptive_root_scan: bool = True,
    initial_root_window_frac: float = 0.01,
    first_initial_root_window_frac: float = 0.0,
    extrapolate_start_level: int = 0,
    collect_eval_counts: bool = True,
) -> List[SweepPoint]:
    """Compute a frequency sweep using previous roots as predictor-quality guesses."""
    freqs = [float(f) for f in frequencies_hz]
    if any(f <= 0.0 for f in freqs):
        raise ValueError("frequencies_hz must be positive")

    out: List[SweepPoint] = []
    prev_freqs: List[float] = []
    prev_c: List[float] = []
    prev_u: List[float] = []

    current_guess = float(c_guess)
    for i, freq in enumerate(freqs):
        period = 1.0 / freq
        window = first_initial_root_window_frac if i == 0 else initial_root_window_frac
        if collect_eval_counts and hasattr(dp, "reset_dispersion_eval_counts"):
            dp.reset_dispersion_eval_counts()
        result = dp.compute_dispersion_romberg(
            depths,
            params,
            is_discontinuity,
            period,
            wave_type,
            current_guess,
            max_steps,
            tolerance,
            phase_only=phase_only,
            adaptive_root_scan=adaptive_root_scan,
            initial_root_window_frac=window,
            extrapolate_start_level=extrapolate_start_level,
        )
        counts = None
        if collect_eval_counts and hasattr(dp, "get_dispersion_eval_counts"):
            counts = int(dp.get_dispersion_eval_counts()["double"])
        point = SweepPoint(
            frequency_hz=freq,
            period_s=period,
            c_guess=current_guess,
            phase_velocity=float(result[0]),
            group_velocity=float(result[1]),
            hv_ratio=float(result[2]),
            level=int(result[3]),
            relative_error=float(result[4]),
            double_evals=counts,
        )
        out.append(point)
        prev_freqs.append(freq)
        prev_c.append(point.phase_velocity)
        prev_u.append(point.group_velocity)
        if i + 1 < len(freqs):
            current_guess = predict_next_phase_velocity(prev_freqs, prev_c, prev_u, freqs[i + 1])

    return out
