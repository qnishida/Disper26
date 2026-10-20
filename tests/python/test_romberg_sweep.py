#!/usr/bin/env python3
"""Tests for the Python-side Romberg frequency-sweep wrapper."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "python"))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))

import disper26py as dp  # noqa: E402
from romberg_sweep import compute_dispersion_romberg_frequency_sweep  # noqa: E402


def model_20node():
    depths = np.linspace(0.0, 100.0, 20).tolist()
    params = []
    for i in range(len(depths)):
        v = 1.0 + 0.5 * (i / (len(depths) - 1))
        params.append(dp.create_isotropic_parameters(3.0 * v, 5.0 * v, 2.5 + 0.5 * (i / (len(depths) - 1))))
    return depths, params, [False] * len(depths)


def test_frequency_sweep_matches_scalar_calls():
    depths, params, disc = model_20node()
    freqs = [0.05, 0.055, 0.06, 0.065, 0.07]
    sweep = compute_dispersion_romberg_frequency_sweep(
        dp,
        depths,
        params,
        disc,
        freqs,
        "Rayleigh",
        3.8,
        5,
        1e-10,
        phase_only=True,
    )

    scalar = [
        dp.compute_dispersion_romberg(
            depths,
            params,
            disc,
            1.0 / f,
            "Rayleigh",
            3.8,
            5,
            1e-10,
            phase_only=True,
        )[0]
        for f in freqs
    ]

    for point, c_ref in zip(sweep, scalar):
        rel = abs(point.phase_velocity - c_ref) / max(abs(c_ref), 1e-12)
        assert rel < 1e-8, f"frequency {point.frequency_hz}: rel diff {rel:.2e}"
        assert math.isnan(point.group_velocity)
        assert math.isnan(point.hv_ratio)


def test_frequency_sweep_reduces_eval_count_with_good_predictor():
    depths, params, disc = model_20node()
    freqs = [0.05, 0.055, 0.06, 0.065, 0.07]

    scalar_evals = 0
    for f in freqs:
        dp.reset_dispersion_eval_counts()
        dp.compute_dispersion_romberg(
            depths,
            params,
            disc,
            1.0 / f,
            "Rayleigh",
            3.8,
            5,
            1e-10,
            phase_only=True,
        )
        scalar_evals += int(dp.get_dispersion_eval_counts()["double"])

    sweep = compute_dispersion_romberg_frequency_sweep(
        dp,
        depths,
        params,
        disc,
        freqs,
        "Rayleigh",
        3.8,
        5,
        1e-10,
        phase_only=True,
    )
    sweep_evals = sum(point.double_evals or 0 for point in sweep)
    assert sweep_evals < scalar_evals, f"expected fewer evals, got {sweep_evals} >= {scalar_evals}"
    print(f"frequency sweep evals: scalar={scalar_evals} sweep={sweep_evals}")


if __name__ == "__main__":
    test_frequency_sweep_matches_scalar_calls()
    test_frequency_sweep_reduces_eval_count_with_good_predictor()
    print("Romberg frequency-sweep tests passed.")
