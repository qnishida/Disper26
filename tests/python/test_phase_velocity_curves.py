#!/usr/bin/env python3
"""Numerical, fallback, batching, and GIL tests for phase-velocity curves."""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))

import disper26py as dp  # noqa: E402


def layer_cake_model():
    layers = [
        dp.create_isotropic_parameters(3.15, 5.70, 2.60),
        dp.create_isotropic_parameters(3.55, 6.25, 2.78),
        dp.create_isotropic_parameters(4.05, 7.25, 3.05),
    ]
    thicknesses = [12.0, 24.0, 55.0]
    halfspace = dp.create_isotropic_parameters(4.55, 8.10, 3.35)
    return layers, thicknesses, halfspace


def velocity_vti_grid(scale: float = 1.0, nodes: int = 20):
    depths = np.linspace(0.0, 100.0, nodes).tolist()
    fraction = np.linspace(0.0, 1.0, nodes)
    rhos = (2.50 + 0.50 * fraction).tolist()
    vp = (scale * (5.00 + 2.50 * fraction)).tolist()
    vs = (scale * (3.00 + 1.50 * fraction)).tolist()
    return (
        depths,
        rhos,
        vp,
        vp,
        vs,
        vs,
        [1.0] * nodes,
        [False] * nodes,
    )


def bisect_near(func, centre: float, tolerance: float = 2.0e-14) -> float:
    half_width = 1.0e-4
    for _ in range(16):
        a = centre - half_width
        b = centre + half_width
        fa = func(a)
        fb = func(b)
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb <= 0.0:
            break
        half_width *= 2.0
    else:
        raise AssertionError(f"could not bracket root near {centre}")
    for _ in range(100):
        mid = 0.5 * (a + b)
        fm = func(mid)
        if abs(b - a) <= tolerance * max(1.0, abs(mid)):
            return mid
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def scalar_grid_curve(model, periods, wave, max_steps, c_guess=3.8, branch=None):
    depths, rhos, vphs, vpvs, vshs, vsvs, etas, disc = model
    roots = []
    levels = []
    guess = c_guess
    for period in periods:
        if branch is None:
            value = dp.compute_dispersion_romberg_velocity_vti(
                depths, rhos, vphs, vpvs, vshs, vsvs, etas, disc,
                period, wave, guess, max_steps, 1.0e-10,
                phase_only=True, initial_root_window_frac=0.01,
            )
        else:
            one = dp.compute_dispersion_curve_romberg_grid_velocity_vti(
                depths, rhos, vphs, vpvs, vshs, vsvs, etas, disc,
                [period], wave, c_guess=guess, max_steps=max_steps,
                tolerance=1.0e-10, initial_root_window_frac=0.01,
                branch=branch,
            )
            value = (
                one["phase_velocities"][0], math.nan, math.nan,
                one["converged_levels"][0], one["relative_errors"][0],
            )
        guess = float(value[0])
        roots.append(guess)
        levels.append(int(value[3]))
    return np.asarray(roots), levels


def batch_grid_curve(model, periods, wave, max_steps, c_guess=3.8, branch=None):
    return dp.compute_dispersion_curve_romberg_grid_velocity_vti(
        *model, periods, wave, c_guess=c_guess, max_steps=max_steps,
        tolerance=1.0e-10, initial_root_window_frac=0.01,
        branch=branch,
    )


def test_layer_cake_curve_matches_tight_roots_and_branch():
    layers, thicknesses, halfspace = layer_cake_model()
    periods = [5.0, 6.0, 7.5, 10.0, 15.0, 25.0, 40.0]
    for wave in ("Rayleigh", "Love"):
        curve = dp.compute_dispersion_curve_layer_cake(
            layers, thicknesses, halfspace, periods, wave,
            branch=0, c_min=2.0, c_max=4.5,
            scan_points=1600, root_tolerance=1.0e-12,
        )
        assert curve["periods"] == periods
        assert len(curve["phase_velocities"]) == len(periods)
        assert curve["converged_levels"] == [-1] * len(periods)
        assert all(math.isnan(value) for value in curve["relative_errors"])
        for period, root in zip(periods, curve["phase_velocities"]):
            omega = 2.0 * math.pi / period
            func = lambda c: float(dp.compute_dispersion_value(  # noqa: E731
                layers, thicknesses, halfspace, omega, c, wave
            ))
            tight = bisect_near(func, root)
            assert abs(root - tight) <= 2.0e-12 * max(1.0, abs(tight))
            assert dp.compute_mode_count(
                layers, thicknesses, halfspace, omega, root, wave
            ) == 0

        verified = dp.compute_dispersion_curve_layer_cake(
            layers, thicknesses, halfspace, periods, wave,
            branch=0, c_min=2.0, c_max=4.5,
            scan_points=1600, root_tolerance=1.0e-12,
            verify_branch=True,
        )
        assert np.allclose(
            verified["phase_velocities"], curve["phase_velocities"],
            rtol=0.0, atol=0.0,
        )

        overtone_periods = [5.0, 6.0, 7.0, 8.0, 10.0]
        overtone = dp.compute_dispersion_curve_layer_cake(
            layers, thicknesses, halfspace, overtone_periods, wave,
            branch=1, c_min=2.0, c_max=4.5,
            scan_points=1600, root_tolerance=1.0e-12,
            verify_branch=True,
        )
        references = []
        for period in overtone_periods:
            roots = sorted(dp.search_roots(
                layers, thicknesses, halfspace,
                2.0 * math.pi / period, 2.0, 4.5, 1600, wave,
            ))
            references.append(roots[1])
        assert np.allclose(
            overtone["phase_velocities"], references,
            rtol=1.0e-12, atol=1.0e-12,
        )
        assert overtone["fallback_count"] > 0


def test_grid_curve_matches_scalar_and_reduces_evaluations():
    model = velocity_vti_grid()
    periods = [20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0]
    for wave in ("Rayleigh", "Love"):
        dp.reset_dispersion_eval_counts()
        scalar, scalar_levels = scalar_grid_curve(
            model, periods, wave, max_steps=3
        )
        scalar_evaluations = int(dp.get_dispersion_eval_counts()["double"])

        dp.reset_dispersion_eval_counts()
        batch = batch_grid_curve(model, periods, wave, max_steps=3)
        batch_evaluations = int(dp.get_dispersion_eval_counts()["double"])
        values = np.asarray(batch["phase_velocities"])
        relative = np.max(np.abs(values - scalar) / np.maximum(np.abs(scalar), 1.0e-300))
        assert relative < 1.0e-12, (wave, relative)
        assert batch["converged_levels"] == scalar_levels
        assert batch_evaluations < scalar_evaluations
        assert batch["direct_prediction_count"] > 0


def test_grid_curve_perturbations_and_mode_order():
    periods = [8.0, 10.0, 13.0, 17.0, 22.0, 30.0]
    fallback_total = 0
    for model_index in range(8):
        scale = 1.0 + 8.0e-4 * (model_index - 3.5)
        model = velocity_vti_grid(scale=scale, nodes=12)
        for wave in ("Rayleigh", "Love"):
            scalar, levels = scalar_grid_curve(
                model, periods, wave, max_steps=3
            )
            batch = batch_grid_curve(model, periods, wave, max_steps=3)
            fallback_total += int(batch["fallback_count"])
            values = np.asarray(batch["phase_velocities"])
            relative = np.max(
                np.abs(values - scalar) /
                np.maximum(np.abs(scalar), 1.0e-300)
            )
            assert relative < 1.0e-12, (model_index, wave, relative)
            assert batch["converged_levels"] == levels

    assert fallback_total > 0, "perturbation set did not exercise fallback"

    model = velocity_vti_grid(nodes=8)
    branch_periods = [12.0, 15.0, 20.0]
    scalar, levels = scalar_grid_curve(
        model, branch_periods, "Love", max_steps=2, branch=0
    )
    batch = batch_grid_curve(
        model, branch_periods, "Love", max_steps=2, branch=0
    )
    assert np.allclose(
        batch["phase_velocities"], scalar, rtol=1.0e-12, atol=1.0e-12
    )
    assert batch["converged_levels"] == levels


def test_grid_curve_vti_discontinuity_short_love_and_ocean():
    vti_discontinuity = (
        [0.0, 15.0, 15.0, 45.0, 100.0],
        [2.60, 2.65, 2.90, 3.10, 3.35],
        [5.90, 5.95, 7.05, 7.55, 8.15],
        [5.70, 5.75, 6.80, 7.30, 7.90],
        [3.35, 3.40, 4.05, 4.30, 4.60],
        [3.20, 3.25, 3.85, 4.10, 4.40],
        [0.93, 0.94, 1.03, 1.01, 0.99],
        [False, True, False, False, False],
    )
    shallow = (
        [0.0, 0.03, 0.03, 0.20, 0.20, 1.00],
        [1.80, 1.80, 2.00, 2.00, 2.30, 2.30],
        [0.60, 0.60, 1.20, 1.20, 2.40, 2.40],
        [0.60, 0.60, 1.20, 1.20, 2.40, 2.40],
        [0.20, 0.20, 0.45, 0.45, 1.20, 1.20],
        [0.20, 0.20, 0.45, 0.45, 1.20, 1.20],
        [1.0] * 6,
        [False, True, False, True, False, False],
    )
    ocean = (
        [0.0, 3.0, 3.0, 20.0, 20.0, 100.0],
        [1.03, 1.03, 2.70, 2.80, 3.30, 3.40],
        [1.50, 1.50, 5.80, 6.20, 8.00, 8.20],
        [1.50, 1.50, 5.80, 6.20, 8.00, 8.20],
        [0.00, 0.00, 3.20, 3.50, 4.50, 4.60],
        [0.00, 0.00, 3.20, 3.50, 4.50, 4.60],
        [1.0] * 6,
        [False, True, False, True, False, False],
    )
    cases = [
        (vti_discontinuity, [5.0, 6.0, 8.0, 12.0, 20.0],
         ("Rayleigh", "Love"), 3.3),
        (shallow, [0.05, 0.07, 0.10, 0.20, 0.50],
         ("Love",), 0.3),
        (ocean, [8.0, 10.0, 15.0, 25.0, 40.0],
         ("Rayleigh",), 2.0),
    ]
    for case_index, (model, periods, waves, guess) in enumerate(cases):
        for wave in waves:
            scalar, levels = scalar_grid_curve(
                model, periods, wave, max_steps=3, c_guess=guess
            )
            batch = batch_grid_curve(
                model, periods, wave, max_steps=3, c_guess=guess
            )
            values = np.asarray(batch["phase_velocities"])
            relative = np.max(
                np.abs(values - scalar) /
                np.maximum(np.abs(scalar), 1.0e-300)
            )
            assert relative < 1.0e-12, (case_index, wave, relative)
            assert batch["converged_levels"] == levels


def test_curve_validation_and_gil_release():
    model = velocity_vti_grid(nodes=8)
    try:
        batch_grid_curve(model, [10.0, 12.0, 11.0], "Rayleigh", 2)
    except ValueError as error:
        assert "strictly monotonic" in str(error)
    else:
        raise AssertionError("non-monotonic periods were accepted")

    layers, thicknesses, halfspace = layer_cake_model()
    running = True
    counter = 0
    started = threading.Event()

    def python_worker():
        nonlocal counter
        started.set()
        while running:
            counter += 1

    worker = threading.Thread(target=python_worker)
    worker.start()
    started.wait()
    before = counter
    dp.compute_dispersion_curve_layer_cake(
        layers, thicknesses, halfspace, [10.0], "Rayleigh",
        branch=0, c_min=2.0, c_max=4.5, scan_points=100000,
    )
    after = counter
    running = False
    worker.join()
    assert after > before, "Python worker did not run while C++ held the call"


if __name__ == "__main__":
    test_layer_cake_curve_matches_tight_roots_and_branch()
    test_grid_curve_matches_scalar_and_reduces_evaluations()
    test_grid_curve_perturbations_and_mode_order()
    test_grid_curve_vti_discontinuity_short_love_and_ocean()
    test_curve_validation_and_gil_release()
    print("Phase-velocity curve tests passed.")
