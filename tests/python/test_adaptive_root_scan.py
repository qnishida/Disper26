#!/usr/bin/env python3
"""
Test the adaptive root scan (adaptive_root_scan kwarg) of the Romberg
dispersion functions.

The adaptive finder must select the same dispersion branch as the legacy
fixed +/-20% / 100-point scan (roots agree to the Brent tolerance) while
spending far fewer plain-double dispersion evaluations.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp
import numpy as np


def model_4node():
    depths = [0.0, 10.0, 20.0, 30.0]
    params = [dp.create_isotropic_parameters(2.5 + 0.05 * z, 4.3 + 0.08 * z, 2.4 + 0.01 * z)
              for z in depths]
    return depths, params, [False] * 4


def model_20node():
    N = 20
    depths = np.linspace(0, 100, N).tolist()
    params = []
    for i in range(N):
        v = 1.0 + 0.5 * (i / (N - 1))
        params.append(dp.create_isotropic_parameters(3.0 * v, 5.0 * v, 2.5 + 0.5 * (i / (N - 1))))
    return depths, params, [False] * N


def model_discontinuity():
    depths = [0.0, 10.0, 10.0, 40.0]
    params = [
        dp.create_isotropic_parameters(2.5, 4.3, 2.4),
        dp.create_isotropic_parameters(2.7, 4.6, 2.5),
        dp.create_isotropic_parameters(3.5, 6.2, 2.8),
        dp.create_isotropic_parameters(4.0, 7.0, 3.0),
    ]
    return depths, params, [False, True, False, False]


def model_ocean():
    ocean = dp.create_fluid_parameters(1.5, 1.03)
    crust_top = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    crust_bottom = dp.create_isotropic_parameters(3.5, 6.2, 2.8)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    depths = [0.0, 3.0, 3.0, 23.0, 23.0]
    params = [ocean, ocean, crust_top, crust_bottom, halfspace]
    return depths, params, [False, True, False, True, False]


def run(depths, params, disc, period, wave, c_guess, adaptive):
    dp.reset_dispersion_eval_counts()
    result = dp.compute_dispersion_romberg(
        depths, params, disc, period, wave, c_guess, 5, 1e-10,
        adaptive_root_scan=adaptive)
    counts = dp.get_dispersion_eval_counts()
    return result, counts


def test_adaptive_matches_legacy():
    models = {
        "4node": (model_4node(), 3.0),
        "20node": (model_20node(), 3.8),
        "disc": (model_discontinuity(), 3.0),
        "ocean": (model_ocean(), None),  # c_guess seeded per period below
    }
    periods = [5.0, 10.0, 20.0, 50.0]

    for name, ((depths, params, disc), c_guess) in models.items():
        for wave in ("Rayleigh", "Love"):
            if name == "ocean" and wave == "Love":
                continue
            for period in periods:
                guess = c_guess
                if guess is None:
                    ocean = dp.create_fluid_parameters(1.5, 1.03)
                    crust = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
                    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
                    roots = dp.search_roots(
                        [ocean, crust], [3.0, 20.0], halfspace,
                        2.0 * math.pi / period, 0.5, 5.0, 500, "Rayleigh")
                    if not roots:
                        continue
                    guess = roots[0]

                ref, ref_counts = run(depths, params, disc, period, wave, guess, False)
                ada, ada_counts = run(depths, params, disc, period, wave, guess, True)

                tag = f"{name}/{wave}/T={period}"
                assert ada[3] == ref[3], f"{tag}: level {ada[3]} != {ref[3]}"
                rel_dc = abs(ada[0] - ref[0]) / abs(ref[0])
                assert rel_dc < 5e-8, f"{tag}: |dc|/c = {rel_dc:.2e}"
                for i, q in ((1, "U"), (2, "HV")):
                    if math.isnan(ref[i]):
                        assert math.isnan(ada[i]), f"{tag}: {q} NaN mismatch"
                    else:
                        rel = abs(ada[i] - ref[i]) / max(abs(ref[i]), 1e-12)
                        assert rel < 1e-7, f"{tag}: {q} rel diff {rel:.2e}"
                print(f"adaptive OK: {tag} c={ada[0]:.8f} "
                      f"double evals {ref_counts['double']} -> {ada_counts['double']}")


def test_adaptive_overtone_branch():
    """Track a higher mode: seed c_guess from a root sweep and verify both
    scanners stay on the same branch."""
    depths, params, disc = model_4node()
    layers = [dp.create_isotropic_parameters(2.5 + 0.05 * z, 4.3 + 0.08 * z, 2.4 + 0.01 * z)
              for z in depths[:-1]]
    thicknesses = [10.0, 10.0, 10.0]
    halfspace = dp.create_isotropic_parameters(4.0, 6.7, 2.7)
    period = 5.0
    roots = dp.search_roots(layers, thicknesses, halfspace,
                            2.0 * math.pi / period, 2.0, 4.5, 800, "Rayleigh")
    assert len(roots) >= 2, f"need an overtone, got roots={roots}"

    for mode_idx, c_guess in enumerate(roots[:3]):
        ref, _ = run(depths, params, disc, period, "Rayleigh", c_guess, False)
        ada, _ = run(depths, params, disc, period, "Rayleigh", c_guess, True)
        rel_dc = abs(ada[0] - ref[0]) / abs(ref[0])
        assert ada[3] == ref[3], f"mode {mode_idx}: level differs"
        assert rel_dc < 5e-8, f"mode {mode_idx}: |dc|/c = {rel_dc:.2e}"
        print(f"adaptive OK: overtone branch {mode_idx} c={ada[0]:.8f}")


def test_adaptive_eval_reduction():
    """Regression: the adaptive scan must cut double evaluations by >= 50%
    on the 20-node / 20 s reference case."""
    depths, params, disc = model_20node()
    _, ref_counts = run(depths, params, disc, 20.0, "Rayleigh", 3.8, False)
    ada, ada_counts = run(depths, params, disc, 20.0, "Rayleigh", 3.8, True)
    ratio = ada_counts["double"] / ref_counts["double"]
    print(f"20node/20s double evals: {ref_counts['double']} -> {ada_counts['double']} "
          f"(ratio {ratio:.2f})")
    assert ratio <= 0.5, f"expected >=50% eval reduction, got ratio {ratio:.2f}"


def test_initial_root_window_uses_sweep_guess():
    """A predictor-quality c_guess should be able to avoid the level-0 full scan."""
    depths, params, disc = model_20node()
    period = 20.0
    ref, ref_counts = run(depths, params, disc, period, "Rayleigh", 3.8, True)

    dp.reset_dispersion_eval_counts()
    sweep = dp.compute_dispersion_romberg(
        depths, params, disc, period, "Rayleigh", ref[0], 5, 1e-10,
        phase_only=True, adaptive_root_scan=True, initial_root_window_frac=0.01)
    counts = dp.get_dispersion_eval_counts()

    rel_dc = abs(sweep[0] - ref[0]) / abs(ref[0])
    assert rel_dc < 5e-8, f"sweep window changed root: |dc|/c={rel_dc:.2e}"
    assert counts["double"] < ref_counts["double"], (
        f"expected fewer double evals, got {counts['double']} >= {ref_counts['double']}"
    )
    assert math.isnan(sweep[1]) and math.isnan(sweep[2])
    print(f"sweep window OK: double evals {ref_counts['double']} -> {counts['double']}")


if __name__ == "__main__":
    test_adaptive_matches_legacy()
    test_adaptive_overtone_branch()
    test_adaptive_eval_reduction()
    test_initial_root_window_uses_sweep_guess()
    print("All adaptive-root-scan tests passed.")
