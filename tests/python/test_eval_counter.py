#!/usr/bin/env python3
"""
Test the dispersion-function evaluation counters
(get_dispersion_eval_counts / reset_dispersion_eval_counts).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp


def make_model():
    depths = [0.0, 10.0, 20.0, 30.0]
    params = []
    for z in depths:
        vs = 2.5 + 0.05 * z
        vp = 4.3 + 0.08 * z
        rho = 2.4 + 0.01 * z
        params.append(dp.create_isotropic_parameters(vs, vp, rho))
    is_discontinuity = [False] * len(depths)
    return depths, params, is_discontinuity


def test_counter_counts_and_resets():
    depths, params, is_discontinuity = make_model()

    dp.reset_dispersion_eval_counts()
    counts = dp.get_dispersion_eval_counts()
    assert counts["total"] == 0
    assert counts["double"] == 0
    assert counts["ad"] == 0

    result = dp.compute_dispersion_romberg(
        depths, params, is_discontinuity, 20.0, "Rayleigh", 3.0, 5, 1e-10)
    assert result[0] > 0

    counts = dp.get_dispersion_eval_counts()
    print(f"Romberg (Rayleigh, 4-node): total={counts['total']} "
          f"double={counts['double']} ad={counts['ad']}")
    # Root scan uses double evaluations; group velocity uses AD evaluations.
    assert counts["total"] > 0
    assert counts["double"] > 0
    assert counts["ad"] > 0
    assert counts["total"] == counts["double"] + counts["ad"]

    dp.reset_dispersion_eval_counts()
    counts = dp.get_dispersion_eval_counts()
    assert counts["total"] == 0
    assert counts["double"] == 0
    assert counts["ad"] == 0


def test_counter_love():
    depths, params, is_discontinuity = make_model()

    dp.reset_dispersion_eval_counts()
    result = dp.compute_dispersion_romberg(
        depths, params, is_discontinuity, 20.0, "Love", 3.0, 5, 1e-10)
    assert result[0] > 0

    counts = dp.get_dispersion_eval_counts()
    print(f"Romberg (Love, 4-node): total={counts['total']} "
          f"double={counts['double']} ad={counts['ad']}")
    assert counts["double"] > 0
    assert counts["ad"] > 0


if __name__ == "__main__":
    test_counter_counts_and_resets()
    test_counter_love()
    print("All eval-counter tests passed.")
