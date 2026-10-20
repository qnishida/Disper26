#!/usr/bin/env python3
"""
Test the phase_only fast path of the Romberg dispersion functions.

phase_only=True must return bit-identical c / converged_level / rel_error
to the default path (the root iteration never depends on U/HV), with
U and HV returned as NaN and no AD evaluations spent per level.
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


def ocean_c_guess():
    ocean = dp.create_fluid_parameters(1.5, 1.03)
    crust = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    roots = dp.search_roots([ocean, crust], [3.0, 20.0], halfspace,
                            2.0 * math.pi / 20.0, 0.5, 5.0, 500, "Rayleigh")
    assert roots, "no ocean Rayleigh root found for c_guess seed"
    return roots[0]


def make_models():
    return {
        "4node": (model_4node, 3.0),
        "20node": (model_20node, 3.8),
        "disc": (model_discontinuity, 3.0),
        "ocean": (model_ocean, ocean_c_guess()),
    }


def test_phase_only_matches_default():
    for name, (make, c_guess) in make_models().items():
        depths, params, disc = make()
        for wave in ("Rayleigh", "Love"):
            if name == "ocean" and wave == "Love":
                continue  # Love has no fluid-top solution in this setup
            ref = dp.compute_dispersion_romberg(
                depths, params, disc, 20.0, wave, c_guess, 5, 1e-10)

            dp.reset_dispersion_eval_counts()
            po = dp.compute_dispersion_romberg(
                depths, params, disc, 20.0, wave, c_guess, 5, 1e-10,
                phase_only=True)
            counts = dp.get_dispersion_eval_counts()

            assert po[0] == ref[0], f"{name}/{wave}: c differs"
            assert po[3] == ref[3], f"{name}/{wave}: level differs"
            assert po[4] == ref[4], f"{name}/{wave}: rel_error differs"
            assert math.isnan(po[1]), f"{name}/{wave}: U should be NaN"
            assert math.isnan(po[2]), f"{name}/{wave}: HV should be NaN"
            # No group-velocity AD passes in phase-only mode.
            assert counts["ad"] == 0, f"{name}/{wave}: expected 0 AD evals, got {counts['ad']}"
            print(f"phase_only OK: {name}/{wave} c={po[0]:.8f} level={po[3]} "
                  f"double_evals={counts['double']}")


def test_extrapolate_start_level_skips_coarse_table_entry():
    depths, params, disc = model_20node()
    ref = dp.compute_dispersion_romberg(
        depths, params, disc, 20.0, "Rayleigh", 3.8, 7, 1e-12,
        phase_only=True)
    skipped = dp.compute_dispersion_romberg(
        depths, params, disc, 20.0, "Rayleigh", 3.8, 5, 1e-6,
        phase_only=True, extrapolate_start_level=1)

    rel_dc = abs(skipped[0] - ref[0]) / max(abs(ref[0]), 1e-12)
    assert rel_dc < 1e-6, f"level-0 skip lost accuracy: |dc|/c={rel_dc:.2e}"
    assert skipped[3] <= 4, f"expected early stop, got level {skipped[3]}"
    assert math.isnan(skipped[1]) and math.isnan(skipped[2])
    print(f"extrapolate_start_level OK: c={skipped[0]:.8f} level={skipped[3]}")


def test_phase_only_velocity_variants():
    depths, params, disc = model_4node()
    rhos = [2.4 + 0.01 * z for z in depths]
    vps = [4.3 + 0.08 * z for z in depths]
    vss = [2.5 + 0.05 * z for z in depths]
    etas = [1.0] * len(depths)

    ref = dp.compute_dispersion_romberg_velocity_iso(
        depths, rhos, vps, vss, disc, 20.0, "Rayleigh", 3.0, 5, 1e-10)
    po = dp.compute_dispersion_romberg_velocity_iso(
        depths, rhos, vps, vss, disc, 20.0, "Rayleigh", 3.0, 5, 1e-10,
        phase_only=True)
    assert po[0] == ref[0] and po[3] == ref[3] and po[4] == ref[4]
    assert math.isnan(po[1]) and math.isnan(po[2])

    ref = dp.compute_dispersion_romberg_velocity_vti(
        depths, rhos, vps, vps, vss, vss, etas, disc, 20.0, "Rayleigh", 3.0, 5, 1e-10)
    po = dp.compute_dispersion_romberg_velocity_vti(
        depths, rhos, vps, vps, vss, vss, etas, disc, 20.0, "Rayleigh", 3.0, 5, 1e-10,
        phase_only=True)
    assert po[0] == ref[0] and po[3] == ref[3] and po[4] == ref[4]
    assert math.isnan(po[1]) and math.isnan(po[2])
    print("phase_only OK: velocity iso/vti variants")


def test_phase_only_with_kernels():
    depths, params, disc = model_4node()

    # phase_only + U/HV kernels is contradictory and must raise.
    for kwargs in ({"want_kernel_u": True, "want_kernel_hv": False},
                   {"want_kernel_u": False, "want_kernel_hv": True}):
        try:
            dp.compute_dispersion_romberg_with_kernels(
                depths, params, disc, 20.0, "Rayleigh", 3.0, "Isotropic", 5, 1e-10,
                compute_c=True, phase_only=True, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for phase_only with {kwargs}")

    # phase_only c-kernels must match the default path's c-kernels.
    ref = dp.compute_dispersion_romberg_with_kernels(
        depths, params, disc, 20.0, "Rayleigh", 3.0, "Isotropic", 5, 1e-10)
    po = dp.compute_dispersion_romberg_with_kernels(
        depths, params, disc, 20.0, "Rayleigh", 3.0, "Isotropic", 5, 1e-10,
        compute_c=True, want_kernel_u=False, want_kernel_hv=False, phase_only=True)
    assert po[0] == ref[0] and po[3] == ref[3]
    assert math.isnan(po[1]) and math.isnan(po[2])
    assert len(po[5]) == len(ref[5])
    for k_po, k_ref in zip(po[5], ref[5]):
        assert k_po.d_c == k_ref.d_c, "phase_only changed c-kernels"
    print("phase_only OK: with_kernels validation and c-kernel equality")


if __name__ == "__main__":
    test_phase_only_matches_default()
    test_extrapolate_start_level_skips_coarse_table_entry()
    test_phase_only_velocity_variants()
    test_phase_only_with_kernels()
    print("All phase_only tests passed.")
