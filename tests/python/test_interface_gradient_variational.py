#!/usr/bin/env python3
"""compute_interface_gradient_variational_grid: dc/dz without the AD pass.

The Jacobian's Depth component -- dc/dz for an interface -- has only ever been
available as the ``Depth`` entry of the AD Romberg kernel list. That made it the
one number blocking a weak-form Jacobian: the AD pass costs 733 ms per datum
against about 30 ms for the weak-form gradient, and keeping it for a single
component would cancel the reason to switch.

The variational route was already there for second order.
compute_interface_hessian_row_variational_ad_grid_extrapolated evaluates K_d on every
refined mesh and Neville-extrapolates the row. First order is the same walk
without the AD: dc/dz_j = sum_b [K_d]_b w_b uses the kernel values, so it runs
in plain double.

What is pinned:

  Agreement with the AD Depth kernel. These are independent implementations of
  the same derivative -- one differentiates the secular function and applies
  the IFT, the other integrates the boundary kernel -- so this is a real
  cross-check, not a restatement.

  Both Rayleigh and Love, and an overtone. Branch 1 is where the AD path's
  defects have shown up.

  Extrapolation matters. A single level carries an O(h^2) truncation
  (~1e-4 relative, per REPORT_KD_ACCURACY_20260807.md), so the level-0 value is
  checked to be visibly worse than the extrapolated one -- otherwise the
  Neville step could be silently doing nothing.
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))

import disper26py as dp                                     # noqa: E402
from disper26_inv.derivatives import kernels_to_depth_gradient  # noqa: E402
from disper26_inv.engine import Datum, ForwardEngine        # noqa: E402
from disper26_inv.model import InterfaceShiftBasis          # noqa: E402
from disper26_inv.shallow import shallow_basis, shallow_reference  # noqa: E402

REF = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
VEL_BASIS = shallow_basis(REF)
PAIR_START = int(np.nonzero(REF.is_discontinuity)[0][1])
DEPTH_BASIS = InterfaceShiftBasis(VEL_BASIS, PAIR_START)
DISC_INDEX = DEPTH_BASIS.discontinuity_index if hasattr(
    DEPTH_BASIS, "discontinuity_index") else 1

ENG = ForwardEngine()
DEPTHS, PARAMS, IS_DISC = REF.to_params()

CASES = [Datum("Rayleigh", 0, 20.0), Datum("Rayleigh", 1, 7.143),
         Datum("Love", 0, 10.0), Datum("Love", 1, 10.0)]

RTOL = 2e-6   # two independent derivations; the AD side is exact to ~1e-8


def variational(
    datum, c, level, disc=DISC_INDEX, roots=None, phase_accepted_level=None
):
    if phase_accepted_level is None:
        phase_accepted_level = level
    return dp.compute_interface_gradient_variational_grid(
        DEPTHS, PARAMS, IS_DISC, datum.period, datum.wave, float(c),
        disc, int(level), phase_velocity_roots=roots,
        phase_accepted_level=int(phase_accepted_level))


def ad_depth_kernel(datum):
    solution, kernels = ENG.phase_velocity_with_kernels(REF, datum)
    return solution, float(kernels_to_depth_gradient(kernels, PAIR_START))


def test_matches_the_ad_depth_kernel():
    worst = 0.0
    for datum in CASES:
        solution, dz_ad = ad_depth_kernel(datum)
        dz_var = variational(
            datum, solution.c, solution.level, roots=solution.romberg_roots
        )
        den = max(abs(dz_ad), 1e-12)
        rel = abs(dz_var - dz_ad) / den
        worst = max(worst, rel)
        assert rel < RTOL, (
            f"{datum.label()}: variational dc/dz {dz_var!r} vs AD {dz_ad!r} "
            f"(rel {rel:.2e})")
        print(f"OK {datum.label():>12}  dc/dz {dz_var:+.10f}  "
              f"AD {dz_ad:+.10f}  rel {rel:.1e}")
    print(f"   worst {worst:.1e} over {len(CASES)} data")


def test_extrapolation_is_doing_something():
    """A single level carries O(h^2); if level 0 were as good as the
    extrapolated value, the Neville step would not be running."""
    datum = CASES[0]
    solution, dz_ad = ad_depth_kernel(datum)
    dz_l0 = variational(
        datum, solution.c, 0, roots=solution.romberg_roots,
        phase_accepted_level=solution.level,
    )
    dz_ext = variational(
        datum, solution.c, solution.level, roots=solution.romberg_roots
    )
    err_l0 = abs(dz_l0 - dz_ad) / abs(dz_ad)
    err_ext = abs(dz_ext - dz_ad) / abs(dz_ad)
    assert err_ext < err_l0 / 10.0, (
        f"extrapolation gained little: level 0 {err_l0:.2e}, "
        f"extrapolated {err_ext:.2e}")
    print(f"OK {datum.label()}: level 0 err {err_l0:.1e} -> "
          f"extrapolated {err_ext:.1e}")


def test_rejects_an_out_of_range_interface():
    datum = CASES[0]
    solution, _ = ad_depth_kernel(datum)
    try:
        variational(
            datum,
            solution.c,
            solution.level,
            disc=999,
            roots=solution.romberg_roots,
        )
    except RuntimeError as exc:
        assert "out of range" in str(exc), f"unhelpful message: {exc}"
        print(f"OK out-of-range interface raises: {str(exc)[:60]}")
    else:
        raise AssertionError("accepted an out-of-range discontinuity_index")


def test_legacy_and_trace_paths_agree_on_a_well_tracked_mode():
    datum = CASES[0]
    solution, _ = ad_depth_kernel(datum)
    traced = variational(
        datum, solution.c, solution.level, roots=solution.romberg_roots
    )
    legacy = variational(datum, solution.c, solution.level)
    assert math.isclose(traced, legacy, rel_tol=2.0e-10, abs_tol=1.0e-12)


if __name__ == "__main__":
    test_matches_the_ad_depth_kernel()
    test_extrapolation_is_doing_something()
    test_rejects_an_out_of_range_interface()
    test_legacy_and_trace_paths_agree_on_a_well_tracked_mode()
    print("OK")
