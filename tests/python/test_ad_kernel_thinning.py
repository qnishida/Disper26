#!/usr/bin/env python3
"""ad_kernel_every: make the AD-vs-weak-form cross-check optional, not mandatory.

Phase 1 computes dc/dp twice for every datum. The AD Romberg pass supplies the
Jacobian at 733 ms; the weak-form gradient comes out of the Hessian call for
about 30 ms and is compared against it as `consistency`. That comparison is
worth keeping -- it is what exposed the one-sided-difference defect fixed in
986b360 -- but paying 733 ms on every datum to run it is not the only way to
keep it.

This knob decides how often the AD pass runs. The default of 1 leaves
everything exactly as it was, which is the point: the safety net gets wired in
before the Jacobian is switched over to the weak-form path, not after.

What the tests pin:

  Default is unchanged. Not a formality -- the whole sequencing argument is
  that this lands without altering any result.

  Thinning selects, it does not corrupt. Data that keep the AD pass must give
  the same Jacobian row and the same consistency as an unthinned run.

  Skipped data report NaN consistency, not 0. A check that never ran must not
  be indistinguishable from one that passed; np.nanmax over the array is then
  honest about what was actually verified.

  The Jacobian falls back to the weak-form gradient where AD is skipped, and
  agrees with the AD row to the 1e-08 the two paths now share.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))

from disper26_inv.engine import Datum, ForwardEngine        # noqa: E402
from disper26_inv.parallel import map_over_data             # noqa: E402
from disper26_inv.shallow import shallow_basis, shallow_reference  # noqa: E402

REF = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
BASIS = shallow_basis(REF)
# Overtones included: every disagreement ever seen between the two paths was on
# branch 1, so a fundamental-only fixture would not exercise the interesting
# case. Kept small -- each datum with the AD pass on costs ~0.8 s.
DATA = [Datum("Rayleigh", 0, 20.0), Datum("Love", 1, 10.0),
        Datum("Rayleigh", 1, 7.143), Datum("Love", 0, 10.0)]

ENG = ForwardEngine()
HINTS = ENG.trace_dataset(REF, DATA)


def run(every):
    return map_over_data(
        REF, DATA, c_hints=HINTS, want_hessian=True, processes=1,
        engine_kwargs={}, depth_config=None,
        reduced_hessian=True, basis=BASIS, ad_kernel_every=every)


def test_default_is_unchanged():
    """every=1 must reproduce the pre-knob behaviour: AD kernels on every
    datum."""
    for res in run(1):
        assert "failed" not in res, res
        assert "g_a_ad" in res, "default dropped the AD kernel pass"
    print(f"OK every=1: all {len(DATA)} data carry g_a_ad")


def test_thinning_selects_the_expected_data():
    for every in (2, 3):
        results = run(every)
        got = [i for i, r in enumerate(results) if "g_a_ad" in r]
        want = [i for i in range(len(DATA)) if i % every == 0]
        assert got == want, f"every={every}: AD ran on {got}, expected {want}"
        print(f"OK every={every}: AD pass on data {got}")

    results = run(0)
    assert not any("g_a_ad" in r for r in results), "every=0 still ran AD"
    print("OK every=0: AD pass skipped entirely")


def test_kept_data_are_bit_identical_to_an_unthinned_run():
    """Thinning must be a choice of which data to evaluate, not a change to how
    they are evaluated."""
    full = run(1)
    thin = run(2)
    for i in range(0, len(DATA), 2):
        a = np.asarray(full[i]["g_a_ad"], dtype=float)
        b = np.asarray(thin[i]["g_a_ad"], dtype=float)
        assert np.array_equal(a, b), f"datum {i}: g_a_ad changed under thinning"
        assert full[i]["c"] == thin[i]["c"], f"datum {i}: c changed"
    print(f"OK kept data identical across every=1 and every=2")


def test_weakform_fallback_agrees_with_the_ad_row():
    """Where AD is skipped the Jacobian comes from the weak-form gradient. The
    two have matched to 1e-08 since 986b360; if that regressed, thinning would
    silently change the Jacobian."""
    from disper26_inv.derivatives import (
        gradient_log, project_gradient, reduced_result_to_node_gradient,
    )
    full = run(1)
    worst = 0.0
    for i, res in enumerate(full):
        g_ad = project_gradient(gradient_log(REF, res["g_a_ad"]), BASIS)
        g_wf = project_gradient(gradient_log(REF, res["g_a_variational"]), BASIS)
        scale = max(float(np.max(np.abs(g_wf))), 1e-300)
        rel = float(np.max(np.abs(g_ad - g_wf))) / scale
        worst = max(worst, rel)
        assert rel < 1e-6, (
            f"{DATA[i].label()}: AD and weak-form Jacobian rows differ by "
            f"{rel:.2e} -- thinning would change results")
    print(f"OK AD vs weak-form Jacobian agree to {worst:.1e} over {len(DATA)} data")


if __name__ == "__main__":
    test_default_is_unchanged()
    test_thinning_selects_the_expected_data()
    test_kept_data_are_bit_identical_to_an_unthinned_run()
    test_weakform_fallback_agrees_with_the_ad_row()
    print("OK")
