#!/usr/bin/env python3
"""disper26_inv.gravity, checked against the fluid Cowling solver.

The Cowling propagator is the only place in the codebase where gravity is
solved for rather than post-corrected, so it is the independent truth for the
fluid part of K_g -- which is the whole reason it was kept when the
non-gravitating solver became the primary path. See docs/energy_integral.md
for the correction conventions.

Two things are pinned here:

  delta_c against c(g) - c(g=0), evaluated through the production entry point
  and the internal fluid Cowling solver.

  delta_u against U(g) - U(g=0).  Group velocity needs no separate kernel, but
  it does need omega d(delta_c)/d omega, which is the dominant term and changes
  sign at the Airy phase.  A delta_u that merely tracked delta_c would pass
  nothing here: the two differ by a factor of -0.7 to +3.4 across the band.

Both are asserted over 3-10 s, where the correction is large enough for a
relative tolerance to mean something.  Past ~20 s the correction is a few
hundredths of a m/s and the Cowling difference is itself at the root-finder's
noise floor, so those periods are covered by an absolute bound instead.

Also pinned: the solid/fluid split is real. Region selection must use the
corresponding kernel contributions, so "solid-only" and "all-layers" cannot
silently evaluate the same correction.
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

import disper26py as dp  # noqa: E402
from disper26_inv.gravity import (  # noqa: E402
    EARTH_GRAVITY_KM_S2,
    rayleigh_gravity_correction,
    rayleigh_gravity_delta_c,
)

THICK = [4.0, 20.0]
HALFSPACE = dp.create_isotropic_parameters(4.5, 8.1, 3.35)
CRUST = dp.create_isotropic_parameters(3.2, 5.8, 2.6)

# Where the fluid correction is large enough that a relative tolerance is
# meaningful: |delta_c| runs 1-7 m/s here and peaks near the 8 s Airy phase.
LOUD_PERIODS = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
QUIET_PERIODS = (20.0, 50.0)

# Worst observed is 1.9%, on delta_u at the 8 s Airy phase -- the peak of
# |delta_c|, where the first-order error in delta_c is amplified by the
# derivative term. Everything else is under 0.9%.
RTOL_LOUD = 0.03
ATOL_QUIET = 5.0e-5     # km/s = 0.05 m/s; the corrections out here are ~0.1 m/s


def ocean(g):
    return [dp.create_fluid_parameters(1.5, 1.03, g, 1.0, 0.0), CRUST]


def solid_only():
    return [dp.create_isotropic_parameters(2.6, 4.6, 2.4), CRUST]


def cowling(period_s, c_near=None):
    """(c, U) from the solver with gravity switched on inside the water."""
    omega = 2.0 * math.pi / period_s
    layers = ocean(EARTH_GRAVITY_KM_S2)
    lo, hi = (0.5, 4.4) if c_near is None else (c_near * 0.99, c_near * 1.01)
    roots = dp.search_roots(layers, THICK, HALFSPACE, omega, lo, hi, 6000, "Rayleigh")
    assert roots, f"{period_s} s: Cowling root search found nothing"
    c = roots[0] if c_near is None else min(roots, key=lambda r: abs(r - c_near))
    return c, dp.calculate_group_velocity(layers, THICK, HALFSPACE, omega, c, "Rayleigh")


def fluid_correction(periods):
    return rayleigh_gravity_correction(ocean(0.0), THICK, HALFSPACE, periods,
                                       gravity=EARTH_GRAVITY_KM_S2, region="fluid")


def test_delta_c_matches_cowling():
    corr = fluid_correction(LOUD_PERIODS)
    for period, c0, dc in zip(LOUD_PERIODS, corr.c, corr.delta_c):
        c_g, _ = cowling(period, c0)
        truth = c_g - c0
        rel = abs(dc - truth) / abs(truth)
        assert rel < RTOL_LOUD, (
            f"{period} s: delta_c {dc*1e3:.3f} m/s vs Cowling {truth*1e3:.3f} m/s "
            f"(rel {rel:.2%})")
        print(f"OK {period:>4.0f} s  delta_c {dc*1e3:>8.3f} m/s  "
              f"Cowling {truth*1e3:>8.3f}  rel {rel:.2%}")


def test_delta_u_matches_cowling():
    corr = fluid_correction(LOUD_PERIODS)
    for period, c0, u0, du in zip(LOUD_PERIODS, corr.c, corr.u, corr.delta_u):
        c_g, u_g = cowling(period, c0)
        truth = u_g - u0
        rel = abs(du - truth) / abs(truth)
        assert rel < RTOL_LOUD, (
            f"{period} s: delta_u {du*1e3:.3f} m/s vs Cowling {truth*1e3:.3f} m/s "
            f"(rel {rel:.2%})")
        print(f"OK {period:>4.0f} s  delta_u {du*1e3:>8.3f} m/s  "
              f"Cowling {truth*1e3:>8.3f}  rel {rel:.2%}")


def test_quiet_periods_stay_small():
    """No relative tolerance out here -- both the correction and the Cowling
    difference are sub-m/s.  The check is that they agree in absolute terms and
    that neither has blown up."""
    corr = fluid_correction(QUIET_PERIODS)
    for period, c0, u0, dc, du in zip(QUIET_PERIODS, corr.c, corr.u,
                                      corr.delta_c, corr.delta_u):
        c_g, u_g = cowling(period, c0)
        assert abs(dc - (c_g - c0)) < ATOL_QUIET, f"{period} s: delta_c off"
        assert abs(du - (u_g - u0)) < ATOL_QUIET, f"{period} s: delta_u off"
        print(f"OK {period:>4.0f} s  delta_c {dc*1e3:>8.3f} m/s  "
              f"delta_u {du*1e3:>8.3f} m/s  (both within {ATOL_QUIET*1e3} m/s)")


def test_omega_derivative_dominates_delta_u():
    """delta_u is not a rescaled delta_c.  If someone ever drops the
    omega-derivative term as a small correction, this fails."""
    corr = fluid_correction(LOUD_PERIODS + QUIET_PERIODS)
    ratio = corr.delta_u / corr.delta_c
    assert ratio.min() < 0.0 < ratio.max(), (
        f"delta_u/delta_c should change sign across the Airy phase, got {ratio}")
    assert np.abs(corr.omega_dc_domega).max() > 1.5 * np.abs(corr.delta_c).max(), (
        "omega d(delta_c)/d omega should dominate delta_c")
    print(f"OK delta_u/delta_c spans {ratio.min():.2f} to {ratio.max():.2f}; "
          f"max |omega d(delta_c)/d omega| = "
          f"{np.abs(corr.omega_dc_domega).max()*1e3:.2f} m/s vs max |delta_c| = "
          f"{np.abs(corr.delta_c).max()*1e3:.2f} m/s")


def test_region_split_is_real():
    """solid + fluid must add up to all, and for an ocean model the fluid part
    must not be the whole thing.  The second half is what the old policy switch
    silently violated."""
    omega = 2.0 * math.pi / 8.0
    layers = ocean(0.0)
    c = dp.search_roots(layers, THICK, HALFSPACE, omega, 0.5, 4.4, 6000, "Rayleigh")[0]
    u = dp.calculate_group_velocity(layers, THICK, HALFSPACE, omega, c, "Rayleigh")

    parts = {r: rayleigh_gravity_delta_c(layers, THICK, HALFSPACE, omega, c, u,
                                         region=r)
             for r in ("all", "solid", "fluid")}
    assert abs(parts["solid"] + parts["fluid"] - parts["all"]) < 1e-12 * abs(parts["all"]), \
        f"solid + fluid != all: {parts}"
    assert abs(parts["fluid"] - parts["all"]) > 0.01 * abs(parts["all"]), (
        f"fluid and all are indistinguishable ({parts}) -- the region split is "
        "not being applied")
    print(f"OK 8 s split: all {parts['all']*1e3:.3f}  fluid {parts['fluid']*1e3:.3f}  "
          f"solid {parts['solid']*1e3:.3f} m/s")


def test_solid_model_has_no_fluid_part():
    omega = 2.0 * math.pi / 10.0
    layers = solid_only()
    c = dp.search_roots(layers, THICK, HALFSPACE, omega, 0.5, 4.4, 6000, "Rayleigh")[0]
    u = dp.calculate_group_velocity(layers, THICK, HALFSPACE, omega, c, "Rayleigh")
    fluid = rayleigh_gravity_delta_c(layers, THICK, HALFSPACE, omega, c, u,
                                     region="fluid")
    every = rayleigh_gravity_delta_c(layers, THICK, HALFSPACE, omega, c, u,
                                     region="all")
    assert fluid == 0.0, f"solid model reported a fluid gravity term: {fluid}"
    assert abs(every) > 0.0, "solid model reported no gravity term at all"
    print(f"OK solid model: fluid part exactly 0, all-layers "
          f"{every*1e3:.3f} m/s")


if __name__ == "__main__":
    test_delta_c_matches_cowling()
    test_delta_u_matches_cowling()
    test_quiet_periods_stay_small()
    test_omega_derivative_dominates_delta_u()
    test_region_split_is_real()
    test_solid_model_has_no_fluid_part()
    print("OK")
