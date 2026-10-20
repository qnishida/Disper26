#!/usr/bin/env python3
"""Group velocity for models with a water layer.

The AD group velocity used to read uninitialized memory for fluid models. Every
conversion of PhysicalParameters into an AD type copies the six moduli and
leaves the four gravity fields alone, and `T gravity{}` does not initialize an
Eigen AutoDiffScalar -- its user-provided empty default constructor means
value-initialization leaves the value as whatever was on the stack. Gravity is
read only in the fluid branches, where a nonzero value selects a different 2x2
liquid propagator, so solid models never noticed while ocean models returned
1.345, 2.080 or NaN for identical inputs depending on what had run before.

Two things are checked here, because either alone would miss it:

  determinism -- the same call must give the same answer regardless of call
  history. This is what actually failed; a single-shot accuracy check can pass
  by luck.

  accuracy -- against U = dw/dk taken from finite differences of the dispersion
  curve, which shares no code with the AD path.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp  # noqa: E402


RTOL = 1e-6   # AD vs the finite-difference reference; the FD step sets the floor


def iso(vs, vp, rho):
    return dp.create_isotropic_parameters(vs, vp, rho)


def fluid(vp, rho):
    return dp.create_fluid_parameters(vp, rho, 0.0, 1.0, 0.0)


def models():
    """(name, layers, thicknesses, halfspace, periods)"""
    return [
        ("solid", [iso(2.0, 3.5, 2.3), iso(3.0, 5.2, 2.6)], [5.0, 15.0],
         iso(4.0, 6.9, 3.0), (5.0, 10.0, 20.0)),
        ("ocean 4km", [fluid(1.5, 1.03), iso(3.2, 5.8, 2.6)], [4.0, 20.0],
         iso(4.5, 8.1, 3.35), (5.0, 10.0, 20.0)),
        ("ocean 1km", [fluid(1.5, 1.03), iso(3.2, 5.8, 2.6)], [1.0, 20.0],
         iso(4.5, 8.1, 3.35), (5.0, 10.0, 20.0)),
        ("ocean 2 layers", [fluid(1.5, 1.03), fluid(1.52, 1.04),
                            iso(3.2, 5.8, 2.6)], [2.0, 2.0, 20.0],
         iso(4.5, 8.1, 3.35), (8.0, 20.0)),
    ]


def branch_at(layers, thick, halfspace, omega, c_near):
    roots = dp.search_roots(layers, thick, halfspace, omega,
                            c_near * 0.97, c_near * 1.03, 4000, "Rayleigh")
    if not roots:
        return float("nan")
    return min(roots, key=lambda r: abs(r - c_near))


def u_reference(layers, thick, halfspace, omega, c0):
    """U = dw/dk with k = w/c(w), central difference. Independent of the AD path."""
    d = 1e-4 * omega
    c_lo = branch_at(layers, thick, halfspace, omega - d, c0)
    c_hi = branch_at(layers, thick, halfspace, omega + d, c0)
    if math.isnan(c_lo) or math.isnan(c_hi):
        return float("nan")
    return (2.0 * d) / ((omega + d) / c_hi - (omega - d) / c_lo)


def test_group_velocity_matches_dwdk_reference():
    for name, layers, thick, halfspace, periods in models():
        for period in periods:
            omega = 2.0 * math.pi / period
            roots = dp.search_roots(layers, thick, halfspace, omega,
                                    0.5, 4.4, 4000, "Rayleigh")
            assert roots, f"{name}/{period}s: no root found"
            c0 = roots[0]
            u_ad = dp.calculate_group_velocity(layers, thick, halfspace,
                                               omega, c0, "Rayleigh")
            u_ref = u_reference(layers, thick, halfspace, omega, c0)
            assert not math.isnan(u_ad), f"{name}/{period}s: U is NaN"
            rel = abs(u_ad - u_ref) / abs(u_ref)
            assert rel < RTOL, (f"{name}/{period}s: U {u_ad} vs reference "
                                f"{u_ref} (rel {rel:.2e})")
            print(f"OK {name}/{period}s: c={c0:.6f} U={u_ad:.8f} "
                  f"(rel to dw/dk {rel:.1e})")


def test_group_velocity_is_independent_of_call_history():
    """The bug this file exists for. Uninitialized reads gave answers that
    depended on whatever had run before, so the same call has to be made from
    several different preceding states and give one answer."""
    for name, layers, thick, halfspace, periods in models():
        period = periods[0]
        omega = 2.0 * math.pi / period
        c0 = dp.search_roots(layers, thick, halfspace, omega,
                             0.5, 4.4, 4000, "Rayleigh")[0]
        call = lambda: dp.calculate_group_velocity(  # noqa: E731
            layers, thick, halfspace, omega, c0, "Rayleigh")

        observed = [call()]
        # after a pile of root searches
        for w in (omega * 0.999, omega * 1.001, omega * 0.99):
            dp.search_roots(layers, thick, halfspace, w, 0.5, 4.4, 2000, "Rayleigh")
        observed.append(call())
        # after rebuilding the parameter objects
        rebuilt = [iso(p.get_vs(), p.get_vp(), p.rho) if p.c44 > 0
                   else fluid(math.sqrt(p.c11 / p.rho), p.rho) for p in layers]
        dp.calculate_group_velocity(rebuilt, thick, halfspace, omega, c0, "Rayleigh")
        observed.append(call())
        # after an unrelated fluid computation
        dp.calculate_group_velocity(
            [fluid(1.4, 1.0), iso(2.5, 4.3, 2.4)], [3.0, 10.0],
            iso(4.0, 6.9, 3.0), 2.0 * math.pi / 12.0, 2.2, "Rayleigh")
        observed.append(call())

        assert len(set(observed)) == 1, (
            f"{name}: U depends on call history: {observed}")
        print(f"OK {name}: U = {observed[0]!r} from 4 different call histories")


if __name__ == "__main__":
    test_group_velocity_matches_dwdk_reference()
    test_group_velocity_is_independent_of_call_history()
    print("OK")
