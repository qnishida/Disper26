#!/usr/bin/env python3
"""GRID dc/dp against finite differences, on overtones as well as fundamentals.

The AD kernel path computes dc/dp = -F_p/F_c. F_p came from AD, but F_c came
from a one-sided difference of two independent NormalizationMode::Vertical
calls -- and Vertical divides by the running Y(0), so each call carries its own
c-dependent gauge s(c). Differencing them left s(c)/s(c+eps) in the quotient,
and being a denominator that error scaled every component of the gradient
equally: -3.85e-05 at Love branch 1, T = 10 s, uniform to 1.4% over eighteen
directions. Overtones are where it showed, their eigenfunctions having nodes
where Y(0) passes near zero and s moves quickly; the fundamentals looked clean
because their gauge is slow-varying and, as it turned out, because the gauge
error was partly cancelling the difference's own truncation.

Two things make this test the one that catches such a defect:

  Finite differences of the forward solve are the reference. They share no code
  with either gradient path, so this cannot be satisfied by both paths being
  wrong in the same way -- which is exactly what pipeline.py's `consistency`
  (AD against weak-form) can be.

  Overtones are included. Every outlier in 2805 saved data points was a
  branch-1 datum; a fundamental-only test passes straight through the bug.

The FD reference uses fourth-order central differences in log-parameter space,
computed by Richardson extrapolation of second-order differences. The former
second-order reference at h=1e-4 lost accuracy for the small Rayleigh branch-1
gradient at direction 9: phase velocities near 4.224 km/s differed by only
3.3e-9 km/s, and the AD/FD relative discrepancy reached 7.95e-7. Fourth-order
references at h=1e-3 and 2e-3 agree within 1e-7 across the selected cases and
directions. Check this step-size stability independently before comparing
either gradient path, retaining the existing relative tolerance of 5e-7.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))

from disper26_inv.derivatives import (                      # noqa: E402
    gradient_log, kernels_to_stiffness_gradient, project_gradient,
    reduced_result_to_node_gradient,
)
from disper26_inv.engine import Datum, ForwardEngine        # noqa: E402
from disper26_inv.shallow import shallow_basis, shallow_reference  # noqa: E402

REF = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
BASIS = shallow_basis(REF)
ENG = ForwardEngine()
P = BASIS.n_params
M0 = np.zeros(P)
H_FD = 1e-3

# Overtones first: they are the cases that failed. R1_7.143s and L1_10s are the
# two the investigation used.
CASES = [
    Datum("Love", 1, 10.0),
    Datum("Rayleigh", 1, 7.143),
    Datum("Love", 0, 10.0),
    Datum("Rayleigh", 0, 20.0),
]
DIRS = (0, 3, 6, 9, 12)
RTOL = 5e-7          # Existing tolerance, also required for FD step-size stability.


def fd_gradient(datum, c0, dirs, h=H_FD):
    """Fourth-order reference: (4 D(h) - D(2h)) / 3 for central D."""
    out = {}
    for j in dirs:
        central = []
        for step in (h, 2.0 * h):
            mp, mm = M0.copy(), M0.copy()
            mp[j] += step
            mm[j] -= step
            cp = ENG.phase_velocity(BASIS.m_to_forward_model(mp), datum, c_hint=c0).c
            cm = ENG.phase_velocity(BASIS.m_to_forward_model(mm), datum, c_hint=c0).c
            central.append((cp - cm) / (2.0 * step))
        out[j] = (4.0 * central[0] - central[1]) / 3.0
    return out


def ad_gradient(datum):
    solution, kernels = ENG.phase_velocity_with_kernels(
        REF, datum, minimum_trace_level=ENG.hessian_max_levels
    )
    g_a = kernels_to_stiffness_gradient(kernels, REF.n_nodes)
    return solution, project_gradient(gradient_log(REF, g_a), BASIS)


def weakform_gradient(datum, solution):
    red = ENG.datum_hessian_reduced(
        REF,
        datum,
        solution.c,
        BASIS.full_log_operator(),
        phase_velocity_roots=solution.romberg_roots,
        phase_accepted_level=solution.level,
    )
    g_a = reduced_result_to_node_gradient(REF, red["gradient_node"])
    return project_gradient(gradient_log(REF, g_a), BASIS)


def test_both_gradient_paths_match_finite_differences():
    for datum in CASES:
        solution, g_ad = ad_gradient(datum)
        g_wf = weakform_gradient(datum, solution)
        g_fd = fd_gradient(datum, solution.c, DIRS)
        g_fd_coarse = fd_gradient(datum, solution.c, DIRS, h=2.0 * H_FD)

        worst_ad = worst_wf = worst_step = 0.0
        for j, fd in g_fd.items():
            den = max(abs(fd), 1e-12)
            e_step = abs(fd - g_fd_coarse[j]) / den
            worst_step = max(worst_step, e_step)
            assert e_step < RTOL, (
                f"{datum.label()} dir {j}: fourth-order FD {fd!r} at h={H_FD:g} "
                f"vs {g_fd_coarse[j]!r} at h={2.0 * H_FD:g} "
                f"(step-size rel {e_step:.2e})")
            e_ad = abs(g_ad[j] - fd) / den
            e_wf = abs(g_wf[j] - fd) / den
            worst_ad = max(worst_ad, e_ad)
            worst_wf = max(worst_wf, e_wf)
            assert e_ad < RTOL, (
                f"{datum.label()} dir {j}: AD {g_ad[j]!r} vs FD {fd!r} "
                f"(rel {e_ad:.2e})")
            assert e_wf < RTOL, (
                f"{datum.label()} dir {j}: weak-form {g_wf[j]!r} vs FD {fd!r} "
                f"(rel {e_wf:.2e})")
        print(f"OK {datum.label():>12}  c={solution.c:.8f}  "
              f"worst AD {worst_ad:.1e}, weak-form {worst_wf:.1e}, "
              f"FD step {worst_step:.1e}")


def test_ad_error_is_not_a_uniform_scale_factor():
    """The defect this file exists for showed up as a constant ratio between the
    AD gradient and the truth -- a wrong shared denominator, not per-direction
    noise. Checking the spread of that ratio detects a reintroduction even if
    someone loosens RTOL: a scale error moves the mean while leaving the spread
    tiny, which the per-direction assertions above would flag only once it grew
    past the tolerance."""
    datum = Datum("Love", 1, 10.0)
    solution, g_ad = ad_gradient(datum)
    g_fd = fd_gradient(datum, solution.c, DIRS)

    ratios = np.array([g_ad[j] / g_fd[j] for j in DIRS])
    offset = float(np.abs(ratios.mean() - 1.0))
    assert offset < RTOL, (
        f"{datum.label()}: AD/FD mean ratio is {ratios.mean():.12f}, off by "
        f"{offset:.2e} -- a shared factor (F_c) is wrong, not the parameter pass")
    print(f"OK {datum.label()}: AD/FD ratio {ratios.mean():.12f} "
          f"(spread {ratios.max() - ratios.min():.1e})")


if __name__ == "__main__":
    test_both_gradient_paths_match_finite_differences()
    test_ad_error_is_not_a_uniform_scale_factor()
    print("OK")
