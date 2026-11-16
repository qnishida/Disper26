#!/usr/bin/env python3
"""Equivalence harness for changes to the per-layer phase-velocity kernel.

Anything that touches how the layer loop is evaluated -- compiler flags for
complex arithmetic, shortcuts in the transcendentals, a different carrier type
-- has to leave the algebra alone. The acceptance criterion is |dc|/c < 1e-12
on the Romberg phase velocity.

Most such changes are baked in at compile time, so the comparison runs across
two builds rather than in one process:

  --dump FILE     compute the whole case table and write it as JSON
  --compare FILE  recompute and diff against a dumped table

Dump with the old build (point DISPER26_SO_DIR at a directory holding its .so),
rebuild, then compare. Running with no arguments just computes the table and
reports that there is nothing to diff against.

Every case reports the layer_cake secular function F(c) on a c-grid (a direct,
branch-free probe of the layer loop) and the grid-mode Romberg c/U/HV.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
# Point this at a directory holding another disper26py .so to run the table
# against a different build (see --dump/--compare below).
if os.environ.get("DISPER26_SO_DIR"):
    sys.path.insert(0, os.environ["DISPER26_SO_DIR"])
import disper26py as dp  # noqa: E402


C_RTOL = 1e-12          # agreed acceptance criterion on the Romberg root
F_ATOL_SCALE = 1e-11    # F(c) is a secular function: compare against its own scale
DERIVED_RTOL = 1e-9     # U and HV, once they have settled with refinement depth
DERIVED_CONVERGED_RTOL = 1e-8   # "settled" = moves less than this over +2 levels

MAX_STEPS = 6
TOLERANCE = 1e-10


# --------------------------------------------------------------------------
# model suite
# --------------------------------------------------------------------------

def iso(vs, vp, rho):
    return dp.create_isotropic_parameters(vs, vp, rho)


def vti(c11, c13, c33, c44, c66, rho):
    p = dp.PhysicalParameters()
    p.rho = rho
    p.c11, p.c13, p.c33, p.c44, p.c66 = c11, c13, c33, c44, c66
    p.gravity = 0.0
    p.gravity_internal_scale = 1.0
    p.gravity_surface_scale = 0.0
    return p


def fluid(vp, rho):
    return dp.create_fluid_parameters(vp, rho, 0.0, 1.0, 0.0)


# This used to exempt ocean_crust from the U/HV comparison, because its group
# velocity moved by up to 1.7e-5 between builds that produced a bit-identical c.
# That was not sensitivity to codegen: the AD group velocity was reading
# uninitialized gravity fields for fluid models, so the answer depended on what
# had run before. With that fixed the ocean U converges with refinement depth
# like any other model, so the exemption is gone and nothing is exempt.
UNSTABLE_DERIVED = set()


def grid_models():
    """(name, depths, params, is_discontinuity, waves, periods, c_guesses)."""
    out = []

    # (a) smooth isotropic gradient
    n = 20
    depths = [100.0 * i / (n - 1) for i in range(n)]
    params = [iso(3.0 + 1.5 * i / (n - 1), 5.0 + 2.5 * i / (n - 1),
                  2.5 + 0.5 * i / (n - 1)) for i in range(n)]
    out.append(("iso_gradient", depths, params, [False] * n,
                ("Rayleigh", "Love"), (5.0, 20.0, 50.0, 100.0),
                (3.3, 3.8, 4.2)))

    # (b) PREM-like crust/mantle with two first-order discontinuities
    depths = [0.0, 15.0, 15.0, 24.4, 24.4, 80.0, 220.0, 400.0]
    params = [iso(3.20, 5.80, 2.60), iso(3.20, 5.80, 2.60),
              iso(3.90, 6.80, 2.90), iso(3.90, 6.80, 2.90),
              iso(4.49, 8.11, 3.38), iso(4.49, 8.08, 3.37),
              iso(4.42, 8.56, 3.44), iso(4.93, 8.91, 3.54)]
    is_disc = [False, True, False, True, False, False, False, False]
    out.append(("prem_like", depths, params, is_disc,
                ("Rayleigh", "Love"), (20.0, 50.0, 100.0, 200.0),
                (3.5, 4.0, 4.5)))

    # (c) shallow sediment over bedrock, engineering-scale periods
    depths = [0.0, 0.03, 0.03, 0.20, 0.20, 1.00]
    params = [iso(0.20, 0.60, 1.80), iso(0.20, 0.60, 1.80),
              iso(0.45, 1.20, 2.00), iso(0.45, 1.20, 2.00),
              iso(1.20, 2.40, 2.30), iso(1.20, 2.40, 2.30)]
    is_disc = [False, True, False, True, False, False]
    out.append(("shallow_sediment", depths, params, is_disc,
                ("Rayleigh", "Love"), (0.05, 0.1, 0.3, 1.0),
                (0.25, 0.5, 0.9)))

    # (d) low-velocity zone (non-monotone profile, awkward branches)
    depths = [0.0, 10.0, 10.0, 30.0, 30.0, 60.0, 60.0, 200.0]
    params = [iso(3.30, 5.90, 2.65), iso(3.30, 5.90, 2.65),
              iso(3.85, 6.75, 2.90), iso(3.85, 6.75, 2.90),
              iso(3.30, 6.10, 2.95), iso(3.30, 6.10, 2.95),
              iso(4.55, 8.10, 3.35), iso(4.70, 8.30, 3.40)]
    is_disc = [False, True, False, True, False, True, False, False]
    out.append(("lvz", depths, params, is_disc,
                ("Rayleigh", "Love"), (10.0, 25.0, 60.0),
                (3.4, 3.9, 4.3)))

    # (e) VTI, moderate anisotropy (real discriminant, exercises c11 != c33)
    def vti_iso_like(vs, vp, rho, aniso):
        c33 = rho * vp * vp
        c11 = c33 * (1.0 + aniso)
        c44 = rho * vs * vs
        c66 = c44 * (1.0 + aniso)
        c13 = c33 - 2.0 * c44
        return vti(c11, c13, c33, c44, c66, rho)

    depths = [0.0, 20.0, 20.0, 120.0]
    params = [vti_iso_like(3.4, 6.0, 2.7, 0.10),
              vti_iso_like(3.4, 6.0, 2.7, 0.10),
              vti_iso_like(4.5, 8.1, 3.35, 0.06),
              vti_iso_like(4.6, 8.3, 3.40, 0.06)]
    out.append(("vti_moderate", depths, params, [False, True, False, False],
                ("Rayleigh", "Love"), (10.0, 30.0, 80.0),
                (3.6, 4.2, 4.6)))

    # (f) ocean layer over solid crust -- Rayleigh (Scholte) only
    depths = [0.0, 4.0, 4.0, 20.0, 20.0, 150.0]
    params = [fluid(1.50, 1.03), fluid(1.50, 1.03),
              iso(3.20, 5.80, 2.60), iso(3.20, 5.80, 2.60),
              iso(4.50, 8.10, 3.35), iso(4.65, 8.30, 3.40)]
    out.append(("ocean_crust", depths, params,
                [False, True, False, True, False, False],
                ("Rayleigh",), (10.0, 25.0, 60.0),
                (2.05, 3.35, 3.58)))

    return out


def layer_cake_stacks():
    """(name, layers, thicknesses, halfspace, waves, periods, c_range).

    Fixed layer stacks fed straight to calculate_dispersion_value. This probes
    the layer loop without any root finding, so it cannot be confounded by a
    branch flip. Includes a strongly anisotropic case whose Christoffel
    discriminant goes negative -- the real-carrier kernel must fall back to
    complex arithmetic there and still agree.
    """
    out = []

    n = 60
    layers = [iso(3.0 + 1.5 * i / n, 5.0 + 2.5 * i / n, 2.5 + 0.5 * i / n)
              for i in range(n)]
    out.append(("iso_stack", layers, [100.0 / n] * n, iso(4.8, 8.5, 3.5),
                ("Rayleigh", "Love"), (5.0, 20.0, 80.0), (2.5, 5.0)))

    layers = [iso(0.20, 0.60, 1.80)] * 6 + [iso(0.45, 1.20, 2.00)] * 10
    thick = [0.005] * 6 + [0.017] * 10
    out.append(("shallow_stack", layers, thick, iso(1.20, 2.40, 2.30),
                ("Rayleigh", "Love"), (0.05, 0.2, 0.8), (0.15, 1.3)))

    layers = [fluid(1.50, 1.03)] * 4 + [iso(3.20, 5.80, 2.60)] * 8
    thick = [1.0] * 4 + [2.0] * 8
    out.append(("ocean_stack", layers, thick, iso(4.50, 8.10, 3.35),
                ("Rayleigh",), (5.0, 20.0, 60.0), (1.2, 4.4)))

    # Strong VTI, chosen so the Christoffel discriminant goes negative and nu1,
    # nu3 become a genuine complex-conjugate pair -- the case ParityReal cannot
    # represent and has to hand back to complex arithmetic.
    #
    # Writing X = rho w^2 - c44 k^2, Y = rho w^2 - c11 k^2, A = c44 X, B = c33 Y
    # and F = (c13+c44)^2 k^2, the discriminant is (A-B)^2 + F^2 + 2F(A+B). For
    # c below both wave speeds A+B < 0, and as a quadratic in F it is negative
    # between its roots, i.e. for (sqrt(u)-sqrt(w))^2 < F < (sqrt(u)+sqrt(w))^2
    # with u = -A, w = -B. An unusually large c13 puts F in that window: with
    # c11 = c33 = 30 and c44 = 6 it needs 18 < c13 < 30, and isotropy would
    # give exactly the lower edge, c13 = c33 - 2*c44 = 18. c13 = 24 sits inside.
    # Verified negative for c below about 1.35 and positive above, so the sweep
    # below crosses the boundary and exercises both paths.
    strong = vti(c11=30.0, c13=24.0, c33=30.0, c44=6.0, c66=6.0, rho=3.0)
    strong2 = vti(c11=34.0, c13=26.0, c33=28.0, c44=5.0, c66=6.5, rho=3.1)
    layers = [strong] * 6 + [strong2] * 6
    out.append(("vti_strong_stack", layers, [3.0] * 12,
                vti(c11=40.0, c13=20.0, c33=40.0, c44=12.0, c66=13.0,
                    rho=3.3),
                ("Rayleigh", "Love"), (5.0, 20.0, 60.0), (1.0, 6.0)))

    return out


# --------------------------------------------------------------------------
# case table
# --------------------------------------------------------------------------

def build_table():
    """Compute every case. Returns {case_key: [values...]}."""
    table = {}

    for name, layers, thick, hs, waves, periods, (c_lo, c_hi) in layer_cake_stacks():
        for wave in waves:
            for period in periods:
                omega = 2.0 * math.pi / period
                vals = []
                for i in range(41):
                    # Offset by an odd fraction so no sample lands exactly on a
                    # layer velocity. At c == Vs of some layer the secular
                    # function is genuinely singular (nu3 = 0, d5/d6 -> 0) and
                    # whether it evaluates to NaN or to a huge finite number is
                    # decided by the last bit, which would make this a test of
                    # round-off rather than of the kernel.
                    c = c_lo + (c_hi - c_lo) * (i + 0.317) / 41.0
                    vals.append(dp.calculate_dispersion_value(
                        layers, thick, hs, omega, c, wave))
                table[f"F|{name}|{wave}|{period}"] = vals

    def _converged(shallow, deep):
        """Has this quantity stopped moving between MAX_STEPS and MAX_STEPS+2?"""
        if math.isnan(shallow) and math.isnan(deep):
            return True
        if math.isnan(shallow) or math.isnan(deep):
            return False
        return abs(shallow - deep) <= DERIVED_CONVERGED_RTOL * max(abs(deep), 1e-30)

    rooted = 0
    total = 0
    for name, depths, params, is_disc, waves, periods, guesses in grid_models():
        for wave in waves:
            for period in periods:
                for c_guess in guesses:
                    key = f"R|{name}|{wave}|{period}|{c_guess}"
                    total += 1
                    try:
                        r = dp.compute_dispersion_romberg(
                            depths, params, is_disc, period, wave, c_guess,
                            MAX_STEPS, TOLERANCE)
                    except Exception as exc:  # noqa: BLE001
                        table[key] = str(exc)
                        continue
                    # When the scan finds nothing the Romberg core hands back
                    # c_guess untouched, and U/HV are then just the group
                    # velocity at an arbitrary velocity -- ill-conditioned
                    # numbers that would make this a test of round-off. Record
                    # the status so a build that starts or stops finding the
                    # root is still caught, but do not compare the values.
                    if abs(r[0] - c_guess) <= 1e-14 * max(1.0, abs(c_guess)):
                        table[key] = "noroot"
                        continue
                    rooted += 1
                    # U and HV are Neville extrapolations of their own level
                    # tables, and those converge far later than c does. The
                    # ocean model at 10 s reports rel_err 1.9e-9 -- that is c's
                    # convergence -- while its U is still 1.5e-3 from the value
                    # it settles on two levels later. Comparing an unconverged
                    # extrapolation across builds measures truncation error
                    # amplifying a ulp, not the kernel. So measure U's own
                    # convergence by re-running deeper, and only compare U/HV
                    # where they have actually settled.
                    deeper = dp.compute_dispersion_romberg(
                        depths, params, is_disc, period, wave, c_guess,
                        MAX_STEPS + 2, TOLERANCE)
                    settled = [
                        _converged(r[i], deeper[i]) for i in (1, 2)
                    ]
                    comparable = name not in UNSTABLE_DERIVED
                    # c, U, HV, level, U comparable, HV comparable
                    table[key] = [r[0], r[1], r[2], float(r[3]),
                                  settled[0] and comparable,
                                  settled[1] and comparable]

    if rooted < 0.7 * total:
        raise SystemExit(
            f"only {rooted}/{total} Romberg cases found a root; the c_guess "
            "values in grid_models() need revisiting -- comparing 'noroot' "
            "statuses alone would not exercise the kernel")

    return table


def compare(ref, cur):
    """Diff two case tables. Returns (n_checked, failures, worst)."""
    failures = []
    worst = {"F": (0.0, None), "c": (0.0, None), "U": (0.0, None),
             "HV": (0.0, None)}
    unsettled = {"U": 0, "HV": 0}
    n_checked = 0

    missing = set(ref) ^ set(cur)
    if missing:
        failures.append(f"case set differs: {sorted(missing)[:5]}")

    for key in sorted(set(ref) & set(cur)):
        a, b = ref[key], cur[key]
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                failures.append(f"{key}: exception mismatch {a!r} vs {b!r}")
            continue

        if key.startswith("F|"):
            scale = max(abs(v) for v in a) or 1.0
            for i, (x, y) in enumerate(zip(a, b)):
                n_checked += 1
                if math.isnan(x) or math.isnan(y):
                    if not (math.isnan(x) and math.isnan(y)):
                        failures.append(f"{key}[{i}]: NaN mismatch "
                                        f"{x!r} vs {y!r}")
                    continue
                d = abs(x - y) / scale
                if d > worst["F"][0]:
                    worst["F"] = (d, f"{key}[{i}]")
                if d > F_ATOL_SCALE:
                    failures.append(f"{key}[{i}]: F {x!r} vs {y!r} "
                                    f"(rel-to-scale {d:.3e})")
        else:
            # a[4], a[5] say whether U and HV are worth comparing: they had
            # stopped moving with refinement depth, and their model is not one
            # of UNSTABLE_DERIVED. Otherwise comparing them across builds
            # measures how a ulp gets amplified, not the kernel. c is always
            # held to the flat acceptance criterion, for every model.
            settled = {
                "U": bool(a[4]) and bool(b[4]),
                "HV": bool(a[5]) and bool(b[5]),
            } if len(a) > 5 and len(b) > 5 else {"U": True, "HV": True}
            for label, x, y in zip(("c", "U", "HV"), a[:3], b[:3]):
                if label in settled and not settled[label]:
                    unsettled[label] += 1
                    continue
                n_checked += 1
                if math.isnan(x) or math.isnan(y):
                    if not (math.isnan(x) and math.isnan(y)):
                        failures.append(f"{key}: {label} NaN mismatch "
                                        f"{x!r} vs {y!r}")
                    continue
                d = abs(x - y) / max(abs(x), 1e-30)
                if d > worst[label][0]:
                    worst[label] = (d, key)
                if label == "c":
                    if d > C_RTOL:
                        failures.append(f"{key}: c {x!r} vs {y!r} (rel {d:.3e})")
                elif d > DERIVED_RTOL:
                    failures.append(
                        f"{key}: {label} {x!r} vs {y!r} (rel {d:.3e})")
            if a[3] != b[3]:
                failures.append(f"{key}: converged_level {a[3]} vs {b[3]}")

    return n_checked, failures, worst, unsettled


def report(n_checked, failures, worst, unsettled):
    print(f"compared {n_checked} values across {len(worst)} quantity classes")
    for label, (d, key) in worst.items():
        print(f"  worst {label:<3} rel diff: {d:.3e}"
              + (f"  at {key}" if key else ""))
    skipped = ", ".join(f"{k}={v}" for k, v in unsettled.items() if v)
    if skipped:
        print(f"  skipped as not converged with refinement depth: {skipped}")
    if failures:
        print(f"\nFAIL: {len(failures)} discrepancies")
        for line in failures[:20]:
            print("  " + line)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1
    print(f"\nPASS (c tolerance {C_RTOL:.0e})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", metavar="FILE")
    ap.add_argument("--compare", metavar="FILE")
    args = ap.parse_args()

    if args.dump:
        table = build_table()
        Path(args.dump).write_text(json.dumps(table))
        print(f"wrote {len(table)} cases to {args.dump}")
        return 0

    if args.compare:
        ref = json.loads(Path(args.compare).read_text())
        return report(*compare(ref, build_table()))

    table = build_table()
    print(f"computed {len(table)} cases; nothing to diff against.")
    print("Dump with the other build first: DISPER26_SO_DIR=<dir> "
          f"{Path(__file__).name} --dump ref.json, then --compare ref.json here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
