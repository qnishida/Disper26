#!/usr/bin/env python3
"""Cross-build regression for full-AD kernels at velocities near located roots."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

if os.environ.get("DISPER26_SO_DIR"):
    sys.path.insert(0, os.environ["DISPER26_SO_DIR"])
import disper26py as dp


def table(reference=None):
    layers = [
        dp.create_isotropic_parameters(3.0 + 0.3 * i, 5.2 + 0.5 * i,
                                       2.5 + 0.1 * i)
        for i in range(4)
    ]
    thicknesses = [5.0] * 4
    halfspace = dp.create_isotropic_parameters(4.6, 8.0, 3.3)
    omega = 2.0 * math.pi / 20.0
    result = {"module": dp.__file__, "cases": {}}
    for wave in ("Love", "Rayleigh"):
        roots = dp.search_roots(layers, thicknesses, halfspace, omega,
                                2.5, 4.5, 400, wave)
        if not roots:
            raise RuntimeError(f"no {wave} root")
        root = roots[0]
        for sign in (-1.0, 1.0):
            key = f"{wave}_{sign:+g}"
            velocity = (reference["cases"][key]["phase_velocity"]
                        if reference else root * (1.0 + sign * 1e-6))
            kernels = dp.compute_kernels_ad_pancake(
                layers, thicknesses, halfspace, omega, velocity, wave,
                "Isotropic", True)
            result["cases"][key] = {
                "located_root": root, "phase_velocity": velocity,
                "kernels": [[k.layer_index, k.parameter_name,
                             k.d_c, k.d_U, k.d_HV] for k in kernels]}
    return result


def compare(reference, current):
    failures = []
    worst = 0.0
    for key, expected in reference["cases"].items():
        got = current["cases"].get(key)
        if got is None or len(got["kernels"]) != len(expected["kernels"]):
            failures.append(f"{key}: missing case or kernel count differs")
            continue
        root_difference = abs(got["located_root"] - expected["located_root"]) / max(
            abs(expected["located_root"]), 1e-30)
        if root_difference > 1e-12:
            failures.append(f"{key}: located-root relative difference {root_difference:.3e}")
        for a, b in zip(expected["kernels"], got["kernels"]):
            if a[:2] != b[:2]:
                failures.append(f"{key}: kernel identity differs")
                continue
            for x, y in zip(a[2:], b[2:]):
                if math.isnan(x) or math.isnan(y):
                    if not (math.isnan(x) and math.isnan(y)):
                        failures.append(f"{key}: NaN classification differs")
                    continue
                difference = abs(y - x) / max(abs(x), 1e-30)
                worst = max(worst, difference)
                if difference > 1e-12:
                    failures.append(f"{key}: derivative relative difference {difference:.3e}")
    print(f"worst derivative relative difference: {worst:.3e}")
    if failures:
        print("FAIL")
        print("\n".join(failures[:20]))
        return 1
    print("PASS (derivative tolerance 1e-12)")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump")
    parser.add_argument("--compare")
    args = parser.parse_args()
    if args.dump:
        current = table()
        Path(args.dump).write_text(json.dumps(current, indent=2) + "\n")
        print(f"wrote {len(current['cases'])} perturbed-root cases")
        return 0
    if args.compare:
        reference = json.loads(Path(args.compare).read_text())
        return compare(reference, table(reference))
    parser.error("one of --dump or --compare is required")


if __name__ == "__main__":
    raise SystemExit(main())
