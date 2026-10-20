"""Regression tests for the fast AD-z mode counter.

These pin the parts of the counting convention that the substep-endpoint reuse
and the analytic endpoint derivative could plausibly have moved: the integer
labels themselves, the per-crossing contribution sequence, the agreement
between the count-only and debug entry points, and the invariance of a label
under splitting a layer at a depth where a crossing sits.

The counter is verified against dispersion roots in test_bran_mode_count.py;
this file is about the label staying put when the propagation is restructured.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp


OMEGA = 2.0 * math.pi * 0.5


def solid_model():
    return (
        [
            dp.create_isotropic_parameters(1.0, 1.73, 2.0),
            dp.create_isotropic_parameters(1.5, 2.6, 2.2),
            dp.create_isotropic_parameters(2.2, 3.8, 2.5),
        ],
        [2.0, 5.0, 8.0],
        dp.create_isotropic_parameters(3.0, 5.2, 2.7),
    )


def ocean_model():
    return (
        [dp.create_fluid_parameters(1.5, 1.03), dp.create_isotropic_parameters(2.2, 3.8, 2.5)],
        [3.0, 12.0],
        dp.create_isotropic_parameters(3.0, 5.2, 2.7),
    )


def split_layers(layers, thicknesses, pieces):
    out_layers, out_thicknesses = [], []
    for layer, thickness in zip(layers, thicknesses):
        for _ in range(pieces):
            out_layers.append(layer)
            out_thicknesses.append(thickness / pieces)
    return out_layers, out_thicknesses


def split_at_depth(layers, thicknesses, depth_from_top):
    """Split whichever layer contains `depth_from_top` exactly at that depth."""
    out_layers, out_thicknesses = [], []
    remaining = depth_from_top
    for layer, thickness in zip(layers, thicknesses):
        if 1e-9 < remaining < thickness - 1e-9:
            out_layers.extend([layer, layer])
            out_thicknesses.extend([remaining, thickness - remaining])
        else:
            out_layers.append(layer)
            out_thicknesses.append(thickness)
        remaining -= thickness
    return out_layers, out_thicknesses


# Recorded before the endpoint reuse and analytic derivative changes, except
# for the supra-Rayleigh labels corrected by halfspace initialization. Those
# were one low away from roots; test_halfspace_mode_count checks the corrected
# cumulative counts against an independent secular scan.
SOLID_RAYLEIGH_LABELS = {
    0.8: 0, 1.2: 1, 1.7: 4, 2.5: 8, 2.95: 9,
}
SOLID_LOVE_LABELS = {
    0.8: 0, 1.2: 1, 1.7: 3, 2.5: 6, 2.95: 8,
}
# c = 2.2 is bit-identical to the crust Vs and is deliberately absent: see
# test_labels_are_stable_around_a_layer_velocity.
OCEAN_RAYLEIGH_LABELS = {
    0.9: 0, 1.4: 0, 2.9: 6,
}


def test_solid_labels_are_pinned():
    layers, thicknesses, halfspace = solid_model()
    for wave, expected in (("Rayleigh", SOLID_RAYLEIGH_LABELS), ("Love", SOLID_LOVE_LABELS)):
        for c, label in expected.items():
            got = dp.compute_mode_count(layers, thicknesses, halfspace, OMEGA, c, wave)
            assert got == label, f"{wave} c={c}: {got} != {label}"


def test_ocean_labels_are_pinned():
    layers, thicknesses, halfspace = ocean_model()
    for c, label in OCEAN_RAYLEIGH_LABELS.items():
        got = dp.compute_mode_count(layers, thicknesses, halfspace, OMEGA, c, "Rayleigh")
        assert got == label, f"ocean c={c}: {got} != {label}"


def test_count_only_and_debug_agree():
    """The count-only path and the diagnostic path must not drift apart."""
    for model in (solid_model(), ocean_model()):
        layers, thicknesses, halfspace = model
        waves = ("Rayleigh",) if any(dp_is_fluid(l) for l in layers) else ("Rayleigh", "Love")
        for wave in waves:
            for c in (0.9, 1.4, 2.0, 2.5, 2.9):
                count = dp.compute_mode_count(layers, thicknesses, halfspace, OMEGA, c, wave)
                debug = dp.debug_mode_count(layers, thicknesses, halfspace, OMEGA, c, wave)
                assert debug["status"] == "ok"
                assert count == debug["n_integer"]
                assert debug["crossing_count"] == len(debug["crossings"])
                assert all(x["contribution"] in (-1, 1) for x in debug["crossings"])


def dp_is_fluid(layer):
    return layer.c44 <= 0.0


def test_label_survives_uniform_layer_splitting():
    """Splitting a layer into equal pieces changes the substep layout but not
    the physics, so the label and the crossing contributions must not move."""
    layers, thicknesses, halfspace = solid_model()
    for wave in ("Rayleigh", "Love"):
        for c in (1.2, 2.5, 2.95):
            reference = dp.debug_mode_count(layers, thicknesses, halfspace, OMEGA, c, wave)
            for pieces in (2, 3, 5, 8):
                sl, st = split_layers(layers, thicknesses, pieces)
                got = dp.compute_mode_count(sl, st, halfspace, OMEGA, c, wave)
                assert got == reference["n_integer"], (
                    f"{wave} c={c} split x{pieces}: {got} != {reference['n_integer']}")
                debug = dp.debug_mode_count(sl, st, halfspace, OMEGA, c, wave)
                assert ([x["contribution"] for x in debug["crossings"]]
                        == [x["contribution"] for x in reference["crossings"]])


def test_label_survives_splitting_exactly_on_a_crossing():
    """A layer boundary placed exactly where a crossing sits exercises the
    dedup that keeps one crossing from being counted on both sides."""
    layers, thicknesses, halfspace = solid_model()
    total = sum(thicknesses)
    for wave in ("Rayleigh", "Love"):
        for c in (1.2, 2.5, 2.95):
            reference = dp.debug_mode_count(layers, thicknesses, halfspace, OMEGA, c, wave)
            for crossing in reference["crossings"]:
                # local_depth_km counts upward from the halfspace top; the
                # splitter works downward from the free surface.
                depth_from_top = total - crossing["local_depth_km"]
                if not (1e-6 < depth_from_top < total - 1e-6):
                    continue
                sl, st = split_at_depth(layers, thicknesses, depth_from_top)
                got = dp.compute_mode_count(sl, st, halfspace, OMEGA, c, wave)
                assert got == reference["n_integer"], (
                    f"{wave} c={c} split at {depth_from_top}: "
                    f"{got} != {reference['n_integer']}")


def test_labels_are_stable_around_a_layer_velocity():
    """Away from the bit-exact degeneracy, the label must not depend on which
    side of a layer velocity the phase velocity sits.

    At c bit-identical to a layer Vs the layer's vertical wavenumber vanishes
    and the counter is degenerate; docs/mode_counting.md excludes that band
    from counter verification, and this test stays outside it.
    """
    layers, thicknesses, halfspace = solid_model()
    for wave in ("Rayleigh", "Love"):
        for vs in (1.0, 1.5, 2.2):
            labels = {
                dp.compute_mode_count(layers, thicknesses, halfspace, OMEGA,
                                      vs * (1.0 + delta), wave)
                for delta in (-1e-6, -1e-9, -1e-12, 1e-12, 1e-9, 1e-6)
            }
            assert len(labels) == 1, f"{wave} vs={vs}: unstable labels {labels}"


def test_monotonic_labels_on_roots():
    """Labels along the root sequence stay 0, 1, 2, ... for both waves."""
    layers, thicknesses, halfspace = solid_model()
    for wave in ("Rayleigh", "Love"):
        roots = dp.search_roots(layers, thicknesses, halfspace, OMEGA, 0.5, 3.0, 40000, wave)
        for expected, c in enumerate(roots):
            got = dp.compute_mode_count(layers, thicknesses, halfspace, OMEGA, c, wave)
            assert got == expected, f"{wave} root {expected} at c={c}: label {got}"


if __name__ == "__main__":
    test_solid_labels_are_pinned()
    test_ocean_labels_are_pinned()
    test_count_only_and_debug_agree()
    test_label_survives_uniform_layer_splitting()
    test_label_survives_splitting_exactly_on_a_crossing()
    test_labels_are_stable_around_a_layer_velocity()
    test_monotonic_labels_on_roots()
    print("OK")
