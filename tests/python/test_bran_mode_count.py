"""Diagnostic AD-z bran-style mode-count checks."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp


def assert_aliases_agree(layers, thicknesses, halfspace, omega, c, wave, expected):
    """calculate_total_mode_count and calculate_bran_mode_count are not separate
    counters -- both reach calculateAdModeCount with the same arguments, and
    compute_mode_count (docs/naming_convention.md #4/#5/#9) is the canonical
    name for it; the calculate_* spellings are deprecated aliases. Checking
    them here keeps that true if anyone rewires one of the aliases."""
    for alias in ("compute_mode_count", "calculate_total_mode_count", "calculate_bran_mode_count"):
        got = getattr(dp, alias)(layers, thicknesses, halfspace, omega, c, wave)
        assert got == expected, f"{alias} gave {got}, calculate_mode_count gave {expected}"
    assert (dp.debug_bran_mode_count(layers, thicknesses, halfspace, omega, c, wave)["n_integer"]
            == dp.debug_mode_count_crossings(layers, thicknesses, halfspace, omega, c, wave)["n_integer"]
            == dp.debug_mode_count(layers, thicknesses, halfspace, omega, c, wave)["n_integer"])


def build_solid_model():
    l1 = dp.create_isotropic_parameters(1.0, 1.73, 2.0)
    l2 = dp.create_isotropic_parameters(1.5, 2.6, 2.2)
    l3 = dp.create_isotropic_parameters(2.2, 3.8, 2.5)
    halfspace = dp.create_isotropic_parameters(3.0, 5.2, 2.7)
    return [l1, l2, l3], [2.0, 5.0, 8.0], halfspace


def test_bran_mode_count_debug_runs_on_solid_rayleigh_roots():
    layers, thicknesses, halfspace = build_solid_model()
    omega = 2.0 * math.pi * 0.5
    roots = dp.search_roots(layers, thicknesses, halfspace, omega, 0.5, 3.0, 500, "Rayleigh")

    assert len(roots) == 10
    previous_bran = -1
    for expected_mode, c_phase in enumerate(roots):
        count = dp.calculate_mode_count(layers, thicknesses, halfspace, omega, c_phase, "Rayleigh")
        debug = dp.debug_mode_count_crossings(layers, thicknesses, halfspace, omega, c_phase, "Rayleigh")

        assert_aliases_agree(layers, thicknesses, halfspace, omega, c_phase, "Rayleigh", count)
        bran = count
        assert debug["status"] == "ok"
        assert bran == debug["n_integer"]
        assert bran == expected_mode
        assert debug["crossing_count"] == len(debug["crossings"])
        assert debug["n_integer"] == abs(debug["calibrated_signed_count"])
        assert bran >= previous_bran
        previous_bran = bran

        for crossing in debug["crossings"]:
            assert set(crossing) == {
                "layer_index",
                "local_depth_km",
                "contribution",
                "secular",
                "secular_derivative",
                "conjugate",
                "test_value",
            }
            assert crossing["contribution"] in (-1, 1)


def test_bran_mode_count_debug_runs_on_solid_love_roots():
    layers, thicknesses, halfspace = build_solid_model()
    omega = 2.0 * math.pi * 0.5
    roots = dp.search_roots(layers, thicknesses, halfspace, omega, 0.5, 3.0, 500, "Love")

    assert len(roots) > 0
    previous_bran = -1
    for expected_mode, c_phase in enumerate(roots):
        count = dp.calculate_mode_count(layers, thicknesses, halfspace, omega, c_phase, "Love")
        debug = dp.debug_mode_count_crossings(layers, thicknesses, halfspace, omega, c_phase, "Love")

        assert_aliases_agree(layers, thicknesses, halfspace, omega, c_phase, "Love", count)
        bran = count
        assert debug["status"] == "ok"
        assert bran == debug["n_integer"]
        assert bran == expected_mode
        assert debug["crossing_count"] == len(debug["crossings"])
        assert debug["n_integer"] == abs(debug["calibrated_signed_count"])
        assert bran >= previous_bran
        previous_bran = bran


def build_ocean_model():
    fluid = dp.create_fluid_parameters(1.5, 1.03)
    crust = dp.create_isotropic_parameters(2.2, 3.8, 2.5)
    halfspace = dp.create_isotropic_parameters(3.0, 5.2, 2.7)
    return [fluid, crust], [3.0, 12.0], halfspace


def test_bran_mode_count_debug_runs_on_ocean_rayleigh_roots():
    layers, thicknesses, halfspace = build_ocean_model()
    omega = 2.0 * math.pi * 0.5
    roots = dp.search_roots(layers, thicknesses, halfspace, omega, 0.5, 3.0, 500, "Rayleigh")

    assert len(roots) > 0
    previous_bran = -1
    for expected_mode, c_phase in enumerate(roots):
        count = dp.calculate_mode_count(layers, thicknesses, halfspace, omega, c_phase, "Rayleigh")
        debug = dp.debug_mode_count_crossings(layers, thicknesses, halfspace, omega, c_phase, "Rayleigh")

        assert_aliases_agree(layers, thicknesses, halfspace, omega, c_phase, "Rayleigh", count)
        bran = count
        assert debug["status"] == "ok"
        assert bran == debug["n_integer"]
        assert bran == expected_mode
        assert debug["crossing_count"] == len(debug["crossings"])
        assert debug["n_integer"] == abs(debug["calibrated_signed_count"])
        assert bran >= previous_bran
        previous_bran = bran


def test_bran_mode_count_reports_unsupported_fluid_love_cases():
    layers, thicknesses, halfspace = build_solid_model()
    omega = 2.0 * math.pi * 0.5
    c_phase = 1.5

    fluid = dp.create_fluid_parameters(1.5, 1.03)
    ocean_layers = [fluid, fluid, layers[0]]
    ocean_thicknesses = [2.0, 2.0, 5.0]
    ocean_debug = dp.debug_mode_count_crossings(ocean_layers, ocean_thicknesses, halfspace, omega, c_phase, "Love")
    assert ocean_debug["status"] == "unsupported_liquid"
    try:
        dp.calculate_mode_count(ocean_layers, ocean_thicknesses, halfspace, omega, c_phase, "Love")
        raise AssertionError("Liquid Love bran count should be unsupported")
    except RuntimeError as exc:
        assert "liquid" in str(exc)


if __name__ == "__main__":
    test_bran_mode_count_debug_runs_on_solid_rayleigh_roots()
    test_bran_mode_count_debug_runs_on_solid_love_roots()
    test_bran_mode_count_debug_runs_on_ocean_rayleigh_roots()
    test_bran_mode_count_reports_unsupported_fluid_love_cases()
    print("OK")
