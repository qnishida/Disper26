#!/usr/bin/env python3
"""Physical mode-identity diagnostics used by Plan L."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))

from disper26_inv.long_period_taylor_mode import (  # noqa: E402
    ModeShapeProfile,
    compute_character_assignment,
    compute_kernel_overlap,
    compute_mode_character_diagnostics,
    compute_mode_energy_metrics,
    compute_mode_shape_overlap,
)
from disper26_inv.long_period_taylor_validation import (  # noqa: E402
    classify_mode_identity,
)


def profiles():
    depth = np.linspace(0.0, 120.0, 241)
    density = 2.7 + 0.006 * depth
    shallow = np.exp(-0.5 * ((depth - 15.0) / 10.0) ** 2)
    deep = np.exp(-0.5 * ((depth - 70.0) / 16.0) ** 2)
    return (
        ModeShapeProfile(depth, shallow, density),
        ModeShapeProfile(depth, 1j * deep, density),
    )


def test_overlap_energy_and_assignment_are_phase_invariant():
    shallow, deep = profiles()
    phase_rotated = ModeShapeProfile(
        shallow.depth_km,
        np.exp(0.7j) * shallow.displacement,
        shallow.density,
    )
    assert compute_mode_shape_overlap(shallow, phase_rotated) > 1.0 - 1.0e-12
    assert compute_mode_shape_overlap(shallow, deep) < 0.1
    assert compute_kernel_overlap([1.0, 2.0], [-2.0, -4.0]) > 1.0 - 1.0e-12
    shallow_energy = compute_mode_energy_metrics(shallow, 35.0)
    deep_energy = compute_mode_energy_metrics(deep, 35.0)
    assert shallow_energy.crustal_fraction > 0.95
    assert deep_energy.crustal_fraction < 0.02
    assignment = compute_character_assignment([[0.01, 0.99], [0.98, 0.02]])
    np.testing.assert_array_equal(assignment, [1, 0])


def test_root_order_swap_is_rejected_then_recovers():
    shallow, deep = profiles()
    profile_array = np.empty((1, 2, 3), dtype=object)
    profile_array[0, 0] = [shallow, deep, deep]
    profile_array[0, 1] = [deep, shallow, shallow]
    phase = np.array([[[3.2, 3.25, 3.3], [4.2, 4.25, 4.3]]])
    group = np.array([[[3.0, 3.02, 3.04], [3.8, 3.82, 3.84]]])
    kernels = np.empty((1, 2, 3, 2))
    kernels[0, 0] = [[1.0, 0.01], [0.01, 1.0], [0.02, 1.0]]
    kernels[0, 1] = [[0.01, 1.0], [1.0, 0.01], [1.0, 0.02]]
    diagnostics = compute_mode_character_diagnostics(
        phase,
        group,
        profile_array,
        kernels,
        np.ones((1, 2, 3), dtype=bool),
        target_root_order=0,
        moho_depth_km=35.0,
    )
    assert not diagnostics.trace_agreement[0, 0]
    assert not diagnostics.trace_agreement[0, 1]
    assert diagnostics.trace_agreement[0, 2]
    assert diagnostics.eigenfunction_overlap[0, 0] < 0.1
    assert diagnostics.eigenfunction_overlap[0, 2] > 1.0 - 1.0e-12
    assert diagnostics.crustal_energy_fraction[0, 0] > 0.95
    assert diagnostics.crustal_energy_fraction[0, 2] < 0.02
    safe = classify_mode_identity(diagnostics)
    np.testing.assert_array_equal(safe, [[False, False, True]])


def test_missing_adjacent_root_is_local_not_global():
    shallow, deep = profiles()
    n_frequency = 5
    profile_array = np.empty((1, 3, n_frequency), dtype=object)
    profile_array[0, 0] = [shallow] * n_frequency
    profile_array[0, 1] = [deep, deep, None, deep, deep]
    profile_array[0, 2] = [None] * n_frequency
    phase = np.full((1, 3, n_frequency), np.nan)
    phase[0, 0] = np.linspace(3.2, 3.3, n_frequency)
    phase[0, 1] = np.asarray([4.2, 4.22, np.nan, 4.26, 4.28])
    group = np.full_like(phase, np.nan)
    group[0, 0] = np.linspace(3.0, 3.04, n_frequency)
    group[0, 1] = np.asarray([3.8, 3.81, np.nan, 3.83, 3.84])
    kernels = np.full((1, 3, n_frequency, 2), np.nan)
    kernels[0, 0] = np.tile([1.0, 0.01], (n_frequency, 1))
    kernels[0, 1, [0, 1, 3, 4]] = [0.01, 1.0]
    success = np.zeros((1, 3, n_frequency), dtype=bool)
    success[0, 0] = True
    success[0, 1, [0, 1, 3, 4]] = True
    diagnostics = compute_mode_character_diagnostics(
        phase,
        group,
        profile_array,
        kernels,
        success,
        target_root_order=0,
        moho_depth_km=35.0,
    )
    np.testing.assert_array_equal(
        diagnostics.trace_agreement[0],
        [True, False, False, False, True],
    )
    assert np.all(diagnostics.exact_success)


def main():
    test_overlap_energy_and_assignment_are_phase_invariant()
    test_root_order_swap_is_rejected_then_recovers()
    test_missing_adjacent_root_is_local_not_global()
    print("long-period Taylor mode-identity checks passed")


if __name__ == "__main__":
    main()
