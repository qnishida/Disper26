"""Plan-L offline eigenfunction diagnostics separating root order from character."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ModeCharacterDiagnostics:
    """Exact trace and mode-shape diagnostics with shape (model, frequency)."""

    adjacent_relative_gap: np.ndarray
    eigenfunction_overlap: np.ndarray
    kernel_overlap: np.ndarray
    crustal_energy_fraction: np.ndarray
    energy_centroid_jump_km: np.ndarray
    group_velocity_relative_jump: np.ndarray
    trace_agreement: np.ndarray
    exact_success: np.ndarray

    def __post_init__(self):
        arrays = {
            "adjacent_relative_gap": np.asarray(
                self.adjacent_relative_gap, dtype=float
            ),
            "eigenfunction_overlap": np.asarray(
                self.eigenfunction_overlap, dtype=float
            ),
            "kernel_overlap": np.asarray(self.kernel_overlap, dtype=float),
            "crustal_energy_fraction": np.asarray(
                self.crustal_energy_fraction, dtype=float
            ),
            "energy_centroid_jump_km": np.asarray(
                self.energy_centroid_jump_km, dtype=float
            ),
            "group_velocity_relative_jump": np.asarray(
                self.group_velocity_relative_jump, dtype=float
            ),
            "trace_agreement": np.asarray(self.trace_agreement, dtype=bool),
            "exact_success": np.asarray(self.exact_success, dtype=bool),
        }
        shape = arrays["adjacent_relative_gap"].shape
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("mode diagnostics must be non-empty matrices")
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("all mode diagnostics must have equal shapes")
        for name, array in arrays.items():
            object.__setattr__(self, name, array.copy())

    @property
    def shape(self) -> tuple[int, int]:
        return self.adjacent_relative_gap.shape


@dataclass(frozen=True)
class ModeShapeProfile:
    """Complex displacement profile and density on one sampled depth grid."""

    depth_km: np.ndarray
    displacement: np.ndarray
    density: np.ndarray

    def __post_init__(self):
        depth = np.asarray(self.depth_km, dtype=float)
        displacement = np.asarray(self.displacement, dtype=complex)
        density = np.asarray(self.density, dtype=float)
        if (
            depth.ndim != 1
            or depth.size < 2
            or np.any(~np.isfinite(depth))
            or np.any(np.diff(depth) <= 0.0)
        ):
            raise ValueError("depth_km must be finite and strictly increasing")
        if displacement.ndim == 1:
            displacement = displacement[:, None]
        if displacement.ndim != 2 or displacement.shape[0] != depth.size:
            raise ValueError("displacement must have shape (depth, component)")
        if density.shape != depth.shape or np.any(~np.isfinite(density)) or np.any(density <= 0.0):
            raise ValueError("density must be finite, positive, and aligned with depth")
        if np.any(~np.isfinite(displacement.real)) or np.any(~np.isfinite(displacement.imag)):
            raise ValueError("displacement must be finite")
        if not np.any(np.abs(displacement) > 0.0):
            raise ValueError("displacement profile must be nonzero")
        object.__setattr__(self, "depth_km", depth.copy())
        object.__setattr__(self, "displacement", displacement.copy())
        object.__setattr__(self, "density", density.copy())


def mode_shape_from_solver_output(output, wave: str, density) -> ModeShapeProfile:
    """Convert one binding eigenfunction dictionary to a common profile type."""

    depth = np.asarray(output["depth"], dtype=float)
    density_values = np.asarray(density, dtype=float)
    wave_key = str(wave).lower()
    if wave_key == "love":
        displacement = np.asarray(output["uy"], dtype=complex)[:, None]
    elif wave_key == "rayleigh":
        displacement = np.column_stack(
            (
                np.asarray(output["ur"], dtype=complex),
                np.asarray(output["uz"], dtype=complex),
            )
        )
    else:
        raise ValueError("wave must be Love or Rayleigh")
    if density_values.shape != depth.shape:
        raise ValueError("density must be evaluated at eigenfunction sample depths")
    # Solver traces may repeat interface depths.  Zero-thickness pairs do not
    # affect integrals; average their displacement/density to obtain a strict
    # grid for interpolation and overlap calculations.
    unique, inverse = np.unique(depth, return_inverse=True)
    if unique.size != depth.size:
        accumulated = np.zeros((unique.size, displacement.shape[1]), dtype=complex)
        accumulated_density = np.zeros(unique.size)
        count = np.zeros(unique.size)
        for index, target in enumerate(inverse):
            accumulated[target] += displacement[index]
            accumulated_density[target] += density_values[index]
            count[target] += 1.0
        displacement = accumulated / count[:, None]
        density_values = accumulated_density / count
        depth = unique
    return ModeShapeProfile(depth, displacement, density_values)


def _interpolate_complex(depth, values, target):
    output = np.empty((target.size, values.shape[1]), dtype=complex)
    for component in range(values.shape[1]):
        output[:, component] = np.interp(target, depth, values[:, component].real)
        output[:, component] += 1j * np.interp(
            target, depth, values[:, component].imag
        )
    return output


def compute_mode_shape_overlap(
    left: ModeShapeProfile, right: ModeShapeProfile, *, sample_count: int = 513
) -> float:
    """Return a phase-invariant density-weighted displacement overlap."""

    if left.displacement.shape[1] != right.displacement.shape[1]:
        raise ValueError("mode profiles must have equal component counts")
    if int(sample_count) != sample_count or sample_count < 3:
        raise ValueError("sample_count must be an integer >= 3")
    lower = max(float(left.depth_km[0]), float(right.depth_km[0]))
    upper = min(float(left.depth_km[-1]), float(right.depth_km[-1]))
    if lower >= upper:
        raise ValueError("mode profiles have no common depth interval")
    depth = np.linspace(lower, upper, int(sample_count))
    left_values = _interpolate_complex(left.depth_km, left.displacement, depth)
    right_values = _interpolate_complex(right.depth_km, right.displacement, depth)
    density = 0.5 * (
        np.interp(depth, left.depth_km, left.density)
        + np.interp(depth, right.depth_km, right.density)
    )
    inner_density = density[:, None]
    inner = np.trapz(
        np.sum(np.conjugate(left_values) * right_values, axis=1) * density,
        depth,
    )
    left_norm = np.trapz(
        np.sum(np.abs(left_values) ** 2 * inner_density, axis=1), depth
    )
    right_norm = np.trapz(
        np.sum(np.abs(right_values) ** 2 * inner_density, axis=1), depth
    )
    denominator = float(np.sqrt(left_norm * right_norm))
    if not np.isfinite(denominator) or denominator <= 1.0e-300:
        raise ValueError("mode profile overlap is numerically singular")
    return float(np.clip(abs(inner) / denominator, 0.0, 1.0))


@dataclass(frozen=True)
class ModeEnergyMetrics:
    crustal_fraction: float
    centroid_depth_km: float


def compute_mode_energy_metrics(
    profile: ModeShapeProfile, moho_depth_km: float
) -> ModeEnergyMetrics:
    """Approximate kinetic-energy partition and depth centroid."""

    if not np.isfinite(moho_depth_km) or moho_depth_km <= profile.depth_km[0]:
        raise ValueError("moho_depth_km must lie below the first sample")
    amplitude = profile.density * np.sum(np.abs(profile.displacement) ** 2, axis=1)
    total = float(np.trapz(amplitude, profile.depth_km))
    if not np.isfinite(total) or total <= 1.0e-300:
        raise ValueError("mode energy is numerically singular")
    depth = profile.depth_km
    if moho_depth_km >= depth[-1]:
        crustal = total
    else:
        insert = np.searchsorted(depth, moho_depth_km)
        crustal_depth = np.concatenate((depth[:insert], [moho_depth_km]))
        crustal_amplitude = np.concatenate(
            (
                amplitude[:insert],
                [np.interp(moho_depth_km, depth, amplitude)],
            )
        )
        crustal = float(np.trapz(crustal_amplitude, crustal_depth))
    centroid = float(np.trapz(depth * amplitude, depth) / total)
    return ModeEnergyMetrics(
        crustal_fraction=float(np.clip(crustal / total, 0.0, 1.0)),
        centroid_depth_km=centroid,
    )


def compute_kernel_overlap(left, right) -> float:
    """Return the absolute cosine similarity of two real sensitivity rows."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("kernel vectors must have equal one-dimensional shapes")
    if np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
        raise ValueError("kernel vectors must be finite")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-300:
        raise ValueError("kernel overlap is numerically singular")
    return float(np.clip(abs(first @ second) / denominator, 0.0, 1.0))


def compute_character_assignment(overlap_matrix) -> np.ndarray:
    """Map each old root to the new root with maximum global overlap."""

    overlap = np.asarray(overlap_matrix, dtype=float)
    if overlap.ndim != 2 or min(overlap.shape) < 1 or np.any(~np.isfinite(overlap)):
        raise ValueError("overlap_matrix must be a finite non-empty matrix")
    rows, columns = linear_sum_assignment(-overlap)
    assignment = np.full(overlap.shape[0], -1, dtype=int)
    assignment[rows] = columns
    return assignment


def compute_adjacent_relative_gap(phase_velocity, target_root_order: int) -> np.ndarray:
    """Minimum relative phase-velocity gap to an adjacent root."""

    phase = np.asarray(phase_velocity, dtype=float)
    if phase.ndim != 3:
        raise ValueError("phase_velocity must have shape (model, root, frequency)")
    if not 0 <= int(target_root_order) < phase.shape[1]:
        raise ValueError("target_root_order is outside the supplied roots")
    target = phase[:, int(target_root_order), :]
    gaps = []
    for neighbor in (int(target_root_order) - 1, int(target_root_order) + 1):
        if 0 <= neighbor < phase.shape[1]:
            scale = np.maximum(
                np.maximum(np.abs(target), np.abs(phase[:, neighbor, :])),
                1.0e-300,
            )
            gaps.append(np.abs(target - phase[:, neighbor, :]) / scale)
    if not gaps:
        return np.full(target.shape, np.inf)
    return np.min(np.stack(gaps, axis=0), axis=0)


def _minimum_update(values: np.ndarray, index, candidate: float):
    if np.isfinite(candidate):
        current = values[index]
        values[index] = candidate if not np.isfinite(current) else min(current, candidate)


def _maximum_update(values: np.ndarray, index, candidate: float):
    if np.isfinite(candidate):
        current = values[index]
        values[index] = candidate if not np.isfinite(current) else max(current, candidate)


def _profile_assignment(profiles, left_index, right_index, root_indices=None):
    n_root = profiles.shape[1]
    if root_indices is None:
        roots = np.arange(n_root, dtype=int)
    else:
        roots = np.asarray(tuple(root_indices), dtype=int)
        if (
            roots.ndim != 1
            or roots.size < 1
            or np.any(roots < 0)
            or np.any(roots >= n_root)
            or np.unique(roots).size != roots.size
        ):
            raise ValueError("root_indices must identify distinct available roots")
    overlap = np.empty((roots.size, roots.size))
    for old_local, old_root in enumerate(roots):
        for new_local, new_root in enumerate(roots):
            left = profiles[left_index[0], old_root, left_index[1]]
            right = profiles[right_index[0], new_root, right_index[1]]
            if not isinstance(left, ModeShapeProfile) or not isinstance(
                right, ModeShapeProfile
            ):
                return None
            try:
                overlap[old_local, new_local] = compute_mode_shape_overlap(left, right)
            except ValueError:
                return None
    local_assignment = compute_character_assignment(overlap)
    assignment = np.full(n_root, -1, dtype=int)
    assignment[roots] = roots[local_assignment]
    return assignment


def compute_mode_character_diagnostics(
    phase_velocity,
    group_velocity,
    profiles,
    kernels,
    exact_success,
    *,
    target_root_order: int,
    moho_depth_km: float,
    model_neighbor_pairs=(),
) -> ModeCharacterDiagnostics:
    """Combine frequency/model continuation diagnostics for one root order.

    Input phase/group arrays have shape (model, root, frequency), profiles is
    an object array with the same shape, and kernels has an additional final
    parameter dimension.  Missing profiles/kernels remain NaN diagnostics and
    are therefore classified as transition_unsupported.
    """

    phase = np.asarray(phase_velocity, dtype=float)
    group = np.asarray(group_velocity, dtype=float)
    profile_values = np.asarray(profiles, dtype=object)
    success = np.asarray(exact_success, dtype=bool)
    if phase.ndim != 3 or group.shape != phase.shape:
        raise ValueError("phase/group velocity must have shape (model, root, frequency)")
    if profile_values.shape != phase.shape or success.shape != phase.shape:
        raise ValueError("profiles and exact_success must match phase_velocity")
    n_model, n_root, n_frequency = phase.shape
    target = int(target_root_order)
    if not 0 <= target < n_root:
        raise ValueError("target_root_order is outside the supplied root dimension")
    kernel_values = np.asarray(kernels, dtype=float)
    if kernel_values.ndim != 4 or kernel_values.shape[:3] != phase.shape:
        raise ValueError("kernels must have shape (model, root, frequency, parameter)")
    pairs = np.asarray(tuple(model_neighbor_pairs), dtype=int)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("model_neighbor_pairs must contain index pairs")
    if np.any(pairs < 0) or np.any(pairs >= n_model) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("model_neighbor_pairs contain invalid indices")

    shape = (n_model, n_frequency)
    adjacent_gap = compute_adjacent_relative_gap(phase, target)
    # A Sturm--Liouville root can exchange character only through an adjacent
    # ordered root.  Requiring every diagnostic root makes an unrelated root-2
    # failure invalidate Love fundamental at all frequencies.  Restrict the
    # assignment to the target and its immediate neighbors; a missing adjacent
    # root still fails closed at that local frequency.
    assignment_roots = tuple(
        root
        for root in (target - 1, target, target + 1)
        if 0 <= root < n_root
    )
    eigen_overlap = np.full(shape, np.nan)
    kernel_overlap = np.full(shape, np.nan)
    crustal_fraction = np.full(shape, np.nan)
    centroid = np.full(shape, np.nan)
    centroid_jump = np.full(shape, np.nan)
    group_jump = np.full(shape, np.nan)
    trace_agreement = success[:, target, :].copy()
    target_success = (
        success[:, target, :]
        & np.isfinite(phase[:, target, :])
        & (phase[:, target, :] > 0.0)
        & np.isfinite(group[:, target, :])
        & (group[:, target, :] > 0.0)
    )

    for model_index in range(n_model):
        for frequency_index in range(n_frequency):
            profile = profile_values[model_index, target, frequency_index]
            if not isinstance(profile, ModeShapeProfile):
                target_success[model_index, frequency_index] = False
                trace_agreement[model_index, frequency_index] = False
                continue
            try:
                metrics = compute_mode_energy_metrics(profile, moho_depth_km)
            except ValueError:
                target_success[model_index, frequency_index] = False
                trace_agreement[model_index, frequency_index] = False
                continue
            crustal_fraction[model_index, frequency_index] = metrics.crustal_fraction
            centroid[model_index, frequency_index] = metrics.centroid_depth_km

        for frequency_index in range(n_frequency - 1):
            left_index = (model_index, frequency_index)
            right_index = (model_index, frequency_index + 1)
            left_profile = profile_values[model_index, target, frequency_index]
            right_profile = profile_values[model_index, target, frequency_index + 1]
            if isinstance(left_profile, ModeShapeProfile) and isinstance(
                right_profile, ModeShapeProfile
            ):
                try:
                    overlap = compute_mode_shape_overlap(left_profile, right_profile)
                    _minimum_update(eigen_overlap, left_index, overlap)
                    _minimum_update(eigen_overlap, right_index, overlap)
                except ValueError:
                    pass
            left_kernel = kernel_values[model_index, target, frequency_index]
            right_kernel = kernel_values[model_index, target, frequency_index + 1]
            try:
                overlap = compute_kernel_overlap(left_kernel, right_kernel)
                _minimum_update(kernel_overlap, left_index, overlap)
                _minimum_update(kernel_overlap, right_index, overlap)
            except ValueError:
                pass
            if np.isfinite(centroid[left_index]) and np.isfinite(centroid[right_index]):
                jump = abs(centroid[right_index] - centroid[left_index])
                _maximum_update(centroid_jump, left_index, jump)
                _maximum_update(centroid_jump, right_index, jump)
            left_group = group[model_index, target, frequency_index]
            right_group = group[model_index, target, frequency_index + 1]
            if np.isfinite(left_group) and np.isfinite(right_group):
                jump = abs(right_group - left_group) / max(
                    abs(left_group), abs(right_group), 1.0e-300
                )
                _maximum_update(group_jump, left_index, jump)
                _maximum_update(group_jump, right_index, jump)
            assignment = _profile_assignment(
                profile_values,
                left_index,
                right_index,
                root_indices=assignment_roots,
            )
            agrees = assignment is not None and assignment[target] == target
            if not agrees:
                trace_agreement[left_index] = False
                trace_agreement[right_index] = False

    # Connected points in the radial ensemble can reveal character exchange in
    # model space even when every individual frequency trace looks smooth.
    for first_model, second_model in pairs:
        for frequency_index in range(n_frequency):
            first_index = (int(first_model), frequency_index)
            second_index = (int(second_model), frequency_index)
            first_profile = profile_values[first_model, target, frequency_index]
            second_profile = profile_values[second_model, target, frequency_index]
            if isinstance(first_profile, ModeShapeProfile) and isinstance(
                second_profile, ModeShapeProfile
            ):
                try:
                    overlap = compute_mode_shape_overlap(first_profile, second_profile)
                    _minimum_update(eigen_overlap, first_index, overlap)
                    _minimum_update(eigen_overlap, second_index, overlap)
                except ValueError:
                    pass
            try:
                overlap = compute_kernel_overlap(
                    kernel_values[first_model, target, frequency_index],
                    kernel_values[second_model, target, frequency_index],
                )
                _minimum_update(kernel_overlap, first_index, overlap)
                _minimum_update(kernel_overlap, second_index, overlap)
            except ValueError:
                pass
            assignment = _profile_assignment(
                profile_values,
                (int(first_model), frequency_index),
                (int(second_model), frequency_index),
                root_indices=assignment_roots,
            )
            agrees = assignment is not None and assignment[target] == target
            if not agrees:
                trace_agreement[first_index] = False
                trace_agreement[second_index] = False

    # A one-point curve has no continuation evidence.  For ordinary curves,
    # edge points inherit their only finite adjacent jump/overlap.
    target_success &= np.isfinite(eigen_overlap)
    target_success &= np.isfinite(kernel_overlap)
    target_success &= np.isfinite(crustal_fraction)
    target_success &= np.isfinite(centroid_jump)
    target_success &= np.isfinite(group_jump)
    trace_agreement &= target_success
    return ModeCharacterDiagnostics(
        adjacent_relative_gap=adjacent_gap,
        eigenfunction_overlap=eigen_overlap,
        kernel_overlap=kernel_overlap,
        crustal_energy_fraction=crustal_fraction,
        energy_centroid_jump_km=centroid_jump,
        group_velocity_relative_jump=group_jump,
        trace_agreement=trace_agreement,
        exact_success=target_success,
    )


__all__ = [
    "ModeCharacterDiagnostics",
    "ModeEnergyMetrics",
    "ModeShapeProfile",
    "compute_adjacent_relative_gap",
    "compute_character_assignment",
    "compute_kernel_overlap",
    "compute_mode_character_diagnostics",
    "compute_mode_energy_metrics",
    "compute_mode_shape_overlap",
    "mode_shape_from_solver_output",
]
