"""Surface-wave dispersion with owned models and reusable mode solutions.

Use km, s, g/cm³, and GPa. The fundamental mode is ``mode=0``. Select a
GRID interpolation explicitly; computing derivatives never changes the model.
The low-level, backwards-compatible functions remain in ``disper26py``.
"""
from __future__ import annotations

from typing import Optional, Sequence
import warnings

import numpy as np
from disper26py import _api

Material = _api.Material
Geometry = _api.Geometry
Options = _api.Options
DispersionContext = _api.DispersionContext
create_dispersion_context = _api.create_dispersion_context
Model = _api.Model
ModeSolution = _api.ModeSolution
DispersionCurve = _api.DispersionCurve
Parameter = _api.Parameter
DerivativeResult = _api.DerivativeResult
Diagnostics = _api.Diagnostics
Eigenfunctions = _api.Eigenfunctions
EnergyIntegrals = _api.EnergyIntegrals
FrequencyCurvature = _api.FrequencyCurvature
GravityCorrection = _api.GravityCorrection
Capability = _api.Capability
Error = _api.Error

create_isotropic_parameters = _api.create_isotropic_parameters
create_fluid_parameters = _api.create_fluid_parameters
create_velocity_vti_parameters = _api.create_velocity_vti_parameters


class ConvergenceWarning(RuntimeWarning):
    """A requested quantity was returned with a convergence warning."""


def _checked(result):
    diagnostic = result.diagnostics
    if diagnostic.status == "warning":
        warnings.warn("; ".join(diagnostic.messages) or "Convergence was not reached",
                      ConvergenceWarning, stacklevel=3)
    return result


def _options(options: Optional[Options]) -> Options:
    if options is not None:
        return options
    return Options()


def create_model(depths: Sequence[float], *, kind: str,
                 vs: Optional[Sequence[float]] = None, vp: Optional[Sequence[float]] = None,
                 rho: Optional[Sequence[float]] = None,
                 vph: Optional[Sequence[float]] = None, vpv: Optional[Sequence[float]] = None,
                 vsh: Optional[Sequence[float]] = None, vsv: Optional[Sequence[float]] = None,
                 eta: Optional[Sequence[float]] = None,
                 A: Optional[Sequence[float]] = None, C: Optional[Sequence[float]] = None,
                 F: Optional[Sequence[float]] = None, L: Optional[Sequence[float]] = None,
                 N: Optional[Sequence[float]] = None, interpolation: Optional[str] = None,
                 geometry: Optional[Geometry] = None, name: str = "") -> Model:
    """Copy depth/material arrays into a Pancake or GRID model.

    Supply exactly one complete set: vs/vp/rho, vph/vpv/vsh/vsv/eta/rho,
    or A/C/F/L/N/rho. Every array is 1D and matches depths in length.
    Pancake depths are strictly increasing layer tops; the last point starts
    the halfspace. GRID duplicate depths specify shallow/deep interface sides.
    GRID requires interpolation='stiffness' or 'velocity'; Pancake forbids it.
    """
    fields = {}
    for key, value in (("vs", vs), ("vp", vp), ("rho", rho), ("vph", vph),
                       ("vpv", vpv), ("vsh", vsh), ("vsv", vsv), ("eta", eta),
                       ("A", A), ("C", C), ("F", F), ("L", L), ("N", N)):
        if value is not None:
            array = np.asarray(value, dtype=float)
            if array.ndim != 1:
                raise ValueError(f"{key} must be a 1D array")
            fields[key] = array
    depths = np.asarray(depths, dtype=float)
    if depths.ndim != 1:
        raise ValueError("depths must be a 1D array")
    return _api.create_model(depths, kind, fields, interpolation, geometry or Geometry(), name)


def create_pancake_model(layers: Sequence[Material], thicknesses: Sequence[float],
                         halfspace: Material, *, geometry: Optional[Geometry] = None,
                         name: str = "") -> Model:
    """Copy finite-layer materials/thicknesses and the explicit halfspace.

    All thicknesses are in km. Changing input lists afterwards does not alter
    the model. Geometry defaults to a flat model.
    """
    return _api.create_pancake_model(layers, thicknesses, halfspace,
                                     geometry or Geometry(), name)


def create_grid_model(depths: Sequence[float], materials: Optional[Sequence[Material]] = None,
                      *, interpolation: str, vs: Optional[Sequence[float]] = None,
                      vp: Optional[Sequence[float]] = None,
                      rho: Optional[Sequence[float]] = None,
                      geometry: Optional[Geometry] = None, name: str = "") -> Model:
    """Copy a GRID model with explicit ``stiffness`` or ``velocity`` interpolation.

    Supply materials, or the three isotropic arrays vs, vp, rho. Duplicate
    depths describe the shallow/deep sides of a discontinuity. The last node
    supplies the halfspace. Input velocities are km/s and density is g/cm³.
    """
    has_isotropic = any(value is not None for value in (vs, vp, rho))
    if materials is not None and has_isotropic:
        raise ValueError("Supply materials or vs/vp/rho, not both")
    if materials is None:
        if not all(value is not None for value in (vs, vp, rho)):
            raise ValueError("Supply materials or all three arrays vs, vp, rho")
        arrays = [np.asarray(value, dtype=float) for value in (vs, vp, rho)]
        if any(a.ndim != 1 or len(a) != len(depths) for a in arrays):
            raise ValueError("vs, vp, and rho must be 1D arrays matching depths")
        materials = [create_isotropic_parameters(s, p, d) for s, p, d in zip(*arrays)]
    return _api.create_grid_model(depths, materials, interpolation, geometry or Geometry(), name)


def compute_dispersion(model: Model, *, period: float, wave: str = "Rayleigh",
                        mode: int = 0, outputs: Sequence[str] = ("phase_velocity",),
                        options: Optional[Options] = None,
                        context: Optional[DispersionContext] = None) -> ModeSolution:
    """Solve one mode at a period in seconds; optional values are otherwise None.

    A context reuses scratch capacity; it does not reuse model values or roots.
    Select phase_velocity, group_velocity and/or hv_ratio. All computation is
    explicit; accessing a result property performs no further calculation.
    Use Options(strict=True) to turn convergence warnings into errors.
    Unchecked continuation uses c_guess to select a nearby root without mode
    counts. Its mode is a label; call compute_mode_count to inspect the root.
    """
    if context is None:
        return _checked(_api.compute_dispersion(model, period, wave, mode, outputs, _options(options)))
    return _checked(_api.compute_dispersion(model, period, wave, mode, outputs, _options(options), context))


def compute_dispersion_curve(model: Model, *, periods: Sequence[float], wave: str = "Rayleigh",
                              mode: int = 0, outputs: Sequence[str] = ("phase_velocity",),
                              options: Optional[Options] = None,
                              context: Optional[DispersionContext] = None) -> DispersionCurve:
    """Return solutions in input order, including repeated periods.

    A shared context serializes calls. Use one context per worker for parallelism.
    Mode means phase-velocity order, not eigenfunction tracking across periods.
    With method="continuation", verify_mode=False (or None) performs no mode
    counts, including initial selection and recovery. Supply c_guess near the
    desired root; mode is an unverified label. Use compute_mode_count explicitly
    at selected solutions and GRID levels. Method="continuation_recover" counts
    during initialization at each level and recovery after failed local tracking;
    intermediate roots remain unverified. verify_mode=True verifies every root
    in either method. Auto uses one startup group-velocity predictor on pancake,
    or at GRID levels 0 and 1; history disables this extra computation.
    """
    if context is None:
        result = _api.compute_dispersion_curve(model, periods, wave, mode, outputs, _options(options))
    else:
        result = _api.compute_dispersion_curve(model, periods, wave, mode, outputs, _options(options), context)
    for solution in result.solutions:
        _checked(solution)
    return result


def compute_mode_count(solution: ModeSolution, *, level: int = -1) -> int:
    """Count once at a saved root; -1 selects the last generated GRID level.

    Uses the unextrapolated root and its matching mesh. Does not rerun root
    finding, verify other levels/periods, or change solution diagnostics.
    """
    return _api.compute_mode_count(solution, level)


def compute_kernels(solution: ModeSolution, *, basis: str,
                    outputs: Sequence[str] = ("phase_velocity",),
                    options: Optional[Options] = None) -> DerivativeResult:
    """Compute gradients on the solution model; parameters describe every axis.

    Basis is stiffness, isotropic, velocity_vti or log_velocity_vti, where
    supported. Options use per-call defaults; root settings remain attached to
    the solution. Check get_capabilities before selecting a specialized method.
    """
    return _checked(_api.compute_kernels(solution, basis, outputs, _options(options)))


def compute_hessian(solution: ModeSolution, *, basis: str,
                    options: Optional[Options] = None) -> DerivativeResult:
    """Compute the phase-velocity Hessian in the explicit parameter basis."""
    return _checked(_api.compute_hessian(solution, basis, _options(options)))


def compute_hessian_reduced(solution: ModeSolution, projection: np.ndarray, *,
                            options: Optional[Options] = None) -> DerivativeResult:
    """Use the reduced algorithm for a (stiffness-axis, reduced-axis) projection.

    The projection can be C-contiguous, Fortran-contiguous, or a strided view.
    Rows use the stiffness parameter order reported by compute_kernels.
    """
    return _checked(_api.compute_hessian_reduced(solution, projection, _options(options)))


def compute_interface_kernels(solution: ModeSolution, *,
                              options: Optional[Options] = None) -> DerivativeResult:
    """Compute interface-depth sensitivity, distinct from layer thickness."""
    return _checked(_api.compute_interface_kernels(solution, _options(options)))


def compute_interface_hessian_row(solution: ModeSolution, *, interface_index: int,
                                  basis: str, options: Optional[Options] = None) -> DerivativeResult:
    """Compute one interface row; interface_index is zero-based, not a node index."""
    return _checked(_api.compute_interface_hessian_row(
        solution, interface_index, basis, _options(options)))


def compute_eigenfunctions(solution: ModeSolution, *, num_sub_layers: int = 10) -> Eigenfunctions:
    """Return depth, complex component arrays, fluid flags and diagnostics."""
    return _checked(_api.compute_eigenfunctions(solution, num_sub_layers))


def compute_energy_integrals(solution: ModeSolution, *,
                             options: Optional[Options] = None) -> EnergyIntegrals:
    """Return named I1, I2, I3, phase_velocity and group_velocity fields."""
    return _checked(_api.compute_energy_integrals(solution, _options(options)))


def compute_frequency_curvature(solution: ModeSolution, *,
                                options: Optional[Options] = None) -> FrequencyCurvature:
    """Compute frequency derivatives, whose field names specify angular frequency."""
    return _checked(_api.compute_frequency_curvature(solution, _options(options)))


def compute_gravity_correction(solution: ModeSolution, *, gravity: float = 9.80665e-3,
                               region: str = "solid", num_sub_layers: int = 40,
                               domega_relative: float = 5e-3) -> GravityCorrection:
    """Evaluate the supported perturbative gravity correction; gravity is km/s²."""
    return _checked(_api.compute_gravity_correction(
        solution, gravity, region, num_sub_layers, domega_relative))


def earth_flattening_transformation(model: Model) -> Model:
    """Return a new flat model, preserving transformation metadata."""
    return _api.earth_flattening_transformation(model)


def apply_mineos_physical_dispersion(reference_model: Model, qkappa: Sequence[float],
                                    qmu: Sequence[float], *, reference_period: float,
                                    period: float) -> Model:
    """Return a new period-corrected model, including the halfspace."""
    return _api.apply_mineos_physical_dispersion(reference_model, qkappa, qmu,
                                               reference_period, period)


def get_capabilities(model: Model, *, wave: str = "Rayleigh") -> Sequence[Capability]:
    """Inspect supported operations/bases/methods and reasons for restrictions."""
    return tuple(_api.get_capabilities(model, wave))


from ._prem import load_prem_flat, load_prem_spherical

__all__ = [
    "Material", "Geometry", "Options", "Model", "ModeSolution", "DispersionCurve",
    "Parameter", "DerivativeResult", "Diagnostics", "Eigenfunctions", "EnergyIntegrals",
    "FrequencyCurvature", "GravityCorrection", "Capability", "Error", "ConvergenceWarning",
    "create_isotropic_parameters", "create_fluid_parameters", "create_velocity_vti_parameters",
    "DispersionContext", "create_dispersion_context",
    "create_model", "create_pancake_model", "create_grid_model", "compute_dispersion", "compute_dispersion_curve", "compute_mode_count",
    "compute_kernels", "compute_hessian", "compute_hessian_reduced", "compute_interface_kernels",
    "compute_interface_hessian_row", "compute_eigenfunctions", "compute_energy_integrals",
    "compute_frequency_curvature", "compute_gravity_correction", "earth_flattening_transformation",
    "apply_mineos_physical_dispersion", "get_capabilities", "load_prem_flat", "load_prem_spherical",
]
