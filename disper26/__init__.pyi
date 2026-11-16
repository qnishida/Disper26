from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np

class Error(RuntimeError):
    code: str
class ConvergenceWarning(RuntimeWarning): ...

class Material:
    def __init__(self, rho: float, A: float, C: float, F: float, L: float, N: float,
                 gravity: float = ..., gravity_internal_scale: float = ...,
                 gravity_surface_scale: float = ...) -> None: ...
    @property
    def rho(self) -> float: ...
    @property
    def A(self) -> float: ...
    @property
    def C(self) -> float: ...
    @property
    def F(self) -> float: ...
    @property
    def L(self) -> float: ...
    @property
    def N(self) -> float: ...
    @property
    def gravity(self) -> float: ...
    @property
    def gravity_internal_scale(self) -> float: ...
    @property
    def gravity_surface_scale(self) -> float: ...

class Geometry:
    def __init__(self, kind: str = ..., planet_radius: float = ...,
                 flattening_exponent: float = ...) -> None: ...
    @property
    def kind(self) -> str: ...
    @property
    def planet_radius(self) -> float: ...
    @property
    def flattening_exponent(self) -> float: ...

class Options:
    max_steps: int
    max_levels: int
    num_sub_layers: int
    tolerance: float
    root_tolerance: float
    derivative_tolerance: float
    c_guess: Optional[float]
    method: str
    predictor: str
    verify_mode: Optional[bool]
    start: str
    c_range: Optional[Tuple[float, float]]
    verify: Optional[str]
    on_violation: Optional[str]
    backend: str
    integration: str
    strict: bool
    def __init__(self, *, max_steps: int = ..., max_levels: int = ..., num_sub_layers: int = ...,
                 tolerance: float = ..., root_tolerance: float = ..., derivative_tolerance: float = ...,
                 c_guess: Optional[float] = ..., method: str = ..., backend: str = ...,
                 integration: str = ..., strict: bool = ..., predictor: str = ...,
                 verify_mode: Optional[bool] = ..., start: str = ...,
                 c_range: Optional[Tuple[float, float]] = ..., verify: Optional[str] = ...,
                 on_violation: Optional[str] = ...) -> None: ...

class Model:
    @property
    def kind(self) -> str: ...
    @property
    def interpolation(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def materials(self) -> Tuple[Material, ...]: ...
    @property
    def depths(self) -> np.ndarray: ...
    @property
    def thicknesses(self) -> np.ndarray: ...
    @property
    def discontinuities(self) -> np.ndarray: ...
    @property
    def halfspace(self) -> Material: ...
    @property
    def geometry(self) -> Geometry: ...
    @property
    def flattened(self) -> bool: ...
    @property
    def flattening_policy(self) -> str: ...
    @property
    def q_period(self) -> Optional[float]: ...
    @property
    def q_reference_period(self) -> Optional[float]: ...

class Diagnostics:
    missing_reason: str
    status: str
    convergence: str
    method: str
    backend: str
    messages: Sequence[str]
    availability: Dict[str, str]
    convergence_by_quantity: Dict[str, str]
    accepted_level: int
    generated_level: int
    fallback_count: int
    relative_error: float
    mode_verified: bool
    level_mode_verified: Sequence[bool]
    predictors: Sequence[str]
    fallback_reasons: Sequence[str]
    mode_count_evaluations: int
    predictor_ad_evaluations: int

class ModeSolution:
    @property
    def phase_velocity(self) -> float: ...
    @property
    def group_velocity(self) -> Optional[float]: ...
    @property
    def hv_ratio(self) -> Optional[float]: ...
    @property
    def period(self) -> float: ...
    @property
    def wave(self) -> str: ...
    @property
    def mode(self) -> int: ...
    @property
    def model(self) -> Model: ...
    @property
    def options(self) -> Options: ...
    @property
    def diagnostics(self) -> Diagnostics: ...

class DispersionCurve:
    @property
    def periods(self) -> np.ndarray: ...
    @property
    def solutions(self) -> Tuple[ModeSolution, ...]: ...
    @property
    def phase_velocity(self) -> np.ndarray: ...
    @property
    def group_velocity(self) -> Optional[np.ndarray]: ...
    @property
    def hv_ratio(self) -> Optional[np.ndarray]: ...

class Parameter:
    name: str
    entity: str
    unit: str
    index: int
    active: bool

class DerivativeResult:
    basis: str
    parameters: Tuple[Parameter, ...]
    gradients: Dict[str, np.ndarray]
    gradient: Optional[np.ndarray]
    hessian: Optional[np.ndarray]
    diagnostics: Diagnostics

class Eigenfunctions:
    depth: np.ndarray
    components: Dict[str, np.ndarray]
    is_fluid: np.ndarray
    diagnostics: Diagnostics

class EnergyIntegrals:
    I1: float
    I2: float
    I3: float
    phase_velocity: float
    group_velocity: float
    diagnostics: Diagnostics

class FrequencyCurvature:
    phase_velocity: float
    group_velocity: float
    group_velocity_omega_derivative: float
    wavenumber_omega_second_derivative: float
    diagnostics: Diagnostics

class GravityCorrection:
    phase_velocity: float
    group_velocity: float
    delta_c: float
    delta_u: float
    omega_delta_c_omega_derivative: float
    region: str
    diagnostics: Diagnostics

class Capability:
    operation: str
    basis: str
    method: str
    reason: str
    supported: bool

def create_isotropic_parameters(vs: float, vp: float, rho: float) -> Material: ...
def create_fluid_parameters(vp: float, rho: float) -> Material: ...
def create_velocity_vti_parameters(rho: float, vph: float, vpv: float, vsh: float,
                                    vsv: float, eta: float) -> Material: ...
def create_model(depths: Sequence[float], *, kind: str,
                 vs: Optional[Sequence[float]] = None, vp: Optional[Sequence[float]] = None,
                 rho: Optional[Sequence[float]] = None,
                 vph: Optional[Sequence[float]] = None, vpv: Optional[Sequence[float]] = None,
                 vsh: Optional[Sequence[float]] = None, vsv: Optional[Sequence[float]] = None,
                 eta: Optional[Sequence[float]] = None,
                 A: Optional[Sequence[float]] = None, C: Optional[Sequence[float]] = None,
                 F: Optional[Sequence[float]] = None, L: Optional[Sequence[float]] = None,
                 N: Optional[Sequence[float]] = None, interpolation: Optional[str] = None,
                 geometry: Optional[Geometry] = None, name: str = "") -> Model: ...

def create_layer_cake_model(layers: Sequence[Material], thicknesses: Sequence[float],
                         halfspace: Material, *, geometry: Optional[Geometry] = ..., name: str = ...) -> Model: ...
# Legacy constructor preserves kind="pancake".
def create_pancake_model(layers: Sequence[Material], thicknesses: Sequence[float],
                         halfspace: Material, *, geometry: Optional[Geometry] = ..., name: str = ...) -> Model: ...
def create_grid_model(depths: Sequence[float], materials: Optional[Sequence[Material]] = ..., *,
                      interpolation: str, vs: Optional[Sequence[float]] = ..., vp: Optional[Sequence[float]] = ...,
                      rho: Optional[Sequence[float]] = ..., geometry: Optional[Geometry] = ..., name: str = ...) -> Model: ...
class DispersionContext:
    """Owned scratch capacity; shared calls are serialized."""
    ...
def create_dispersion_context() -> DispersionContext: ...

def compute_dispersion(model: Model, *, period: float, wave: str = ..., mode: int = ...,
                        outputs: Sequence[str] = ..., options: Optional[Options] = ..., context: Optional[DispersionContext] = ...) -> ModeSolution: ...
def compute_dispersion_curve(model: Model, *, periods: Sequence[float], wave: str = ..., mode: int = ...,
                              outputs: Sequence[str] = ..., options: Optional[Options] = ..., context: Optional[DispersionContext] = ...) -> DispersionCurve: ...
def compute_mode_count(solution: ModeSolution, *, level: int = ...) -> int: ...
def compute_kernels(solution: ModeSolution, *, basis: str, outputs: Sequence[str] = ...,
                    options: Optional[Options] = ...) -> DerivativeResult: ...
def compute_hessian(solution: ModeSolution, *, basis: str,
                    options: Optional[Options] = ...) -> DerivativeResult: ...
def compute_hessian_reduced(solution: ModeSolution, projection: np.ndarray, *,
                            options: Optional[Options] = ...) -> DerivativeResult: ...
def compute_interface_kernels(solution: ModeSolution, *, options: Optional[Options] = ...) -> DerivativeResult: ...
def compute_interface_hessian_row(solution: ModeSolution, *, interface_index: int, basis: str,
                                  options: Optional[Options] = ...) -> DerivativeResult: ...
def compute_eigenfunctions(solution: ModeSolution, *, num_sub_layers: int = ...) -> Eigenfunctions: ...
def compute_energy_integrals(solution: ModeSolution, *, options: Optional[Options] = ...) -> EnergyIntegrals: ...
def compute_frequency_curvature(solution: ModeSolution, *, options: Optional[Options] = ...) -> FrequencyCurvature: ...
def compute_gravity_correction(solution: ModeSolution, *, gravity: float = ..., region: str = ...,
                               num_sub_layers: int = ..., domega_relative: float = ...) -> GravityCorrection: ...
def earth_flattening_transformation(model: Model) -> Model: ...
def apply_mineos_physical_dispersion(reference_model: Model, qkappa: Sequence[float], qmu: Sequence[float], *,
                                    reference_period: float, period: float) -> Model: ...
def get_capabilities(model: Model, *, wave: str = ...) -> Sequence[Capability]: ...
def load_prem_flat(*, interpolation: str = ..., max_depth_km: float = ..., max_spacing_km: Optional[float] = ...,
                   density_exponent: float = ..., csv_path: Optional[Union[str, Path]] = ..., name: str = ...) -> Model: ...
def load_prem_spherical(*, interpolation: str = ..., max_depth_km: float = ..., max_spacing_km: Optional[float] = ...,
                        density_exponent: float = ..., csv_path: Optional[Union[str, Path]] = ..., name: str = ...) -> Model: ...
