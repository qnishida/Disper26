"""disper26_inv: surrogate-driven multistart GMM inversion on top of disper26py.

Method reference: docs/vti_inversion_method_and_validation_plan.md.
All public APIs take periods in seconds and km / km/s / g/cm^3 model units;
the omega/Hz inconsistencies of the raw bindings are hidden inside engine.py.
"""

__version__ = "0.1.0"

from .engine import BranchError, Datum, ForwardEngine, ModeSolution  # noqa: F401
from .depth_hessian import (  # noqa: F401
    AdDepthHessianBackend,
    DepthHessianBackend,
    DepthHessianColumn,
    DepthHessianError,
    FdDepthHessianBackend,
)
from .model import (  # noqa: F401
    ActiveSubspaceBasis,
    Brocher2005CrustLaw,
    ForwardGridModel,
    GridRefinementPolicy,
    InterfaceShiftBasis,
    InversionBasis,
    Masters2000MantleLaw,
    MohoRegionMap,
    RegionalScaledShearBasis,
    RegionalScalingProfile,
    ScaledShearBasis,
    shift_discontinuity,
)
from .priors import ConditionalPriorCovariance, PriorCovariance  # noqa: F401
from .physics import PosdefReport, check_stiffness_posdef  # noqa: F401
from .prem import load_prem_flat, load_prem_spherical, refine_forward_grid  # noqa: F401
from .data import CorrelatedNoiseModel, NoiseModel, ObservationCovariance  # noqa: F401
from .shallow import (  # noqa: F401
    SHALLOW_INV_DEPTHS_KM,
    shallow_basis,
    shallow_dataset,
    shallow_independent_basis,
    shallow_reference,
)
