"""disper26_inv: surrogate-driven multistart GMM inversion on top of disper26py.

Public entry-point overview: examples/README.md#advanced.
Derivative implementation map: docs/sensitivity_implementation_map.md.
All public APIs take periods in seconds and km / km/s / g/cm^3 model units;
the omega/Hz inconsistencies of the raw bindings are hidden inside engine.py.
"""

__version__ = "0.1.104"

try:
    import scipy as _scipy
except ModuleNotFoundError as exc:
    if exc.name != "scipy":
        raise
    raise ImportError(
        "disper26_inv requires SciPy. Install the inversion extra from the "
        "source root with: python -m pip install '.[inversion]'"
    ) from exc
del _scipy

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
from .long_period_taylor import (  # noqa: F401
    EXACT_REQUIRED,
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND,
    PHASE_VELOCITY_RESPONSE_KIND,
    PRIMARY_FREQUENCY_HZ,
    ExactRequiredError,
    LongPeriodTaylorArtifact,
    LongPeriodTaylorBatchPrediction,
    LongPeriodTaylorCurve,
    LongPeriodTaylorPrediction,
    LongPeriodTaylorProvenance,
    build_long_period_taylor_curve,
)
from .long_period_taylor_io import (  # noqa: F401
    compute_long_period_mixed_response_artifact_hash,
    compute_long_period_taylor_artifact_hash,
    load_long_period_mixed_response_artifact,
    load_long_period_taylor_artifact,
    save_long_period_mixed_response_artifact,
    save_long_period_taylor_artifact,
)
