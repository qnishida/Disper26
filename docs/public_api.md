# Shared public API

[Japanese edition](../docs_jp/public_api.md)

Python (`disper26`), C++ (`disper26::api`), C (`api_c.h`) and Fortran (`disper26_api`) use
the same C++ implementation for forward calculations. The legacy `disper26py`
and Fortran `disper26` interfaces remain available. For runnable Python
examples, see the [practice guide](../examples/README.md). For method-specific
checks and diagnostics, see the [low-level API](lowlevel_api.md).

## Migration from the seabed gravity option

The solver enforces continuity of normal displacement and Lagrangian normal
traction at the seabed. The former `gravity_seabed_scale` option introduced
an unverified additional traction jump. Remove the `seabed_scale` keyword
or sixth positional argument from low-level Python `create_fluid_parameters`
calls. Remove the `gravity_seabed_scale` keyword or tenth positional argument from
`Material` and any assignment to the removed field. Calculations that used
a zero scale retain their previous numerical behavior.

Recompile C++, C and Fortran clients for the nine-component material type.
In the C API, `model_info.materials` has nine rows in column-major order
and `halfspace` has nine elements. Both use the order `rho, A, C, F, L, N,
gravity, gravity_internal_scale, gravity_surface_scale`. The three remaining
gravity fields retain their meanings.

## From a model to a solution and its derivatives

```python
import disper26 as d26

model = d26.create_model(
    [0.0, 10.0], kind="layer_cake",
    vs=[3.0, 4.5], vp=[5.2, 8.0], rho=[2.5, 3.3],
)
solution = d26.compute_dispersion(
    model, period=5.0, wave="Rayleigh", mode=0,
    outputs=("phase_velocity", "group_velocity", "hv_ratio"),
)
kernels = d26.compute_kernels(solution, basis="velocity_iso")
hessian = d26.compute_hessian(solution, basis="velocity_iso")
print(solution.phase_velocity, solution.group_velocity)
print([(p.entity, p.index, p.name, p.active) for p in kernels.parameters])
print(kernels.gradient, hessian.hessian.shape)
```

Use km, s, g/cm³ and GPa. Specify frequency by the period in seconds.
Frequency-curvature derivatives use angular frequency ω as indicated by their
field names. Gravitational acceleration is in km/s². All languages use
`mode=0` for the fundamental mode.

The model copies its inputs into an immutable snapshot. The solution holds
the model, wave type, period, mode, options and accepted roots for subsequent
derivatives. Accessing a property reads the stored result.

```python
grid = d26.create_model(
    [0.0, 10.0, 30.0], kind="grid", interpolation="stiffness",
    vs=[3.0, 3.4, 4.5], vp=[5.2, 6.0, 8.0], rho=[2.5, 2.8, 3.3],
)
curve = d26.compute_dispersion_curve(grid, periods=[8.0, 5.0, 8.0])
assert curve.phase_velocity[0] == curve.phase_velocity[2]
print(curve.periods)  # Preserves the input order and repeated periods.
```

GRID requires `interpolation="stiffness"` or `"velocity"`. These choices
define different models even for identical node values. A pair of nodes at
the same depth represents the two sides of a discontinuity. The last node
defines the half-space. The curve solver sorts and deduplicates periods
internally and restores input order. With this GRID and the default refinement
limit, the examples can emit `ConvergenceWarning`. Inspect the solution
`diagnostics` for the accepted level and estimated error. At each period,
`mode` indexes roots
in phase-velocity order. This ordering does not track eigenfunction identity.

### Numerical options

| Option | Default | Use |
|---|---|---|
| `max_steps` / `tolerance` | 5 / 1e-10 | GRID phase and curve refinement, and frequency curvature |
| `max_levels` / `derivative_tolerance` | 5 / 1e-6 | GRID derivative refinement and extrapolation where supported |
| `root_tolerance` | 1e-12 | Root searches on individual meshes |
| `num_sub_layers` | 10 | Within-layer sampling for operations taking this option |
| `backend` | `"dual"` | Variational AD Hessians also support `"vector"` and `"vector_dyn"` |
| `integration` | `"auto"` | Automatic, analytical or trapezoidal quadrature |

For the layer-cake Love Hessian, `integration="trapezoidal"` uses
`num_sub_layers`. Earlier builds used 200 regardless of the option.
Use 200 explicitly to reproduce those trapezoidal calculations.

### Mode selection and tracking

| `Options.method` | When the solver checks the mode index |
|---|---|
| `standard` (default) or `mode_order` | At every period and every level used |
| `continuation_recover` | At each level's initial search and during recovery from local tracking failure |
| `continuation` | Follows the nearby root with zero internal counts, including initialization and recovery, under the default tracking options |

For `continuation`, supply `c_guess` near the desired root. The mode label is
unverified. By default, `continuation_recover` checks the requested mode
during initialization and recovery, which can require multiple counts. It
skips routine checks at intermediate points and the endpoint. For either
tracking method, `verify_mode=True` checks every point. `None` or `False`
selects the behavior in the table. Both `standard` and `mode_order` reject `verify_mode=False`.
Choose an explicit policy such as `verify="every"` in place of `verify_mode`.
Only roots actually checked have `level_mode_verified=True`.

`predictor="history"` uses up to three recent points at the same level and
information across levels. `"auto"` reuses group velocity U at the same level.
Even when U is omitted from the output, auto can compute it at startup to
predict the next period. This adds one AD evaluation for a layer-cake model.
For GRID with checks at every period, it adds one at level 0. Other GRID
tracking methods add at most two, at levels 0 and 1. No extra evaluation is
needed for a single period, with no next distinct period, or for a frequency
change exceeding 25%. A requested U output shares the evaluation.

Unchecked auto tracking uses the prediction error to size its local search.
From level 2 it also considers the adaptive cross-level search used by the
low-level GRID curve. It prefers the direct Brent search when the cross-level
change satisfies that criterion. Otherwise it chooses the narrower estimated
interval. The returned root includes its residual for the acceptance check.

To check one stored root, call `compute_mode_count(solution, level=-1)` in
Python or C++. Each call runs the integer counter once on the stored mesh
and the root before extrapolation. For GRID, select a stored level or use
`-1` for the last generated level. Layer-cake models accept `0` or `-1`.
The C and Fortran names are `d26_api_compute_mode_count` and
`compute_mode_count`, respectively. Refinement levels and returned mode
indices are zero-based in every language. An unavailable level raises an error.
A mismatch neither triggers a search nor changes the solution's `mode`,
`mode_verified` or `level_mode_verified`. The check applies to that one root.

```python
periods = [5.0, 5.2, 5.4]
curve = d26.compute_dispersion_curve(
    grid, periods=periods,
    options=d26.Options(method="continuation", verify_mode=False, c_guess=3.0))
last_mode = d26.compute_mode_count(curve.solutions[-1])
first_coarse_mode = d26.compute_mode_count(curve.solutions[0], level=0)
```

By default, internal tracking proceeds from longest to shortest period.
`c_guess` therefore applies at the longest period. `Options(start="short")`
selects the reverse direction. In the example, `[-1]` selects the last
*input* period. Both counters in this example return `0`. Derivative
calculations check stored roots separately.

With verification, auto uses the coarser root at the same period from
level 1 onward. At level 0, auto can compute U on its first step even for
phase-only output. It uses `dc/dω=(c/ω)(1-c/U)` for frequency changes up to
25%, with history or a coarser level available as alternatives.
`predictor="group_velocity"` allows AD evaluation and shares it with a
requested U output. An unrequested public group velocity remains `None`.
If local tracking fails, plain `continuation` searches within ±20% of the
prediction and chooses the nearest root. Recovery and fully verified
methods return to a mode-ordered search. With default tracking controls, an
empty unchecked search raises `root_not_found`.

`solution.mode` always stores the requested index. Ordinary tracked points
in `continuation_recover` may carry an unverified index. Only a solution
checked at all used levels has `diagnostics.mode_verified=True`. Consult
`level_mode_verified` for individual levels. A partial recovery verifies
only the recovered levels. These flags are independent of convergence, and
`strict=True` does not add mode checks.

Inspect `predictors`, `fallback_reasons`, `fallback_count` and
`mode_count_evaluations` for the search history. `predictor_ad_evaluations`
counts AD used for auto startup and explicit group-velocity prediction,
including evaluations shared with a requested U. Before computing kernels,
a Hessian or frequency curvature from an unverified solution, the API checks
each stored level root. A mismatch raises an error with code `mode_mismatch` in Python and C++. Successful checks
are shared internally. Their status is reported on the derived result.

Python and C++ expose tracking options directly. Fortran uses
`computation_options`. In C, initialize the additional `d26_api_tracking_options`
with `d26_api_default_tracking_options` and pass it alongside the existing
`d26_api_options` to `d26_api_compute_dispersion_ex` or
`d26_api_compute_dispersion_curve_ex`. A NULL context allocates a fresh workspace.
Empty `verify` and `on_violation` strings and `verify_mode=-1` leave the option
unspecified. Low-level `disper26py` batch functions share the numerical
kernels but have their own search options and defaults. A shared-API curve
reuses meshes and tracking state within one call. See `DispersionContext`
below for workspace reuse across calls.

### Starting period, velocity range and verification frequency

| Option | Setting | Default |
|---|---|---|
| `start` | `"long"` starts at the longest period, `"short"` at the shortest | `"long"` |
| `c_range` | Initial phase-velocity interval `(lo, hi)` in km/s | Unspecified |
| `verify` | `"none"` skips routine index checks, `"endpoints"` checks the ends of each valid segment, `"every"` checks every period | Method-specific |
| `on_violation` | `"fallback"` searches for the requested index, `"error"` raises, `"missing"` reports a missing value | `"fallback"` with explicit verification or range settings |

```python
options = d26.Options(
    start="long", c_range=(2.8, 4.2),
    verify="endpoints", on_violation="fallback")
curve = d26.compute_dispersion_curve(model, periods=[2., 5., 10., 20.], options=options)
```

This example computes 20, 10, 5 and 2 s and returns results at 2, 5, 10 and
20 s. Repeated periods are computed once. `c_range` overrides `c_guess` at
every generated GRID level of the first valid period. If an overtone is
absent at the longest period, the interval applies at the first period where
it exists. Bounds must be finite and positive with `lo < hi`. The upper
bound is clipped to 0.9995 times the half-space shear velocity (vertical for
Rayleigh, horizontal for Love). An empty clipped range is an input error.
A verified initial range enclosing just the requested index and a finite
sign change of the secular function permits a direct Brent solve. Tangencies
or residual problems use the narrow-interval search.

`verify` overrides the method's verification frequency and is mutually
exclusive with `verify_mode`. With `verify="none"`, a scan checks that it
resolves a single root. Its resolution limits detection of nearby roots,
and the index is unverified. With `on_violation="fallback"`, recovery counts
modes and selects the requested index. Plain `continuation` retains zero
internal counts when verification, range and recovery options are omitted.

`endpoints` checks each level's unextrapolated root at the first and last
period of each valid segment. On an endpoint mismatch, automatic recovery
recomputes from the previous verified point with a check at every period.
The `missing` policy marks that interval missing. Endpoint checks can miss
a jump to another branch and back. Use `every` when an inversion requires
a verified index at every period.

If the count at the upper search limit establishes that the requested mode
is absent, the default returns `NaN`, sets `diagnostics.status="missing"`
and `missing_reason="mode_not_found"`, and continues. A tracking violation
handled as missing gives `missing_reason="tracking_violation"`. Requested U
and H/V are also missing. Unrequested quantities and Love H/V remain `None`.
Tracking restarts after a gap and excludes missing points from the predictor
history. Strict mode raises for missing results. Invalid input and unsupported
conditions always raise. Kernels, Hessians, eigenfunctions, energy integrals and gravity
corrections check the saved solution. Missing roots report `missing_solution`,
and a tracked root belonging to a different mode reports `mode_mismatch` during
derived calculations. The explicit mode-count operation returns the measured index.

### Model arrays

`create_model` requires `kind`, depths and material arrays. Layer-cake depths
start at zero and increase strictly. Each depth marks the top of a constant
layer, thicknesses are depth differences and the last depth starts the
half-space. A single point describes a homogeneous half-space and supports
`compute_dispersion(..., wave="Rayleigh", mode=0)`. Omit `interpolation`
for layer-cake models. GRID needs at least two nondecreasing depths starting
at zero, with at most two entries at a discontinuity. A positive GRID node spacing must be at least 1e-12 km.

Supply exactly one complete set of material arrays: `vs/vp/rho`,
`vph/vpv/vsh/vsv/eta/rho` or `A/C/F/L/N/rho`. Arrays must be finite,
one-dimensional and match the length of the depths. The caller provides
the order and all values. An isotropic fluid has `vs=0`. GRID interpolation
is independent of this input parameterization. The immutable model and
derivative axes follow the same contract as the object-based constructors
`create_layer_cake_model` and `create_grid_model`. Layer-cake metadata expose
`thicknesses` and a separate `halfspace`.

The canonical kind is `layer_cake`. The legacy `kind="pancake"` and
`create_pancake_model` preserve their arguments and defaults. All languages
preserve the construction spelling in model metadata, solutions, flattening
and Q corrections. Both names use the same solver.

## Results and diagnostics

| Item | Contract |
|---|---|
| Output selection | `outputs` selects optional quantities. Phase velocity is always present. U and H/V are opt-in. |
| Unrequested or inapplicable output | Python uses `None`, C++ uses `std::optional`. Love H/V is inapplicable. |
| Numerical zero | Use the output status and active mask to distinguish a computed zero from an uncomputed or fixed component. |
| Gradient | Access e.g. `gradients["phase_velocity"]`. Python's `gradient` is the phase-velocity gradient. |
| Parameter axis | Each entry in `parameters` has `name, entity, index, unit, active`. |
| Fixed parameter | GRID fluid nodes have `active=False` and zero derivatives. |
| Hessian | Second derivatives of phase velocity. Python returns a 2-D array. C++/C return column-major data and shape. |
| Diagnostics | Method, convergence status, accepted level, error estimate and output status. |
| Ownership | Python receives array copies. C handles are reference-counted. Fortran manages ownership through assignment and finalization. |

By default, an extrapolation that misses its tolerance returns a result and
diagnostics, with a Python `ConvergenceWarning`. `Options(strict=True)`
raises when the requested accuracy cannot be verified. Quantities without an
independent error estimate, including some Hessians, report
`convergence="not_evaluated"` and also fail in strict mode.

The dispersion `relative_error` estimates phase-velocity extrapolation error.
Check `convergence_by_quantity` for each observable. Requesting U or H/V leaves
overall accuracy `not_evaluated` even if phase velocity converged, so strict
mode fails. Derived calculations use their own default options unless you pass options for that call. Root tracking uses
the mode and root tolerance stored in the solution and generates additional
levels only as needed.

### Signed Rayleigh ellipticity

`hv_ratio` is the signed ratio epsilon = y3/y1, and `d_HV` and the
`hv_ratio` gradient contain derivatives of that signed quantity. This is a
semantic change from the former amplitude-only value; API names and shapes
are unchanged, including deprecated aliases and the C/Fortran interfaces.
With z upward, exp(i(k*x-omega*t)), u_z=y1*exp(...) and
u_x=i*y3*exp(...), epsilon < 0 is retrograde and epsilon > 0 is prograde
for propagation in positive x. The eigenfunction arrays `ur` and `uz`
store y3 and y1, respectively; the horizontal phase factor i is implicit.

Both layer-cake (`layer_cake`) and GRID models evaluate the solid free surface,
or the solid side of the seabed when surface fluid layers are present.
The common expression is -Y12/Y14, obtained from zero shear traction.
This also corrects the former seabed amplitude, which used a free-surface
normal-traction relation under fluid loading. GRID values and gradients are
extrapolated while signed. A uniform layer perturbation corresponds to
varying both endpoints of its constant GRID segment; sum their node
sensitivities to compare with the layer sensitivity.

For an amplitude observable use E = abs(epsilon). Away from epsilon=0,
dE/dp = sign(epsilon)*d(epsilon)/dp, not abs(d(epsilon)/dp).
Across a smooth zero crossing predict abs(epsilon + gradient @ delta_p).
The signed ratio is differentiable at a horizontal-displacement zero;
the amplitude and normalized sensitivity (p/epsilon)*d(epsilon)/dp are not
regular there. A vertical-displacement zero is a pole of the ratio: values
and sensitivities are undefined. A nonfinite requested dispersion value raises
`numerical_failure` in the shared API.
The existing Love-wave and unrequested-output availability rules still apply.

Both shared and low-level material and interface derivatives require a
non-gravitating reference model. Cowling propagation provides a forward check
of the separate first-order gravity correction.

## Derivative coordinates and supported operations

Specify `basis` independently of model interpolation.

| Basis | Axis order at each layer or node |
|---|---|
| `stiffness_vti` | A, C, F, L, N, Rho |
| `stiffness_iso` | Lambda, Mu, Rho |
| `velocity_iso` | Vs, Vp, Rho (isotropic models only) |
| `velocity_vti` | rho, vph, vpv, vsh, vsv, eta |
| `log_velocity_vti` | Natural logs of the same six quantities. Active coordinates must be positive. |

The two `_iso` bases require an isotropic reference and preserve isotropy during
perturbation. `stiffness_iso` uses the Lamé moduli (GPa). Its density derivative
holds both moduli fixed, while `velocity_iso` holds both wave speeds fixed.
On the same isotropic solution, use `compute_kernels(solution, basis="velocity_iso")`
to vary Vs and Vp or `basis="velocity_vti"` to vary horizontal and vertical speeds
independently. `isotropic` and `stiffness` remain aliases for `velocity_iso` and
`stiffness_vti`. Results echo the requested spelling.

At A = 2L, supply VTI velocity arrays with explicit eta to select the VTI
reference coordinates. Isotropic velocity arrays select eta = 1. Stiffness
alone cannot determine eta at this point. On a velocity-interpolated GRID,
independent VTI stiffness gradients require an invertible velocity map.

The half-space uses `entity="halfspace"`. Hessians in velocity or log
coordinates include `Jᵀ H J` and the gradient contracted with the second
derivative of the material map. `compute_hessian_reduced(solution, projection)`
takes a **linear** stiffness-by-reduced projection. It omits nonlinear map
curvature and computes directly in the reduced basis.

| Operation | Models |
|---|---|
| Dispersion, curves, kernels | Layer-cake, stiffness GRID, velocity GRID |
| Full Hessian | Layer-cake, stiffness GRID |
| Reduced Hessian | Stiffness GRID |
| `compute_interface_kernels` | All-solid models, either layer-cake or flat stiffness GRID |
| `compute_interface_hessian_row` | All-solid flat stiffness GRID |
| `compute_eigenfunctions`, `compute_energy_integrals`, `compute_frequency_curvature` | Models with a stored solution. GRID wavefields and integrals use the accepted mesh. |
| `compute_gravity_correction` | Rayleigh in a flat, non-Cowling layer-cake model, with `region="fluid"`, `"solid"` or `"all"` |

For a contiguous surface fluid stack over solid material, `compute_energy_integrals` with Rayleigh `integration="auto"` uses analytical integrals. The low-level point-based `compute_energy_integrals_analytical_rayleigh` accepts solid models only.

GRID frequency curvature has its own Romberg stopping criterion, so the phase
velocity it returns may differ from that of the original solution. Love mode
searches exclude models with fluid. `get_capabilities(model, wave=...)` reports
support and its reasons. Unavailable methods report `unsupported`. Invalid arguments, including incompatible bases and repeated transformations, report `invalid_argument`. C and Fortran report errors through status codes.
The shared API rejects material and interface derivatives for Cowling-gravity
models because the AD paths omit gravity terms. The usual workflow uses a
non-gravitating solution and a separate Kg correction.

`Options(method="standard")` selects the default. Support for explicit `ad`,
`variational` and `variational_ad` depends on the operation. Use the low-level
API for checks such as fixed-level full AD on a GRID.

## Transformations and PREM

Specify spherical geometry with `Geometry("spherical_ba97")` or
`Geometry("spherical_experimental_density", flattening_exponent=...)`.
Interpolating in spherical geometry and interpolating between preflattened
nodes give different GRID models. `earth_flattening_transformation` returns a
new flat model carrying the transformation history and rejects a second application.
Spherical layer-cake models require explicit flattening. The homogeneous
half-space uses properties flattened at its top depth.

`apply_mineos_physical_dispersion` accepts Q arrays including the half-space
(layer count + 1 for layer-cake, node count for GRID). It stores the target
and reference periods. A corrected model is valid only at its target period
and rejects another correction. For a curve, correct the original model
independently at each period.

The Python helpers `load_prem_spherical` and `load_prem_flat` extend the upper
crust to the surface using bundled data. Their keyword-only controls include
`max_depth_km=500.0`, `max_spacing_km=10.0` and `density_exponent=-2.275`. The mantle-side node at the core boundary (2,891 km in bundled PREM)
can supply the halfspace. A deeper cutoff raises `ValueError`.

The first-order corrections are `delta_c = c(g) − c(0)` and `delta_u = U(g) − U(0)`. Subtract them from the observations for comparison with the reference model. The defaults are `gravity=9.80665e-3` km/s² and `region="solid"`.

The gravity correction uses the perturbation formula with 40 subdivisions per
layer and a relative angular-frequency difference step of 0.005 by default.
Check convergence on the target model when changing these values. The result
reports the original velocities and `delta_c` and `delta_u` separately.

## C++, C and Fortran

For C++, include `"api.hpp"`. Projects using `add_subdirectory` can link the in-tree CMake target `disper26::api`.
The public header has no Eigen or Python dependency. The target can be
used independently of the legacy `disper26_lib`.

```cpp
#include "api.hpp"
int main() {
    namespace d26 = disper26::api;
    d26::MaterialArrays parameters;
    parameters.vs = std::vector<double>{3.0, 4.5};
    parameters.vp = std::vector<double>{5.2, 8.0};
    parameters.rho = std::vector<double>{2.5, 3.3};
    auto model = d26::create_model({0.0, 10.0}, "layer_cake", parameters);
    auto solution = d26::compute_dispersion(model, 5.0);
    auto kernels = d26::compute_kernels(solution, "velocity_iso");
    return kernels.parameters.empty();
}
```

In C, use `d26_api_*` from `api_c.h`. `d26_api_create_model` accepts depths
and a column-major component-by-point matrix. Its three input parameterization
names and component orders are Vs,Vp,Rho
for `isotropic`, rho,vph,vpv,vsh,vsv,eta for `velocity_vti` and A,C,F,L,N,Rho
for `stiffness`. Pass NULL for layer-cake interpolation. Enumerate result
fields and query their sizes before copying them. Complex data use separate
real and imaginary arrays. Errors use a caller-owned buffer. After a successful
call, inspect diagnostics for numerical convergence.

In Fortran, `use disper26_api` and link `disper26_fortran`. Models, solutions
and results are derived types. Pass `status` to each operation. Fortran
arrays, curve solutions, parameters, interfaces and nodes are one-based.
Modes, refinement levels and raw field keys such as `capability.0.operation`
are zero-based in every language. Copy Fortran handles and enclosing types by
assignment. For an allocatable copy, allocate the destination and then assign
to it. Sourced allocation (`allocate(copy, source=original)`) bypasses the
handle's reference-counted assignment. For example, call
`create_model(depths, "layer_cake", model_out, status, vs=vs, vp=vp, rho=rho)`.
For GRID add `interpolation="stiffness"` or `"velocity"`. See
[the Fortran API test](../tests/test_api_fortran.f90) for assignment and lifetime.

## Implementation and verification

`api_backend.hpp` contains the computational adapters formerly in the Python
bindings. `api_impl.hpp` and `api_related_impl.hpp` validate inputs, manage
results and dispatch operations. `api_python.hpp` and `api_c.cpp` convert types.
Fortran delegates to the C ABI. The inversion and long-period Taylor surrogate modules use the same
numerical implementation through their own adapters, parameterization and cache
contracts. Full-AD Hessian work dimensions are thread-local. Parallel tests
compare with serial results using the original formulas and operation order.

First follow the [development and native verification steps](../README.md#verification) to install the build and test dependencies and build the Fortran-enabled targets in `build-native`. The API checks can then be run separately.

```bash
python setup.py build_ext --inplace
python -m unittest discover -s tests/python -p 'test_public_api*.py' -v
python -m unittest discover -s tests/python -p test_model_arrays.py -v
(cd build-native && ctest -R 'public_api|fortran_' --output-on-failure)
```

At introduction (2026-09-24), 26 public Python tests, seven native CTests,
`test_wrapper_full.py`, `test_romberg_level_root_trace.py` and
`test_variational_root_trace_consistency.py` passed. Checks included finite
differences for velocity/log Hessians, EFT/Q and gravity comparisons with
existing paths, PREM from a separately installed wheel, and the Python/C++
examples above. The tracking integration (2026-09-25) passed 64 Python tests
and three native tests covering C++ curves, the C API and concurrent calls.
Those runs excluded large performance measurements. The tracking check also
excluded a Fortran rebuild.

## Reusing workspace capacity (2026-09-26)

Pass `context=d26.create_dispersion_context()` to `compute_dispersion` or
`compute_dispersion_curve`, retaining the context for subsequent calls.
Each call repopulates its arrays from the model. Roots, mesh contents and
predictions are specific to the call. Previously returned results own their
data. Concurrent calls on one context serialize, so create one per worker
for parallel work. A context is reusable after failure.

C++ provides a move-only `DispersionContext` and overloads. C provides
`d26_api_dispersion_context`, creation, retain/release and `*_with_context`
operations. Fortran provides `dispersion_context` and an optional `context`
argument. Assignment shares ownership. Existing C structures and calls retain
their signatures. An uninitialized context is an error.

```python
context = d26.create_dispersion_context()
first = d26.compute_dispersion_curve(model, periods=[8., 10.], context=context)
second = d26.compute_dispersion_curve(model, periods=[12., 14.], context=context)
# first remains usable after the second calculation or context destruction.
```

The context retains capacity for each GRID level's layer arrays, interpolation
arrays and Romberg tables. Destroy the context to release this memory after a
large model. In C/C++, keep the context alive and in place while a call uses
it. In Fortran, create contexts separately for independent workspaces.
