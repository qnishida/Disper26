# API naming convention

[Japanese edition](../docs_jp/naming_convention.md)

## 2. Function names

```
<verb>_<quantity>[_<method>][_<wave>][_<model>][_<variant>…]
```

The slots follow the order in the historical name
`compute_phase_velocity_hessian_weakform_rayleigh_grid_reduced`.

## 3. Vocabulary

### Verb

Use `compute` for numerical quantities. Older code also uses `calculate`.
Use `debug_` for a diagnostic, as in `debug_hessian_variational_ad_love_layer_cake`.
Established verbs for other operations include `create_*`, `search_roots`,
`apply_*`, `earth_flattening_*`, `get_*`, `eval_*` and `propagate_*`.

### Quantity

| Token | Quantity |
|---|---|
| `dispersion` | Phase velocity c, optionally with group velocity U and H/V |
| `dispersion_curve` | Phase velocities along a monotonic period sequence with convergence and fallback diagnostics |
| `group_velocity`, `hv_ratio` | U or H/V returned individually |
| `frequency_curvature` | dU/dω, d²k/dω² and dimensionless curvature diagnostics on a real dispersion branch |
| `kernels` | Depth distribution of ∂c/∂p by layer or GRID node |
| `gradient` | Assembled vector of ∂c/∂p |
| `hessian` | ∂²c/∂p² |
| `interface_kernels` | Raw boundary kernels K_d for a layer-cake model |
| `interface_gradient` | GRID depth derivative dc/dz assembled from K_d and geometric weights |
| `interface_hessian_row` | A row of the GRID Hessian obtained by differentiating the interface gradient |
| `eigenfunctions`, `energy_integrals`, `mode_count`, `continuous_mode_count` | Eigenfunctions, energy integrals and the two mode-count diagnostics |

### Method

| Token | Differentiation method | AD order |
|---|---|---|
| `variational` | Closed-form variational gradient at stationary displacement (δ_uL = 0) | 0 |
| `variational_ad` | First-order AD of the variational gradient or K_d, historically called a weak-form Hessian | 1 |
| `variational_fd` | Finite differences of the variational gradient | — |
| `ad` | AD through the propagator and implicit differentiation of the root | 1 for kernels, 2 (nested) for Hessians |
| `fd` | Finite differences of c | — |

Method-specific low-level names for `kernels`, `gradient`, `hessian` and
`interface_*` must include the method. A bare name such as
`compute_phase_velocity_hessian` is ambiguous at this layer. The vocabulary
distinguishes use of the variational principle and the order of AD. Names for
ordinary dispersion, group velocity, H/V, eigenfunctions, integrals and mode
counting do not require a derivative-method slot. Historical tokens include
`weakform`, `hybrid`, `full_ad`, `stable` and `reference`. The transitional
`hybrid` implementations need individual interpretation.

### Wave

Use `rayleigh` or `love`. Omit the slot when a wave argument selects the
implementation (`WaveType` in C++ or a string in Python).

### Model

Write `layer_cake` or `grid` explicitly when the operation distinguishes
models. Treat `layer_cake` as one model name. Omit the slot for utilities
that have no model-specific implementation.

### Variant

Use `_reduced` for basis reduction. Use `_extrapolated` only when there is
also a non-extrapolated function with the same base name. GRID Hessians that
always use Neville extrapolation need no suffix. The two variants of the
interface Hessian row illustrate the paired naming.

## 4. Terms with multiple historical meanings

| Older terms | Convention |
|---|---|
| `Reduced` | Reserve for basis reduction. Use `MinorIndex` for the five-component compound-matrix representation historically called `ReducedIndex` in `disper26.cpp`. |
| `Romberg` | Reserve for extrapolation across levels. Use `quadrature` for within-layer integration, historically `integrateGridLayer_Romberg` and `propagate_romberg_vector`. |
| `Liquid`, `Fluid` | Use `fluid`. |
| `Boundary`, `Interface`, `Depth` for K_d | Use `interface`. |
| `Bran`, `AdModeCount`, `Total`, `Continuous` | Use `mode_count` for the canonical count and `continuous_mode_count` for the legacy diagnostic. |
| `Pieces`, `Breakdown`, `Trace`, `Probe` | Prefix diagnostic functions with `debug_`. |
| `Weakform`, `WeakForm` | Use the method vocabulary above. |
| `compute`, `calculate` | Use `compute`. |

## 5. Keyword arguments, dictionary keys and fields

- Distinguish requests for values (`want_value_u`, `want_value_hv`, historically
  combined in `phase_only`) from requests for kernels (`want_kernel_u`,
  `want_kernel_hv`). Older bindings used `compute_u` and `compute_hv`, and
  inversion code used `want_u_kernel` and `want_hv_kernel`.
- Use `sensitivity_mode` for the derivative parameterization (older spellings
  include `mode`, `mode_str` and `parameter_mode`) and `wave_type` for the
  low-level wave selector (formerly also `wave_type_str`).
- `SensitivityKernel.d_c` is the layerwise distribution. `gradient` and
  `gradient_node` are assembled vectors in `HessianResult` and
  `GridHessianReducedResult`. When the source matters, use method names,
  e.g. `g_a_ad`, `g_a_variational` and `g_variational_depth` for the quantities
  historically called `g_a_kernel`, `g_a_hessian` and `g_wf_depth`.
- Use `interface_hessian_row` in both Python and C++. The older
  `VariationalDepthHessianBackend.column()` returned the transposed quantity.
- Pass `phase_velocity_roots` (raw roots at each level) together with
  `phase_accepted_level` (the last level used in direct phase extrapolation).
  `minimum_trace_level` specifies how far to extend the trace.
  `root_tolerance` controls root solving and `tolerance` controls phase extrapolation.

The depth Hessian backend registry and its reported method use the same key.

| Historical registry key | Historical class | Historical method string | Canonical key and method |
|---|---|---|---|
| `finite_difference` | `FiniteDifferenceDepthHessianBackend` | `central_fd_of_ad_gradient` | `fd` |
| `nested_ad` | `AutoDiffDepthHessianBackend` | `nested_ad` | `ad` |
| `variational` | `VariationalDepthHessianBackend` | `variational_extrapolated` | `variational_ad` |

The last method applies first-order AD to the closed-form K_d. The corresponding
class names are `FdDepthHessianBackend`, `AdDepthHessianBackend` and
`VariationalAdDepthHessianBackend`. Define defaults at the highest calling
layer, here `ForwardEngine`. Historically the binding default was `"dual"`
and the engine default was `"vector"`. Preserve C++ defaults needed for
compatibility and pass an explicit backend in new calls.

A historical Phase-1 cache marker in `inversion/disper26_inv/pipeline.py`
used `"depth_backend": "central_fd_of_ad_gradient"` even after the default
changed to `"variational"` by 2026-08-11. That marker distinguished cache
contracts. The canonical method string for this backend is `"variational_ad"`.

## 6. Compatibility

### Shared public API (September 2026)

Use `create_model(depths, kind=..., ...)` with `kind="layer_cake"` or
`"grid"` and one complete set of isotropic, VTI velocity or stiffness
arrays. The object-based constructors are `create_layer_cake_model` and
`create_grid_model`.

Preserve the `pancake` aliases, arguments and defaults without introducing
a runtime warning or a removal deadline. Use `layer_cake` in new code and
documentation. Models built with `kind="layer_cake"` or `create_layer_cake_model`
report `"layer_cake"`. The legacy `kind="pancake"` and `create_pancake_model`
report `"pancake"`. The spelling persists in solutions, flattening, Q
corrections and C/Fortran model metadata. Both spellings use the same solver.

In the typed `disper26`, `disper26::api` and `disper26_api` interfaces,
`Model` and `ModeSolution` carry the model and wave, and `Options.method`
selects the method. This layer omits method/model suffixes in `compute_kernels`,
`compute_hessian`, `compute_hessian_reduced`, `compute_interface_kernels` and
`compute_interface_hessian_row`. These interfaces own their results and
validate inputs and parameter axes. Method-specific low-level `disper26py`
names retain their conventions. The [shared API guide](public_api.md)
defines `wave`, `mode`, `basis`, `outputs`, options, fields and axis order.
Reserve `mode` for the mode index in the shared API.

### Low-level aliases

Register both names with `m.def` and begin the old docstring with
`Deprecated alias. Use <canonical name>.` Keep the old C++ signature as
an inline forwarding function. Use canonical names in new code and in the
README. Previously documented aliases must remain callable.
