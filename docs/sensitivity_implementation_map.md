# Sensitivity kernels and Hessians

[Japanese edition](../docs_jp/sensitivity_implementation_map.md)

Disper26 has several implementations of phase velocity c, group velocity U,
H/V, their derivatives dc/dp, dU/dp and d(H/V)/dp, and the phase-velocity
Hessian d²c/dp². This guide identifies the methods used for forward modelling,
inversion and independent checks. Dated measurements and commit identifiers
refer to development versions. Some historical scripts and reports are held
in the research archive. For current usage see the [shared API](public_api.md)
and [low-level API](lowlevel_api.md). Runnable regression tests are listed below.

## Classifying the methods

### Variational gradients and AD order

A variational formula provides the first derivative of phase velocity in
closed form. Differentiating that gradient once gives the Hessian. Direct
differentiation through the propagator requires two orders of AD for the
same Hessian. Here L counts layers and N counts parameter directions.

| Target | Variational method | Direct propagator method |
|---|---|---|
| First derivative (dc/dp, dU/dp) | Closed-form phase gradient, or first-order AD of the energy expression `U=I₃/(c·I₁)` | First-order AD of the propagator and implicit differentiation of the root |
| Second derivative (d²c/dp²) | First-order AD of the variational gradient, O(L·N) | Nested second-order `HessianAD<N>`, O(L·N²) |

The GRID material Hessian uses the variational method. Nested AD also appears
in `u_kernel_mode="full_ad"` for dU/dp and the full-AD layer-cake Hessian.
Computing the U value by full AD once per level took about 0.1% of the total
time in the historical timing below.

### Model and basis

Layer-cake calls take `layers, thicknesses, half_space` and operate on fixed
layers without extrapolation. GRID calls take `depths, params, is_discontinuity`,
subdivide segments into 2ⁿ layers and use Neville extrapolation. A `_reduced`
Hessian seeds only basis directions, reducing the number of AD passes from
`n_nodes*6` to `P`. Functions beginning with `debug_` expose diagnostic details.

| Order and method | Layer-cake | GRID |
|---|---|---|
| First, variational | `compute_kernels_variational_layer_cake` | `compute_kernels_variational_grid` |
| First, propagator AD | `compute_kernels_ad_layer_cake` | `compute_kernels_ad_grid` |
| Second, variational | `compute_hessian_variational_ad_rayleigh_layer_cake` / `_love_layer_cake` | `compute_hessian_variational_ad_*_grid[_reduced]` |
| Second, full material propagator Hessian | `compute_hessian_ad_layer_cake` | No equivalent entry in this table |

The standalone first-order GRID variational kernel was added on 2026-08-11.
It returns the same quantity as `gradient_node` from the reduced Hessian.
Its dedicated loop monitors relative changes in the gradient across levels,
so it can stop on the gradient itself and avoids the empty Hessian norm at
`P=0`. See the [naming convention](naming_convention.md).

### Branch identity and direct dispersion roots (2026-08-26)

`compute_dispersion_romberg_levels` owns the discrete root $c_k$ at each
level. Pass its `phase_velocity_roots` and `phase_accepted_level` together to
AD kernels, standalone variational kernels, full or reduced Hessians, and
extrapolated interface gradients or Hessian rows. Each consumer matches the
extrapolation of the accepted prefix against the direct phase velocity,
validates the level residuals and uses those roots directly.

At one known point the direct solver selected Love0, while the former
level-0 local search window included Love1 and missed Love0. For this
reason, a short trace, mismatched accepted phase or residual outside the
direct solver's $|F|<0.1$ criterion raises an error. Extra levels require
an explicit branch and are generated in mode order. Pass both trace fields
from a single solution. `ForwardEngine` retains this provenance in the
`ModeSolution`. Trace API v1 accepts extrapolation starting at level 0.
Ordinary forward calculations generate extra derivative levels only on
request. See the [Romberg trace contract](Romberg.md) and the
[root consistency regression](../tests/python/test_variational_root_trace_consistency.py).

## First-order sensitivity kernels

| Binding | Model | dc/dp | dU/dp | Extrapolation |
|---|---|---|---|---|
| `compute_kernels_ad_grid` | GRID | `AutoDiffScalar<VectorXd>` (528 directions in the example), IFT −(dF/dp)/(dF/dc) | `variational_frequency_ad` by default, also `variational` or `full_ad` | Per-level kernels and Neville extrapolation by default |
| `compute_kernels_ad_grid_velocity_iso` / `_vti` | Velocity GRID | As above | As above | As above |
| `compute_kernels_variational_layer_cake` | Layer-cake | Closed-form variational | `variational_frequency_ad` by default, also `variational` (dual, N passes) | None |
| `compute_kernels_ad_layer_cake` | Layer-cake | Full AD | Nested AD | None |
| `compute_kernels_fd_layer_cake` | Layer-cake | Finite differences | Finite differences | Reference calculation |
| `compute_interface_kernels_variational_rayleigh_layer_cake` / `_love_layer_cake` | Layer-cake | Boundary kernels K_d | — | None |

`compute_kernels_variational_grid` provides the standalone variational phase
kernel for GRID. In layer-cake models, `variational_frequency_ad` supports
Rayleigh analytical integrals and Love material kernels. Rayleigh trapezoidal
integration (including fluid) and Love thickness components fall back to
parameter-direction variational AD component by component. This retains the
requested thickness and interface contributions.

### Layer-cake kernel options

| Keyword | Default | Effect |
|---|---|---|
| `compute_c`, `want_kernel_u`, `want_kernel_hv` | `True` | Select dc/dp, dU/dp and d(H/V)/dp. Unrequested U/HV kernels are skipped and returned as zeros. Computing dU/dp needs dc/dp internally. |
| `u_kernel_mode` | `"variational_frequency_ad"` | Method for dU/dp. `"variational"` differentiates energy integrals in the parameter directions. |
| `compute_thickness` | `False` | Differentiate layer thicknesses. For Love the fallback applies only to requested thickness components. |
| `use_analytical` | `False` | Rayleigh within-layer quadrature. Frequency AD applies to solid models with `True`. |

### GRID kernel options

`compute_kernels_ad_grid` combines the requested derivatives within the AD
calculation. Its switches gate the work for each requested quantity.

| Keyword | Default | Effect |
|---|---|---|
| `compute_c`, `want_kernel_u`, `want_kernel_hv` | `True` | Request dc/dp, dU/dp and d(H/V)/dp kernels. |
| `want_value_u`, `want_value_hv` (formerly `phase_only`) | `True` | Request U and H/V values from the per-level full-AD evaluation. Phase-only evaluation requires both `want_kernel_u=False` and `want_kernel_hv=False`. |
| `u_kernel_mode` | `"variational_frequency_ad"` | Frequency AD of a variational phase kernel. Alternatives are `"variational"` (energy-integral parameter AD) and `"full_ad"` (nested AD). A nonfinite value triggers the existing fallback path. |
| `kernel_extrapolation` | `True` | Compute kernels at all levels and extrapolate with Neville's algorithm. |
| `adaptive_root_scan` | `True` | Adaptive root search (5.8× speedup in the historical P2 measurement). |
| `phase_velocity_roots` | `None` | Use the roots selected by direct Romberg dispersion. A residual mismatch raises an exception. |
| `phase_accepted_level` | `-1` | When passing roots, supply the last level accepted for phase extrapolation by the same direct solve. |

An 88-node shallow GRID run (2026-08-11) distinguished the cost of derivatives
from that of scalar outputs.

| Configuration | Seconds per datum | Computation omitted |
|---|---:|---|
| `want_kernel_u=True, want_kernel_hv=True` (former default configuration) | 6.833 | — |
| `want_kernel_u=False, want_kernel_hv=False` | 1.091 | U and H/V derivatives (84%) |
| Also `want_value_u=want_value_hv=False` (formerly `phase_only=True`) | 1.083 | U and H/V values (0.1% of original runtime) |

Phase velocity and dc/dp agreed bit for bit in all three cases. In a separate
comparison, kernel extrapolation took 0.2020 s against 0.1085 s without
extrapolation and improved the dU error from 2e-4 to 1.6e-9.

Since 2026-08-20, the U-only `variational_frequency_ad` path evaluates
$K_p=dc/dp$ and uses width-one frequency AD in
$dU/dp=U^2 d(\omega K_p/c^2)/d\omega$. This reduces the group-velocity
cost from $O(LP)$ to $O(L+P)$. With `compute_c=False` and
`want_kernel_hv=False`, it skips the width-P AD model altogether. If H/V
kernels are requested, their calculation still uses width P. The
[GRID](../tests/python/test_u_kernel_frequency_variational_grid.py) and
[layer-cake](../tests/python/test_u_kernel_frequency_variational_layer_cake.py)
tests cover this method.

## Second-order derivatives

| Binding | Model | Implementation | Extrapolation |
|---|---|---|---|
| `compute_hessian_ad_layer_cake` | Layer-cake | Nested `HessianAD<N>`, O(L·N²). Canonical name for the former unsuffixed, `_full_ad` and `_reference` variants, checked for Rayleigh and Love on 2026-08-11. | None |
| `compute_hessian_variational_ad_rayleigh_layer_cake` / `_love_layer_cake` | Layer-cake | First-order dual AD of the variational gradient, O(L·N). Production Love implementation. | None |
| `compute_hessian_variational_fd_love_layer_cake` | Layer-cake | Finite differences of the variational gradient, O(L·N²), for comparison | None |
| `compute_hessian_variational_ad_rayleigh_grid` / `_love_grid` | GRID | Variational, all node columns, in `grid_hessian.hpp` | Yes |
| `..._rayleigh_grid_reduced` / `..._love_grid_reduced` | GRID | Seed only the supplied basis directions in `J` | Yes |
| `compute_interface_hessian_row_ad_grid` | GRID | Interface-depth derivatives by nested forward AD | — |
| `compute_interface_hessian_row_variational_ad_grid[_extrapolated]` | GRID | Variational interface Hessian row | In the suffixed variant |

The reduced path seeds `P` basis vectors in place of `n_nodes*6` node-VTI unit
vectors (about 19 passes in place of 528 in the measured case). It returns
`gradient_node` before projection, needed for the curvature correction, and
`hessian_reduced`. It avoids constructing a square node-space Hessian.

### Love Hessians and historical aliases (2026-08-11)

Four historical Love names covered three implementations. The production
`compute_phase_velocity_hessian_weakform_love` used a variational gradient.
`_hybrid_love` recovered eigenfunctions from an AD propagator with two
nested AD levels, O(L·N²). `_variational_love` was an eight-line forwarding
alias for that function. `_variational_fd_love` differenced the gradient.
Their use outside inversion was confined to diagnostics, including five
`diagnose_*` scripts and the archived `verify_hessian_taylor_random.py`.
The hybrid, its alias and `debug_variational_love_hessian_breakdown` were
removed and seven callers updated (historical sequence `77e566d`,
`1de506a`, `2cdf53b`). In particular, the former four-way comparison in
`diagnose_prem_grid_love_layer_cake_bridge.py` compared two aliases for the
same implementation.

The canonical variational AD routine is
`compute_hessian_variational_ad_love_layer_cake`. It seeds the IFT with
`computeLoveIFTGradientFirstOrder` before forming the Hessian. Replacing
the former `HessianAD<80>` seed in `wf-love-seed-gradient` reduced the
10-layer isotropic time from 1.45 to 0.079 ms. Use full AD for an independent
reference or `compute_hessian_variational_fd_love_layer_cake` for a finite-
difference check. The former hybrid was a development diagnostic.

The [naming convention](naming_convention.md) uses *variational* for
stationarity of the Takeuchi & Saito (1972) Lagrangian. The historical
*weakform* label referred to this principle. A previous rename (`0d89d0f`)
had made the distinction from variational finite differences ambiguous.

The binding defaults to `backend="dual"` (the N-pass reference) and
`ForwardEngine` defaults to `"vector"` (blocked vectorized AD). A development
comparison gave 3.4× speedup for GRID Rayleigh and agreement within 3e-12.

## The GRID phase-velocity Jacobian in inversion

The pipeline builds the Jacobian from the variational gradient
(`g_a_variational`, obtained as `gradient_node` with the reduced Hessian).
With `depth_config`, it uses the variational K_d entry `g_variational_depth`
for the depth component. `hessian_reduced` supplies the Hessian. The
independent AD calculation (`compute_kernels_ad_grid`) records a consistency
check on the points selected by `ad_kernel_every`. Other points report NaN.
The following pseudocode shows the row comparison.

```text
G_hess = gradient_log(model0, res["g_a_variational"])
J[i] = project_gradient(G_hess, basis)
if "g_a_ad" in res:
    G_kernel = gradient_log(model0, res["g_a_ad"])
    ad_row = res.get("g_inv_kernel", project_gradient(G_kernel, basis))
    consistency[i] = max(abs(J[i] - ad_row)) / scale
```

| Stage (24 shallow points, serial, 2026-08-11) | Implementation | ms per datum |
|---|---|---:|
| Forward | `compute_dispersion_romberg` | 29 |
| AD Jacobian | U and H/V kernels disabled | 733 |
| Additional variational Hessian | `_reduced`, including `gradient_node` | 36 |

After the 2026-08-11 correction, both derivative methods agreed with finite
differences at about 1e-8. The lower variational cost motivated its use in
production, with propagator AD as an independent check.

### Diagnosing the discrepancies

The research archive retained 42 historical runs with 2,805 points. The
pre-correction `consistency` distribution was as follows.

| Quantile | Consistency |
|---|---:|
| p50 | 5.9e-08 |
| p90 | 1.2e-06 |
| p99 | 5.2e-05 |
| Worst | 4.0e+00 |

The median was close to the measured AD kernel truncation error of 1.7e-08.
The large outliers all occurred on branch 1 (the first overtone). At the
worst point, the root had `rel_err=5.4e-11` and `branch_confirmation=accepted`.
The discrepancies varied smoothly with period. Proximity to layer Vs did not
explain their distribution. For the Love first overtone at 10 s (L1_10s),
`|c−Vs|/c = 0.65`.

At L1_10s, finite differences along 18 basis directions gave relative errors
of order 1e-08 for the variational method and −3.85e-05 for AD in every
direction. The recorded consistency of 3.84e-05 matched the AD error of
3.849e-05. The mean `g_AD/g_FD` was 0.999961511, with a spread of 5.2e-07.
This pointed to a common scalar in `dc/dp = −F_p/F_c`.

### Normalization and the one-sided difference (corrected 2026-08-11)

The numerator used AD. The denominator used a one-sided difference between
two calls using `NormalizationMode::Vertical`. Dividing by the running
`Y(0)` makes the normalization factor s(c) depend on c. At a root, F = 0,
and the two derivative evaluations had the following factors.

```text
computed dF/dc ≈ s(c+ε)·F_c
AD dF/dp       = s(c)·F_p
```

The resulting relative error in the derivative ratio was approximately
−(d ln s/dc)·ε in every parameter direction. Near an overtone node,
`Y(0)` approaches zero and s can change rapidly. The smoother factor on the
fundamental mode made the error less conspicuous.

The correction fixed the gauge using `Provided` and evaluated `F_c` with
width-one AD (`ScalarADc`). The same issue affected `dU/dc` on the full-AD
and fluid paths and `dHV/dc`. Three sites were corrected.

| Case | Before | After |
|---|---:|---:|
| L1_10s | −3.85e−05 | Order 1e−08 |
| L0_10s | −2.39e−08 | 1.6e−09 |
| `consistency` L1_10s | 3.84e−05 | 2.81e−11 |
| `consistency` L0_10s | 2.00e−08 | 6.53e−12 |
| AD/FD ratio, L1_10s | 0.999961511 | 0.999999986 |

The fundamental-mode comparison also improved. Fixing only the gauge had
changed its error from 2.4e−08 to 2.5e−06. The two errors had partially
cancelled. Increasing refinement from levels 4 to 5 and 6 with the experimental
`kernel_extra_levels` left the 3.85e−05 error unchanged and quadrupled the
cost, with convergence order 2.00–2.03. That keyword was removed.

### Finite differences retained in Newton iteration and reference calculations

Fourteen historical sites estimated Newton slopes with
`df = (f(c+eps) − f(c))/eps`. They were in `sensitivity_utils.hpp` (lines
1427, 1515, 1735, 1907, 3184, 5277, 5443, 5853, 5999, 6081), `bindings.cpp`
(1434, 1872) and `grid_hessian.hpp` (348, 545). The approximate Jacobian
affects the iteration path. For a converged solve the stopping condition
sets root accuracy. Switching these sites to AD would add work per iteration.

Finite differences are intentional in `compute_kernels_fd_layer_cake`,
`compute_hessian_variational_fd_love_layer_cake` and `FdDepthHessianBackend`.
The historical inventory also identified `computeNumericalDerivative` in
`disper26.cpp` as unused. At that stage `depth_backend.gradient_row` took
interface-depth derivatives from AD `Depth` entries. The standalone
variational alternative is described below.

A previous error in `compute_kernels_ad_grid` (formerly
`compute_dispersion_romberg_with_kernels`) evaluated kernels at only one
level. A Taylor check in the research archive used the variational gradient
and missed this error. The correction (`e5faa1d`, 2026-07-05) brought the
two gradients into agreement at about 1e-8.

## Options used by inversion

| `ForwardEngine` argument | Default | Purpose |
|---|---|---|
| `want_kernel_u`, `want_kernel_hv` | `False` | Request derivatives of U and H/V |
| `hessian_backend` | `"vector"` | AD backend for the variational Hessian |
| `max_steps`, `tolerance` | 6, 1e-10 | Dispersion Romberg settings |
| `hessian_max_levels`, `hessian_tolerance` | 5, 1e-8 | Hessian refinement |

The engine uses the C++ defaults for `compute_c`, `u_kernel_mode`,
`kernel_extrapolation`, `want_value_u` and `want_value_hv`. Suppressing scalar
values saved about 0.1% in the historical comparison, and `_group_predict`
in `trace_dataset` needs `solution.U` for a warm start. `derivatives.py`
reads `kernel.d_c`. The inversion's `Datum` describes a wave, branch and
period, and its current observables do not use `d_U` or `d_HV`.

`map_over_data` and `run_phase1` expose `ad_kernel_every` in `parallel.py`
and `pipeline.py`. The default of 1 runs AD at every point for
`consistency`. A value of 0 skips this comparison, and a positive N checks
indices 0, N, 2N, and so on. The production Jacobian always uses the variational gradient
and K_d. When AD is skipped, the depth backend must implement
`depth_gradient_variational` if depth is requested. Otherwise the call
raises `RuntimeError`.

## Naming and migration to variational kernels

The [naming convention](naming_convention.md) records the canonical and
compatibility names. The 2026-08-11 audit covered 74 `m.def` names and the
public inversion names. It aligned method tokens with variational evaluation
and AD order, distinguished `d_c`, `d_U` and `d_HV` from the kernel and
value switches, and exposed `compute_kernels_variational_grid`. The detailed
audit is held in the research archive.

After the accuracy correction, the measured costs were 733 ms for AD and
30–77 ms for the variational path.
The production switch on 2026-08-11 proceeded in three steps.

1. B2 (`cfa628a`) added `ad_kernel_every` for independent comparisons,
   retaining every-point checks by default.
2. B3 (`bfe6002`) exposed `compute_interface_gradient_variational_grid`
   (then `compute_interface_gradient_variational_extrapolated`). Using
   doubles only, it agreed with AD `Depth` to 7.7e-12 and took 2–6 ms.
3. B1 switched `pipeline.py` to the variational Jacobian with K_d. The
   velocity part comes from `g_a_variational` and the depth part from
   `VariationalAdDepthHessianBackend.depth_gradient_variational`.
   A backend such as `fd` or `ad` without that depth method raises if AD
   is skipped and a depth gradient is required.

`VariationalAdDepthHessianBackend.interface_hessian_row` (formerly `column`)
uses the level cached by `depth_gradient_variational`. It does not need
`dc_dz`. On the default backend this avoids a separate
`phase_velocity_with_kernels` call on points that skip AD. Compute depth
gradients only when `want_kernels or want_hessian`, so pure forward evaluations
in the `landing.py` Levenberg–Marquardt line search avoid the cost of K_d.

In `test_phase1_weakform_jacobian.py`, `ad_kernel_every=1` and `0` gave
bitwise-identical Jacobians and consistency quantiles within 2.8e-11.
The archived 24-point `verify_inv_phase1.py` comparison gave maximum
consistency 6.33e-11 against the old threshold of 1e-4. Serial and eight-worker
runs gave bitwise-identical Jacobians and Hessians.

## References

- [API naming convention](naming_convention.md)
- [GRID variational Hessian design and convergence](grid_wf_hessian.md)
- [Romberg extrapolation, root traces and options](Romberg.md)
- [Shared API parameter axes and derivative outputs](public_api.md)
- [Output selection regression](../tests/python/test_kernel_target_switch.py)
- [Variational root trace consistency](../tests/python/test_variational_root_trace_consistency.py)
- [GRID group-velocity kernel regression](../tests/python/test_u_kernel_frequency_variational_grid.py)
- [AD kernel sampling regression](../tests/python/test_ad_kernel_thinning.py)
