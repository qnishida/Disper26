# GRID weak-form Hessian and Romberg extrapolation

[Japanese edition](../docs_jp/grid_wf_hessian.md)

This document records the design and measurements for the GRID variational
Hessian. Sections 1–5 describe the original trapezoidal-quadrature proposal.
Section 6 records what was implemented. The current shared GRID Hessian uses trapezoidal integrals for Rayleigh and analytical integrals for Love. For the quadrature used by each
observable and wave type today, see the [implementation map](sensitivity_implementation_map.md)
and [shared API](public_api.md). The first design considered three analytical
integration options (A/B/C). The revised design considered the convergence
order of the energy integrals.

## 1. Convergence order and quadrature

### 1.1 Richardson extrapolation

For an even-power error expansion

```
f(h) = f(0) + C₂h² + C₄h⁴ + ...
```

halving the spacing reduces the leading error by a factor of four. Midpoint
sampling in the layer-cake approximation gives this behavior for phase
velocity (observed ratio ≈ 4.0, see [Romberg convergence verification](Romberg.md#convergence-verification)).

### 1.2 Energy integrals

The weak-form gradient depends on the energy integrals through
`g_i = c³/(2ω²I₃) · (∂I₂/∂pᵢ − ω²∂I₁/∂pᵢ)`. The design compared two ways
to evaluate each sublayer contribution.

| Quadrature | Evaluation within a sublayer | Behavior in the original comparison |
|---|---|---|
| Analytical | Closed-form ∫exp(μᵢz)exp(μⱼz)dz for a homogeneous sublayer | The observed errors departed from the expected even-power sequence and extrapolation worsened. |
| Trapezoidal | (f₀+f₁)h/2 from the two boundary states | The observed errors retained the O(h²) structure and extrapolated well. |

The proposed explanation was that the propagator evaluates boundary states
to O(h²) using midpoint material properties, while the analytical integral
is exact within the homogeneous approximation. For the gradient model in
this comparison, these errors combined poorly in a ratio-four extrapolation.
The trapezoidal rule uses the same endpoint states and is also O(h²).
This was consistent with the improved Romberg convergence observed when
the energy calculation switched to trapezoidal integration.

### 1.3 Integration routines at the design stage

| Routine | Quadrature | Wave | Role in the proposal |
|---|---|---|---|
| `computeLoveAnalyticalIntegrals` | Analytical | Love | Examined in the original comparison |
| `computeRayleighEnergyIntegrals_Analytical` | Analytical | Rayleigh | Examined in the original comparison |
| `computeRayleighEnergyIntegrals_CT` with `useAnalytical=false` | Trapezoidal | Rayleigh | Available for extrapolation |
| Love trapezoidal integration | Trapezoidal | Love | Required a new routine |

`computeVariationalGradientRayleigh_T<T>` already selected quadrature with
`useAnalytical`.

## 2. Design of the node-based Hessian

### 2.1 Parameters and matrix size

The layer-cake weak-form Hessian uses the properties p₁,…,p_L of L homogeneous
layers. For npl parameters per layer it has (L×npl)² entries. The GRID model
uses properties q₁,…,q_M at M nodes, with M ≪ L for refined meshes.
The mesher computes `p_layer = f(q_nodes)` by linear interpolation inside
the AD calculation. The Hessian has (M×npl)² entries. An even-power expansion
permits successive extrapolations from O(h²) to O(h⁴) to O(h⁶).

### 2.2 Trapezoidal quadrature with Hessian extrapolation

```
for k = 0, 1, ..., K:
    layers_k, thick_k = mesher.get_refined_layers_ad<adual>(k, node_params_dual)
    H_k = compute_WF_Hessian_trapezoidal(layers_k, thick_k, hs, ω, c)
    # N-pass Hessian using trapezoidal energy integrals (useAnalytical=false)
H_final = neville_extrapolation_matrix(H_0, ..., H_K)  # Elementwise extrapolation
```

The proposal assumed that the O(h²) approximation for propagation and
quadrature would extend to each Hessian entry. If the phase-velocity error
ratio of about four carried over, extrapolation at K=2–3 might attain the
accuracy of a single calculation at K=5–6. Section 6 reports the measured errors.

```cpp
std::vector<Eigen::MatrixXd> H_table;

for (int k = 0; k <= K; ++k) {
    // Generate 2^k sublayers per segment inside the AD calculation.
    auto [layers_k, thick_k] = mesher.get_refined_layers_ad<adual>(k, node_dual);
    auto hs_k = mesher.get_halfspace_ad<adual>(node_dual);

    // Evaluate the weak-form Hessian with trapezoidal integration.
    auto H_k = compute_node_wf_hessian_trapezoidal(
        layers_k, thick_k, hs_k, omega, c_root, g0_node, mode);
    H_table.push_back(H_k);

    // Adaptive stopping proposed for option B
    if (k >= 1) {
        auto err = (H_table.back() - H_table[H_table.size()-2]).norm()
                   / H_table.back().norm();
        if (err < tol || err > prev_err) break;  // adaptive stopping
        prev_err = err;
    }
}

// Apply Richardson extrapolation to each matrix entry and enforce symmetry.
H_final = neville_extrapolation_matrix(H_table);
H_final = (H_final + H_final.transpose()) * 0.5;
```

### 2.3 Two-phase alternative

First converge phase velocity with Romberg to obtain `converged_level`.
Then compute the Hessian once on that level's sublayers. This approach
works with analytical (`useAnalytical=true`) or trapezoidal integration.
Without Hessian extrapolation the result retains the O(h²) layer-cake
discretization error at the selected level.

## 3. Implementation questions at the design stage

### 3.1 Rayleigh quadrature

The Rayleigh trapezoidal path already ran through
`computeRayleighEnergyIntegrals_CT` and `computeVariationalGradientRayleigh_T<T>`.
The proposed change extended `computePhaseVelocityHessianWeakFormRayleigh`
to node-based parameters.

### 3.2 Love quadrature

The existing Love routine was `computeLoveAnalyticalIntegrals`. The proposed
interface below illustrates the design. The implemented routine returns
`LoveEnergyTerms<T>` and takes the subdivision count as an argument.

```cpp
template <typename T>
LoveEnergyResult<T> computeLoveEnergyIntegrals_Trap(
    const std::vector<LoveLayerState<T>>& states,  // Boundary states
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp
) {
    // Integrate u² with the trapezoidal rule: sum (u[j]² + u[j+1]²)/2 * dz.
}
```

The proposal called for a `numSubLayers` argument to `propagateLoveEigenstates`
to return more boundary states. During implementation the existing mesher
proved sufficient (§6.2).

### 3.3 Implicit differentiation in node coordinates

The layer-based implicit-function seed is `g0[j] = ∂c/∂p_layer_j`. GRID
needs `g0[j] = ∂c/∂q_node_j`. One option was to apply the interpolation
transpose to the weak-form layer gradient at the start of phase 2.

```cpp
auto g_layer = compute_wf_gradient_double(layers_k, thick_k, ...);
Eigen::VectorXd g0_node = interpolation_transpose(g_layer, mesher, k);
```

The interpolation weights in stiffness coordinates are parameter-independent,
so this gradient transformation is exact. The alternative used existing
Romberg kernels but required interoperability with `Eigen::AutoDiffScalar`.

### 3.4 Mesher compatibility

`get_refined_layers_ad<T>` constructs `T h = T(sub_thickness)`. This worked
with `Eigen::AutoDiffScalar`. The proposal called for a check with
`autodiff::dual`, which accepts an implicit conversion from `double`.

### 3.5 AD types

| Component | Existing type | Proposed GRID Hessian use |
|---|---|---|
| Romberg kernel dc/dq | `Eigen::AutoDiffScalar` | Retain existing kernel |
| Weak-form Hessian | `autodiff::dual` | Use the same type |
| Mesher interpolation | Templated | Verify instantiation with `autodiff::dual` |

## 4. Estimated cost

The historical estimate counted 15 active parameters (five nodes with three
isotropic parameters each) and five refined segments. A continuous GRID made
from just those five nodes has four intervals. To apply the table to a given
model, use its actual number of intervals. At each level from 0 to K,
multiply the number of AD passes by the total sublayer count, then sum over levels.

| Method | Quadrature | Romberg target | Parameters | Passes | Sublayers per segment | Work estimate |
|---|---|---|---:|---|---|---|
| Layer-cake weak form (10 layers) | Analytical | None | 30 | 30 | — | 30×10=300 |
| Two-phase GRID | Analytical or trapezoidal | Phase 1 | 15 | 15 | 16 (k=4) | 15×80=1200 |
| GRID Romberg Hessian | Trapezoidal | Hessian | 15 | 15×(K+1) | 1,2,4,8 (K=3) | 15×75=1125 |
| GRID Romberg Hessian | Trapezoidal | Hessian | 15 | 15×(K+1) | 1,2,4,8,16 (K=4) | 15×155=2325 |

The estimated saving at K=3 assumed accuracy comparable to a single
high-level calculation at k=5–6.

## 5. Proposed implementation sequence

The main proposal generated 2^k sublayers per segment for k=0,…,K (K=2–4),
computed an N-pass node-based trapezoidal Hessian H_k at each level and
monitored relative changes in its norm. Stopping when the changes grew was
intended to avoid roundoff. The final steps were elementwise Neville
extrapolation and symmetrization. The simpler alternative evaluated one
Hessian on the mesh accepted for phase velocity.

| Priority | Work proposed | Wave |
|---:|---|---|
| 1 | Connect the existing trapezoidal path to a node-based Hessian | Rayleigh |
| 2 | Add the Romberg loop | Rayleigh |
| 3 | Implement trapezoidal integration | Love |
| 4 | Use the new quadrature for the node-based Hessian | Love |
| 5 | Add Python bindings | Both |
| 6 | Compare against finite differences and test extrapolation | Both |

| Operation | Existing Romberg kernels | Proposed GRID Hessian |
|---|---|---|
| Phase 1 | Converge phase velocity | Same |
| Quadrature | Propagation only | Trapezoidal energy integrals |
| AD | First-order AutoDiffScalar, all directions at once | First-order autodiff::dual, N passes |
| Nodes to layers | Mesher interpolation in AD | Same interpolation, transpose for node gradients |
| Implicit-function seed | −(dF/dp)/(dF/dc) | g0_node[j] = ∂c/∂q_node_j |
| Richardson extrapolation in the design | Phase velocity | Hessian entries |
| Output | M×npl gradient | (M×npl)² Hessian entries |

## 6. Implementation and measured accuracy

### 6.1 Routines added

The historical implementation (`0b37bfe`) added `computeLoveEnergyIntegrals_Trap<T>`
and a `useAnalytical` flag on `computeVariationalGradientLove_T` in
`src/sensitivity_utils.hpp`. It added `Mesher::apply_sublayer_grad_to_node()`
in `src/romberg_engine.hpp`, the Rayleigh and Love GRID weak-form Hessians
in `src/grid_hessian.hpp`, and two Python bindings in `src/bindings.cpp`.
Nine historical tests checked accuracy, symmetry and Romberg convergence.
The original scripts are held in the research archive.

### 6.2 Changes during implementation

- The mesher already generates 2^k thin sublayers. The two boundary states
  for each sublayer from `propagateLoveEigenstates` are sufficient for
  `computeLoveEnergyIntegrals_Trap`.
- All internal passes use VTI stiffness coordinates (ρ, A, C, F, L, N).
  Interpolation is linear in these coordinates. The code applies
  `g_node = J^T · g_sub` using the mesher's constant interpolation weights.
  Vs and Vp enter the stiffness map nonlinearly. The Jacobian `J_ip` maps isotropic perturbations (Vs, Vp, ρ) to VTI stiffness perturbations (ρ, A, C, F, L, N) per node. The isotropic Hessian
  is `H_iso = J_ip^T · H_vti · J_ip` plus the second-derivative correction
  from the material map. Only same-node diagonal blocks receive that correction.
- The GRID Hessians reside in `src/grid_hessian.hpp`, included after
  `disper26.cpp`, `sensitivity_utils.hpp` and `romberg_engine.hpp`.
- `Mesher::apply_sublayer_grad_to_node()` implements the interpolation
  transpose proposed in §3.3.

### 6.3 Comparison against finite differences

The early check used ε=0.5. Comparisons with finer central differences
(ε=0.01–0.05) gave the following relative errors.

| Romberg level | Gradient relative error | Hessian relative error |
|---:|---:|---:|
| 3 | 5.8e-6 | 4.6e-5 |
| 4 | 2.8e-7 | 1.8e-5 |
| **5** | **2.1e-9** | **2.2e-7** |
| 6 | 1.4e-10 | 5.9e-8 |

The measured gradient error was 5.8e-6 at level 3 and reached 2e-9
at level 5. The improvement at levels 4–5 was consistent with cancellation
of even powers as the Neville polynomial degree increased.

The [historical convergence experiment](grid_wf_hessian_convergence.md)
gives the recorded model, finite-difference references and limits of the comparison.
