# Eigenfunction Calculation Methodology

This document describes the methodology used in **Disper26** to calculate eigenfunctions (displacement and stress profiles) for Rayleigh and Love waves. The implementation aims for high numerical stability and consistency with the legacy **Disper80C** (Saito) algorithm.

## Two-pass reconstruction for solid Rayleigh waves

We employ a robust **Two-Pass Integration Method** to determine the eigenfunctions. This approach separates the root-finding phase (which uses stable 5x5 Minor Matrices) from the eigenfunction reconstruction phase (which uses 4x4 Propagator Matrices).

### 1. Root Finding (Standard)
The phase velocity $c$ is first determined using the **Saito/Dunkin** method with 5x5 Minor Matrices (for Rayleigh) or 2x2 matrices (for Love). This ensures numerical stability even for high frequencies and thick layers, avoiding the "large exponent" problem.

### 2. Eigenfunction Pass 1: Upward Propagation (Minors)
Once the phase velocity $c$ is known, we perform an upward integration of the **5x5 Minor Matrix** from the bottom half-space to the surface.
*   The five-component minor vector contains two-by-two determinants of the two independent decaying half-space solutions.
*   We **store the Minor Vector** at each layer interface during this upward pass. These stored minors are crucial for the stabilization in the next step.

### 3. Eigenfunction Pass 2: Downward Integration (Reconstruction)

At a solid free surface, the implementation selects the larger pivot from
`[Y14, 0, -Y12, 0]` and `[-Y12, 0, Y23, 0]` before normalization. These
represent the same traction-free state at a root; selecting the pivot avoids
a vanishing initial vector at either horizontal or vertical displacement zeros.

With the Minors stored, we perform a **Downward Integration** of the four-component state vector from the surface to the bottom half-space.
*   **Initial Condition**: The surface state $(u_r, u_z)$ is derived from the Surface Minor Vector to strictly satisfy the Stress-Free Boundary Condition $(\sigma_{zz}=0, \sigma_{zr}=0)$.
*   **Interface Stabilization**: At each layer interface, instead of simply propagating the state, we **reconstruct** the stress components $(\sigma_{zz}, \sigma_{zr})$ using the stored Minor Vector and the current displacements $(u_r, u_z)$.
    *   This technique (adopted from **Disper80C**) prevents the "Large Exponent Problem" where the numerical solution can drift onto a growing non-physical mode.
    *   Formula: $\sigma_{zz}, \sigma_{zr}$ are calculated algebraically from $u_r, u_z$ and the Minors $Y_{ij}$ using Plucker relations.
*   **Sublayer Integration**: Within each layer, standard 4x4 propagation is used to obtain high-resolution profiles.

> [!NOTE]
> **Variable Definitions & Consistency**:
> *   **Disper80C (Saito)** uses variables $y_1=u_r, y_3=u_z$. Thus, the ellipticity `ELIP` in legacy code corresponds to $V/H = u_z/u_r$.
> *   **Disper26 (Ikeda)** returns the reduced amplitudes `uz=y1`, `ur=y3`. For $e^{i(kx-\omega t)}$ and upward $z$, the physical horizontal displacement is $u_x=i y_3 e^{i(kx-\omega t)}$. Thus signed ellipticity is $y_3/y_1$, with negative retrograde and positive prograde motion for $k>0$.
> *   The Disper26 implementation internally handles this mapping, ensuring that the stabilized reconstruction formulas are physically consistent with the Ikeda basis.

## Verification Results

The public regression suite checks energy balance, group-velocity consistency,
and eigenfunction boundary conditions; see the test scripts below.

### Energy Integral Balance
A stringent test of eigenfunction accuracy is the **Energy Integral Balance**:
$$ \frac{\text{Potential Energy Integral } (I_2)}{\omega^2 \cdot \text{Kinetic Energy Integral } (I_1)} \approx 1.0 $$

The original convergence studies reported typical errors $< 10^{-6}$ for
Rayleigh waves and $< 10^{-8}$ for Love waves. Their diagnostic scripts are
historical research records and are not bundled. These are historical results,
not error bounds for arbitrary models or a newly tested release.

### Group Velocity Consistency
We calculated Group Velocity using two independent methods:
1.  **AutoDiff (AD)**: Exact derivative $U = d\omega/dk$.
2.  **Energy Integrals**: Variational principle $U = I_3 / (c I_1)$.

The cited tests compare the two group velocities on their specified models. For example, the multilayer Love analytical-integral test uses a relative tolerance of $10^{-10}$.

## Test Scripts

The current regression suite includes:

*   [Regular finite-layer integrals](../tests/python/test_regular_energy_integrals.py).
*   [Love energy balance and group velocity](../tests/python/test_love_energy_integrals.py).
*   [Fluid Rayleigh eigenfunctions](../tests/python/test_fluid_rayleigh_eigenfunctions.py).

See [energy-integral conventions](energy_integral.md) for signs and the
specific tolerances used in these checks.

## Unified State Evaluator

The shared API returns profiles on the mesh used by a stored solution.
For a solid layer-cake model, `disper26py.eval_global_state` can also propagate
a supplied surface state to a depth in km. All eleven arguments are positional.
The example evaluates a Love eigenfunction 2 km below the surface and converts
the depth-oriented traction returned by the low-level call to the public convention.

```python
import math
import numpy as np
import disper26 as d26
import disper26py as dp

model = d26.create_model([0.0, 15.0], kind="layer_cake",
                         vs=[3.2, 4.5], vp=[5.8, 8.0], rho=[2.6, 3.3])
period = 10.0  # s
solution = d26.compute_dispersion(model, period=period, wave="Love")
fields = d26.compute_eigenfunctions(solution).components
surface = np.array([fields["uy"][0], -fields["tyz"][0]])
layer = dp.create_isotropic_parameters(3.2, 5.8, 2.6)
halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
state = dp.eval_global_state(
    [layer], [15.0], halfspace, surface, 2.0,
    2 * math.pi / period, solution.phase_velocity,  # rad/s, km/s
    "Love", "LAYER_CAKE", 5, 1e-8,
)
print(state[0], -state[1])  # uy, physical tyz at depth 2 km
```

The final two arguments set the maximum quadrature depth and tolerance for
`"GRID"` propagation. This low-level call takes one material per finite layer
and uses it at both ends. For profiles in a model with depth-varying material,
pass its GRID solution to `d26.compute_eigenfunctions`. Use that same shared
API for Rayleigh depth profiles: the generic low-level Rayleigh evaluator uses
a positive propagator increment for increasing depth, while the public
Rayleigh reconstruction uses a negative increment for downward propagation. A surface ocean uses
the acoustic state and solid-side reconstruction described in [fluid layers](fluid_layers.md).

Love propagation internally uses `[u_y,T]`, with `T=c44*du_y/dz` and `z`
positive upward; `compute_eigenfunctions_love` and the common eigenfunction
API return `tyz=T` at depths positive downward. For compatibility, the generic
Python state-vector utilities `eval_global_state`, `eval_layer_state`,
`propagate_layer_quadrature_vector`, and `integrate_grid_layer_quadrature`
retain their historical Love vector `[u_y,c44*du_y/ddepth]=[u_y,-T]` at their
inputs and vector outputs. Convert the second component when supplying a
public eigenfunction to these generic utilities. Their bindings perform the
reflection to/from the internal upward state. `compute_layer_matrix_sh` keeps
its matrix entries and interprets its signed distance as an upward increment
for `[u_y,T]`.
