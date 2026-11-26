# Eigenfunction Calculation Methodology

This document describes how **Disper26** calculates displacement and stress profiles for Rayleigh and Love waves. The Rayleigh reconstruction follows the two-pass method used in Disper80C, a C port of Saito's flat-layer DISPER80 code.

## Two-pass reconstruction for solid Rayleigh waves

We employ a robust **Two-Pass Integration Method** to determine the eigenfunctions. This approach separates the root-finding phase (which uses stable 5x5 Minor Matrices) from the eigenfunction reconstruction phase (which uses 4x4 Propagator Matrices).

### 1. Root Finding (Standard)

For Rayleigh waves, the Saito/Dunkin method finds the phase velocity $c$ using
$5\times5$ reduced compound matrices. At high frequencies or in thick layers,
large exponential terms in a direct propagator can cancel, causing a loss of
precision in the dispersion function. Propagating the minors reduces this
cancellation. Rescaling the state between layers limits amplitude growth,
but an individual layer propagator can still overflow if its exponential
terms exceed the floating-point range. The Love-wave calculation uses a
$2\times2$ SH propagator.

### 2. Eigenfunction Pass 1: Upward Propagation (Minors)
Once the phase velocity $c$ is known, we perform an upward integration of the **5x5 Minor Matrix** from the bottom half-space to the surface.
*   The five-component minor vector contains two-by-two determinants of the two independent decaying half-space solutions.
*   We **store the Minor Vector** at each layer interface during this upward pass. These stored minors are crucial for the stabilization in the next step.

### 3. Eigenfunction Pass 2: Downward Integration (Reconstruction)

At a solid free surface, the implementation selects the larger pivot from
$[Y_{14},0,-Y_{12},0]$ and $[-Y_{12},0,Y_{23},0]$ before normalization. These
represent the same traction-free state at a root; selecting the pivot avoids
a vanishing initial vector at either horizontal or vertical displacement zeros.

With the Minors stored, we perform a **Downward Integration** of the four-component state vector from the surface to the bottom half-space.
*   **Initial Condition**: The surface state $(u_r, u_z)$ is derived from the Surface Minor Vector to strictly satisfy the Stress-Free Boundary Condition $(\sigma_{zz}=0, \sigma_{zr}=0)$.
*   **Interface stabilization.** When the stored minor $Y_{13}$ is large
    enough to use as a pivot, the Plücker coordinates $Y_{ij}$ and the
    propagated displacements $(u_r,u_z)$ determine the interface tractions
    $(\sigma_{zz},\sigma_{zr})$. Reconstructing the tractions this way limits
    contamination by the parasitic growing solution excited by roundoff
    during downward propagation.
*   **Sublayer Integration**: Within each layer, standard 4x4 propagation is used to obtain high-resolution profiles.

> [!NOTE]
> In Saito's DISPER80 FLAT-VER.1 and its C port, $y_1$ is the vertical displacement amplitude and $y_3$ is the reduced horizontal amplitude along the propagation direction. At a solid free surface, `ELIP` is the signed horizontal-to-vertical ratio $H/V=y_3/y_1=-Y_{12}/Y_{14}$, where the uppercase $Y_{ij}$ are minors. See the `RAYINT` routine in [Saito's Fortran listing](https://www.jsce.or.jp/library/eq10/book/49198/3-0000.pdf).
>
> Disper26 returns the same reduced amplitudes as `uz` ($y_1$) and `ur` ($y_3$). With $z$ positive upward and phase factor $e^{i(kx-\omega t)}$, the complex displacement amplitudes are $y_1$ vertically and $i y_3$ horizontally. For $k>0$, negative $y_3/y_1$ indicates retrograde motion and positive $y_3/y_1$ indicates prograde motion.

## Verification Results

The public regression suite checks energy balance, group-velocity consistency,
and eigenfunction boundary conditions; see the test scripts below.

### Kinetic and potential energy equipartition

For each normal mode of a lossless elastic model without gravity, the
cycle-averaged kinetic energy equals the cycle-averaged strain energy after
integration over the full depth, including the half-space. At the eigenfrequency,
Rayleigh's variational principle gives $I_2=\omega^2 I_1$. We check the
computed eigenfunction using the dimensionless ratio

$$
\frac{I_2}{\omega^2 I_1} \approx 1.
$$

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

Use the high-level API (`disper26`) to get the eigenfunction profile on the
mesh of a stored solution. This example computes the Love profile for a solid
layer-cake model.

```python
import disper26 as d26

model = d26.create_model([0.0, 15.0], kind="layer_cake",
                         vs=[3.2, 4.5], vp=[5.8, 8.0], rho=[2.6, 3.3])
period = 10.0  # s
solution = d26.compute_dispersion(model, period=period, wave="Love")
fields = d26.compute_eigenfunctions(solution).components
```

To evaluate the state at a specific depth between mesh points, continue with
the low-level API (`disper26py`). Here `disper26py.eval_global_state` takes
eleven positional arguments and propagates the surface state to 2 km depth.
The sign change converts between the two traction conventions.

```python
import math
import numpy as np
import disper26py as d26_lowlevel

surface = np.array([fields["uy"][0], -fields["tyz"][0]])
layer = d26_lowlevel.create_isotropic_parameters(3.2, 5.8, 2.6)
halfspace = d26_lowlevel.create_isotropic_parameters(4.5, 8.0, 3.3)
state = d26_lowlevel.eval_global_state(
    [layer], [15.0], halfspace, surface, 2.0,
    2 * math.pi / period, solution.phase_velocity,  # rad/s, km/s
    "Love", "LAYER_CAKE", 5, 1e-8,
)
print(state[0], -state[1])  # uy, physical tyz at depth 2 km
```

The final two arguments set the maximum quadrature depth and tolerance for
`"GRID"` propagation. This low-level call takes one material per finite layer
and uses it at both ends. For profiles in a model with depth-varying material,
pass its GRID solution to `disper26.compute_eigenfunctions`. Use that same shared
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
