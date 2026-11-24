# Fluid Layer (Ocean) Implementation

Disper26 supports fluid (liquid) layers for modeling ocean acoustic propagation and Rayleigh waves in ocean-seabed structures.

## Mathematical Formulation

### State Vector

In fluid layers, the motion is described by a 2-component state vector:

$$\mathbf{Z} = \begin{pmatrix} w \\ P \end{pmatrix}$$

where:
- $w=U$ = upward vertical displacement
- $P=T_{zz}$ = Lagrangian normal traction

### Propagator Matrix

For zero gravity, the propagator maps the bottom state to the top across a layer of positive thickness $H$:

$$\mathbf{B} = \begin{pmatrix} 
\cosh(\nu H) & -\frac{\nu}{\rho \omega^2} \sinh(\nu H) \\
-\frac{\rho \omega^2}{\nu} \sinh(\nu H) & \cosh(\nu H)
\end{pmatrix}$$

where:
- $\nu^2 = k^2(1 - c^2/V_p^2)$ (vertical wavenumber squared)
- $H$ = layer thickness
- $\rho$ = density
- $V_p = \sqrt{c_{33}/\rho}$ = P-wave velocity

At $c=V_p$, $\nu^2=0$. The implementation evaluates the regular limit
$\sinh(\nu H)/\nu=H$ at zero, so the diagonal entries are one and the
off-diagonals are $0$ and $-\rho\omega^2 H$ (upper and lower, respectively).

### Solid-Liquid Transition

At the solid-liquid interface, the 5D minor vector $\mathbf{Y}$ is reduced to the 2D state:

$$Z_1 = Y(I_{14}), \quad Z_2 = Y(I_{24})$$

At the liquid free surface, the default secular condition is $P = 0$.

Here $P=T_{zz}$ is the Lagrangian normal traction, not the Eulerian pressure
perturbation. The standard seabed conditions retain continuity of normal
displacement and Lagrangian normal traction, with zero shear traction on the
solid side. No separate $\Delta\rho\,g\,U$ jump is added. When fluid Cowling
gravity is enabled, it modifies fluid propagation; the solid propagation
equations remain non-gravitating.

### Rayleigh Eigenfunctions

`compute_eigenfunctions_rayleigh` supports an ocean stack at the top of a solid
model. In fluid layers the returned Rayleigh fields use the same public keys as
solid layers:

$$
u_z = U,\qquad \sigma_{zz}=T_{zz},\qquad \sigma_{zr}=0,
$$

and the horizontal displacement is reconstructed from the fluid equations,

$$
u_r =
\frac{k}{\rho\omega^2}
\left(\rho g_{\rm internal}U-T_{zz}\right).
$$

Here $g_{\rm internal}$ is `gravity * gravity_internal_scale` in km/s²
(default `gravity=0`). The optional `gravity_surface_scale`, zero by default,
controls the independent pressure correction at the free surface.
$T_{zz}$ is the Lagrangian traction, so the Eulerian pressure perturbation
is $p' = \rho g_{\rm internal} U - T_{zz}$ and horizontal momentum balance
$\omega^2 \rho\, u_r = k\, p'$ gives the sign above.

The returned dictionary also includes `is_fluid`, a boolean array marking
samples inside the fluid stack. At the seabed, the solid-side state is recovered
from the existing minor-vector interface relation; duplicated interface depths
may appear so that fluid-side and solid-side states remain distinct.

### Rayleigh Variational Gradients

Both the trapezoidal and the analytical Rayleigh weak-form paths support top
fluid stacks. Fluid layers use the acoustic frozen-displacement kernels

$$
K_\rho = -\omega^2\int (u_z^2+u_r^2)\,dz,\qquad
K_C = \int \frac{T_{zz}^2}{C^2}\,dz ,
$$

with no sensitivity to shear moduli. In isotropic output, the fluid `Vs`
gradient is zero and the `Vp`/`Rho` gradients use
$C=\rho V_p^2$.

With `useAnalytical=true`, the integrals over a fluid layer are closed-form
moments of the state $[U,T_{zz}]$ taken from the top of the layer
(`fluid_moments` in `src/energy_entire.hpp`). The coefficient matrix $G$ of the
fluid propagator has zero trace, so $G^2=\delta^2 I$ and the state at local
depth $x$ is $\mathcal C\,y_0-x\mathcal S\,Gy_0$ with the entire functions of
$\delta^2x^2$; the moments need $\int\mathcal C^2$, $\int x\mathcal C\mathcal S$
and $\int x^2\mathcal S^2$ only, and remain regular at $\delta=0$. The
horizontal displacement follows from the fluid relation above, including the
constant-$g$ term. The kernels agree with Full AD to roundoff.

By design the energy functional and the variational kernels are those of the
non-gravitating reference model; background gravity is applied separately as
the first-order $K_g$ correction, and the shared and low-level derivative APIs reject
gravitating models. Fluid Cowling propagation is kept as a cross-check only.
Use a non-gravitating reference for material kernels, Hessians and interface
derivatives. The forward Cowling root, group velocity and signed ellipticity
remain available.

For Layer-cake models the public `integration="auto"` option selects these
analytical integrals when the fluid layers form a contiguous stack at the top
of the model above a solid halfspace. Buried fluids and a fluid halfspace are outside
the supported ocean geometry. Pass `integration="trapezoidal"` to use the
trapezoidal integrals as a check. The point-based
`compute_energy_integrals_analytical_rayleigh` remains solid-only.

Interface-depth derivatives (`compute_interface_kernels`,
`compute_interface_hessian_row`) are not available for models with fluid
layers: the interface jump uses the solid spatial Hamiltonian, which contains
$\sigma_{zr}^2/L$ and has no fluid form. Both entry points raise, and
`get_capabilities` reports the reason.

### Grid/Romberg Rayleigh Weak-Form Hessian

`compute_hessian_variational_ad_rayleigh_grid` supports a top fluid
stack as a fixed known medium. The Rayleigh wavefield and weak-form integrals
include the fluid layers, but fluid grid nodes are not AD variables. Their
gradient entries and Hessian rows/columns are returned as zero in both
isotropic and VTI output modes.

Represent the seabed as a duplicate-depth discontinuity, for example
`[ocean@z_bed, solid@z_bed]` with `is_discontinuity` set on the upper seabed
node. Continuous interpolation between `c44=0` fluid nodes and `c44>0` solid
nodes is rejected because it creates an artificial transition layer.

## Usage

### C++ API

Use `add_subdirectory` and link the `disper26::api` CMake target. This example prints the fundamental
Rayleigh phase velocity in km/s at a period of 2 s.

```cpp
#include "api.hpp"
#include <iostream>

int main() {
    namespace d26 = disper26::api;
    const auto ocean = d26::create_fluid_parameters(1.5, 1.0);
    const auto sediment = d26::create_isotropic_parameters(0.5, 2.0, 1.8);
    const auto basement = d26::create_isotropic_parameters(3.0, 5.0, 2.5);
    const auto model = d26::create_layer_cake_model({ocean, sediment}, {2.0, 1.0}, basement);
    const auto solution = d26::compute_dispersion(model, 2.0, "Rayleigh", 0);
    std::cout << solution.phase_velocity() << '\n';
}
```

### Python API

Use the high-level API (`disper26`) with `mode=0` to request the fundamental
as in the C++ example. The low-level API (`disper26py`) can scan an explicit
interval. This example finds sign-changing Rayleigh roots between 1.6 and
2.9 km/s and returns them in order within that interval.

```python
import disper26py as d26_lowlevel
import math

# Create fluid layer
ocean = d26_lowlevel.create_fluid_parameters(vp=1.5, rho=1.0)

# Create solid layer
sediment = d26_lowlevel.create_isotropic_parameters(vs=0.5, vp=2.0, rho=1.8)
halfspace = d26_lowlevel.create_isotropic_parameters(vs=3.0, vp=5.0, rho=2.5)

layers = [ocean, sediment]
thicknesses = [2.0, 1.0]  # km

omega = 2.0 * math.pi * 0.5
roots = d26_lowlevel.search_roots(layers, thicknesses, halfspace, omega, 1.6, 2.9, 100, "Rayleigh")
print(roots)  # Phase velocities in km/s at 0.5 Hz
```

## Multi-Layer Ocean (SOFAR Channel)

To model a depth-varying sound speed, replace the single ocean material and
thickness in the C++ example with four fluid layers above the sediment.
In top-to-bottom order, their (Vp in km/s, density in g/cm³, thickness in km)
are (1.52, 1.025, 0.2), (1.48, 1.025, 0.5), (1.46, 1.025, 0.5)
and (1.50, 1.025, 1.0).

## Known Limitations

| Issue | Description |
|-------|-------------|
| Liquid half-space | Not supported (ocean must be on top of solid) |
| Buried fluid | Fluid layers must form a contiguous stack at the top of the model |
| Analytical Rayleigh weak-form integrals | Top fluid layers supported; a fluid half-space and the point-based energy-integral routine are not |
| Grid/Romberg Rayleigh weak-form Hessian | Fluid nodes are treated as fixed; use a duplicate-depth seabed discontinuity |

## References

- Ikeda & Matsuoka (2013): Formulation of layer matrix
- Woodhouse (1988): Phase definition for mode counting
