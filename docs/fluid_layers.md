# Fluid Layer (Ocean) Implementation

Disper26 supports fluid (liquid) layers for modeling ocean acoustic propagation and Rayleigh waves in ocean-seabed structures.

## Mathematical Formulation

### State Vector

In fluid layers, the motion is described by a 2-component state vector:

$$\mathbf{Z} = \begin{pmatrix} w \\ P \end{pmatrix}$$

where:
- $w$ = vertical displacement
- $P=T_{zz}$ = Lagrangian normal traction

### Propagator Matrix

The layer propagator is:

$$\mathbf{B} = \begin{pmatrix} 
\cosh(\nu H) & -\frac{\nu}{\rho \omega^2} \sinh(\nu H) \\
-\frac{\rho \omega^2}{\nu} \sinh(\nu H) & \cosh(\nu H)
\end{pmatrix}$$

where:
- $\nu^2 = k^2(1 - c^2/V_p^2)$ (vertical wavenumber squared)
- $H$ = layer thickness
- $\rho$ = density
- $V_p = \sqrt{c_{33}/\rho}$ = P-wave velocity

### Solid-Liquid Transition

At the solid-liquid interface, the 5D minor vector $\mathbf{Y}$ is reduced to the 2D state:

$$Z_1 = Y(I_{14}), \quad Z_2 = Y(I_{24})$$

At the liquid free surface, the secular condition is $P = 0$.

Here $P=T_{zz}$ is the Lagrangian normal traction, not the Eulerian pressure
perturbation. The standard seabed conditions retain continuity of normal
displacement and Lagrangian normal traction, with zero shear traction on the
solid side. No separate $\Delta\rho\,g\,U$ jump is added. When fluid Cowling
gravity is enabled, it modifies fluid propagation; the solid propagation
equations remain non-gravitating.

### Rayleigh Eigenfunctions

`compute_rayleigh_eigenfunctions` supports an ocean stack at the top of a solid
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

Here $T_{zz}$ is the Lagrangian traction, so the Eulerian pressure perturbation
is $p' = \rho g_{\rm internal} U - T_{zz}$ and horizontal momentum balance
$\omega^2 \rho\, u_r = k\, p'$ gives the sign above.

The returned dictionary also includes `is_fluid`, a boolean array marking
samples inside the fluid stack. At the seabed, the solid-side state is recovered
from the existing minor-vector interface relation; duplicated interface depths
may appear so that fluid-side and solid-side states remain distinct.

### Rayleigh Variational Gradients

The trapezoidal Rayleigh weak-form path supports top fluid stacks. Fluid layers
use the acoustic frozen-displacement kernels

$$
K_\rho = -\omega^2\int (u_z^2+u_r^2)\,dz,\qquad
K_C = \int \frac{T_{zz}^2}{C^2}\,dz ,
$$

with no sensitivity to shear moduli. In isotropic output, the fluid `Vs`
gradient is zero and the `Vp`/`Rho` gradients use
$C=\rho V_p^2$. Analytical Rayleigh weak-form integrals are still solid-only;
use `useAnalytical=false` for fluid models.

### Grid/Romberg Rayleigh Weak-Form Hessian

`compute_phase_velocity_hessian_weakform_rayleigh_grid` supports a top fluid
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

```cpp
#include "disper26.cpp"

// Create fluid layer parameters
template <typename T>
PhysicalParameters<T> createFluidParameters(T vp, T rho) {
    PhysicalParameters<T> p;
    p.rho = rho;
    p.c44 = rho * T(0.0);  // No shear
    p.c11 = rho * vp * vp;
    p.c33 = p.c11;
    p.c13 = p.c11;
    p.c66 = rho * T(0.0);
    return p;
}

// Build model: Ocean -> Sediment -> Basement
std::vector<PhysicalParameters<double>> layers;
std::vector<double> thicknesses;

// Ocean (liquid)
layers.push_back(createFluidParameters(1.5, 1.0));  // Vp=1.5 km/s
thicknesses.push_back(2.0);  // 2 km

// Sediment (solid)
layers.push_back(createIsotropicParameters(0.5, 2.0, 1.8));
thicknesses.push_back(1.0);

// Half-space
PhysicalParameters<double> hs = createIsotropicParameters(3.0, 5.0, 2.5);

// Calculate dispersion
double omega = 2.0 * M_PI * 0.5;  // 0.5 Hz
double c = 1.5;  // km/s
auto result = calculateRayleighDispersion(layers, thicknesses, hs, omega, c);
```

### Python API

```python
import disper26py as d26
import math

# Create fluid layer
ocean = d26.create_fluid_parameters(vp=1.5, rho=1.0)

# Create solid layer
sediment = d26.create_isotropic_parameters(vs=0.5, vp=2.0, rho=1.8)
halfspace = d26.create_isotropic_parameters(vs=3.0, vp=5.0, rho=2.5)

layers = [ocean, sediment]
thicknesses = [2.0, 1.0]  # km

omega = 2.0 * math.pi * 0.5
roots = d26.search_roots(layers, thicknesses, halfspace, omega, 0.3, 3.0, 100)
```

## Multi-Layer Ocean (SOFAR Channel)

For depth-varying sound speed profiles:

```cpp
// SOFAR-like profile
layers.push_back(createFluidParameters(1.52, 1.025));  // Surface
thicknesses.push_back(0.2);

layers.push_back(createFluidParameters(1.48, 1.025));  // Thermocline
thicknesses.push_back(0.5);

layers.push_back(createFluidParameters(1.46, 1.025));  // SOFAR axis
thicknesses.push_back(0.5);

layers.push_back(createFluidParameters(1.50, 1.025));  // Deep
thicknesses.push_back(1.0);
```

## Known Limitations

| Issue | Description |
|-------|-------------|
| c = Vp singularity | Near-singularity handling via fallback (not exact limit) |
| Liquid half-space | Not supported (ocean must be on top of solid) |
| Liquid→Solid transition | Ocean below solid not supported |
| Analytical Rayleigh weak-form integrals | Fluid layers are not supported; use trapezoidal integrals |
| Grid/Romberg Rayleigh weak-form Hessian | Fluid nodes are treated as fixed; use a duplicate-depth seabed discontinuity |

## References

- Ikeda & Matsuoka (2013): Formulation of layer matrix
- Woodhouse (1988): Phase definition for mode counting
