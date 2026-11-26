# Diagnostics with the low-level Python API

[Japanese edition](../docs_jp/lowlevel_api.md)

For routine dispersion, sensitivities and Hessians, use the high-level API
(`disper26`) with a model or a stored solution. See the [practice examples](../examples/README.md).

This guide uses the low-level API (`disper26py`), imported as `d26_lowlevel`,
to inspect a mode count at a trial phase velocity, compare methods or supply
roots for individual refinement levels. Its functions take material properties,
thickness or depth arrays and the frequency or period for the calculation.
Use `*_layer_cake` in new code. The corresponding `*_pancake` aliases retain
their arguments and defaults.

Run the Python blocks in sections 1–8 in order in a fresh session.
`alpha_v` ($V_{PV}$) and `alpha_h` ($V_{PH}$) are the P-wave speeds for
vertical and horizontal propagation, respectively. For horizontally propagating
S waves in a VTI solid, `beta_v` ($V_{SV}=\sqrt{c_{44}/\rho}$) is the speed
for vertical polarization and `beta_h` ($V_{SH}=\sqrt{c_{66}/\rho}$) is the
speed for horizontal polarization. Model arrays, including
`depth`, `rho` and `eta`, are indexed by layer or node. Depth and thickness
are in km, velocity in km/s, density in g/cm³ and stiffness in GPa.

| Operation | Period or frequency argument |
|---|---|
| Layer-cake root search, single-period values, kernels and Hessians | `omega` [rad/s] |
| Layer-cake and GRID `compute_dispersion_curve_*` | `periods` [s] |
| `compute_dispersion_romberg*`, `compute_kernels_ad_grid*`, `compute_frequency_curvature_ad_grid`, GRID interface derivatives | `period` [s] |
| `compute_kernels_variational_grid`, `compute_hessian_ad_grid`, `compute_hessian_variational_ad_*_grid*` | `freq` [Hz] |

Use `omega = 2π / period` and `freq = 1 / period`.

1. [Prepare the inputs](#lowlevel-model)
2. [Roots and mode counts](#lowlevel-roots)
3. [Batch dispersion curves](#lowlevel-curves)
4. [Group velocity, H/V, eigenfunctions and integrals](#lowlevel-observables)
5. [Sensitivity kernels](#lowlevel-kernels)
6. [Phase-velocity Hessians](#lowlevel-hessian)
7. [Interface-depth derivatives](#lowlevel-interface)
8. [Earth flattening and Q correction](#lowlevel-eft-q)

<a id="lowlevel-model"></a>

## 1. Prepare the inputs

```python
import math
import numpy as np
import disper26py as d26_lowlevel

# Isotropic material: create_isotropic_parameters(vs, vp, rho)
layer_solid = d26_lowlevel.create_isotropic_parameters(3.2, 5.8, 2.6)
half_space = d26_lowlevel.create_isotropic_parameters(4.5, 8.0, 3.3)

# Fluid layer: create_fluid_parameters(vp, rho)
ocean_water = d26_lowlevel.create_fluid_parameters(1.5, 1.0)

# Arguments: A, C, F, L, N (GPa), density (g/cm³), top and bottom depth (km).
vti_layer, thick = d26_lowlevel.create_vti_layer_from_depth(
    20.0, 20.0, 8.0, 4.0, 4.0, 2.5, 0.0, 10.0
)

# Only the finite layers go into the layer and thickness lists.
layers = [layer_solid]
thicknesses = [15.0]

# GRID inputs in the low-level representation
depth = [0.0, 10.0, 30.0, 30.0, 100.0]
rho = [2.5, 2.7, 2.9, 3.3, 3.4]
alpha_v = [5.5, 6.0, 6.5, 8.0, 8.1]
beta_v = [3.2, 3.5, 3.8, 4.5, 4.6]
is_disc = [False, False, True, False, False]  # Shallow side at 30 km
grid_params = [d26_lowlevel.create_isotropic_parameters(b, a, r)
               for b, a, r in zip(beta_v, alpha_v, rho)]
# In this isotropic example, VPH = VPV and VSH = VSV.
alpha_h = list(alpha_v)
beta_h = list(beta_v)
eta = [1.0] * len(depth)
```

<a id="lowlevel-roots"></a>

## 2. Roots and mode counts

```python
period_layer_cake = 5.0  # s
omega = 2.0 * math.pi / period_layer_cake

# The last four positional arguments are c_min, c_max, n_search and wave_type.
roots_rayleigh = d26_lowlevel.search_roots(
    layers, thicknesses, half_space, omega, 2.0, 4.4, 200, "Rayleigh"
)
roots_love = d26_lowlevel.search_roots(
    layers, thicknesses, half_space, omega, 3.2, 4.4, 200, "Love"
)
if not roots_rayleigh or not roots_love:
    raise RuntimeError("No roots found. Check the model and velocity range.")
c_rayleigh = roots_rayleigh[0]
c_love = roots_love[0]

# Check the count jump across the fundamental root of this simple model.
delta_rel = 3e-4  # Relative half-width used in the branch check
for wave, cp in (("Rayleigh", c_rayleigh), ("Love", c_love)):
    counts = [d26_lowlevel.compute_mode_count(
        layers, thicknesses, half_space, omega, cp * (1.0 + sign * delta_rel), wave
    ) for sign in (-1, 1)]
    assert counts == [0, 1], (wave, counts)

# The legacy continuous count is a diagnostic. Its integer part is unreliable as a mode index.
cont_mode = d26_lowlevel.compute_continuous_mode_count(
    layers, thicknesses, half_space, omega, c_rayleigh, "Rayleigh"
)
```

Rounding can affect a count evaluated exactly at the root. Adjust the window
and check the branch when nearby roots could fall in the same interval.
The sign-change scan in `search_roots` can also miss roots. For branch-specific
GRID calculations use `branch` in section 5 or the
[ForwardEngine example](../examples/README.md#advanced). See
[mode counting](mode_counting.md) for diagnostics.

<a id="lowlevel-curves"></a>

## 3. Batch dispersion curves

These `disper26py` batch functions share numerical kernels with the
[shared API examples](../examples/README.md) and have their own search and
prediction controls. The shared API's `predictor` and `verify_mode` options
are unavailable here. Supply positive, finite, strictly increasing or
decreasing periods. The C++ batch releases the GIL, allowing independent
models to run in Python threads. Tracking within each curve is sequential.

```python
periods = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

# Batch calculation for the layer-cake model
layer_cake_curve = d26_lowlevel.compute_dispersion_curve_layer_cake(
    layers, thicknesses, half_space,
    periods=periods, wave_type="Rayleigh", branch=0,
    c_min=2.0, c_max=4.4, verify_branch=True
)
c_layer_cake = layer_cake_curve["phase_velocities"]

# Velocity-interpolated GRID using the arrays from section 1
grid_curve = d26_lowlevel.compute_dispersion_curve_romberg_grid_velocity_vti(
    depth, rho, alpha_h, alpha_v, beta_h, beta_v, eta,
    is_disc, periods, wave_type="Rayleigh", c_guess=3.5, branch=0
)
c_grid_velocity = grid_curve["phase_velocities"]
print("GRID fallback count:", grid_curve["fallback_count"])
```

For layer-cake models, `branch=0` selects the lowest-velocity root at
initialization and recovery. `verify_branch=True` also checks the integer
count of predicted roots. For GRID, `branch=0` specifies the mode order at
each level. The default `branch=None` follows the nearby root, with different
runtime characteristics.

The result also contains `relative_errors`, `converged_levels` and
`fallback_mask`. A fallback means that the solver returned to its regular
search after prediction. Layer-cake results use level `-1` and relative
error `NaN`. See the [batch and single-period regression](../tests/python/test_phase_velocity_curves.py).

<a id="lowlevel-observables"></a>

## 4. Group velocity, H/V, eigenfunctions and energy integrals

```python
# Group velocity from full AD
U = d26_lowlevel.compute_group_velocity_ad(
    layers, thicknesses, half_space, omega, c_rayleigh, "Rayleigh"
)

# Signed ellipticity y3/y1, evaluated on the solid side of a fluid-loaded seabed
hv = d26_lowlevel.compute_hv_ratio(
    layers, thicknesses, half_space, omega, c_rayleigh, "Rayleigh"
)

# Rayleigh fields: ur, uz, szr, szz. Love fields: uy, tyz.
# Pass the subdivision count as a positional argument.
ef_rayleigh = d26_lowlevel.compute_eigenfunctions_rayleigh(
    layers, thicknesses, half_space, omega, c_rayleigh, 50
)

# Analytical energy integrals for a solid model
I1, I2, I3, c_energy = d26_lowlevel.compute_energy_integrals_analytical_rayleigh(
    ef_rayleigh, layers, thicknesses, half_space, omega, c_rayleigh
)
```

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
Zero shear traction gives $\epsilon=-Y_{12}/Y_{14}$ wherever $Y_{14}\ne 0$.
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
and sensitivities are undefined (NaN), not clipped or replaced by zero.
The existing Love-wave and unrequested-output availability rules still apply.

<a id="lowlevel-kernels"></a>

## 5. First-order sensitivity kernels for c, U and H/V

In a VTI GRID, `compute_kernels_variational_grid` and the variational Hessian
use `(Rho,A,C,F,L,N)` within each node. The AD kernels from
`compute_kernels_ad_grid` and the material part of an interface Hessian row
use `(A,C,F,L,N,Rho)`. Reorder each six-component block before combining an
interface row with the variational Hessian. Keep trailing depth entries in place.

### Layer-cake model

Full AD gives phase-velocity, group-velocity and H/V sensitivities.
Love H/V is undefined and its value and sensitivity are `NaN`.

```python
kernels = d26_lowlevel.compute_kernels_ad_layer_cake(
    layers, thicknesses, half_space, omega, c_rayleigh,
    wave_type="Rayleigh",
    sensitivity_mode="Isotropic",  # Or "VTI"
    compute_thickness=False        # Set True for thickness sensitivities.
)

for k in kernels:
    print(f"Layer {k.layer_index} [{k.parameter_name}]: dc/dp={k.d_c:.6e}, dU/dp={k.d_U:.6e}, d(HV)/dp={k.d_HV:.6e}")
```

`layer_index` is zero-based for finite layers and `-1` for the half-space.
The material order in this API is `(Vs,Vp,Rho)` for `Isotropic` and
`(Rho,A,C,F,L,N)` for `VTI`. Each derivative has the units of the observable
divided by the units of its parameter.

### GRID model with Romberg extrapolation

Sections 5–7 use linear interpolation of **stiffness**. Recompute the 10 s
Rayleigh fundamental root for this model. Passing its level roots to each
derivative calculation keeps the kernels and Hessian on the same branch.

```python
period_grid = 10.0
derivative_level = 5
trace = d26_lowlevel.compute_dispersion_romberg_levels(
    depth, grid_params, is_disc, period=period_grid,
    wave_type="Rayleigh", c_guess=3.5, branch=0,
    max_steps=derivative_level, minimum_trace_level=derivative_level,
)
grid_roots = trace["phase_velocity_roots"]
accepted_level = trace["phase_accepted_level"]

c_grid, U_grid, hv_grid, level, rel_err, kernels_grid = d26_lowlevel.compute_kernels_ad_grid(
    depth, grid_params, is_disc,
    period=period_grid, wave_type="Rayleigh", c_guess=trace["phase_velocity"],
    sensitivity_mode="Isotropic", max_steps=derivative_level,
    phase_velocity_roots=grid_roots, phase_accepted_level=accepted_level,
)
print("GRID c, level, rel_err:", c_grid, level, rel_err)
```

For GRID, `layer_index` identifies a node. Results include discontinuity
depth (`Depth`) as well as material sensitivities. The default `dU/dp` method
is `variational_frequency_ad`. Unrequested sensitivities are filled with
zeros. Consult the requested quantities when interpreting a zero. To check
an interface-depth derivative by finite differences, move both coincident
nodes together, as in the [Love group-velocity regression](../tests/python/test_u_kernel_frequency_variational_grid.py).

`max_steps` and `max_levels` cap refinement. The interface routines in
Section 7 call the upper level `max_level`. Inspect the returned `rel_err`
and accepted level to assess accuracy. The five-level example demonstrates
the calls and can reach the cap before all requested quantities converge.

<a id="lowlevel-hessian"></a>

## 6. Phase-velocity Hessians (d²c/dp²)

Layer-cake Hessians have material coordinates only. The legacy
`compute_thickness` argument adds no rows or columns. First-order kernels
support thickness derivatives through `compute_thickness=True`.

The primary method applies first-order AD to a variational gradient. Runtime
and accuracy depend on the model, quadrature and backend. See the
[implementation map](sensitivity_implementation_map.md) for the comparison
conditions. Full AD via `compute_hessian_ad_layer_cake` provides an independent
check.

### Layer-cake model

```python
# Rayleigh waves
res_rayleigh = d26_lowlevel.compute_hessian_variational_ad_rayleigh_layer_cake(
    layers, thicknesses, half_space, omega, c_rayleigh,
    sensitivity_mode="Isotropic",
    use_analytical=True  # Closed-form integrals, including contiguous surface fluid layers over solid.
)
H = res_rayleigh["hessian"]   # 2-D NumPy array
g = res_rayleigh["gradient"]  # 1-D NumPy array

# Love waves
res_love = d26_lowlevel.compute_hessian_variational_ad_love_layer_cake(
    layers, thicknesses, half_space, omega, c_love,
    sensitivity_mode="Isotropic", use_analytical=True,
)
```

### GRID node parameters

```python
res_grid = d26_lowlevel.compute_hessian_variational_ad_rayleigh_grid(
    depth, grid_params, is_disc,
    freq=1.0 / period_grid, c_guess=c_grid,
    sensitivity_mode="Isotropic", max_levels=derivative_level,
    phase_velocity_roots=grid_roots, phase_accepted_level=accepted_level,
)
H_grid = res_grid["hessian"]
parameter_labels = list(zip(res_grid["layer_indices"], res_grid["parameter_names"]))
print("GRID Hessian:", H_grid.shape, parameter_labels)
```

The material Hessian is 15×15 for these five isotropic nodes. Section 7
treats interface depth separately. Use `parameter_names` and `layer_indices`
to identify columns in each API.

For contiguous surface fluid layers over solid, the Rayleigh layer-cake weak-form
Hessian supports analytical integration. Its low-level default, `use_analytical=False`,
selects trapezoidal integration. The GRID
Hessian includes the fluid in the wavefield and treats **fluid nodes as
known and fixed**. Their gradient and Hessian entries are zero. Represent
the seabed as a repeated-depth discontinuity.

<a id="lowlevel-interface"></a>

## 7. Interface-depth derivatives

Layer-cake $K_d$ is a kernel for each boundary. For a geometry parameter $q$,
$dc/dq=\sum_b [K_d]_b\,dz_b/dq$. Changing layer thickness can move several
boundaries. Moving a single interface contributes only its own term. The
following GRID example uses flat depths and the variational boundary kernel.

```python
# Layer-cake interface kernels Kd
kd_rayleigh = d26_lowlevel.compute_interface_kernels_variational_rayleigh_layer_cake(
    layers, thicknesses, half_space, omega, c_rayleigh
)

# Discontinuity 0 is at 30 km. Its shallow-side node has index 2.
discontinuity_index = 0
grad_z = d26_lowlevel.compute_interface_gradient_variational_grid(
    depth, grid_params, is_disc,
    period=period_grid, wave_type="Rayleigh", c_root=c_grid,
    discontinuity_index=discontinuity_index, max_level=derivative_level,
    phase_velocity_roots=grid_roots, phase_accepted_level=accepted_level,
)
row_z = d26_lowlevel.compute_interface_hessian_row_variational_ad_grid_extrapolated(
    depth, grid_params, is_disc,
    period=period_grid, wave_type="Rayleigh", c_root=c_grid,
    discontinuity_index=discontinuity_index, max_level=derivative_level,
    sensitivity_mode="Isotropic",
    phase_velocity_roots=grid_roots, phase_accepted_level=accepted_level,
)
```

Here `row_z` has 16 elements. The first 15 are material components ordered
`(Vs,Vp,Rho)` at each node. The final component for the discontinuity is
$d^2c/dz^2$.

<a id="lowlevel-eft-q"></a>

## 8. Earth flattening and Mineos Q correction

These independent examples interpret the same input values as a spherical
model. The flat-model solutions from the preceding sections remain available.

```python
r_earth = 6371.0

# Flatten the layer-cake model (Bhattacharya & Arora 1997, default p=-2).
flat_layers, flat_thick = d26_lowlevel.earth_flattening_transformation_layers(
    layers, thicknesses, planet_radius=r_earth
)

# Flatten the GRID model.
flat_depths, flat_params, flat_disc = d26_lowlevel.earth_flattening_transformation_grid(
    depth, grid_params, is_disc, planet_radius=r_earth
)

# Correct the reference-period model for a target period of 20 s.
corrected_layers = d26_lowlevel.apply_mineos_physical_dispersion_model(
    layers, qkappas=[57823.0], qmus=[600.0], tref_s=1.0, period_s=20.0
)
```

The layer-cake flattening function transforms only the finite layers passed
to it. The half-space requires separate treatment to complete the flat model.
The GRID transformation includes the deepest node.

The GRID transformation above produces flat nodes for `grid_policy="flat"`.
To interpolate in physical depth, pass the original nodes to the forward and
derivative calls with the same `grid_policy="spherical_ba97"` or
`"spherical_experimental_density"`. See [choosing depth coordinates for a stiffness GRID](public_api.md#grid-depth-coordinates).

`compute_kernels_ad_grid` includes one `Depth` entry per interface. In an
all-solid spherical stiffness GRID it carries the physical-depth derivatives
of the requested phase velocity, group velocity and Rayleigh H/V.
Pass `include_interface_depths=False` for material entries alone.
Spherical interface derivatives in models containing fluid are unsupported.
The fixed-level `compute_interface_hessian_row_ad_grid` accepts the same
`grid_policy`, `planet_radius` and `flattening_exponent` as the forward
calculation and takes that level's `dc_dz` as its outer root seed.

Q correction also applies only to the supplied list. Apply
`apply_mineos_physical_dispersion` separately to the half-space, and match
the Q array length to the material list. Start from the original reference
model at each target period. The correction uses $Q_\kappa$ and $Q_\mu$
to adjust the real elastic moduli for anelastic dispersion. Solve the real
dispersion equation again with those moduli at the target period.
