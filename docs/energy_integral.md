# Energy integrals

The integrals use positive depth increments from the surface through the
decaying half-space. Here $dz$ denotes that positive integration measure. The displacement and traction components use an upward
vertical axis. For Rayleigh waves the reduced state is
`[y1, y2, y3, y4] = [uz, szz, ur, szr]`. The horizontal displacement amplitude
is $i y_3$ for the phase $e^{i(kx-\omega t)}$. For Love waves $y_1=W$ and
$y_2=T$ as defined below. Here $k=\omega/c$ (rad/km), $\omega$ is in rad/s,
density $\rho$ is in g/cm³, and $(A,C,F,L,N)=(c_{11},c_{33},c_{13},c_{44},c_{66})$
are VTI stiffnesses in GPa.

## Rayleigh wave in a solid
- $I_1 = \int \rho (y_1^2+y_3^2) dz$
- $I_2 = \int [y_2^2/C+y_4^2/L+k^2(A-F^2/C)y_3^2]dz$
- $I_3 = \int [(A-F^2/C)y_3^2+(y_1*y_4-F/C*y_2*y_3)/k]dz$

For a fluid layer the $I_2$ and $I_3$ integrands reduce to $y_2^2/C$ and $-y_2 y_3/k$, respectively.

## Love wave
- $I_1 = \int \rho y_1^2 dz$
- $I_2 = \int [y_2^2/L+ k^2*N*y_1^2]dz$
- $I_3 = \int N y_1^2 dz$

## Group velocity
- $U = I_3/ (c\cdot I_1)$, where $c$ is phase velocity.

## Value for accuracy check
- $\omega^2 I_1 - I_2 = 0$

## Love traction convention and verification

SH propagation uses $z$ positive upward and the state
$[W,T]=[u_y,L\,du_y/dz]$. The decaying halfspace starts from $[1,+L\nu]$;
upward propagation over thickness $h$ uses $A(+h)$ and downward propagation
uses $A(-h)$. Stored depths remain positive downward, $d=-z$.
The public Love eigenfunction field `tyz=T=-L*du_y/dd` keeps its existing
physical sign and surface normalization. It is emitted directly from the
upward-stress state.

Analytical modal integrals still integrate a local downward depth $x$. Their
decomposition therefore uses $\tau_x=L\,du_y/dx=-T$ and must retain the
conversion `tau_depth = -tyz` before forming modal amplitudes. The regular
moments take upward `T` and convert to the downward slope `-T/L` internally.
Trapezoidal integration squares the traction and needs no reflection.

The older DISPER80 fallback energy paths retain their pre-existing moment
values for compatibility in this coordinate-only change. Their evanescent
bottom-state moment historically used reflected traction, whereas the usual
upward analytical path uses physical traction. That fallback discrepancy is
not a correction made by this refactor; the conversion is explicit at its
moment boundary and its propagated state now uses upward `T` throughout.

The missing conversion was corrected in `computeLoveEnergyIntegrals_Analytical`
on 2026-09-25, affecting both `disper26.compute_energy_integrals` with analytical
integration (including solid-model `auto`) and
`disper26py.compute_energy_integrals_analytical_love`. A single finite layer
does not expose this error because its top is a traction-free surface;
multilayer interfaces generally have nonzero traction.

The regression `tests/python/test_love_energy_integrals.py` covers isotropic
and VTI multilayer fundamental and first-overtone modes, and the accepted
GRID mesh. Analytical group velocity and energy balance use a relative
tolerance of `1e-10` for root and
state-propagation roundoff. Independent frequency differences use `1e-6`,
matching the fluid-group-velocity regression's finite-difference allowance.
The 200-sublayer trapezoidal comparison uses `2e-6` for its second-order
quadrature error and also checks improvement from 100 sublayers.

For the isotropic three-layer example at 20 s, the erroneous analytical
group velocity was `3.58336046113 km/s`; AD and frequency differences give
`2.94184492337 km/s` (a 21.8% discrepancy before the correction).
After correction, the four isotropic/VTI mode cases agree with AD within
`2.3e-16` relative, with energy-balance residuals below `4.5e-16` relative.

## Critical finite-layer speeds (historical study, 2026-09-25)

`fix/regular-energy-integrals-20260925` shares regular finite-layer moments
between the public Rayleigh integrals and variational derivatives/Hessians.
Love also uses regular moments near zero vertical wavenumber, where the public
modal formula previously returned NaN. Halfspace decay integrals are unchanged.
The four diagnosed near-Vs roots now agree with the independent group-velocity
reference within `1.33e-15` relative; previously the error reached `4.43e-6`.
These are historical development-branch measurements, not a statement about
the current public Git history. The fixed-state checks and derivative controls
available in this repository are in
[the regular-integral regression](../tests/python/test_regular_energy_integrals.py)
and [the independent group-velocity reference](../tests/python/test_group_velocity_accuracy_reference.py).
Detailed historical timing reports are kept in the research archive.

## Rayleigh gravity post-correction

The inverse-model helper uses a first-order correction evaluated from the
non-gravitating solution. With flat depth positive downward and positive
gravity magnitude,

$$
\delta c = \int K_g(z) g(z)\,dz,\qquad
K_g = \frac{c^2}{\omega^2 U I_1}\,\rho k y_1 y_3.
$$

The returned correction estimates $c(g)-c(g=0)$. Subtract it from observed
phase velocities before comparing them with the ordinary non-gravitating
forward solver. The corresponding group correction is

$$
\delta U = \frac{U^2}{c^2}
\left[\left(\frac{2c}{U}-1\right)\delta c
+\omega\frac{d\delta c}{d\omega}\right].
$$

Subtract `delta_u` from the observed group velocity before comparing it with
the non-gravitating solution. For the frequency derivative, the helper samples
two neighboring angular frequencies at a default relative offset of 0.005.
The correction is applied to observations and does not enter the forward
Jacobian or Hessian. Love waves have zero gravity correction here. The shared
`compute_gravity_correction` function accepts only Rayleigh solutions.
These conventions are implemented in
[the gravity helper](../inversion/disper26_inv/gravity.py) and checked against
the fluid Cowling solver in
[the gravity regression](../tests/python/test_gravity_kg_correction.py).
Historical stationarity and Mineos comparison reports are retained in the
research archive; they are not a requirement for running the public examples.
