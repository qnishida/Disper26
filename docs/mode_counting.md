# Mode Counting in Disper26

Disper26 now uses an AD-z zero-crossing counter as the primary public integer
mode counter.  The method is closer in spirit to Mineos `bran/zknt` than the
older Woodhouse-phase `atan2` counter: it follows the secular quantity through
the layers, detects its zero crossings, and uses the local derivative to assign
the crossing direction.

The older phase-accumulation counter is still available as a legacy diagnostic.

## Public API

### Primary Counter

In Python, `compute_mode_count(...)` is the integer mode label
(`docs/naming_convention.md` #4/#5/#9; `calculate_mode_count` is the same
function kept callable as a deprecated alias):

```python
import disper26py

n = disper26py.compute_mode_count(
    layers, thicknesses, half_space, omega, c, "Rayleigh")
```

`calculate_total_mode_count(...)` and `calculate_bran_mode_count(...)` are the
same function under older names, kept so existing scripts keep working.  There
has never been a separate "bran" counter: both reach `calculateAdModeCount`
with the same arguments.  Use `compute_mode_count` in new code.

### Checked curve tracking (2026-09-26)

The common API's checked Rayleigh curve path first tries a secular root near
the period/coarse-level prediction. It accepts that candidate only when:

- the root is finite, within the local search window, and satisfies the existing
  secular-residual limit;
- a narrow interval enclosing it has finite secular values with a sign change;
- the integer counts at the interval endpoints are exactly `mode` and `mode+1`.

The certificate interval uses the existing count-bisection width limit; root
and residual tolerances are unchanged. Every generated GRID level is checked
before its value enters Romberg extrapolation. The initial unpredicted root,
failed candidate searches, rejected certificates, and invalid local counts use
the original mode-order search. Failed certificates are recorded as
`local_mode_certificate` in fallback diagnostics. Unchecked continuation keeps
its existing semantics. Checked Love keeps the original count search because
candidate-first searching was slower with its cheaper counter.

See implementation and measurements. A
certificate relies on the existing integer counter and does not extend its
supported model domain or remove its near-degeneracy limitations.

### Legacy continuous diagnostic

The old phase-accumulation implementation survives only as a *continuous*
diagnostic.  Its integer wrapper, `calculate_legacy_total_mode_count(...)`, has
been removed: the AD-z counter is the integer label, and for Love the legacy
integer was wrong by more than an offset (see below).

```python
n_cont = disper26py.compute_continuous_mode_count(
    layers, thicknesses, half_space, omega, c, "Rayleigh")

legacy_debug = disper26py.debug_continuous_mode_count(
    layers, thicknesses, half_space, omega, c, "Rayleigh")
```

`calculate_continuous_mode_count(...)` and `debug_legacy_mode_count(...)` are
the same two functions under their older names, kept callable as deprecated
aliases; so is `calculate_legacy_continuous_mode_count(...)`, a name that says
where it comes from.  Note that the continuous count *is* legacy machinery
despite the plain name -- the AD-z counter has no continuous variant, so this
is the available phase-based diagnostic; it is not globally smooth.  `test_low_c.py` and
`test_safe_range.py` use it to probe the edges of the valid `c` range.

An unrelated, since-removed `debug_mode_count(...)` alias for
`debug_legacy_mode_count(...)` once existed and was removed because its name
suggested it was the primary debug entry point when it was the legacy one.
The name `debug_mode_count` has since been reintroduced (2026-08-11) for a
different function -- see "Crossing Debug Output" below -- so do not confuse
the two if reading old history.

### Crossing Debug Output

Use `debug_mode_count(...)` to inspect the counter
(`debug_mode_count_crossings(...)` and `debug_bran_mode_count(...)` are the
same function under older names, kept callable as deprecated aliases):

```python
debug = disper26py.debug_mode_count(
    layers, thicknesses, half_space, omega, c, "Rayleigh")
```

Important fields are:

- `n_integer`: final AD-z integer mode label.
- `signed_count`: raw signed crossing sum.
- `initial_signed_count`: analytical halfspace contribution (Rayleigh: 0 or
  -1; Love: 0), separate from the depth crossings.
- `calibrated_signed_count`: count after adding the initial contribution and
  applying endpoint/regime calibration.
- `crossings`: per-crossing diagnostics.
- `legacy_continuous_count`: legacy continuous phase count at the same point.
- `status`: `ok`, `unsupported_liquid`, `unsupported_love`, or
  `nan_or_radiative`.

Each crossing contains `layer_index`, `local_depth_km`, `contribution`,
`secular`, `secular_derivative`, `conjugate`, and `test_value`.
`local_depth_km` is the cumulative height above the halfspace top, a single
global coordinate shared by all layers so that crossings can be deduplicated
at layer interfaces. The analytical halfspace contribution is not a depth
crossing and does not appear in `crossings`.

## AD-z Crossing Algorithm

The counter propagates from the halfspace upward through the model.  In each
layer, it evaluates the relevant secular variable and its derivative with
respect to the local layer coordinate.

Within each substep, the code builds a cubic Hermite interpolant for the
secular variable using endpoint values and endpoint derivatives.  It then finds
zero crossings of that interpolant.  The crossing direction is determined from
the sign of

```text
test_value = conjugate * d(secular)/dz
```

The substep size is chosen adaptively from the estimated vertical phase or
decay scale, with extra subdivision near small vertical wavenumber.

### How the endpoint derivative is obtained

The name is historical: the derivative used to come from seeding `wave.z` with
`Eigen::AutoDiffScalar` at both ends of every substep.  It no longer does.  A
single mode-count call is about 9x cheaper for Rayleigh and about 300x cheaper
for Love, which makes the counter usable as a routine guard on dispersion
calculations: a Love count now costs about what one secular-function evaluation
costs on the same model, and a Rayleigh count about 80 of them.  The counting
convention is unchanged throughout.

Several properties of a homogeneous layer are used.

*The bottom endpoint of a substep is the top endpoint of the previous one.*
The value is already carried in the propagated state, and since the layer
propagator is `B(z) = exp(A z)`, its derivative satisfies `B'(z) = A B(z)`, so
the derivative at the top of a substep is the derivative at the bottom of the
next.  Only the first substep of a layer evaluates the bottom explicitly.  When
the overflow guard renormalises the state, the carried derivative is rescaled
with it, the propagation being linear in the state.

*The SH propagator can be differentiated in closed form.*  With
`A(z) = [[C, S/eta], [eta S, C]]`, `C = cosh(nu z)`, `S = sinh(nu z)` and
`eta = c44 nu`, the Love counter uses

```text
dA/dz = [[nu S, C/c44], [c44 nu^2 C, nu S]]
```

and propagates upward with `z = s`, so `d/ds = d/dz`. The SH state is
`[W,T]=[u_y,c44*du_y/dz]` with `z` positive upward. Both derivative
entries and the entire-function value stay regular as `nu -> 0`, including
`A01 -> z/c44`. `nu` and `eta` are computed once per
layer.  This is used where a Love endpoint derivative is still needed, which
after the next point is only at the crossings themselves.

### Love layers are counted in closed form

The Love counter does not use a substep mesh at all.  In a homogeneous SH layer
`dW/ds = T/c44` and `dT/ds = -c44 r^2 W` with
`r^2 = (rho omega^2 - c66 k^2)/c44 = -nu^2`, so `T'' = -r^2 T`: the secular
variable is a single harmonic, and its zeros are known.

```text
r^2 > 0   T(s) = T0 cos(r s) - c44 r W0 sin(r s)
          = R cos(r s - phi),  R = hypot(T0, c44 r W0)
                              phi = atan2(-c44 r W0, T0)
          zeros at s = (phi + pi/2 + m pi) / r

r^2 < 0   T(s) = T0 cosh(kappa s) + c44 kappa W0 sinh(kappa s)
          one zero at s = atanh(-T0 / (c44 kappa W0)) / kappa

r^2 = 0   T is constant; no isolated interior traction zero
```

The oscillatory count is an interval of integers `m`; the evanescent case has
at most one zero because `tanh` is monotone.  The orientation is not evaluated
either: at a zero of `T`,

```text
test_value = W dT/ds = -c44 r^2 W^2
Love contribution = sign(test_value)
```

so every oscillatory crossing contributes `-1` and every evanescent one `+1`.
This preserves the Love signed sequence. Rayleigh and liquid crossings use
`-sign(test_value)` in their own state conventions; that rule must not be
applied to the upward SH state. `W` cannot vanish where `T` does except for
the trivial solution.

A layer therefore costs `O(1)` plus one propagator evaluation per crossing
actually found, instead of the 50 evaluations per radian of vertical phase that
the mesh needed.  Substeps survive only as range control in strongly evanescent
layers, where a piece is capped at 20 decay lengths so `cosh` cannot overflow
the rescaling guard; that has nothing to do with resolving crossings.

Zeros are enumerated on the half-open interval `(0, d]`, so a zero on an
interface belongs to the layer below.  The window is opened by the dedup
tolerance at both ends and the dedup discards the duplicate -- testing it
strictly lets rounding push an interface zero just outside both neighbours and
lose it.

The same closed form is not available for Rayleigh: the secular minor combines
two vertical wavenumbers, `nu1` and `nu3`, so it is quasi-periodic rather than
harmonic and its zeros have no closed form.  Rayleigh keeps the substep mesh.

*The Rayleigh generator is z-independent.*  `A` is read off once per layer as
the z derivative of `B` at `z = 0`, which is the one place automatic
differentiation is still used, and each substep endpoint then needs only the
plain `double` propagator and the product `A B(z)`.  Taking `A` from the same
`computeLayerMatrix` call keeps whichever propagator branch the layer selects
self-consistent, including the small-`nu1` regular form.  The derivative matrix
is formed before contracting with the state rather than applying `A` to the
propagated vector, so the near-zero secular component stays a result of the
final cancellation instead of an input the generator would amplify.  The liquid
`[U, P]` propagator is treated the same way.

### What this does and does not preserve

Labels are preserved.  On a randomised sweep of 400 models -- monotone stacks,
low-velocity zones, thin stacks, ocean models and thick single layers -- at
four frequencies, 40208 mode-count evaluations and 312670 crossings, the
integer labels, signed and calibrated counts, status and the full per-crossing
`(layer_index, contribution)` sequence are identical to the earlier
implementation.  The sweep includes phase velocities 0.1% either side of every
layer velocity.  `tests/python/test_mode_count_endpoint_reuse.py` pins the same
properties on the repository fixtures.

Diagnostic floats are not bit-identical.  The analytic and generator routes
agree with the old AD derivative to about 1e-12 to 1e-14 relative rather than
to the last bit, so `debug_mode_count` fields -- `secular`, `secular_derivative`,
`conjugate`, `test_value` and `local_depth_km` -- can move in their last digits.
On a grazing crossing, where the secular derivative sits at the cancellation
noise floor (one fixture has `1e-7` against a conjugate of `1e7`), the crossing
depth is not determined to better than a substep by either route and can move
visibly.  The label does not, because it depends on the sign of `test_value`.

Evaluated exactly on a root, whether the free-surface crossing is registered
can flip, since its secular value is zero there and the detection threshold is
absolute.  Four survey roots report one crossing more or less in
`debug_mode_count` than before.  Their labels are unchanged: the endpoint
calibration subtracts exactly that crossing when it is present.

At a phase velocity bit-identical to a layer `Vs` the label can change -- the
ocean fixture at `c = 2.2` returns 1 where it used to return 0.  That is the
degenerate point itself, not a change of convention: the layer's vertical
wavenumber vanishes there, and both implementations return 2 on either side of
it, within 1e-15 relative.  This is the same band that
[Comparing a counter against root indices](#comparing-a-counter-against-root-indices)
already excludes from counter verification.

### The endpoint operators are built once per layer (2026-09-01)

`B(z)` and `A B(z)` do not depend on the state they are applied to, and every
substep inside a layer uses the same two depths, `z = 0` and `z = dz`.  They are
therefore loop-invariant: the counters build them once per layer
(`build_endpoint`, `build_liquid_endpoint`) and apply them `num_steps` times
(`apply_endpoint`, `apply_liquid_endpoint`).  What is left inside a substep is a
matrix-vector product.

This composes with the section above rather than replacing it: the generator
still supplies the derivative, and the state is still contracted last, so the
arithmetic each substep performs is unchanged and the endpoints come out bit for
bit as they did when the matrices were rebuilt every substep.  Verified as
byte-identical output -- integers, crossing depths at 17 significant digits,
contributions -- over 21840 Rayleigh `(model, omega, c)` points across seven
models including two ocean models and one VTI model, `f` from 0.01 to 5 Hz.

One Rayleigh mode count on the 50-layer Level-1 PREM scan model, `f = 0.05 Hz`,
measured back to back: 7039 us before any of this work, 783 us with the
generator alone, **194 us with both** (36x).  Love is untouched -- it has no
substep mesh left to hoist out of.

**Do not replace the substep chain without revalidating surface roots.** Inside a layer the
reduced Rayleigh propagator's entries live in a fixed five-dimensional function
space (`span{1, cosh(sigma z), sinh(sigma z), cosh(delta z), sinh(delta z)}`
with `sigma = nu1 + nu3`, `delta = nu1 - nu3`), so the secular minor can be
evaluated at any depth from five per-layer coefficients, about 40x cheaper per
sample. That was implemented and rejected under the previous convention
(before the halfspace initialization correction below). At a **supra-Rayleigh root** the
endpoint calibration was disabled, so the count included the free-surface
crossing, and whether that crossing is found depends on the sign of the secular
value at the surface -- which at a root is zero to well below the round-off
accumulated along the substep chain.  For the `test_bran_mode_count.py` fixture
at `f = 0.5 Hz`, the 10th root has `|Y| ~ 3.3e3` at the surface and a secular
value of `+1.7e-10` after 296 chained substeps; a one-shot closed-form
evaluation of the same quantity returns `-1.1e-10`, the crossing is missed, and
the root labels as 9 instead of 10.  A chained evaluation reproduces the sign
(`+1.6e-10`) because the accumulation pattern, not the exact arithmetic, is what
determines it.  This is the same wall the refinement-stability experiment hit
from the other side (the rejected Rayleigh mesh experiment is recorded in the
research archive dated 2026-08-31, not bundled in this source distribution).

Any future rewrite that changes the accumulation must first decide what the
counter should do when the secular value at the free surface is below its own
noise floor; `value_eps` in `countBranHermiteCrossings` is an absolute `1e-14`
and does not cover it.

## Rayleigh Waves in Solid Layers

For solid Rayleigh layers, Disper26 uses the same Woodhouse minor pair that was
used by the legacy phase definition:

```text
secular  y = I24
conjugate t = -(I23 + I14)
```

Here `I24` is the Rayleigh secular minor in the reduced Dunkin vector.  Crossings
of `I24` are counted using `t * dI24/dz`.

### Analytical halfspace initialization

The halfspace state is the compound vector of the two decaying solutions. In
the same homogeneous medium, propagation only multiplies this vector by a
positive exponential; it introduces no depth crossings. The primary counter
therefore uses the analytical boundary state directly, without a virtual
halfspace segment or a halfspace root search.

With the `I12 >= 0` orientation enforced by `computeInitialVector`, the
halfspace secular minor `I24` changes from positive to negative across the
halfspace Rayleigh root. Its initial signed contribution is 0 below that root
and -1 above, matching the orientation of solid-layer crossings. When there
are no finite layers, the halfspace boundary is also the free surface, and a
numerical zero is assigned to the lower count using `64 * epsilon(double)`
times the absolute sum of the uncancelled terms in
`c44 * (d1*d4 - d2*d3)`. This allows for roundoff through the fixed-size
initial-vector arithmetic and tracks the secular determinant's own frequency
scaling; other minors have different units and are unsuitable as its scale.
This is not a change to the root solver's velocity tolerance.
With finite layers the halfspace boundary is internal, so its exact secular
sign is retained: the seed and a crossing entering/leaving the bottom layer
must cancel together across the halfspace Rayleigh root.
At an initial numerical zero, the first layer's derivative determines whether
the boundary crossing lies on the interior side consistent with that seed.
An exterior crossing is suppressed using the existing depth deduplication
tolerance (1e-8 km). This applies to both solid and liquid first layers;
the liquid check uses the Lagrangian normal traction extracted at the seabed.

For an all-solid stack, the final label is

```text
abs(initial_signed_count + signed_count - surface_endpoint_contribution).
```

The existing surface depth tolerance (1e-6 km) now applies on both sides of
the halfspace Rayleigh root. Thus the label at a resolved root remains its
zero-based index, while values away from roots give the number of slower
roots. For an ocean stack the initial contribution is added to the solid
signed sum before combining the absolute solid and liquid sums; adding it
after taking absolute values would mishandle crossings of opposite signs.
Love initialization and its counting algorithm are unchanged.

Previously the integer counter started from zero even above the halfspace
Rayleigh root. Disabling the endpoint correction on that side hid the missing
unit at some roots but left counts between roots one low. In particular, a
homogeneous halfspace had no count jump, so mode-order root selection failed.
The previous sub-Rayleigh virtual segment also inverse-propagated a growing
compound mode and could generate numerical crossings near the halfspace
root. It is removed from the primary counter; the legacy continuous
diagnostic retains its previous algorithm.

The analytical initialization has constant time and memory cost per count.
Finite-layer propagation and its substep mesh are unchanged, and removing the
virtual segment saves its inverse matrix and propagation work.

`tests/python/test_halfspace_mode_count.py` checks the isotropic Rayleigh
cubic, invariance under adding identical finite layers, and 3,337 off-root
counts against secular sign-change scans at two resolutions. The scan models
include an isolated halfspace, identical-layer splits, ordinary solid and
ocean stacks, a fast surface layer, VTI media, and a low-velocity zone at
0.02, 0.1 and 0.5 Hz. The public mode-order path is checked for Layer-cake and
both GRID interpolations. Existing tests retain root labels and Love behavior;
the pinned off-root values at c=2.95 (solid) and c=2.9 (ocean) increase by one
to agree with the scanned root counts.
Additional checks cover PREM (flat, 400 km, 20 km input spacing, refinement
level 1) at 0.02/0.05/0.1 Hz, an ocean directly on the halfspace, and the
boundary transfer at the halfspace Rayleigh speed itself.

### Halfspace correction validation and timing (2026-09-24)

Baseline: `4be4984`, built with `python setup.py
build_ext --inplace`. Before the change, the 11 existing counter regressions
passed, but 325 of the 3,337 off-root scan comparisons failed. After the
change all 3,337 agree, including the roots above the halfspace Rayleigh speed.
The homogeneous example (`vs=4.5`, `vp=8`, `rho=3.3`) now solves through the
public mode-order path at `c=4.150909460643233` km/s, versus
`4.150909460643234` from the independent Rayleigh cubic. Its group velocity
is `4.150909460643235` km/s.

Checks run after rebuilding:

```sh
python -m pytest -q \
  tests/python/test_halfspace_mode_count.py tests/python/test_bran_mode_count.py \
  tests/python/test_mode_count_endpoint_reuse.py tests/python/test_model_arrays.py
# 65 passed
python -m pytest -q \
  tests/python/test_public_api.py tests/python/test_public_api_related.py \
  tests/python/test_phase_velocity_curves.py tests/python/test_adaptive_root_scan.py \
  tests/python/test_wrapper_full.py
# 36 passed
python tests/python/verify_integer_alignment.py
# 10/10 root labels, SUCCESS
ctest --test-dir build -R '^(public_api_|fortran_)' --output-on-failure
# 7 passed, after rebuilding the C++, C and Fortran targets
```

These are targeted checks; the full Python suite was not rerun for this change.

The table measures one `compute_mode_count` call through Python, using the
same Conda build configuration before and after. Each value is the median of
7 batches of 2,000 calls after a warmup call. The old extension was preserved
before rebuilding and loaded in a separate process. `Vs_hs=sqrt(c44/rho)`;
frequency is 0.5 Hz except PREM (0.05 Hz). The four small models come from
`comparison_models()` in the new regression test, and PREM uses the settings
above. No compilation or tests ran concurrently with the timing.

| Model | c / Vs_hs | Before (µs) | After (µs) |
|---|---:|---:|---:|
| Halfspace only | 0.80 | 3.928 | 0.354 |
| Halfspace only | 0.98 | 0.369 | 0.355 |
| Solid | 0.80 | 40.990 | 36.070 |
| Solid | 0.98 | 44.105 | 42.771 |
| Ocean | 0.80 | 24.306 | 20.857 |
| Ocean | 0.98 | 30.123 | 30.003 |
| VTI | 0.80 | 13.818 | 9.932 |
| VTI | 0.98 | 18.140 | 17.441 |
| PREM, 50 layers | 0.80 | 177.819 | 166.828 |
| PREM, 50 layers | 0.98 | 185.771 | 180.120 |

The sub-Rayleigh cases benefit from removal of the virtual segment. The
supra-Rayleigh cases have comparable cost; differences of a few percent
should not be interpreted as a guaranteed speedup. To reproduce the workload
from the repository root (preimport a preserved old extension before the
fixture module to measure the baseline):

```python
import math, statistics, sys, timeit
sys.path[:0] = ["tests/python", "inversion"]
from test_halfspace_mode_count import comparison_models
from disper26_inv.prem import load_prem_flat
import disper26py as dp

models = [x for x in comparison_models() if x[0] in {"halfspace", "solid", "ocean", "vti"}]
prem = load_prem_flat(max_depth_km=400., max_spacing_km=20.)
models.append(("prem", *prem.refined_layers(level=1)))
for name, layers, thicknesses, halfspace in models:
    omega = 2*math.pi*(0.05 if name == "prem" else 0.5)
    for fraction in (0.8, 0.98):
        c = fraction*math.sqrt(halfspace.c44/halfspace.rho)
        call = lambda: dp.compute_mode_count(layers, thicknesses, halfspace, omega, c, "Rayleigh")
        call()
        samples = timeit.repeat(call, repeat=7, number=2000)
        print(name, fraction, statistics.median(samples)*1e6/2000)
```

## Love Waves

For solid Love waves, the state vector is the SH displacement and traction:

```text
state     = [W,T] = [u_y, c44*du_y/dz], z positive upward
halfspace = [1,+eta], eta=c44*nu
upward h  = A(+h) * state_bottom
secular   = T
conjugate = u_y
```

The free-surface condition is `T = 0`, so the counter follows traction
crossings through the solid layers. Stored depths and API depth arguments
remain positive downward, `depth=-z`. The halfspace integer seed remains zero;
the final label is `abs(signed_count - surface_endpoint_contribution)`.

The upward-coordinate change reflects the previous internal stress: `T=-T_old`.
Crossing `secular_derivative` and `test_value` consequently reverse sign;
`conjugate`, integer labels and signed contributions keep their meanings.
The closed-form counter records `secular=0` exactly at a crossing. Its debug
derivative is the physical `dT/ds=-c44*r^2*W` in the propagated gauge.
The 5-component dispersion result and `compute_dispersion_value` keep their
historical Love residual `F=-T_surface`, including its sign. This residual is
separate from the physical state and the public eigenfunction `tyz=T`.

### Status: the AD-z counter is verified for Love

Checked by evaluating the counter at true Love roots, where it must return the
root index `n`.  Five models, 64 roots; the four roots sitting within 5e-3
relative of a layer `Vs` are excluded and counted separately, because the
counter there legitimately turns on the last bit (see
[Comparing a counter against root indices](#comparing-a-counter-against-root-indices)).

The legacy integer counter is measured alongside for the record.  It is what
motivated removing it, and it is the reason the AD-z column is worth stating
explicitly for Love rather than assumed from the Rayleigh case.

| model | roots used | `compute_mode_count` | legacy integer (removed) |
|---|---:|---:|---:|
| 3-layer (as in `test_bran_mode_count`) | 8 | 8/8 | 7/8 |
| 1-layer thin | 4 | 4/4 | 4/4 |
| 1-layer thick | 26 | 26/26 | 26/26 |
| crust/mantle at 20 s | 1 | 1/1 | 1/1 |
| 3-layer at high frequency | 21 | 21/21 | 18/21 |
| **total** | **60** | **60/60** | **56/60** |

There is only one production counter, not two: `calculate_total_mode_count` and
`calculate_bran_mode_count` are the same function under older names (both
deprecated aliases of `compute_mode_count`), so a table column for each would
have compared a function with itself.

The legacy integer's error was not a small offset -- see
[Love integer labels were offset](#love-integer-labels-were-offset-by-about-half-a-step-why-the-integer-was-removed).
That section surveys the same five models but over **all** roots rather than
excluding the four near `c ~ Vs`, so its per-model root counts are slightly
larger than the table above (1-layer thick is 29 there, 26 here).

## Ocean Rayleigh Models

Rayleigh models with a top ocean over solid layers are supported.  Once the
upward propagation reaches a liquid layer, the counter switches from the solid
minor vector to the liquid state

```text
state     = [U, P]
secular  = P
conjugate = U
```

The solid-to-liquid transition uses the same extraction as the dispersion path:

```text
U = I14
P = I24
```

The Lagrangian normal traction is continuous at the seabed; no separate
density-contrast gravity jump is applied before liquid propagation.

For ocean Rayleigh models, the final integer label combines per-regime signed
sums, `|signed_solid| + |signed_liquid|`, rather than the raw signed total or
an unsigned crossing tally.  The raw cross-regime total is unusable because
the solid and liquid crossing orientations can have opposite signs and cancel
even when the branch order is increasing correctly; a fully unsigned tally, on
the other hand, double-counts spurious +/- crossing pairs (grazing zeros near
mode cutoffs), which was observed to produce non-monotonic labels in
thin-water/soft-sediment models.  Summing with sign within each regime avoids
both failure modes.

Liquid Love models and unsupported liquid-to-solid transitions are reported as
unsupported by the AD-z counter.

## Endpoint Calibration

At a true root, the free-surface secular condition is zero.  If the crossing at
the top boundary is counted naively, it introduces an off-by-one label.  The
counter therefore applies a small endpoint calibration: when the final crossing
lies at the top free-surface endpoint, that endpoint contribution is removed
from the final label.

This calibration is used for solid Rayleigh, solid Love, and ocean Rayleigh
models.  It is a counting convention, not a change to the dispersion equation.

## Legacy Phase Counter

The legacy algorithm is the previous Woodhouse-phase accumulation method.  For
Rayleigh waves it uses

```text
theta = atan2(I24, -(I23 + I14))
```

It initializes from the halfspace Woodhouse phase, applies a `-0.5*pi` offset,
uses a virtual halfspace contribution on the sub-Rayleigh side, and applies a
supra-Rayleigh correction when the halfspace secular minor changes sign.

For liquid Rayleigh layers, the legacy path uses the liquid phase pair
`atan2(P, U)` and a liquid-layer winding count. For Love, compatibility requires
the previous angle `atan2(-T,W)` and the previous reverse-direction layer
sampling. The physical boundary states advance upward with `A(+h)`, while the
legacy layer diagnostic samples `A(-s)` from each bottom state. This preserves
the public continuous phase and debug fields; it does not define a physical
upward phase integral or the production integer count.

This legacy path remains useful for debugging because it gives a continuous
phase-like diagnostic, but it can duplicate or skip integer labels in difficult
cases.  The AD-z counter is the production integer label path.

### What the legacy layer diagnostic guarantees

`compute_continuous_mode_count` and `debug_continuous_mode_count` expose the
legacy phase diagnostic, not the production integer label. The latter returns
layer phase details, not individual zero crossings. Its phase-derived integer
fields are diagnostic only. `debug_mode_count` also evaluates the legacy path
for its comparison field `legacy_continuous_count` on supported models; that
field does not determine `n_integer`. Public names and aliases are unchanged.

The angle `atan2(I24, -(I23+I14))` is undefined when both arguments vanish.
Near that point, roundoff can dominate the angle and its accumulated winding;
being far from an integer value of `abs(phase)/pi` does not ensure stability.
The implementation's finite return value at a singular phase pair is not a
well-defined mathematical phase. No new endpoint convention is imposed here.

The former `tests/verify_mode_count.cpp` fixture set only `Y12=1` in the
reduced state (`Y34=-Y12`). Besides having a zero initial phase pair, it is not
a decomposable minor state: `Y12*Y34-Y13*Y24+Y14*Y23=-1`, rather than zero.
For its isotropic material (`rho=mu=lambda=1`), `omega=1`, `k=0.5`, and
`d=6*pi/sqrt(3/4)`, the vertical P and S phases are exactly `2*pi` and `6*pi`.
The state propagator is therefore the identity and the final phase pair is
also zero. The old expectation `2.1841892928944762` was an angle of roundoff,
not an independently defined reference value.

On Ubuntu 24.04 / GCC 13.3, with assertions enabled, that test passes at its
introduction (`cc79e33`) and fails after the real-kernel reversion (`94310d1`),
as well as before and after the macOS build change (`66221fd`, `27bb401`).
At `27bb401` its final pair was approximately
`(x,y)=(-3.36e-15,2.22e-15)`. Increasing the propagation steps from 943 to
1886 and 3772 changed `phase/pi` from `-2.18585` to `-3.33151` and `-0.12746`.
This is an ill-defined test input, not evidence of a changed integer-label
convention.

The replacement uses the decomposable state `(.3*e1+e2) wedge e4`, with
`Y14=.3`, `Y24=1`. Its reference solves the isotropic four-state ODE using
spectral projectors onto the P/S subspaces, then forms the second compound
at each depth. It does not call the production propagator or phase-unwrapping
helper. Direct-depth grids of 2048 and 4096 intervals verify the winding
independently. Cases cover evanescent waves, S-only propagation, P/S
propagation and thicknesses on both sides of `5*pi` and `6*pi` in S phase.
The existing `1e-9` phase tolerance is retained, with a tighter `1e-11`
reference-refinement check. Exact integer boundaries test the phase; the
nearby cases test the integer too. The Love tests retain their analytic
quarter/full-turn endpoint checks. No numerical production code is changed.

Validation on Ubuntu 24.04 / GCC 13.3 / Eigen 3.4: both `Debug` (`-g`, no
`NDEBUG`) and `Release` with `CMAKE_CXX_FLAGS_RELEASE="-O3 -UNDEBUG"` pass
`verify_mode_count`, including the Love cases. Maximum phase error is
`5.68434e-14` radians; maximum reference-grid difference is `4.68375e-17`.
The source-build command is `cmake --build <build-dir> --target verify_mode_count -j4`,
followed by `<build-dir>/tests/verify_mode_count`.

After `PYTHONNOUSERSITE=1 python -m pip install .` into a dedicated Conda
Python 3.12 environment, the installed extension passes
`test_bran_mode_count.py`, `test_mode_count_endpoint_reuse.py`,
`verify_integer_alignment.py` (10/10 roots), and `test_wrapper_full.py`.
The runner preimports the installed extension with `python -I` before running
the scripts via `runpy`, preventing their historical `sys.path` modifications
from selecting a repository-local binary. macOS was not rerun for this change.

### Love integer labels were offset by about half a step (why the integer was removed)

This section is about the legacy counter alone.  Love mode counting itself is
fine: `compute_mode_count` labels every measured Love root correctly, 60 out
of 60 -- see [Status: the AD-z counter is verified for
Love](#status-the-ad-z-counter-is-verified-for-love).  It was the legacy
diagnostic that was wrong, and only for Love.  Its integer wrapper has since
been removed; the measurements below are why, and they still describe the
continuous phase that remains.

`calculate_legacy_total_mode_count(..., "Love")` disagreed with the AD-z and
Bran counters over wide bands, not only at boundaries.  Sweeping one layer
(vs 1.0, vp sqrt(3), rho 1.0) over a fast half-space (vs 3.0), at omega 1.0 and
c 2.0, with `r_beta * thickness` from `3*pi` to `6*pi`, disagrees at 46 of 81
sample points -- including points where the continuous count sits half a unit
away from any integer:

| `r_beta*d` | legacy | AD-z | Bran | continuous |
|---|---:|---:|---:|---|
| `3.4500*pi` | 3 | 4 | 4 | 3.477708529889536 |
| `4.4250*pi` | 4 | 5 | 5 | 4.454985133044042 |
| `5.4375*pi` | 5 | 6 | 6 | 5.466392592651707 |

The legacy integer is `floor(|total_phase| / pi)`.  There is a real structural
difference between the two branches: the Rayleigh branch seeds `total_phase`
from the half-space Woodhouse phase minus `0.5*pi` (`src/disper26.cpp:3368`,
added by `2774ab7`), whereas the Love branch starts from zero and sums layer
contributions only, with no half-space phase reference at all.
`countModesLayerLove` also never fills in the integer half of its return value
(`n_sub` is declared 0 and never assigned), though that is harmless because the
legacy wrapper derives its integer from the summed phase instead.

**Porting the Rayleigh seed to Love does not fix this, and was measured rather
than assumed.**  Evaluating `C = |total_phase| / pi` at true Love roots, where
the answer must be the root index `n`, across the same five models, over all
roots this time -- including the four near `c ~ Vs` that the status table
excludes, which is why the root counts here run slightly higher:

| model | `C - n` min | max | spread |
|---|---:|---:|---:|
| 3-layer (as in `test_bran_mode_count`) | -0.047 | +0.827 | 0.874 |
| 1-layer thin | +0.155 | +0.499 | 0.344 |
| 1-layer thick | +0.054 | +0.504 | 0.450 |
| crust/mantle at 20 s | +0.198 | +0.198 | 0.000 |
| 3-layer at high frequency | -0.822 | +0.964 | 1.786 |

An additive calibration `s` works only if `floor(C + s) == n` everywhere, which
needs `C - n` to span less than one unit.  It spans 1.786.  The reason is
visible in the high-frequency model, where `C` is not even monotonic in `n`:

```text
n = 15  C = 15.935
n = 16  C = 15.554     <- decreases
n = 17  C = 16.178
n = 18  C = 18.174     <- jumps by two
```

No offset, constant or derived from the half-space eigenvector, can turn a
non-monotonic quantity into a correct integer label.  The Love accumulation
itself loses and gains phase; the missing seed is a symptom of the branch never
having been calibrated, not the defect.  A real repair means reworking the Love
phase accumulation, which is a different and much larger job than `2774ab7`.

So this stays recorded rather than repaired.  Nothing in production depends on
it: `inversion/disper26_inv/engine.py:159` labels branches with
`dp.calculate_mode_count(...)`, which is the same AD-z/Bran counter as
`calculate_bran_mode_count()` (`bindings.cpp:121` dispatches both aliases to
`calculateAdModeCount`), and the legacy integer is opt-in behind a name that
says legacy.  Use the AD-z or Bran integer for Love. (An earlier version of
this paragraph named `calculate_bran_mode_count()` specifically, which was
itself an alias-induced slip of the kind `docs/naming_convention.md`
addresses -- `engine.py` calls a different one of the three interchangeable
names.)

The scans behind both tables use 40000 points, for the reason given in
[Comparing a counter against root indices](#comparing-a-counter-against-root-indices)
-- a first pass at this measurement reached the opposite conclusion purely
because the root indices it compared against were wrong.

The regression test that used to pin this disagreement went with the API it
pinned.  `tests/python/test_bran_mode_count.py` now checks instead that
`compute_mode_count`, `calculate_total_mode_count` and
`calculate_bran_mode_count` all return the same value, so the aliases cannot
drift apart.

## Practical Notes

- `search_roots(...)` still finds roots of the dispersion function; the counter
  labels those roots after the fact.
- For branch tracing, sorted roots and mode counts should be checked together.
- Near Stoneley/Scholte-like branches or mode crossings, inspect
  `debug_mode_count(...)` rather than relying only on the integer label.
- `compute_continuous_mode_count(...)` is legacy/debug information.  Its
  integer part should not be treated as the primary mode label.
- Take the integer from `compute_mode_count(...)`.  The legacy integer
  counter has been removed; what is left of the legacy path is the continuous
  count, whose integer part is not a mode label -- for Love it is not even
  monotonic in mode number.
- Branch-labeled GRID/Romberg evaluation uses the AD-z count to bracket the
  requested mode order separately on every refinement mesh when the normal
  nearest-root solve fails its final branch confirmation.  Eigenfunction
  character remains diagnostic information and does not change the mode-order
  label.

## Verification

Recommended checks after mode-counting changes, using the Python interpreter
from the dedicated environment described in [installation](install.md):

```bash
python setup.py build_ext --inplace --force
python tests/python/test_bran_mode_count.py
python -m pytest tests/python/test_bran_mode_count.py \
  tests/python/test_mode_count_endpoint_reuse.py -q
python tests/python/verify_integer_alignment.py
```

`test_bran_mode_count.py` checks the counter against dispersion roots;
`test_mode_count_endpoint_reuse.py` checks that the labels stay put when the
propagation itself is restructured.  Both are needed after a change to the
counter: the first would not notice a label that moved consistently with its
own roots, and the second does not know where the roots are.

Note that `tests/test_ocean_mode_count.cpp` does not compile: it calls
`calculateTotalModeCount`, a C++ entry point that no longer exists (only the
binding of that name does).  Ocean coverage comes from the Python fixtures.

### Comparing a counter against root indices

The natural way to check a counter is to find the roots at fixed `omega` and
assert that the counter returns each root's index.  Three things make that
harder than it looks, all of them learned the expensive way -- the first survey
of the Love counters produced two wrong verdicts in a row before these were
accounted for.

**Evaluate at the root, not around it.**  At a root `c_n` the counters return
`n`, and the value is stable across at least `c_n * (1 +/- 1e-9)`.  It is
tempting to assume the count is a step at each root, `n` just below and `n + 1`
just above, and to assert that instead; it is not, and doing so makes correct
counters look broken on nearly every root.

**Skip roots that sit on top of a layer velocity.**  Where `c` is within about
5e-3 relative of some layer `Vs`, the secular function changes sign within
about 5e-9 of the root, so which side Brent lands on -- and therefore the
counter's answer -- is decided by the last bit.  A 1-layer model with `Vs = 1.0`
has a root at `c = 1.0012500`, where `compute_mode_count` returns 1 or 2
depending on a 4.7e-9 difference in the refined root, tracking `F(c)` exactly.
That is the known `c ~ Vs` conditioning cliff, not a counter defect.  Four of
the 64 roots in the Love survey fall in this band.

**Check the root indices before trusting them.**  `search_roots(...)` misses
closely spaced roots: in a 30 km layer of `Vs = 1.0` over a fast half-space it
finds 27 roots over `[0.5, 3.0]` at 800 scan points and 29 at 5000, because
approaching `c -> Vs` the spacing drops below the scan step.  With two roots
missing, every index above them is off by two and the counters appear to be the
ones at fault.  Refine the scan until the root count stops changing before
using the index as an oracle.

## References

- Woodhouse, J.H. (1988). "The calculation of eigenfrequencies and
  eigenfunctions of the free oscillations of the Earth and the Sun." In
  *Seismological Algorithms*, ed. D.J. Doornbos, pp. 321-370.
