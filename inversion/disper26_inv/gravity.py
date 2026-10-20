"""First-order gravity correction for Rayleigh waves, applied outside the solver.

The solver is non-gravitating.  Gravity enters as a post-correction built from
the g = 0 solution, which is what makes the whole approach cheap: nothing in
here perturbs the eigenvalue problem. See docs/energy_integral.md for the
energy and correction conventions. Earlier Mineos validation reports are
historical research records, not bundled here.

    delta c(T) = integral K_g(z) g(z) dz
    K_g = c^2 / (omega^2 U I1) * rho * k * y1 * y3

with positive-downward flat depth and positive gravity magnitude.  ``delta_c``
below is the gravity contribution to the phase velocity, i.e. it estimates
``c(g) - c(g=0)``, so the intended use is to remove it from the data:

    c_target(T) = c_observed(T) - delta_c(T)
    U_target(T) = U_observed(T) - delta_u(T)

and then compare against the ordinary non-gravitating forward calculation.
Because the correction is subtracted from observations rather than added to the
forward model, no derivative of it with respect to model parameters is ever
needed -- it does not enter the Jacobian or the Hessian.

Group velocity needs no separate kernel.  delta_c is a function of frequency,
so differentiating k = omega/c gives

    delta U = (U^2/c^2) [ (2c/U - 1) delta_c + omega d(delta_c)/d omega ]

The omega-derivative term is not a small correction to this: it runs 1.5-4.7x
the size of delta_c itself and changes sign across the Airy phase, so
delta_u/delta_c sweeps from -0.7 to +3.4 for an ocean model.  It is therefore
evaluated by an explicit central difference in omega rather than by
differentiating a user-supplied period grid, which is generally too coarse near
the Airy phase (a 10% step in omega costs 5% there, against 1% at 20 s).

Rayleigh only, and Love needs nothing.  SH motion has no vertical displacement
and div u = 0, so it sources no potential perturbation and feels no buoyancy or
pre-stress term; the toroidal equations contain no g.  K_g is proportional to
y1 y3, a product of the vertical and horizontal P-SV eigenfunctions, and SH has
no such pair.

Validity rests on the g = 0 problem being stationary; historical diagnostics
recorded stationarity to 1e-14. Accuracy against the
fluid Cowling solver is 0.2-0.6% over 1-10 s, where the correction matters;
tests/python/test_gravity_kg_correction.py pins both delta_c and delta_u
against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

import disper26py as dp

EARTH_GRAVITY_KM_S2 = 9.80665e-3
"""Surface gravity in the km/s^2 units the bindings use."""

REGIONS = ("all", "solid", "fluid")


@dataclass
class GravityCorrection:
    """Gravity contribution to a Rayleigh dispersion curve.

    Every array is aligned with ``periods_s``.  ``delta_c`` and ``delta_u`` are
    in km/s and are quantities to subtract from observations.  ``c`` and ``u``
    are the underlying g = 0 curve, returned because the caller usually wants to
    confirm the branch that was corrected.
    """

    periods_s: np.ndarray
    c: np.ndarray
    u: np.ndarray
    delta_c: np.ndarray
    delta_u: np.ndarray
    omega_dc_domega: np.ndarray   # omega * d(delta_c)/d omega, the dominant term in delta_u
    region: str


def _gravity_fn(gravity) -> Callable[[float], float]:
    if callable(gravity):
        return gravity
    value = float(gravity)
    return lambda _depth_km: value


def _layer_at(layers, thicknesses, halfspace, depth_km: float):
    top = 0.0
    for layer, thickness in zip(layers, thicknesses):
        if depth_km <= top + thickness + 1.0e-10:
            return layer
        top += thickness
    return halfspace


def _is_fluid(layer) -> bool:
    """c44 is the shear modulus the propagator switches on; a water layer has
    c44 = 0 exactly, so this needs no tolerance."""
    return float(layer.c44) <= 0.0


def _find_root(layers, thicknesses, halfspace, omega, c_range, near=None):
    if near is None:
        lo, hi = c_range
    else:
        lo, hi = near * 0.99, near * 1.01
    roots = dp.search_roots(layers, thicknesses, halfspace, omega,
                            lo, hi, 6000, "Rayleigh")
    if not roots:
        raise RuntimeError(
            f"no Rayleigh root in [{lo:.4f}, {hi:.4f}] km/s at omega={omega:.6f}")
    if near is None:
        return roots[0]
    return min(roots, key=lambda r: abs(r - near))


def _polish_root(layers, thicknesses, halfspace, omega, c):
    """search_roots stops at a 1e-6 absolute Brent tolerance.  delta_c is
    differenced across neighbouring frequencies, which amplifies any jitter in c
    by 1/(2 eps), so bisect down to machine precision first."""
    def f(x):
        return dp.calculate_dispersion_value(layers, thicknesses, halfspace,
                                             omega, x, "Rayleigh")

    lo, hi = c * (1.0 - 1e-7), c * (1.0 + 1e-7)
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        return c
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo <= 1e-15 * mid:
            break
        f_mid = f(mid)
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def rayleigh_gravity_delta_c(layers, thicknesses, halfspace, omega, c, u, *,
                             gravity=EARTH_GRAVITY_KM_S2,
                             region: str = "all",
                             num_sub_layers: int = 40) -> float:
    """integral K_g g dz over ``region``, at one frequency.

    ``c`` and ``u`` are the g = 0 phase and group velocities of the branch being
    corrected.  ``region`` restricts the depth integral to ``"solid"`` or
    ``"fluid"`` layers; the split is by c44, evaluated at each sub-interval
    midpoint.
    """
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
    g_at = _gravity_fn(gravity)

    ef = dp.compute_rayleigh_eigenfunctions(layers, thicknesses, halfspace,
                                            omega, c, num_sub_layers)
    i1 = float(dp.compute_energy_integrals(ef, layers, thicknesses, halfspace,
                                           omega, c, "Rayleigh")[0])
    if not np.isfinite(i1) or abs(i1) <= 1e-30:
        raise RuntimeError(f"non-finite Rayleigh I1 for the gravity kernel: {i1}")
    if not np.isfinite(u) or abs(u) <= 1e-30:
        raise RuntimeError(f"non-finite group velocity for the gravity kernel: {u}")

    depth = np.asarray(ef["depth"], dtype=float)
    uz = np.real(np.asarray(ef["uz"]))
    ur = np.real(np.asarray(ef["ur"]))

    k = omega / c
    prefactor = c * c / (omega * omega * u * i1)

    total = 0.0
    for i in range(len(depth) - 1):
        dz = depth[i + 1] - depth[i]
        if dz <= 0.0:
            continue
        z_mid = 0.5 * (depth[i] + depth[i + 1])
        layer = _layer_at(layers, thicknesses, halfspace, z_mid)
        if region == "solid" and _is_fluid(layer):
            continue
        if region == "fluid" and not _is_fluid(layer):
            continue
        y1 = 0.5 * (uz[i] + uz[i + 1])
        y3 = 0.5 * (ur[i] + ur[i + 1])
        total += prefactor * float(layer.rho) * k * y1 * y3 * g_at(z_mid) * dz
    return total


def rayleigh_gravity_correction(layers, thicknesses, halfspace,
                                periods_s: Sequence[float], *,
                                gravity=EARTH_GRAVITY_KM_S2,
                                region: str = "all",
                                phase_velocity: Sequence[float] | None = None,
                                num_sub_layers: int = 40,
                                c_range: tuple[float, float] = (0.5, 10.0),
                                domega_rel: float = 5e-3) -> GravityCorrection:
    """Gravity contribution to phase and group velocity over a period grid.

    With ``phase_velocity=None`` the fundamental mode is taken at each period
    (the lowest Rayleigh root).  Pass an explicit array to correct an overtone
    branch that the caller has already resolved -- neighbouring frequencies are
    then tracked by nearest root, so the branch is preserved as long as it is
    not crossing another one within 1% in c.

    ``domega_rel`` is the central-difference step for d(delta_c)/d omega, in
    relative omega.  The default is well inside the plateau: 1e-3 and 1e-2 agree
    to 0.05% even at the Airy phase.
    """
    periods = np.asarray(periods_s, dtype=float)
    if phase_velocity is not None:
        phase_velocity = np.asarray(phase_velocity, dtype=float)
        if phase_velocity.shape != periods.shape:
            raise ValueError("phase_velocity must have the same shape as periods_s")

    def at(omega, near):
        c = _find_root(layers, thicknesses, halfspace, omega, c_range, near=near)
        c = _polish_root(layers, thicknesses, halfspace, omega, c)
        u = dp.calculate_group_velocity(layers, thicknesses, halfspace,
                                        omega, c, "Rayleigh")
        dc = rayleigh_gravity_delta_c(layers, thicknesses, halfspace, omega, c, u,
                                      gravity=gravity, region=region,
                                      num_sub_layers=num_sub_layers)
        return c, u, dc

    c_out, u_out, dc_out, du_out, wd_out = [], [], [], [], []
    for idx, period in enumerate(periods):
        omega = 2.0 * np.pi / period
        near = None if phase_velocity is None else float(phase_velocity[idx])
        c, u, dc = at(omega, near)

        # omega * d(delta_c)/d omega by central difference in log omega.
        _, _, dc_hi = at(omega * (1.0 + domega_rel), c)
        _, _, dc_lo = at(omega * (1.0 - domega_rel), c)
        omega_ddc = (dc_hi - dc_lo) / (2.0 * domega_rel)

        du = (u * u / (c * c)) * ((2.0 * c / u - 1.0) * dc + omega_ddc)

        c_out.append(c)
        u_out.append(u)
        dc_out.append(dc)
        du_out.append(du)
        wd_out.append(omega_ddc)

    return GravityCorrection(
        periods_s=periods,
        c=np.array(c_out),
        u=np.array(u_out),
        delta_c=np.array(dc_out),
        delta_u=np.array(du_out),
        omega_dc_domega=np.array(wd_out),
        region=region,
    )
