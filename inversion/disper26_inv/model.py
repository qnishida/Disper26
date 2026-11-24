"""Forward grid model in the velocity basis (rho, Vph, Vpv, Vsh, Vsv, eta).

Units: depth km, velocities km/s, density g/cm^3, eta dimensionless — the
conventions of the disper26py grid entry points. Discontinuities are duplicated
depth nodes with is_discontinuity=True on the first node of the pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import disper26py as dp

VELOCITY_FIELDS = ("rho", "vph", "vpv", "vsh", "vsv", "eta")


@dataclass(frozen=True)
class GridRefinementPolicy:
    """Coordinate and flattening convention used by a forward Grid call."""

    kind: str = "flat"
    planet_radius_km: float = 6371.0
    flattening_exponent: Optional[float] = None

    def __post_init__(self):
        aliases = {
            "Flat": "flat",
            "spherical-ba97": "spherical_ba97",
            "BA97": "spherical_ba97",
            "spherical-experimental-density": "spherical_experimental_density",
            "experimental_density": "spherical_experimental_density",
        }
        canonical = aliases.get(self.kind, self.kind)
        if canonical not in {"flat", "spherical_ba97", "spherical_experimental_density"}:
            raise ValueError(f"unknown grid refinement policy: {self.kind}")
        object.__setattr__(self, "kind", canonical)
        if not np.isfinite(self.planet_radius_km) or self.planet_radius_km <= 0.0:
            raise ValueError("planet_radius_km must be finite and positive")
        exponent = self.flattening_exponent
        if exponent is None:
            exponent = -2.0 if canonical == "spherical_ba97" else -2.275
        if not np.isfinite(exponent):
            raise ValueError("flattening_exponent must be finite")
        object.__setattr__(self, "flattening_exponent", float(exponent))

    def to_binding_kwargs(self) -> dict:
        return {
            "grid_policy": self.kind,
            "planet_radius": float(self.planet_radius_km),
            "flattening_exponent": float(self.flattening_exponent),
        }

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "planet_radius_km": float(self.planet_radius_km),
            "flattening_exponent": float(self.flattening_exponent),
        }


@dataclass
class ForwardGridModel:
    depths: np.ndarray
    rho: np.ndarray
    vph: np.ndarray
    vpv: np.ndarray
    vsh: np.ndarray
    vsv: np.ndarray
    eta: np.ndarray
    is_discontinuity: np.ndarray = field(default=None)
    name: str = ""
    refinement_policy: GridRefinementPolicy = field(default_factory=GridRefinementPolicy)

    def __post_init__(self):
        self.depths = np.asarray(self.depths, dtype=float)
        for name in VELOCITY_FIELDS:
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if self.is_discontinuity is None:
            self.is_discontinuity = np.zeros(self.depths.shape[0], dtype=bool)
        else:
            self.is_discontinuity = np.asarray(self.is_discontinuity, dtype=bool)
        if not isinstance(self.refinement_policy, GridRefinementPolicy):
            self.refinement_policy = GridRefinementPolicy(**self.refinement_policy)
        self.validate()

    @property
    def n_nodes(self) -> int:
        return int(self.depths.shape[0])

    def validate(self):
        n = self.depths.shape[0]
        if not np.all(np.isfinite(self.depths)):
            raise ValueError("depths must be finite")
        for name in VELOCITY_FIELDS:
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"field {name} has shape {arr.shape}, expected ({n},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values")
        if self.is_discontinuity.shape != (n,):
            raise ValueError("is_discontinuity shape mismatch")
        if n < 2:
            raise ValueError("grid model needs at least two nodes")
        spacing = np.diff(self.depths)
        if np.any((spacing > 0) & (spacing < 1e-12)):
            raise ValueError("Positive GRID depth intervals must be at least 1e-12 km")
        if np.any(spacing < 0):
            raise ValueError("depths must be non-decreasing")
        paired = np.r_[self.depths[:-1] == self.depths[1:], False]
        invalid = (self.is_discontinuity != paired) | (paired & np.r_[False, paired[:-1]])
        if np.any(invalid):
            i = int(np.flatnonzero(invalid)[0])
            raise ValueError(f"Expected a flagged two-node discontinuity at node {i}, depth {self.depths[i]:g} km")
        for name in ("rho", "vph", "vpv", "vsh", "vsv", "eta"):
            if np.any(getattr(self, name) <= 0.0):
                raise ValueError(f"{name} must be strictly positive (solid VTI stack only)")

    # -- stiffness (Love notation, units rho*v^2) -----------------------------

    def stiffness(self) -> dict:
        A = self.rho * self.vph**2
        C = self.rho * self.vpv**2
        L = self.rho * self.vsv**2
        N = self.rho * self.vsh**2
        F = self.eta * (A - 2.0 * L)
        return {"A": A, "C": C, "F": F, "L": L, "N": N, "rho": self.rho}

    # -- adapters for disper26py entry points ---------------------------------

    def vti_args(self):
        """Positional arguments for compute_dispersion_romberg_velocity_vti[_with_kernels]."""
        return (
            list(self.depths),
            list(self.rho),
            list(self.vph),
            list(self.vpv),
            list(self.vsh),
            list(self.vsv),
            list(self.eta),
            list(bool(b) for b in self.is_discontinuity),
        )

    def refinement_kwargs(self) -> dict:
        """Keyword metadata forwarded to every policy-aware Grid binding."""
        return self.refinement_policy.to_binding_kwargs()

    def to_params(self):
        """(depths, params, is_discontinuity) for the grid Hessian entry points."""
        stiff = self.stiffness()
        params = []
        for i in range(self.n_nodes):
            p = dp.PhysicalParameters()
            p.rho = float(self.rho[i])
            p.c11 = float(stiff["A"][i])
            p.c33 = float(stiff["C"][i])
            p.c13 = float(stiff["F"][i])
            p.c44 = float(stiff["L"][i])
            p.c66 = float(stiff["N"][i])
            params.append(p)
        return list(self.depths), params, list(bool(b) for b in self.is_discontinuity)

    def refined_layers(self, level: int = 2):
        """(layers, thicknesses, halfspace) via the solver's own Mesher refinement.

        Used for root scanning, mode counting and eigenfunctions so that branch
        labels are assigned on the same discretization family the Romberg solver
        extrapolates from.
        """
        depths, params, is_disc = self.to_params()
        out = dp.debug_refined_grid_layers(
            depths, params, is_disc, int(level), **self.refinement_kwargs())
        return out["layers"], out["thicknesses"], out["halfspace"]

    def copy(self, name: str = None) -> "ForwardGridModel":
        return ForwardGridModel(
            depths=self.depths.copy(),
            rho=self.rho.copy(),
            vph=self.vph.copy(),
            vpv=self.vpv.copy(),
            vsh=self.vsh.copy(),
            vsv=self.vsv.copy(),
            eta=self.eta.copy(),
            is_discontinuity=self.is_discontinuity.copy(),
            name=self.name if name is None else name,
            refinement_policy=self.refinement_policy,
        )


class InversionBasis:
    """Log-perturbation parameterization on a coarse inversion depth grid.

    m is ordered parameter-major: m[p * n_inv + k] is the log perturbation of
    field param_names[p] at inversion node k. The interpolation operator
    P_node maps inversion-node values to forward-node values:

      - linear interpolation in depth between inversion nodes,
      - constant extension above the shallowest inversion node (the shallow
        structure follows the top node instead of jumping back to reference),
      - linear taper to zero over one node spacing below the deepest node
        (the halfspace stays pinned to the reference model, without a jump).

    The map to a forward model is field = field_ref * exp(P_node @ m_p), so
    positivity is automatic; the AC - F^2 > 0 constraint (eta range) is checked
    separately via physics.check_stiffness_posdef.
    """

    def __init__(self, reference: ForwardGridModel, inv_depths, param_names=VELOCITY_FIELDS):
        self.reference = reference
        self.inv_depths = np.asarray(inv_depths, dtype=float)
        if np.any(np.diff(self.inv_depths) <= 0):
            raise ValueError("inv_depths must be strictly increasing")
        unknown = set(param_names) - set(VELOCITY_FIELDS)
        if unknown:
            raise ValueError(f"unknown parameter names: {sorted(unknown)}")
        self.param_names = tuple(param_names)
        self.P_node = self._build_interpolation(reference.depths, self.inv_depths)

    @property
    def n_inv(self) -> int:
        return int(self.inv_depths.shape[0])

    @property
    def n_params(self) -> int:
        return len(self.param_names) * self.n_inv

    @staticmethod
    def _build_interpolation(fwd_depths: np.ndarray, inv_depths: np.ndarray) -> np.ndarray:
        n_fwd = fwd_depths.shape[0]
        n_inv = inv_depths.shape[0]
        P = np.zeros((n_fwd, n_inv))
        if n_inv == 1:
            taper_lo = inv_depths[0]
            taper_hi = inv_depths[0] + 50.0
        else:
            taper_lo = inv_depths[-1]
            taper_hi = inv_depths[-1] + (inv_depths[-1] - inv_depths[-2])
        for j, z in enumerate(fwd_depths):
            if z <= inv_depths[0]:
                P[j, 0] = 1.0  # constant extension to the surface
            elif z >= taper_hi:
                continue  # pinned to reference below the taper
            elif z > taper_lo:
                P[j, n_inv - 1] = (taper_hi - z) / (taper_hi - taper_lo)
            else:
                k = int(np.searchsorted(inv_depths, z, side="right")) - 1
                k = min(k, n_inv - 2)
                t = (z - inv_depths[k]) / (inv_depths[k + 1] - inv_depths[k])
                P[j, k] = 1.0 - t
                P[j, k + 1] = t
        return P

    def split(self, m: np.ndarray) -> dict:
        m = np.asarray(m, dtype=float)
        if m.shape != (self.n_params,):
            raise ValueError(f"m has shape {m.shape}, expected ({self.n_params},)")
        return {
            name: m[p * self.n_inv : (p + 1) * self.n_inv]
            for p, name in enumerate(self.param_names)
        }

    def m_to_forward_model(self, m: np.ndarray, name: str = None) -> ForwardGridModel:
        out = self.reference.copy(name=name)
        for field_name, m_p in self.split(m).items():
            scale = np.exp(self.P_node @ m_p)
            setattr(out, field_name, getattr(self.reference, field_name) * scale)
        # Assignment after construction bypasses dataclass validation.  Check
        # the completed model so overflow/NaN LM trials are rejected at the
        # model boundary instead of reaching the dispersion kernels.
        out.validate()
        return out

    def describe(self) -> dict:
        """Stable cache/provenance description of the inversion coordinates."""
        return {
            "type": type(self).__name__,
            "param_names": list(self.param_names),
            "inv_depths_km": self.inv_depths.tolist(),
        }

    def full_log_operator(self) -> np.ndarray:
        """Map inversion coordinates to all six forward-node log fields.

        The historical basis only perturbs the fields listed in
        ``param_names``; unselected fields have zero rows.  A constrained basis
        can override this method to express dependent fields while retaining
        the same derivative pullback interface.
        """
        n_fwd = self.P_node.shape[0]
        out = np.zeros((len(VELOCITY_FIELDS) * n_fwd, self.n_params))
        for p, name in enumerate(self.param_names):
            field_index = VELOCITY_FIELDS.index(name)
            out[field_index * n_fwd : (field_index + 1) * n_fwd,
                p * self.n_inv : (p + 1) * self.n_inv] = self.P_node
        return out

    def log_field_operator(self, model: ForwardGridModel = None) -> np.ndarray:
        """Map current inversion coordinates to forward-node log fields.

        Linear bases are state independent.  The optional ``model`` argument
        is accepted so nonlinear constrained bases can evaluate their local
        Jacobian without changing the public pullback API.
        """
        return self.full_log_operator()

    def log_field_curvature_contraction(
        self, model: ForwardGridModel, g_log: np.ndarray
    ) -> np.ndarray:
        """Return the basis-map second-order chain-rule correction.

        ``g_log`` is the gradient with respect to the six forward-node log
        fields.  The historical linear bases have no such correction.
        """
        return np.zeros((self.n_params, self.n_params), dtype=float)

    # -- pullbacks -------------------------------------------------------
    # Forward-node log-space vectors are ordered parameter-major with the
    # same param_names order: index = p * n_fwd + j.

    def full_operator(self) -> np.ndarray:
        """Block-diagonal P over the selected parameters (n_p*n_fwd, n_p*n_inv)."""
        n_p = len(self.param_names)
        n_fwd = self.P_node.shape[0]
        P_full = np.zeros((n_p * n_fwd, n_p * self.n_inv))
        for p in range(n_p):
            P_full[p * n_fwd : (p + 1) * n_fwd, p * self.n_inv : (p + 1) * self.n_inv] = self.P_node
        return P_full

    def pullback_gradient(self, g_log: np.ndarray) -> np.ndarray:
        n_p = len(self.param_names)
        n_fwd = self.P_node.shape[0]
        g_log = np.asarray(g_log, dtype=float).reshape(n_p, n_fwd)
        return np.concatenate([self.P_node.T @ g_log[p] for p in range(n_p)])

    def pullback_hessian(self, H_log: np.ndarray) -> np.ndarray:
        P_full = self.full_operator()
        return P_full.T @ np.asarray(H_log, dtype=float) @ P_full


class ScaledShearBasis(InversionBasis):
    """VTI basis with independent ``vsv``/``vsh`` and scaled dependent fields.

    The independent coordinates are parameter-major ``(vsv, vsh)`` log
    perturbations.  At every forward node, with
    ``m_beta = 0.5 * (m_vsv + m_vsh)``, the dependent fields obey

      dlnrho = rho_beta * m_beta
      dlnvph = dlnvpv = vp_beta * m_beta
      dlneta = 0

    Thus the S-wave anisotropy can vary, while eta and the P-wave anisotropy
    ratio remain fixed to the reference model.
    """

    def __init__(self, reference: ForwardGridModel, inv_depths,
                 rho_beta: float = 0.4, vp_beta: float = 0.6):
        super().__init__(reference, inv_depths, param_names=("vsv", "vsh"))
        self.rho_beta = float(rho_beta)
        self.vp_beta = float(vp_beta)
        if not np.isfinite(self.rho_beta) or not np.isfinite(self.vp_beta):
            raise ValueError("scaling coefficients must be finite")
        self._full_log_operator_cache = None

    def describe(self) -> dict:
        return {
            **super().describe(),
            "type": type(self).__name__,
            "rho_beta": self.rho_beta,
            "vp_beta": self.vp_beta,
            "eta_fixed": True,
            "p_anisotropy_fixed": True,
        }

    def full_log_operator(self) -> np.ndarray:
        if self._full_log_operator_cache is not None:
            return self._full_log_operator_cache.copy()
        n_fwd = self.P_node.shape[0]
        out = np.zeros((len(VELOCITY_FIELDS) * n_fwd, self.n_params))
        vsv = VELOCITY_FIELDS.index("vsv")
        vsh = VELOCITY_FIELDS.index("vsh")
        rho = VELOCITY_FIELDS.index("rho")
        vph = VELOCITY_FIELDS.index("vph")
        vpv = VELOCITY_FIELDS.index("vpv")
        first = slice(0, self.n_inv)
        second = slice(self.n_inv, 2 * self.n_inv)
        out[vsv * n_fwd : (vsv + 1) * n_fwd, first] = self.P_node
        out[vsh * n_fwd : (vsh + 1) * n_fwd, second] = self.P_node
        beta = 0.5 * self.P_node
        for field_index, coefficient in ((rho, self.rho_beta),
                                         (vph, self.vp_beta),
                                         (vpv, self.vp_beta)):
            block = out[field_index * n_fwd : (field_index + 1) * n_fwd]
            block[:, first] = coefficient * beta
            block[:, second] = coefficient * beta
        self._full_log_operator_cache = out
        return out.copy()

    def m_to_forward_model(self, m: np.ndarray, name: str = None) -> ForwardGridModel:
        m = np.asarray(m, dtype=float)
        if m.shape != (self.n_params,):
            raise ValueError(f"m has shape {m.shape}, expected ({self.n_params},)")
        out = self.reference.copy(name=name)
        n_fwd = self.P_node.shape[0]
        log_fields = self.full_log_operator() @ m
        for field_index, field_name in enumerate(VELOCITY_FIELDS):
            setattr(
                out,
                field_name,
                getattr(self.reference, field_name)
                * np.exp(log_fields[field_index * n_fwd : (field_index + 1) * n_fwd]),
            )
        # This override constructs the fields directly, so explicitly repeat
        # the finite/positivity validation performed by the base mapping.
        out.validate()
        return out


def _log_derivative_pair(value, first, second):
    """Convert first/second derivatives of a positive quantity to log form."""
    value = np.asarray(value, dtype=float)
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("scaling law produced non-positive or non-finite values")
    d1 = first / value
    d2 = second / value - d1 * d1
    if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(d2)):
        raise ValueError("scaling law produced non-finite logarithmic derivatives")
    return d1, d2


@dataclass(frozen=True)
class Brocher2005CrustLaw:
    """Brocher (2005) crustal ``Vp(Vs)`` and ``rho(Vp)`` scaling.

    Velocities are in km/s and density is in g/cm^3.  The maps are anchored
    to the supplied reference point, so a zero perturbation reproduces the
    reference model exactly.  The direct ``Vp(Vs)`` polynomial is used rather
    than numerically inverting the published ``Vs(Vp)`` fit.
    """

    name: str = "brocher2005"
    vs_min: float = 0.0
    vs_max: float = 4.5

    @staticmethod
    def vp_from_vs(vs):
        vs = np.asarray(vs, dtype=float)
        return (
            0.9409
            + 2.0947 * vs
            - 0.8206 * vs**2
            + 0.2683 * vs**3
            - 0.0251 * vs**4
        )

    @staticmethod
    def vp_from_vs_first(vs):
        vs = np.asarray(vs, dtype=float)
        return 2.0947 - 2.0 * 0.8206 * vs + 3.0 * 0.2683 * vs**2 - 4.0 * 0.0251 * vs**3

    @staticmethod
    def vp_from_vs_second(vs):
        vs = np.asarray(vs, dtype=float)
        return -2.0 * 0.8206 + 6.0 * 0.2683 * vs - 12.0 * 0.0251 * vs**2

    @staticmethod
    def rho_from_vp(vp):
        vp = np.asarray(vp, dtype=float)
        return (
            1.6612 * vp
            - 0.4721 * vp**2
            + 0.0671 * vp**3
            - 0.0043 * vp**4
            + 0.000106 * vp**5
        )

    @staticmethod
    def rho_from_vp_first(vp):
        vp = np.asarray(vp, dtype=float)
        return 1.6612 - 2.0 * 0.4721 * vp + 3.0 * 0.0671 * vp**2 - 4.0 * 0.0043 * vp**3 + 5.0 * 0.000106 * vp**4

    @staticmethod
    def rho_from_vp_second(vp):
        vp = np.asarray(vp, dtype=float)
        return -2.0 * 0.4721 + 6.0 * 0.0671 * vp - 12.0 * 0.0043 * vp**2 + 20.0 * 0.000106 * vp**3

    def evaluate(self, shear, shear_ref, vp_ref, rho_ref):
        shear = np.asarray(shear, dtype=float)
        shear_ref = np.asarray(shear_ref, dtype=float)
        vp_ref = np.asarray(vp_ref, dtype=float)
        rho_ref = np.asarray(rho_ref, dtype=float)
        if np.any(~np.isfinite(shear)) or np.any(shear <= self.vs_min) or np.any(shear > self.vs_max):
            raise ValueError(
                f"Brocher 2005 Vp(Vs) is outside its supported Vs range "
                f"({self.vs_min:g}, {self.vs_max:g}] km/s"
            )
        p_raw = self.vp_from_vs(shear)
        dp_du = shear * self.vp_from_vs_first(shear)
        d2p_du2 = dp_du + shear**2 * self.vp_from_vs_second(shear)
        vp = vp_ref + p_raw - self.vp_from_vs(shear_ref)
        vp_d1, vp_d2 = _log_derivative_pair(vp, dp_du, d2p_du2)

        # Apply rho(Vp) to the anchored Vp itself.  Subtracting the polynomial
        # at ``vp_ref`` preserves an arbitrary starting model exactly at zero
        # perturbation, even when it does not lie on the Brocher fit.
        rho = rho_ref + self.rho_from_vp(vp) - self.rho_from_vp(vp_ref)
        drho_du = self.rho_from_vp_first(vp) * dp_du
        d2rho_du2 = (
            self.rho_from_vp_second(vp) * dp_du**2
            + self.rho_from_vp_first(vp) * d2p_du2
        )
        rho_d1, rho_d2 = _log_derivative_pair(rho, drho_du, d2rho_du2)
        return {
            "vp": vp,
            "rho": rho,
            "log_vp_d1": vp_d1,
            "log_vp_d2": vp_d2,
            "log_rho_d1": rho_d1,
            "log_rho_d2": rho_d2,
        }

    def describe(self):
        return {"type": type(self).__name__, "name": self.name, "vs_range_km_s": [self.vs_min, self.vs_max]}


@dataclass(frozen=True)
class Masters2000MantleLaw:
    """Anchored mantle scaling with Masters et al. (2000) density law.

    ``vp_slope`` is the explicitly configured absolute ``dVp/dVs`` slope in
    the mantle.  Density follows ``rho/rho0 = (Vs/Vs0)**density_exponent``;
    the default exponent 0.4 corresponds to ``d ln Vs / d ln rho = 2.5``.
    """

    vp_slope: float
    density_exponent: float = 0.4
    name: str = "masters2000_log_density"

    def __post_init__(self):
        if not np.isfinite(self.vp_slope):
            raise ValueError("vp_slope must be finite")
        if not np.isfinite(self.density_exponent):
            raise ValueError("density_exponent must be finite")

    def evaluate(self, shear, shear_ref, vp_ref, rho_ref):
        shear = np.asarray(shear, dtype=float)
        shear_ref = np.asarray(shear_ref, dtype=float)
        vp_ref = np.asarray(vp_ref, dtype=float)
        rho_ref = np.asarray(rho_ref, dtype=float)
        if np.any(~np.isfinite(shear)) or np.any(shear <= 0.0) or np.any(shear_ref <= 0.0):
            raise ValueError("mantle scaling requires positive finite shear velocities")
        vp = vp_ref + self.vp_slope * (shear - shear_ref)
        dp_du = self.vp_slope * shear
        d2p_du2 = dp_du
        vp_d1, vp_d2 = _log_derivative_pair(vp, dp_du, d2p_du2)
        rho = rho_ref * (shear / shear_ref) ** self.density_exponent
        if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
            raise ValueError("mantle density scaling produced non-positive values")
        return {
            "vp": vp,
            "rho": rho,
            "log_vp_d1": vp_d1,
            "log_vp_d2": vp_d2,
            "log_rho_d1": np.full_like(shear, self.density_exponent),
            "log_rho_d2": np.zeros_like(shear),
        }

    def describe(self):
        return {
            "type": type(self).__name__,
            "name": self.name,
            "vp_slope_dvp_dvs": self.vp_slope,
            "density_exponent_dlnrho_dlnvs": self.density_exponent,
        }


@dataclass(frozen=True)
class MohoRegionMap:
    """Sharp crust/mantle node partition defined by a duplicated Moho pair."""

    pair_start: int
    depth_tolerance_km: float = 1e-10

    def validate(self, reference: ForwardGridModel):
        pair = int(self.pair_start)
        if pair < 0 or pair + 1 >= reference.n_nodes:
            raise IndexError("Moho pair_start is outside the reference grid")
        if not reference.is_discontinuity[pair]:
            raise ValueError("Moho pair_start must flag the first node of a discontinuity")
        if abs(float(reference.depths[pair + 1] - reference.depths[pair])) > self.depth_tolerance_km:
            raise ValueError("Moho discontinuity must be represented by a duplicated depth pair")

    def masks(self, reference: ForwardGridModel):
        self.validate(reference)
        indices = np.arange(reference.n_nodes)
        return indices <= int(self.pair_start), indices >= int(self.pair_start) + 1

    def describe(self, reference: ForwardGridModel):
        self.validate(reference)
        return {
            "pair_start": int(self.pair_start),
            "depth_km": float(reference.depths[self.pair_start]),
            "depth_tolerance_km": self.depth_tolerance_km,
            "switch": "sharp_duplicate_pair",
        }


@dataclass(frozen=True)
class RegionalScalingProfile:
    """Regional crust/mantle scaling laws with a sharp Moho switch."""

    moho: MohoRegionMap
    # Laws are duck-typed so a project-specific implementation can be supplied
    # without changing the inversion basis.  It must provide ``evaluate`` and
    # ``describe`` with the same contract as the built-ins below.
    crust_law: object
    mantle_law: object

    def validate(self, reference: ForwardGridModel):
        self.moho.validate(reference)
        for name, law in (("crust_law", self.crust_law), ("mantle_law", self.mantle_law)):
            if not callable(getattr(law, "evaluate", None)) or not callable(getattr(law, "describe", None)):
                raise TypeError(f"{name} must provide evaluate(...) and describe()")

    def evaluate(self, reference: ForwardGridModel, vsv, vsh):
        self.validate(reference)
        vsv = np.asarray(vsv, dtype=float)
        vsh = np.asarray(vsh, dtype=float)
        if vsv.shape != (reference.n_nodes,) or vsh.shape != (reference.n_nodes,):
            raise ValueError("regional scaling shear arrays must match reference.n_nodes")
        if np.any(~np.isfinite(vsv)) or np.any(~np.isfinite(vsh)) or np.any(vsv <= 0.0) or np.any(vsh <= 0.0):
            raise ValueError("regional scaling requires positive finite Vsv and Vsh")
        shear = np.sqrt(vsv * vsh)
        shear_ref = np.sqrt(reference.vsv * reference.vsh)
        vp_ref = np.sqrt(reference.vph * reference.vpv)
        vp = np.empty(reference.n_nodes)
        rho = np.empty(reference.n_nodes)
        vp_d1 = np.empty(reference.n_nodes)
        vp_d2 = np.empty(reference.n_nodes)
        rho_d1 = np.empty(reference.n_nodes)
        rho_d2 = np.empty(reference.n_nodes)
        crust, mantle = self.moho.masks(reference)
        for mask, law in ((crust, self.crust_law), (mantle, self.mantle_law)):
            values = law.evaluate(shear[mask], shear_ref[mask], vp_ref[mask], reference.rho[mask])
            vp[mask] = values["vp"]
            rho[mask] = values["rho"]
            vp_d1[mask] = values["log_vp_d1"]
            vp_d2[mask] = values["log_vp_d2"]
            rho_d1[mask] = values["log_rho_d1"]
            rho_d2[mask] = values["log_rho_d2"]
        return {
            "shear": shear,
            "vp": vp,
            "rho": rho,
            "log_vp_d1": vp_d1,
            "log_vp_d2": vp_d2,
            "log_rho_d1": rho_d1,
            "log_rho_d2": rho_d2,
        }

    def describe(self, reference: ForwardGridModel):
        self.validate(reference)
        return {
            "type": type(self).__name__,
            "moho": self.moho.describe(reference),
            "crust_law": self.crust_law.describe(),
            "mantle_law": self.mantle_law.describe(),
            "switch_continuity": "not_enforced_sharp_interface",
        }


class RegionalScaledShearBasis(InversionBasis):
    """Two-parameter shear basis with crust/mantle regional scaling.

    ``Vsv`` and ``Vsh`` are independent.  Their geometric-mean shear speed
    controls common P-wave scaling and density, while the reference P-wave
    anisotropy ratio and ``eta`` remain fixed.  The local operator and the
    basis-map curvature are evaluated at the current model for AD/variational
    Hessian pullbacks.
    """

    def __init__(self, reference: ForwardGridModel, inv_depths, profile: RegionalScalingProfile):
        super().__init__(reference, inv_depths, param_names=("vsv", "vsh"))
        if not isinstance(profile, RegionalScalingProfile):
            raise TypeError("profile must be a RegionalScalingProfile")
        profile.validate(reference)
        self.profile = profile

    def describe(self):
        return {**super().describe(), "type": type(self).__name__, "profile": self.profile.describe(self.reference), "eta_fixed": True, "p_anisotropy_fixed": True}

    def _profile_values(self, model):
        if model is None:
            model = self.reference
        if model.n_nodes != self.reference.n_nodes:
            raise ValueError("model node count does not match the basis reference")
        return self.profile.evaluate(self.reference, model.vsv, model.vsh)

    def log_field_operator(self, model: ForwardGridModel = None) -> np.ndarray:
        values = self._profile_values(model)
        n_fwd = self.reference.n_nodes
        out = np.zeros((len(VELOCITY_FIELDS) * n_fwd, self.n_params))
        first = slice(0, self.n_inv)
        second = slice(self.n_inv, 2 * self.n_inv)
        vsv = VELOCITY_FIELDS.index("vsv")
        vsh = VELOCITY_FIELDS.index("vsh")
        rho = VELOCITY_FIELDS.index("rho")
        vph = VELOCITY_FIELDS.index("vph")
        vpv = VELOCITY_FIELDS.index("vpv")
        out[vsv * n_fwd : (vsv + 1) * n_fwd, first] = self.P_node
        out[vsh * n_fwd : (vsh + 1) * n_fwd, second] = self.P_node
        for field_index, dlog_du in ((rho, values["log_rho_d1"]), (vph, values["log_vp_d1"]), (vpv, values["log_vp_d1"])):
            block = out[field_index * n_fwd : (field_index + 1) * n_fwd]
            q = 0.5 * dlog_du[:, None] * self.P_node
            block[:, first] = q
            block[:, second] = q
        return out

    def full_log_operator(self) -> np.ndarray:
        return self.log_field_operator(self.reference)

    def log_field_curvature_contraction(self, model: ForwardGridModel, g_log: np.ndarray) -> np.ndarray:
        values = self._profile_values(model)
        g = np.asarray(g_log, dtype=float).reshape(len(VELOCITY_FIELDS), self.reference.n_nodes)
        out = np.zeros((self.n_params, self.n_params), dtype=float)
        first = slice(0, self.n_inv)
        second = slice(self.n_inv, 2 * self.n_inv)
        q = 0.5 * self.P_node
        for field_name, d2log_du2 in (("rho", values["log_rho_d2"]), ("vph", values["log_vp_d2"]), ("vpv", values["log_vp_d2"])):
            weight = g[VELOCITY_FIELDS.index(field_name)] * d2log_du2
            block = np.einsum("i,ij,ik->jk", weight, q, q, optimize=True)
            out[first, first] += block
            out[first, second] += block
            out[second, first] += block
            out[second, second] += block
        return out

    def m_to_forward_model(self, m: np.ndarray, name: str = None) -> ForwardGridModel:
        m = np.asarray(m, dtype=float)
        if m.shape != (self.n_params,):
            raise ValueError(f"m has shape {m.shape}, expected ({self.n_params},)")
        out = self.reference.copy(name=name)
        m_parts = self.split(m)
        out.vsv = self.reference.vsv * np.exp(self.P_node @ m_parts["vsv"])
        out.vsh = self.reference.vsh * np.exp(self.P_node @ m_parts["vsh"])
        values = self.profile.evaluate(self.reference, out.vsv, out.vsh)
        vp_ref = np.sqrt(self.reference.vph * self.reference.vpv)
        vp_factor = values["vp"] / vp_ref
        out.vph = self.reference.vph * vp_factor
        out.vpv = self.reference.vpv * vp_factor
        out.rho = values["rho"]
        out.eta = self.reference.eta.copy()
        out.validate()
        return out


class ActiveSubspaceBasis:
    """Restrict an inversion basis to an active parameter subspace.

    The wrapped basis still defines the full forward model and interpolation
    grid.  ``active_indices`` selects the coordinates that may vary; all
    other full-basis coordinates are held at zero.  This is used for
    depth-fixed comparisons so that the forward parameterization is not
    silently replaced by a different grid or scaling law.
    """

    def __init__(self, basis: InversionBasis, active_indices):
        self.base_basis = basis
        self.reference = basis.reference
        self.inv_depths = basis.inv_depths.copy()
        self.P_node = basis.P_node
        self.param_names = basis.param_names
        active = np.asarray(active_indices, dtype=int)
        if active.ndim != 1 or active.size == 0:
            raise ValueError("active_indices must be a non-empty one-dimensional array")
        if np.any(active < 0) or np.any(active >= basis.n_params):
            raise IndexError("active_indices contains an out-of-range parameter")
        if np.unique(active).size != active.size:
            raise ValueError("active_indices must be unique")
        self.active_indices = active.copy()

    @property
    def n_inv(self) -> int:
        return self.base_basis.n_inv

    @property
    def n_params(self) -> int:
        return int(self.active_indices.size)

    def expand(self, m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=float)
        if m.shape != (self.n_params,):
            raise ValueError(f"m has shape {m.shape}, expected ({self.n_params},)")
        full = np.zeros(self.base_basis.n_params, dtype=float)
        full[self.active_indices] = m
        return full

    def restrict(self, m_full: np.ndarray) -> np.ndarray:
        m_full = np.asarray(m_full, dtype=float)
        if m_full.shape != (self.base_basis.n_params,):
            raise ValueError(
                f"m_full has shape {m_full.shape}, expected ({self.base_basis.n_params},)"
            )
        return m_full[self.active_indices].copy()

    def split(self, m: np.ndarray) -> dict:
        return self.base_basis.split(self.expand(m))

    def m_to_forward_model(self, m: np.ndarray, name: str = None) -> ForwardGridModel:
        return self.base_basis.m_to_forward_model(self.expand(m), name=name)

    def full_log_operator(self) -> np.ndarray:
        return self.base_basis.full_log_operator()[:, self.active_indices]

    def log_field_operator(self, model: ForwardGridModel = None) -> np.ndarray:
        return self.base_basis.log_field_operator(model)[:, self.active_indices]

    def log_field_curvature_contraction(self, model: ForwardGridModel, g_log: np.ndarray) -> np.ndarray:
        correction = self.base_basis.log_field_curvature_contraction(model, g_log)
        return correction[np.ix_(self.active_indices, self.active_indices)]

    def full_operator(self) -> np.ndarray:
        return self.base_basis.full_operator()[:, self.active_indices]

    def describe(self) -> dict:
        return {
            "type": type(self).__name__,
            "base_basis": self.base_basis.describe(),
            "active_indices": self.active_indices.tolist(),
            "n_params": self.n_params,
        }


def shift_discontinuity(
    model: ForwardGridModel,
    pair_start: int,
    delta_km: float,
    min_spacing_km: float = 1e-6,
) -> ForwardGridModel:
    """Return a copy with one duplicated interface shifted in depth.

    ``pair_start`` is the first node of a duplicated discontinuity pair.  The
    two depths move together, while all material values and the discontinuity
    flag remain unchanged.  This is a model-family helper for explicit
    interface-depth experiments.  The GRID C++ kernel path computes ``Depth``
    kernels; the depth-derivative backend maps those entries separately from
    the velocity pullback.
    """
    pair_start = int(pair_start)
    if pair_start < 0 or pair_start + 1 >= model.n_nodes:
        raise IndexError("pair_start is outside the model grid")
    if not model.is_discontinuity[pair_start]:
        raise ValueError("pair_start must be flagged as the first node of a discontinuity")
    z0 = float(model.depths[pair_start])
    if abs(float(model.depths[pair_start + 1]) - z0) > 1e-10:
        raise ValueError("discontinuity flag does not identify a duplicated depth pair")
    new_depth = z0 + float(delta_km)
    lower_bound = -np.inf if pair_start == 0 else float(model.depths[pair_start - 1]) + min_spacing_km
    upper_bound = (
        np.inf if pair_start + 2 >= model.n_nodes
        else float(model.depths[pair_start + 2]) - min_spacing_km
    )
    if not lower_bound < new_depth < upper_bound:
        raise ValueError(
            f"shifted interface depth {new_depth:g} km violates neighboring grid bounds "
            f"({lower_bound:g}, {upper_bound:g})"
        )
    out = model.copy(name=f"{model.name}_shift_{delta_km:+g}km")
    out.depths[pair_start : pair_start + 2] = new_depth
    return out


class InterfaceShiftBasis:
    """Augment a velocity basis with one explicit interface-depth coordinate.

    The first ``base_basis.n_params`` coordinates are unchanged log-velocity
    parameters.  The final coordinate is an interface shift in km.  The
    ``full_log_operator`` deliberately contains a zero column for this depth
    coordinate.  A depth-Hessian backend appends the ``Depth`` kernel to the
    velocity pullback and supplies the depth Hessian column separately.
    """

    def __init__(self, base_basis: InversionBasis, pair_start: int):
        if not base_basis.reference.is_discontinuity[int(pair_start)]:
            raise ValueError("pair_start must identify a reference discontinuity")
        self.base_basis = base_basis
        self.reference = base_basis.reference
        self.pair_start = int(pair_start)
        self.inv_depths = base_basis.inv_depths.copy()
        self.P_node = base_basis.P_node
        self.param_names = base_basis.param_names

    @property
    def n_inv(self) -> int:
        return self.base_basis.n_inv

    @property
    def n_params(self) -> int:
        return self.base_basis.n_params + 1

    @property
    def velocity_n_params(self) -> int:
        return self.base_basis.n_params

    @property
    def depth_parameter_index(self) -> int:
        return self.velocity_n_params

    def depth_shift_bounds(self, min_spacing_km: float = 1e-6) -> tuple[float, float]:
        """Feasible lower/upper bounds for the interface shift in km."""
        z0 = float(self.reference.depths[self.pair_start])
        lower = (
            -np.inf
            if self.pair_start == 0
            else float(self.reference.depths[self.pair_start - 1])
            + min_spacing_km
            - z0
        )
        upper = (
            np.inf
            if self.pair_start + 2 >= self.reference.n_nodes
            else float(self.reference.depths[self.pair_start + 2])
            - min_spacing_km
            - z0
        )
        return lower, upper

    def project_feasible(self, m: np.ndarray) -> np.ndarray:
        """Clip only the explicit depth coordinate to physical grid bounds."""
        m = np.asarray(m, dtype=float).copy()
        if m.shape != (self.n_params,):
            raise ValueError(f"m has shape {m.shape}, expected ({self.n_params},)")
        lower, upper = self.depth_shift_bounds()
        m[self.depth_parameter_index] = np.clip(
            m[self.depth_parameter_index], lower, upper
        )
        return m

    def split(self, m: np.ndarray) -> dict:
        m = np.asarray(m, dtype=float)
        if m.shape != (self.n_params,):
            raise ValueError(f"m has shape {m.shape}, expected ({self.n_params},)")
        out = self.base_basis.split(m[: self.velocity_n_params])
        out["interface_shift_km"] = float(m[-1])
        return out

    def m_to_forward_model(self, m: np.ndarray, name: str = None) -> ForwardGridModel:
        m = np.asarray(m, dtype=float)
        if m.shape != (self.n_params,):
            raise ValueError(f"m has shape {m.shape}, expected ({self.n_params},)")
        out = self.base_basis.m_to_forward_model(m[: self.velocity_n_params], name=name)
        return shift_discontinuity(out, self.pair_start, m[-1])

    def full_log_operator(self) -> np.ndarray:
        velocity_operator = self.base_basis.full_log_operator()
        return np.column_stack([velocity_operator, np.zeros(velocity_operator.shape[0])])

    def log_field_operator(self, model: ForwardGridModel = None) -> np.ndarray:
        velocity_operator = self.base_basis.log_field_operator(model)
        return np.column_stack([velocity_operator, np.zeros(velocity_operator.shape[0])])

    def log_field_curvature_contraction(self, model: ForwardGridModel, g_log: np.ndarray) -> np.ndarray:
        base = self.base_basis.log_field_curvature_contraction(model, g_log)
        out = np.zeros((self.n_params, self.n_params), dtype=float)
        out[: self.velocity_n_params, : self.velocity_n_params] = base
        return out

    def describe(self) -> dict:
        return {
            "type": type(self).__name__,
            "base_basis": self.base_basis.describe(),
            "pair_start": self.pair_start,
            "interface_depth_km": float(self.reference.depths[self.pair_start]),
            "depth_derivative": "grid_ad_kernel",
            "n_params": self.n_params,
        }
