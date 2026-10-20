"""Anisotropic PREM loading with the earth-flattening transformation.

Source CSV: tests/model_PREM_aniso.csv, columns
radius[m], density[kg/m^3], Vpv[m/s], Vsv[m/s], Q-kappa, Q-mu, Vph[m/s], Vsh[m/s], eta
ordered from the Earth's center outward. Ocean (Vs=0) nodes are dropped, so the
returned model is the ocean-free PREM used as reference/prior center.

Flattening follows the convention of tests/python/verify_hessian_taylor_random.py:
flat depth R*ln(R/r), velocity scale R/r, density scale (r/R)^-2.275; eta is a
dimensionless stiffness ratio and is left unscaled.

Ocean handling (ocean_mode):
  "crust_fill" (default) - extend the uppermost crust to r=6371 km, matching the
      standard PREM-noocean convention (tests/reference/mineos/DEMO/models/
      prem_noocean.txt).
  "drop" - simply remove fluid rows; the free surface sits at the seafloor.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .model import VELOCITY_FIELDS, ForwardGridModel, GridRefinementPolicy

EARTH_RADIUS_KM = 6371.0
DENSITY_EXPONENT = -2.275
DEFAULT_PREM_CSV = Path(__file__).resolve().parents[2] / "tests" / "model_PREM_aniso.csv"


def load_prem_flat(
    csv_path: Path = DEFAULT_PREM_CSV,
    max_depth_km: float = 500.0,
    name: str = "prem_noocean_flat",
    density_exponent: float = DENSITY_EXPONENT,
    max_spacing_km: float = 10.0,
    ocean_mode: str = "crust_fill",
    _model_space: str = "flat",
) -> ForwardGridModel:
    if ocean_mode not in ("crust_fill", "drop"):
        raise ValueError("ocean_mode must be 'crust_fill' or 'drop'")
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            try:
                radius_m = float(row[0])
            except ValueError:
                continue
            vsv = float(row[3]) / 1000.0
            if vsv <= 0.0:
                continue  # ocean / fluid layers
            radius_km = radius_m / 1000.0
            rows.append(
                {
                    "radius_km": radius_km,
                    "depth_km": EARTH_RADIUS_KM - radius_km,
                    "rho": float(row[1]) / 1000.0,
                    "vpv": float(row[2]) / 1000.0,
                    "vsv": vsv,
                    "vph": float(row[6]) / 1000.0,
                    "vsh": float(row[7]) / 1000.0,
                    "eta": float(row[8]),
                }
            )

    # The CSV runs center -> surface, so at a discontinuity (duplicated radius)
    # the deeper-side row comes first. Reverse before the stable sort so that
    # in depth-ascending order the shallow-side value precedes the deep-side one.
    rows.reverse()
    rows.sort(key=lambda r: r["depth_km"])
    if not rows:
        raise RuntimeError(f"no solid nodes found in {csv_path}")

    if ocean_mode == "crust_fill" and rows[0]["depth_km"] > 1e-9:
        surface = dict(rows[0])
        surface["radius_km"] = EARTH_RADIUS_KM
        surface["depth_km"] = 0.0
        rows.insert(0, surface)

    top_depth = rows[0]["depth_km"]
    trimmed = [r for r in rows if r["depth_km"] - top_depth <= max_depth_km + 1e-9]
    if len(trimmed) < 2:
        raise RuntimeError("PREM grid needs at least two solid nodes in the requested depth range")

    # Radial refinement before flattening: the CSV samples PREM every ~20-45 km,
    # while the flattening scales velocities by R/r (exponential in flat depth).
    # Linear interpolation of the raw fields in radius down to max_spacing_km
    # keeps the model-discretization error well below the solver tolerance.
    if max_spacing_km is not None and max_spacing_km > 0.0:
        refined = []
        fields = ("rho", "vpv", "vsv", "vph", "vsh", "eta")
        for a, b in zip(trimmed, trimmed[1:]):
            refined.append(a)
            gap = a["radius_km"] - b["radius_km"]
            if gap > max_spacing_km:
                n_sub = int(math.ceil(gap / max_spacing_km))
                for k in range(1, n_sub):
                    t = k / n_sub
                    node = {
                        "radius_km": a["radius_km"] - t * gap,
                    }
                    node["depth_km"] = EARTH_RADIUS_KM - node["radius_km"]
                    for key in fields:
                        node[key] = (1.0 - t) * a[key] + t * b[key]
                    refined.append(node)
        refined.append(trimmed[-1])
        trimmed = refined

    if _model_space not in ("flat", "spherical"):
        raise ValueError("_model_space must be 'flat' or 'spherical'")

    if _model_space == "spherical":
        depths = np.asarray([r["depth_km"] for r in trimmed], dtype=float)
        is_disc = np.zeros(depths.shape[0], dtype=bool)
        for i in range(depths.shape[0] - 1):
            if abs(depths[i] - depths[i + 1]) < 1e-12:
                is_disc[i] = True
        return ForwardGridModel(
            depths=depths,
            rho=np.asarray([r["rho"] for r in trimmed], dtype=float),
            vph=np.asarray([r["vph"] for r in trimmed], dtype=float),
            vpv=np.asarray([r["vpv"] for r in trimmed], dtype=float),
            vsh=np.asarray([r["vsh"] for r in trimmed], dtype=float),
            vsv=np.asarray([r["vsv"] for r in trimmed], dtype=float),
            eta=np.asarray([r["eta"] for r in trimmed], dtype=float),
            is_discontinuity=is_disc,
            name=name,
            refinement_policy=GridRefinementPolicy(
                kind="spherical_experimental_density",
                planet_radius_km=EARTH_RADIUS_KM,
                flattening_exponent=density_exponent,
            ),
        )

    flat_abs = [EARTH_RADIUS_KM * math.log(EARTH_RADIUS_KM / r["radius_km"]) for r in trimmed]
    origin = flat_abs[0]
    depths = np.asarray([d - origin for d in flat_abs], dtype=float)

    def scaled(key: str, kind: str) -> np.ndarray:
        values = []
        for r in trimmed:
            if kind == "velocity":
                values.append(r[key] * EARTH_RADIUS_KM / r["radius_km"])
            elif kind == "density":
                values.append(r[key] * (r["radius_km"] / EARTH_RADIUS_KM) ** density_exponent)
            else:
                values.append(r[key])
        return np.asarray(values, dtype=float)

    is_disc = np.zeros(depths.shape[0], dtype=bool)
    for i in range(depths.shape[0] - 1):
        if abs(depths[i] - depths[i + 1]) < 1e-12:
            is_disc[i] = True

    return ForwardGridModel(
        depths=depths,
        rho=scaled("rho", "density"),
        vph=scaled("vph", "velocity"),
        vpv=scaled("vpv", "velocity"),
        vsh=scaled("vsh", "velocity"),
        vsv=scaled("vsv", "velocity"),
        eta=scaled("eta", "plain"),
        is_discontinuity=is_disc,
        name=name,
        refinement_policy=GridRefinementPolicy(kind="flat"),
    )


def load_prem_spherical(
    csv_path: Path = DEFAULT_PREM_CSV,
    max_depth_km: float = 500.0,
    name: str = "prem_noocean_spherical",
    density_exponent: float = DENSITY_EXPONENT,
    max_spacing_km: float = 10.0,
    ocean_mode: str = "crust_fill",
) -> ForwardGridModel:
    """Load PREM as spherical-depth, unflattened node values.

    The forward Grid binding applies the selected flattening convention at
    each refined sublayer midpoint.  The default density exponent matches the
    historical PREM loader and is recorded in the model metadata.
    """
    return load_prem_flat(
        csv_path=csv_path,
        max_depth_km=max_depth_km,
        name=name,
        density_exponent=density_exponent,
        max_spacing_km=max_spacing_km,
        ocean_mode=ocean_mode,
        _model_space="spherical",
    )


def refine_forward_grid(
    model: ForwardGridModel,
    extra_depths_km,
    min_spacing_km: float = 1e-6,
) -> ForwardGridModel:
    """Insert extra forward nodes at the given depths by linear interpolation.

    Used to give a coarse ``load_prem_flat`` grid (uniform ``max_spacing_km``
    everywhere) finer resolution only where a synthetic anomaly needs it (e.g.
    a shallow LVZ core that the standard 15 km grid does not sample -- see
    tests/REPORT_INV_TOBA_CAPACITY_20260720.md), without changing the
    inversion basis or refining the whole 1000 km column. Fields are
    interpolated directly in flat-depth space (not re-derived from the raw
    PREM radius profile like ``load_prem_flat``'s own refinement): over a few
    km near the surface the flattening scale R/r changes by <<1%, well below
    solver tolerance, so this is an adequate approximation.

    A candidate depth is skipped (not inserted) if it falls within
    ``min_spacing_km`` of an existing node, or if it does not land strictly
    inside a single interpolation interval (e.g. between the two zero-width
    sides of a discontinuity pair) -- both are silently safe no-ops rather
    than errors, so callers can pass a dense candidate list without special-
    casing existing nodes/discontinuities.
    """
    depths = model.depths
    extra = np.atleast_1d(np.asarray(extra_depths_km, dtype=float))
    new_depths, new_fields = [], {name: [] for name in VELOCITY_FIELDS}
    for z in sorted(extra):
        if np.any(np.abs(depths - z) < min_spacing_km):
            continue
        idx = np.searchsorted(depths, z)
        if idx <= 0 or idx >= depths.shape[0]:
            continue
        lo, hi = depths[idx - 1], depths[idx]
        if not (lo < z < hi):
            continue
        t = (z - lo) / (hi - lo)
        new_depths.append(z)
        for name in VELOCITY_FIELDS:
            arr = getattr(model, name)
            new_fields[name].append((1.0 - t) * arr[idx - 1] + t * arr[idx])
    if not new_depths:
        return model.copy()

    all_depths = np.concatenate([depths, new_depths])
    order = np.argsort(all_depths, kind="stable")
    fields_out = {}
    for name in VELOCITY_FIELDS:
        combined = np.concatenate([getattr(model, name), new_fields[name]])
        fields_out[name] = combined[order]
    is_disc_out = np.concatenate(
        [model.is_discontinuity, np.zeros(len(new_depths), dtype=bool)]
    )[order]

    return ForwardGridModel(
        depths=all_depths[order],
        is_discontinuity=is_disc_out,
        name=f"{model.name}_refined",
        refinement_policy=model.refinement_policy,
        **fields_out,
    )
