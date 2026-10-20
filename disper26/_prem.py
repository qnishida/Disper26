"""Packaged, ocean-free anisotropic PREM reference model.

The CSV is an unchanged copy of tests/model_PREM_aniso.csv. Refinement follows
the existing inversion loader: interpolate radial velocity fields before EFT,
preserve duplicate-radius ordering, and extend the top crust to the surface.
"""
from __future__ import annotations

import csv
from importlib import resources
import math
from pathlib import Path
from typing import Optional, Union

from disper26py import _api


def load_prem_spherical(*, interpolation: str = "stiffness", max_depth_km: float = 500.0,
                        max_spacing_km: Optional[float] = 10.0,
                        density_exponent: float = -2.275,
                        csv_path: Optional[Union[str, Path]] = None,
                        name: str = "prem_noocean_spherical") -> _api.Model:
    """Load unflattened solid PREM, with the ocean replaced by upper crust.

    Interpolate raw radial velocity fields to max_spacing_km before constructing
    the model. The final retained source node at or above max_depth_km supplies
    the halfspace; the function does not interpolate an extra bottom node.
    density_exponent selects the spherical refinement/EFT convention explicitly.
    The default CSV is an installed package resource, independent of cwd.
    """
    if not math.isfinite(max_depth_km) or max_depth_km <= 0 or max_depth_km >= 6371:
        raise ValueError("max_depth_km must be finite and between 0 and 6371")
    if max_spacing_km is not None and (not math.isfinite(max_spacing_km) or max_spacing_km <= 0):
        raise ValueError("max_spacing_km must be positive and finite, or None")
    if not math.isfinite(density_exponent):
        raise ValueError("density_exponent must be finite")
    rows = []
    # open_text is available on every supported Python (>=3.8).
    stream = (Path(csv_path).open(encoding="utf-8") if csv_path is not None
              else resources.open_text("disper26.data", "model_PREM_aniso.csv", encoding="utf-8"))
    with stream:
        for record in csv.reader(stream):
            if not record:
                continue
            try:
                radius = float(record[0]) / 1000.0
            except ValueError:
                continue
            vsv = float(record[3]) / 1000.0
            if vsv <= 0:
                continue
            rows.append((6371.0 - radius, float(record[1]) / 1000.0,
                         float(record[6]) / 1000.0, float(record[2]) / 1000.0,
                         float(record[7]) / 1000.0, vsv, float(record[8])))
    rows.reverse()  # Shallow side precedes deep side at duplicate depths.
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise ValueError("PREM source contains no solid nodes")
    if rows[0][0] > 0:
        rows.insert(0, (0.0, *rows[0][1:]))
    rows = [row for row in rows if row[0] <= max_depth_km + 1e-9]
    if len(rows) < 2:
        raise ValueError("PREM needs at least two source nodes in the requested depth range")
    if max_spacing_km is not None:
        refined = []
        for a, b in zip(rows, rows[1:]):
            refined.append(a)
            subdivisions = max(1, int(math.ceil((b[0] - a[0]) / max_spacing_km)))
            for k in range(1, subdivisions):
                t = k / subdivisions
                refined.append(tuple((1.0 - t) * x + t * y for x, y in zip(a, b)))
        refined.append(rows[-1])
        rows = refined
    return _api.create_grid_model(
        [row[0] for row in rows], [_api.create_velocity_vti_parameters(*row[1:]) for row in rows],
        interpolation, _api.Geometry("spherical_experimental_density", 6371.0, density_exponent), name)


def load_prem_flat(*, interpolation: str = "stiffness", max_depth_km: float = 500.0,
                   max_spacing_km: Optional[float] = 10.0,
                   density_exponent: float = -2.275,
                   csv_path: Optional[Union[str, Path]] = None,
                   name: str = "prem_noocean_flat") -> _api.Model:
    """Load ocean-free PREM and apply EFT to the nodes, retaining metadata.

    This differs from load_prem_spherical, which lets the solver apply the
    selected spherical convention inside refined intervals.
    """
    spherical = load_prem_spherical(
        interpolation=interpolation, max_depth_km=max_depth_km,
        max_spacing_km=max_spacing_km, density_exponent=density_exponent,
        csv_path=csv_path, name=name)
    return _api.earth_flattening_transformation(spherical)
