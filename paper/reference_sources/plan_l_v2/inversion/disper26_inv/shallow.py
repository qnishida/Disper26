"""Canonical setup for the shallow (0--60 km) VTI validation problem."""

from __future__ import annotations

import numpy as np

from .data import DatasetSpec
from .model import InversionBasis, ScaledShearBasis
from .prem import load_prem_flat


SHALLOW_INV_DEPTHS_KM = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])


def shallow_reference(max_spacing_km: float = 15.0, max_depth_km: float = 1000.0):
    """Ocean-free PREM forward model for the shallow inversion tests."""
    return load_prem_flat(max_depth_km=max_depth_km, max_spacing_km=max_spacing_km)


def shallow_basis(reference, rho_beta: float = 0.4, vp_beta: float = 0.6) -> ScaledShearBasis:
    """18-parameter standard basis: independent vsv/vsh on 0--60 km nodes."""
    return ScaledShearBasis(
        reference, SHALLOW_INV_DEPTHS_KM, rho_beta=rho_beta, vp_beta=vp_beta
    )


def shallow_independent_basis(reference) -> InversionBasis:
    """Legacy 54-parameter six-field basis for diagnostic comparisons."""
    return InversionBasis(reference, SHALLOW_INV_DEPTHS_KM)


def shallow_dataset() -> DatasetSpec:
    """140 phase-velocity data: R/L x n=0/1 x 0.03--0.2 Hz."""
    return DatasetSpec.shallow_frequency_grid()
