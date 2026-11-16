"""Group-kernel regression against the independent 90-digit state ODE.

Keep the existing full-AD comparison's rtol=1e-12. The old sinh/sqrt formula
misses that criterion against this reference by up to 6.3e-10.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pytest
import disper26py as dp

FIXTURE = json.loads((Path(__file__).resolve().parents[1] /
    'fixtures/regular_group_kernel_reference.json').read_text())


@lru_cache(None)
def kernels(velocity):
    model = FIXTURE['model']
    p = [dp.create_isotropic_parameters(*args) for args in
         zip(model['vs'], model['vp'], model['rho'])]
    rows = dp.compute_kernels_ad_layer_cake(p[:-1], model['thicknesses'], p[-1],
        model['omega'], velocity, 'Rayleigh', 'Isotropic', True)
    return {(row.layer_index, row.parameter_name): row.d_U for row in rows}


@pytest.mark.parametrize('row', FIXTURE['cases'], ids=lambda r:
    f"{r['case']}/{r['layer_index']}/{r['parameter']}")
def test_group_kernel_matches_high_precision_ode(row):
    value = kernels(row['phase_velocity'])[(row['layer_index'], row['parameter'])]
    np.testing.assert_allclose(value, float(row['reference_dU']), rtol=1e-12, atol=0.)
