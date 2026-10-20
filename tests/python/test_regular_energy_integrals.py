"""Public energy integrals at critical finite-layer speeds.

Use the existing energy/root tolerance (1e-10) and an independent high-
precision state-ODE integral. Layer tests fix the boundary state, so a
compensating error in eigenfunction reconstruction cannot hide an error.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import disper26py as dp

from verify_regular_energy_integrals import reference, cases

ROOT=Path(__file__).resolve().parents[2]
ROWS=json.loads((ROOT/'tests/fixtures/regular_layer_reference.json').read_text())['cases']


def material(raw):
    p=dp.PhysicalParameters()
    for name,value in zip(('rho','c11','c13','c33','c44','c66'),raw):setattr(p,name,value)
    return p


@pytest.mark.parametrize('name,case',list(cases().items()),ids=list(cases()))
def test_fixed_boundary_layer_integrals(name,case):
    p=material(case['material']); h=case['h']; y=case['state']
    hs=dp.create_isotropic_parameters(4.6,8.,3.3)
    fields={'depth':np.array([0.,h])}
    if case['wave']=='Rayleigh':
        fields.update({key:np.array([v,0.],dtype=complex) for key,v in zip(('uz','szz','ur','szr'),y)})
        f=dp.compute_energy_integrals_analytical_rayleigh
    else:
        fields.update({key:np.array([v,0.],dtype=complex) for key,v in zip(('uy','tyz'),y[:2])})
        f=dp.compute_energy_integrals_analytical_love
    actual=f(fields,[p],[h],hs,case['omega'],case['c'])
    expected=np.array(reference(case),dtype=float)
    np.testing.assert_allclose(actual[:3],expected,rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize('name',[name for name in ROWS if name.startswith('root_near_vs/')])
def test_root_energy_matches_independent_group_velocity(name):
    row=ROWS[name];case=row['input'];p=[material(raw) for raw in case['layers']]
    args=(p,case['thicknesses'],material(case['halfspace']),case['omega'],case['c'])
    fields=dp.compute_eigenfunctions_rayleigh(*args,1)
    energy=dp.compute_energy_integrals_analytical_rayleigh(fields,*args)
    np.testing.assert_allclose(energy[2]/(case['c']*energy[0]),row['reference']['U'],rtol=1e-10,atol=0.)
    np.testing.assert_allclose(energy[1],case['omega']**2*energy[0],rtol=1e-10,atol=0.)
