"""Signed ellipticity: physical polarity, tied-node derivatives and zero crossing.

Value tolerances follow test_public_api; FD uses the paper's 1e-5 scaled
criterion and relative steps 1e-4 / 5e-5. Roots are independently polished.
The 1e-9 absolute derivative allowance covers root roundoff divided by the
FD step for near-zero components; nonzero components use the relative check.
"""
import math
import numpy as np
import pytest
from scipy.optimize import brentq
import disper26 as api
import disper26py as raw


def material(vs=.5, vp=1., rho=2.):
    return raw.create_isotropic_parameters(vs, vp, rho)


def stack():
    return [material()], [1.], material(3.5, 6., 2.7)


def solve(layers, h, hs, period):
    # Use the public mode selector, then an independent bracketed secular root.
    mats = [api.Material(p.rho, p.c11, p.c33, p.c13, p.c44, p.c66,
                         gravity=p.gravity, gravity_internal_scale=p.gravity_internal_scale,
                         gravity_surface_scale=p.gravity_surface_scale) for p in layers+[hs]]
    model = api.create_layer_cake_model(mats[:-1], h, mats[-1])
    s = api.compute_dispersion(model, period=period, outputs=('phase_velocity','group_velocity','hv_ratio'))
    w = 2*math.pi/period
    c = brentq(lambda c: raw.compute_dispersion_value(layers,h,hs,w,c,'Rayleigh'),
               s.phase_velocity-1e-6, s.phase_velocity+1e-6, xtol=1e-14, rtol=1e-14)
    return c, raw.compute_hv_ratio(layers,h,hs,w,c,'Rayleigh'), s


@pytest.mark.parametrize('period,sign,c0,u0', [
    (4.,-1,.6274264369191523,.20853119626794428),
    (4.6,1,1.0004899708460282,.4088709686919485),
    (7.8,1,2.2975569052911426,.5242928811601599),
    (8.,-1,2.4901599478165832,.6398485359543568)])
def test_polarity_eigenfunctions_and_velocity_baseline(period,sign,c0,u0):
    l,h,hs=stack(); c,e,s=solve(l,h,hs,period)
    ef=raw.compute_eigenfunctions_rayleigh(l,h,hs,2*math.pi/period,c,4)
    np.testing.assert_allclose(e, (ef['ur'][0]/ef['uz'][0]).real, rtol=1e-9,atol=1e-11)
    assert sign*e>0
    np.testing.assert_allclose([s.phase_velocity,s.group_velocity],[c0,u0],rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(s.hv_ratio,e,rtol=1e-8,atol=1e-10)
    assert raw.calculate_hv_ratio(l,h,hs,2*math.pi/period,c,'Rayleigh') == e


@pytest.mark.parametrize('interpolation',['stiffness','velocity'])
@pytest.mark.parametrize('period',[4.,4.6])
def test_constant_grid_matches_layer_cake(interpolation,period):
    l,h,hs=stack(); c,e,s=solve(l,h,hs,period)
    g=api.create_grid_model([0.,1.,1.],interpolation=interpolation,
        vs=[.5,.5,3.5],vp=[1.,1.,6.],rho=[2.,2.,2.7])
    gs=api.compute_dispersion(g,period=period,outputs=('phase_velocity','hv_ratio'))
    np.testing.assert_allclose(gs.hv_ratio,e,rtol=1e-8,atol=1e-10)
    pk=api.compute_kernels(s,basis='stiffness',outputs=('hv_ratio',)).gradients['hv_ratio'].reshape(-1,6)
    gk=api.compute_kernels(gs,basis='stiffness',outputs=('hv_ratio',)).gradients['hv_ratio'].reshape(-1,6)
    np.testing.assert_allclose(np.stack([gk[0]+gk[1],gk[2]]),pk,rtol=1e-5,atol=1e-9)
    var=api.compute_kernels(s,basis='stiffness',outputs=('hv_ratio',),options=api.Options(method='variational',integration='analytical'))
    np.testing.assert_allclose(var.gradients['hv_ratio'].reshape(-1,6),pk,rtol=1e-5,atol=1e-9)
    # Replay roots and test the geometric derivative of the tied interface.
    traced=raw.compute_dispersion_romberg_levels([0.,1.,1.],[l[0],l[0],hs],[False,True,False],period,'Rayleigh',c,3,1e-12)
    r=raw.compute_kernels_ad_grid([0.,1.,1.],[l[0],l[0],hs],[False,True,False],period,'Rayleigh',traced['phase_velocity'],
        max_steps=3,tolerance=1e-12,phase_velocity_roots=traced['phase_velocity_roots'],phase_accepted_level=traced['phase_accepted_level'])
    np.testing.assert_allclose(r[2],e,rtol=1e-8,atol=1e-10)
    ka=raw.compute_kernels_ad_layer_cake(l,h,hs,2*math.pi/period,c,compute_thickness=True)
    dh=next(k.d_HV for k in ka if k.parameter_name=='Thickness')
    dz=next(k.d_HV for k in r[5] if k.parameter_name=='Depth')
    np.testing.assert_allclose(dz,dh,rtol=1e-5,atol=1e-9)


def test_zero_crossing_thickness_derivative():
    l,h,hs=stack(); period=4.4
    h0=brentq(lambda v:solve(l,[v],hs,period)[1],.96,1.1,xtol=1e-12)
    c,e,_=solve(l,[h0],hs,period)
    assert abs(e)<1e-8
    ks=raw.compute_kernels_ad_layer_cake(l,[h0],hs,2*math.pi/period,c,compute_thickness=True)
    k=next(k.d_HV for k in ks if k.parameter_name=='Thickness')
    assert np.isfinite(k) and k != 0
    var=raw.compute_kernels_variational_layer_cake(l,[h0],hs,2*math.pi/period,c,
        compute_thickness=True,use_analytical=True,want_kernel_u=False)
    np.testing.assert_allclose([v.d_HV for v in var],[v.d_HV for v in ks],rtol=1e-5,atol=1e-9)
    ef=raw.compute_eigenfunctions_rayleigh(l,[h0],hs,2*math.pi/period,c,4)
    np.testing.assert_allclose((ef['ur'][0]/ef['uz'][0]).real,e,rtol=1e-9,atol=1e-11)
    grid=raw.compute_kernels_ad_grid([0.,h0,h0],[l[0],l[0],hs],
        [False,True,False],period,'Rayleigh',c,max_steps=3,tolerance=1e-12,
        want_kernel_u=False)
    assert abs(grid[2]) < 1e-8
    kg=next(v.d_HV for v in grid[5] if v.parameter_name=='Depth')
    np.testing.assert_allclose(kg,k,rtol=1e-5,atol=1e-9)
    for step in [1e-4,5e-5]:
        delta=step*h0
        em=solve(l,[h0-delta],hs,period)[1]; ep=solve(l,[h0+delta],hs,period)[1]
        assert em*ep<0
        np.testing.assert_allclose((ep-em)/(2*delta),k,rtol=1e-5)
        assert abs(abs(e+k*delta)-abs(ep)) < 1e-5


@pytest.mark.parametrize('gravity',[0.,.00980665])
def test_seabed_value_and_fluid_thickness_derivative(gravity):
    l=[raw.create_fluid_parameters(1.5,1.03,gravity),material(3.2,5.8,2.7)]
    h=[3.,20.]; hs=material(4.5,8.,3.3); period=20.
    c,e,_=solve(l,h,hs,period); w=2*math.pi/period
    ef=raw.compute_eigenfunctions_rayleigh(l,h,hs,w,c,8)
    j=np.flatnonzero(~np.asarray(ef['is_fluid']))[0]
    np.testing.assert_allclose(e,(ef['ur'][j]/ef['uz'][j]).real,rtol=1e-9,atol=1e-11)
    if gravity:
        for op in (raw.compute_kernels_ad_layer_cake, raw.compute_kernels_variational_layer_cake,
                   raw.compute_kernels_fd_layer_cake, raw.compute_hessian_ad_layer_cake):
            with pytest.raises(ValueError, match="Cowling gravity"):
                op(l,h,hs,w,c,wave_type="Rayleigh",compute_thickness=True)
        return
    ka=raw.compute_kernels_ad_layer_cake(l,h,hs,w,c,compute_thickness=True)
    kv=raw.compute_kernels_variational_layer_cake(l,h,hs,w,c,compute_thickness=True,want_kernel_u=False)
    for kernels in [ka,kv]:
        k=next(k.d_HV for k in kernels if k.parameter_name=='Thickness' and k.layer_index==0)
        for delta in [3e-4,1.5e-4]:
            fd=(solve(l,[3+delta,20.],hs,period)[1]-solve(l,[3-delta,20.],hs,period)[1])/(2*delta)
            np.testing.assert_allclose(k,fd,rtol=1e-5,atol=1e-9)

@pytest.mark.parametrize('period',[4.,4.6])
def test_material_derivatives_against_resolved_roots(period):
    l,h,hs=stack(); c,e,_=solve(l,h,hs,period)
    ks=raw.compute_kernels_ad_layer_cake(l,h,hs,2*math.pi/period,c)
    for idx,values in [(0,[.5,1.,2.]),(-1,[3.5,6.,2.7])]:
        for col,name in enumerate(['Vs','Vp','Rho']):
            k=next(k.d_HV for k in ks if k.layer_index==idx and k.parameter_name==name)
            for step in [1e-4,5e-5]:
                pair=[]
                for sign in [-1,1]:
                    p=values.copy(); p[col]*=1+sign*step
                    pair.append(solve([material(*p)] if idx==0 else l,h,
                                      material(*p) if idx==-1 else hs,period)[1])
                fd=(pair[1]-pair[0])/(2*step*values[col])
                np.testing.assert_allclose(k,fd,rtol=1e-5,atol=1e-9)

@pytest.mark.parametrize('gravity',[0.,.00980665])
def test_fluid_grid_seabed_and_tied_gradients(gravity):
    water=api.Material(1.03,1.03*1.5**2,1.03*1.5**2,1.03*1.5**2,0.,0.,gravity=gravity)
    crust=api.create_isotropic_parameters(3.2,5.8,2.7)
    hs=api.create_isotropic_parameters(4.5,8.,3.3)
    models=[api.create_layer_cake_model([water,crust],[3.,20.],hs),
            api.create_grid_model([0.,3.,3.,23.,23.],[water,water,crust,crust,hs],interpolation='stiffness')]
    solutions=[api.compute_dispersion(m,period=20.,outputs=('hv_ratio',)) for m in models]
    np.testing.assert_allclose(solutions[0].hv_ratio,solutions[1].hv_ratio,rtol=1e-9,atol=1e-11)
    if gravity:
        # Preserve the public API's existing Cowling derivative restriction.
        for solution in solutions:
            with pytest.raises(api.Error, match='Cowling gravity'):
                api.compute_kernels(solution,basis='stiffness',outputs=('hv_ratio',))
        water_raw=raw.create_fluid_parameters(1.5,1.03,gravity)
        crust_raw=material(3.2,5.8,2.7); hs_raw=material(4.5,8.,3.3)
        with pytest.raises(ValueError, match="Cowling gravity"):
            raw.compute_kernels_ad_grid([0.,3.,3.,23.,23.],
                [water_raw,water_raw,crust_raw,crust_raw,hs_raw],
                [False,True,False,True,False],20.,'Rayleigh',solutions[1].phase_velocity,
                sensitivity_mode='VTI',want_kernel_u=False)
        return
    kernels=[api.compute_kernels(s,basis='stiffness',outputs=('hv_ratio',)).gradients['hv_ratio'].reshape(-1,6) for s in solutions]
    # Solid parameters and halfspace; fluid stiffness components are constrained.
    np.testing.assert_allclose(np.stack([kernels[1][2]+kernels[1][3],kernels[1][4]]),kernels[0][1:],rtol=1e-5,atol=1e-9)
