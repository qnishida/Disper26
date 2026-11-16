#!/usr/bin/env python3
"""Recalculate manuscript Figures 5/6 with current Disper26; no solver edits.
See README.md. Plotting is separate and uses only the exported files.
"""
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np
PAPER=Path(__file__).resolve().parents[1]
DATA=PAPER/'data/results'
NAMES=('rho','c11','c33','c13','c44','c66')
K_NAMES=('Rho','A','C','F','L','N')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extrap(v):
    a=np.array(v,dtype=float).copy()
    for m in range(1,len(a)):
        a[m:]=(4**m*a[m:]-a[m-1:-1])/(4**m-1)
    return a[-1]
def parameters(row):
    p=dp.PhysicalParameters()
    for k,v in zip(NAMES,row):setattr(p,k,float(v))
    return p

def mesh(model,level):
    z=np.array(model['depth_km']);a=np.array(model['materials']);disc=model['is_discontinuity']
    mats=[];h=[];mapping=[]
    for j in range(len(z)-1):
        if disc[j] or z[j+1]<=z[j]:continue
        for i in range(2**level):
            t=(i+.5)/2**level
            mats.append((1-t)*a[j]+t*a[j+1]);h.append((z[j+1]-z[j])/2**level);mapping.append((j,t))
    return np.array(mats),np.array(h),mapping

def layer_cake_curve(*args, **kwargs):
    """Prefer the current API; retain compatibility with frozen source snapshots."""
    compute = getattr(dp, 'compute_dispersion_curve_layer_cake', None)
    if compute is None:
        compute = dp.compute_dispersion_curve_pancake
    return compute(*args, **kwargs)


def root(mats,h,half,period,wave):
    r=layer_cake_curve([parameters(a) for a in mats],h.tolist(),parameters(half),[period],wave,0,.1,6.,2400,1e-12,True)
    c=r['phase_velocities'][0]
    assert np.isfinite(c) and c>0
    return c

def diagnostic(exe,mats,h,half,c,f,wave,q):
    x,w=np.polynomial.legendre.leggauss(q);x=(x+1)/2;w=w/2
    lines=[f'{len(h)} {int(wave=="Love")} {2*np.pi*f:.17g} {c:.17g} {q}']
    lines += [' '.join(format(v,'.17g') for v in list(a)+[hj]) for a,hj in zip(np.vstack([mats,half]),np.r_[h,0])]
    lines += [f'{a:.17g} {b:.17g}' for a,b in zip(x,w)]
    result=subprocess.run([str(exe)],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
    out={k:[] for k in ('P','I','A','META')}
    for line in result.stdout.splitlines():
        a=line.split();out[a[0]].append([float(v) for v in a[1:]])
    return {k:np.array(v) for k,v in out.items()}

def project(integrals,mapping,count):
    a=np.zeros((count,6))
    for v,(j,t) in zip(integrals[:-1],mapping):a[j]+=(1-t)*v;a[j+1]+=t*v
    a[-1]+=integrals[-1]
    return a

def velocity_basis(g,a):
    rho,A,C,F,L,N=a.T
    vp=np.sqrt(A/rho);vv=np.sqrt(C/rho);vs=np.sqrt(N/rho);sv=np.sqrt(L/rho);eta=F/(A-2*L)
    out=np.empty_like(g)
    out[:,0]=g[:,0]+g[:,1]*vp**2+g[:,2]*vv**2+g[:,3]*eta*(vp**2-2*sv**2)+g[:,4]*sv**2+g[:,5]*vs**2
    out[:,1]=2*rho*vp*(g[:,1]+eta*g[:,3]);out[:,2]=2*rho*vv*g[:,2]
    out[:,3]=2*rho*vs*g[:,5];out[:,4]=2*rho*sv*(g[:,4]-2*eta*g[:,3]);out[:,5]=g[:,3]*rho*(vp**2-2*sv**2)
    return out

def api_kernel(model,period,wave,roots,mode):
    accepted=len(roots)-1
    r=dp.compute_kernels_ad_grid(depths=model['depth_km'],params=[parameters(a) for a in model['materials']],is_discontinuity=model['is_discontinuity'],period=period,wave_type=wave,c_guess=float(extrap(roots)),sensitivity_mode='VTI',max_steps=accepted,tolerance=1e-12,compute_c=True,want_kernel_u=True,want_kernel_hv=False,u_kernel_mode=mode,kernel_extrapolation=True,phase_velocity_roots=list(roots),phase_accepted_level=accepted)
    arr=np.zeros((len(model['depth_km']),6))
    for k in r[5]:
        if k.parameter_name in K_NAMES:arr[k.layer_index,K_NAMES.index(k.parameter_name)]=k.d_U
    assert np.all(np.isfinite(arr))
    return arr,r[:5]

def scaled(a,b):
    scale=max(np.max(abs(b)),np.finfo(float).tiny)
    return float(np.max(abs(a-b))/scale)

def main():
    global DATA
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--disper26-root',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--only',choices=['all','convergence','kernels'],default='all');p.add_argument('--output-dir',type=Path,default=PAPER/'results/kernel_recompute');args=p.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper.provenance import validate_output
    args.output_dir = validate_output(args.output_dir)
    import shutil
    frozen=DATA
    DATA=args.output_dir.resolve()
    if DATA==frozen.resolve():p.error('Frozen inputs cannot be the output directory')
    DATA.mkdir(parents=True,exist_ok=True)
    shutil.copy2(frozen/'prem_like_kernel_model.json',DATA/'prem_like_kernel_model.json')
    sys.path.insert(0,str(PAPER.parent))
    from paper.provenance import reference_for_root
    source_provenance=reference_for_root(args.disper26_root)
    sys.path.insert(0,str(args.disper26_root));global dp
    import disper26py as dp
    assert Path(dp.__file__).resolve().parent==args.disper26_root.resolve()
    model=json.loads((DATA/'prem_like_kernel_model.json').read_text());half=np.array(model['materials'][-1]);count=len(model['depth_km'])
    report={'source_tree_sha256':source_provenance['source_tree_sha256'],'extension_path':dp.__file__,'extension_sha256':sha(dp.__file__),'diagnostic_sha256':sha(args.diagnostic),'model_sha256':sha(DATA/'prem_like_kernel_model.json'),'generator_sha256':sha(__file__),'helper_source_sha256':sha(PAPER/'scripts/continuous_group_kernel.cpp'),'root_tolerance':1e-12,'kernel_normalized_tolerance':1e-5}
    if args.only in ('all','convergence'):
        arrays={};stats={}
        for wave in ('Rayleigh','Love'):
            roots=[]
            for level in range(9):
                a,h,_=mesh(model,level);roots.append(root(a,h,half,1/.15,wave))
            diagonals=np.array([extrap(roots[:i+1]) for i in range(9)])
            r=dp.compute_dispersion_romberg(model['depth_km'],[parameters(a) for a in model['materials']],model['is_discontinuity'],1/.15,wave,roots[0],8,1e-12)
            arrays[wave+'_raw']=roots;arrays[wave+'_diagonal']=diagonals;arrays[wave+'_adaptive']=r[:5]
            stats[wave]={'reference_7_vs_8_relative':float(abs(diagonals[7]-diagonals[8])/abs(diagonals[8])),'adaptive_relative_error':float(abs(r[0]-diagonals[8])/abs(diagonals[8])),'adaptive_reported_error':r[4],'adaptive_level':r[3]}
            print('convergence',wave,stats[wave],flush=True)
        np.savez_compressed(DATA/'prem_convergence_updated.npz',**arrays)
        r=dict(report,cases=stats,data_sha256=sha(DATA/'prem_convergence_updated.npz'))
        (DATA/'prem_convergence_updated_report.json').write_text(json.dumps(r,indent=2)+'\n')
    if args.only in ('all','kernels'):
        arrays={'depth_km':np.array(model['depth_km'])};stats={}
        for wave in ('Rayleigh','Love'):
            for f in (.05,.1):
                key=wave+('_05' if f==.05 else '_10');roots=[];projections=[];quadrature_diffs=[];integral_diffs=[]
                for level in range(9):
                    a,h,mapping=mesh(model,level);c=root(a,h,half,1/f,wave);roots.append(c)
                    d=diagnostic(args.diagnostic,a,h,half,c,f,wave,16)
                    d2=diagnostic(args.diagnostic,a,h,half,c,f,wave,32)
                    quad=np.vstack([d2['I'][:,1:],d2['A'][-1,1:]])
                    analytic=d2['A'][:,1:]
                    integral_diffs.append(scaled(quad,analytic));quadrature_diffs.append(scaled(d['I'][:,1:],d2['I'][:,1:]))
                    projections.append(project(quad,mapping,count))
                    if level in (7,8):
                        pts=d2['P'];arrays[key+f'_profile_l{level}']=pts
                        arrays[key+f'_materials_l{level}']=a
                    print(key,'level',level,'integral difference',integral_diffs[-1],flush=True)
                projected=extrap(projections)
                freq,meta=api_kernel(model,1/f,wave,roots,'variational_frequency_ad')
                full,_=api_kernel(model,1/f,wave,roots,'full_ad')
                materials=np.array(model['materials'])
                arrays[key+'_projection']=velocity_basis(projected,materials);arrays[key+'_frequency_ad']=velocity_basis(freq,materials);arrays[key+'_full_ad']=velocity_basis(full,materials);arrays[key+'_roots']=roots
                stats[key]={'quadrature_16_vs_32_max':max(quadrature_diffs),'quadrature_vs_analytic_max':max(integral_diffs),'projected_vs_frequency_ad':scaled(projected,freq),'frequency_vs_full_ad':scaled(freq,full),'projection_l7_vs_l8':scaled(extrap(projections[:8]),projected),'c':meta[0],'U':meta[1],'level':meta[3]}
                print(key,stats[key],flush=True)
        np.savez_compressed(DATA/'prem_group_kernels_updated.npz',**arrays)
        (DATA/'prem_group_kernels_updated_report.json').write_text(json.dumps(dict(report,cases=stats,data_sha256=sha(DATA/'prem_group_kernels_updated.npz')),indent=2)+'\n')
        assert all(v['projected_vs_frequency_ad']<1e-5 and v['frequency_vs_full_ad']<1e-5 and v['quadrature_vs_analytic_max']<1e-5 for v in stats.values()),'kernel validation failed; data retained for diagnosis'
if __name__=='__main__':main()
