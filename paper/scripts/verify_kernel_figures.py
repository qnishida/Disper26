#!/usr/bin/env python3
"""Independent finite differences and pointwise refinement checks for Figure 6."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import generate_kernel_figures as g

def profile_velocity(a,key,level):
    p=a[key+f'_profile_l{level}'];materials=a[key+f'_materials_l{level}'][p[:,0].astype(int)]
    return p[:,1],g.velocity_basis(p[:,8:14],materials)

def main():
    p=argparse.ArgumentParser();p.add_argument('--disper26-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=g.PAPER/'results/kernel_verification');args=p.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper.provenance import validate_output
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(g.PAPER.parent))
    from paper.provenance import reference_for_root
    reference_for_root(args.disper26_root)
    sys.path.insert(0,str(args.disper26_root));import disper26py as dp
    g.dp=dp
    model=json.loads((g.DATA/'prem_like_kernel_model.json').read_text());a=np.load(g.DATA/'prem_group_kernels_updated.npz');report={}
    def value(m,period,wave):
        vals=[];half=np.array(m['materials'][-1])
        for level in range(9):
            material,h,_=g.mesh(m,level);c=g.root(material,h,half,period,wave)
            vals.append(dp.compute_group_velocity_ad([g.parameters(v) for v in material],h.tolist(),g.parameters(half),2*np.pi/period,c,wave))
        return float(g.extrap(vals))
    for wave in ('Rayleigh','Love'):
        for f in (.05,.1):
            key=wave+('_05' if f==.05 else '_10');index=4 if wave=='Rayleigh' else 3;node=0
            mat=model['materials'][node];rho,A,C,F,L,N=mat;eta=F/(A-2*L);v0=np.sqrt((L if wave=='Rayleigh' else N)/rho)
            slopes=[]
            for rel in (1e-4,5e-5):
                h=v0*rel;values=[]
                for sign in (-1,1):
                    m=copy.deepcopy(model);v=v0+sign*h
                    if wave=='Rayleigh':m['materials'][node][4]=rho*v*v;m['materials'][node][3]=eta*(A-2*rho*v*v)
                    else:m['materials'][node][5]=rho*v*v
                    values.append(value(m,1/f,wave))
                slopes.append((values[1]-values[0])/(2*h))
            target=float(a[key+'_full_ad'][node,index]);scale=np.max(abs(a[key+'_full_ad'][:,index]));errors=[abs(v-target)/scale for v in slopes]
            d7,k7=profile_velocity(a,key,7);d8,k8=profile_velocity(a,key,8)
            # Compare at the same depths, inside each material interval. Never interpolate across interfaces.
            errors_profile=[];differences=[];reference=[];depth=np.array(model['depth_km'])
            for j in range(len(depth)-1):
                if model['is_discontinuity'][j] or depth[j+1]<=depth[j]:continue
                h=depth[j+1]-depth[j];points=depth[j]+h*np.linspace(.01,.99,199)
                mask7=(d7>depth[j])&(d7<depth[j+1]);mask8=(d8>depth[j])&(d8<depth[j+1])
                p7=np.array([np.interp(points,d7[mask7],k7[mask7,i]) for i in range(6)]).T
                p8=np.array([np.interp(points,d8[mask8],k8[mask8,i]) for i in range(6)]).T
                differences.append(p7-p8);reference.append(p8)
            diff=np.vstack(differences);ref=np.vstack(reference)
            for i in range(6):
                sc=np.max(abs(ref[:,i]));errors_profile.append(float(np.max(abs(diff[:,i]))/sc) if sc>0 else 0.)
            report[key]={'node':node,'parameter':'Vsv' if wave=='Rayleigh' else 'Vsh','relative_steps':[1e-4,5e-5],'fd_values':slopes,'full_ad':target,'fd_normalized_errors':errors,'profile_l7_l8_component_normalized_errors':errors_profile,'profile_note':'Fixed-depth comparison of fine piecewise-constant reconstructions; not an extrapolation of pointwise values.'}
            print(key,report[key],flush=True)
    r={'cases':report,'kernel_data_sha256':g.sha(g.DATA/'prem_group_kernels_updated.npz'),'script_sha256':g.sha(__file__),'profile_display_tolerance':.005,'profile_display_tolerance_rationale':'0.5% of each component peak resolves the plotted shape; this is separate from the 1e-5 nodal-derivative criterion.'}
    (args.output_dir/'prem_group_kernels_verification.json').write_text(json.dumps(r,indent=2)+'\n')
    assert all(max(v['fd_normalized_errors'])<1e-5 for v in report.values())
    assert all(max(v['profile_l7_l8_component_normalized_errors'])<.005 for v in report.values())
if __name__=='__main__':main()
