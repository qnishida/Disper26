#!/usr/bin/env python3
"""Regenerate Figure 4 using the frozen model shared with Figures 5/6."""
import argparse,json,sys,subprocess
from pathlib import Path
import numpy as np
import generate_kernel_figures as g
p=argparse.ArgumentParser();p.add_argument('--disper26-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=g.PAPER/'results/model_recompute');args=p.parse_args()
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper.provenance import validate_output
args.output_dir = validate_output(args.output_dir)
import shutil
frozen=g.DATA
g.DATA=args.output_dir.resolve()
if g.DATA==frozen.resolve():p.error('Frozen inputs cannot be the output directory')
g.DATA.mkdir(parents=True,exist_ok=True)
shutil.copy2(frozen/'prem_like_kernel_model.json',g.DATA/'prem_like_kernel_model.json')
sys.path.insert(0,str(g.PAPER.parent))
from paper.provenance import reference_for_root
source_provenance=reference_for_root(args.disper26_root)
sys.path.insert(0,str(args.disper26_root));import disper26py as dp
g.dp=dp
assert Path(dp.__file__).resolve().parent==args.disper26_root.resolve()
model=json.loads((g.DATA/'prem_like_kernel_model.json').read_text());half=g.parameters(model['materials'][-1])
f=np.linspace(.05,.5,46);out={'frequency_hz':f};stats={}
for wave in ('Rayleigh','Love'):
    raw=[]
    for level in range(9):
        a,h,_=g.mesh(model,level)
        r=g.layer_cake_curve([g.parameters(v) for v in a],h.tolist(),half,(1/f).tolist(),wave,0,.1,6.,2400,1e-12,True)
        c=np.array(r['phase_velocities']);assert np.all(np.isfinite(c)) and np.all(c>0)
        raw.append(c);print(wave,level,flush=True)
    out[wave+'_raw']=raw;out[wave+'_diagonal']=g.extrap(raw)
    stats[wave]={'level7_vs8_max_relative':float(np.max(abs(g.extrap(raw[:8])-g.extrap(raw))/g.extrap(raw)))}
np.savez_compressed(g.DATA/'prem_model_updated.npz',**out)
(g.DATA/'prem_model_updated_report.json').write_text(json.dumps({'source_tree_sha256':source_provenance['source_tree_sha256'],'model_sha256':g.sha(g.DATA/'prem_like_kernel_model.json'),'extension_sha256':g.sha(dp.__file__),'generator_sha256':g.sha(__file__),'data_sha256':g.sha(g.DATA/'prem_model_updated.npz'),'root_tolerance':1e-12,'cases':stats},indent=2)+'\n')
