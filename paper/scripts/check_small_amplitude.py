#!/usr/bin/env python3
"""Extend opened Plan L rays for a convergence diagnostic, without refitting.
Run with --disper26-root and --plan-l-dir in an environment with the paper extra.
The original holdout and its acceptance statistics are never modified.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'data/results'
BASE = dict(max_steps=6, tolerance=1e-9, hessian_max_levels=5,
            hessian_tolerance=1e-6)
TIGHT = dict(BASE, max_steps=8, tolerance=1e-12, root_tolerance=1e-13)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup(root, expected_hash=None):
    from paper.provenance import load_reference_extension, verify_reference
    if expected_hash is not None:
        verify_reference("plan_l_v2", root)
    extension = load_reference_extension(root)
    if expected_hash is not None and sha(extension.__file__) != expected_hash:
        raise ValueError("Worker and parent loaded different snapshot extensions")
    sys.path.insert(0, str(Path(root) / "inversion"))
    global basis, ForwardEngine
    from disper26_inv.engine import ForwardEngine
    from disper26_inv.long_period_taylor_reference import build_plan_l_objects
    _, basis, _ = build_plan_l_objects()

def solve(task):
    wave, dm, freq, tight = task
    model = basis.m_to_forward_model(dm)
    engine = ForwardEngine(**(TIGHT if tight else BASE))
    values = np.array([[s.c for s in engine.trace_dispersion_curve(
        model, wave.capitalize(), m, freq, minimum_trace_level=5)] for m in (0, 1)])
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    return values

def summarize(arrays, report):
    """Record slope and reproduction diagnostics from saved, unfiltered arrays."""
    for wave in ('rayleigh', 'love'):
        src = np.load(report['sources'][wave]['path'])
        rows = np.flatnonzero(np.isclose(src['target_maximum_anomaly'], .005))
        amp = arrays[wave+'_amplitude']
        for m in (0, 1):
            key = wave+str(m)
            stats = report['curves'][key]
            stats['baseline_0p5pct_max_abs_difference_km_s'] = float(np.max(np.abs(
                arrays[key+'_exact'][128:] - src['phase_velocity'][rows,m,:])))
            for order in (1, 2):
                err = 100*np.abs(arrays[key+'_order'+str(order)]-arrays[key+'_exact'])/arrays[key+'_exact']
                q = [float(np.quantile(err[amp==h], .9)) for h in (.001,.002,.005)]
                stats['order%d_p90_percent' % order] = q
                stats['order%d_p90_slope_0p1_to_0p5pct' % order] = float(np.polyfit(
                    np.log([.001,.002,.005]), np.log(q), 1)[0])
    report['amplitudes'] = [.001, .002, .005]
    report['diagnostic_script_sha256'] = sha(__file__)
    return report


def main():
    global OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--disper26-root', type=Path, required=True)
    p.add_argument('--plan-l-dir', type=Path, required=True)
    p.add_argument('--processes', type=int, default=8)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results/small_amplitude')
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper.provenance import validate_output, verify_reference
    args.output_dir = validate_output(args.output_dir)
    OUT = args.output_dir
    source_provenance = verify_reference('plan_l_v2', args.disper26_root, args.plan_l_dir / 'long_period_mixed_response_v2.npz')
    setup(args.disper26_root)
    from disper26_inv.long_period_taylor_manifest import (
        compute_shear_log_operator, _scale_ray_to_physical_anomaly,
        physical_shear_anomaly)
    import disper26py
    artifact = args.plan_l_dir / 'long_period_mixed_response_v2.npz'
    jet = np.load(artifact)
    assert sha(artifact) == '4e92574a29a6a1398052b4a7744fc3c4131de4ae779906e348c05719fe555bb5'
    operator = compute_shear_log_operator(basis)
    arrays, report = {}, {'role': 'post-holdout small-amplitude convergence diagnostic',
        'base_settings': BASE, 'tight_settings': TIGHT, 'minimum_trace_level': 5,
        'reference_id': source_provenance['reference_id'],
        'source_tree_sha256': source_provenance['source_tree_sha256'],
        'extension_sha256': sha(disper26py.__file__), 'artifact_sha256': sha(artifact),
        'source_files_sha256': {str(p.relative_to(args.disper26_root)): sha(p)
            for p in sorted((args.disper26_root / 'inversion/disper26_inv').glob('*.py'))},
        'sources': {}, 'curves': {}}
    with ProcessPoolExecutor(args.processes, initializer=setup,
                             initargs=(args.disper26_root, report['extension_sha256'])) as pool:
        for wi, wave in enumerate(('rayleigh','love')):
            src = args.plan_l_dir / 'holdout' / ('split_'+wave+'.npz')
            s = np.load(src)
            report['sources'][wave] = {'path':str(src), 'sha256':sha(src)}
            rows = np.flatnonzero(np.isclose(s['target_maximum_anomaly'], .005))
            assert len(rows) == 64
            dm, z, amps, dirs, signs, achieved = [], [], [], [], [], []
            for h in (.001, .002, .005):
                for row in rows:
                    scale = _scale_ray_to_physical_anomaly(operator, s['delta_m'][row], h)
                    dm.append(scale*s['delta_m'][row]); z.append(scale*s['z'][row])
                    amps.append(h); dirs.append(s['direction_index'][row]); signs.append(s['sign'][row])
                    achieved.append(physical_shear_anomaly(operator, dm[-1])[0])
            dm, z, amps = np.array(dm), np.array(z), np.array(amps)
            freq = s['frequency_hz']; arrays['frequency_hz'] = freq
            np.testing.assert_allclose(achieved, amps, rtol=0, atol=2e-12)
            tasks = [(wave, d, freq, False) for d in dm]
            exact = np.array(list(pool.map(solve, tasks)))
            print(wave, 'base complete', flush=True)
            tight = np.array(list(pool.map(solve, [(wave,d,freq,True) for d in dm[:128]])))
            print(wave, 'tight complete', flush=True)
            np.testing.assert_allclose(exact[128:], s['phase_velocity'][rows,:2,:], rtol=0, atol=1e-12)
            for name, value in [('amplitude',amps),('achieved_amplitude',achieved),
                                ('direction_index',dirs),('sign',signs),('delta_m',dm),('z',z)]:
                arrays[wave+'_'+name] = np.asarray(value)
            for m in (0,1):
                key = wave+str(m); prefix = 'curve_%02d_'%(wi*2+m)
                c0=jet[prefix+'reference_phase_velocity']
                y1=jet[prefix+'reference_response']+z@jet[prefix+'response_gradient_z'].T
                y2=y1+.5*np.einsum('mp,fpq,mq->mf',z,jet[prefix+'response_hessian_z'],z)
                c1,c2=(c0/np.sqrt(y1),c0/np.sqrt(y2)) if wave=='love' else (y1,y2)
                arrays[key+'_exact']=exact[:,m]; arrays[key+'_tight_exact']=tight[:,m]
                arrays[key+'_order1']=c1; arrays[key+'_order2']=c2
                e=100*abs(c2-exact[:,m])/exact[:,m]
                diff=100*abs(tight[:,m]-exact[:128,m])/tight[:,m]
                report['curves'][key]={'order2_p90_percent':[float(np.quantile(e[amps==h],.9)) for h in (.001,.002,.005)],
                    'tight_difference_max_percent':float(diff.max()),
                    'tight_difference_p90_percent':float(np.quantile(diff,.9)),
                    'order2_tight_p90_percent':[float(np.quantile(100*abs(c2[j:j+64]-tight[j:j+64,m])/tight[j:j+64,m],.9)) for j in (0,64)]}
    np.savez_compressed(OUT/'plan_l_v2_small_amplitude.npz', **arrays)
    summarize(arrays, report)
    report['data_sha256']=sha(OUT/'plan_l_v2_small_amplitude.npz')
    (OUT/'plan_l_v2_small_amplitude_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
