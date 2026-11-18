#!/usr/bin/env python3
"""Recompute the archived mixed-response experiment with Rayleigh p=0, Love p=3.

Uses immutable plan_l_v2 sources; all new outputs go outside frozen inputs.
The opened perturbation coordinates are reused, never relabelled as fresh holdout.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER / 'reference_sources/plan_l_v2'
OLD = PAPER / 'data/plan_l_v2'
BASE = dict(max_steps=6, tolerance=1e-9, hessian_max_levels=5, hessian_tolerance=1e-6)
TIGHT = dict(BASE, max_steps=8, tolerance=1e-12, root_tolerance=1e-13)
WAVES = ('rayleigh', 'love')
EXPONENTS = {'rayleigh': 2., 'love': 5.}
sys.path[:0] = [str(ROOT), str(ROOT / 'inversion'), str(PAPER.parent)]
from paper.provenance import verify_reference, validate_output
import disper26py as dp
from disper26_inv.prem import load_prem_flat
from disper26_inv.shallow import shallow_basis
from disper26_inv.priors import PriorCovariance
from disper26_inv.data import DatasetSpec
from disper26_inv.engine import Datum, ForwardEngine
from disper26_inv.pipeline import run_phase1
from disper26_inv.long_period_taylor import (PRIMARY_FREQUENCY_HZ, PRIMARY_TARGETS,
    LongPeriodTaylorArtifact, LongPeriodTaylorProvenance, build_long_period_taylor_curve,
    LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME, LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION,
    PHASE_VELOCITY_RESPONSE_KIND, NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND)
from disper26_inv.long_period_taylor_manifest import (LongPeriodTaylorPriorManifest,
    LongPeriodTaylorGate, reference_fingerprint, basis_fingerprint, prior_fingerprint,
    compute_shear_log_operator, physical_shear_anomaly, _scale_ray_to_physical_anomaly,
    radial_neighbor_pairs)
from disper26_inv.long_period_taylor_exact import collect_exact_mode_trace_ensemble
from disper26_inv.long_period_taylor_reference import _branch_identity_disagreements
from disper26_inv.long_period_taylor_mode import ModeCharacterDiagnostics
from disper26_inv.long_period_taylor_validation import (validate_long_period_taylor_curve,
    benchmark_long_period_taylor_curve)
from disper26_inv.long_period_taylor_io import save_long_period_mixed_response_artifact as save_long_period_taylor_artifact
from disper26_inv.derivatives import gradient_log, kernels_to_stiffness_gradient, project_gradient

FREQ = np.array(PRIMARY_FREQUENCY_HZ)
RESP = {'rayleigh': PHASE_VELOCITY_RESPONSE_KIND, 'love': NORMALIZED_SQUARED_SLOWNESS_RESPONSE_KIND}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    def default(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, indent=2, default=default, allow_nan=False)+'\n')

def load(path):
    with np.load(path, allow_pickle=False) as a:
        return dict(a)

def objects(wave, depth=2400, legacy=False):
    model = load_prem_flat(max_depth_km=depth, max_spacing_km=15,
                          density_exponent=-2.275 if legacy else EXPONENTS[wave])
    basis = shallow_basis(model)
    prior = PriorCovariance(basis, **LongPeriodTaylorPriorManifest().prior_kwargs())
    return model, basis, prior

def moho(model):
    return float(model.depths[np.flatnonzero(model.is_discontinuity)[1]])

def cohort(kind='holdout'):
    a = load(OLD/kind/'split_rayleigh.npz')
    b = load(OLD/kind/'split_love.npz')
    for key in ('delta_m','z','direction_index','sign','target_maximum_anomaly'):
        np.testing.assert_array_equal(a[key], b[key])
    return a

def direct(task):
    wave, dm, tight, legacy = task
    _, basis, _ = objects(wave, legacy=legacy)
    engine = ForwardEngine(**(TIGHT if tight else BASE))
    model = basis.m_to_forward_model(dm)
    values=[]
    for m in (0,1):
        sol=engine.trace_dispersion_curve(model,wave.capitalize(),m,FREQ,minimum_trace_level=5)
        assert all(s.confirmation['accepted'] for s in sol)
        values.append([s.c for s in sol])
    return np.asarray(values)

def baseline(out):
    a=cohort(); rows=np.flatnonzero((a['direction_index']==0)&np.isclose(a['target_maximum_anomaly'],.005))
    errors={}
    for wave in WAVES:
        exact=load(OLD/'holdout'/('split_'+wave+'.npz'))['phase_velocity'][rows,:2]
        got=np.array([direct((wave,d,False,True)) for d in a['delta_m'][rows]])
        np.testing.assert_allclose(got,exact,rtol=0,atol=1e-12)
        errors[wave]=float(np.max(abs(got-exact)))
    write_json(out/'baseline.json',dict(maximum_absolute_error_km_s=errors,tolerance=1e-12))

def reference(out, processes):
    baseline(out)
    all_traces=[]; times={}; records={}; all_jets={}
    old, oldbasis, _=objects('rayleigh',legacy=True)
    for depth in (2200,2400,2600):
        start=time.perf_counter(); traces=[]
        for wave in WAVES:
            model,basis,prior=objects(wave,depth)
            if depth==2400:
                np.testing.assert_array_equal(model.depths,old.depths)
                np.testing.assert_array_equal(model.vsv,old.vsv)
                np.testing.assert_array_equal(compute_shear_log_operator(basis),compute_shear_log_operator(oldbasis))
            dataset=DatasetSpec([Datum(wave.capitalize(),m,1/f) for m in (0,1) for f in FREQ])
            result=run_phase1(model,basis,dataset,processes=processes,engine_kwargs=BASE,
                              reduced_hessian=True,ad_kernel_every=0,cache_dir=None)
            arrays=dict(c=result.c0.reshape(2,10),g=result.J.reshape(2,10,18),
                        h=result.H.reshape(2,10,18,18),phase1_error=result.rel_err,
                        hessian_asymmetry=result.asymmetry)
            assert all(np.all(np.isfinite(v)) for v in arrays.values())
            trace=collect_exact_mode_trace_ensemble(basis,np.zeros((1,18)),wave.capitalize(),FREQ,engine_kwargs=BASE)
            assert np.all(trace.success & trace.trace_continuity)
            mismatch=float(np.max(abs(arrays['c']/trace.phase_velocity[0,:2]-1)))
            assert mismatch<=2e-8
            traces.append(trace)
            arrays.update(exact_reference=trace.phase_velocity,reference_group_velocity=trace.group_velocity)
            np.savez_compressed(out/f'reference_{wave}_{depth}.npz',**arrays)
            all_jets[wave,depth]=arrays
            records[f'{wave}_{depth}']=dict(reference_fingerprint=reference_fingerprint(model),
                phase_trace_relative_difference=mismatch,basis_fingerprint=basis_fingerprint(basis),
                density_exponent=EXPONENTS[wave])
            if depth==2400:
                np.savez_compressed(out/f'model_{wave}.npz',**{k:getattr(model,k) for k in
                    ('depths','rho','vpv','vph','vsv','vsh','eta','is_discontinuity')})
            print('reference',wave,depth,flush=True)
        times[str(depth)]=time.perf_counter()-start
        all_traces.append(SimpleNamespace(exact_by_wave=traces))
    disagreements=sum(_branch_identity_disagreements(a,b) for a,b in zip(all_traces,all_traces[1:]))
    cal=cohort('calibration'); dm=cal['delta_m'][np.isclose(cal['target_maximum_anomaly'],.08)]
    differences=[]
    for wave in WAVES:
        for mode in (0,1):
            predictions=[]
            for depth in (2400,2600):
                a=all_jets[wave,depth]
                curve=build_long_period_taylor_curve(wave=wave,root_order=mode,frequency_hz=FREQ,
                    reference_phase_velocity=a['c'][mode],phase_gradient_m=a['g'][mode],
                    phase_hessian_m=a['h'][mode],prior_cholesky=np.eye(18),response_kind=RESP[wave])
                predictions.append(curve.evaluate_z_batch(dm,order=2)[0])
            differences.extend((abs(predictions[0]-predictions[1])/predictions[0]).ravel())
    p99=float(np.quantile(differences,.99))
    report=dict(records=records,wall_seconds_by_depth=times,processes=processes,
                depth_prediction_relative_p99=p99,depth_prediction_relative_maximum=float(max(differences)),
                depth_tolerance=1e-3/30,branch_disagreements=disagreements,
                passed=disagreements==0 and p99<=1e-3/30)
    write_json(out/'reference_report.json',report)
    assert report['passed'],report
    artifact(out)

def artifact(out):
    model,basis,prior=objects('rayleigh'); curves=[]
    for wave,mode in PRIMARY_TARGETS:
        a=load(out/f'reference_{wave}_2400.npz')
        curves.append(build_long_period_taylor_curve(wave=wave,root_order=mode,frequency_hz=FREQ,
            reference_phase_velocity=a['c'][mode],phase_gradient_m=a['g'][mode],phase_hessian_m=a['h'][mode],
            prior_cholesky=prior.chol(),response_kind=RESP[wave]))
    refs={w:reference_fingerprint(objects(w)[0]) for w in WAVES}
    prov=LongPeriodTaylorProvenance(code_commit='c72cf1d961f2bc50118fd69869b88dbc5c015629',
        plan_sha256=sha(__file__),reference_manifest_sha256=sha(PAPER/'reference_manifest.json'),
        depth_convergence_report_sha256=sha(out/'reference_report.json'),
        reference_fingerprint=hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest(),
        basis_fingerprint=basis_fingerprint(basis),prior_fingerprint=prior_fingerprint(prior),
        dataset_fingerprint=sha(OLD/'holdout/split_manifest.json'),
        calibration_manifest_sha256=sha(OLD/'calibration/split_manifest.json'))
    value=LongPeriodTaylorArtifact(prior.chol(),compute_shear_log_operator(basis),tuple(curves),prov,
        metadata_json=json.dumps(dict(role='reassessment_of_opened_directions',p_by_wave={'rayleigh':0,'love':3},
        reference_fingerprints=refs,reference_fingerprint_semantics='hash of wave-to-model fingerprint mapping')),
        schema_name=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_NAME,schema_version=LONG_PERIOD_MIXED_RESPONSE_SCHEMA_VERSION)
    save_long_period_taylor_artifact(value,out/'reference_jets.npz')
    return value

def _fd_case(task):
    out, wave, mode, fi = task
    _,basis,_=objects(wave); engine=ForwardEngine(**BASE)
    a=load(out/f'reference_{wave}_2400.npz')
    columns=(4,13); raw={}
    datum=Datum(wave.capitalize(),mode,1/FREQ[fi]); H=a['h'][mode,fi]; c0=a['c'][mode,fi]
    def grad(dm):
        model=basis.m_to_forward_model(dm)
        sol,k=engine.phase_velocity_with_kernels(model,datum,c_hint=float(c0))
        assert sol.branch==datum.branch and sol.confirmation['accepted']
        g=project_gradient(gradient_log(model,kernels_to_stiffness_gradient(k,model.n_nodes)),basis,model=model)
        return np.asarray(g),sol.c
    g0,c=grad(np.zeros(18)); assert abs(c/c0-1)<=2e-8
    fd=[]; norms=[]; errors=[]; changes=[]; gradients=[]; roots=[]
    for col in columns:
        estimates=[]
        for step in (2e-4,1e-4):
            dm=np.eye(18)[col]*step; gp,cp=grad(dm); gm,cm=grad(-dm)
            estimates.append((gp-gm)/(2*step)); gradients.extend([gp,gm]); roots.extend([cp,cm])
        norm=np.linalg.norm(H[:,col]); norms.append(norm); fd.append(estimates[1])
        errors.append(float(np.linalg.norm(estimates[1]-H[:,col])/norm))
        changes.append(float(np.linalg.norm(estimates[1]-estimates[0])/norm))
    cross=float(max(abs(fd[1][4]-H[4,13]),abs(fd[0][13]-H[13,4]))/max(norms))
    case=dict(wave=wave,mode=mode,frequency_hz=float(FREQ[fi]),column_errors=errors,
              step_changes=changes,cross_error=cross,confirmed_roots=9)
    key=f'{wave}{mode}_{fi}'
    for k,v in dict(gradients=gradients,roots=roots,hessian=H,zero_gradient=g0,fd=fd).items(): raw[key+'_'+k]=np.asarray(v)
    print('FD',case,flush=True)
    return case, raw


def finite_difference(out, processes):
    tasks=[(out,wave,mode,fi) for wave in WAVES for mode in (0,1) for fi in (0,5,9)]
    with ProcessPoolExecutor(max_workers=processes) as pool:
        computed=list(pool.map(_fd_case,tasks))
    cases=[case for case,raw in computed]
    raw={k:v for case,values in computed for k,v in values.items()}
    passed=all(max(c['column_errors'])<=5e-3 and max(c['step_changes'])<=5e-3 and c['cross_error']<=1e-2 for c in cases)
    np.savez_compressed(out/'hessian_verification.npz',**raw)
    write_json(out/'hessian_verification.json',dict(cases=cases,passed=passed,
        tolerances=dict(column=5e-3,step=5e-3,cross=1e-2),steps=[2e-4,1e-4]))
    assert passed

def validation(out,processes):
    art=artifact(out); src=cohort(); results=[]; exported={'frequency_hz':FREQ}
    pairs=radial_neighbor_pairs(SimpleNamespace(**src))
    for wave in WAVES:
        model,basis,_=objects(wave)
        operator=compute_shear_log_operator(basis)
        achieved=np.array([physical_shear_anomaly(operator,d)[0] for d in src['delta_m']])
        np.testing.assert_allclose(achieved,src['target_maximum_anomaly'],rtol=0,atol=2e-12)
        ensemble=collect_exact_mode_trace_ensemble(basis,src['delta_m'],wave.capitalize(),FREQ,
                                                  processes=processes,engine_kwargs=BASE)
        saved={k:src[k] for k in ('delta_m','z','direction_index','sign','target_maximum_anomaly')}
        for k in ('phase_velocity','group_velocity','kernels','success','trace_continuity'): saved[k]=getattr(ensemble,k)
        write_json(out/f'trace_errors_{wave}.json',ensemble.errors)
        for mode in (0,1):
            diag=ensemble.diagnostics(mode,moho(model),model_neighbor_pairs=pairs)
            for k,v in asdict(diag).items(): saved[f'diagnostic_{mode}_{k}']=np.asarray(v)
            result=validate_long_period_taylor_curve(art,wave,mode,src['delta_m'],ensemble.phase_velocity[:,mode],
                ensemble.success[:,mode]&ensemble.trace_continuity[:,mode],diag,src['target_maximum_anomaly'])
            results.append(asdict(result))
            key=wave+str(mode); exported[key+'_exact']=ensemble.phase_velocity[:,mode]
            for order in (1,2): exported[key+f'_order{order}']=art.evaluate_unchecked_batch(wave,mode,src['delta_m'],order=order)[0]
        for name,key in [('amplitude','target_maximum_anomaly'),('achieved_amplitude','achieved_maximum_anomaly'),('direction_index','direction_index'),('sign','sign')]:
            exported[wave+'_'+name]=src[key]
        np.savez_compressed(out/f'validation_{wave}.npz',**saved)
        print('validation complete',wave,flush=True)
    np.savez_compressed(out/'plan_l_v2_holdout_curves.npz',**exported)
    write_json(out/'validation_report.json',dict(role='reassessment_of_previously_opened_holdout_directions',
        curve_results=results,passed=all(r['passed'] for r in results),gate=LongPeriodTaylorGate().as_dict()))

def small(out,processes):
    art=artifact(out); src=cohort(); rows=np.flatnonzero(np.isclose(src['target_maximum_anomaly'],.005))
    operator=compute_shear_log_operator(objects('rayleigh')[1]); models=[]; amps=[]; indices=[]
    for amp in (.001,.002,.005):
        for i in rows:
            d=src['delta_m'][i]; models.append(d*_scale_ray_to_physical_anomaly(operator,d,amp)); amps.append(amp); indices.append(i)
    models=np.array(models); amps=np.array(amps); exported={'frequency_hz':FREQ}; report={}
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for wave in WAVES:
            exact=np.array(list(pool.map(direct,[(wave,d,False,False) for d in models])))
            tight=np.array(list(pool.map(direct,[(wave,d,True,False) for d in models[:128]])))
            previous=load(out/f'validation_{wave}.npz')['phase_velocity'][rows,:2]
            np.testing.assert_allclose(exact[128:],previous,rtol=0,atol=1e-12)
            exported[wave+'_amplitude']=amps; exported[wave+'_achieved_amplitude']=amps
            exported[wave+'_direction_index']=src['direction_index'][indices]; exported[wave+'_sign']=src['sign'][indices]
            exported[wave+'_delta_m']=models
            for mode in (0,1):
                key=wave+str(mode); exported[key+'_exact']=exact[:,mode]; exported[key+'_tight_exact']=tight[:,mode]
                r={}
                for order in (1,2):
                    pred=art.evaluate_unchecked_batch(wave,mode,models,order=order)[0]
                    exported[key+f'_order{order}']=pred
                    e=100*abs(pred-exact[:,mode])/exact[:,mode]
                    q=[float(np.quantile(e[amps==h],.9)) for h in (.001,.002,.005)]
                    r[f'order{order}_p90_percent']=q
                    r[f'order{order}_slope']=float(np.polyfit(np.log([.001,.002,.005]),np.log(q),1)[0])
                r['tight_difference_max_percent']=float(np.max(100*abs(exact[:128,mode]-tight[:,mode])/tight[:,mode]))
                report[key]=r
            print('small complete',wave,flush=True)
    np.savez_compressed(out/'plan_l_v2_small_amplitude.npz',**exported)
    write_json(out/'small_amplitude_report.json',dict(curves=report,base=BASE,tight=TIGHT,minimum_trace_level=5))

def benchmark(out):
    art=artifact(out); src=cohort('development'); dm=src['delta_m'][src['target_maximum_anomaly']<=.08+2e-12][:8]; results=[]
    for wave,mode in PRIMARY_TARGETS:
        _,basis,_=objects(wave); engine=ForwardEngine(**BASE)
        def exact(d):
            return np.array([s.c for s in engine.trace_dispersion_curve(basis.m_to_forward_model(d),wave.capitalize(),mode,FREQ)])
        def approx(batch):
            before=dict(engine.counters)
            r=art.predict_batch(wave,mode,dm) if batch else art.predict(wave,mode,dm[0])
            assert np.all(r.supported) and before==engine.counters
            return r.phase_velocity
        result=benchmark_long_period_taylor_curve(wave=wave,root_order=mode,exact_scalar=lambda:exact(dm[0]),
            exact_batch=lambda:np.stack([exact(d) for d in dm]),surrogate_scalar=lambda:approx(False),
            surrogate_batch=lambda:approx(True),repeats=5)
        results.append(asdict(result))
    ref=json.loads((out/'reference_report.json').read_text())
    cold=ref['wall_seconds_by_depth']['2400']
    write_json(out/'plan_l_v2_benchmark_report.json',dict(curve_results=results,repeats=5,batch_size=8,
        surrogate_exact_solver_call_count=0,reference_build_seconds=cold,reference_build_processes=ref['processes'],
        reference_build_scope='two wave-specific reference expansions including reference mode diagnostics; no cache',
        amortization_queries=float(cold/sum(r['exact_scalar_seconds']-r['surrogate_scalar_seconds'] for r in results))))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['reference','fd','validate','small','benchmark','manifest','all'])
    ap.add_argument('--output-dir',type=Path,default=PAPER/'results/plan_l_eft_p0_p3')
    ap.add_argument('--processes',type=int,default=8)
    args=ap.parse_args(); out=validate_output(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    provenance=verify_reference('plan_l_v2',ROOT,OLD/'long_period_mixed_response_v2.npz')
    assert Path(dp.__file__).resolve().parent==ROOT.resolve()
    for kind in ('holdout','development','calibration'):
        manifest=json.loads((OLD/kind/'split_manifest.json').read_text())
        for wave in WAVES: assert sha(OLD/kind/manifest['curves'][wave]['file'])==manifest['curves'][wave]['sha256']
    funcs={'reference':lambda:reference(out,args.processes),'fd':lambda:finite_difference(out,args.processes),
           'validate':lambda:validation(out,args.processes),'small':lambda:small(out,args.processes),
           'benchmark':lambda:benchmark(out)}
    for name,func in funcs.items():
        if args.stage in ('all',name):
            print('START',name,flush=True); func(); print('DONE',name,flush=True)
    write_json(out/'manifest.json',dict(role='EFT p=0/p=3 reassessment using archived solver and opened coordinates',
        p_by_wave={'rayleigh':0,'love':3},density_exponent_by_wave=EXPONENTS,source=provenance,
        extension_sha256=sha(dp.__file__),generator_sha256=sha(__file__),settings=BASE,
        python=sys.version,numpy=np.__version__,exports={p.name:sha(p) for p in sorted(out.iterdir())
        if p.suffix in ('.npz','.json') and p.name!='manifest.json'}))

if __name__=='__main__': main()
