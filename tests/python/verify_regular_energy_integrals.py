#!/usr/bin/env python3
"""Compare fixed-state analytical layer energies with an independent ODE.

The reference integrates the quadratic state by a high-precision power series
of M'=G M+M G^T. It has no modal roots, eigenvectors or sinhc evaluations.
The same binary64 boundary state is supplied to each production path.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

from mpmath import mp

ROOT = Path(__file__).resolve().parents[2]


def reference(case, digits=70):
    with mp.workdps(digits):
        rho,a,f,cc,l,nn = map(mp.mpf,case['material'])
        h,w,c = map(mp.mpf,(case['h'],case['omega'],case['c']))
        k=w/c
        if case['wave']=='Rayleigh':
            g=-mp.matrix([[0,1/cc,f*k/cc,0],[-rho*w*w,0,0,k],
                          [-k,0,0,1/l],[0,-f*k/cc,(a-f*f/cc)*k*k-rho*w*w,0]])
            y=mp.matrix(case['state'])
        else:
            # Public outward traction is minus the depth-derivative stress.
            g=mp.matrix([[0,-1/l],[rho*w*w-nn*k*k,0]])
            y=mp.matrix(case['state'][:2])
        term=y*y.T
        integral=h*term
        gh=h*g
        small=0
        for n in range(1,1000):
            term=(gh*term+term*gh.T)/n
            add=h*term/(n+1)
            integral+=add
            small=small+1 if mp.norm(add)<mp.eps*max(1,mp.norm(integral)) else 0
            if small>=4: break
        else: raise RuntimeError('reference did not converge')
        z=integral
        if case['wave']=='Rayleigh':
            aa=a-f*f/cc
            values=[rho*(z[0,0]+z[2,2]),z[1,1]/cc+z[3,3]/l+k*k*aa*z[2,2],
                    aa*z[2,2]+(z[0,3]-f/cc*z[1,2])/k]
        else: values=[rho*z[0,0],z[1,1]/l+k*k*nn*z[0,0],nn*z[0,0]]
        return [mp.nstr(v,digits-5) for v in values]


def cases():
    rows=json.loads((ROOT/'tests/fixtures/regular_layer_reference.json').read_text())['cases']
    result={}
    for name,row in rows.items():
        c=row['input']; j=c['selected']; p=c['layers'][j]
        if not p[4]: continue
        result[name]=dict(material=p,h=c['thicknesses'][j],omega=c['omega'],c=c['c'],
                          wave=c['wave'],state=[.7,-.2,.3,.4])
    # Exact binary64 critical speeds, both sides, normal and reversed VTI roots.
    for model,p in [('iso',[2.,2.,1.,2.,.5,.5]),
                    ('vti',[3.1,34.,26.,28.,5.,6.5]),
                    ('complex_vti',[2.,30.,0.,20.,12.,10.])]:
        for label,index in [('vs',4),('vp',1)]:
            speed=math.sqrt(p[index]/p[0])
            for eps in [-1e-8,0.,1e-8]:
                result[f'{model}/{label}/{eps}']=dict(material=p,h=.2,omega=.7,
                    c=speed*(1+eps),wave='Rayleigh',state=[.7,-.2,.3,.4])
    for wave in ('Rayleigh','Love'):
        for h in (.002,1.,8.):
            result[f'normal/{wave}/{h}']=dict(material=[2.,2.,1.,2.,.5,.5],h=h,
                omega=.7,c=.4,wave=wave,state=[.7,-.2,.3,.4])
    result['uncoupled']=dict(material=[2.,2.,-.5,2.,.5,.5],h=.2,
                omega=.7,c=.5,wave='Rayleigh',state=[.7,-.2,.3,.4])
    result['coalescent']=dict(material=[1.,4.,.5,1.,1.,1.],h=.2,
                omega=.7,c=math.sqrt(15/16),wave='Rayleigh',state=[.7,-.2,.3,.4])
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    process=subprocess.Popen([str(args.probe.resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    report={'rtol':1e-10,'atol':1e-12,'cases':{}}
    failures=[]
    try:
        for name,case in cases().items():
            expected=reference(case)
            check=reference(case,90)
            with mp.workdps(95):
                convergence=max(float(abs(mp.mpf(a)-mp.mpf(b))/max(1,abs(mp.mpf(b)))) for a,b in zip(expected,check))
            if convergence>1e-60: raise RuntimeError('reference convergence: '+name)
            values=[int(case['wave']=='Love')]+case['material']+[case['h'],case['omega'],case['c']]+case['state']
            process.stdin.write(' '.join(map(str,values))+'\n');process.stdin.flush()
            actual=json.loads(process.stdout.readline())
            errors={}
            for path,vals in actual.items():
                errors[path]=[None if a is None else abs(a-float(b))/max(abs(float(b)),1e-30) for a,b in zip(vals,expected)]
            passed=all(a is not None and abs(a-float(b))<=1e-12+1e-10*abs(float(b)) for a,b in zip(actual['public'],expected))
            if not passed:failures.append(name)
            report['cases'][name]=dict(input=case,reference=expected,convergence=convergence,actual=actual,relative_errors=errors,passed=passed)
            print(name,errors,flush=True)
    finally:
        process.stdin.close(); process.wait()
    report['failures']=failures
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(report["cases"])-len(failures)}/{len(report["cases"])} passed')
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
