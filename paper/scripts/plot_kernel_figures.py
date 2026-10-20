#!/usr/bin/env python3
"""Plot Figures 4--6 from archived arrays; no numerical solver is required."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import generate_kernel_figures as g
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'legend.handlelength':3.5,'pdf.fonttype':42})
FIG=g.PAPER/'Fig'
FIG.mkdir(parents=True, exist_ok=True)
model=json.loads((g.DATA/'prem_like_kernel_model.json').read_text())
z=np.array(model['depth_km']);materials=np.array(model['materials'])

def load(stem):
    report=json.loads((g.DATA/(stem+'_report.json')).read_text())
    assert report['data_sha256']==g.sha(g.DATA/(stem+'.npz'))
    assert report['model_sha256']==g.sha(g.DATA/'prem_like_kernel_model.json')
    return np.load(g.DATA/(stem+'.npz')),report

def save(fig,name):
    for suffix in ('pdf','png'):fig.savefig(FIG/(name+'.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)

def physical(a):
    rho,A,C,F,L,N=a.T
    return np.array([rho,np.sqrt(A/rho),np.sqrt(C/rho),np.sqrt(N/rho),np.sqrt(L/rho),F/(A-2*L)]).T

a,report=load('prem_convergence_updated')
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
for ax,wave in zip(axes,('Rayleigh','Love')):
    raw=a[wave+'_raw'];diag=a[wave+'_diagonal'];ref=diag[-1];levels=np.arange(8);h=2.**(-levels)
    floor=max(report['root_tolerance']/abs(ref),abs(diag[-2]-ref)/abs(ref),10*np.finfo(float).eps)
    for v,color,marker,label in ((raw,'0.45','o','Piecewise constant'),(diag,'#1765ad','s','Romberg diagonal')):
        err=abs(v[:8]-ref)/abs(ref);ax.loglog(h,np.maximum(err,floor),color=color,linestyle='-' if marker=='o' else '--',marker=marker,ms=4,label=label)
        small=err<=floor
        ax.scatter(h[small],np.full(sum(small),floor),marker='v',s=40,color=color,zorder=5)
    ax.axhline(floor,color='0.65',lw=.8,ls=':',label='Reference resolution')
    ad=a[wave+'_adaptive'];err=abs(ad[0]-ref)/abs(ref)
    ax.scatter([2.**(-ad[3])],[max(err,floor)],marker='*',s=105,facecolor='#e49b13',edgecolor='black',lw=.5,zorder=6,label='Adaptive result')
    ax.set(title=f'{wave}, 0.15 Hz',xlabel=r'Relative layer thickness $h/h_0$',ylabel='Relative phase-velocity error')
    ax.text(.04,.42,f'Adaptive level: {int(ad[3])}\nEstimated error: {ad[4]:.2e}\nObserved error: {err:.2e}',transform=ax.transAxes,fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    ax.grid(alpha=.2,which='major');ax.legend(loc='upper left');ax.set_ylim(floor*.4,.1)
save(fig,'prem_convergence_updated')

a,report=load('prem_group_kernels_updated')
fig,axes=plt.subplots(1,4,figsize=(7.2,3.5),sharey=True)
colors=['0.45','#1765ad','#39a9b8','#bc243c','#e58c21','#42954b']
styles=['-', '--', ':', '-.', (0,(6,2,1,2,1,2)), (0,(3,1,1,1))]
labels=[r'$\rho$',r'$V_{ph}$',r'$V_{pv}$',r'$V_{sh}$',r'$V_{sv}$',r'$\eta$']
keys=['Rayleigh_05','Rayleigh_10','Love_05','Love_10']
for ax,key in zip(axes,keys):
    p=a[key+'_profile_l8'];mat=a[key+'_materials_l8'][p[:,0].astype(int)];U=report['cases'][key]['U']
    continuous=g.velocity_basis(p[:,8:14],mat)*physical(mat)/U
    wave=key.split('_')[0];frequency='0.05' if key.endswith('05') else '0.10'
    for i,(color,label) in enumerate(zip(colors,labels)):
        xx=[];dd=[]
        for j in range(len(z)-1):
            indices=np.flatnonzero((p[:,1]>z[j])&(p[:,1]<z[j+1]))
            if not len(indices):continue
            indices=indices[np.unique(np.linspace(0,len(indices)-1,min(1000,len(indices))).astype(int))]
            # Extend the nearest interior samples to the exact interface depth.
            # Repeated depths produce horizontal jump connectors, not smoothing.
            xx.extend([continuous[indices[0],i],*continuous[indices,i],continuous[indices[-1],i]])
            dd.extend([z[j],*p[indices,1],z[j+1]])
        ax.plot(xx,dd,color=color,linestyle=styles[i],lw=1.2,label=label)
    ax.axvline(0,color='0.7',lw=.5,zorder=0);ax.grid(alpha=.15)
    ax.set_ylim(100,0);ax.set_title(f'{wave}\n{frequency} Hz')
    ax.tick_params(axis='x',labelsize=7)
    ax.locator_params(axis='x',nbins=3)
axes[0].set_ylabel('Flattened depth (km)')
fig.supxlabel(r'$(p/U)K_{U,p}$ (km$^{-1}$)',fontsize=9,y=.01)
fig.legend([Line2D([],[],color=c,linestyle=st) for c,st in zip(colors,styles)],labels,ncol=6,loc='upper center',bbox_to_anchor=(.5,1.04),frameon=False)
fig.tight_layout(rect=(0,.045,1,.96));save(fig,'prem_group_kernels_updated')

if (g.DATA/'prem_model_updated.npz').exists():
    a,report=load('prem_model_updated')
    fig,(left,right)=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'width_ratios':[1,1.3]},layout='constrained')
    pv=physical(materials)
    model_styles={4:['-','--',':'],3:['-.',(0,(6,2,1,2,1,2)),(0,(3,1,1,1))]}
    for index,color,name in ((4,'#1765ad',r'$V_{sv}$'),(3,'#bc243c',r'$V_{sh}$')):
        for j in range(len(z)-1):
            t=np.linspace(0,1,100) if z[j+1]>z[j] else np.array([0.,1.])
            mat=(1-t[:,None])*materials[j]+t[:,None]*materials[j+1]
            left.plot(physical(mat)[:,index],(1-t)*z[j]+t*z[j+1],color=color,linestyle=model_styles[index][0],label=name+' grid' if j==0 else None)
        for level,ls in ((0,'--'),(1,':')):
            mat,h,_=g.mesh(model,level);d=np.r_[0,np.cumsum(h)];v=physical(mat)[:,index]
            left.plot(np.repeat(v,2),np.column_stack((d[:-1],d[1:])).ravel(),color=color,ls=model_styles[index][level+1],lw=1.1,label=name+f' level {level}')
    left.set(xlabel='Shear velocity (km/s)',ylabel='Flattened depth (km)',ylim=(z[-1],0));left.legend(fontsize=6)
    for wave,color in (('Rayleigh','#1765ad'),('Love','#bc243c')):
        wave_styles=model_styles[4 if wave=='Rayleigh' else 3]
        for level,ls in ((0,'--'),(1,':')):right.plot(a['frequency_hz'],a[wave+'_raw'][level],color=color,ls=wave_styles[level+1],lw=1.2,label=f'{wave}, level {level}')
        right.plot(a['frequency_hz'],a[wave+'_diagonal'],color=color,ls=wave_styles[0],label=f'{wave}, Romberg')
    right.set(xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)');right.legend(fontsize=6)
    for ax in (left,right):ax.grid(alpha=.2)
    save(fig,'prem_model_updated')

# Refresh provenance after every visual edit, without rerunning the solver.
import platform
paths=[g.PAPER/'scripts'/name for name in ('plot_kernel_figures.py','generate_kernel_figures.py','generate_model_figure.py','continuous_group_kernel.cpp','verify_kernel_figures.py')]
paths += [g.DATA/'prem_like_kernel_model.json']
for stem in ('prem_model_updated','prem_convergence_updated','prem_group_kernels_updated'):
    paths += [g.DATA/(stem+s) for s in ('.npz','_report.json')]
    paths += [FIG/(stem+s) for s in ('.pdf','.png')]
paths += [g.DATA/'prem_group_kernels_verification.json']
manifest={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'sha256':{str(p.relative_to(g.PAPER)):g.sha(p) for p in paths}}
(g.PAPER/'results').mkdir(parents=True, exist_ok=True)
(g.PAPER/'results/kernel_figures_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
