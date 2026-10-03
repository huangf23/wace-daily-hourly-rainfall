"""Focused manuscript figures from frozen outputs; no fitting or selection."""
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.stats import genextreme
import draw_research_v3 as old
from draw_research_v3 import PAPER, OUT, METHODS, MEMBERS, DURATIONS, T, COLORS, STYLES, MARKERS, panel, taxis
from draw_visual_revision_wace import value_grid

DEST=PAPER/'figures_wace'/'focused_revision_20261002'
DEST.mkdir(exist_ok=True,parents=True)
old.FIG=DEST
d=dict(np.load(PAPER/'figures_v3'/'source_data.npz'))
CAP={}; DATA={}
PAIRS=[(1,0),(2,1),(3,1)]; LABELS=['UCF − NC','HQT − UCF','TPS − UCF']
def save(f,name,caption):
    CAP[name]=caption
    for ext in ['pdf','png','svg']: f.savefig(DEST/(name+'.'+ext),dpi=300,facecolor='white')
    plt.close(f);print('Rendered '+name,flush=True)
def legend(ax):
    ax.legend([Line2D([],[],color=COLORS[m],ls=STYLES[m],marker=MARKERS[m],ms=3) for m in range(4)],METHODS,frameon=False,ncol=2,fontsize=6.5)

def overall():
    f=plt.figure(figsize=(7.1,5.0),layout='constrained');g=f.add_gridspec(2,3,width_ratios=[1.1,1.1,1])
    axs=[f.add_subplot(g[0,:2]),f.add_subplot(g[0,2]),f.add_subplot(g[1,2]),f.add_subplot(g[1,:2])]
    means=d['scores'].mean(axis=-1)
    for v in range(3):
        ax=axs[v]
        for m in range(4):
            for i in range(4):ax.plot(DURATIONS,means[i,m,:,v],color=COLORS[m],lw=.6,alpha=.3)
            ax.plot(DURATIONS,means[:,m,:,v].mean(axis=0),color=COLORS[m],ls=STYLES[m],marker=MARKERS[m],ms=3,lw=1.8)
        ax.set_xticks([1,3,6,12]);ax.set_xlabel('Duration (h)');ax.set_ylabel([r'Mean $D_{RL}$',r'Mean $B$',r'Mean $S_Q$'][v]);ax.axhline(0,color='.8',lw=.6)
        panel(ax,'abd'[v],['Overall error','Signed bias','Curve distortion'][v])
    legend(axs[0]);ax=axs[3];point=means[:,:,:,0].mean(axis=-1)
    for k,(m,b) in enumerate(PAIRS):
        vals=point[:,m]-point[:,b]
        for i in range(4):ax.scatter(vals[i],k+(i-1.5)*.11,color=COLORS[m],marker=['o','s','^','v'][i],s=18)
        ax.scatter(vals.mean(),k,s=45,color=COLORS[m],marker='D',edgecolor='white',zorder=5)
    ax.axvline(0,color='.4',ls=':',lw=.8);ax.set_yticks(range(3),LABELS);ax.invert_yaxis();ax.set_xlabel(r'Paired difference in mean $D_{RL}$');panel(ax,'c','Aggregate gain and member spread')
    save(f,'fig02_overall',r'Updating value and explanatory error metrics. (a,b,d) Four thin lines per method show member land means; thick lines show their equal-member mean. (c) Small circles, squares, upward and downward triangles show contrasts for members 01, 04, 07 and 08 averaged over durations; diamonds are their equal-member means. Negative values favor the first named method. All errors are calculated locally before aggregation. Member spread is descriptive, not a confidence interval.')

def local():
    scores=d['scores'][:,:,:,0];dif=np.array([scores[:,m]-scores[:,b] for m,b in PAIRS]);qs=np.quantile(dif,[0,.1,.25,.5,.75,.9,1],axis=(1,3))
    f=plt.figure(figsize=(7.1,5.2),layout='constrained');g=f.add_gridspec(2,3,height_ratios=[1.2,1]);axs=[f.add_subplot(g[0,k]) for k in range(3)]
    for k,ax in enumerate(axs):
        for y,j in enumerate([0,5,6]):
            z=qs[:,k,j];ax.hlines(y,z[0],z[-1],color=COLORS[k+1],lw=.5,alpha=.5);ax.hlines(y,z[1],z[5],color=COLORS[k+1],lw=2,alpha=.4);ax.hlines(y,z[2],z[4],color=COLORS[k+1],lw=7,alpha=.65);ax.scatter(z[3],y,s=18,color='white',edgecolor=COLORS[k+1],zorder=4);ax.scatter(dif[k,:,j].mean(),y+.16,color='black',marker='D',s=12,zorder=5)
        ax.axvline(0,color='.4',ls=':',lw=.7);ax.set_xscale('symlog',linthresh=.03);ax.set_yticks(range(3),['1 h','6 h','12 h']);ax.set_ylim(2.5,-.5);ax.set_xlabel(r'Paired $\Delta D_{RL}$');panel(ax,chr(97+k),LABELS[k]);old.difference_ticks(ax)
    ax=f.add_subplot(g[1,:]);freq=(dif<0).mean(axis=(1,3))*100
    im=value_grid(ax,freq,DURATIONS,LABELS,old.SEQUENTIAL,Normalize(0,100),'.1f');ax.set_xlabel('Duration (h)');panel(ax,'d','Local improvement frequency (%)')
    DATA['paired_quantiles']=qs;DATA['improvement_percent']=freq
    save(f,'fig03_local',r'Local gains and penalties. (a--c) Equal-weight member--cell distributions at selected durations: hairlines span the full range, medium lines P10--P90, thick bars P25--P75 and open circles the median; black diamonds mark means. Symmetric-log axes share a linear threshold of $\pm0.03$ but have panel-specific limits to retain each full range. (d) Percentage of comparisons favoring the first method at each duration. These are descriptive distributions, not independent replicates or confidence intervals. Complete ECDFs remain in Fig.~S7.')

def maps():
    scores=d['scores'][:,:,:,0];dif=np.array([scores[:,m,0]-scores[:,b,0] for m,b in PAIRS]);med=np.median(dif,axis=1);counts=(dif<0).sum(axis=1)
    f,axs=plt.subplots(2,3,figsize=(7.1,6.4),gridspec_kw={'height_ratios':[1.25,1]},layout='constrained');lim=abs(med).max()
    for k in range(3):
        im=old.spatial(axs[0,k],med[k],d,SymLogNorm(.03,vmin=-lim,vmax=lim),old.DIVERGING);panel(axs[0,k],chr(97+k),LABELS[k])
        im2=old.spatial(axs[1,k],counts[k],d,BoundaryNorm(np.arange(-.5,5.5),old.SEQUENTIAL.N),old.SEQUENTIAL);panel(axs[1,k],chr(100+k),'Member agreement')
    cb=f.colorbar(im,ax=axs[0],orientation='horizontal',fraction=.055,pad=.02,aspect=45);cb.set_label(r'Member-median $\Delta D_{RL}$; negative favors first')
    cb=f.colorbar(im2,ax=axs[1],orientation='horizontal',fraction=.055,pad=.02,aspect=45,ticks=range(5));cb.set_label('Number of improving members')
    save(f,'fig04_spatial',r'Spatial performance at 1 h. Columns show UCF--NC, HQT--UCF and TPS--UCF. (a--c) Within-cell median of four member contrasts on a shared zero-centered symmetric-log scale (linear threshold 0.03). (d--f) Number of members with negative contrasts, using a separate discrete scale. Native land cells are shown without smoothing; white indicates cells outside the mask. Agreement counts are not significance probabilities. Individual-member and additional-duration maps are in Figs.~S4--S6.')

def distributions():
    f,axs=plt.subplots(2,2,figsize=(7.1,5.8),layout='constrained');sample=np.load(OUT/'point_diagnostics'/'case_ams_01.npz');tf=np.geomspace(41/40,100,401)
    for k,role in enumerate(['daily','hourly']):
        ax=axs[0,k]
        for s,period in enumerate(['historical','future']):
            p=d['daily'][0,s,0,:,1] if k==0 else d['casepars'][0,s,2,:,1];x=sample[f'{period}_{role}'][:,0 if k==0 else 2,1]
            q=genextreme.ppf(1-1/tf,c=-p[2],loc=p[0],scale=p[1]);ax.plot(q,1/tf,color=old.PERIOD_COLORS[s],ls=['--','-'][s],lw=1.4,label=['Historical','Target'][s]);ax.scatter(np.sort(x),1-np.arange(1,len(x)+1)/(len(x)+1),s=12,marker=['o','^'][s],facecolors='none',edgecolors=old.PERIOD_COLORS[s],lw=.6)
        ax.axhspan(.009,1/41,color='.92');ax.set_yscale('log');ax.set_ylim(.009,1);ax.set_xlabel('Annual maximum depth (mm)');ax.set_ylabel('Exceedance probability');panel(ax,chr(97+k),['Whole-day marginal, member 01','3 h marginal, member 01'][k])
    axs[0,0].legend(frameon=False)
    for k,i in enumerate([0,2]):
        ax=axs[1,k];old.methodlines(ax,T,d['caseq'][i,:,2,:,1]);ax.plot(T,d['casetr'][i,2,:,1],color='black',lw=1.5,label='Reference');taxis(ax);ax.set_ylabel('3 h depth (mm)');panel(ax,chr(99+k),f'Transferred tail, member {MEMBERS[i]}')
    lo=min(ax.get_ylim()[0] for ax in axs[1]);hi=max(ax.get_ylim()[1] for ax in axs[1]);[ax.set_ylim(lo,hi) for ax in axs[1]];axs[1,0].legend(frameon=False,ncol=3,fontsize=6.1)
    save(f,'fig05_tails',r'Extreme distributions and local transfer at the historically fixed P50 cell ($53.11^\circ$N, $2.56^\circ$W). (a,b) Empirical plotting positions and fitted whole-day and 3-h survival curves for member 01. Shading below $1/41$ identifies extrapolation beyond the largest plotting position. (c,d) Predicted 3-h return levels and target-hourly references at the same coordinate. Member 01 is the fixed illustration; member 07 is a disclosed post-hoc counterexample. The bottom panels share a depth scale. Complete cases, quantile correspondence and transfer-relation diagnostics are in Figs.~S8--S13 and S27.')

def controls():
    sim=pd.read_csv(OUT/'controls'/'synthetic_summary.csv');names=sim.scenario.unique();f,(a,b)=plt.subplots(1,2,figsize=(7.1,4.0),gridspec_kw={'width_ratios':[1,1.25]},layout='constrained')
    mat=sim.pivot(index='scenario',columns='method',values='population_D').loc[names,METHODS]
    im=value_grid(a,mat.values,METHODS,['S0','S1','S2','S3','S4'],old.SEQUENTIAL,Normalize(0,mat.values.max()),'.3f');panel(a,'a','Known-parameter discrepancy');f.colorbar(im,ax=a,orientation='horizontal',pad=.07,shrink=.8).set_label(r'Population $D_{RL}$')
    for m in range(4):
        r=sim[sim.method==METHODS[m]].set_index('scenario').loc[names];y=np.arange(5)+(m-1.5)*.17
        b.hlines(y,r.finite_sample_low,r.finite_sample_high,color=COLORS[m],alpha=.55,lw=.9);b.scatter(r.finite_sample_median_D,y,color=COLORS[m],marker=MARKERS[m],s=18);b.scatter(r.finite_sample_mean_D,y,s=25,facecolors='none',edgecolors=COLORS[m],marker='D')
    b.set_yticks(range(5),['S0','S1','S2','S3','S4']);b.invert_yaxis();b.set_xlabel(r'Finite-sample $D_{RL}$');panel(b,'b','Forty-year estimation');legend(b)
    save(f,'fig06_controls',r'Transfer assumptions under known distributions. S0: no change; S1: common amplification; S2: extra sub-daily amplification; S3: hourly shape change; S4: changed temporal scaling. (a) Error using known population parameters. (b) Errors from 1,000 correlated 40-year samples per scenario: method symbols are medians, open diamonds means, and lines P2.5--P97.5 sampling ranges. Ranges describe individual sample estimates, not confidence intervals for Monte Carlo means. These scenarios specify statistical conditions, not diagnosed physical regimes at UK cells.')

def length():
    s=pd.read_csv(OUT/'controls'/'sample_length_scores.csv');w=pd.read_csv(OUT/'controls'/'sample_length_widths.csv');f,(a,b)=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for m in range(4):
        sub=s[s.method==METHODS[m]]
        for _,r in sub.groupby('member'):
            v=r.groupby('n_years').mean_D.mean();a.plot(v.index,v,color=COLORS[m],alpha=.3,lw=.6)
        v=sub.groupby('n_years').mean_D.mean();a.plot(v.index,v,color=COLORS[m],ls=STYLES[m],marker=MARKERS[m],ms=3,lw=1.6)
    a.set_xticks([15,20,30,40]);a.set_xlabel('Input years per period');a.set_ylabel(r'Mean $D_{RL}$');panel(a,'a','Error with shorter input records');legend(a)
    mat=w.groupby(['method','n_years']).W_median.mean().unstack().loc[METHODS];im=value_grid(b,mat.values,mat.columns,METHODS,old.SEQUENTIAL,Normalize(0,mat.values.max()),'.2f');b.set_xlabel('Input years per period');panel(b,'b','100-year interval width');f.colorbar(im,ax=b,orientation='horizontal',pad=.07,shrink=.8).set_label('Mean spatial-median W')
    save(f,'fig07_length',r'Cost of shorter input records at 128 historically stratified cells per member. Each length uses 200 circular 3-year resamples and a fixed full-40-year target-hourly reference. (a) Error averaged across cells, durations and resamples; thin lines show members and thick lines their equal-weight mean. (b) Quantile-estimate interval widths at 100 years, averaged over within-member/duration spatial medians. The reference is held fixed to isolate sensitivity to input length; width values are descriptive summaries.')

def uncertainty():
    f,axs=plt.subplots(2,2,figsize=(7.1,5.8),layout='constrained');ci=old.grid_ci(d)
    ax=axs[0,0]
    for i,member in enumerate(MEMBERS):
        for k,(m,b) in enumerate(PAIRS):
            r=ci[(ci.member==member)&(ci.comparison==f'{METHODS[m]} minus {METHODS[b]}')].iloc[0];y=i+(k-1)*.22;ax.hlines(y,r.low,r.high,color=COLORS[m],lw=1.2);ax.scatter(r.point,y,s=18,color=COLORS[m],marker=MARKERS[m])
    ax.axvline(0,color='.4',ls=':',lw=.8);ax.set_yticks(range(4),MEMBERS);ax.invert_yaxis();ax.set_ylabel('Member');ax.set_xlabel(r'Paired $\Delta D_{RL}$');panel(ax,'a','Aggregate percentile intervals')
    ax=axs[0,1];boot=np.load(OUT/'joint_bootstrap'/'01'/'joint_results.npz');point=d['scores'][0].mean(axis=(1,3))[:,0]
    for k,(m,b) in enumerate(PAIRS):
        x=np.sort(boot['aggregate'][m,0]-boot['aggregate'][b,0]-(point[m]-point[b]));ax.step(x,np.arange(1,len(x)+1)/len(x),color=COLORS[m],ls=STYLES[m],lw=1.2,label=LABELS[k],rasterized=True)
    ax.axvline(0,color='.4',ls=':',lw=.8);ax.set_ylabel('Cumulative fraction');ax.set_xlabel('Resampled contrast − original contrast');panel(ax,'b','Interval displacement, member 01');ax.legend(frameon=False,fontsize=6.4)
    ax=axs[1,0];w=np.load(OUT/'uncertainty_audit'/'land_widths.npz')['W'];vals=[w[:,m,0,1].ravel() for m in range(4)];bp=ax.boxplot(vals,vert=False,patch_artist=True,whis=(5,95),showfliers=True,widths=.5,flierprops=dict(marker='.',markersize=1.2,alpha=.12,rasterized=True),medianprops=dict(color='black'))
    for m in range(4):bp['boxes'][m].set(facecolor=COLORS[m],alpha=.35);bp['fliers'][m].set_color(COLORS[m])
    ax.set_yticks(range(1,5),METHODS);ax.invert_yaxis();ax.set_xlabel(r'Width $W=\ln(Q_{.975}/Q_{.025})$');panel(ax,'c','Local widths: 1 h, 100 years')
    ax=axs[1,1];qci=boot['case_prediction_ci'][:,2,:,:,1];tci=boot['case_reference_ci'][:,2,:,1]
    for m in [1,3]:ax.fill_between(T,qci[0,m],qci[2,m],color=COLORS[m],alpha=.13,lw=0);ax.plot(T,d['caseq'][0,m,2,:,1],color=COLORS[m],ls=STYLES[m],label=METHODS[m])
    ax.fill_between(T,tci[0],tci[2],color='.3',alpha=.12,lw=0);ax.plot(T,d['casetr'][0,2,:,1],color='black',label='Reference');taxis(ax,True);ax.set_ylabel('3 h depth (mm)');panel(ax,'d','Fixed P50 case, member 01');ax.legend(frameon=False,fontsize=6.4)
    save(f,'fig08_uncertainty',r'Sampling stability from 500 circular 3-year draws per member. (a) Original paired aggregate contrasts and percentile 95\% intervals; colors and symbols identify the contrasts in panel b. (b) Full resampling distributions centered on the original contrasts for member 01; a distribution displaced from zero explains why percentile intervals may not contain the point estimate. All-member basic/percentile comparisons are in Fig.~S23. (c) Member--cell distributions of interval widths: P25--P75 boxes, medians, P5--P95 whiskers and all outside values. (d) Point predictions and pointwise 95\% intervals at the fixed 3-h case, including the jointly resampled target-hourly reference. These intervals have not been validated for nominal coverage.')

if __name__=='__main__':
    for fn in [overall,local,maps,distributions,controls,length,uncertainty]:fn()
    (DEST/'captions.json').write_text(json.dumps(CAP,indent=2),encoding='utf-8');np.savez_compressed(DEST/'summary_data.npz',**DATA)
    files=[PAPER/'figures_v3'/'source_data.npz',OUT/'controls'/'synthetic_summary.csv',OUT/'controls'/'sample_length_scores.csv',OUT/'controls'/'sample_length_widths.csv',OUT/'uncertainty_audit'/'land_widths.npz']+[OUT/'joint_bootstrap'/m/'joint_results.npz' for m in MEMBERS]+[OUT/'point_diagnostics'/'case_ams_01.npz']
    provenance=dict(date='2026-10-02',status='rendered_pending_visual_review',refitting=False,future_hourly_role='evaluation only; all source predictions already frozen',figures=list(CAP),inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (DEST/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
