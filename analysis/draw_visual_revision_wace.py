"""Alternative encodings of frozen results; no refitting or outcome selection."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import draw_research_v3 as old
from draw_research_v3 import COLORS, METHODS, MEMBERS, DURATIONS, T, OUT, PAPER, panel, clean

DEST=PAPER/'figures_wace'/'visual_revision'
DEST.mkdir(parents=True,exist_ok=True)
SOURCE=PAPER/'figures_v3'/'source_data.npz'
d=dict(np.load(SOURCE))
old.FIG=DEST
CAPTIONS={}
TABLES={}

def capture(fn):
    captured=[]
    original=old.save
    old.save=lambda fig,*args,**kwargs:captured.append(fig)
    try: fn(d)
    finally: old.save=original
    return captured[0]

def save(f,name,caption):
    CAPTIONS[name]=caption
    for ext in ['png','pdf','svg']:
        f.savefig(DEST/(name+'.'+ext),dpi=300,facecolor='white')
    plt.close(f)
    print('Rendered '+name,flush=True)

def value_grid(ax,values,xlabels,ylabels,cmap,norm,fmt='.3f'):
    im=ax.imshow(values,aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
    for (i,j),v in np.ndenumerate(values):
        rgba=np.array(im.cmap(im.norm(v))[:3]);linear=np.where(rgba<=.04045,rgba/12.92,((rgba+.055)/1.055)**2.4);lum=linear@np.array([.2126,.7152,.0722])
        ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=7,color='white' if lum<.179 else 'black')
    ax.set_xticks(range(len(xlabels)),xlabels);ax.set_yticks(range(len(ylabels)),ylabels)
    ax.tick_params(length=0);ax.spines[:].set_visible(False)
    return im

def relationships():
    f,axs=plt.subplots(2,2,figsize=(7.1,5.7),layout='constrained')
    rt=np.array([2,5,10,25,50,100]);y=np.arange(6)
    # Interpolate the already computed log-ratio summaries on the log-T grid.
    vals=np.empty((3,3,6))
    for m,ax in enumerate(axs.ravel()[:3]):
        vals[:,m]=np.exp(np.array([np.interp(np.log(rt),np.log(T),d['rel_main'][k,m]) for k in range(3)]))
        v=vals[:,m]
        ax.hlines(y,v[0],v[2],color=COLORS[m+1],lw=3,alpha=.35)
        ax.scatter(v[1],y,color=COLORS[m+1],marker=old.MARKERS[m+1],s=27,zorder=3)
        ax.axvline(1,color='.35',ls=':',lw=.8);ax.set_yticks(y,rt);ax.invert_yaxis()
        ax.set_xscale('log');ax.xaxis.set_major_formatter(old.FuncFormatter(lambda x,p:f'{x:g}'));ax.xaxis.set_minor_formatter(old.NullFormatter())
        ax.set_xlim(min(.85,v.min()*.95),max(1.15,v.max()*1.05))
        ax.set_xlabel('Target / historical relation (log scale)');ax.set_ylabel('Return period (years)')
        panel(ax,chr(97+m),[r'UCF: ratio $K$',r'HQT: mapping $G$',r'TPS: discrepancy $C$'][m])
        if m==1: ax.text(.02,.02,f'100 yr support coverage: {d["rel_coverage"][m,-1]*100:.1f}%',transform=ax.transAxes,fontsize=6.4)
    ax=axs[1,1]
    low,high=vals.min()*.95,vals.max()*1.05
    for a in axs.ravel()[:3]:
        a.set_xlim(low,high);a.set_ylim(6,-.45)
        a.xaxis.set_major_locator(old.FixedLocator([x for x in [.25,.5,.75,1,1.25,1.5,2,3,4] if low<=x<=high]))
    for m in [1,2,3]:
        v=d['A_main'][:,m];ax.fill_between(T,v[0],v[2],color=COLORS[m],alpha=.09,lw=0)
        ax.plot(T,v[1],color=COLORS[m],ls=old.STYLES[m],lw=1.4,label=METHODS[m])
    ax.axhline(1,color='.35',ls=':',lw=.8);old.taxis(ax);ax.set_ylim(bottom=0);ax.set_ylabel(r'Exceedance ratio $A(T)$');panel(ax,'d','Probability consequence');ax.legend(frameon=False)
    TABLES['relation_intervals']=vals;TABLES['relation_return_periods']=rt
    save(f,'fig04_relationships',r'Changes in transfer relationships and their probability consequences at 1 h. (a--c) Points and horizontal ranges show spatial medians and P10--P90 of the target/historical ratios of $K$, $G$ and $C$, after taking each cell\textquoteright s four-member median; $G$ is evaluated at the target daily quantile. Selected return periods interpolate log-ratio summaries on the log-$T$ grid. One denotes preservation. Raw ratios require all four members within valid support; HQT coverage at 100 years is shown. (d) Spatial median and P10--P90 of $A(T)$ from regularized predictions, retaining zero probabilities; $A>1$ denotes exceedance more frequent than $1/T$. Ranges and bands describe spatial variation, not confidence intervals.')

def local_distributions():
    scores=d['scores'][:,:,:,0];pairs=[(1,0),(2,1),(3,1)]
    diff=np.array([scores[:,m]-scores[:,b] for m,b in pairs])
    # Equal-width signed-log bins. Heights are fractions per bin, NOT densities.
    threshold=.03
    transform=lambda x:np.sign(x)*np.log1p(abs(x)/threshold)
    inverse=lambda z:np.sign(z)*threshold*np.expm1(abs(z))
    extent=max(abs(transform(diff.min())),abs(transform(diff.max())))*1.00001
    zedges=np.linspace(-extent,extent,57);edges=inverse(zedges)
    assert edges[0]<=diff.min() and edges[-1]>=diff.max()
    f,axs=plt.subplots(2,2,figsize=(7.1,5.8),layout='constrained')
    hist=np.empty((3,3,len(edges)-1));ticks=np.array([-10,-1,-.1,0,.1,1,10]);ticks=ticks[abs(transform(ticks))<=extent]
    for k,ax in enumerate(axs.ravel()[:3]):
        m,b=pairs[k]
        for ii,j in enumerate([0,5,6]):
            counts,_=np.histogram(diff[k,:,j].ravel(),edges);assert counts.sum()==diff[k,:,j].size
            hist[k,ii]=counts/counts.sum()*100
            if ii==0: ax.stairs(hist[k,ii],zedges,fill=True,color=COLORS[m],alpha=.17,lw=0)
            ax.stairs(hist[k,ii],zedges,color=COLORS[m],lw=1.2,ls=['-','--',':'][ii],label=f'{DURATIONS[j]} h')
        ax.axvline(0,color='.4',lw=.7);ax.set_xticks(transform(ticks),[f'{x:g}' for x in ticks]);ax.set_xlim(-extent,extent);ax.set_ylim(bottom=0)
        ax.set_xlabel(r'Paired $\Delta D_{RL}$ (signed-log spacing)');ax.set_ylabel('Member-cell combinations / bin (%)')
        panel(ax,chr(97+k),f'{METHODS[m]} - {METHODS[b]}')
    ymax=max(ax.get_ylim()[1] for ax in axs.ravel()[:3])
    for ax in axs.ravel()[:3]:ax.set_ylim(0,ymax)
    axs[0,0].legend(frameon=False)
    prevalence=(diff<0).mean(axis=(1,3))*100
    ax=axs[1,1];im=value_grid(ax,prevalence,DURATIONS,['UCF - NC','HQT - UCF','TPS - UCF'],old.SEQUENTIAL,Normalize(0,100),'.0f')
    ax.set_xlabel('Duration (h)');panel(ax,'d','Combinations improved (%)')
    cb=f.colorbar(im,ax=ax,orientation='horizontal',pad=.05,shrink=.9,ticks=[0,50,100]);cb.set_label('Negative paired difference (%)',fontsize=7)
    TABLES['difference_bin_edges']=edges;TABLES['difference_histogram_percent']=hist;TABLES['improvement_percent']=prevalence
    save(f,'fig05_local_distributions',r'Local distributions of paired performance. (a--c) Histograms of member--land-cell differences at 1, 6 and 12 h; heights are percentages per bin. All values are retained in 56 common bins equally spaced in $z=\mathrm{sign}(\Delta)\log(1+|\Delta|/0.03)$; horizontal ticks show the original differences. Solid, dashed and dotted outlines distinguish durations; the solid histogram is also shaded. Full unbinned ECDFs are in Fig.~S7. (d) Percentages with negative differences at every target duration. Member--cell entries receive equal weights and describe spatial/member variation. Pairwise improvement differs from ranking best among all methods.')

def cases():
    f,axs=plt.subplots(3,2,figsize=(7.1,7.3),layout='constrained');errors=[];change=[]
    for col,i in enumerate([0,2]):
        q=d['caseq'][i,:,2,:,1];tr=d['casetr'][i,2,:,1];hist=q[0]
        ax=axs[0,col];old.methodlines(ax,T,q);ax.plot(T,tr,color='black',lw=1.4,label='Reference');old.taxis(ax);ax.set_ylabel('3 h depth (mm)');panel(ax,chr(97+col),f'Member {MEMBERS[i]}: rainfall depth')
        vals=np.vstack([q/hist,tr/hist]);change.append(vals)
        ax=axs[1,col]
        for m in range(5):
            color=(COLORS+['black'])[m];lo,hi=vals[m,[0,-1]]
            ax.plot([lo,hi],[m,m],color=color,lw=1.4,alpha=.65)
            ax.scatter(lo,m,color=color,marker='o',s=25);ax.scatter(hi,m,color=color,marker='s',s=25,facecolors='white',linewidths=1)
        ax.set_yticks(range(5),METHODS+['Reference']);ax.invert_yaxis();ax.axvline(1,color='.45',ls=':',lw=.8);ax.set_xscale('log');ax.xaxis.set_major_formatter(old.FuncFormatter(lambda x,p:f'{x:g}'));ax.xaxis.set_minor_formatter(old.NullFormatter());ax.set_xlabel('Target / historical (log scale)');panel(ax,chr(99+col),'Change factors: 2 and 100 years')
        errors.append(np.log(q/tr))
    axs[0,0].legend(frameon=False,ncol=2,fontsize=6.4)
    axs[1,0].legend([old.Line2D([],[],marker='o',color='.3',ls='none'),old.Line2D([],[],marker='s',color='.3',markerfacecolor='white',ls='none')],['2 years','100 years'],loc='lower right',frameon=False,fontsize=6.4)
    norm=TwoSlopeNorm(vmin=-max(abs(np.array(errors)).max(),.01),vcenter=0,vmax=max(abs(np.array(errors)).max(),.01))
    for col in range(2):
        ax=axs[2,col];im=ax.pcolormesh(T,np.arange(4),errors[col],shading='nearest',cmap=old.DIVERGING,norm=norm,rasterized=True)
        old.taxis(ax);ax.set_yticks(range(4),METHODS);ax.invert_yaxis();panel(ax,chr(101+col),'Local error along the probability axis')
    cb=f.colorbar(im,ax=axs[2],orientation='horizontal',pad=.03,shrink=.82,aspect=40);cb.set_label('ln(predicted / reference)',fontsize=7)
    for row in [0,1]:
        limits=[a.get_ylim() if row==0 else a.get_xlim() for a in axs[row]];lo=min(x[0] for x in limits);hi=max(x[1] for x in limits)
        for ax in axs[row]: ax.set_ylim(lo,hi) if row==0 else ax.set_xlim(lo,hi)
    for ax in axs[1]:
        lo,hi=ax.get_xlim();ax.xaxis.set_major_locator(old.FixedLocator([x for x in [.5,.75,1,1.5,2,3,4,5,6] if lo<=x<=hi]))
    TABLES['case_log_errors']=np.array(errors);TABLES['case_change_factors']=np.array(change)
    save(f,'fig07_cases',r'Local rainfall amount, change and error at the same historical P50 cell ($53.11^\circ$N, $2.56^\circ$W), 3 h. Member 01 is the fixed illustration and member 07 is a disclosed post-hoc counterexample. (a,b) Return-level curves. (c,d) Filled circles and open squares show change factors at 2 and 100 years; connectors join the two return periods, not uncertainty bounds. (e,f) Log prediction errors over the full probability grid, with a common zero-centered color scale. TPS projects a decrease in member 07 although the fitted reference increases. These panels transform the same predictions. All members and three historically selected cells remain in Figs.~S8--S10.')

def uncertainty():
    f=capture(old.fig8);axs=f.axes
    ax=axs[1];ax.clear();w=np.load(OUT/'uncertainty_audit'/'land_widths.npz')['W']
    values=[w[:,m,0,1].ravel() for m in range(4)]
    bp=ax.boxplot(values,vert=False,patch_artist=True,showfliers=True,whis=(5,95),widths=.5,
        flierprops=dict(marker='.',markersize=1.4,alpha=.15,rasterized=True),medianprops=dict(color='black',lw=1.1))
    for m in range(4):
        bp['boxes'][m].set(facecolor=COLORS[m],alpha=.3,edgecolor=COLORS[m]);bp['fliers'][m].set_color(COLORS[m])
        ax.scatter(np.median(w[:,m,0,1],axis=-1),np.full(4,m+1)+np.linspace(-.12,.12,4),marker='D',s=10,color=COLORS[m],edgecolor='white',lw=.3,zorder=5)
    ax.set_yticks(range(1,5),METHODS);ax.invert_yaxis();ax.set_xlabel(r'Width $W=\ln(Q_{97.5}/Q_{2.5})$');panel(ax,'b','Local width: 1 h, 100 years')
    ax=axs[3];ax.clear();curves=[]
    for member in MEMBERS:
        z=np.load(OUT/'joint_bootstrap'/member/'joint_results.npz');curves.append(z['support_counts'].mean(axis=1)/5)
    curves=np.array(curves);x=np.arange(7)
    ax.bar(x,curves.mean(axis=0),color='#b6bec3',width=.66)
    for i in range(4):ax.scatter(x+(i-1.5)*.1,curves[i],color='.2',s=12,marker=['o','s','^','D'][i],label=MEMBERS[i],zorder=3)
    ax.set_xticks(x,DURATIONS);ax.set_ylim(bottom=0);ax.set_xlabel('Duration (h)');ax.set_ylabel('Resamples with support flags (%)');panel(ax,'d','Mean bars and individual members');ax.legend(frameon=False,ncol=4,fontsize=6,loc='lower right')
    TABLES['interval_width_spatial_quantiles']=np.quantile(np.array(values),[0,.05,.25,.5,.75,.95,1],axis=1);TABLES['support_percent_by_member']=curves
    save(f,'fig08_uncertainty',r'Sampling uncertainty and tail diagnostics. (a) Paired land- and duration-mean differences from 500 circular 3-year draws per member; symbols are original estimates and bars are percentile 95\% intervals. Figure~S23 compares basic intervals. (b) Spatial/member distributions of quantile-estimate interval widths at 1 h and 100 years: boxes span P25--P75, center lines are medians, whiskers are P5--P95, and all outside values are plotted. Diamonds mark four member medians. These spreads describe heterogeneity of interval widths. (c) UCF, TPS and jointly resampled reference intervals at the fixed P50 case, member 01, 3 h. (d) Bars show the four-member mean support-flag frequency; symbols show members. Flagged draws remain under the original admissibility policy.')

def controls():
    f=capture(old.fig9);s=pd.read_csv(OUT/'controls'/'sample_length_scores.csv');w=pd.read_csv(OUT/'controls'/'sample_length_widths.csv')
    for k,(source,col,title,cblab) in enumerate([(s,'mean_D','Record-length sensitivity',r'Mean $D_{RL}$'),(w,'W_median','100-year interval sensitivity','Mean spatial-median W')]):
        ax=f.axes[k];ax.clear();matrix=source.groupby(['method','n_years'])[col].mean().unstack().loc[METHODS]
        im=value_grid(ax,matrix.values,matrix.columns,METHODS,old.SEQUENTIAL,Normalize(0,matrix.values.max()),'.3f' if k==0 else '.2f');panel(ax,chr(97+k),title);ax.set_xlabel('Input years per period')
        cb=f.colorbar(im,ax=ax,orientation='horizontal',pad=.035,shrink=.85);cb.set_label(cblab,fontsize=7)
        TABLES['length_'+col]=matrix.values
    f.axes[2].legend([plt.Rectangle((0,0),1,1,color=c) for c in COLORS],METHODS,frameon=False,ncol=2,fontsize=6.5)
    save(f,'fig09_controls',r'Sample length and specified distribution conditions. (a,b) Annotated matrices show sensitivity at 128 historically stratified cells per member, with 200 circular 3-year draws at each input length and a fixed full-40-year target-hourly reference. Widths average within-member/duration spatial medians; the two matrices use separate color scales and display their values. (c) Population-parameter discrepancy. (d) Median and P2.5--P97.5 of the finite-sample increment over 1,000 correlated 40-year samples. S0: no change; S1: common amplitude change; S2: extra sub-daily amplification; S3: hourly shape change; S4: changed temporal scaling. Sampling ranges are not confidence intervals for the Monte Carlo mean.')

if __name__=='__main__':
    for fn in [relationships,local_distributions,cases,uncertainty,controls]:fn()
    CAPTIONS.update({
        'fig04_relationships':r'Transfer relationships and probability consequences at 1 h. (a--c) Spatial medians and P10--P90 of target/historical $K$, $G$ and $C$ ratios, after cellwise four-member medians; $G$ uses the target daily quantile. Selected periods interpolate log-ratio summaries on log-$T$. One denotes preservation. Raw ratios require all four members within valid support; HQT coverage is shown. (d) Spatial median and P10--P90 of regularized $A(T)$, including zeros; $A>1$ denotes exceedance more frequent than $1/T$. Ranges describe spatial variation.',
        'fig05_local_distributions':r'Local paired performance. (a--c) Histograms of member--cell differences at 1, 6 and 12 h, retaining all values. Heights are percentages per bin: 56 common bins equally spaced in $z=\mathrm{sign}(\Delta)\log(1+|\Delta|/0.03)$; ticks show original differences. Solid, dashed and dotted outlines distinguish durations. Unbinned ECDFs remain in Fig.~S7. (d) Percentages with negative differences at each duration. Equally weighted member--cell entries describe heterogeneity, not independent replicates. Pairwise improvement differs from ranking best.',
        'fig07_cases':r'Local amount, change and error at the historical P50 cell ($53.11^\circ$N, $2.56^\circ$W), 3 h. Member 01 is the fixed illustration; member 07 is a post-hoc counterexample. (a,b) Return levels. (c,d) Filled circles and open squares show 2- and 100-year change factors; connectors join return periods, not uncertainty bounds. (e,f) Log errors across the probability grid with a common zero-centered scale. TPS predicts a decrease in member 07 while the reference increases. Complete cases remain in Figs.~S8--S10.',
        'fig08_uncertainty':r'Sampling uncertainty and tail diagnostics. (a) Paired aggregate differences from 500 circular 3-year draws per member; symbols are original estimates and bars are percentile 95\% intervals (basic intervals: Fig.~S23). (b) Spatial/member distributions of 1-h, 100-year interval widths: P25--P75 boxes, median lines, P5--P95 whiskers and all outside values; diamonds mark member medians. This spread describes heterogeneity of widths. (c) UCF, TPS and jointly resampled reference intervals at the P50 case, member 01, 3 h. (d) Mean support-flag frequency (bars) and individual members (symbols); flagged draws remain under original admissibility.',
        'fig09_controls':r'Record length and specified distribution conditions. (a,b) Annotated matrices use 128 historically stratified cells per member, 200 circular 3-year draws per length and the full-40-year target reference. Widths average within-member/duration spatial medians; matrices have separate color scales. (c) Population-parameter discrepancy. (d) Median and P2.5--P97.5 sampling ranges of the estimation increment over 1,000 correlated 40-year samples. S0: no change; S1: common amplification; S2: extra sub-daily amplification; S3: hourly shape change; S4: changed temporal scaling. Ranges are not confidence intervals for Monte Carlo means.'
    })
    np.savez_compressed(DEST/'visual_summary_data.npz',**TABLES)
    (DEST/'captions.json').write_text(json.dumps(CAPTIONS,indent=2),encoding='utf-8')
    provenance=dict(date='2026-10-01',source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        refitting=False,model_members=MEMBERS,land_cells=10397,width_inches=7.1,encodings={'4':'Interval dots plus probability ribbons','5':'Signed-log binned frequencies plus annotated prevalence matrix','7':'Return-level curves plus endpoint dumbbells plus error maps','8':'Bootstrap forest plus descriptive boxplots plus uncertainty curves plus member dots on bars','9':'Record-length matrices plus population bars and sampling ranges'},
        hist_bins='56 equal signed-log bins over the complete global paired-difference range, threshold 0.03; bin heights are percentages, no KDE or density interpretation',
        boxplot='Descriptive widths: P25/P75 box, median, P5/P95 whiskers, all fliers; diamonds member medians',
        preserved_figures=[2,3,6,10],color_policy='Fixed method colors across manuscript; transforms and variables determine sequential/diverging scales')
    (DEST/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
