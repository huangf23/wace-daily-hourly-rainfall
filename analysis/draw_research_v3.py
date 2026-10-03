"""Paper composite figures from completed v3 analyses. No fitting or selection here."""
import os,json,string,argparse,sys
sys.modules.setdefault('draw_research_v3',sys.modules[__name__])
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap,SymLogNorm,Normalize,BoundaryNorm
from matplotlib.ticker import ScalarFormatter,FuncFormatter,NullFormatter,FixedLocator
from scipy.stats import genextreme
from research_v3_common import *

PAPER=ROOT/'manuscript_elsevier';FIG=PAPER/'figures_v3'
COLORS=['#777777','#0072B2','#C65D20','#00836B']
STYLES=[':','-','--','-.'];MARKERS=['x','o','s','^']
SEQUENTIAL=LinearSegmentedColormap.from_list('quiet_blue',['#F3F6F8','#D4E3EB','#93B8CF','#4D84A7','#204D70'])
DIVERGING=LinearSegmentedColormap.from_list('quiet_diverging',['#315D7C','#86ACC3','#FAF9F6','#C99381','#8D4E40'])
PERIOD_COLORS=['#3E6785','#AC684D']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.65,'pdf.fonttype':42,
    'svg.fonttype':'none','savefig.facecolor':'white'})
MANIFEST=[]

def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3,width=.65)

def panel(ax,label,title):
    ax.set_title(f'({label}) {title}',loc='left',pad=7,fontweight='normal');clean(ax)

def taxis(ax,short=False):
    ax.set_xscale('log');ax.set_xticks([2,10,100] if short else [2,5,10,25,50,100]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.minorticks_off();ax.set_xlabel('Return period (years)')


def difference_ticks(ax):
    """Keep the linear symlog center readable at final print size."""
    low,high=ax.get_xlim()
    ticks=[x for x in [-10,-1,-.1,0,.1,1,10] if low<=x<=high]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    ax.xaxis.set_minor_locator(FixedLocator([]))

def methodlines(ax,x,ys,selection=range(4),**kwargs):
    for m in selection:ax.plot(x,ys[m],color=COLORS[m],ls=STYLES[m],lw=1.3,label=METHODS[m],**kwargs)

def save(fig,name,title,caption,group='main'):
    folder=FIG/group;folder.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        path=folder/f'{name}.{ext}';tmp=folder/f'{name}.tmp.{ext}'
        fig.savefig(tmp,dpi=300,facecolor='white');os.replace(tmp,path)
    plt.close(fig)
    MANIFEST.append(dict(id=name,title=title,caption=caption,group=group,pdf=str(folder/f'{name}.pdf'),png=str(folder/f'{name}.png')))
    print('DRAWN '+name,flush=True)

def prepare():
    cache=FIG/'source_data.npz'
    if cache.exists():return dict(np.load(cache))
    geo=geometry();ci=geo['case_land_indices'];scores=[];heat=[];caseq=[];casetr=[];casepars=[];daily=[]
    relsummary=[];probsummary=[];relfirst=[];probfirst=[];fit=[];depend=[];overlap=[]
    for member in MEMBERS:
        p=np.load(OUT/'point_diagnostics'/f'predictions_{member}.npz');e=np.load(OUT/'point_diagnostics'/f'diagnostics_{member}.npz')
        q=p['pred'];tr=e['truth'];scores.append(e['scores']);caseq.append(q[...,ci]);casetr.append(tr[...,ci])
        hh=np.empty((2,4,7,len(T)))
        for j in range(7):
            err=np.log(q[:,j]/tr[j]);hh[0,:,j]=err.mean(axis=-1);hh[1,:,j]=abs(err).mean(axis=-1)
        heat.append(hh);casepars.append(np.stack([p['h'][...,ci],e['truth_parameters'][...,ci]]));daily.append(p['daily'][...,ci])
        rr=e['log_relations'];relsummary.append(np.nanquantile(rr,[.1,.5,.9],axis=-1));relfirst.append(rr[:,0])
        aa=e['A'];probsummary.append(np.quantile(aa,[.1,.5,.9],axis=-1));probfirst.append(aa[:,0])
        fit.append(e['fit_ks']);depend.append(e['rank_correlation']);overlap.append(e['window_overlap'])
        print('FIGURE SOURCES READ '+member,flush=True)
        del q,tr,rr,aa
    rf=np.array(relfirst);all_members=np.isfinite(rf).all(axis=0)
    # Only full-member raw support is summarized; its coverage is separately reported.
    med=np.nanmedian(rf,axis=0);med[~all_members]=np.nan
    rm=np.nanquantile(med,[.1,.5,.9],axis=-1)
    ap=np.median(np.array(probfirst),axis=0);am=np.quantile(ap,[.1,.5,.9],axis=-1)
    result=dict(scores=np.array(scores),heat=np.array(heat),caseq=np.array(caseq),casetr=np.array(casetr),
        casepars=np.array(casepars),daily=np.array(daily),rel_summary=np.array(relsummary),prob_summary=np.array(probsummary),
        rel_main=rm,rel_coverage=all_members.mean(axis=-1),A_main=am,fit_ks=np.array(fit),
        dependency=np.array(depend),overlap=np.array(overlap),land=geo['land'],cells=geo['cells'],
        x=geo['projection_x_coordinate']/1000,y=geo['projection_y_coordinate']/1000,
        latitude=geo['latitude'],longitude=geo['longitude'],T=T,case_ids=ci)
    np.savez_compressed(cache,**result)
    save_json(FIG/'source_provenance.json',dict(input_root=str(OUT),members=MEMBERS,methods=METHODS,
        fixed_cases=geo['cases'],score_unit='member-duration-land cell before any averaging',
        relation_summary='Spatial quantiles of within-cell member medians, all4 raw member relations required at a given T',
        case_policy='Historical fixed P50 point; member07 is a disclosed post-hoc counterexample at that same point',
        renderer='Matplotlib; final size 7.1 inches unless a supplementary matrix needs more height; journal-specific requirements pending'))
    return result

def fig2(d):
    means=d['scores'].mean(axis=-1);f,axs=plt.subplots(2,2,figsize=(7.1,5.4),layout='constrained')
    for v,ax,title,ylabel in zip(range(3),axs.ravel()[:3],['Overall quantile error','Signed log bias','Return-period error variation'],[r'Mean $D_{RL}$',r'Mean $B$',r'Mean $S_Q$']):
        for m in range(4):
            for i in range(4):ax.plot(DURATIONS,means[i,m,:,v],color=COLORS[m],lw=.6,alpha=.28)
            ax.plot(DURATIONS,means[:,m,:,v].mean(axis=0),color=COLORS[m],ls=STYLES[m],marker=MARKERS[m],ms=3,lw=1.7,label=METHODS[m])
        ax.set_xticks(DURATIONS);ax.set_xlabel('Duration (h)');ax.set_ylabel(ylabel);ax.grid(axis='y',color='.92',lw=.5)
        if v==1:ax.axhline(0,color='.4',lw=.7)
        else:ax.set_ylim(bottom=0)
        panel(ax,string.ascii_lowercase[v],title)
    axs[0,0].legend(frameon=False,ncol=2)
    ax=axs[1,1];overall=means.mean(axis=2)[:,:,0]
    for k,(m,b) in enumerate([(1,0),(2,1),(3,1)]):
        val=overall[:,m]-overall[:,b]
        ax.scatter(val,k+np.linspace(-.17,.17,4),s=19,color=COLORS[m],marker='o',alpha=.65)
        ax.scatter(val.mean(),k,s=52,color=COLORS[m],marker='D',edgecolor='white',lw=.6,zorder=4)
    ax.axvline(0,color='.4',lw=.7);ax.set_yticks(range(3),['UCF − NC','HQT − UCF','TPS − UCF']);ax.invert_yaxis();ax.set_xlabel(r'Paired difference in mean $D_{RL}$')
    panel(ax,'d','Gain relative to each benchmark')
    ax.text(.03,.03,'Small points: members\nDiamonds: four-member mean',transform=ax.transAxes,fontsize=6.7,va='bottom')
    save(f,'fig02_overall','Updating value and error components',
         'Panels a–c show land-mean scores for each member as thin lines and their equal-member mean as thick lines. NC retains the historical hourly curve. Panel d shows member-level differences and their mean, without assigning a confidence interval to member spread. Negative differences favor the first named method. All scores are formed at cells before aggregation.')

def fig3(d):
    f,axs=plt.subplots(2,2,figsize=(7.1,5.8),layout='constrained');member=0;case=1;j=2
    sample=np.load(OUT/'point_diagnostics'/'case_ams_01.npz');Tfull=np.geomspace(41/40,100,401)
    for k,(role,ax) in enumerate(zip(['daily','hourly'],axs[0])):
        for s,period in enumerate(['historical','future']):
            p=d['daily'][member,s,0,:,case] if role=='daily' else d['casepars'][member,s,j,:,case]
            x=sample[f'{period}_{role}'][:,0 if role=='daily' else j,case]
            q=genextreme.ppf(1-1/Tfull,c=-p[2],loc=p[0],scale=p[1])
            ax.plot(q,1/Tfull,color=PERIOD_COLORS[s],ls=['--','-'][s],lw=1.4,label=['Historical','Target'][s])
            ax.scatter(np.sort(x),1-np.arange(1,len(x)+1)/(len(x)+1),s=13,marker=['o','^'][s],facecolors='none',edgecolors=PERIOD_COLORS[s],lw=.7)
        ax.axhspan(.009,1/41,color='.9',alpha=.55,zorder=-1)
        ax.set_yscale('log');ax.set_ylim(.009,1);ax.set_xlabel('Annual maximum depth (mm)');ax.set_ylabel('Annual exceedance probability')
        panel(ax,string.ascii_lowercase[k],['Whole-day extreme distribution','3 h extreme distribution'][k])
    axs[0,0].legend(frameon=False,fontsize=7)
    ax=axs[1,0]
    for s in range(2):
        dp=d['daily'][0,s,0,:,case];hp=d['casepars'][0,s,j,:,case]
        qd=genextreme.ppf(1-1/T,c=-dp[2],loc=dp[0],scale=dp[1]);qh=genextreme.ppf(1-1/T,c=-hp[2],loc=hp[0],scale=hp[1])
        ax.plot(qd,qh,color=PERIOD_COLORS[s],ls=['--','-'][s],lw=1.4)
        for t,mk in [(2,'o'),(10,'s'),(100,'^')]:
            a=genextreme.ppf(1-1/t,c=-dp[2],loc=dp[0],scale=dp[1]);b=genextreme.ppf(1-1/t,c=-hp[2],loc=hp[0],scale=hp[1])
            ax.scatter(a,b,marker=mk,color=PERIOD_COLORS[s],s=25,edgecolor='white',lw=.4)
    ax.set_xlabel('Whole-day quantile (mm)');ax.set_ylabel('3 h quantile (mm)');panel(ax,'c','Same-probability correspondence')
    ax.text(.03,.97,'○ 2 yr   □ 10 yr   △ 100 yr',transform=ax.transAxes,va='top',fontsize=7)
    ax=axs[1,1]
    for s in range(2):
        for k,(p,label) in enumerate([(d['casepars'][0,s,j,:,case],'3 h'),(d['daily'][0,s,0,:,case],'1 day')]):
            q=genextreme.ppf(1-1/T,c=-p[2],loc=p[0],scale=p[1]);ax.plot(T,q/q[0],color=PERIOD_COLORS[s],ls=['--','-'][s],marker=['o','s'][k],markevery=45,ms=2.8,
                lw=1.2,label=f'{["Historical","Target"][s]}, {label}')
    taxis(ax);ax.set_ylabel(r'Tail growth $Q(T)/Q(2)$');panel(ax,'d','Scale-normalized tail growth');ax.legend(frameon=False,fontsize=6.3)
    save(f,'fig03_distributions','Marginal distributions and cross-duration correspondence',
        'Member01 at the fixed historical P50 3h case. Empirical positions are i/(n+1), with n=40; shading below 1/41 marks extrapolation beyond the largest empirical plotting position. Panels c and d compare marginal quantiles, not paired storms. This illustration does not estimate spatial prevalence; complete cases are in the supplement.')

def fig4(d):
    f,axs=plt.subplots(2,2,figsize=(7.1,5.6),layout='constrained');titles=['UCF: same-probability ratio','HQT: amount-based mapping','TPS: scaling discrepancy']
    for m,ax in enumerate(axs.ravel()[:3]):
        vals=np.exp(d['rel_main'][:,m]);ax.fill_between(T,vals[0],vals[2],color=COLORS[m+1],alpha=.18,lw=0)
        ax.plot(T,vals[1],color=COLORS[m+1],lw=1.5);ax.axhline(1,color='.3',ls=':',lw=.8)
        ax.set_yscale('log');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'));ax.yaxis.set_minor_formatter(NullFormatter());taxis(ax)
        low,high=ax.get_ylim();ax.yaxis.set_major_locator(FixedLocator([x for x in [.25,.5,.75,1,1.25,1.5,2,3,4] if low<=x<=high]))
        ax.set_ylabel('Target / historical relation');panel(ax,string.ascii_lowercase[m],titles[m])
        if m==1:
            coverage=d['rel_coverage'][m,-1]*100
            ax.text(.03,.03,f'100 yr raw-support coverage: {coverage:.1f}%',transform=ax.transAxes,fontsize=6.3)
    ax=axs[1,1]
    for m in [1,2,3]:
        vals=d['A_main'][:,m];ax.fill_between(T,vals[0],vals[2],color=COLORS[m],alpha=.09,lw=0)
        ax.plot(T,vals[1],color=COLORS[m],ls=STYLES[m],lw=1.4,label=METHODS[m])
    ax.axhline(1,color='.3',ls=':',lw=.8);taxis(ax);ax.set_ylabel(r'Exceedance ratio $A(T)$');ax.set_ylim(bottom=0);panel(ax,'d','Probability consequence');ax.legend(frameon=False)
    save(f,'fig04_relationships','Relationship changes and exceedance consequences',
        'All panels use 1h targets. Each cell is summarized by its median over the four members; lines and shading then show the spatial median and P10–P90. Raw relation summaries require all4 member values to be in valid support. Their coverage is disclosed; full boundary and clipping diagnostics remain in the source and supplement. A uses the regularized prediction and the fitted target-hourly reference, retaining zero exceedance probabilities. A>1 denotes a more frequent exceedance than nominal. These are explanatory re-expressions, not independent performance tests.')

def fig5(d):
    scores=d['scores'][:,:,:,0];pairs=[(1,0),(2,1),(3,1)]
    dif=[scores[:,m]-scores[:,b] for m,b in pairs];mn=min(x.min() for x in dif);mx=max(x.max() for x in dif)
    f,axs=plt.subplots(2,2,figsize=(7.1,5.5),layout='constrained')
    for k,(ax,(m,b)) in enumerate(zip(axs.ravel()[:3],pairs)):
        for j,ls in zip([0,5,6],['-','--',':']):
            x=np.sort(dif[k][:,j].ravel());ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=COLORS[m],ls=ls,lw=1.2,label=f'{DURATIONS[j]} h',rasterized=True)
        ax.axvline(0,color='.4',lw=.7);ax.set_xscale('symlog',linthresh=.03);ax.set_xlim(mn*1.04,mx*1.04);ax.set_ylim(0,1);difference_ticks(ax)
        ax.set_xlabel(r'Paired $\Delta D_{RL}$ (symlog, threshold 0.03)');ax.set_ylabel('Cumulative fraction');panel(ax,string.ascii_lowercase[k],f'{METHODS[m]} − {METHODS[b]}')
    axs[0,0].legend(frameon=False)
    ax=axs[1,1]
    for k,(m,b) in enumerate(pairs):
        ax.plot(DURATIONS,(dif[k]<0).mean(axis=(0,2))*100,color=COLORS[m],marker=MARKERS[m],ms=3,ls=STYLES[m],label=f'{METHODS[m]} − {METHODS[b]}')
    ax.axhline(50,color='.5',ls=':',lw=.7);ax.set_ylim(0,100);ax.set_xticks(DURATIONS);ax.set_xlabel('Duration (h)');ax.set_ylabel('Combinations improved (%)');panel(ax,'d','Prevalence of improvement');ax.legend(frameon=False,fontsize=6.7)
    save(f,'fig05_local_distributions','Distributions of paired local performance',
        'ECDFs retain all member–land-cell differences with equal member weights and show 1,6,12h. Symlog uses a linear region of ±0.03 and retains full tails. The final panel shows improvement fractions at all target durations. These describe heterogeneity; the member–cell entries are not independent statistical replicates. Complete member panels are supplementary.')

def spatial(ax,values,d,norm,cmap):
    arr=np.full(d['land'].shape,np.nan);arr.ravel()[d['cells']]=values
    im=ax.pcolormesh(d['x'],d['y'],arr,cmap=cmap,norm=norm,shading='nearest',rasterized=True)
    use=np.where(d['land']);ax.set_xlim(d['x'][use[1]].min()-15,d['x'][use[1]].max()+15);ax.set_ylim(d['y'][use[0]].min()-15,d['y'][use[0]].max()+15)
    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
    return im

def fig6(d):
    score=d['scores'][:,:,:,0];pairs=[(1,0),(2,1),(3,1)]
    dif=np.array([score[:,m,0]-score[:,b,0] for m,b in pairs]);med=np.median(dif,axis=1);count=(dif<0).sum(axis=1)
    lim=float(abs(med).max());norm=SymLogNorm(linthresh=.03,vmin=-lim,vmax=lim,base=10)
    f,axs=plt.subplots(2,3,figsize=(7.1,7.5),layout='constrained')
    for k,(m,b) in enumerate(pairs):
        im=spatial(axs[0,k],med[k],d,norm,DIVERGING);panel(axs[0,k],string.ascii_lowercase[k],f'{METHODS[m]} − {METHODS[b]}')
        im2=spatial(axs[1,k],count[k],d,BoundaryNorm(np.arange(-.5,5.5),SEQUENTIAL.N),SEQUENTIAL);panel(axs[1,k],string.ascii_lowercase[k+3],'Members with improvement')
    cb=f.colorbar(im,ax=axs[0],orientation='horizontal',fraction=.045,pad=.015,aspect=40);cb.set_label(r'Member-median $\Delta D_{RL}$ (symlog; negative favors first)')
    cb=f.colorbar(im2,ax=axs[1],orientation='horizontal',fraction=.045,pad=.015,aspect=40,ticks=range(5));cb.set_label('Count out of four members; not a significance measure')
    save(f,'fig06_spatial','Spatial magnitude and member agreement',
        '1h targets. Top: the cellwise median of the four member-specific score differences. Bottom: the count of members in which that same paired difference is negative. Equal values are not counted as improvements. Shared top-row normalization retains extremes with a symlog threshold0.03. Counts express member consistency, not probabilities or significance. Full individual-member maps and 6/12h counterparts are supplementary.')

def fig7(d):
    f,axs=plt.subplots(3,2,figsize=(7.1,7.3),layout='constrained');case=1;j=2
    for col,i in enumerate([0,2]):
        q=d['caseq'][i,:,j,:,case];tr=d['casetr'][i,j,:,case];hist=q[0]
        curves=[q,q/hist,np.log(q/tr)];refs=[tr,tr/hist,np.zeros_like(T)]
        for row in range(3):
            ax=axs[row,col];methodlines(ax,T,curves[row]);ax.plot(T,refs[row],color='black',lw=1.4,label='Reference')
            taxis(ax);panel(ax,string.ascii_lowercase[row*2+col],f'Member {MEMBERS[i]}: '+['depth','change factor','log error'][row])
            ax.set_ylabel(['Depth (mm)','Target / historical (log axis)','ln(predicted / reference)'][row])
            if row==1:ax.set_yscale('log');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'));ax.axhline(1,color='.5',ls=':',lw=.6)
            if row==2:ax.axhline(0,color='.5',ls=':',lw=.6)
    for row in range(3):
        low=min(axs[row,c].get_ylim()[0] for c in range(2));high=max(axs[row,c].get_ylim()[1] for c in range(2))
        for c in range(2):axs[row,c].set_ylim(low,high)
    axs[0,0].legend(frameon=False,ncol=2,fontsize=6.4)
    save(f,'fig07_cases','Local amount, climate signal and tail error',
        'The same historical P50 3h cell is shown for member01 (fixed illustration) and member07 (a disclosed post-hoc counterexample). Rows show absolute rainfall, change relative to the corresponding historical curve, and log error. Member07 TPS predicts a decrease over the curve while the fitted reference increases. The panels are complementary transformations of the same predictions, not independent evidence. All four members at all three fixed cells are supplementary.')

def grid_ci(d):
    rows=[]
    for i,member in enumerate(MEMBERS):
        b=np.load(OUT/'joint_bootstrap'/member/'joint_results.npz');agg=b['aggregate'];point=d['scores'][i].mean(axis=(1,3))
        for m,b0 in [(1,0),(2,1),(3,1)]:
            delta=agg[m,0]-agg[b0,0];valid=np.isfinite(delta);qs=np.quantile(delta[valid],[.025,.5,.975])
            rows.append(dict(member=member,comparison=f'{METHODS[m]} minus {METHODS[b0]}',point=point[m,0]-point[b0,0],low=qs[0],median=qs[1],high=qs[2],valid_replicates=int(valid.sum())))
    frame=pd.DataFrame(rows);frame.to_csv(FIG/'grid_paired_CI.csv',index=False);return frame

def fig8(d):
    ci=grid_ci(d);f,axs=plt.subplots(2,2,figsize=(7.1,6.0),layout='constrained')
    f.legend([Line2D([],[],color=COLORS[k+1],marker=MARKERS[k+1],lw=1) for k in range(3)],
             ['Panel a: UCF − NC','Panel a: HQT − UCF','Panel a: TPS − UCF'],
             loc='outside upper center',ncol=3,frameon=False,fontsize=6.7)
    ax=axs[0,0]
    for i,memb in enumerate(MEMBERS):
        for k,label in enumerate(['UCF minus NC','HQT minus UCF','TPS minus UCF']):
            r=ci[(ci.member==memb)&(ci.comparison==label)].iloc[0];y=i+(k-1)*.22
            ax.hlines(y,r.low,r.high,color=COLORS[k+1],lw=1.2);ax.scatter(r.point,y,color=COLORS[k+1],s=18,marker=MARKERS[k+1])
    ax.axvline(0,color='.4',ls=':',lw=.7);ax.set_yticks(range(4),MEMBERS);ax.invert_yaxis();ax.set_ylabel('Member');ax.set_xlabel(r'Paired $\Delta D_{RL}$, percentile 95% CI');panel(ax,'a','Uncertainty in overall differences')
    ax=axs[0,1];w=np.load(OUT/'uncertainty_audit'/'land_widths.npz')['W']
    for m in range(4):
        vals=np.sort(w[:,m,0,1].ravel());ax.step(vals,np.arange(1,len(vals)+1)/len(vals),color=COLORS[m],ls=STYLES[m],lw=1.2,label=METHODS[m],rasterized=True)
    ax.set_xlabel(r'Width $W=\ln(Q_{97.5}/Q_{2.5})$');ax.set_ylabel('Cumulative fraction');ax.set_ylim(0,1);panel(ax,'b','Local width: 1 h, 100 years');ax.legend(frameon=False,fontsize=6.5)
    ax=axs[1,0];b=np.load(OUT/'joint_bootstrap'/'01'/'joint_results.npz');qci=b['case_prediction_ci'][:,2,:,:,1];tci=b['case_reference_ci'][:,2,:,1]
    ax.fill_between(T,tci[0],tci[2],color='.4',alpha=.12,lw=0);ax.plot(T,d['casetr'][0,2,:,1],color='black',lw=1.4,label='Reference')
    for m in [1,3]:
        ax.fill_between(T,qci[0,m],qci[2,m],color=COLORS[m],alpha=.14,lw=0);ax.plot(T,d['caseq'][0,m,2,:,1],color=COLORS[m],ls=STYLES[m],lw=1.4,label=METHODS[m])
    taxis(ax);ax.set_ylabel('3 h depth (mm)');panel(ax,'c','Fixed P50 case, member 01');ax.legend(frameon=False,fontsize=6.5)
    ax=axs[1,1];curves=[]
    for member in MEMBERS:
        b=np.load(OUT/'joint_bootstrap'/member/'joint_results.npz');r=b['support_counts'].mean(axis=1)/5;curves.append(r);ax.plot(DURATIONS,r,color='.6',lw=.8)
    ax.plot(DURATIONS,np.mean(curves,axis=0),color='.15',lw=1.7,marker='o',ms=3);ax.set_xticks(DURATIONS);ax.set_xlabel('Duration (h)');ax.set_ylabel('Resamples with support flags (%)');ax.set_ylim(bottom=0);panel(ax,'d','Fitted-support diagnostic')
    save(f,'fig08_uncertainty','Sampling uncertainty and interval diagnostics',
        'Panel a refits the target-hourly reference using the same target-year draws as the daily inputs; 500 original 3-year block draws are retained per member. Symbols are original point estimates and bars are bootstrap percentiles, which need not enclose the points; basic-interval sensitivity is in Fig. S23. Panel b is a descriptive spatial/member distribution of quantile-estimate interval widths, including all tails. Panel c contrasts UCF and TPS against a jointly resampled reference at one fixed case; other methods/cases are supplementary. Panel d shows the land-mean fraction of resamples with any required observed value outside its fitted GEV support; thin lines identify members and the thick line is their mean. Finite predictions do not certify physically credible tail bounds.')

def fig9(d):
    f,axs=plt.subplots(2,2,figsize=(7.1,5.7),layout='constrained');s=pd.read_csv(OUT/'controls'/'sample_length_scores.csv');w=pd.read_csv(OUT/'controls'/'sample_length_widths.csv')
    for m,method in enumerate(METHODS):
        a=s[s.method==method].groupby('n_years').mean(numeric_only=True);b=w[w.method==method].groupby('n_years').mean(numeric_only=True)
        axs[0,0].plot(a.index,a.mean_D,color=COLORS[m],ls=STYLES[m],marker=MARKERS[m],ms=3,label=method)
        axs[0,1].plot(b.index,b.W_median,color=COLORS[m],ls=STYLES[m],marker=MARKERS[m],ms=3)
    for ax in axs[0]:ax.set_xticks([15,20,30,40]);ax.set_xlabel('Input years per period');ax.set_ylim(bottom=0)
    axs[0,0].set_ylabel(r'Mean $D_{RL}$');axs[0,1].set_ylabel('Mean spatial-median width W')
    panel(axs[0,0],'a','Record-length sensitivity');panel(axs[0,1],'b','100-year interval sensitivity');axs[0,0].legend(frameon=False,ncol=2)
    sim=pd.read_csv(OUT/'controls'/'synthetic_summary.csv');names=sim.scenario.unique();labels=['S0','S1','S2','S3','S4']
    for m,method in enumerate(METHODS):
        x=np.arange(5)+(m-1.5)*.18;r=sim[sim.method==method].set_index('scenario').loc[names]
        axs[1,0].bar(x,r.population_D,width=.16,color=COLORS[m],alpha=.85)
        ax=axs[1,1];v=r.finite_sample_median_D-r.population_D
        ax.errorbar(x,v,yerr=np.vstack([r.finite_sample_median_D-r.finite_sample_low,r.finite_sample_high-r.finite_sample_median_D]),fmt=MARKERS[m],color=COLORS[m],ms=3,lw=.7,capsize=1.5)
    for ax in axs[1]:ax.set_xticks(range(5),labels);ax.set_xlabel('Prespecified synthetic scenario');ax.axhline(0,color='.4',lw=.6)
    axs[1,0].set_ylabel(r'Population $D_{RL}$');axs[1,1].set_ylabel(r'$D_{sample}-D_{population}$')
    panel(axs[1,0],'c','Discrepancy with known parameters');panel(axs[1,1],'d','Finite-sample estimation increment')
    save(f,'fig09_controls','Sample length and controlled distribution conditions',
        'Panels a–b use128 fixed historical-intensity-stratified cells per member,200 circular3-year resamples for each input length, and a fixed full40-year target-hourly reference. The displayed width averages within-member/duration spatial medians. Synthetic scenarios are S0 no change,S1 common amplitude,S2 extra subdaily change,S3 hourly tail-shape change,S4 changed temporal scaling. Panel c uses known population parameters. Panel d shows the median and P2.5–P97.5 sampling range of the estimation increment over1000 correlated40-year samples. These ranges are not CIs for the Monte Carlo mean. Simulations illustrate the specified marginal conditions, not physical storm sequences or empirical validation.')

def fig10(d):
    source=OUT/'stations';ind=pd.read_csv(source/'individual_D.csv',dtype={'station_id':str});ind=ind[(ind.block_length==3)&(ind.policy=='original_admissible')&ind.primary]
    diff=pd.read_csv(source/'aggregate_differences.csv');diff=diff[(diff.block_length==3)&(diff.policy=='original_admissible')&(diff.group=='primary_6')]
    cohort=pd.read_csv(ROOT/'midas_station_validation_7'/'transfer_v1'/'cohort.csv',dtype={'station_id':str});st=cohort[cohort.primary]
    f,axs=plt.subplots(2,2,figsize=(7.1,6.0),layout='constrained');ax=axs[0,0]
    labels=[]
    for i,s in enumerate(st.itertuples()):
        labels.append(s.station_name.replace('-',' ').title())
        for m,method in enumerate(METHODS):
            r=ind[(ind.station_id==s.station_id)&(ind.method==method)].iloc[0];y=i+(m-1.5)*.16
            ax.hlines(y,r.low,r.high,color=COLORS[m],lw=.8);ax.scatter(r.point,y,s=14,color=COLORS[m],marker=MARKERS[m])
    ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel(r'Station-mean $D_{RL}$ (log axis)');panel(ax,'a','Individual stations and sampling intervals')
    ax=axs[0,1]
    for k,label in enumerate(['UCF minus NC','HQT minus UCF','TPS minus UCF']):
        r=diff[diff.comparison==label].iloc[0];ax.hlines(k,r.low,r.high,color=COLORS[k+1],lw=1.4);ax.scatter(r.point,k,color=COLORS[k+1],marker=MARKERS[k+1],s=24)
    ax.axvline(0,color='.4',ls=':',lw=.7);ax.set_yticks(range(3),['UCF − NC','HQT − UCF','TPS − UCF']);ax.invert_yaxis();ax.set_xlabel(r'Paired $\Delta D_{RL}$, percentile 95% CI');panel(ax,'b','Six-station aggregate comparison')
    p=np.load(source/'point_predictions.npz');ev=np.load(source/'point_evaluation.npz');band=np.load(source/'bands_block3.npz')['reference']
    raw=pd.read_csv(ROOT/'midas_station_validation_7'/'transfer_v1'/'heldout_late_hourly.csv',dtype={'station_id':str})
    for col,sid in enumerate(['00708','00889']):
        i=int(np.flatnonzero(cohort.station_id==sid)[0]);j=2;ax=axs[1,col]
        methodlines(ax,T,p['q'][:,j,IDS,i]);tr=gev_curves(ev['truthp'][j,:,i:i+1],reduced_variate(T))[:,0]
        ax.fill_between(T,band[0,j,:,i],band[2,j,:,i],color='.5',alpha=.12,lw=0);ax.plot(T,tr,color='black',lw=1.4,label='Reference')
        y=np.sort(raw[(raw.station_id==sid)&(raw.duration_h==3)].ams_mm.to_numpy());tt=(len(y)+1)/(len(y)+1-np.arange(1,len(y)+1));use=tt>=2
        ax.scatter(tt[use],y[use],color='black',s=12,marker='.',zorder=5)
        taxis(ax);ax.set_ylabel('3 h depth (mm)');panel(ax,string.ascii_lowercase[col+2],f'{cohort.iloc[i].station_name.replace("-"," ").title()}, n={len(y)}')
    axs[1,0].legend(frameon=False,ncol=2,fontsize=6.3)
    save(f,'fig10_stations','Historical station evidence and local exceptions',
        'Six primary stations are retained;Boulmer is supplementary because the historical complete-year sample is14. Panels a–b use the joint2000-draw3-calendar-year block bootstrap and original fit admissibility. Panel a averages scores over target durations within each station. Heathrow and Boscombe Down are disclosed post-hoc contrasting cases; the full station set supports aggregate inference. Black dots in c–d are empirical target-period AMS positions and shading is the reference interval. Station results are historical complementary tests and do not directly validate future gridded rainfall.')

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    for part in ['point_diagnostics','stations','controls','joint_bootstrap','uncertainty_audit','supplementary']:
        marker=OUT/part/'complete.json';assert marker.exists(),f'Computation unfinished: {part}'
        assert json.loads(marker.read_text(encoding='utf-8'))['status']=='complete'
    d=prepare()
    for fun in [fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10]:fun(d)
    from draw_supplement_v3 import draw_all
    draw_all(d)
    save_json(FIG/'manifest.json',dict(status='rendered_pending_visual_review',figure1='reserved as instructed',figures=MANIFEST,
        method_palette=dict(zip(METHODS,COLORS)),all_computations_completed_before_rendering=True))

if __name__=='__main__':main()
