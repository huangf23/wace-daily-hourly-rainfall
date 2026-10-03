"""Complete member/case displays and diagnostic figures accompanying the main composites."""
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,SymLogNorm,TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter,FuncFormatter
from scipy.stats import genextreme
import draw_research_v3 as v
from research_v3_common import *

def export(f,id,title,caption):v.save(f,id,title,caption,'supplement')

def metrics(d):
    means=d['scores'].mean(axis=-1);f,axs=plt.subplots(4,3,figsize=(7.1,8.7),layout='constrained',sharex=True,sharey='col')
    for i,member in enumerate(MEMBERS):
        for j in range(3):
            ax=axs[i,j];v.methodlines(ax,DURATIONS,means[i,:,:,j]);ax.set_xticks(DURATIONS)
            v.panel(ax,string.ascii_lowercase[i*3+j],['D','B','S_Q'][j]+f' | member {member}')
            ax.set_ylabel(['Mean D','Mean B','Mean S_Q'][j])
            if j==1:ax.axhline(0,color='.4',lw=.6)
            else:ax.set_ylim(bottom=0)
            if i==3:ax.set_xlabel('Duration (h)')
    axs[0,0].legend(frameon=False,fontsize=6,ncol=2)
    export(f,'S01_member_metrics','Complete member-level score curves','Each panel uses one member and all10,397 land cells; no AMS or GEV parameters are pooled. NC is the unchanged historical-hourly baseline. S_Q is error variation across return period, not GEV shape-parameter error.')

def heatmaps(d):
    edges=np.exp(np.r_[np.log(T)[0],(np.log(T)[1:]+np.log(T)[:-1])/2,np.log(T)[-1]])
    for typ,id,title in [(0,'S02_signed_error','Signed log error'),(1,'S03_absolute_error','Absolute log error')]:
        z=d['heat'][:,typ];lim=float(abs(z).max());norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim) if typ==0 else Normalize(0,lim)
        f,axs=plt.subplots(4,4,figsize=(7.1,8.0),layout='constrained',sharex=True,sharey=True)
        for i,member in enumerate(MEMBERS):
            for m,method in enumerate(METHODS):
                ax=axs[i,m];im=ax.pcolormesh(edges,np.arange(8)-.5,z[i,m],norm=norm,cmap=v.DIVERGING if typ==0 else v.SEQUENTIAL,shading='flat',rasterized=True)
                v.taxis(ax,True);ax.set_yticks(range(7),DURATIONS);v.panel(ax,string.ascii_lowercase[i*4+m],f'{member} | {method}')
                if i<3:ax.set_xlabel('')
                if m==0:ax.set_ylabel('Duration (h)')
        cb=f.colorbar(im,ax=axs,orientation='horizontal',fraction=.025,pad=.025,aspect=40);cb.set_label('Land mean ln(prediction / reference)' if typ==0 else 'Land mean |ln(prediction / reference)|')
        export(f,id,title,'All four members, four methods and seven target durations. A shared color scale is used across the entire figure; errors are calculated at cells before averaging. Return period is logarithmic. Signed averages may conceal cancellation; the absolute counterpart is supplied separately.')

def maps(d):
    s=d['scores'][:,:,:,0];pairs=[(1,0),(2,1),(3,1)]
    selected=np.array([s[:,m,[0,5,6]]-s[:,b,[0,5,6]] for m,b in pairs]);lim=float(abs(selected).max());norm=SymLogNorm(linthresh=.03,vmin=-lim,vmax=lim)
    for j,id in zip([0,5,6],['S04_maps_1h','S05_maps_6h','S06_maps_12h']):
        f,axs=plt.subplots(4,3,figsize=(7.1,10.1),layout='constrained')
        for i,member in enumerate(MEMBERS):
            for k,(m,b) in enumerate(pairs):
                ax=axs[i,k];im=v.spatial(ax,s[i,m,j]-s[i,b,j],d,norm,v.DIVERGING)
                v.panel(ax,string.ascii_lowercase[i*3+k],f'{member}: {METHODS[m]} − {METHODS[b]}')
        cb=f.colorbar(im,ax=axs,orientation='horizontal',fraction=.015,pad=.015,aspect=45);cb.set_label('Paired ΔD (symlog threshold0.03; negative favors first)')
        export(f,id,f'Individual-member maps: {DURATIONS[j]}h','Rows retain the separate members and columns compare UCF−NC,HQT−UCF,TPS−UCF. Shared normalization across the1/6/12h supplementary maps retains the full range, with a symlog linear region ±0.03. These maps describe spatial heterogeneity without a cellwise significance claim.')

def ecdf(d):
    score=d['scores'][:,:,:,0];pairs=[(1,0),(2,1),(3,1)];colors=plt.get_cmap('viridis')(np.linspace(.1,.9,7));f,axs=plt.subplots(4,3,figsize=(7.1,8.5),layout='constrained',sharex=True,sharey=True)
    lims=[np.inf,-np.inf]
    for i,member in enumerate(MEMBERS):
        for k,(m,b) in enumerate(pairs):
            ax=axs[i,k]
            for j,dur in enumerate(DURATIONS):
                vals=np.sort(score[i,m,j]-score[i,b,j]);lims=[min(lims[0],vals[0]),max(lims[1],vals[-1])]
                ax.step(vals,np.arange(1,len(vals)+1)/len(vals),color=colors[j],ls=['-','--',':','-.','-','--',':'][j],lw=.8,label=f'{dur}h',rasterized=True)
            ax.set_xscale('symlog',linthresh=.03);ax.axvline(0,color='.5',lw=.6);ax.set_ylim(0,1);v.panel(ax,string.ascii_lowercase[i*3+k],f'{member}: {METHODS[m]} − {METHODS[b]}')
            if k==0:ax.set_ylabel('Cumulative land fraction')
            if i==3:ax.set_xlabel('Paired ΔD (symlog)')
    axs[0,0].legend(frameon=False,fontsize=5.8,ncol=2);axs[0,0].set_xlim(lims[0]*1.04,lims[1]*1.04)
    for ax in axs.ravel():v.difference_ticks(ax)
    export(f,'S07_complete_ecdf','Complete member-duration paired distributions','Every target duration is shown within each member, retaining complete ECDF tails. Colors and line patterns denote durations here. These distributions are descriptive and do not treat grid cells as independent replicates.')

def cases(d):
    for kind,id in [('depth','S08_case_depth'),('change','S09_case_change'),('qq','S10_case_QQ')]:
        f,axs=plt.subplots(4,3,figsize=(7.1,8.8),layout='constrained');j=2
        for i,member in enumerate(MEMBERS):
            for k,pct in enumerate([10,50,90]):
                ax=axs[i,k]
                if kind=='qq':
                    for s in range(2):
                        hp=d['casepars'][i,s,j,:,k];dp=d['daily'][i,s,0,:,k]
                        x=genextreme.ppf(1-1/T,c=-dp[2],loc=dp[0],scale=dp[1]);y=genextreme.ppf(1-1/T,c=-hp[2],loc=hp[0],scale=hp[1]);ax.plot(x,y,color=v.PERIOD_COLORS[s],ls=['--','-'][s],lw=1.1,label=['Historical','Target'][s])
                    ax.set_xlabel('Whole-day quantile (mm)');ax.set_ylabel('3h quantile (mm)')
                else:
                    q=d['caseq'][i,:,j,:,k];tr=d['casetr'][i,j,:,k];hist=q[0]
                    if kind=='change':q=q/hist;tr=tr/hist;ax.axhline(1,color='.6',lw=.6)
                    v.methodlines(ax,T,q);ax.plot(T,tr,color='black',lw=1.2,label='Reference');v.taxis(ax,True)
                    ax.set_yscale('log');ax.set_ylabel('Depth (mm, log axis)' if kind=='depth' else 'Change factor (log axis)')
                v.panel(ax,string.ascii_lowercase[i*3+k],f'{member} | historical P{pct}')
        axs[0,0].legend(frameon=False,fontsize=5.7,ncol=2)
        export(f,id,f'All fixed cases: {kind}','Three coordinates were selected from member01 historical3h Q10 spatial P10/P50/P90 and retained unchanged across all members. Each row uses the corresponding member-specific historical baseline and target-hourly reference. Q–Q pairs match non-exceedance probability, not storm identity. Log axes retain large positive tails without magnitude capping.')

def relations(d):
    for j,id in zip([0,5,6],['S11_relations_1h','S12_relations_6h','S13_relations_12h']):
        f,axs=plt.subplots(4,3,figsize=(7.1,8.3),layout='constrained',sharex=True)
        for i,member in enumerate(MEMBERS):
            for m,name in enumerate(['K_F/K_H','G_F/G_H','C_F/C_H']):
                ax=axs[i,m];a=d['rel_summary'][i,:,m,j]
                ax.fill_between(T,a[0],a[2],color=v.COLORS[m+1],alpha=.18,lw=0);ax.plot(T,a[1],color=v.COLORS[m+1],lw=1.2);ax.axhline(0,color='.4',ls=':',lw=.7)
                v.taxis(ax,True);v.panel(ax,string.ascii_lowercase[i*3+m],f'{member} | {name}');ax.set_ylabel('Log relation ratio')
        export(f,id,f'Raw relation changes: {DURATIONS[j]}h','Lines and bands show within-member spatial median and P10–P90 of the log relation ratio. HQT summaries exclude undefined raw mappings and preserve their flags in the source diagnostics. These panels explain raw transfer discrepancies algebraically, not as an independent validation test. The main probability-consequence panel uses the regularized prediction, including boundary cases.')

def fits(d):
    f,axs=plt.subplots(2,2,figsize=(7.1,5.7),layout='constrained');roles=[('Historical hourly',slice(0,7)),('Historical daily',slice(7,10)),('Target daily',slice(10,13)),('Target hourly',slice(13,20))]
    for i,(label,sl) in enumerate(roles):
        z=np.sort(d['fit_ks'][:,sl].ravel());axs[0,0].step(z,np.arange(1,len(z)+1)/len(z),color=v.COLORS[i],ls=v.STYLES[i],label=label,rasterized=True)
    axs[0,0].set_xlabel('Maximum empirical–fitted CDF discrepancy');axs[0,0].set_ylabel('Cumulative fraction');axs[0,0].legend(frameon=False,fontsize=6);v.panel(axs[0,0],'a','Descriptive GEV fit check')
    fit=pd.read_csv(OUT/'point_diagnostics'/'fit_diagnostics_all.csv');actualroles=['historical_hourly','historical_daily','future_daily','future_hourly_reference']
    vals=[100*(fit[fit.role==r].support_violation_cells/fit[fit.role==r].cells).mean() for r in actualroles]
    axs[0,1].bar(range(4),vals,color=v.COLORS);axs[0,1].set_xticks(range(4),['H hour','H day','F day','F hour']);axs[0,1].set_ylabel('Fits with sample outside support (%)');v.panel(axs[0,1],'b','Original-fit support flags')
    shape=pd.read_csv(OUT/'supplementary'/'case_shape_change_ci.csv',dtype={'member':str});ax=axs[1,0]
    for i,member in enumerate(MEMBERS):
        for k,pct in enumerate([10,50,90]):
            row=shape[(shape.member==member)&(shape.case_percentile==pct)&(shape.duration_h==3)].iloc[0];y=i+(k-1)*.2
            point=d['casepars'][i,1,2,2,k]-d['casepars'][i,0,2,2,k]
            ax.hlines(y,row.xi_change_low,row.xi_change_high,color=['#4D84A7','#222222','#B17250'][k],lw=1);ax.scatter(point,y,color=['#4D84A7','#222222','#B17250'][k],s=17,marker=['o','s','^'][k])
    ax.axvline(0,color='.5',lw=.7);ax.set_yticks(range(4),MEMBERS);ax.invert_yaxis();ax.set_xlabel('Target − historical ξ, percentile95% CI');ax.set_ylabel('Member');v.panel(ax,'c','Shape-change uncertainty: fixed3h cases')
    ax=axs[1,1];high=[];boundary=[]
    for member in MEMBERS:
        e=np.load(OUT/'point_diagnostics'/f'diagnostics_{member}.npz');r=e['hqt_mapped_T'];high.append((r>41).mean(axis=1));boundary.append(np.isinf(r).mean(axis=1))
        ax.plot(T,100*high[-1],color='.65',lw=.65)
    ax.plot(T,100*np.mean(high,axis=0),color=v.COLORS[2],lw=1.5,label='Mapped rarity >41years')
    ax.plot(T,100*np.mean(boundary,axis=0),color='.25',ls='--',lw=1.2,label='Historical support exceeded')
    v.taxis(ax);ax.set_ylim(0,100);ax.set_ylabel('Land cells (%)');ax.legend(frameon=False,fontsize=6.2);v.panel(ax,'d','Historical mapping extrapolation')
    export(f,'S14_fit_and_support','Fitted distributions, support and historical extrapolation','The CDF discrepancy is descriptive, not a KS test with unadjusted fitted-parameter p-values. Original support checks and bootstrap support flags are distinct. Shape intervals use500 paired draws at each of three fixed3h cases; blue/black/brown indicate P10/P50/P90. Historical mapping beyond41years exceeds the largest plotting position for40 maxima; a fitted-support endpoint is a stronger boundary and is shown separately.')

def dependency(d):
    f,axs=plt.subplots(2,2,figsize=(7.1,5.9),layout='constrained')
    for s in range(2):
        for a,key in enumerate(['dependency','overlap']):
            arr=d[key][:,s]
            for i in range(4):axs[0,a].plot(DURATIONS,arr[i].mean(axis=-1),color=v.PERIOD_COLORS[s],alpha=.3,lw=.6)
            axs[0,a].plot(DURATIONS,arr.mean(axis=(0,2)),color=v.PERIOD_COLORS[s],ls=['--','-'][s],lw=1.6,marker=['o','^'][s],ms=3,label=['Historical','Target'][s])
    for ax in axs[0]:ax.set_xticks(DURATIONS);ax.set_xlabel('Duration (h)');ax.set_ylim(0,1)
    axs[0,0].set_ylabel('Mean within-year Spearman ρ');axs[0,1].set_ylabel('Mean window-overlap fraction');axs[0,0].legend(frameon=False)
    v.panel(axs[0,0],'a','Grid annual-maximum association');v.panel(axs[0,1],'b','Grid accumulation-window overlap')
    st=pd.read_csv(OUT/'supplementary'/'station_rank_window_relationship.csv',dtype={'station_id':str});ids=st.station_id.unique();names=[]
    for i,sid in enumerate(ids):
        sub=st[st.station_id==sid];names.append(sub.iloc[0].station_name.replace('-airport','').replace('-down','').title()+(' *' if sid=='00315' else ''))
        for a,metric in enumerate(['rank_rho','window_overlap_fraction']):
            vals=[sub[sub.period==period][metric].mean() for period in ['historical','target']]
            axs[1,a].hlines(i,min(vals),max(vals),color='.7',lw=1)
            for s in range(2):axs[1,a].scatter(vals[s],i,color=v.PERIOD_COLORS[s],marker=['o','^'][s],s=23)
    for ax in axs[1]:ax.set_yticks(range(7),names);ax.invert_yaxis();ax.set_xlim(0,1)
    axs[1,0].set_xlabel('Mean Spearman ρ across durations');axs[1,1].set_xlabel('Mean overlap across durations')
    v.panel(axs[1,0],'c','Station association');v.panel(axs[1,1],'d','Station overlap')
    export(f,'S15_rank_and_windows','Same-year association and accumulation-window overlap','Hourly maxima at1–12h are compared with whole-day24h maxima within the same year. Correlations use ranks and overlap means positive intersection of the maximum accumulation intervals; neither identifies independent storms. Grid thin lines show member means; bold lines are their mean. Station summaries average the7 durations. Boulmer (*) is exploratory. These joint descriptors do not change the marginal transfer algorithms.')

def station_sensitivity():
    tab=pd.read_csv(OUT/'stations'/'aggregate_differences.csv');pairs=['UCF minus NC','HQT minus UCF','TPS minus UCF'];f,axs=plt.subplots(3,3,figsize=(7.1,7.8),layout='constrained',sharey='col')
    for k,pair in enumerate(pairs):
        for row,policy in enumerate(['original_admissible','support_clean']):
            ax=axs[row,k];a=tab[(tab.group=='primary_6')&(tab.policy==policy)&(tab.comparison==pair)].sort_values('block_length')
            ax.vlines(a.block_length,a.low,a.high,color=v.COLORS[k+1],lw=1.2);ax.scatter(a.block_length,a.point,color=v.COLORS[k+1],marker=v.MARKERS[k+1],s=20)
            ax.set_xticks([1,3,5]);ax.set_xlabel('Calendar-block length (years)');ax.axhline(0,color='.5',lw=.6)
            v.panel(ax,string.ascii_lowercase[row*3+k],pair.replace(' minus ',' − '));ax.set_ylabel('Paired ΔD' if k==0 else '')
            ax.text(.02,.98,('Original admissibility' if row==0 else 'Support-clean sensitivity')+'\nvalid: '+ '/'.join(a.valid_replicates.astype(str)),transform=ax.transAxes,va='top',fontsize=5.7)
        ax=axs[2,k];groups=['primary_6','all_7_exploratory','no_nominal_trend_3_exploratory']
        for i,group in enumerate(groups):
            r=tab[(tab.block_length==3)&(tab.policy=='original_admissible')&(tab.group==group)&(tab.comparison==pair)].iloc[0]
            ax.vlines(i,r.low,r.high,color=v.COLORS[k+1],lw=1.2);ax.scatter(i,r.point,color=v.COLORS[k+1],s=20)
        ax.axhline(0,color='.5',lw=.6);ax.set_xticks(range(3),['Primary6','All7','Trend\nsubset3']);ax.set_ylabel('Paired ΔD' if k==0 else '');v.panel(ax,string.ascii_lowercase[6+k],'Station-set sensitivity')
    export(f,'S16_station_sensitivity','Block length, support and station-set sensitivity','Top two rows use the same six primary stations with2000 draws for block lengths1/3/5years. Support-clean draws require all relevant fits to contain their sampled observations, jointly across sites/durations/methods; valid counts are shown and conditioning may change the sampling population. The last row uses block3 and compares the primary6, all7 including Boulmer, and the exploratory three-station subset without nominal trend flags. Intervals are pointwise95% percentiles; a nonsignificant trend screen does not prove stationarity.')

def station_coverage():
    cohort=pd.read_csv(ROOT/'midas_station_validation_7'/'transfer_v1'/'cohort.csv',dtype={'station_id':str});f,axs=plt.subplots(2,2,figsize=(7.1,6.0),layout='constrained');names=[x.replace('-airport','').replace('-down','').title() for x in cohort.station_name]
    for i,r in enumerate(cohort.itertuples()):
        axs[0,0].hlines(i,min(r.n_1961_1990,r.n_1991_2020),max(r.n_1961_1990,r.n_1991_2020),color='.75',lw=1)
        for s,n in enumerate([r.n_1961_1990,r.n_1991_2020]):axs[0,0].scatter(n,i,color=v.PERIOD_COLORS[s],marker=['o','^'][s],s=24)
    axs[0,0].set_yticks(range(7),names);axs[0,0].invert_yaxis();axs[0,0].set_xlim(0,31);axs[0,0].set_xlabel('Qualified paired years');v.panel(axs[0,0],'a','Historical / target sample coverage')
    samples=np.load(OUT/'stations'/'block_3'/'evaluation.npz')['sample_counts'].min(axis=1)
    for i,z in enumerate(samples):
        a=np.quantile(z,[0,.1,.5,.9,1]);axs[0,1].hlines(i,a[0],a[4],color='.7',lw=.7);axs[0,1].hlines(i,a[1],a[3],color=v.COLORS[1],lw=3);axs[0,1].scatter(a[2],i,color='.2',s=18)
    axs[0,1].set_yticks(range(7),names);axs[0,1].invert_yaxis();axs[0,1].set_xlim(0,31);axs[0,1].set_xlabel('Smaller effective period sample size');v.panel(axs[0,1],'b','Bootstrap sample-count distribution')
    qs=pd.read_csv(OUT/'supplementary'/'station_raw_quantile_scores.csv');qs=qs[qs.primary]
    for m,method in enumerate(METHODS):
        a=qs[qs.method==method].groupby('T').normalized_QS.mean();axs[1,0].plot(a.index,a.values,color=v.COLORS[m],ls=v.STYLES[m],marker=v.MARKERS[m],ms=3,label=method)
    axs[1,0].set_xticks([2,5,10]);axs[1,0].set_xlabel('Return period (years)');axs[1,0].set_ylabel('Mean normalized quantile loss');axs[1,0].legend(frameon=False,ncol=2,fontsize=6);v.panel(axs[1,0],'c','Raw-AMS score: primary stations')
    tab=pd.read_csv(OUT/'stations'/'individual_D.csv',dtype={'station_id':str});tab=tab[(tab.block_length==3)&(tab.method=='UCF')]
    for k,policy in enumerate(['original_admissible','support_clean']):
        a=tab[tab.policy==policy].set_index('station_id').loc[cohort.station_id];axs[1,1].barh(np.arange(7)+(k-.5)*.28,a.valid_replicates,height=.25,color=['#9CBFD5','#315D7C'][k],label=['Original','Support-clean'][k])
    axs[1,1].set_yticks(range(7),names);axs[1,1].invert_yaxis();axs[1,1].set_xlim(0,2050);axs[1,1].set_xlabel('Jointly valid draws, out of2000');axs[1,1].legend(frameon=False,fontsize=6);v.panel(axs[1,1],'d','Interval conditioning')
    export(f,'S17_station_coverage','Coverage, variable bootstrap counts and raw-data checks','Coverage requires common usable hourly and whole-day years. Calendar resampling preserves missing-year masks, so effective sample counts vary; thin bars in b span minima–maxima, thick bars P10–P90 and points medians, not confidence intervals. Boulmer has14 historical years and is supplemental. Raw-AMS scores are normalized by the same historical-hourly median for all methods at a site; target daily fitting uses the same period, so this is not independent annual out-of-sample validation.')

def station_curves():
    cohort=pd.read_csv(ROOT/'midas_station_validation_7'/'transfer_v1'/'cohort.csv',dtype={'station_id':str});p=np.load(OUT/'stations'/'point_predictions.npz');e=np.load(OUT/'stations'/'point_evaluation.npz');b=np.load(OUT/'stations'/'bands_block3.npz')['reference']
    for part,ids in enumerate([list(range(4)),list(range(4,7))]):
        f,axs=plt.subplots(len(ids),3,figsize=(7.1,2.05*len(ids)+.35),layout='constrained',squeeze=False)
        for row,i in enumerate(ids):
            for col,j in enumerate([0,2,6]):
                ax=axs[row,col];v.methodlines(ax,T,p['q'][:,j,IDS,i]);true=gev_curves(e['truthp'][j,:,i:i+1],reduced_variate(T))[:,0]
                ax.plot(T,true,color='black',lw=1.2,label='Reference');ax.fill_between(T,b[0,j,:,i],b[2,j,:,i],color='.4',alpha=.1,lw=0);ax.set_yscale('log');v.taxis(ax,True);ax.set_ylabel('Depth (mm, log axis)')
                v.panel(ax,string.ascii_lowercase[row*3+col],f'{cohort.iloc[i].station_name.replace("-airport","").replace("-down","").title()} | {DURATIONS[j]}h')
        axs[0,0].legend(frameon=False,fontsize=5.5,ncol=2)
        export(f,f'S{18+part:02d}_station_curves','Complete station examples, part'+str(part+1),'All seven stations are displayed across the two figures at1,3,12h. Lines are point predictions and the black target-hourly GEV reference; shading is the2000-draw block3 reference interval. Log rainfall axes preserve wide positive tails. Boulmer is exploratory and remains excluded from the six-station primary average. Full numerical curves and intervals at all seven durations accompany the figures.')

def local_intervals(d):
    f,axs=plt.subplots(4,3,figsize=(7.1,9.0),layout='constrained');j=2
    for i,member in enumerate(MEMBERS):
        b=np.load(OUT/'joint_bootstrap'/member/'joint_results.npz');qc=b['case_prediction_ci'];tc=b['case_reference_ci']
        for k,pct in enumerate([10,50,90]):
            ax=axs[i,k]
            for m in range(4):
                ax.fill_between(T,qc[0,j,m,:,k],qc[2,j,m,:,k],color=v.COLORS[m],alpha=.09,lw=0);ax.plot(T,d['caseq'][i,m,j,:,k],color=v.COLORS[m],ls=v.STYLES[m],lw=1,label=METHODS[m])
            ax.fill_between(T,tc[0,j,:,k],tc[2,j,:,k],color='.35',alpha=.09,lw=0);ax.plot(T,d['casetr'][i,j,:,k],color='black',lw=1.15,label='Reference')
            ax.set_yscale('log');v.taxis(ax,True);ax.set_ylabel('3h depth (mm, log axis)');v.panel(ax,string.ascii_lowercase[i*3+k],f'{member} | P{pct}')
    axs[0,0].legend(frameon=False,fontsize=5.4,ncol=2)
    export(f,'S20_case_intervals','Complete fixed-case sampling intervals','Point curves and pointwise percentile95% intervals for all four methods and the target-hourly reference, using500 paired block3 resamples within each member. Input fits are frozen before reference access; target daily/hourly resampling uses identical year draws. Log axes retain extreme upper bounds. These intervals are conditional on the selected model/GEV/admissibility rules and are not simultaneous bands or guarantees of physical plausibility.')

def boundary(d):
    f,axs=plt.subplots(2,2,figsize=(7.1,5.5),layout='constrained');omit=pd.read_csv(OUT/'point_diagnostics'/'omit_first_all.csv',dtype={'member':str});cross=pd.read_csv(OUT/'point_diagnostics'/'cross_duration_all.csv',dtype={'member':str})
    for m,method in enumerate(METHODS):
        a=omit[omit.method==method].groupby('duration_h').mean(numeric_only=True)
        axs[0,0].plot(a.index,a.mean_change,color=v.COLORS[m],ls=v.STYLES[m],marker=v.MARKERS[m],ms=3,label=method)
        axs[0,1].fill_between(a.index,a.cell_change_P10,a.cell_change_P90,color=v.COLORS[m],alpha=.1,lw=0);axs[0,1].plot(a.index,a.mean_change,color=v.COLORS[m],ls=v.STYLES[m],lw=1.2)
        a=cross[cross.method==method].groupby('longer_h').mean(numeric_only=True)
        for k,metric in enumerate(['depth_crossing_cell_fraction','intensity_crossing_cell_fraction']):axs[1,k].plot(a.index,100*a[metric],color=v.COLORS[m],ls=v.STYLES[m],marker=v.MARKERS[m],ms=3)
    for ax in axs[0]:ax.axhline(0,color='.5',lw=.6);ax.set_xlabel('Duration (h)');ax.set_xticks(DURATIONS)
    axs[0,0].set_ylabel('Mean D change after omitting1981');axs[0,1].set_ylabel('Change in D: member-mean spatial ranges');axs[0,0].legend(frameon=False,ncol=2,fontsize=6)
    for ax in axs[1]:ax.set_xlabel('Longer duration in adjacent pair (h)');ax.set_xticks(DURATIONS[1:]);ax.set_ylabel('Cells with a crossing (%)');ax.set_ylim(bottom=0)
    for k,title in enumerate(['First-year omission sensitivity','Spatial sensitivity range','Depth decreases with duration','Intensity increases with duration']):v.panel(axs.ravel()[k],string.ascii_lowercase[k],title)
    export(f,'S21_boundaries_and_crossings','Record-start and cross-duration diagnostics','Historical1981 is omitted jointly from hourly and whole-day fitting; target daily inputs and the target-hourly reference are unchanged. This is a39-vs40year sensitivity, not recovery of unobserved pre-record rainfall. Panel b averages member-specific P10/P90 spatial bounds and means; it is not a confidence interval. Crossings are diagnosed on the fixed probability grid for adjacent target durations. Return-period PAVA does not enforce duration consistency; no new duration constraint is imposed.')

def raw_scores():
    model=pd.read_csv(OUT/'point_diagnostics'/'raw_scores_all.csv');station=pd.read_csv(OUT/'supplementary'/'station_raw_quantile_scores.csv');station=station[station.primary]
    f,axs=plt.subplots(2,3,figsize=(7.1,5.4),layout='constrained',sharex=True)
    for row,frame in enumerate([model,station]):
        for col,period in enumerate([2,5,10]):
            ax=axs[row,col];a=frame[frame['T']==period].groupby(['method','duration_h']).normalized_QS.mean().unstack(0)
            for m in [1,2,3]:ax.plot(a.index,a[METHODS[m]]-a['NC'],color=v.COLORS[m],ls=v.STYLES[m],marker=v.MARKERS[m],ms=2.5,label=f'{METHODS[m]} − NC')
            ax.axhline(0,color='.5',lw=.7);ax.set_xticks(DURATIONS);ax.set_xlabel('Duration (h)');ax.set_ylabel('Normalized quantile-loss difference');v.panel(ax,string.ascii_lowercase[row*3+col],['Grid','Six stations'][row]+f' | T={period}')
    axs[0,0].legend(frameon=False,fontsize=5.8)
    export(f,'S22_raw_AMS_scores','Low-return-period quantile scores against raw AMS','The pinball loss at T2,5,10 is scored against target annual maxima and normalized by the corresponding positive historical-hourly median, common to all methods. Grid summaries weight members/cells equally and station summaries weight six primary sites equally. Negative differences favor updating over NC. Since target daily inputs are fitted from the same period, these are complementary same-period scores, not strict held-out annual probability calibration and not a validation of the100year tail.')

def interval_construction():
    tab=pd.read_csv(OUT/'interval_method_sensitivity.csv',dtype={'member':str})
    pairs=['UCF minus NC','HQT minus UCF','TPS minus UCF']
    f,axs=plt.subplots(2,3,figsize=(7.1,5.4),layout='constrained')
    for k,pair in enumerate(pairs):
        for row,domain in enumerate(['model','station']):
            a=tab[(tab.domain==domain)&(tab.comparison==pair)]
            if domain=='station':a=a[(a.group=='primary_6')&(a.policy=='original_admissible')].sort_values('block_length')
            else:a=a.sort_values('member')
            ax=axs[row,k]
            for i,r in enumerate(a.itertuples()):
                ax.hlines(i-.09,r.low,r.high,color=v.COLORS[k+1],lw=1.5)
                ax.hlines(i+.09,r.basic_low,r.basic_high,color=v.COLORS[k+1],lw=1.2,ls='--')
                ax.scatter(r.point,i,s=17,color='black',marker='D',zorder=4)
            ax.set_yticks(range(len(a)),a.member if domain=='model' else [f'Block {int(b)}' for b in a.block_length])
            ax.invert_yaxis();ax.axvline(0,color='.5',lw=.7);ax.set_xlabel('Paired difference in mean D')
            v.panel(ax,string.ascii_lowercase[row*3+k],pair.replace(' minus ',' − '))
            if k==0:ax.set_ylabel('Model member' if domain=='model' else 'Six primary stations')
    handles=[v.Line2D([],[],color='.3',lw=1.5),v.Line2D([],[],color='.3',ls='--',lw=1.2),v.Line2D([],[],color='black',marker='D',lw=0,ms=4)]
    f.legend(handles,['Percentile interval','Basic interval','Original point estimate'],loc='outside upper center',ncol=3,frameon=False,fontsize=7)
    export(f,'S23_interval_construction','Sensitivity to bootstrap interval construction',
           'The same saved paired draws are summarized by percentile [q2.5,q97.5] and basic [2d−q97.5,2d−q2.5] intervals, where d is the original difference. This post-hoc diagnostic was prompted by displaced percentile intervals; no predictors, exclusions or resamples were changed. Top: each model member, 500 block-3 draws. Bottom: six primary stations, 2000 draws at each block length. Original fit admissibility is used. Interval disagreement, especially for the station TPS−UCF contrast, limits claims of robust separation; neither construction establishes nominal coverage with these short, non-smooth estimators.')


def draw_all(d):
    metrics(d);heatmaps(d);maps(d);ecdf(d);cases(d);relations(d);fits(d);dependency(d)
    station_sensitivity();station_coverage();station_curves();local_intervals(d);boundary(d);raw_scores()
    if (OUT/'interval_method_sensitivity.csv').exists():interval_construction()
