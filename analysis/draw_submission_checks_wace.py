"""Figures for the frozen submission-check extension."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize,TwoSlopeNorm,ListedColormap
from matplotlib.lines import Line2D
import draw_research_v3 as old
from complete_submission_checks_wace import DEST as DATA,EPS
P=old.PAPER;DEST=P/'figures_wace'/'submission_checks_20261002';DEST.mkdir(parents=True,exist_ok=True)
CAP={};COL=old.COLORS;PAIRS=['UCF-NC','HQT-UCF','TPS-UCF'];LABELS=['UCF − NC','HQT − UCF','TPS − UCF']
def value_grid(ax,values,xlabels,ylabels,cmap,norm,fmt='.3f'):
    im=ax.imshow(values,aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
    for (i,j),v in np.ndenumerate(values):
        rgb=np.array(im.cmap(im.norm(v))[:3]);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);lum=linear@np.array([.2126,.7152,.0722])
        ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=7,color='white' if lum<.179 else 'black')
    ax.set_xticks(range(len(xlabels)),xlabels);ax.set_yticks(range(len(ylabels)),ylabels);ax.tick_params(length=0);ax.spines[:].set_visible(False)
    return im
def save(f,name,caption):
    for ext in ['pdf','png','svg']:f.savefig(DEST/(name+'.'+ext),dpi=300,facecolor='white')
    plt.close(f);CAP[name]=caption;print('DRAWN '+name,flush=True)
def workflow():
    f=plt.figure(figsize=(7.1,5.0),layout='constrained');g=f.add_gridspec(2,3,height_ratios=[1,1.5],width_ratios=[1,1.2,1.2]);ax=f.add_subplot(g[0,0]);geo=dict(np.load(DATA/'input_snapshot_geometry.npz' if (DATA/'input_snapshot_geometry.npz').exists() else P/'figures_v3'/'source_data.npz'))
    old.spatial(ax,np.ones(len(geo['cells'])),geo,Normalize(0,1),ListedColormap(['#aec7d4','#668ca3']));old.panel(ax,'a','Model domain')
    ax=f.add_subplot(g[0,1:]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');ax.set_title('(b) Two 40-year periods',loc='left',fontsize=9)
    ax.text(.02,.93,'UKCP18 members 01, 04, 07, 08 • 10,397 land cells',fontsize=8,va='top')
    for x,w,c,txt in [(.02,.39,'#dae6ee','Historical\n1981–2020'),(.57,.39,'#f0dcd2','Target\n2041–2080')]:
        ax.add_patch(FancyBboxPatch((x,.32),w,.4,boxstyle='round,pad=.008',facecolor=c,edgecolor='none'));ax.text(x+w/2,.52,txt,ha='center',va='center',fontsize=9)
    ax.plot([.42,.55],[.5,.5],color='.5',ls=':');ax.text(.5,.12,'Year = preceding December–November; 360-day calendar',ha='center',fontsize=7)
    ax=f.add_subplot(g[1,:]);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title('(c) Predictor information and evaluation',loc='left',fontsize=9)
    def box(x,y,w,h,text,color,ls='-'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,edgecolor='.5',lw=.8,linestyle=ls));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7.5)
    def arrow(a,b,dashed=False):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='.4',lw=1,linestyle='--' if dashed else '-'))
    box(.02,.73,.28,.20,'Historical hourly AMS\n1, 2, 3, 4, 5, 6, 12 h','#dae6ee')
    box(.36,.73,.28,.20,'Historical + target daily AMS\nMidnight-aligned 1, 2, 3 days','#e1ebdf')
    box(.70,.73,.28,.20,'Target hourly AMS\nEvaluation only','#f5e5dd','--')
    box(.10,.35,.49,.22,'L-moment GEV → transfer predictions\nNC baseline • UCF • HQT • TPS','#eef0f2')
    box(.70,.35,.28,.22,'Target hourly GEV\nEvaluation reference','#f5e5dd','--')
    arrow((.16,.73),(.25,.57));arrow((.5,.73),(.46,.57));arrow((.84,.73),(.84,.57),True)
    box(.20,.01,.60,.18,'Paired local errors → member and regional summaries\nConditional gains • tail behavior • sampling sensitivity','#f6f5ef')
    arrow((.35,.35),(.4,.19));arrow((.84,.35),(.68,.19),True)
    save(f,'fig01_workflow',r'Study domain, periods and information flow. (a) Native land-cell mask used for evaluation. (b) Four members from one UKCP18 framework are analyzed separately over historical and target 40-year periods. (c) Historical hourly and historical/target midnight-aligned daily maxima supply the predictors. Target hourly maxima enter the reference and evaluation only (dashed path). Predictions and input-defined strata are frozen before their evaluation; annual maxima and GEV parameters are never pooled across members or cells.')

def conditions():
    s=pd.read_csv(DATA/'strata_aggregate.csv');f,axs=plt.subplots(2,2,figsize=(7.1,5.5),layout='constrained')
    for name,ax,label,title in [('daily_change',axs[0,0],'a','Daily change signal'),('historical_tail',axs[0,1],'b','Historical tail mismatch'),('historical_seasonality',axs[1,0],'c','Historical seasonal concentration')]:
        for k,pair in enumerate(PAIRS):
            r=s[(s.feature==name)&(s.comparison==pair)].sort_values('group');x=np.arange(3)+(k-1)*.13
            ax.errorbar(x,r.mean_delta,yerr=[r.mean_delta-r.low,r.high-r.mean_delta],fmt=old.MARKERS[k+1]+'-',color=COL[k+1],ms=4,lw=1,capsize=2,label=LABELS[k])
        ax.axhline(0,color='.5',ls=':',lw=.8);ax.set_xticks(range(3),['Low','Middle','High']);ax.set_xlabel('Input-defined tercile');ax.set_ylabel(r'Mean paired $\Delta D_{RL}$');old.panel(ax,label,title)
    axs[0,0].legend(frameon=False,fontsize=6.4)
    ax=axs[1,1];r=s[(s.feature=='daily_x_tail')&(s.comparison=='HQT-UCF')].sort_values('group');z=r.mean_delta.to_numpy().reshape(3,3);lim=abs(z).max();im=value_grid(ax,z,['Low','Middle','High'],['Low','Middle','High'],old.DIVERGING,TwoSlopeNorm(0,vmin=-lim,vmax=lim),'.4f');ax.set_xlabel('Historical tail-mismatch tercile');ax.set_ylabel('Daily-change tercile');old.panel(ax,'d','Joint conditions: HQT − UCF');f.colorbar(im,ax=ax,orientation='horizontal',pad=.07,shrink=.8).set_label(r'Mean paired $\Delta D_{RL}$')
    save(f,'fig06_conditions',r'Associations between allowed-input conditions and updating performance. (a--c) Within-member terciles of signed daily log-change, absolute historical hourly--daily tail-growth mismatch and historical hourly AMS seasonal concentration. Tail and seasonality terciles are duration-specific. Points are equal-member means after pooling local durations within each stratum; bars are nominal 95\% spatial-block resampling intervals using 53 native-coordinate 100-km blocks and 1,000 draws. (d) Joint daily-change/tail terciles for HQT--UCF. Negative contrasts favor the first method. Group definitions use no target-hourly information. These post-hoc associations are not independent physical attribution or a validated method-selection rule; individual members are retained in Fig.~S30.')

def epsilon():
    e=pd.read_csv(DATA/'epsilon_all.csv',dtype={'member':str});a=e[(e.T_low==2)&(e.T_high==100)];f,axs=plt.subplots(2,2,figsize=(7.1,5.5),layout='constrained')
    for member,r in a.groupby('member'):
        rr=r.groupby('epsilon').mean(numeric_only=True);axs[0,0].plot(rr.index,rr.mean_D,color=COL[2],alpha=.3,lw=.7);axs[0,1].plot(rr.index,rr.clip_100*100,color=COL[2],alpha=.3,lw=.7);axs[1,1].plot(rr.index,rr.improves_UCF*100,color=COL[2],alpha=.3,lw=.7)
    r=a.groupby('epsilon').mean(numeric_only=True)
    for ax,values in [(axs[0,0],r.mean_D),(axs[0,1],r.clip_100*100),(axs[1,1],r.improves_UCF*100)]:ax.plot(r.index,values,color=COL[2],lw=1.7,marker='s',ms=4);ax.set_xscale('log');ax.set_xticks(EPS);ax.axvline(1e-6,color='.4',ls=':',lw=.7);ax.set_xlabel(r'HQT probability bound $\epsilon$')
    for m,v in [(0,.2213547523),(1,.1215149539),(3,.1922633193)]:axs[0,0].axhline(v,color=COL[m],ls=old.STYLES[m],lw=1,label=old.METHODS[m])
    axs[0,0].legend(frameon=False,fontsize=6.5);axs[0,0].set_ylabel(r'Mean $D_{RL}$');axs[0,1].set_ylabel('100-year queries bounded (%)');axs[1,1].set_ylabel('HQT improves on UCF (%)')
    z=e[e.T_high!=100].copy();z=e[~((e.T_low==2)&(e.T_high==100))];mat=z.groupby(['epsilon','T_low']).mean_D.mean().unstack().sort_index();ax=axs[1,0];im=value_grid(ax,mat.values,['2–10','10–25','25–100'],[f'{x:.0e}' for x in mat.index],old.SEQUENTIAL,Normalize(0,mat.values.max()),'.3f');ax.set_xlabel('Return-period interval (years)');ax.set_ylabel(r'$\epsilon$');f.colorbar(im,ax=ax,orientation='horizontal',pad=.07,shrink=.8).set_label(r'Mean HQT $D_{RL}$')
    for ax,l,t in zip(axs.ravel(),'abcd',['Aggregate error','Numerical bound activation','Where the error changes','Local pairwise improvement']):old.panel(ax,l,t)
    save(f,'S28_hqt_epsilon',r'HQT bound sensitivity with all other inputs and numerical rules fixed. Five bounds were fixed before evaluating their outcomes; the original value is $10^{-6}$. Thin curves show members and thick curves their equal-member means after averaging durations and land cells. (a) HQT error with the unchanged NC, UCF and TPS means. (b) Fraction of 100-year HQT queries hitting a probability bound. (c) Mean errors by return-period interval. (d) Fraction of local HQT errors below UCF. No bound is selected or calibrated using target-hourly results; the original main analysis is retained.')

def bands():
    b=pd.read_csv(DATA/'bands_all.csv',dtype={'member':str});b=b[~((b.T_low==2)&(b.T_high==100))];f,axs=plt.subplots(2,2,figsize=(7.1,5.0),layout='constrained');mat=b.groupby(['method','T_low']).mean_D.mean().unstack().loc[old.METHODS];im=value_grid(axs[0,0],mat.values,['2–10','10–25','25–100'],old.METHODS,old.SEQUENTIAL,Normalize(0,mat.values.max()),'.3f');old.panel(axs[0,0],'a','Mean error by return-period interval')
    vals=b.groupby(['member','T_low','method']).mean_D.mean().unstack()
    for k,(ax,(m,base)) in enumerate(zip(axs.ravel()[1:],[(1,0),(2,1),(3,1)])):
        for i,member in enumerate(old.MEMBERS):
            v=vals.loc[member,old.METHODS[m]]-vals.loc[member,old.METHODS[base]];ax.scatter(np.arange(3)+(i-1.5)*.06,v,color=COL[m],marker=['o','s','^','D'][i],s=20,label=member)
        av=(vals[old.METHODS[m]]-vals[old.METHODS[base]]).groupby('T_low').mean();ax.plot(range(3),av,color=COL[m],lw=1.3);ax.axhline(0,color='.5',ls=':',lw=.7);ax.set_xticks(range(3),['2–10','10–25','25–100']);ax.set_xlabel('Return-period interval (years)');ax.set_ylabel(r'Mean paired $\Delta D_{RL}$');old.panel(ax,chr(98+k),LABELS[k])
    axs[0,1].legend(frameon=False,ncol=4,fontsize=6)
    save(f,'S29_return_periods',r'Sensitivity of method comparison to return-period range. The primary absolute log-quantile error is integrated separately over 2--10, 10--25 and 25--100 years, with exact boundaries interpolated in log return period. (a) Equal-member/duration/land means. (b--d) Duration/land-mean paired contrasts, with symbols for each member and lines their mean. No new score is introduced. These summaries reproduce and expose the previously saved band diagnostics; the target reference remains fitted from 40 annual maxima.')

def members():
    s=pd.read_csv(DATA/'strata_member.csv',dtype={'member':str});f,axs=plt.subplots(4,3,figsize=(7.1,8.2),sharex=True,sharey=True,layout='constrained')
    for i,member in enumerate(old.MEMBERS):
        for j,name in enumerate(['daily_change','historical_tail','historical_seasonality']):
            ax=axs[i,j]
            for k,pa in enumerate(PAIRS):
                r=s[(s.member==member)&(s.feature==name)&(s.comparison==pa)].sort_values('group');ax.plot(range(3),r.mean_delta,color=COL[k+1],ls=old.STYLES[k+1],marker=old.MARKERS[k+1],ms=3,label=LABELS[k])
            ax.axhline(0,color='.5',ls=':',lw=.7);ax.set_xticks(range(3),['Low','Middle','High']);old.panel(ax,chr(97+i*3+j),f'{member}: '+['daily change','tail mismatch','seasonality'][j]);ax.set_ylabel(r'Mean $\Delta D_{RL}$' if j==0 else '')
    axs[0,0].legend(frameon=False,fontsize=6)
    save(f,'S30_condition_members',r'Allowed-input stratification retained separately for all four members. Each panel reports three pairwise mean contrasts within the same frozen terciles used in main Fig.~6. Tail and seasonal terciles are defined separately by duration; daily-change terciles are shared across hourly durations. All panels share the vertical scale. These descriptive member curves complement the paired spatial-block intervals in the main figure. Full bin counts and cutpoints accompany the numerical release.')

def coverage():
    c=pd.read_csv(DATA/'coverage_contrasts.csv');q=pd.read_csv(DATA/'coverage_quantiles.csv');f,axs=plt.subplots(1,2,figsize=(7.1,3.8),layout='constrained')
    for ax,rho,l in zip(axs,[0.,.3],'ab'):
        values=[];labels=[]
        for pair in PAIRS:
            for typ in ['percentile','basic']:
                values.append(c[(c.rho==rho)&(c.comparison==pair)&(c.interval==typ)].sort_values('scenario').coverage.to_numpy()*100);labels.append(pair+'\n'+typ)
        value_grid(ax,np.array(values),['S0','S1','S4'],labels,old.SEQUENTIAL,Normalize(0,100),'.1f');old.panel(ax,l,f'Latent annual correlation: {rho:g}');ax.set_xlabel('Known-distribution scenario')
    save(f,'S31_coverage_contrasts',r'Empirical coverage (\%) of nominal 95\% paired-error intervals in known-distribution experiments. S0: no change; S1: common amplification; S4: changed temporal scaling. Each configuration uses 500 independent pairs of 40-year records and 499 paired circular 3-year resamples per pair. Within-year Gaussian latent shared variance is 0.8; annual latent AR(1) correlations are 0 and 0.3. Coverage targets the known population contrast in mean $D_{RL}$ over 1, 3 and 12 h. All outer and inner fits are usable. Monte Carlo standard errors are reported in the source table (at most 2.3 percentage points here). These local controlled experiments do not directly estimate coverage of the spatially aggregated UKCP18 contrasts.')
    f,axs=plt.subplots(2,2,figsize=(7.1,5.4),layout='constrained')
    for row,colname in enumerate(['estimator_coverage','true_target_inclusion']):
        for col,rho in enumerate([0.,.3]):
            z=q[q.rho==rho].groupby(['method','scenario','T'])[colname].mean().unstack(['scenario','T']).loc[old.METHODS]*100
            value_grid(axs[row,col],z.values,['S0\n10','S0\n100','S1\n10','S1\n100','S4\n10','S4\n100'],old.METHODS,old.SEQUENTIAL,Normalize(0,100),'.1f');old.panel(axs[row,col],chr(97+row*2+col),['Estimator coverage','True-target inclusion'][row]+f', ρ={rho:g}');axs[row,col].set_xlabel('Scenario / return period (years)')
    save(f,'S32_coverage_quantiles',r'Pointwise percentile-interval diagnostics (\%), averaging the separately evaluated 1-, 3- and 12-h rates; no observations are pooled for fitting. (a,b) Coverage of each method\textquoteright s own population-prediction quantile, which evaluates the sampling interval around that estimator. (c,d) Inclusion of the true future quantile, which additionally reflects structural transfer error and is not nominal estimator coverage. The same six configurations and resampling scheme as Fig.~S31 are used. Full duration-specific rates, validity counts, estimands and Monte Carlo standard errors accompany the release.')

if __name__=='__main__':
    assert (DATA/'grid_complete.json').exists() and (DATA/'coverage_complete.json').exists()
    for fn in [workflow,conditions,epsilon,bands,members,coverage]:fn()
    (DEST/'captions.json').write_text(json.dumps(CAP,indent=2),encoding='utf-8')
