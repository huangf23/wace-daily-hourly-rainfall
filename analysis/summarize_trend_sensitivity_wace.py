"""Exact land summaries, local sensitivity and supplementary composite figure."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import set_num_threads
from complete_trend_sensitivity_wace import *

set_num_threads(20)
P=ROOT/'manuscript_elsevier';FIG=P/'figures_wace';cells=geometry()['cells']
ROLES=['historical_hourly','historical_daily','future_daily','future_hourly']
screens={r:{k:np.stack([read_screen(m,r,cells)[k] for m in MEMBERS]) for k in ['mk_p','mk_s','mk_p_tfpw','sen_slope','sample_median','lag1_p_approx','ljung_box_p5']} for r in ROLES}
rows=[]
for r,s in screens.items():
    durations=DURATIONS if r.endswith('hourly') else [24,48,72]
    for d,j in [('all',slice(None))]+[(str(d),j) for j,d in enumerate(durations)]:
        p=s['mk_p'][:,j].ravel();sign=s['mk_s'][:,j].ravel();b=s['sen_slope'][:,j].ravel();median=s['sample_median'][:,j].ravel();rel=1000*b/median
        rows.append(dict(role=r,duration_h=d,n=len(p),mk_flag_pct=100*np.mean(p<.05),increase_pct=100*np.mean((p<.05)&(sign>0)),decrease_pct=100*np.mean((p<.05)&(sign<0)),
          tfpw_flag_pct=100*np.mean(s['mk_p_tfpw'][:,j]<.05),lag1_flag_pct=100*np.mean(s['lag1_p_approx'][:,j]<.05),ljung_box_flag_pct=100*np.mean(s['ljung_box_p5'][:,j]<.05),
          mm_decade_median=np.median(10*b),pct_decade_p10=np.quantile(rel,.1),pct_decade_median=np.median(rel),pct_decade_p90=np.quantile(rel,.9),
          abs_39yr_pct_median=np.median(abs(rel)*3.9),abs_39yr_pct_p90=np.quantile(abs(rel)*3.9,.9),slope_over20pct_span_pct=100*np.mean(abs(rel)*3.9>20)))
pd.DataFrame(rows).to_csv(DEST/'pooled_trend_summary.csv',index=False)
original=np.load(P/'figures_v3'/'source_data.npz')['scores']
changed=np.stack([np.load(DEST/f'evaluation_{m}.npz')['scores'] for m in MEMBERS])
assert np.isfinite(changed).all()
local=[]
for j,d in enumerate(DURATIONS):
    old=original[:,:,j,0];new=changed[:,:,j,0]
    local.append(dict(duration_h=int(d),winner_changed_pct=100*np.mean(old.argmin(axis=1)!=new.argmin(axis=1)),
      ucf_vs_nc_sign_changed_pct=100*np.mean((old[:,1]<old[:,0])!=(new[:,1]<new[:,0])),
      ucf_D_abs_change_median=np.median(abs(new[:,1]-old[:,1])),ucf_D_abs_change_p90=np.quantile(abs(new[:,1]-old[:,1]),.9)))
pd.DataFrame(local).to_csv(DEST/'local_ranking_changes.csv',index=False)
qfile=DEST/'engineering_quantile_changes.npz'
if not qfile.exists():
    delta=np.zeros((4,5,7,3,len(cells)))
    for mi,m in enumerate(MEMBERS):
        f=np.load(DEST/f'frozen_prediction_parameters_{m}.npz');oldh,oldd,olds=frozen_parameters(m,cells)
        ref=np.load(DEST/f'evaluation_{m}.npz')['reference_parameters'];oldref=reference_parameters(m,cells)
        ids=[int(np.argmin(abs(BASE_T-rp))) for rp in [2,10,100]]
        for j,d in enumerate(DURATIONS):
            qp=prediction_curves(f['h'][j],f['daily'][0,0],f['daily'][1,0],f['scale'],d)[0][:,ids]
            qo=prediction_curves(oldh[j],oldd[0,0],oldd[1,0],olds,d)[0][:,ids]
            delta[mi,:4,j]=100*(qp/qo-1)
            delta[mi,4,j]=100*(gev_curves(ref[j],reduced_variate([2,10,100]))/gev_curves(oldref[j],reduced_variate([2,10,100]))-1)
        print('EXACT QUANTILE CHANGE',m,flush=True)
    np.savez_compressed(qfile,delta_pct=delta,return_periods=[2,10,100],methods=METHODS+['Reference'])
else:delta=np.load(qfile)['delta_pct']
qrows=[]
for k,label in enumerate(METHODS+['Reference']):
    for ri,rp in enumerate([2,10,100]):
        a=delta[:,k,:,ri].ravel()
        qrows.append(dict(method=label,return_period=rp,n=len(a),median_pct=np.median(a),p10_pct=np.quantile(a,.1),p90_pct=np.quantile(a,.9),median_abs_pct=np.median(abs(a)),p90_abs_pct=np.quantile(abs(a),.9),more_than10pct_pct=100*np.mean(abs(a)>10)))
pd.DataFrame(qrows).to_csv(DEST/'pooled_quantile_changes.csv',index=False)
group=pd.read_csv(DEST/'score_groups.csv');grows=[]
for (groupname,method),sub in group.groupby(['group','method'],sort=False):
    grows.append(dict(group=groupname,method=method,n=int(sub.n.sum()),original_D=np.average(sub.original_D,weights=sub.n),adjusted_D=np.average(sub.adjusted_D,weights=sub.n)))
pd.DataFrame(grows).to_csv(DEST/'pooled_group_scores.csv',index=False)
save_json(DEST/'local_summary.json',dict(winner_changed_pct=float(100*np.mean(original[:,:,:,0].argmin(axis=1)!=changed[:,:,:,0].argmin(axis=1))),
    ucf_vs_nc_sign_changed_pct=float(100*np.mean((original[:,1,:,0]<original[:,0,:,0])!=(changed[:,1,:,0]<changed[:,0,:,0]))),
    valid=291116,total=291116,mean_scores_by_member={m:{method:float(changed[i,k,:,0].mean()) for k,method in enumerate(METHODS)} for i,m in enumerate(MEMBERS)}))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(11,7.9),layout='constrained')
colors=['#777777','#0072B2','#D55E00','#009E73']
ax=axes[0,0];s=screens['historical_daily'];x=np.arange(12)
up=100*((s['mk_p']<.05)&(s['mk_s']>0)).mean(axis=-1).ravel();down=100*((s['mk_p']<.05)&(s['mk_s']<0)).mean(axis=-1).ravel()
ax.bar(x,up,color='#0072B2',label='Increasing');ax.bar(x,down,bottom=up,color='#D55E00',label='Decreasing')
ax.set_xticks(x,[f'{m}\n{d}' for m in MEMBERS for d in [24,48,72]],fontsize=8);ax.set_xlabel('Member / whole-day duration (h)');ax.set_ylabel('Nominal MK flags (%)');ax.set_title('(a) Historical whole-day AMS',loc='left');ax.legend(frameon=False,ncol=2,fontsize=8)
ax=axes[0,1];rel=(1000*s['sen_slope']/s['sample_median']).reshape(12,-1);q=np.quantile(rel,[.1,.5,.9],axis=1)
ax.errorbar(x,q[1],yerr=[q[1]-q[0],q[2]-q[1]],fmt='o',capsize=3,color='#0072B2');ax.axhline(0,color='0.5',lw=.7)
ax.set_xticks(x,[f'{m}\n{d}' for m in MEMBERS for d in [24,48,72]],fontsize=8);ax.set_xlabel('Member / whole-day duration (h)');ax.set_ylabel('Sen slope (% of median per decade)');ax.set_title('(b) Spatial median and P10–P90',loc='left')
ax=axes[1,0]
for k,m in enumerate(METHODS):
    a=original[:,k,:,0].mean(axis=(1,2));b=changed[:,k,:,0].mean(axis=(1,2))
    for mi in range(4):ax.plot([k-.17,k+.17],[a[mi],b[mi]],c=colors[k],alpha=.45,lw=1)
    ax.scatter(np.full(4,k-.17),a,s=20,facecolors='white',edgecolors=colors[k]);ax.scatter(np.full(4,k+.17),b,s=20,c=colors[k])
    ax.plot([k-.17,k+.17],[a.mean(),b.mean()],c=colors[k],lw=2);ax.scatter([k-.17,k+.17],[a.mean(),b.mean()],c=colors[k],marker='D',s=33,zorder=4)
ax.set_xticks(range(4),METHODS);ax.set_ylabel(r'Mean $D_{\rm RL}$');ax.set_title('(c) Original → trend-adjusted scores',loc='left');ax.text(.02,.98,'Circles: members; diamonds: equal-member mean',transform=ax.transAxes,va='top',fontsize=8)
ax.set_ylim(top=.29)
ax=axes[1,1]
for k,label in enumerate(METHODS+['Reference']):
    a=np.sort(abs(delta[:,k,:,2].ravel()));p=np.arange(1,len(a)+1)/len(a)
    ids=np.unique(np.r_[np.arange(0,len(a),100),len(a)-1]);ax.plot(a[ids],p[ids],color=colors[k] if k<4 else 'black',ls='--' if k==4 else '-',label=label)
ax.set_xscale('symlog',linthresh=1);ax.axvline(10,color='0.6',ls=':',lw=.8);ax.set_ylim(0,1);ax.set_xlabel('Absolute 100-year quantile change (%)');ax.set_ylabel('Cumulative proportion');ax.set_title('(d) Local amount sensitivity',loc='left');ax.legend(frameon=False,fontsize=8,loc='lower right')
for ax in axes.ravel():ax.grid(axis='y',alpha=.14)
for ext in ['pdf','png','svg']:fig.savefig(FIG/f'S25_trend_sensitivity.{ext}',dpi=200)
plt.close(fig)
save_json(DEST/'figure_provenance.json',dict(figure='S25',data_sources=['pooled_trend_summary.csv','scores.csv','engineering_quantile_changes.npz','local_ranking_changes.csv'],point_counts=dict(historical_daily=124764,score_combinations=291116),intervals='Spatial P10-P90, not confidence intervals; no new uncertainty estimation',adjustment='Period-centered linear Sen location detrending of all marginals; no future hourly input to prediction'))
print(pd.DataFrame(rows).query('duration_h == "all"').to_string(index=False))
print(pd.DataFrame(qrows).query('return_period == 100').to_string(index=False))
print((DEST/'local_summary.json').read_text())
