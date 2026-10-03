"""Evaluation-only engineering stress test. No coefficient fitting or selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json, hashlib, concurrent.futures
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_v3_common import ROOT,OUT,T,IDS,BASE_A,MEMBERS,METHODS,gev_curves

DEST=ROOT/'research_wace'/'fixed_uplift'
DEST.mkdir(parents=True,exist_ok=True)
FACTORS=np.array([1,1.05,1.10,1.15,1.20,1.25])
# This protocol is written before any target-hourly reference is opened.
protocol=dict(factors=FACTORS.tolist(),frozen_before_reference_access=True,
 objective='Descriptive trade-off; no optimization, recommended factor, or retraining',
 tolerance=0.10,tolerance_role='Diagnostic only, not a risk acceptance standard',
 weighting='Equal cells/stations, durations, and (for grids) members',
 uncertainty='No new confidence intervals; member lines are descriptive',
 source_prediction='Previously frozen regularized predictions',
 evaluation_only=True)
(DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
weights=np.ones(201)/200;weights[[0,-1]]*=.5
def summaries(q,tr,scope,member):
 assert q.shape[:3]==(4,7,201) and np.isfinite(q).all() and (q>0).all()
 err=np.log(q/tr[None])
 rows=[]
 for c in FACTORS:
  ec=err+np.log(c)
  D=np.einsum('mdtn,t->mdn',np.abs(ec),weights).mean(axis=(1,2))
  B=np.einsum('mdtn,t->mdn',ec,weights).mean(axis=(1,2))
  for m,name in enumerate(METHODS):
   tail=ec[m,:,-1,:]
   rows.append(dict(scope=scope,member=member,method=name,factor=c,
    D_RL=float(D[m]),B=float(B[m]),
    fraction_below_90pct_at_T100=float((tail<np.log(.9)).mean()),
    fraction_above_110pct_at_T100=float((tail>np.log(1.1)).mean()),
    fraction_below_reference_at_T100=float((tail<0).mean())))
 return rows
def grid(member):
 with np.load(OUT/'point_diagnostics'/f'predictions_{member}.npz') as f: q=f['pred']
 with np.load(OUT/'point_diagnostics'/f'diagnostics_{member}.npz') as f: tr=f['truth'];old=f['scores'][:, :, 0].mean(axis=(1,2))
 rows=summaries(q,tr,'grid',member)
 assert np.allclose([r['D_RL'] for r in rows[:4]],old,atol=1e-7)
 print('Member completed '+member,flush=True)
 return rows
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 gridrows=sum(list(pool.map(grid,MEMBERS)),[])
cohort=pd.read_csv(ROOT/'midas_station_validation_7'/'transfer_v1'/'cohort.csv')
mask=cohort.primary.to_numpy(bool)
with np.load(OUT/'stations'/'point_predictions.npz') as f:q=f['q'][:,:,IDS,:][...,mask]
with np.load(OUT/'stations'/'point_evaluation.npz') as f:p=f['truthp']
tr=np.stack([gev_curves(p[j],BASE_A)[IDS] for j in range(7)])[...,mask]
stationrows=summaries(q,tr,'stations','primary6')
df=pd.DataFrame(gridrows+stationrows)
df.to_csv(DEST/'tradeoff_by_member.csv',index=False)
mean=df.groupby(['scope','method','factor'],sort=False).mean(numeric_only=True).reset_index()
mean.to_csv(DEST/'tradeoff_summary.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
colors=['#777777','#0072B2','#C65D20','#00836B']; styles=[':','-','--','-.'];marks=['x','o','s','^']
fig,axes=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained')
for ax,scope,title in zip(axes[0],['grid','stations'],['(a) Model-grid integrated error','(b) Historical-station integrated error']):
 for m,name in enumerate(METHODS):
  sel=mean[(mean.scope==scope)&(mean.method==name)]
  if scope=='grid':
   for member in MEMBERS:
    v=df[(df.scope==scope)&(df.method==name)&(df.member==member)]
    ax.plot(v.factor,v.D_RL,color=colors[m],alpha=.27,lw=.6)
  ax.plot(sel.factor,sel.D_RL,color=colors[m],ls=styles[m],marker=marks[m],ms=3,lw=1.4,label=name)
 ax.set_title(title,loc='left');ax.set_ylabel('Mean integrated absolute log error');ax.legend(ncol=2,frameon=False)
for ax,scope,title in zip(axes[1],['grid','stations'],['(c) Model grids: UCF at 100 years','(d) Stations: UCF at 100 years']):
 for field,label,col,ls,marker in [('fraction_below_90pct_at_T100','More than 10% below reference','#0072B2','-','o'),('fraction_above_110pct_at_T100','More than 10% above reference','#C65D20','--','s')]:
  sel=mean[(mean.scope==scope)&(mean.method=='UCF')]
  if scope=='grid':
   for member in MEMBERS:
    v=df[(df.scope==scope)&(df.method=='UCF')&(df.member==member)]
    ax.plot(v.factor,100*v[field],color=col,alpha=.3,lw=.6)
  ax.plot(sel.factor,100*sel[field],label=label,color=col,ls=ls,marker=marker,ms=3,lw=1.4)
 ax.set_title(title,loc='left');ax.set_ylabel('Location-duration combinations (%)');ax.set_ylim(0,100);ax.legend(frameon=False,loc='upper left')
for ax in axes.flat:
 ax.spines[['top','right']].set_visible(False)
 ax.set_xlabel('Fixed multiplier c');ax.set_xticks(FACTORS);ax.grid(axis='y',alpha=.13)
figfolder=ROOT/'manuscript_elsevier'/'figures_wace';figfolder.mkdir(exist_ok=True)
for ext in ['pdf','png','svg']:fig.savefig(figfolder/f'S24_fixed_uplift.{ext}',dpi=300)
plt.close(fig)
print(mean[(mean.method=='UCF')&mean.factor.isin([1,1.10,1.25])].to_string(index=False),flush=True)
(DEST/'complete.json').write_text(json.dumps(dict(status='complete',protocol=protocol,rows=len(df),factor_fitted=False),indent=2),encoding='utf-8')
