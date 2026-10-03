"""Evaluate frozen design depths in the fitted future marginal; no calibration."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,concurrent.futures
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.stats import genextreme
R=Path(__file__).resolve().parent
OUT=R/'research_wace/engineering_design_20261002'
MEMBERS=['01','04','07','08'];METHODS=['NC','UCF','HQT','TPS']
DURS=np.array([1,2,3,4,5,6,12]);TS=np.array([10,25,50,100]);HORIZON=30

def task(member):
    snap=R/'research_wace/submission_checks_20261002'/f'input_snapshot_{member}.npz'
    with np.load(R/'research_wace/submission_checks_20261002/input_snapshot_geometry.npz') as z:cells=z['cells']
    with np.load(snap) as z:h=z['h']
    src=R/'transfer_experiment_v2_regularized'/f'predictions_{member}.nc'
    with nc.Dataset(src) as ds:
        tids=[int(np.flatnonzero(np.asarray(ds['engineering_return_period'][:])==t)[0]) for t in TS]
        q=np.asarray(ds['predicted_depth_engineering'][:],float).reshape(3,7,7,-1)[:,:,tids,:][...,cells]
    historical=genextreme.isf(1/TS[None,:,None],c=-h[:,2,None,:],loc=h[:,0,None,:],scale=h[:,1,None,:])
    q=np.concatenate([historical[None],q],axis=0)
    assert q.shape==(4,7,4,10397) and np.isfinite(q).all() and (q>0).all()
    # Only after loading and freezing the predictions is the evaluation reference used.
    with np.load(snap) as z:p=z['evaluation_reference_parameters']
    truth=genextreme.isf(1/TS[None,:,None],c=-p[:,2,None,:],loc=p[:,0,None,:],scale=p[:,1,None,:])
    prob=genextreme.sf(q,c=-p[None,:,2,None,:],loc=p[None,:,0,None,:],scale=p[None,:,1,None,:])
    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    np.testing.assert_allclose(genextreme.sf(truth,c=-p[:,2,None,:],loc=p[:,0,None,:],scale=p[:,1,None,:]),np.broadcast_to(1/TS[None,:,None],truth.shape),rtol=1e-11,atol=1e-13)
    # Independent closed-form support-aware check, including Gumbel limit.
    z=(q-p[None,:,0,None,:])/p[None,:,1,None,:];xi=p[None,:,2,None,:];support=1+xi*z
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        ell=np.where(np.abs(xi)<1e-9,-z,-np.log1p(xi*z)/xi)
        check=-np.expm1(-np.exp(ell))
    check=np.where(support<=0,np.where(xi>0,1.,0.),check)
    np.testing.assert_allclose(prob,check,rtol=1e-8,atol=1e-11)
    # Existing 100-year endpoint must be exactly the same evaluated prediction.
    with np.load(R/'research_v3/point_diagnostics'/f'predictions_{member}.npz') as z:prior=z['pred'][:,:,-1]
    np.testing.assert_allclose(q[:,:,-1],prior,rtol=2e-6,atol=1e-5)
    with np.errstate(divide='ignore'):risk=-np.expm1(HORIZON*np.log1p(-prob))
    np.testing.assert_allclose(risk,1-(1-prob)**HORIZON,atol=1e-14)
    err=100*(q/truth[None]-1)
    arr={'relative_depth_error_pct':err,'annual_exceedance':prob,'exceedance_ratio':prob*TS[None,None,:,None],'horizon_exceedance':risk,'depth_mm':q,'reference_depth_mm':truth}
    np.savez_compressed(OUT/f'local_{member}.npz',**arr,durations=DURS,return_periods=TS,cells=cells)
    (OUT/f'check_{member}.json').write_text(json.dumps({'member':member,'status':'passed','max_sf_difference':float(np.max(abs(prob-check))),'prediction_source':str(src),'prediction_depth_sha256':hashlib.sha256(q.tobytes()).hexdigest(),'zero_survival_count':int((prob==0).sum()),'evaluation_only':True},indent=2),encoding='utf-8')
    print('Completed '+member,flush=True)
    return arr

def summarize(arr,member):
    rows=[]
    for m,name in enumerate(METHODS):
        for j,d in enumerate(DURS):
            for k,t in enumerate(TS):
                err=arr['relative_depth_error_pct'][m,j,k].ravel();p=arr['annual_exceedance'][m,j,k].ravel();a=arr['exceedance_ratio'][m,j,k].ravel();r=arr['horizon_exceedance'][m,j,k].ravel()
                row=dict(member=member,method=name,duration=int(d),T=int(t),n=len(p),below90_pct=float((err < -10).mean()*100),above110_pct=float((err>10).mean()*100),within10_pct=float(((err>=-10)&(err<=10)).mean()*100),A_gt2_pct=float((a>2).mean()*100),zero_survival_pct=float((p==0).mean()*100),nominal_R30_pct=float((1-(1-1/t)**HORIZON)*100))
                for label,x in [('depth_error',err),('A',a),('R30_pct',r*100)]:
                    for quant,val in zip([10,25,50,75,90],np.percentile(x,[10,25,50,75,90])):row[f'{label}_p{quant}']=float(val)
                pm=float(np.median(p));row['equivalent_T_median']=1/pm if pm>0 else None
                rows.append(row)
    return rows

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    design={'intent':'Post-hoc engineering interpretation of existing fixed estimates; no optimization or new method','members':MEMBERS,'durations_h':DURS.tolist(),'return_periods':TS.tolist(),'horizon_years':HORIZON,'depth_tolerance_pct':10,'exceedance_ratio_threshold':2,'threshold_role':'Descriptive diagnostics; not regulatory acceptance thresholds','horizon_assumption':'Illustrative independent years with stationary fitted future marginal, not a calendar-specific nonstationary lifetime forecast','future_hourly_role':'Fitted evaluation reference only','summary':'Equal-weight member-cell distribution within each duration and return period; all members retained separately','intervals':'Spatial/member descriptive quantiles; no new confidence intervals','zeros':'Keep zero survival and report support endpoint frequency; no artificial large return period','choice_timing':'This design fixed before reading new engineering outcomes; prior main results already known'}
    (OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:arrays=list(pool.map(task,MEMBERS))
    pd.DataFrame(sum([summarize(a,m) for a,m in zip(arrays,MEMBERS)],[])).to_csv(OUT/'by_member.csv',index=False)
    combined={k:np.concatenate([a[k] for a in arrays],axis=-1) for k in arrays[0]}
    df=pd.DataFrame(summarize(combined,'all4'));df.to_csv(OUT/'summary.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps({'status':'complete','rows':len(df),'members':4,'land_cells':10397,'fits_added':0,'future_hourly_evaluation_only':True,'tests':'GEV inverse check, independent survival expression, original endpoint consistency, horizon expression'},indent=2),encoding='utf-8')
    print(df[(df.duration.isin([1,6]))&(df['T'].isin([10,100]))][['method','duration','T','below90_pct','above110_pct','equivalent_T_median','R30_pct_p50','A_gt2_pct']].to_string(index=False))
