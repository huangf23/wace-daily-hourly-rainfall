"""Station analogue of the frozen UKCP18 regularized transfer experiment.

Separate prepare/predict/evaluate processes enforce held-out hourly values.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NUMBA_NUM_THREADS']='7'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.stats import genextreme
from gev_lmoments_kernel import fit_batch
from transfer_core_regularized import (METHODS,DURATIONS,T_ENGINEERING,
    reduced_variate,scaling,predict_curves,gev_curves,score_curves,valid_curves,LAST_DIAGNOSTICS)

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'midas_station_validation_7'/'whole_day_AMS'
OUT=ROOT/'midas_station_validation_7'/'transfer_v1'
T=np.geomspace(2,100,201)
TFINE=np.geomspace(2,100,401)
ALLD=[1,2,3,4,5,6,12,24,48,72]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def savejson(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare():
    OUT.mkdir(exist_ok=True)
    h=pd.read_csv(SOURCE/'hourly_AMS_paired_common_years.csv',dtype={'station_id':str})
    d=pd.read_csv(SOURCE/'whole_day_AMS_paired_common_years.csv',dtype={'station_id':str})
    for name,a,lo,hi in [('early_hourly',h,1961,1990),('early_daily',d,1961,1990),
                         ('late_daily',d,1991,2020),('heldout_late_hourly',h,1991,2020)]:
        a[a.year.between(lo,hi)].to_csv(OUT/f'{name}.csv',index=False)
    cohort=h[['station_id','station_name']].drop_duplicates().sort_values('station_id')
    for start,end in [(1961,1990),(1991,2020)]:
        counts=h[h.year.between(start,end)].groupby('station_id').year.nunique()
        cohort[f'n_{start}_{end}']=cohort.station_id.map(counts)
    cohort['primary']=(cohort.n_1961_1990>=15)&(cohort.n_1991_2020>=15)
    cohort.to_csv(OUT/'cohort.csv',index=False)
    savejson('config.json',dict(periods=[[1961,1990],[1991,2020]],target_durations=DURATIONS.tolist(),
        daily_durations=[24,48,72],return_periods=T_ENGINEERING.tolist(),integration_points=201,
        projection_points='401 log-T nodes plus engineering nodes',methods=METHODS,
        hqt_epsilon=1e-6,estimation='Same stationary L-moment GEV kernel as UKCP18; xi=-scipy_c',
        primary_stations=6,exploratory_stations=1,year_definition='Existing interval-end calendar-year labels',
        eligibility='Ten hourly and three whole-day series share coverage-qualified station years; no missing-year imputation',
        hourly_holdout='Late hourly values used only for fitting evaluation reference after predictions frozen. Coverage/availability used to define paired sample.',
        source_hashes={p.name:sha(p) for p in [SOURCE/'hourly_AMS_paired_common_years.csv',SOURCE/'whole_day_AMS_paired_common_years.csv']},
        core_hashes={name:sha(ROOT/name) for name in ['gev_lmoments_kernel.py','transfer_core.py','transfer_core_regularized.py']}))
    print(cohort.to_string(index=False),flush=True)

def fit(a,cohort,durations,role):
    pars=np.empty((len(durations),3,len(cohort)));records=[]
    for k,station in enumerate(cohort.itertuples()):
        sub=a[a.station_id==station.station_id]
        pivot=sub.pivot(index='year',columns='duration_h',values='ams_mm').sort_index()
        sample=np.ascontiguousarray(pivot[durations].to_numpy().T)
        p,m,q,status,support=fit_batch(sample,T_ENGINEERING)
        assert np.all(status==0),(role,station.station_id,status)
        assert np.isfinite(q).all() and (q>0).all()
        ref=genextreme.ppf(1-1/T_ENGINEERING[None,:],c=-p[:,2,None],loc=p[:,0,None],scale=p[:,1,None])
        assert np.allclose(q,ref,rtol=1e-10,atol=1e-9)
        pars[:,:,k]=p
        for j,duration in enumerate(durations):
            records.append(dict(role=role,station_id=station.station_id,station_name=station.station_name,
                duration_h=duration,n_years=len(pivot),years=';'.join(map(str,pivot.index)),
                mu=p[j,0],sigma=p[j,1],xi=p[j,2],l1=m[j,0],l2=m[j,1],tau3=m[j,2],
                sample_values_outside_fitted_support=int(support[j])))
    return pars,records

def predict():
    cohort=pd.read_csv(OUT/'cohort.csv',dtype={'station_id':str})
    allowed=['early_hourly','early_daily','late_daily']
    inputs={r:pd.read_csv(OUT/f'{r}.csv',dtype={'station_id':str}) for r in allowed}
    h,hr=fit(inputs['early_hourly'],cohort,ALLD,'early_hourly')
    hd,hdr=fit(inputs['early_daily'],cohort,[24,48,72],'early_daily')
    fd,fdr=fit(inputs['late_daily'],cohort,[24,48,72],'late_daily')
    daily=np.stack([hd,fd]);sc,res=scaling(daily)
    assert np.isfinite(sc).all()
    q=[];eng=[];adjusted=[];clips=[];bounds=[]
    for j,duration in enumerate(DURATIONS):
        v,u,lt,bd=predict_curves(h[j],hd[0],fd[0],sc,float(duration),reduced_variate(TFINE))
        assert valid_curves(v)[2].all()
        q.append(v);adjusted.append(LAST_DIAGNOSTICS['adjusted'].copy());clips.append(LAST_DIAGNOSTICS['clipped'].copy());bounds.append(bd)
        eng.append(predict_curves(h[j],hd[0],fd[0],sc,float(duration),reduced_variate(T_ENGINEERING))[0])
    np.savez_compressed(OUT/'frozen_predictions.npz',early_hourly=h,daily=daily,tps_coefficients=sc,tps_fit_rmse=res,
        curves401=np.stack(q,axis=1),engineering=np.stack(eng,axis=1),adjusted=np.stack(adjusted,axis=1),
        clipped=np.stack(clips),support_boundary=np.stack(bounds))
    pd.DataFrame(hr+hdr+fdr).to_csv(OUT/'prediction_GEV_parameters.csv',index=False)
    savejson('prediction_complete.json',dict(status='complete',time=time.strftime('%Y-%m-%dT%H:%M:%S'),
        heldout_hourly_files_opened=0,allowed_inputs={r:sha(OUT/f'{r}.csv') for r in allowed},
        prediction_sha256=sha(OUT/'frozen_predictions.npz'),
        monotonicity_adjusted_curves=int(np.sum(adjusted)),hqt_clipped_nodes=int(np.sum(clips)),
        clipping_count_note='Count includes repeated daily mapping across seven target durations'))
    print('Predictions frozen before held-out hourly fitting.',flush=True)

def evaluate():
    marker=json.loads((OUT/'prediction_complete.json').read_text(encoding='utf-8'))
    assert marker['heldout_hourly_files_opened']==0
    assert sha(OUT/'frozen_predictions.npz')==marker['prediction_sha256']
    cohort=pd.read_csv(OUT/'cohort.csv',dtype={'station_id':str})
    frozen=np.load(OUT/'frozen_predictions.npz')
    a=pd.read_csv(OUT/'heldout_late_hourly.csv',dtype={'station_id':str})
    truth,rows=fit(a,cohort,ALLD,'late_hourly_evaluation_only')
    pd.DataFrame(rows).to_csv(OUT/'evaluation_GEV_parameters.csv',index=False)
    scores=[];engineering=[];curve_rows=[];convergence=[];truthcurves=[]
    for j,duration in enumerate(DURATIONS):
        pred=frozen['curves401'][:,j];tr=gev_curves(truth[j],reduced_variate(TFINE));truthcurves.append(tr)
        assert valid_curves(tr)[2].all()
        values=score_curves(pred[:,::2],tr[::2]);fine=score_curves(pred,tr)
        # Independent vectorized numerical integration of the three metrics.
        error=np.log(pred[:,::2]/tr[None,::2]);mean=np.trapz(error,axis=1)/200
        assert np.allclose(values[:,0],np.trapz(abs(error),axis=1)/200)
        assert np.allclose(values[:,1],mean)
        assert np.allclose(values[:,2],np.sqrt(np.trapz((error-mean[:,None])**2,axis=1)/200))
        qt=gev_curves(truth[j],reduced_variate(T_ENGINEERING))
        qh=gev_curves(frozen['early_hourly'][j],reduced_variate(T_ENGINEERING))
        for k,station in enumerate(cohort.itertuples()):
            best=int(np.argmin(values[:,0,k]))
            for m,method in enumerate(METHODS):
                scores.append(dict(station_id=station.station_id,station_name=station.station_name,primary=station.primary,
                    duration_h=int(duration),method=method,D_RL=values[m,0,k],B=values[m,1,k],S=values[m,2,k],
                    geometric_bias_pct=100*np.expm1(values[m,1,k]),best_method=METHODS[best],
                    monotonicity_adjusted=bool(frozen['adjusted'][m,j,k])))
                convergence.append(abs(values[m,0,k]-fine[m,0,k]))
                for z,period in enumerate(T_ENGINEERING):
                    q=frozen['engineering'][m,j,z,k]
                    engineering.append(dict(station_id=station.station_id,station_name=station.station_name,primary=station.primary,
                        duration_h=int(duration),return_period_years=period,method=method,predicted_depth_mm=q,
                        predicted_intensity_mm_h=q/duration,true_depth_mm=qt[z,k],true_intensity_mm_h=qt[z,k]/duration,
                        early_depth_mm=qh[z,k],relative_error_pct=100*(q/qt[z,k]-1),
                        predicted_change_factor=q/qh[z,k],true_change_factor=qt[z,k]/qh[z,k]))
                for z,period in enumerate(T):
                    curve_rows.append(dict(station_id=station.station_id,duration_h=int(duration),method=method,
                        return_period_years=period,predicted_depth_mm=pred[m,2*z,k],true_depth_mm=tr[2*z,k]))
    score=pd.DataFrame(scores);score.to_csv(OUT/'station_duration_scores.csv',index=False)
    pd.DataFrame(engineering).to_csv(OUT/'engineering_return_levels.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(OUT/'dense_curves.csv',index=False)
    np.savez_compressed(OUT/'evaluation_reference.npz',parameters=truth,curves401=np.stack(truthcurves))
    summaries=[]
    for group,subset in [('primary_6',score[score.primary]),('all_7_exploratory',score),
        ('boulmer_only',score[~score.primary]),('no_nominal_trend_3_exploratory',score[score.station_id.isin(['00708','01145','01198'])])]:
        for method,g in subset.groupby('method',sort=False):
            summaries.append(dict(group=group,method=method,stations=g.station_id.nunique(),station_duration_pairs=len(g),
                mean_D_RL=g.D_RL.mean(),median_D_RL=g.D_RL.median(),mean_B=g.B.mean(),mean_S=g.S.mean(),
                best_fraction=(g.best_method==method).mean(),underestimate_fraction=(g.B<0).mean()))
    pd.DataFrame(summaries).to_csv(OUT/'group_summary.csv',index=False)
    score.groupby(['station_id','station_name','primary','method'])[['D_RL','B','S']].mean().reset_index().to_csv(OUT/'station_summary.csv',index=False)
    score[score.primary].groupby(['duration_h','method'])[['D_RL','B','S']].mean().reset_index().to_csv(OUT/'duration_summary_primary.csv',index=False)
    # Same mean aggregation over members, durations and UK land cells for context.
    import netCDF4 as nc
    grids=[]
    for member in ['01','04','07','08']:
        with nc.Dataset(ROOT/'transfer_experiment_v2_regularized'/f'evaluation_{member}.nc') as ds:
            land=np.asarray(ds['land_mask'][:]).ravel().astype(bool)
            arr=np.stack([np.asarray(ds[v][:]).reshape(3,7,-1)[:,:,land] for v in ['D_RL','B_log_bias','S_shape_error']],axis=2)
            grids.append(arr)
    grid=np.stack(grids)
    gs=[]
    for m,method in enumerate(METHODS):
        gs.append(dict(method=method,mean_D_RL=float(grid[:,m,:,0].mean()),mean_B=float(grid[:,m,:,1].mean()),
            mean_S=float(grid[:,m,:,2].mean()),best_fraction=float((grid[:,:,:,0].argmin(axis=1)==m).mean())))
    pd.DataFrame(gs).to_csv(OUT/'ukcp18_comparable_summary.csv',index=False)
    savejson('FINAL_REPORT.json',dict(status='complete',primary_stations=6,exploratory_stations=1,
        predictions_frozen_before_holdout_fit=True,prediction_sha256=marker['prediction_sha256'],
        all_curves_finite_positive_monotone=True,score_rows=len(score),engineering_rows=len(engineering),
        maximum_201_401_D_difference=float(max(convergence)),summary=summaries,ukcp18_summary=gs,
        uncertainty='Point estimates only; no station bootstrap or significance claim on method differences',
        comparability='Different periods, sample lengths, spatial support and climates; comparison is qualitative validation, not like-for-like model skill test',
        monotonicity_adjusted_curves=marker['monotonicity_adjusted_curves'],hqt_clipped_nodes=marker['hqt_clipped_nodes']))
    assert sha(OUT/'frozen_predictions.npz')==marker['prediction_sha256']
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print('UKCP18 equivalent aggregation:',pd.DataFrame(gs).to_string(index=False),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'predict':predict,'evaluate':evaluate}[sys.argv[1]]()
