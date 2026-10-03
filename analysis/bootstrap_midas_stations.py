"""Synchronous calendar-block bootstrap with missing station years retained.

Prediction stage has no access to withheld hourly values. Evaluation refits
the reference using the same late-year draws as the allowed daily data.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NUMBA_NUM_THREADS']='7'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from gev_lmoments_kernel import fit_batch
from transfer_core import predict_curves as raw_predict
from transfer_core_regularized import (BASE_A,WEIGHTS,bounded_hqt,project_log_curves,
    scaling,gev_curves,score_curves,DURATIONS,METHODS,T_ENGINEERING,reduced_variate)

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'midas_station_validation_7'/'transfer_v1'
OUT=SOURCE/'uncertainty_2000'
B=2000
T=np.geomspace(2,100,201)
Q=[.025,.5,.975]
IDS=np.abs(BASE_A[:,None]-reduced_variate(T)[None,:]).argmin(axis=0)
ENG=np.abs(BASE_A[:,None]-reduced_variate(T_ENGINEERING)[None,:]).argmin(axis=0)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def draws(seed):
    rng=np.random.default_rng(seed)
    start=rng.integers(0,30,(B,10))
    return ((start[:,:,None]+np.arange(3))%30).reshape(B,30)

def refit(frame,sid,durations,start,index):
    p=frame[frame.station_id==sid].pivot(index='year',columns='duration_h',values='ams_mm')
    arr=p.reindex(np.arange(start,start+30))[durations].to_numpy()
    present=np.isfinite(arr).all(axis=1)
    assert np.all(np.isfinite(arr).any(axis=1)==present)
    sampled=arr[index];mask=present[index];n=mask.sum(axis=1)
    pars=np.full((len(durations),3,B),np.nan)
    status=np.full((len(durations),B),1,dtype=np.int16)
    support=np.zeros_like(status)
    for count in np.unique(n):
        if count<3:continue
        select=np.flatnonzero(n==count)
        x=np.stack([sampled[r,mask[r]].T for r in select])
        flat=np.ascontiguousarray(x.reshape(-1,count))
        pp,_,_,ss,sp=fit_batch(flat,np.empty(0))
        pars[:,:,select]=pp.reshape(len(select),len(durations),3).transpose(1,2,0)
        status[:,select]=ss.reshape(len(select),len(durations)).T
        support[:,select]=sp.reshape(len(select),len(durations)).T
    return pars,n,status,support

def curves(h,hd,fd,sc,d):
    raw,u,lt,bound=raw_predict(h,hd,fd,sc,float(d),BASE_A)
    clipped=bounded_hqt(h,lt,raw,u)
    good=np.all(np.isfinite(raw)&(raw>0),axis=1)
    projected=np.full_like(raw,np.nan);adjusted=np.zeros((3,B),bool)
    for m in range(3):
        use=np.flatnonzero(good[m])
        if len(use):
            v,changed=project_log_curves(np.ascontiguousarray(raw[m:m+1][:,:,use]),WEIGHTS)
            projected[m][:,use]=v[0];adjusted[m,use]=changed[0]
    return projected,good,adjusted,clipped

def quantiles(x,axis):
    with np.errstate(invalid='ignore'):
        return np.nanquantile(x,Q,axis=axis)

def predict():
    OUT.mkdir(exist_ok=True)
    started=time.time()
    cohort=pd.read_csv(SOURCE/'cohort.csv',dtype={'station_id':str})
    allowed=['early_hourly','early_daily','late_daily']
    data={r:pd.read_csv(SOURCE/f'{r}.csv',dtype={'station_id':str}) for r in allowed}
    ih,il=draws(930061),draws(930091)
    np.savez_compressed(OUT/'calendar_draws.npz',early_indices=ih,late_indices=il,
        early_years=ih+1961,late_years=il+1991)
    save('config.json',dict(replicates=B,block_length=3,calendar_years_per_period=30,
        seeds=[930061,930091],interval='Pointwise 2.5/50/97.5 percentiles; not simultaneous',
        pairing='Common calendar-year draws across all stations and all durations; early and late independent. Late hourly evaluation uses identical late draws to daily prediction input.',
        missing='Resample complete calendar with original masks; omit missing station years after drawing. Replicate station sample counts may vary. Never join nonadjacent observed years into a block.',
        boundary='Circular blocks permit 1990→1961 and 2020→1991 wrap, as a bootstrap convention',
        assumptions='Period-wise stationarity approximation; blocks do not remove trends; uncertainty conditional on fixed station cohort, data screening, GEV and transfer structure',
        invalid='No replacing, clipping rainfall magnitude, or redrawing failed replicates. Counts recorded; intervals conditional on valid replicates.',
        aggregation='Fixed six primary stations, equal weights; only jointly valid complete replicates used for method differences and aggregate comparisons',
        input_hashes={r:sha(SOURCE/f'{r}.csv') for r in allowed},
        original_point_prediction_hash=sha(SOURCE/'frozen_predictions.npz')))
    diagnostics=[];files={};cirows=[]
    point=np.load(SOURCE/'frozen_predictions.npz')
    for k,station in enumerate(cohort.itertuples()):
        sid=station.station_id
        h,nh,hs,hsp=refit(data['early_hourly'],sid,DURATIONS.tolist(),1961,ih)
        hd,nd,hds,hdsp=refit(data['early_daily'],sid,[24,48,72],1961,ih)
        fd,nf,fds,fdsp=refit(data['late_daily'],sid,[24,48,72],1991,il)
        assert np.array_equal(nh,nd)
        sc,rmse=scaling(np.stack([hd,fd]))
        file=OUT/f'frozen_parameters_{sid}.npz'
        np.savez_compressed(file,h=h,hd=hd,fd=fd,sc=sc,nh=nh,nf=nf,
            hs=hs,hds=hds,fds=fds,hsp=hsp,hdsp=hdsp,fdsp=fdsp)
        files[sid]=sha(file)
        dense=[]
        for j,d in enumerate(DURATIONS):
            q,good,adjusted,clipped=curves(h[j],hd[0],fd[0],sc,d)
            dense.append(quantiles(q[:,IDS],2))
            band=quantiles(q[:,ENG],2)
            for m,method in enumerate(METHODS):
                for z,period in enumerate(T_ENGINEERING):
                    cirows.append(dict(station_id=sid,station_name=station.station_name,primary=station.primary,
                        method=method,duration_h=int(d),return_period_years=period,
                        point_depth_mm=point['engineering'][m,j,z,k],depth_low=band[0,m,z],depth_median=band[1,m,z],depth_high=band[2,m,z],
                        point_intensity_mm_h=point['engineering'][m,j,z,k]/d,
                        intensity_low=band[0,m,z]/d,intensity_median=band[1,m,z]/d,intensity_high=band[2,m,z]/d,
                        valid_replicates=int(good[m].sum()),total_replicates=B))
                diagnostics.append(dict(station_id=sid,duration_h=int(d),method=method,
                    valid_prediction_replicates=int(good[m].sum()),monotonicity_adjustments=int(adjusted[m].sum()),
                    early_n_min=int(nh.min()),early_n_max=int(nh.max()),late_n_min=int(nf.min()),late_n_max=int(nf.max()),
                    hqt_clipped_replicates=int(clipped.any(axis=0).sum()),
                    hourly_fit_failures=int((hs[j]!=0).sum()),historical_daily_fit_failures=int((hds!=0).any(axis=0).sum()),
                    future_daily_fit_failures=int((fds!=0).any(axis=0).sum()),
                    fitted_support_violation_replicates=int(((hsp[j]>0)|(hdsp>0).any(axis=0)|(fdsp>0).any(axis=0)).sum())))
        np.savez_compressed(OUT/f'prediction_bands_{sid}.npz',depth=np.stack(dense),T=T,quantiles=Q)
        print(f'PREDICTED {sid}; elapsed={time.time()-started:.1f}s',flush=True)
    pd.DataFrame(cirows).to_csv(OUT/'predicted_rainfall_intensity_CI95.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'prediction_replicate_diagnostics.csv',index=False)
    save('prediction_complete.json',dict(status='complete',replicates=B,future_hourly_values_accessed=0,
        frozen_parameter_hashes=files,draw_hash=sha(OUT/'calendar_draws.npz'),elapsed_seconds=time.time()-started))

def evaluate():
    started=time.time()
    marker=json.loads((OUT/'prediction_complete.json').read_text(encoding='utf-8'))
    assert marker['status']=='complete' and marker['future_hourly_values_accessed']==0
    cohort=pd.read_csv(SOURCE/'cohort.csv',dtype={'station_id':str})
    data=pd.read_csv(SOURCE/'heldout_late_hourly.csv',dtype={'station_id':str})
    index=np.load(OUT/'calendar_draws.npz')['late_indices']
    point=pd.read_csv(SOURCE/'station_duration_scores.csv',dtype={'station_id':str})
    engpoint=pd.read_csv(SOURCE/'engineering_return_levels.csv',dtype={'station_id':str})
    allscores=[];score_rows=[];truth_rows=[];delta_rows=[];checks=[]
    for station in cohort.itertuples():
        sid=station.station_id;f=OUT/f'frozen_parameters_{sid}.npz'
        assert sha(f)==marker['frozen_parameter_hashes'][sid]
        frozen=np.load(f)
        truth,n,status,support=refit(data,sid,DURATIONS.tolist(),1991,index)
        assert np.array_equal(n,frozen['nf'])
        saved=[];bands=[]
        for j,d in enumerate(DURATIONS):
            q,good,_,_=curves(frozen['h'][j],frozen['hd'][0],frozen['fd'][0],frozen['sc'],d)
            tr=gev_curves(truth[j],BASE_A)
            validtruth=np.all(np.isfinite(tr)&(tr>0),axis=0)
            common=good.all(axis=0)&validtruth
            values=score_curves(q[:,IDS],tr[IDS]);values[:,:,~common]=np.nan
            saved.append(values);bands.append(quantiles(np.where(validtruth[None],tr[IDS],np.nan),1))
            tb=quantiles(np.where(validtruth[None],tr[ENG],np.nan),1)
            for z,period in enumerate(T_ENGINEERING):
                pt=engpoint[(engpoint.station_id==sid)&(engpoint.duration_h==d)&(engpoint.return_period_years==period)].iloc[0]
                truth_rows.append(dict(station_id=sid,station_name=station.station_name,primary=station.primary,duration_h=int(d),
                    return_period_years=period,point_depth_mm=pt.true_depth_mm,depth_low=tb[0,z],depth_median=tb[1,z],depth_high=tb[2,z],
                    intensity_low=tb[0,z]/d,intensity_median=tb[1,z]/d,intensity_high=tb[2,z]/d,
                    valid_replicates=int(validtruth.sum())))
            for m,method in enumerate(METHODS):
                pp=point[(point.station_id==sid)&(point.duration_h==d)&(point.method==method)].iloc[0]
                for v,metric in enumerate(['D_RL','B','S']):
                    band=quantiles(values[m,v],0)
                    score_rows.append(dict(station_id=sid,station_name=station.station_name,primary=station.primary,
                        duration_h=int(d),method=method,metric=metric,point=pp[metric],low=band[0],median=band[1],high=band[2],
                        valid_replicates=int(common.sum())))
            for m in [1,2]:
                delta=values[m,0]-values[0,0];band=quantiles(delta,0)
                pp=point[(point.station_id==sid)&(point.duration_h==d)].set_index('method')
                delta_rows.append(dict(station_id=sid,station_name=station.station_name,duration_h=int(d),
                    comparison=f'{METHODS[m]} minus UCF',point=pp.loc[METHODS[m],'D_RL']-pp.loc['UCF','D_RL'],
                    low=band[0],median=band[1],high=band[2],fraction_UCF_lower=float(np.mean(delta[common]>0)),
                    valid_replicates=int(common.sum())))
            checks.append(dict(station_id=sid,duration_h=int(d),common_valid=int(common.sum()),truth_fit_failures=int((status[j]!=0).sum()),
                truth_support_violation_replicates=int((support[j]>0).sum())))
        allscores.append(np.stack(saved)) # duration, method, metric, replicate
        np.savez_compressed(OUT/f'truth_bands_{sid}.npz',depth=np.stack(bands),T=T,quantiles=Q)
        print(f'EVALUATED {sid}; elapsed={time.time()-started:.1f}s',flush=True)
    scores=np.stack(allscores) # station,duration,method,metric,replicate
    np.savez_compressed(OUT/'score_replicates.npz',scores=scores,station_ids=cohort.station_id.to_numpy(dtype=str))
    pd.DataFrame(score_rows).to_csv(OUT/'station_duration_metric_CI95.csv',index=False)
    pd.DataFrame(truth_rows).to_csv(OUT/'reference_rainfall_intensity_CI95.csv',index=False)
    pd.DataFrame(delta_rows).to_csv(OUT/'station_duration_method_difference_CI95.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'evaluation_replicate_diagnostics.csv',index=False)
    aggregates=[];differences=[];stationmeans=[];wins=[]
    for label,mask in [('primary_6',cohort.primary.to_numpy()),('all_7_exploratory',np.ones(7,bool)),
                       ('no_nominal_trend_3_exploratory',cohort.station_id.isin(['00708','01145','01198']).to_numpy())]:
        arr=scores[mask];common=np.isfinite(arr).all(axis=(0,1,2,3))
        ave=arr[...,common].mean(axis=(0,1)) # method,metric,replicate
        orig=point[point.station_id.isin(cohort.loc[mask,'station_id'])].groupby('method')[['D_RL','B','S']].mean()
        for m,method in enumerate(METHODS):
            for v,metric in enumerate(['D_RL','B','S']):
                band=quantiles(ave[m,v],0)
                aggregates.append(dict(group=label,method=method,metric=metric,point=orig.loc[method,metric],
                    low=band[0],median=band[1],high=band[2],valid_replicates=int(common.sum()),total_replicates=B))
            wins.append(dict(group=label,method=method,best_fraction=float((ave[:,0].argmin(axis=0)==m).mean()),valid_replicates=int(common.sum())))
        for m in [1,2]:
            delta=ave[m,0]-ave[0,0];band=quantiles(delta,0)
            differences.append(dict(group=label,comparison=f'{METHODS[m]} minus UCF',point=orig.loc[METHODS[m],'D_RL']-orig.loc['UCF','D_RL'],
                low=band[0],median=band[1],high=band[2],fraction_UCF_lower=float((delta>0).mean()),valid_replicates=int(common.sum())))
    for k,station in enumerate(cohort.itertuples()):
        arr=scores[k];common=np.isfinite(arr).all(axis=(0,1,2));ave=arr[...,common].mean(axis=0)
        orig=point[point.station_id==station.station_id].groupby('method')[['D_RL','B','S']].mean()
        for m,method in enumerate(METHODS):
            for v,metric in enumerate(['D_RL','B','S']):
                band=quantiles(ave[m,v],0)
                stationmeans.append(dict(station_id=station.station_id,station_name=station.station_name,primary=station.primary,
                    method=method,metric=metric,point=orig.loc[method,metric],low=band[0],median=band[1],high=band[2],valid_replicates=int(common.sum())))
    pd.DataFrame(aggregates).to_csv(OUT/'aggregate_metric_CI95.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'aggregate_method_difference_CI95.csv',index=False)
    pd.DataFrame(wins).to_csv(OUT/'bootstrap_ranking_frequencies.csv',index=False)
    pd.DataFrame(stationmeans).to_csv(OUT/'station_mean_metric_CI95.csv',index=False)
    diag=pd.read_csv(OUT/'prediction_replicate_diagnostics.csv')
    save('FINAL_REPORT.json',dict(status='complete',replicates=B,primary_stations=6,supplementary_stations=1,
        block_length=3,interval='95% pointwise percentile, conditional on admissible fits',
        minimum_prediction_valid=int(diag.valid_prediction_replicates.min()),
        minimum_evaluation_common_valid=int(min(c['common_valid'] for c in checks)),
        aggregate_differences=differences,ranking_frequencies=wins,
        aggregate_metrics=aggregates,prediction_elapsed_seconds=marker['elapsed_seconds'],evaluation_elapsed_seconds=time.time()-started,
        future_hourly_role='Evaluation reference only; daily/hourly evaluation dependence retained through identical late-year draws',
        note='Bootstrap ranking fractions are not p-values or posterior probabilities. Intervals approximate sampling uncertainty under period-wise stationarity and fixed cohort; no multiple-comparison adjustment.',
        original_point_predictions_unchanged=sha(SOURCE/'frozen_predictions.npz')==json.loads((OUT/'config.json').read_text(encoding='utf-8'))['original_point_prediction_hash']))
    print(pd.DataFrame(differences).to_string(index=False),flush=True)
    print(pd.DataFrame(wins).to_string(index=False),flush=True)

if __name__=='__main__':{'predict':predict,'evaluate':evaluate}[sys.argv[1]]()
