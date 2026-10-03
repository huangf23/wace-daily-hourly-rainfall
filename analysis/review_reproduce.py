"""Portable numerical audit of the manuscript's grid and station comparisons."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from numba import set_num_threads, config
from research_v3_common import prediction_curves, METHODS, MEMBERS, DURATIONS, IDS, T
from transfer_core import gev_curves, reduced_variate, score_curves, scaling
from gev_lmoments_kernel import fit_batch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reviewer_output'

def compare(actual,expected,keys,cols):
    a=actual.sort_values(keys).reset_index(drop=True)
    b=expected.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]), 'Mismatching table keys'
    np.testing.assert_allclose(a[cols],b[cols],rtol=1e-9,atol=1e-11)
    return dict(rows=len(a),max_absolute_difference=float(np.max(np.abs(a[cols].to_numpy()-b[cols].to_numpy()))),rtol=1e-9,atol=1e-11)

def grids():
    rows=[]
    source=ROOT/'research_wace/submission_checks_20261002'
    for member in MEMBERS:
        path=source/f'input_snapshot_{member}.npz'
        # np.load reads NPZ members lazily: the reference field is not accessed here.
        with np.load(path,allow_pickle=False) as z:
            h=z['h'];daily=z['daily'];sc=z['scale']
        assert h.shape==(7,3,10397)
        for j,d in enumerate(DURATIONS):
            pred,valid,*_=prediction_curves(h[j],daily[0,0],daily[1,0],sc,d)
            assert valid.all()
            pred=pred[:,IDS]
            # Evaluation only, after this duration's predictions exist.
            with np.load(path,allow_pickle=False) as z: reference=z['evaluation_reference_parameters'][j]
            truth=gev_curves(reference,reduced_variate(T))
            scores=score_curves(pred,truth)
            assert np.isfinite(scores).all()
            for m,method in enumerate(METHODS):
                rows.append(dict(member=member,method=method,duration_h=int(d),
                    D_RL=float(scores[m,0].mean()),B=float(scores[m,1].mean()),S_Q=float(scores[m,2].mean()),
                    median_D=float(np.median(scores[m,0])),D_P10=float(np.quantile(scores[m,0],.1)),D_P90=float(np.quantile(scores[m,0],.9))))
        print(f'Grid {member}: 10,397 cells x 7 durations recalculated',flush=True)
    a=pd.DataFrame(rows)
    b=pd.read_csv(ROOT/'expected_results/grid_summary.csv',dtype={'member':str})
    report=compare(a,b,['member','method','duration_h'],['D_RL','B','S_Q','median_D','D_P10','D_P90'])
    a.to_csv(OUT/'grid_by_member_duration.csv',index=False)
    mean=a.groupby('method')[['D_RL','B','S_Q']].mean().reindex(METHODS)
    mean.to_csv(OUT/'grid_overall.csv')
    report.update(members=4,land_cells=10397,durations=7,ucf_improvement_percent=float(100*(1-mean.loc['UCF','D_RL']/mean.loc['NC','D_RL'])))
    return report

def stations():
    source=ROOT/'midas_station_validation_7/transfer_v1'
    cohort=pd.read_csv(source/'cohort.csv',dtype={'station_id':str})
    def fit_file(filename,durations):
        a=pd.read_csv(source/filename,dtype={'station_id':str})
        pars=np.empty((len(durations),3,len(cohort)))
        for k,station in enumerate(cohort.itertuples()):
            pivot=a[a.station_id==station.station_id].pivot(index='year',columns='duration_h',values='ams_mm').sort_index()
            samples=np.ascontiguousarray(pivot[list(durations)].to_numpy().T)
            assert np.isfinite(samples).all()
            p,*_=fit_batch(samples,np.empty(0))
            assert np.isfinite(p).all() and (p[:,1]>0).all()
            pars[:,:,k]=p
        return pars
    h=fit_file('early_hourly.csv',DURATIONS)
    hd=fit_file('early_daily.csv',[24,48,72]);fd=fit_file('late_daily.csv',[24,48,72])
    sc,_=scaling(np.stack([hd,fd]))
    predicted=[]
    for j,d in enumerate(DURATIONS):
        q,valid,*_=prediction_curves(h[j],hd[0],fd[0],sc,d)
        assert valid.all();predicted.append(q[:,IDS])
    # All station predictions have been computed before fitting late hourly AMS.
    truthp=fit_file('heldout_late_hourly.csv',DURATIONS)
    rows=[]
    for j,d in enumerate(DURATIONS):
        scores=score_curves(predicted[j],gev_curves(truthp[j],reduced_variate(T)))
        assert np.isfinite(scores).all()
        for k,station in enumerate(cohort.itertuples()):
            for m,method in enumerate(METHODS):
                rows.append(dict(station_id=station.station_id,station_name=station.station_name,primary=bool(station.primary),duration_h=int(d),method=method,D_RL=scores[m,0,k],B=scores[m,1,k],S_Q=scores[m,2,k]))
    a=pd.DataFrame(rows);b=pd.read_csv(ROOT/'expected_results/station_point_scores.csv',dtype={'station_id':str})
    report=compare(a,b,['station_id','duration_h','method'],['D_RL','B','S_Q'])
    a.to_csv(OUT/'station_scores.csv',index=False)
    a[a.primary].groupby('method')[['D_RL','B','S_Q']].mean().reindex(METHODS).to_csv(OUT/'station_primary6_overall.csv')
    report.update(primary_stations=6,exploratory_stations=1,refitted_from_supplied_AMS=True)
    print('Stations: supplied AMS refitted, all 196 station-duration-method scores checked',flush=True)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=4);args=ap.parse_args()
    if args.threads<1:ap.error('--threads must be positive')
    set_num_threads(min(args.threads,config.NUMBA_NUM_THREADS));OUT.mkdir(exist_ok=True)
    start=time.perf_counter()
    report=dict(status='passed',grid=grids(),stations=stations(),seconds=time.perf_counter()-start,
      scope='Full grid comparison from supplied fits; station fits and comparison from supplied AMS. No raw-hourly extraction or full bootstrap rerun.',
      target_hourly_role='Evaluation only; not supplied to transfer prediction functions.')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
