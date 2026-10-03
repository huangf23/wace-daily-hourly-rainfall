"""Evaluation only: future-hourly data first become accessible after prediction freeze."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,csv,time,hashlib
from pathlib import Path
import numpy as np
import netCDF4 as nc
import shapefile
from numba import set_num_threads
from transfer_core_regularized import METHODS,DURATIONS,T_ENGINEERING,reduced_variate,predict_curves,gev_curves,valid_curves,score_curves

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'transfer_experiment_v2_regularized'

def land_mask(x,y):
    mask=np.zeros((len(y),len(x)),bool)
    for r in shapefile.Reader(str(ROOT/'source'/'ukcp_land_mask'/'ukcp18-uk-land-5km.shp')).records():
        ix=int(np.argmin(abs(x-r['x_coord'])));iy=int(np.argmin(abs(y-r['y_coord'])))
        assert x[ix]==r['x_coord'] and y[iy]==r['y_coord'];mask[iy,ix]=True
    assert mask.sum()==10397
    return mask

def main():
    marker=json.loads((OUT/'prediction_complete.json').read_text())
    assert marker['status']=='complete' and marker['future_hourly_files_opened']==0
    set_num_threads(20);started=time.perf_counter();rows=[];audit=[]
    T401=np.geomspace(2,100,401);a401=reduced_variate(T401)
    convergence=[];member_stats=[]
    for member in ['01','04','07','08']:
        predpath=OUT/f'predictions_{member}.nc'
        truthpath=ROOT/'ukcp18_gev_lmoments_40yr'/f'ukcp18_{member}_AMS_GEV_2041-2080_DecNov.nc'
        with nc.Dataset(predpath) as pred:
            h=np.ma.filled(pred['historical_hourly_parameters'][:],np.nan).reshape(7,3,43920)
            daily=np.ma.filled(pred['daily_parameters'][:],np.nan).reshape(2,3,3,43920)
            scale=np.ma.filled(pred['tps_coefficients'][:],np.nan).reshape(2,5,43920)
            frozen_hash=hashlib.sha256(h.tobytes()+daily.tobytes()+scale.tobytes()).hexdigest()
            x=pred['projection_x_coordinate'][:];y=pred['projection_y_coordinate'][:];land=land_mask(x,y).ravel()
            # This is the first access to future-hourly data, in evaluation only.
            with nc.Dataset(truthpath) as src:
                ids=[int(np.flatnonzero(src['duration'][:]==d)[0]) for d in DURATIONS]
                truth=np.stack([np.ma.filled(src[name][ids],np.nan) for name in ['gev_location','gev_scale','gev_shape']],axis=1).reshape(7,3,43920)
                for name in ['projection_x_coordinate','projection_y_coordinate']:assert np.array_equal(src[name][:],pred[name][:])
            audit.append(dict(stage='evaluation',member=member,role='withheld_future_hourly_pseudo_reality',path=str(truthpath),
                              prediction_completion=marker['completed_at'],frozen_prediction_parameter_sha256=frozen_hash,
                              evaluation_started=time.strftime('%Y-%m-%dT%H:%M:%S')))
            dest=OUT/f'evaluation_{member}.nc';tmp=dest.with_suffix('.tmp.nc')
            allcommon=0
            with nc.Dataset(tmp,'w') as ds:
                for name,size in [('method',3),('duration',7),('return_period',201),('engineering_return_period',7),('projection_y_coordinate',244),('projection_x_coordinate',180),('bnds',2)]:ds.createDimension(name,size)
                ds.title='Validation against withheld future-hourly GEV; prediction coefficients frozen beforehand'
                ds.member=member;ds.truth_file=str(truthpath);ds.prediction_file=str(predpath)
                ds.integration='Trapezoidal mean over uniformly spaced log(T), T=2..100; primary 201 points, check 401 points'
                ds.metric_definition='D_RL=mean(abs(log(pred/true))); B=mean(log(pred/true)); S=sqrt(mean((log(pred/true)-B)^2))'
                ds.common_mask='All three predictions finite, positive and nondecreasing on 401-point grid, and truth finite positive nondecreasing'
                ds.truth_role='Comparison only; no fitting, selection or tuning of transfer hyperparameters with future hourly'
                ds.uncertainty='Point estimates only; bootstrap not included in this deterministic run'
                for name in ['method','duration','return_period','engineering_return_period','projection_y_coordinate','projection_x_coordinate','projection_y_coordinate_bnds','projection_x_coordinate_bnds','latitude','longitude','transverse_mercator']:
                    v=pred[name];out=ds.createVariable(name,str if name=='method' else v.dtype,v.dimensions);out[:]=v[:]
                    out.setncatts({a:v.getncattr(a) for a in v.ncattrs() if a!='_FillValue'})
                spatial=('projection_y_coordinate','projection_x_coordinate');mdims=('method','duration')+spatial
                for name in ['D_RL','B_log_bias','S_shape_error','D_RL_401','B_401','S_401','integration_abs_difference_D','geometric_bias_pct']:
                    ds.createVariable(name,'f8',mdims,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,61,180))
                for name in ['valid_curve_401','positive_curve_401','monotone_curve_401']:
                    ds.createVariable(name,'u1',mdims,zlib=True,complevel=1)
                ds.createVariable('common_valid','u1',('duration',)+spatial,zlib=True,complevel=1)
                v=ds.createVariable('best_method','i1',('duration',)+spatial,fill_value=-1,zlib=True,complevel=1);v.description='0 UCF, 1 HQT, 2 TPS; -1 outside common valid set; minimum D_RL, ties in listed order'
                ds.createVariable('land_mask','u1',spatial)[:]=land.reshape(244,180)
                for name in ['true_depth_engineering','true_log_change_engineering']:
                    ds.createVariable(name,'f8',('duration','engineering_return_period')+spatial,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,61,180))
                ds.createVariable('relative_error_engineering_pct','f8',('method','duration','engineering_return_period')+spatial,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,1,61,180))
                # Dense reference and error curves are recoverable from coefficients;
                # retain the true curve for convenient inspection alongside frozen predictions.
                ds.createVariable('true_depth','f4',('duration','return_period')+spatial,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,61,180))
                for j,t in enumerate(DURATIONS):
                    # Pure prediction function accepts ONLY the frozen allowable inputs.
                    q=predict_curves(h[j],daily[0,0],daily[1,0],scale,float(t),a401)[0]
                    tr=gev_curves(truth[j],a401)
                    stored=np.ma.filled(pred['predicted_depth'][:,j],np.nan).reshape(3,201,43920)
                    assert np.allclose(stored,q[:,::2],rtol=1e-6,atol=1e-6,equal_nan=True),'Frozen prediction reproducibility failure'
                    scores=score_curves(q[:,::2],tr[::2]);fine=score_curves(q,tr)
                    pos,mono,good=valid_curves(q);truegood=valid_curves(tr)[2]
                    common=np.all(good,axis=0)&truegood;allcommon+=int(common.sum())
                    assert np.all(common), 'Unexpected invalid prediction/truth curve'
                    assert np.isfinite(scores).all() and np.isfinite(fine).all()
                    best=np.full(43920,-1,np.int8);best[common]=np.argmin(scores[:,0,common],axis=0)
                    for name,values in [('D_RL',scores[:,0]),('B_log_bias',scores[:,1]),('S_shape_error',scores[:,2]),
                                        ('D_RL_401',fine[:,0]),('B_401',fine[:,1]),('S_401',fine[:,2]),
                                        ('integration_abs_difference_D',abs(scores[:,0]-fine[:,0])),('geometric_bias_pct',100*np.expm1(scores[:,1]))]:
                        ds[name][:,j]=values.reshape(3,244,180)
                    for name,values in [('valid_curve_401',good),('positive_curve_401',pos),('monotone_curve_401',mono)]:ds[name][:,j]=values.reshape(3,244,180)
                    ds['common_valid'][j]=common.reshape(244,180);ds['best_method'][j]=best.reshape(244,180)
                    ds['true_depth'][j]=tr[::2].reshape(201,244,180)
                    qt=gev_curves(truth[j],reduced_variate(T_ENGINEERING));qh=gev_curves(h[j],reduced_variate(T_ENGINEERING))
                    qe=np.ma.filled(pred['predicted_depth_engineering'][:,j],np.nan).reshape(3,7,43920)
                    ds['true_depth_engineering'][j]=qt.reshape(7,244,180)
                    ds['true_log_change_engineering'][j]=np.log(qt/qh).reshape(7,244,180)
                    ds['relative_error_engineering_pct'][:,j]=(100*(qe/qt[None]-1)).reshape(3,7,244,180)
                    for domain,mask in [('full_grid',np.ones(43920,bool)),('uk_land',land)]:
                        use=common&mask;n=int(use.sum())
                        for m,method in enumerate(METHODS):
                            val=scores[m,0,use];diff=abs(scores[m,0,use]-fine[m,0,use])
                            rows.append(dict(member=member,duration_hours=int(t),domain=domain,method=method,
                                domain_cells=int(mask.sum()),common_valid_cells=n,own_valid_cells=int((good[m]&mask).sum()),
                                nonpositive_or_nonfinite_cells=int((~pos[m]&mask).sum()),nonmonotone_positive_cells=int((pos[m]&~mono[m]&mask).sum()),
                                mean_D_RL=float(val.mean()) if n else np.nan,median_D_RL=float(np.median(val)) if n else np.nan,
                                median_multiplicative_abs_error_pct=float(100*np.expm1(np.median(val))) if n else np.nan,
                                median_B=float(np.median(scores[m,1,use])) if n else np.nan,median_S=float(np.median(scores[m,2,use])) if n else np.nan,
                                best_method_cells=int(((best==m)&use).sum()),integration_diff_median=float(np.median(diff)) if n else np.nan,
                                integration_diff_p99=float(np.quantile(diff,.99)) if n else np.nan,integration_diff_max=float(diff.max()) if n else np.nan,
                                integration_diff_over_1e4=int((diff>1e-4).sum())))
                    difference=abs(scores[:,0,common]-fine[:,0,common])
                    convergence.append(dict(member=member,duration=int(t),common_valid_cells=int(common.sum()),
                        max_D_difference=float(difference.max()) if common.any() else None,
                        over_1e4=int((difference>1e-4).sum()),comparisons=int(difference.size)))
                    print(f'EVALUATED {member} {t}h common={common.sum()}/43920',flush=True)
            with nc.Dataset(tmp) as ds:
                assert ds['D_RL'].shape==(3,7,244,180)
                assert np.allclose(np.ma.filled(ds['D_RL'][:,-1],np.nan).reshape(3,-1),scores[:,0],equal_nan=True)
            os.replace(tmp,dest)
            member_stats.append(dict(member=member,common_valid_duration_grid_pairs=allcommon,total_duration_grid_pairs=7*43920))
    with (OUT/'evaluation_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'evaluation_access_audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'integration_convergence.json').write_text(json.dumps(convergence,indent=2))
    report=dict(status='complete',prediction_files=4,evaluation_files=4,target_durations_hours=DURATIONS.tolist(),
                primary_integration=[2,100,201],convergence_points=401,engineering_return_periods=T_ENGINEERING.tolist(),
                prediction_future_hourly_access_count=0,evaluation_future_hourly_access_count=len(audit),
                evaluation_seconds=time.perf_counter()-started,prediction_seconds=marker['elapsed_seconds'],members=member_stats,
                bootstrap_performed=False,summary_rows=len(rows),source='Existing validated L-moment GEV fits',
                integration_comparisons=sum(r['comparisons'] for r in convergence),integration_difference_gt_1e4=sum(r['over_1e4'] for r in convergence))
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2))
    print('FINISHED '+json.dumps(report),flush=True)

if __name__=='__main__':main()
