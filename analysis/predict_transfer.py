"""Prediction stage. Exact input allowlist excludes ALL future-hourly files."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,hashlib,time
from pathlib import Path
import numpy as np
import netCDF4 as nc
from numba import set_num_threads
from transfer_core import METHODS,DURATIONS,T_ENGINEERING,reduced_variate,scaling,predict_curves,valid_curves,tests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'transfer_experiment_v1'
MEMBERS=['01','04','07','08']
T=np.geomspace(2,100,201)
CONFIG=dict(historical_years=[1981,2020],future_years=[2041,2080],members=MEMBERS,
            durations_hours=DURATIONS.tolist(),daily_durations_hours=[24,48,72],methods=METHODS,
            return_period_range=[2,100],integration_points=201,convergence_points=401,
            engineering_return_periods=T_ENGINEERING.tolist(),tps_shape='same-period 1-day xi fixed across durations',
            tps_scaling='separate log least squares exponents for intensity location and scale, anchored at 1 day',
            confidence_intervals=False,validity='finite positive and nondecreasing curves; common valid locations for method comparison')

def sources(member):
    return {
        'historical_hourly':ROOT/'ukcp18_gev_lmoments_40yr'/f'ukcp18_{member}_AMS_GEV_1981-2020_DecNov.nc',
        'historical_daily':ROOT/'ukcp18_historical_calendar_day_ams_gev'/f'ukcp18_{member}_AMS_GEV_1981-2020_calendar_day.nc',
        'future_daily':ROOT/'ukcp18_future_calendar_day_ams_gev'/f'ukcp18_{member}_AMS_GEV_2041-2080_calendar_day.nc'}

def load_allowed(path,allowed):
    assert path.resolve() in {p.resolve() for p in allowed.values()},'Unapproved prediction input'
    with nc.Dataset(path) as ds:
        params=np.stack([np.ma.filled(ds[name][:],np.nan) for name in ['gev_location','gev_scale','gev_shape']],axis=1).reshape(-1,3,43920)
        duration=np.array(ds['duration'][:])
        coords={}
        for name in ['projection_y_coordinate','projection_x_coordinate','projection_y_coordinate_bnds','projection_x_coordinate_bnds','latitude','longitude','transverse_mercator']:
            v=ds[name];coords[name]=(np.array(v[:]),v.dimensions,{a:v.getncattr(a) for a in v.ncattrs() if a!='_FillValue'})
    return params,duration,coords

def main():
    OUT.mkdir(exist_ok=True);set_num_threads(20);tests()
    (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    (OUT/'prediction_complete.json').unlink(missing_ok=True)
    audit=[];summary=[];started=time.perf_counter()
    for member in MEMBERS:
        allowed=sources(member);inputs={};coordinates=None
        for role,path in allowed.items():
            params,d,coords=load_allowed(path,allowed)
            if coordinates is None:coordinates=coords
            else:
                for key in coords:assert np.array_equal(coords[key][0],coordinates[key][0])
            ids=[int(np.flatnonzero(d==duration)[0]) for duration in (DURATIONS if role=='historical_hourly' else np.array([24,48,72]))]
            inputs[role]=params[ids]
            audit.append(dict(stage='prediction',member=member,role=role,path=str(path),bytes=path.stat().st_size,
                              parameter_sha256=hashlib.sha256(inputs[role].tobytes()).hexdigest()))
        daily=np.stack([inputs['historical_daily'],inputs['future_daily']]);sc,res=scaling(daily)
        dest=OUT/f'predictions_{member}.nc';tmp=dest.with_suffix('.tmp.nc')
        counts=np.zeros((3,3),np.int64)
        with nc.Dataset(tmp,'w') as ds:
            for name,size in [('method',3),('duration',7),('return_period',201),('engineering_return_period',7),('projection_y_coordinate',244),('projection_x_coordinate',180),('bnds',2),('period',2),('daily_duration',3),('parameter',3),('scaling_parameter',5),('scaled_parameter',2)]:ds.createDimension(name,size)
            ds.title='Frozen daily-informed predictions: UCF, HQT, historical-hourly-anchored GEV TPS'
            ds.member=member;ds.prediction_inputs=json.dumps({k:str(v) for k,v in allowed.items()})
            ds.future_hourly_access='NONE: only the three allowlisted inputs were opened'
            ds.config=json.dumps(CONFIG);ds.source_units='Rainfall depth mm; intensity = depth/duration'
            ds.tps_shape='Historical and future each use their own 1-day xi at every target duration'
            ds.curve_storage='Dense depth curves float32; frozen coefficients and engineering quantiles float64'
            for name,values in [('method',np.array(METHODS,dtype=object)),('period',np.array(['historical','future'],dtype=object)),('parameter',np.array(['location','scale','xi'],dtype=object)),('scaling_parameter',np.array(['intensity_a1','intensity_b1','xi_1day','eta_a','eta_b'],dtype=object)),('scaled_parameter',np.array(['location','scale'],dtype=object))]:
                v=ds.createVariable(name,str,(name,));v[:]=values
            for name,values in [('duration',DURATIONS),('return_period',T),('engineering_return_period',T_ENGINEERING),('daily_duration',[24,48,72])]:
                v=ds.createVariable(name,'f8',(name,));v[:]=values;v.units='hours' if 'duration' in name else 'years'
            for name,(values,dims,attrs) in coordinates.items():
                v=ds.createVariable(name,values.dtype,dims);v[:]=values;v.setncatts(attrs)
            spatial=('projection_y_coordinate','projection_x_coordinate')
            for name,values,dims in [('historical_hourly_parameters',inputs['historical_hourly'],('duration','parameter')+spatial),
                                    ('daily_parameters',daily,('period','daily_duration','parameter')+spatial),
                                    ('tps_coefficients',sc,('period','scaling_parameter')+spatial),
                                    ('tps_log_scaling_rmse',res,('period','scaled_parameter')+spatial)]:
                v=ds.createVariable(name,'f8',dims,zlib=True,complevel=1,fill_value=np.nan);v[:]=values.reshape(tuple(len(ds.dimensions[d]) for d in dims))
            vq=ds.createVariable('predicted_depth','f4',('method','duration','return_period')+spatial,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,1,61,180));vq.units='mm'
            for name in ['predicted_depth_engineering','predicted_intensity_engineering','predicted_log_change_engineering']:
                v=ds.createVariable(name,'f8',('method','duration','engineering_return_period')+spatial,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,1,61,180));v.units='mm' if name=='predicted_depth_engineering' else ('mm hour-1' if 'intensity' in name else '1')
            for name in ['finite_positive_curve','monotone_curve','valid_curve']:
                ds.createVariable(name,'u1',('method','duration')+spatial,zlib=True,complevel=1)
            for name,typ in [('hqt_u_star','f8'),('hqt_log_negative_log_u','f8'),('hqt_support_boundary','u1')]:
                ds.createVariable(name,typ,('return_period',)+spatial,zlib=True,complevel=1,chunksizes=(1,61,180))
            ds['hqt_support_boundary'].description='0 interior; 1 above historical daily finite upper support; 2 below finite lower support. Exact CDF endpoint used; no clipping.'
            for j,t in enumerate(DURATIONS):
                h=inputs['historical_hourly'][j]
                q,u,lt,boundary=predict_curves(h,daily[0,0],daily[1,0],sc,float(t),reduced_variate(T))
                pos,mono,good=valid_curves(q)
                vq[:,j]=q.reshape(3,201,244,180)
                for name,values in [('finite_positive_curve',pos),('monotone_curve',mono),('valid_curve',good)]:ds[name][:,j]=values.reshape(3,244,180)
                counts[:,0]+=np.sum(~pos,axis=1);counts[:,1]+=np.sum(pos&~mono,axis=1);counts[:,2]+=np.sum(good,axis=1)
                if j==0:
                    ds['hqt_u_star'][:]=u.reshape(201,244,180);ds['hqt_log_negative_log_u'][:]=lt.reshape(201,244,180);ds['hqt_support_boundary'][:]=boundary.reshape(201,244,180)
                eng=predict_curves(h,daily[0,0],daily[1,0],sc,float(t),reduced_variate(T_ENGINEERING))[0]
                from transfer_core import gev_curves
                historical=gev_curves(h,reduced_variate(T_ENGINEERING))
                ds['predicted_depth_engineering'][:,j]=eng.reshape(3,7,244,180)
                ds['predicted_intensity_engineering'][:,j]=(eng/t).reshape(3,7,244,180)
                with np.errstate(invalid='ignore',divide='ignore'):change=np.log(eng/historical[None])
                ds['predicted_log_change_engineering'][:,j]=change.reshape(3,7,244,180)
        with nc.Dataset(tmp) as ds:
            assert ds['predicted_depth'].shape==(3,7,201,244,180)
            assert np.allclose(np.ma.filled(ds['predicted_depth'][:,-1],np.nan).reshape(3,201,-1),q,rtol=1e-6,atol=1e-6,equal_nan=True)
        os.replace(tmp,dest)
        summary.append(dict(member=member,nonpositive_or_nonfinite=counts[:,0].tolist(),nonmonotone_positive=counts[:,1].tolist(),valid=counts[:,2].tolist()))
        print('PREDICTED '+json.dumps(summary[-1]),flush=True)
    (OUT/'prediction_access_audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'prediction_complete.json').write_text(json.dumps(dict(status='complete',future_hourly_files_opened=0,files=4,elapsed_seconds=time.perf_counter()-started,completed_at=time.strftime('%Y-%m-%dT%H:%M:%S'),results=summary),indent=2))
    print('ALL PREDICTIONS FROZEN; future-hourly inputs have not been opened',flush=True)

if __name__=='__main__':main()
