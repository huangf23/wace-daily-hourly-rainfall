"""Shared readers for v3 supplementary analyses; prediction inputs stay explicit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import json, hashlib
import numpy as np
import netCDF4 as nc
from transfer_core import DURATIONS, T_ENGINEERING, reduced_variate, gev_curves, score_curves, scaling
from transfer_core_regularized import BASE_A, BASE_T, WEIGHTS, bounded_hqt, project_log_curves
from transfer_core import predict_curves as raw_predict
from predict_transfer import sources

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_v3'
SRC=ROOT/'transfer_experiment_v2_regularized'
MEMBERS=['01','04','07','08']
METHODS=['NC','UCF','HQT','TPS']
T=np.geomspace(2,100,201)
IDS=np.abs(BASE_A[:,None]-reduced_variate(T)[None,:]).argmin(axis=0)
ENG=np.abs(BASE_A[:,None]-reduced_variate(T_ENGINEERING)[None,:]).argmin(axis=0)

def save_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def geometry():
    with nc.Dataset(SRC/'evaluation_01.nc') as ds:
        land=np.asarray(ds['land_mask'][:],bool)
        result={k:np.asarray(ds[k][:]) for k in ['latitude','longitude','projection_x_coordinate','projection_y_coordinate']}
    result.update(land=land,cells=np.flatnonzero(land.ravel()))
    cases=json.loads((ROOT/'paper_figures_by_member'/'cases.json').read_text(encoding='utf-8'))
    result['cases']=cases
    result['case_land_indices']=np.array([np.flatnonzero(result['cells']==c['grid_index'])[0] for c in cases])
    return result

def read_ams(path,durations,cells):
    with nc.Dataset(path) as ds:
        ids=[int(np.flatnonzero(np.asarray(ds['duration'][:])==d)[0]) for d in durations]
        years=np.asarray(ds['year'][:])
        a=np.asarray(ds['ams_depth'][:,ids],float).reshape(len(years),len(ids),-1)[:,:,cells]
    return np.ascontiguousarray(a),years

def allowed_ams(member,cells):
    data={};audit=[]
    for role,path in sources(member).items():
        assert not ('2041-2080_DecNov' in str(path))
        arr,years=read_ams(path,DURATIONS if role=='historical_hourly' else [24,48,72],cells)
        assert np.array_equal(years,np.arange(2041,2081) if role=='future_daily' else np.arange(1981,2021))
        assert np.isfinite(arr).all() and np.all(arr>0)
        data[role]=arr
        audit.append(dict(role=role,path=str(path),sha256_land_ams=hashlib.sha256(arr.tobytes()).hexdigest()))
    return data,audit

def reference_ams(member,cells):
    return read_ams(ROOT/'ukcp18_gev_lmoments_40yr'/f'ukcp18_{member}_AMS_GEV_2041-2080_DecNov.nc',DURATIONS,cells)[0]

def frozen_parameters(member,cells):
    with nc.Dataset(SRC/f'predictions_{member}.nc') as ds:
        h=np.asarray(ds['historical_hourly_parameters'][:],float).reshape(7,3,-1)[:,:,cells]
        daily=np.asarray(ds['daily_parameters'][:],float).reshape(2,3,3,-1)[:,:,:,cells]
        scale=np.asarray(ds['tps_coefficients'][:],float).reshape(2,5,-1)[:,:,cells]
    return np.ascontiguousarray(h),np.ascontiguousarray(daily),np.ascontiguousarray(scale)

def reference_parameters(member,cells):
    path=ROOT/'ukcp18_gev_lmoments_40yr'/f'ukcp18_{member}_AMS_GEV_2041-2080_DecNov.nc'
    with nc.Dataset(path) as ds:
        ids=[int(np.flatnonzero(np.asarray(ds['duration'][:])==d)[0]) for d in DURATIONS]
        p=np.stack([np.asarray(ds[k][ids],float) for k in ['gev_location','gev_scale','gev_shape']],axis=1).reshape(7,3,-1)[:,:,cells]
    return np.ascontiguousarray(p)

def prediction_curves(h,hd,fd,sc,d):
    """Four methods on BASE_T; no target-hourly argument or file access."""
    raw,u,lt,boundary=raw_predict(h,hd,fd,sc,float(d),BASE_A)
    clipped=bounded_hqt(h,lt,raw,u)
    hist=gev_curves(h,BASE_A)
    out=np.full((4,len(BASE_A),h.shape[-1]),np.nan)
    out[0]=hist
    good=np.zeros((4,h.shape[-1]),bool)
    adjusted=np.zeros_like(good)
    good[0]=np.all(np.isfinite(hist)&(hist>0),axis=0)
    out[0,:,~good[0]]=np.nan
    for m in range(3):
        ok=np.all(np.isfinite(raw[m])&(raw[m]>0),axis=0)
        good[m+1]=ok
        if ok.any():
            v,flags=project_log_curves(np.ascontiguousarray(raw[m:m+1,:,ok].reshape(1,len(BASE_A),-1)),WEIGHTS)
            out[m+1][:,ok]=v[0]
            adjusted[m+1,ok]=flags[0]
    return out,good,adjusted,clipped,boundary

def cf_survival(x,p):
    """Stable GEV survival, keeping true support endpoints (including zero)."""
    from scipy.stats import genextreme
    return genextreme.sf(x,c=-p[2],loc=p[0],scale=p[1])
