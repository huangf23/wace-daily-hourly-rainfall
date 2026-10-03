"""Paired spatial/duration circular year-block bootstrap; prediction inputs only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time,traceback,hashlib
from pathlib import Path
import numpy as np
import netCDF4 as nc
from numba import set_num_threads,njit,prange
from predict_transfer import sources
from gev_lmoments_kernel import fit_batch
from transfer_core import predict_curves as raw_predict
from transfer_core_regularized import (BASE_A,BASE_T,WEIGHTS,bounded_hqt,project_log_curves,
    scaling,gev_curves,DURATIONS,T_ENGINEERING,METHODS,reduced_variate)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'transfer_uncertainty_v1'
PERIODS=np.unique(np.r_[np.geomspace(2,100,201),T_ENGINEERING])
IDS=np.abs(BASE_A[:,None]-reduced_variate(PERIODS)[None,:]).argmin(axis=0)
QUANTILES=[.025,.5,.975]

@njit(parallel=True,cache=True,nogil=True)
def percentiles(sample):
    nt,nb,ng=sample.shape
    out=np.full((3,nt,ng),np.nan)
    for cell in prange(nt*ng):
        t=cell//ng;g=cell%ng
        values=sample[t,:,g]
        values=np.sort(values[np.isfinite(values)])
        if len(values)==0:continue
        for k,p in enumerate((.025,.5,.975)):
            pos=(len(values)-1)*p;i=int(pos);fraction=pos-i
            out[k,t,g]=values[i]*(1-fraction)+values[min(i+1,len(values)-1)]*fraction
    return out

def indices(seed,b,length=3):
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,40,size=(b,int(np.ceil(40/length))))
    return ((starts[:,:,None]+np.arange(length))%40).reshape(b,-1)[:,:40]

def read_inputs(member):
    data={};audit=[];coordinates=None
    for role,path in sources(member).items():
        assert '2041-2080_DecNov' not in str(path)
        with nc.Dataset(path) as ds:
            target=DURATIONS if role=='historical_hourly' else [24,48,72]
            ids=[int(np.flatnonzero(ds['duration'][:]==d)[0]) for d in target]
            years=np.asarray(ds['year'][:]);assert np.array_equal(years,np.arange(2041,2081) if role=='future_daily' else np.arange(1981,2021))
            arr=np.asarray(ds['ams_depth'][:,ids],dtype=np.float64).reshape(40,len(ids),43920)
            assert np.isfinite(arr).all() and (arr>0).all()
            data[role]=arr
            xy=[np.asarray(ds[k][:]) for k in ['projection_y_coordinate','projection_x_coordinate']]
            if coordinates is None:coordinates=xy
            else:assert all(np.array_equal(a,b) for a,b in zip(xy,coordinates))
        audit.append(dict(role=role,path=str(path),ams_sha256=hashlib.sha256(arr.tobytes()).hexdigest()))
    return data,audit

def refit(a,index):
    # a: 40 x duration x pixel. Flatten bootstrap first, pixel second.
    b=len(index);n=a.shape[2];result=[];bad=[]
    for d in range(a.shape[1]):
        sample=a[:,d][index].transpose(0,2,1).reshape(b*n,40)
        p,_,_,status,_=fit_batch(sample,np.empty(0,dtype=np.float64))
        result.append(p.T);bad.append(status.reshape(b,n))
    return np.stack(result),np.stack(bad)

def chunk(data,ih,iff,start,end):
    b=len(ih);n=end-start
    h,hs=refit(data['historical_hourly'][:,:,start:end],ih)
    hd,hds=refit(data['historical_daily'][:,:,start:end],ih)
    fd,fds=refit(data['future_daily'][:,:,start:end],iff)
    sc,_=scaling(np.stack([hd,fd]))
    shape=(3,3,7,len(PERIODS),n)
    depths=np.empty(shape,np.float32);changes=np.empty(shape,np.float32)
    valid_counts=np.zeros((3,7,n),np.uint16);adjusted=np.zeros_like(valid_counts)
    clipped_counts=np.zeros((len(PERIODS),n),np.uint16)
    for d,t in enumerate(DURATIONS):
        q,u,lt,boundary=raw_predict(h[d],hd[0],fd[0],sc,float(t),BASE_A)
        clipped=bounded_hqt(h[d],lt,q,u)
        if d==0:clipped_counts[:]=clipped[IDS].reshape(len(PERIODS),b,n).sum(axis=1)
        hist=gev_curves(h[d],reduced_variate(PERIODS))
        for m in range(3):
            good=np.all(np.isfinite(q[m])&(q[m]>0),axis=0)&np.all(np.isfinite(hist)&(hist>0),axis=0)
            valid_counts[m,d]=good.reshape(b,n).sum(axis=0)
            # No invented values or hidden replacement for inadmissible replicates.
            projected,modified=project_log_curves(q[m:m+1,:,good].reshape(1,len(BASE_A),-1),WEIGHTS)
            output=np.full((len(PERIODS),b*n),np.nan)
            output[:,good]=projected[0,IDS]
            adj=np.zeros(b*n,np.uint8);adj[good]=modified[0]
            adjusted[m,d]=adj.reshape(b,n).sum(axis=0)
            sampled=output.reshape(len(PERIODS),b,n)
            depths[:,m,d]=percentiles(sampled)
            logchange=np.log(output/hist).reshape(len(PERIODS),b,n)
            changes[:,m,d]=percentiles(logchange)
    assert np.isfinite(depths).all() and np.isfinite(changes).all(),'No estimable CI at a grid cell; inspect bootstrap diagnostics'
    assert np.all(np.diff(depths,axis=0)>=0)
    return dict(depth_ci=depths,log_change_ci=changes,valid_replicates=valid_counts,
        monotonicity_adjusted_replicates=adjusted,hqt_clipped_replicates=clipped_counts,
        fit_failures=np.array([(hs!=0).sum(),(hds!=0).sum(),(fds!=0).sum()],np.int64))

def create_file(path,b,chunk_size,audit,ih,iff):
    ds=nc.Dataset(path,'w')
    for k,s in [('quantile',3),('method',3),('duration',7),('return_period',len(PERIODS)),('pixel',43920),('replicate',b),('year_draw',40),('chunk',int(np.ceil(43920/chunk_size))),('source_role',3)]:ds.createDimension(k,s)
    ds.config=json.dumps(dict(replicates=b,block_length=3,interval='pointwise percentile 95%; not simultaneous',
        pairing='same years across durations and spatial pixels within a period; independent H and F',
        regularization='v2 epsilon=1e-6 and fixed-grid weighted log-PAVA',future_hourly_access_count=0,
        invalid_replicates='Excluded, counts retained; CI conditional on admissible fits. Low-count flags must be reviewed.'))
    ds.input_audit=json.dumps(audit);ds.chunk_size=chunk_size
    for k,v in [('quantile',QUANTILES),('duration',DURATIONS),('return_period',PERIODS),('pixel',np.arange(43920))]:ds.createVariable(k,'f8',(k,))[:]=v
    v=ds.createVariable('method',str,('method',));v[:]=np.array(METHODS,object)
    for k,idx in [('historical_year_indices',ih),('future_year_indices',iff)]:ds.createVariable(k,'i2',('replicate','year_draw'))[:]=idx
    with nc.Dataset(ROOT/'transfer_experiment_v2_regularized'/'predictions_01.nc') as ref:
        for k in ['latitude','longitude']:ds.createVariable(k,'f8',('pixel',))[:]=np.asarray(ref[k][:]).ravel()
        ds.createVariable('grid_y','f8',('pixel',))[:]=np.repeat(ref['projection_y_coordinate'][:],180)
        ds.createVariable('grid_x','f8',('pixel',))[:]=np.tile(ref['projection_x_coordinate'][:],244)
    for k in ['depth_ci','log_change_ci']:
        v=ds.createVariable(k,'f4',('quantile','method','duration','return_period','pixel'),fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,1,len(PERIODS),chunk_size));v.units='mm' if k=='depth_ci' else '1'
    ds.intensity_ci='Divide depth_ci by duration in hours; same percentile ordering'
    for k in ['valid_replicates','monotonicity_adjusted_replicates']:ds.createVariable(k,'u2',('method','duration','pixel'),zlib=True,complevel=1)
    ds.createVariable('hqt_clipped_replicates','u2',('return_period','pixel'),zlib=True,complevel=1)
    ds.createVariable('fit_failures','i8',('source_role','chunk'))
    ds.createVariable('chunk_complete','u1',('chunk',))[:]=0
    ds.sync();return ds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--replicates',type=int,default=500);ap.add_argument('--chunk',type=int,default=64)
    args=ap.parse_args();set_num_threads(20);OUT.mkdir(exist_ok=True)
    started=time.time()
    for member in ['01','04','07','08']:
        data,audit=read_inputs(member)
        b=20 if args.pilot else args.replicates
        ih=indices(923100+int(member),b);iff=indices(923200+int(member),b)
        if args.pilot:
            tic=time.time();result=chunk(data,ih,iff,0,16)
            report=dict(seconds=time.time()-tic,replicates=b,pixels=16,min_valid=int(result['valid_replicates'].min()),fit_failures=result['fit_failures'].tolist())
            (OUT/'pilot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
        path=OUT/f'bootstrap_{member}.nc'
        ds=nc.Dataset(path,'r+') if path.exists() else create_file(path,b,args.chunk,audit,ih,iff)
        assert len(ds.dimensions['replicate'])==b and ds.chunk_size==args.chunk
        assert np.array_equal(ds['historical_year_indices'][:],ih) and np.array_equal(ds['future_year_indices'][:],iff)
        assert json.loads(ds.input_audit)==audit, 'Input AMS changed since checkpoint; refusing mixed-input resume'
        for c,start in enumerate(range(0,43920,args.chunk)):
            if ds['chunk_complete'][c]==1:continue
            end=min(start+args.chunk,43920);tic=time.time();result=chunk(data,ih,iff,start,end)
            for k,v in result.items():
                if k=='fit_failures':ds[k][:,c]=v
                else:ds[k][...,start:end]=v
            ds.sync();ds['chunk_complete'][c]=1;ds.sync()
            progress=dict(status='running',member=member,chunk=c+1,total_chunks=len(ds.dimensions['chunk']),
                seconds_per_chunk=time.time()-tic,elapsed_seconds=time.time()-started,
                min_valid_replicates=int(result['valid_replicates'].min()),future_hourly_access_count=0)
            tmp=OUT/'progress.tmp.json';tmp.write_text(json.dumps(progress,indent=2));os.replace(tmp,OUT/'progress.json')
            print(json.dumps(progress),flush=True)
        ds.status='complete';ds.close()
    report=dict(status='complete',replicates=args.replicates,block_length=3,members=4,future_hourly_access_count=0,elapsed_seconds=time.time()-started)
    (OUT/'bootstrap_complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    (OUT/'progress.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());raise
