"""Locate bootstrap extremes on the actual land domain and add NC width summaries."""
import pandas as pd
from numba import set_num_threads
from research_v3_common import *
from bootstrap_transfer import percentiles

set_num_threads(2)
DEST=OUT/'uncertainty_audit';DEST.mkdir(parents=True,exist_ok=True)
geo=geometry();cells=geo['cells'];n=len(cells);rows=[];extremes=[]
width=np.empty((4,4,7,2,n),np.float32);bands=np.empty((4,3,4,7,2,n),np.float32)
targets=np.array([10.,100.]);aa=reduced_variate(np.array([2.,10.,100.]))
for i,member in enumerate(MEMBERS):
    p=np.load(OUT/'joint_bootstrap'/member/'allowed_parameters.npy',mmap_mode='r')
    for j,d in enumerate(DURATIONS):
        h=np.ascontiguousarray(p[j]).reshape(3,-1)
        sampled=gev_curves(h,aa).reshape(3,500,n)
        ok=np.all(np.isfinite(sampled)&(sampled>0),axis=0);sampled[:,~ok]=np.nan
        band=percentiles(sampled[1:])
        bands[i,:,0,j]=band;width[i,0,j]=np.log(band[2]/band[0])
    del p
    # Legacy HDF5 chunks were written by pixel. Sequentially preload the file
    # to RAM before selecting duration/T, avoiding thousands of HDD seeks.
    blob=(ROOT/'transfer_uncertainty_v1'/f'bootstrap_{member}.nc').read_bytes()
    print(f'UNCERTAINTY FILE IN RAM {member}',flush=True)
    with nc.Dataset('audit_in_memory.nc',memory=blob) as ds:
        periods=np.asarray(ds['return_period'][:]);ids=[int(np.flatnonzero(np.isclose(periods,t,rtol=0,atol=1e-10))[0]) for t in targets]
        ds['depth_ci'].set_var_chunk_cache(256*1024*1024,10007,.75)
        for m in range(3):
            for j,d in enumerate(DURATIONS):
                a=np.asarray(ds['depth_ci'][:,m,j,ids,:],float)[:,:,cells]
                assert np.isfinite(a).all() and np.all(a>0) and np.all(np.diff(a,axis=0)>=0)
                bands[i,:,m+1,j]=a;width[i,m+1,j]=np.log(a[2]/a[0])
    del blob
    point=np.load(OUT/'point_diagnostics'/f'predictions_{member}.npz')
    for m,method in enumerate(METHODS):
        for j,d in enumerate(DURATIONS):
            for k,period in enumerate(targets):
                w=width[i,m,j,k];hi=bands[i,2,m,j,k]
                rows.append(dict(member=member,method=method,duration_h=int(d),T=period,
                    W_median=float(np.median(w)),W_P90=float(np.quantile(w,.9)),W_P99=float(np.quantile(w,.99)),
                    upper_median_mm=float(np.median(hi)),upper_max_mm=float(hi.max()),land_cells=n))
                for rank,g in enumerate(np.argsort(hi)[-5:][::-1],1):
                    pixel=cells[g];y,x=np.unravel_index(pixel,geo['land'].shape)
                    qbase=prediction_curves(point['h'][j,:,g:g+1],point['daily'][0,0,:,g:g+1],point['daily'][1,0,:,g:g+1],point['scale'][:,:,g:g+1],d)[0]
                    tid=int(np.argmin(abs(BASE_T-period)))
                    extremes.append(dict(member=member,method=method,duration_h=int(d),T=period,rank=rank,
                        grid_index=int(pixel),latitude=float(geo['latitude'][y,x]),longitude=float(geo['longitude'][y,x]),
                        point_mm=float(qbase[m,tid,0]),low_mm=float(bands[i,0,m,j,k,g]),upper_mm=float(hi[g]),W=float(w[g])))
    print(f'UNCERTAINTY DOMAIN AUDIT {member}',flush=True)
np.savez_compressed(DEST/'land_widths.npz',W=width,depth_ci=bands,T=targets,cells=cells)
pd.DataFrame(rows).to_csv(DEST/'width_summary.csv',index=False)
pd.DataFrame(extremes).to_csv(DEST/'extreme_intervals.csv',index=False)
save_json(DEST/'complete.json',dict(status='complete',members=MEMBERS,land_cells=n,T=targets.tolist(),methods=METHODS,
    interval='Original 500-draw pointwise prediction percentile interval; NC derived from the same historical draws',
    maximum_land_upper_mm=float(bands[:,2].max()),no_magnitude_capping=True,
    scope='Audit of both 10- and 100-year levels for all seven target durations; maximum over T=2..100 occurs at100 for monotone curves.'))
