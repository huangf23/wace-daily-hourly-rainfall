"""Split UKCP18 AMS into confirmed 40-year periods and fit each series separately."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import json
import csv
import time
import numpy as np
import netCDF4 as nc
from numba import set_num_threads
from scipy.stats import genextreme
from lmoments3 import distr
from gev_lmoments_kernel import fit_batch,self_test,gev_tau

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'ukcp18_ams_dec_nov'/'series'
OUT=ROOT/'ukcp18_gev_lmoments_40yr'
DUR=np.array([1,2,3,4,5,6,12,24,48,72])
PERIODS=[(1981,2020),(2041,2080)]
RP=np.array([2,5,10,25,50,100],dtype=np.float64)

def main():
    OUT.mkdir(exist_ok=True)
    set_num_threads(20)
    tests=self_test()
    (OUT/'validation_reference.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')
    summary=[];started=time.perf_counter(); max_reference_difference=0.
    for member in ['01','04','07','08']:
        source=SOURCE/f'ukcp18_{member}_AMS_1981-2080_DecNov.nc'
        with nc.Dataset(source) as src:
            years=np.asarray(src['year'][:])
            for y0,y1 in PERIODS:
                begin=time.perf_counter()
                ids=np.flatnonzero((years>=y0)&(years<=y1))
                assert len(ids)==40 and np.array_equal(years[ids],np.arange(y0,y1+1))
                selection=slice(int(ids[0]),int(ids[-1])+1)
                depth=np.ma.filled(src['ams_depth'][selection],np.nan)
                samples=np.ascontiguousarray(depth.transpose(1,2,3,0).reshape(-1,40))
                t=time.perf_counter()
                pars,moments,levels,status,support=fit_batch(samples,RP)
                fit_seconds=time.perf_counter()-t
                good=status==0
                assert np.all(pars[good,1]>0)
                assert np.all(np.isfinite(levels[good]))
                assert np.all(np.diff(levels[good],axis=1)>0)
                # Independent references on a reproducible, spread-out selection.
                check_ids=np.linspace(0,len(samples)-1,200,dtype=int)
                for index in check_ids:
                    if not good[index]:continue
                    ref=distr.gev.lmom_fit(samples[index])
                    ref_par=np.array([ref['loc'],ref['scale'],-ref['c']])
                    delta=float(np.max(abs(pars[index]-ref_par)))
                    max_reference_difference=max(max_reference_difference,delta)
                    assert np.allclose(pars[index],ref_par,rtol=3e-4,atol=3e-4),(member,y0,index,pars[index],ref_par)
                    q=genextreme.ppf(1-1/RP,c=-pars[index,2],loc=pars[index,0],scale=pars[index,1])
                    assert np.allclose(q,levels[index],rtol=1e-10,atol=1e-9)
                    assert abs(gev_tau(pars[index,2])-moments[index,2])<1e-7
                dest=OUT/f'ukcp18_{member}_AMS_GEV_{y0}-{y1}_DecNov.nc'
                tmp=dest.with_suffix('.tmp.nc')
                with nc.Dataset(tmp,'w',format='NETCDF4') as ds:
                    ds.setncatts({a:src.getncattr(a) for a in src.ncattrs()})
                    ds.title=f'UKCP18 member {member}, {y0}-{y1}, split AMS and L-moment GEV'
                    ds.source_ams_file=str(source)
                    ds.period_start_year=y0;ds.period_end_year=y1
                    ds.estimation='Unbiased sample probability weighted moments (order 0,1,2); L1,L2,L-skewness; numerical solution of GEV L-skewness equation'
                    ds.gev_shape_convention='Conventional xi: xi > 0 has heavy upper tail; scipy.stats.genextreme c = -xi'
                    ds.gev_cdf='F(x)=exp(-(1+xi*(x-mu)/sigma)**(-1/xi)), with Gumbel limit at xi=0'
                    ds.return_level_equation='F(q_T)=1-1/T; q_T=mu+sigma/xi*((-log(1-1/T))**(-xi)-1)'
                    ds.reference='Hosking, Wallis & Wood (1985), doi:10.1080/00401706.1985.10488049'
                    ds.model_assumption='Separate stationary GEV for each member, duration, grid cell and 40-year period; no member pooling'
                    ds.input_boundary_note='The 1981-2020 sample retains 1981 with original incomplete pre-record window context flags; no annual observation dropped.'
                    ds.support_note='L-moment estimates may exclude observed values from fitted support. Retained estimates are flagged with support_violation_count; no silent censoring or clipping.'
                    ds.fit_seconds=fit_seconds
                    ds.history=time.strftime('%Y-%m-%dT%H:%M:%S')+' fit_gev_periods.py'
                    for name,dim in src.dimensions.items():ds.createDimension(name,40 if name=='year' else len(dim))
                    ds.createDimension('return_period',len(RP))
                    rp=ds.createVariable('return_period','i4',('return_period',));rp[:]=RP;rp.units='years'
                    # Preserve the actual split AMS, event timestamps, quality and coordinates.
                    for name,v in src.variables.items():
                        opts={}
                        if '_FillValue' in v.ncattrs():opts['fill_value']=v.getncattr('_FillValue')
                        if len(v.dimensions)>=3:
                            opts.update(zlib=True,complevel=1,shuffle=True)
                            opts['chunksizes']=(1,1,61,180) if len(v.dimensions)==4 else (1,61,180)
                        out=ds.createVariable(name,v.dtype,v.dimensions,**opts)
                        out.setncatts({a:v.getncattr(a) for a in v.ncattrs() if a!='_FillValue'})
                        if name=='ams_depth':out[:]=depth
                        elif 'year' in v.dimensions:out[:]=v[selection]
                        else:out[:]=v[:]
                    dims=('duration','projection_y_coordinate','projection_x_coordinate')
                    for name,values,unit,long_name in [
                        ('gev_location',pars[:,0],'mm','GEV location mu'),
                        ('gev_scale',pars[:,1],'mm','GEV scale sigma'),
                        ('gev_shape',pars[:,2],'1','GEV conventional shape xi; scipy c = -xi'),
                        ('sample_l1',moments[:,0],'mm','Unbiased sample first L-moment'),
                        ('sample_l2',moments[:,1],'mm','Unbiased sample second L-moment'),
                        ('sample_tau3',moments[:,2],'1','Sample L-skewness L3/L2')]:
                        v=ds.createVariable(name,'f8',dims,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,61,180))
                        v[:]=values.reshape(10,244,180);v.units=unit;v.long_name=long_name
                        v.grid_mapping='transverse_mercator';v.coordinates='latitude longitude'
                    for name,values in [('fit_status',status),('support_violation_count',support),('sample_size',np.isfinite(samples).sum(axis=1))]:
                        v=ds.createVariable(name,'i2',dims,zlib=True,complevel=1,chunksizes=(1,61,180));v[:]=values.reshape(10,244,180)
                        if name=='fit_status':
                            v.flag_values=np.arange(5,dtype=np.int16)
                            v.flag_meanings='success missing_or_insufficient_sample degenerate_sample invalid_lmoments invalid_parameters'
                    rdims=('return_period',)+dims
                    q=levels.reshape(10,244,180,len(RP)).transpose(3,0,1,2)
                    for name,values,unit in [('return_level_depth',q,'mm'),('return_level_intensity',q/DUR[None,:,None,None],'mm hour-1')]:
                        v=ds.createVariable(name,'f8',rdims,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,61,180))
                        v[:]=values;v.units=unit;v.grid_mapping='transverse_mercator';v.coordinates='latitude longitude'
                with nc.Dataset(tmp) as ds:
                    assert ds['ams_depth'].shape==(40,10,244,180)
                    assert np.array_equal(ds['year'][:],np.arange(y0,y1+1))
                    assert np.allclose(np.ma.filled(ds['return_level_depth'][:],np.nan),q,equal_nan=True)
                    assert np.array_equal(ds['sample_size'][:],np.full((10,244,180),40))
                os.replace(tmp,dest)
                for j,duration in enumerate(DUR):
                    sub=slice(j*43920,(j+1)*43920)
                    row=dict(member=member,period_start=y0,period_end=y1,duration_hours=int(duration),series=43920,
                             successful_fits=int((status[sub]==0).sum()),failed_fits=int((status[sub]!=0).sum()),
                             support_flagged=int((support[sub]>0).sum()),
                             xi_min=float(np.nanmin(pars[sub,2])),xi_max=float(np.nanmax(pars[sub,2])),
                             negative_return_level_series=int(np.any(levels[sub]<0,axis=1).sum()))
                    summary.append(row)
                print(json.dumps(dict(member=member,period=[y0,y1],fits=len(samples),fit_seconds=fit_seconds,
                                      total_seconds=time.perf_counter()-begin,failed=int((status!=0).sum()),
                                      support_flagged=int((support>0).sum()),file=str(dest))),flush=True)
    with (OUT/'fit_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    report=dict(status='complete',files=8,members=['01','04','07','08'],periods=PERIODS,
                durations_hours=DUR.tolist(),return_periods_years=RP.astype(int).tolist(),sample_size=40,
                total_fits=sum(r['series'] for r in summary),successful_fits=sum(r['successful_fits'] for r in summary),
                failed_fits=sum(r['failed_fits'] for r in summary),support_flagged_fits=sum(r['support_flagged'] for r in summary),
                negative_return_level_series=sum(r['negative_return_level_series'] for r in summary),
                max_parameter_absolute_difference_vs_lmoments3_real_samples=max_reference_difference,
                reference_checked_real_series=1600,elapsed_seconds=time.perf_counter()-started,
                shape_convention='xi=-scipy_genextreme_c',method='unbiased sample L-moments')
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
