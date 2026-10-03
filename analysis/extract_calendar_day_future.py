"""Future UKCP18 whole-calendar-day AMS and L-moment GEV, end to end."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import concurrent.futures as cf
import json
from pathlib import Path
import time
import traceback
import numpy as np
import netCDF4 as nc
from numba import njit,prange,set_num_threads
from scipy.stats import genextreme
from lmoments3 import distr
import extract_ukcp18 as old
from gev_lmoments_kernel import fit_batch,self_test

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'ukcp18_future_calendar_day_ams_gev'
DAYS=np.array([1,2,3],dtype=np.int64)
DUR=DAYS*24
RP=np.array([2,5,10,25,50,100],dtype=np.float64)
YEARS=range(2041,2081)
MEMBERS=['01','04','07','08']

@njit(parallel=True,nogil=True,cache=True)
def aggregate_days(hourly):
    nd=hourly.shape[0]//24;npix=hourly.shape[1]
    daily=np.zeros((nd,npix),np.float64)
    valid=np.zeros((nd,npix),np.int16)
    for d in prange(nd):
        for h in range(24):
            for p in range(npix):
                value=hourly[24*d+h,p]
                if np.isfinite(value) and value>=0 and value<1e19:
                    daily[d,p]+=value;valid[d,p]+=1
        for p in range(npix):
            if valid[d,p]!=24:daily[d,p]=np.nan
    return daily,valid

def aggregate_task(hourly):
    set_num_threads(5)
    daily,valid=aggregate_days(hourly)
    # Independent NumPy check on five spatially separated hourly series.
    ids=np.linspace(0,hourly.shape[1]-1,5,dtype=int)
    reference=hourly[:,ids].reshape(-1,24,5).sum(axis=1,dtype=np.float64)
    assert np.allclose(reference,daily[:,ids],rtol=1e-12,atol=1e-9,equal_nan=True)
    return daily,valid.sum(axis=0,dtype=np.int32)

def tests():
    a=np.zeros((720,3),np.float32);a[12:36,0]=1;a[3,1]=np.nan
    daily,valid=aggregate_days(a)
    assert daily[0,0]==12 and daily[1,0]==12
    assert np.isnan(daily[0,1]) and valid[0,1]==23
    fixed=old.maxima(daily.T.copy(),DAYS,0)
    hourly=old.maxima(a.T.copy(),np.array([24],dtype=np.int64),0)
    assert fixed[0][0,0]==12 and hourly[0][0,0]==24
    assert fixed[0][1,0]==24
    b=np.zeros((1,362));b[0,:3]=[5,7,11]
    r=old.maxima(b,DAYS,2)
    assert r[0][2,0]==23 and r[1][2,0]==3 and r[2][2,0]==360
    b[0,:2]=np.nan
    r=old.maxima(b,DAYS,2)
    assert np.array_equal(r[2][:,0],np.array([360,359,358]))
    print('TESTS PASSED: midnight alignment, daily vs hourly windows, missing hour, cross-year halo',flush=True)

def path_for(member,year,month):
    return old.SOURCE/f'pr_rcp85_land-cpm_uk_5km_{member}_1hr_{year}{month:02d}01-{year}{month:02d}30.nc'

def annual_path(member,year):return OUT/f'member_{member}'/f'ams_calendar_day_{member}_{year}.nc'

def coords_create(ds,coords):
    ds.createDimension('duration',3);ds.createDimension('projection_y_coordinate',244)
    ds.createDimension('projection_x_coordinate',180);ds.createDimension('bnds',2)
    v=ds.createVariable('duration','i4',('duration',));v[:]=DUR;v.units='hours'
    for name,(a,dims,attrs) in coords.items():
        v=ds.createVariable(name,a.dtype,dims);v[:]=a;v.setncatts(attrs)

def metadata(ds,member):
    ds.ensemble_member=member;ds.calendar=old.CAL
    ds.title=f'UKCP18 {min(YEARS)}-{max(YEARS)} whole-calendar-day annual maxima and L-moment GEV'
    ds.statistical_year='December 1 of year-1 through December 1 of year, exclusive'
    ds.daily_definition='00:00 to 00:00 in native 360_day model calendar; exactly 24 hourly intervals'
    ds.window_step_hours=24
    ds.window_assignment='Statistical year containing the last included calendar day'
    ds.event_definition='Fixed 1-, 2- or 3-calendar-day maximum window; not independent storm onset'
    ds.missing_rule='A missing hourly value invalidates its day and every window containing that day'
    ds.tie_rule='Earliest exactly equal maximum retained; ties counted'
    ds.context='Two preceding complete days used across each statistical-year boundary when available; 1981 has no pre-record context, giving at most 360/359/358 windows for 24/48/72 h'
    ds.history=time.strftime('%Y-%m-%dT%H:%M:%S')+' extract_calendar_day_future.py'

def save_year(member,year,data,hours,coords):
    set_num_threads(20)
    peak,end,counts,ties,valid_days=old.maxima(np.ascontiguousarray(data.T),DAYS,2)
    first=(year-1-1970)*8640+11*720;base=first-48
    ok=end>=0
    finish=np.where(ok,base+24*end,-99999999)
    start=np.where(ok,finish-DUR[:,None],-99999999)
    flags=np.zeros_like(counts,dtype=np.uint8)
    flags[:,hours<8640]|=1;flags[counts<360]|=2;flags[counts==0]|=4
    # All-day windows are a subset of the already-verified hourly sliding windows.
    oldpath=ROOT/'ukcp18_ams_dec_nov'/f'member_{member}'/f'ams_{member}_{year}.nc'
    with nc.Dataset(oldpath) as ref:
        refdepth=np.ma.filled(ref['ams_depth'][[7,8,9]],np.nan).reshape(3,-1)
        both=np.isfinite(peak)&np.isfinite(refdepth)
        assert np.all(peak[both]<=refdepth[both]+1e-7),(member,year,'fixed exceeds hourly AMS')
    assert np.all((finish-start)[ok]==np.broadcast_to(DUR[:,None],peak.shape)[ok])
    assert np.all(start[ok]%24==0) and np.all(finish[ok]%24==0)
    # Brute-force all possible daily windows for selected pixels.
    for p in [0,213,11000,22000,43919]:
        for j,d in enumerate(DAYS):
            w=np.lib.stride_tricks.sliding_window_view(data[:,p],int(d)).sum(axis=1)[3-d:]
            finite=np.isfinite(w)
            assert finite.sum()==counts[j,p]
            if finite.any():assert np.isclose(np.nanmax(w),peak[j,p],rtol=1e-11,atol=1e-8)
    dest=annual_path(member,year);dest.parent.mkdir(exist_ok=True)
    tmp=dest.with_suffix('.tmp.nc')
    with nc.Dataset(tmp,'w') as ds:
        metadata(ds,member);ds.year=year;coords_create(ds,coords)
        dims=('duration','projection_y_coordinate','projection_x_coordinate')
        for name,values,dtype,fill,units in [
            ('ams_depth',peak,'f8',np.nan,'mm'),('ams_intensity',peak/DUR[:,None],'f8',np.nan,'mm hour-1'),
            ('event_start',start,'i8',-99999999,old.UNITS),('event_end',finish,'i8',-99999999,old.UNITS),
            ('valid_window_count',counts,'i4',-1,'1'),('tie_count',ties,'i4',-1,'1'),
            ('valid_window_fraction',counts/360.,'f4',np.nan,'1'),('quality_flag',flags,'u1',255,'1')]:
            v=ds.createVariable(name,dtype,dims,fill_value=fill,zlib=True,complevel=1,chunksizes=(1,61,180))
            v[:]=values.reshape(3,244,180);v.units=units
            if name.startswith('event_'):v.calendar=old.CAL
            if name=='quality_flag':v.flag_masks=np.array([1,2,4],np.uint8);v.flag_meanings='incomplete_year_hours incomplete_window_coverage no_valid_window'
        for name,values in [('valid_hours',hours),('valid_days',valid_days)]:
            v=ds.createVariable(name,'i4',dims[1:],zlib=True,complevel=1);v[:]=values.reshape(244,180)
    with nc.Dataset(tmp) as ds:
        assert np.allclose(np.ma.filled(ds['ams_depth'][:],np.nan).reshape(3,-1),peak,equal_nan=True)
    os.replace(tmp,dest)
    return dict(member=member,year=year,missing_grid_hours=int((8640-hours).sum()),
                missing_grid_days=int((360-valid_days).sum()),minimum_window_counts=counts.min(axis=1).tolist(),
                strict_reduction_vs_hourly_count=int((peak[both]<refdepth[both]-1e-7).sum()))

def merge_and_fit(member):
    dest=OUT/f'ukcp18_{member}_AMS_GEV_{min(YEARS)}-{max(YEARS)}_calendar_day.nc';tmp=dest.with_suffix('.tmp.nc')
    paths=[annual_path(member,y) for y in YEARS]
    varying=['ams_depth','ams_intensity','event_start','event_end','valid_window_count','tie_count','valid_window_fraction','quality_flag','valid_hours','valid_days']
    data=np.empty((40,3,244,180),np.float64);missing=0
    with nc.Dataset(paths[0]) as template,nc.Dataset(tmp,'w') as ds:
        ds.setncatts({a:template.getncattr(a) for a in template.ncattrs() if a!='year'})
        ds.createDimension('year',40)
        for name,dim in template.dimensions.items():ds.createDimension(name,len(dim))
        y=ds.createVariable('year','i4',('year',));y[:]=list(YEARS)
        ds.createDimension('return_period',6)
        r=ds.createVariable('return_period','i4',('return_period',));r[:]=RP;r.units='years'
        for name,v in template.variables.items():
            dims=(('year',)+v.dimensions) if name in varying else v.dimensions
            opts={}
            if '_FillValue' in v.ncattrs():opts['fill_value']=v.getncattr('_FillValue')
            if name in varying:opts.update(zlib=True,complevel=1,chunksizes=(1,1,61,180) if len(dims)==4 else (1,61,180))
            out=ds.createVariable(name,v.dtype,dims,**opts)
            out.setncatts({a:v.getncattr(a) for a in v.ncattrs() if a!='_FillValue'})
            if name not in varying:out[:]=v[:]
        for i,path in enumerate(paths):
            with nc.Dataset(path) as annual:
                for name in varying:ds[name][i]=annual[name][:]
                data[i]=np.ma.filled(annual['ams_depth'][:],np.nan)
                missing+=int((8640-annual['valid_hours'][:]).sum())
        samples=np.ascontiguousarray(data.transpose(1,2,3,0).reshape(-1,40))
        set_num_threads(20)
        pars,mom,levels,status,support=fit_batch(samples,RP)
        good=status==0
        assert np.all(np.isfinite(levels[good])) and np.all(np.diff(levels[good],axis=1)>0)
        reference_error=0.
        for p in np.linspace(0,len(samples)-1,200,dtype=int):
            if not good[p]:continue
            ref=distr.gev.lmom_fit(samples[p]); refp=np.array([ref['loc'],ref['scale'],-ref['c']])
            assert np.allclose(pars[p],refp,rtol=3e-4,atol=3e-4)
            reference_error=max(reference_error,float(abs(pars[p]-refp).max()))
            q=genextreme.ppf(1-1/RP,c=-pars[p,2],loc=pars[p,0],scale=pars[p,1])
            assert np.allclose(levels[p],q,rtol=1e-10,atol=1e-8)
        dims=('duration','projection_y_coordinate','projection_x_coordinate')
        for name,values,units in [('gev_location',pars[:,0],'mm'),('gev_scale',pars[:,1],'mm'),('gev_shape',pars[:,2],'1'),
                                 ('sample_l1',mom[:,0],'mm'),('sample_l2',mom[:,1],'mm'),('sample_tau3',mom[:,2],'1')]:
            v=ds.createVariable(name,'f8',dims,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,61,180))
            v[:]=values.reshape(3,244,180);v.units=units
        for name,values in [('fit_status',status),('support_violation_count',support),('sample_size',np.isfinite(samples).sum(axis=1))]:
            v=ds.createVariable(name,'i2',dims,zlib=True,complevel=1);v[:]=values.reshape(3,244,180)
            if name=='fit_status':v.flag_values=np.arange(5,dtype=np.int16);v.flag_meanings='success missing_or_insufficient_sample degenerate_sample invalid_lmoments invalid_parameters'
        q=levels.reshape(3,244,180,6).transpose(3,0,1,2)
        for name,values,units in [('return_level_depth',q,'mm'),('return_level_intensity',q/DUR[None,:,None,None],'mm hour-1')]:
            v=ds.createVariable(name,'f8',('return_period',)+dims,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,61,180));v[:]=values;v.units=units
        ds.estimation='Unbiased sample L-moments; numerical GEV L-skewness equation solution; 40 years per series'
        ds.gev_shape_convention='Conventional xi; scipy.stats.genextreme c=-xi'
        ds.model_assumption='Independent stationary GEV per member, duration and grid; no cross-duration constraints'
        ds.support_note='Observed values outside fitted support flagged; estimates not silently changed'
    with nc.Dataset(tmp) as ds:
        assert ds['ams_depth'].shape==(40,3,244,180)
        assert np.array_equal(ds['duration'][:],DUR)
        assert np.allclose(np.ma.filled(ds['return_level_depth'][:],np.nan),q,equal_nan=True)
    os.replace(tmp,dest)
    return dict(member=member,file=str(dest),fits=len(samples),failed_fits=int((status!=0).sum()),
                support_flagged_fits=int((support>0).sum()),missing_grid_hours=missing,
                max_reference_parameter_difference=reference_error)

def main():
    OUT.mkdir(exist_ok=True);old.OUT=OUT
    set_num_threads(20);tests();reference=self_test()
    (OUT/'reference_validation.json').write_text(json.dumps(reference,indent=2))
    # Exact chronological read plan, with only November needed for a restart halo.
    planned=[];manifest=[]
    for member in MEMBERS:
        previous=None
        for year in YEARS:
            if annual_path(member,year).exists():continue
            if previous!=year-1 and year!=1981:planned.append((member,year-1,11,'context'))
            planned.extend((member,yy,mm,'target') for yy,mm in [(year-1,12)]+[(year,m) for m in range(1,12)])
            previous=year
    for member,yy,mm,role in planned:
        p=path_for(member,yy,mm)
        assert p.is_file(),p
        manifest.append(dict(file=p.name,role=role,bytes=p.stat().st_size))
    (OUT/'read_manifest_latest_run.json').write_text(json.dumps(manifest,indent=2))
    old.PREFETCH=old.SequentialPrefetch([path_for(m,y,mo) for m,y,mo,role in planned])
    started=time.perf_counter();summaries=[]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for member in MEMBERS:
            previous=None;tail=None
            for year in YEARS:
                if annual_path(member,year).exists():continue
                tick=time.perf_counter()
                if previous!=year-1:
                    if year==1981:
                        tail=np.full((2,43920),np.nan,np.float64)
                    else:
                        hourly,coords=old.load_month(path_for(member,year-1,11),year-1,11)
                        tail=aggregate_task(hourly[-48:])[0]
                jobs=[];refcoords=None
                for yy,mm in [(year-1,12)]+[(year,m) for m in range(1,12)]:
                    hourly,coords=old.load_month(path_for(member,yy,mm),yy,mm)
                    if refcoords is None:refcoords=coords
                    else:
                        for key in coords:assert np.array_equal(coords[key][0],refcoords[key][0]),(member,year,key)
                    jobs.append(pool.submit(aggregate_task,hourly))
                data=np.empty((362,43920),np.float64);data[:2]=tail
                hours=np.zeros(43920,np.int32)
                for j,future in enumerate(jobs):
                    daily,count=future.result();data[2+j*30:2+(j+1)*30]=daily;hours+=count
                tail=data[-2:].copy();previous=year
                summary=save_year(member,year,data,hours,coords)
                summary['seconds']=time.perf_counter()-tick
                with (OUT/'completed_years.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(summary)+'\n')
                progress=dict(status='extracting',annual_files=sum(annual_path(m,y).is_file() for m in MEMBERS for y in YEARS),target=160,
                              last=summary,elapsed_seconds=time.perf_counter()-started)
                (OUT/'progress.json').write_text(json.dumps(progress,indent=2))
                print('DONE '+json.dumps(progress),flush=True)
            summary=merge_and_fit(member);summaries.append(summary)
            print('MERGED AND FIT '+json.dumps(summary),flush=True)
    report=dict(status='complete',members=MEMBERS,years=[min(YEARS),max(YEARS)],annual_files=160,combined_files=4,
                durations_hours=DUR.tolist(),window_step_hours=24,return_periods_years=RP.tolist(),
                total_fits=sum(s['fits'] for s in summaries),failed_fits=sum(s['failed_fits'] for s in summaries),
                missing_grid_hours=sum(s['missing_grid_hours'] for s in summaries),
                support_flagged_fits=sum(s['support_flagged_fits'] for s in summaries),
                elapsed_seconds=time.perf_counter()-started,results=summaries,
                checks=['daily totals checked against raw hourly NumPy sums','all event boundaries at midnight','all daily AMS <= hourly-sliding AMS',
                        'brute-force daily window maxima for selected cells','output roundtrip','L-moment reference and SciPy return levels'])
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2))
    (OUT/'progress.json').write_text(json.dumps(report,indent=2))
    print('FINISHED '+json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());raise
