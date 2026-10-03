"""UKCP18 December-November AMS. Source files are never modified."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '20')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import time
import traceback
import queue
import threading
import numpy as np
import netCDF4 as nc
from numba import njit, prange, set_num_threads
import psutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source' / 'ukcp18_pr'
OUT = ROOT / 'ukcp18_ams_dec_nov'
DUR = np.array([1,2,3,4,5,6,12,24,48,72], dtype=np.int64)
UNITS = 'hours since 1970-01-01 00:00:00'
CAL = '360_day'
PREFETCH = None

class SequentialPrefetch:
    """Keep the source HDD reading while decoding, validating and writing outputs."""
    def __init__(self, paths):
        self.queue=queue.Queue(maxsize=24)
        def produce():
            try:
                for path in paths:self.queue.put((path,path.read_bytes()))
            except BaseException as exc:self.queue.put((None,exc))
        self.thread=threading.Thread(target=produce,daemon=True,name='sequential-disk-reader')
        self.thread.start()
    def read(self,path):
        expected,raw=self.queue.get()
        if expected is None:raise raw
        assert expected==path,(expected,path)
        return raw

@njit(parallel=True, nogil=True, cache=True)
def maxima(a, durations, halo):
    # a is pixel x hour, with NaN for missing rainfall, including missing halo.
    npix, nt = a.shape
    nd = len(durations)
    peak = np.full((nd,npix), np.nan)
    end = np.full((nd,npix), -1, dtype=np.int32)
    counts = np.zeros((nd,npix), dtype=np.int32)
    ties = np.zeros((nd,npix), dtype=np.int32)
    hours = np.zeros(npix, dtype=np.int32)
    for p in prange(npix):
        sums = np.zeros(nt+1, dtype=np.float64)
        valid = np.zeros(nt+1, dtype=np.int32)
        for t in range(nt):
            x = a[p,t]
            ok = np.isfinite(x) and x >= 0 and x < 1e19
            sums[t+1] = sums[t] + (x if ok else 0.0)
            valid[t+1] = valid[t] + int(ok)
        hours[p] = valid[nt]-valid[halo]
        for j in range(nd):
            d = durations[j]
            best = -1.0
            for e in range(halo+1,nt+1):
                s = e-d
                if s >= 0 and valid[e]-valid[s] == d:
                    counts[j,p] += 1
                    value = sums[e]-sums[s]
                    if value > best:
                        best = value
                        peak[j,p] = value
                        end[j,p] = e
                        ties[j,p] = 1
                    elif value == best:
                        ties[j,p] += 1
    return peak,end,counts,ties,hours

def compute(data, threads):
    set_num_threads(threads)
    start=time.perf_counter()
    result=maxima(np.ascontiguousarray(data.T), DUR, 71)
    return result, time.perf_counter()-start

def brute_test():
    rng=np.random.default_rng(381)
    a=rng.integers(0,20,(9,180)).astype(np.float32)/8
    a[0,:]=0; a[1,:]=np.nan; a[2,76]=np.nan; a[3,85]=-1; a[4,91]=1e20
    actual=maxima(a,DUR,71)
    for j,d in enumerate(DUR):
        for p in range(len(a)):
            candidates=[]
            for e in range(72,181):
                x=a[p,e-d:e]
                if np.all(np.isfinite(x)&(x>=0)&(x<1e19)):
                    candidates.append((float(x.sum(dtype=np.float64)),e))
            assert actual[2][j,p]==len(candidates)
            if candidates:
                best=max(x[0] for x in candidates)
                assert actual[0][j,p]==best
                assert actual[1][j,p]==next(e for x,e in candidates if x==best)
                assert actual[3][j,p]==sum(x==best for x,e in candidates)
            else: assert np.isnan(actual[0][j,p]) and actual[1][j,p]==-1
    # A 72 h maximum may start in the preceding statistical year.
    b=np.zeros((1,200),np.float32); b[0,30:102]=2
    r=maxima(b,DUR,71)
    assert r[0][-1,0]==144 and r[1][-1,0]-72==30
    print('TESTS PASSED: all durations, missing values, ties, cross-year halo',flush=True)

def manifest():
    files={}
    listing=[]
    for member in ['01','04','07','08']:
        for year in range(1980,2081):
            for month in range(1,13):
                if (year,month)<(1980,12) or (year,month)>(2080,11): continue
                name=f'pr_rcp85_land-cpm_uk_5km_{member}_1hr_{year}{month:02d}01-{year}{month:02d}30.nc'
                p=SOURCE/name
                if not p.is_file(): raise FileNotFoundError(p)
                stat=p.stat()
                files[member,year,month]=p
                listing.append(dict(member=member,year=year,month=month,file=name,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    (OUT/'input_manifest.json').write_text(json.dumps(listing,indent=2),encoding='utf-8')
    return files

def load_month(path, year, month):
    # One large sequential read, then decode from RAM; only the coordinator uses netCDF/HDF5.
    raw=path.read_bytes() if PREFETCH is None else PREFETCH.read(path)
    with nc.Dataset('in_memory.nc',memory=raw) as ds:
        v=ds['pr']; v.set_auto_mask(False)
        assert v.shape==(1,720,244,180), (path.name,v.shape)
        assert v.units=='mm/hour', (path.name,v.units)
        tv=ds['time']; assert tv.calendar==CAL and tv.units==UNITS
        start=(year-1970)*8640+(month-1)*720
        expected=start+np.arange(720)
        assert np.array_equal(tv[:],expected+.5), path.name
        bounds_name=getattr(tv,'bounds','time_bnds')
        if bounds_name in ds.variables:
            bounds=np.asarray(ds[bounds_name][:])
        else:
            assert getattr(ds,'frequency','')=='1hr' and v.cell_methods=='time: mean',path.name
            bounds=np.column_stack((np.asarray(tv[:])-.5,np.asarray(tv[:])+.5))
            with (OUT/'derived_time_bounds.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(dict(file=path.name,method='validated hourly midpoint +/- 0.5 h',calendar=CAL))+'\n')
        assert np.array_equal(bounds[:,0],expected), path.name
        assert np.array_equal(bounds[:,1],expected+1), path.name
        x=np.asarray(v[0],dtype=np.float32).reshape(720,-1)
        x[(~np.isfinite(x))|(x<0)|(x>=1e19)]=np.nan
        coords={}
        assert int(ds['ensemble_member'][0])==int(path.name.split('_')[5]),path.name
        for name in ['projection_y_coordinate','projection_x_coordinate','projection_y_coordinate_bnds','projection_x_coordinate_bnds','latitude','longitude','transverse_mercator']:
            z=ds[name]
            coords[name]=(np.asarray(z[:]),z.dimensions,{a:z.getncattr(a) for a in z.ncattrs() if a!='_FillValue'})
    return x, coords

def write_year(member, year, result, coords, compute_s, io_s):
    (peak,end,counts,ties,hours)=result
    path=OUT/f'member_{member}'/f'ams_{member}_{year}.nc'
    path.parent.mkdir(exist_ok=True)
    temp=path.with_suffix('.tmp.nc')
    first=(year-1-1970)*8640+11*720
    base=first-71
    ok=end>=0
    starts=np.where(ok,base+end-DUR[:,None],-99999999)
    ends=np.where(ok,base+end,-99999999)
    flags=np.zeros_like(counts,dtype=np.uint8)
    flags[:,hours<8640]|=1
    flags[counts<8640]|=2
    flags[counts==0]|=4
    with nc.Dataset(temp,'w',format='NETCDF4') as ds:
        ds.createDimension('duration',10); ds.createDimension('projection_y_coordinate',244); ds.createDimension('projection_x_coordinate',180); ds.createDimension('bnds',2)
        ds.title='UKCP18 annual maximum rolling rainfall, December-November statistical year'
        ds.year=year; ds.ensemble_member=member; ds.calendar=CAL
        ds.statistical_year='December 1 of year-1 through December 1 of year (exclusive)'
        ds.window_assignment='Year containing the last hourly interval start (equivalent to end minus epsilon)'
        ds.event_definition='Maximum fixed-duration rolling window, not independent storm onset'
        ds.tie_rule='Earliest window retained; exact equal maxima counted'
        ds.missing_rule='Any missing hour invalidates a window; no gap filling'
        ds.source='UKCP18 pr_rcp85_land-cpm_uk_5km, local source files listed in input_manifest.json'
        ds.history=time.strftime('%Y-%m-%dT%H:%M:%S')+' extract_ukcp18.py'
        ds.compute_seconds=compute_s; ds.read_seconds=io_s
        dv=ds.createVariable('duration','i4',('duration',)); dv[:]=DUR; dv.units='hours'
        for name,(a,dims,attrs) in coords.items():
            v=ds.createVariable(name,a.dtype,dims); v.setncatts(attrs); v[:]=a
        dims=('duration','projection_y_coordinate','projection_x_coordinate')
        for name,a,dtype,fill,units in [
            ('ams_depth',peak,'f8',np.nan,'mm'),('ams_intensity',peak/DUR[:,None],'f8',np.nan,'mm hour-1'),
            ('event_start',starts,'i8',-99999999,UNITS),('event_end',ends,'i8',-99999999,UNITS),
            ('valid_window_count',counts,'i4',-1,'1'),('tie_count',ties,'i4',-1,'1'),
            ('valid_window_fraction',counts/8640.,'f4',np.nan,'1'),('quality_flag',flags,'u1',255,'1')]:
            v=ds.createVariable(name,dtype,dims,fill_value=fill,zlib=True,complevel=1,shuffle=True,chunksizes=(1,61,180))
            v[:]=a.reshape(10,244,180); v.units=units
            if name.startswith('event_'): v.calendar=CAL
            if name.startswith('ams_'): v.grid_mapping='transverse_mercator'; v.coordinates='latitude longitude'
            if name=='quality_flag':
                v.flag_masks=np.array([1,2,4],dtype=np.uint8)
                v.flag_meanings='incomplete_year_hours incomplete_window_coverage no_valid_window'
        v=ds.createVariable('valid_hours','i4',dims[1:],zlib=True,complevel=1);v[:]=hours.reshape(244,180)
    # Check round-trip values before declaring the annual file complete.
    with nc.Dataset(temp) as ds:
        assert ds['ams_depth'].shape==(10,244,180)
        assert np.allclose(ds['ams_depth'][:].filled(np.nan).reshape(10,-1),peak,equal_nan=True)
        assert np.array_equal(ds['valid_hours'][:].reshape(-1),hours)
    os.replace(temp,path)
    return dict(member=member,year=year,file=str(path),compute_seconds=round(compute_s,3),read_seconds=round(io_s,3),
                missing_grid_hours=int((8640-hours).sum()),cells_with_missing_hours=int((hours<8640).sum()),
                no_valid_windows=int((counts==0).sum()),minimum_valid_window_count=counts.min(axis=1).tolist())

def validate_real(data,result):
    # Independent direct sums on raw hourly data: all durations and multiple cells.
    peak,end,counts,ties,hours=result
    for p in [0,213,11000,22000,43919]:
        x=data[:,p].astype(np.float64)
        for j,d in enumerate(DUR):
            windows=np.lib.stride_tricks.sliding_window_view(x,int(d))
            values=windows.sum(axis=1)
            values=values[72-d:]
            finite=np.isfinite(values)
            assert finite.sum()==counts[j,p]
            if finite.any():
                best=np.nanmax(values)
                assert np.isclose(best,peak[j,p],rtol=1e-10,atol=1e-8)
                e=end[j,p]
                assert np.isclose(x[e-d:e].sum(),peak[j,p],rtol=1e-10,atol=1e-8)

def main():
    global PREFETCH
    ap=argparse.ArgumentParser();ap.add_argument('--test-only',action='store_true');ap.add_argument('--limit',type=int,default=400)
    args=ap.parse_args(); OUT.mkdir(exist_ok=True)
    brute_test()
    files=manifest()
    sample,_=load_month(files['01',1980,12],1980,12)
    # Benchmark real rainfall, same annual time length and a representative spatial subset.
    bench=np.ascontiguousarray(np.tile(sample[:,:2048],(13,1))[:8711].T)
    timings={}
    for threads in [1,5,10,20]:
        set_num_threads(threads); t=time.perf_counter(); maxima(bench,DUR,71); timings[threads]=time.perf_counter()-t
    print('BENCHMARK seconds '+json.dumps(timings),flush=True)
    best=min(timings,key=timings.get)
    threads=best; workers=max(1,min(4,20//threads))
    real_benchmark=OUT/'full_year_benchmark.json'
    if real_benchmark.exists():
        winner=max(json.loads(real_benchmark.read_text()),key=lambda row:row['years_per_second'])
        threads=winner['threads_per_worker']; workers=winner['workers']
    config=dict(threads_per_worker=threads,workers=workers,max_pending_years=12,benchmark_seconds=timings,
                durations=DUR.tolist(),members=['01','04','07','08'],years=[1981,2080],calendar=CAL)
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('CONFIG '+json.dumps(config),flush=True)
    if args.test_only:return
    planned=[]; planned_years=0
    for member in ['01','04','07','08']:
        previous=1980
        for year in range(1981,2081):
            if planned_years>=args.limit:break
            if (OUT/f'member_{member}'/f'ams_{member}_{year}.nc').exists():continue
            if previous!=year-1:planned.append(files[member,year-1,11])
            planned.extend(files[member,yy,mm] for yy,mm in [(year-1,12)]+[(year,m) for m in range(1,12)])
            previous=year;planned_years+=1
    PREFETCH=SequentialPrefetch(planned)
    pending=[]; summaries=[]; began=time.perf_counter(); submitted=0
    logpath=OUT/'completed_years.jsonl'
    def finish(item):
        future,member,year,coords,io_s,data=item
        result,seconds=future.result()
        validate_real(data,result)
        summary=write_year(member,year,result,coords,seconds,io_s)
        summaries.append(summary)
        with logpath.open('a',encoding='utf-8') as f:f.write(json.dumps(summary)+'\n')
        state=dict(completed_this_run=len(summaries),last=summary,elapsed_seconds=time.perf_counter()-began,
                   total_annual_files=len(list(OUT.glob('member_*/*.nc'))),available_memory_gb=psutil.virtual_memory().available/1e9)
        (OUT/'progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        print(f"DONE {member}/{year} total={state['total_annual_files']}/400 io={io_s:.1f}s compute={seconds:.1f}s missing={summary['missing_grid_hours']} elapsed={state['elapsed_seconds']:.0f}s",flush=True)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for member in ['01','04','07','08']:
            tail=np.full((71,43920),np.nan,np.float32)
            tail_year=1980
            for year in range(1981,2081):
                if submitted>=args.limit:break
                target=OUT/f'member_{member}'/f'ams_{member}_{year}.nc'
                if target.exists():
                    continue
                if tail_year!=year-1:
                    # Resume: recover context only immediately before an unfinished year.
                    x,_=load_month(files[member,year-1,11],year-1,11);tail=x[-71:].copy()
                while len(pending)>=12 or (pending and psutil.virtual_memory().available<20e9):finish(pending.pop(0))
                t=time.perf_counter(); data=np.empty((8711,43920),np.float32);data[:71]=tail
                reference_coords=None
                for i,(yy,mm) in enumerate([(year-1,12)]+[(year,m) for m in range(1,12)]):
                    x,coords=load_month(files[member,yy,mm],yy,mm)
                    if reference_coords is None: reference_coords=coords
                    else:
                        for key in coords: assert np.array_equal(coords[key][0],reference_coords[key][0]),(yy,mm,key)
                    data[71+i*720:71+(i+1)*720]=x
                tail=data[-71:].copy();tail_year=year;io_s=time.perf_counter()-t
                future=pool.submit(compute,data,threads)
                pending.append((future,member,year,coords,io_s,data)); submitted+=1
                while pending and pending[0][0].done():finish(pending.pop(0))
            if submitted>=args.limit:break
        while pending:finish(pending.pop(0))
    report=dict(status='complete' if len(list(OUT.glob('member_*/*.nc')))==400 else 'partial',
                annual_files=len(list(OUT.glob('member_*/*.nc'))),elapsed_seconds=time.perf_counter()-began,
                completed_this_run=len(summaries),missing_grid_hours_this_run=sum(s['missing_grid_hours'] for s in summaries),
                first_year_note='1981 has complete annual hours, but the first d-1 windows lack pre-record context for duration d>1.')
    (OUT/'run_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINISHED '+json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
