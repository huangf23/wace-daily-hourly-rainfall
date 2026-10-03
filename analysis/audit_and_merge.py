"""Audit all 400 annual files and create one analysis-ready AMS series per member."""
import csv
import json
from pathlib import Path
import os
import numpy as np
import netCDF4 as nc
from extract_ukcp18 import OUT, DUR, UNITS, CAL

def main():
    rows=[]
    merged=OUT/'series'; merged.mkdir(exist_ok=True)
    total_missing=0
    for member in ['01','04','07','08']:
        paths=[OUT/f'member_{member}'/f'ams_{member}_{year}.nc' for year in range(1981,2081)]
        assert all(p.is_file() for p in paths), f'Member {member} incomplete'
        dest=merged/f'ukcp18_{member}_AMS_1981-2080_DecNov.nc'
        temp=dest.with_suffix('.tmp.nc')
        with nc.Dataset(paths[0]) as template,nc.Dataset(temp,'w',format='NETCDF4') as target:
            target.setncatts({k:template.getncattr(k) for k in template.ncattrs() if k not in ['year','compute_seconds','read_seconds']})
            target.createDimension('year',100)
            for name,d in template.dimensions.items():target.createDimension(name,len(d))
            yr=target.createVariable('year','i4',('year',)); yr[:]=np.arange(1981,2081)
            yr.long_name='Statistical year labelled by ending year (December-November)'
            bounds=target.createVariable('statistical_year_bounds','i8',('year','bnds'))
            first=(np.arange(1981,2081)-1-1970)*8640+11*720
            bounds[:]=np.column_stack([first,first+8640]);bounds.units=UNITS;bounds.calendar=CAL
            varying=[]
            for name,v in template.variables.items():
                annual=name in ['ams_depth','ams_intensity','event_start','event_end','valid_window_count','tie_count','valid_window_fraction','quality_flag','valid_hours']
                dims=(('year',)+v.dimensions) if annual else v.dimensions
                opts={}
                if '_FillValue' in v.ncattrs():opts['fill_value']=v.getncattr('_FillValue')
                if annual:
                    opts.update(zlib=True,complevel=1,shuffle=True,chunksizes=((1,1,61,180) if len(dims)==4 else (1,61,180)))
                    varying.append(name)
                out=target.createVariable(name,v.dtype,dims,**opts)
                out.setncatts({k:v.getncattr(k) for k in v.ncattrs() if k!='_FillValue'})
                if not annual:out[:]=v[:]
            for i,path in enumerate(paths):
                year=i+1981
                with nc.Dataset(path) as ds:
                    assert ds.year==year and ds.ensemble_member==member
                    assert np.array_equal(ds['duration'][:],DUR)
                    depth=np.ma.filled(ds['ams_depth'][:],np.nan)
                    intensity=np.ma.filled(ds['ams_intensity'][:],np.nan)
                    start=np.ma.filled(ds['event_start'][:],-99999999)
                    end=np.ma.filled(ds['event_end'][:],-99999999)
                    valid=np.isfinite(depth)
                    assert np.allclose(depth/DUR[:,None,None],intensity,equal_nan=True)
                    assert np.all((end-start)[valid]==np.broadcast_to(DUR[:,None,None],depth.shape)[valid])
                    assert np.all(end[valid]>first[i]) and np.all(end[valid]<=first[i]+8640)
                    assert np.all(start[valid]>=first[i]-71)
                    assert np.all(depth[valid]>=0)
                    hours=np.asarray(ds['valid_hours'][:]);counts=np.asarray(ds['valid_window_count'][:])
                    assert np.all((hours>=0)&(hours<=8640)) and np.all((counts>=0)&(counts<=8640))
                    missing=int((8640-hours).sum());total_missing+=missing
                    if missing==0:
                        expected=8640-(DUR-1) if year==1981 else np.full(10,8640)
                        assert np.all(counts==expected[:,None,None]),(member,year,'window counts')
                    for name in varying:target[name][i]=ds[name][:]
                    row=dict(member=member,statistical_year=year,missing_grid_hours=missing,grid_cells=43920)
                    for j,d in enumerate(DUR):row[f'max_depth_{d}h_mm']=float(np.nanmax(depth[j]))
                    rows.append(row)
        with nc.Dataset(temp) as ds:
            assert ds['ams_depth'].shape==(100,10,244,180)
            assert ds['event_start'].calendar==CAL
            assert ds['year'][-1]==2080
        os.replace(temp,dest)
        print(f'AUDITED AND MERGED {member}: {dest}',flush=True)
    with (OUT/'annual_quality_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status='complete',annual_files=400,series_files=4,grid_cells=43920,durations_hours=DUR.tolist(),
                members=['01','04','07','08'],statistical_years=[1981,2080],total_missing_grid_hours=total_missing,
                annual_time_steps=8640,total_source_files=4800,
                first_year_context_note='1981: 8640 valid hourly positions; for duration d, at most 8640-d+1 complete windows because pre-record context is unavailable.',
                checks=['source time bounds and midpoints hourly continuity','monthly grid consistency','brute-force synthetic tests','direct raw-hour window sums for 5 cells x 10 durations in each annual job','annual file round-trip','all annual timing/intensity/window-count checks','100-year merged-file dimensions'])
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
