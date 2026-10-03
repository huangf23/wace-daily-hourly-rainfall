"""Midnight-aligned AMS from the supplied station hourly tables; explicit audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'midas_station_validation_7'/'whole_day_AMS'
SOURCE=Path('F:/中国降雨/MIDAS-hourly-2025alive')
DURS=[1,2,3,4,5,6,12,24,48,72]

def worker(st):
    sid,name=st
    path=SOURCE/f'{sid}_{name}.csv.gz'
    a=pd.read_csv(path,parse_dates=['time'])
    assert not a.time.duplicated().any()
    assert (a.time==a.time.dt.floor('h')).all()
    idx=pd.date_range('1960-12-28','2020-12-31 23:00',freq='h')
    rain=a.set_index('time').rain_mm.reindex(idx).to_numpy(dtype=float)
    rain[rain<0]=np.nan
    masks=pd.read_csv(ROOT.parent/'MIDAS-AMS-56'/'mask_hours.csv',parse_dates=['time'])
    applied=[]
    for t in masks.loc[masks.station==f'{sid}_{name}','time']:
        if t in idx:
            k=idx.get_loc(t);applied.append({'time':str(t),'previous_value':float(rain[k])});rain[k]=np.nan
    cs=np.r_[0.,np.cumsum(np.nan_to_num(rain,nan=0.))]
    bad=np.r_[0,np.cumsum(~np.isfinite(rain))]
    years=idx.year.to_numpy();midnight=idx.hour==0
    rows=[];checks=[]
    old=pd.read_csv(ROOT.parent/'MIDAS-AMS-56'/'ams_long.csv',dtype={'station_id':str})
    old=old[(old.station_id==sid)&old.year.between(1961,2020)]
    old=old.set_index(['year','duration_h'])
    for dur in DURS:
        sums=cs[dur:]-cs[:-dur];valid=(bad[dur:]-bad[:-dur])==0
        # Position i denotes END of interval (i-1h,i].
        vals=np.full(len(idx),np.nan);vals[dur-1:]=np.where(valid,np.maximum(sums,0),np.nan)
        for year in range(1961,2021):
            mask=years==year;hour_frac=float(np.isfinite(rain[mask]).mean())
            v=vals[mask]
            if (year,dur) in old.index:
                prev=old.loc[(year,dur)]
                vmax=float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
                checks.append(dict(station_id=sid,year=year,duration_h=dur,old_mm=prev.ams_mm,recomputed_mm=vmax,
                    difference_mm=vmax-prev.ams_mm,old_hour_frac=prev.hour_frac,old_win_frac=prev.win_frac))
            if dur not in [24,48,72]:continue
            positions=np.flatnonzero(mask&midnight);v=vals[positions];good=np.isfinite(v)
            end=idx[positions[np.flatnonzero(good & np.isclose(v,np.nanmax(v),rtol=0,atol=1e-8))[0]]] if good.any() else pd.NaT
            maximum=float(np.nanmax(v)) if good.any() else np.nan
            # Direct summation independently verifies the chosen event.
            if good.any():
                k=idx.get_loc(end);assert np.isclose(rain[k-dur+1:k+1].sum(),maximum,atol=1e-7)
            rows.append(dict(station_id=sid,station_name=name,year=year,duration_h=dur,duration_days=dur//24,
                ams_mm=maximum,mean_intensity_mm_h=maximum/dur,event_start=end-pd.Timedelta(hours=dur),
                window_end=end,hour_frac=hour_frac,win_frac=float(good.mean()),valid_windows=int(good.sum()),
                expected_windows=len(v)))
    return rows,checks,dict(station_id=sid,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),explicit_masks=applied)

def main():
    OUT.mkdir(exist_ok=True)
    selected=pd.read_csv(ROOT/'midas_station_validation_7'/'selected_stations.csv',dtype={'station_id':str})
    with ProcessPoolExecutor(max_workers=7) as pool:
        result=list(pool.map(worker,zip(selected.station_id,selected.station_name)))
    rows=pd.DataFrame([x for r in result for x in r[0]])
    checks=pd.DataFrame([x for r in result for x in r[1]])
    common=pd.read_csv(ROOT/'midas_station_validation_7'/'existing_AMS_trends'/'input_AMS_selected_common_years.csv',dtype={'station_id':str})
    keys=common[['station_id','year']].drop_duplicates().assign(old_common_year=True)
    rows=rows.merge(keys,how='left',on=['station_id','year']);rows['old_common_year']=rows.old_common_year.fillna(False).astype(bool)
    rows['coverage_qualified']=(rows.hour_frac>=.95)&(rows.win_frac>=.95)&(rows.ams_mm>0)
    checks=checks.merge(keys,how='left',on=['station_id','year'])
    checks['matches']=np.isclose(checks.old_mm,checks.recomputed_mm,atol=1e-6,rtol=0)
    rows.to_csv(OUT/'whole_day_AMS_all_years.csv',index=False)
    checks.to_csv(OUT/'hourly_AMS_reconstruction_audit.csv',index=False)
    checks[(checks.old_common_year==True)&~checks.matches].to_csv(OUT/'retained_year_discrepancies.csv',index=False)
    good=rows[rows.old_common_year&rows.coverage_qualified].copy()
    good.to_csv(OUT/'whole_day_AMS_old_common_years.csv',index=False)
    report=dict(status='extraction_complete',stations=7,annual_rows=len(rows),retained_rows=len(good),
        retained_hourly_comparison_rows=int((checks.old_common_year==True).sum()),
        retained_hourly_mismatches=int(((checks.old_common_year==True)&~checks.matches).sum()),
        year_definition='Year of interval-end timestamp, matching existing AMS labels; Jan 1 midnight assigned to end year',
        daily_definition='24/48/72-hour windows ending at 00:00, stepping 24h; timestamp assumed hourly interval end in UTC; no DST conversion',
        missing_policy='No imputation; any missing hour invalidates window; explicit mask_hours.csv applied; suspect_hourly.csv not automatically removed',
        sources=[r[2] for r in result])
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
