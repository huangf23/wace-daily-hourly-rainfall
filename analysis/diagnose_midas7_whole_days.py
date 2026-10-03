"""Diagnostics on extracted daily AMS, retaining explicit sample limitations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from run_midas7_existing_trends import serial_check, mk, reference_mk

ROOT=Path(__file__).resolve().parent/'midas_station_validation_7'
OUT=ROOT/'whole_day_AMS'
a=pd.read_csv(OUT/'whole_day_AMS_all_years.csv',dtype={'station_id':str})
qualified=a[a.old_common_year&a.coverage_qualified]
count=qualified.groupby(['station_id','year']).duration_h.nunique()
keys=count[count==3].index
b=qualified.set_index(['station_id','year']).loc[keys].reset_index()
b.to_csv(OUT/'whole_day_AMS_paired_common_years.csv',index=False)
old=pd.read_csv(ROOT/'existing_AMS_trends'/'input_AMS_selected_common_years.csv',dtype={'station_id':str})
old=old.set_index(['station_id','year']).loc[keys].reset_index()
old.to_csv(OUT/'hourly_AMS_paired_common_years.csv',index=False)
result=[]
for (sid,name),station in b.groupby(['station_id','station_name']):
    for start,end in [(1961,1990),(1991,2020)]:
        for dur,d in station[station.year.between(start,end)].groupby('duration_h'):
            d=d.sort_values('year');x=d.ams_mm.to_numpy();years=d.year.to_numpy()
            s,tau,z,p,var=mk(x);rs,rz,rp=reference_mk(x)
            assert s==rs and np.allclose([z,p],[rz,rp])
            slope=theilslopes(x,years)
            r=dict(station_id=sid,station_name=name,start_year=start,end_year=end,duration_h=int(dur),
                n_years=len(x),meets_15_year_minimum=len(x)>=15,valid_years=';'.join(map(str,years)),
                mk_p=p,mk_tau_b=tau,mk_direction=int(np.sign(s)) if p<.05 else 0,
                sen_mm_per_decade=10*slope.slope,sen_ci95_low_iid=10*slope.low_slope,
                sen_ci95_high_iid=10*slope.high_slope)
            r.update(serial_check(x,years,923000+int(sid)+start));result.append(r)
df=pd.DataFrame(result);df.to_csv(OUT/'whole_day_trend_results.csv',index=False)
previous=pd.read_csv(ROOT/'existing_AMS_trends'/'station_duration_results.csv',dtype={'station_id':str})
previous['series_type']='hourly_sliding_original_cohort'
df['series_type']='midnight_aligned_coverage_checked'
all_tests=pd.concat([previous,df],ignore_index=True)
sig=all_tests[all_tests.mk_p<.05]
sig.to_csv(OUT/'significant_trends_hourly_and_daily.csv',index=False)
summary=[]
for (sid,name),g in all_tests.groupby(['station_id','station_name']):
    row=dict(station_id=sid,station_name=name)
    for tag in ['hourly_sliding_original_cohort','midnight_aligned_coverage_checked']:
        for start in [1961,1991]:
            h=g[(g.series_type==tag)&(g.start_year==start)]
            row[f'{tag}_{start}_significant_durations']=';'.join(str(int(x)) for x in h.loc[h.mk_p<.05,'duration_h'])
    row['any_nominal_trend']=bool((g.mk_p<.05).any())
    row['both_periods_at_least_15_years']=bool((g[g.series_type=='midnight_aligned_coverage_checked'].n_years>=15).all())
    summary.append(row)
pd.DataFrame(summary).to_csv(OUT/'station_trend_concentration.csv',index=False)
report=json.loads((OUT/'FINAL_REPORT.json').read_text(encoding='utf-8'))
report.update(daily_trend_series=len(df),daily_significant=df[df.mk_p<.05][['station_name','start_year','duration_h','n_years','mk_p','sen_mm_per_decade']].to_dict('records'),
    daily_lag1_flags=int(df.lag1_flag.sum()),paired_station_years=len(keys),paired_daily_rows=len(b),paired_hourly_rows=len(old),
    coverage_exclusions=a[a.old_common_year&~a.coverage_qualified][['station_id','year','duration_h','hour_frac','win_frac']].to_dict('records'),
    sample_warning='Boulmer 1975 is only ~25% complete despite qualification in old AMS; exclude this year. Boulmer early period now 14 years, below original 15-year minimum. Retained as flagged exploratory station; no station silently deleted.',
    trend_rule='Two-sided MK alpha=.05 nominal, no FDR; Sen uses actual years; conditional residual permutation autocorrelation as in prior diagnostics; non-significance is not proof of stationarity',
    trend_selection_warning='Excluding stations because of same-data trend significance is exploratory sensitivity, not unbiased stationarity screening')
(OUT/'FINAL_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(df[df.mk_p<.05][['station_name','start_year','duration_h','n_years','mk_p','sen_mm_per_decade']].to_string(index=False))
print(pd.DataFrame(summary).to_string(index=False))
