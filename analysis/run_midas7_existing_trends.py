"""User-selected seven sites: existing AMS, original year labels and fixed QC."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import theilslopes,kendalltau
from test_ams_trends_40yr import mk,reference_mk

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'MIDAS-AMS-56'/'ams_long.csv'
COHORT=ROOT/'midas_station_validation_7'/'selected_stations.csv'
OUT=ROOT/'midas_station_validation_7'/'existing_AMS_trends'

def serial_check(x,years,seed):
    slope=theilslopes(x,years).slope
    e=x-slope*(years-years.min());e-=e.mean()
    denom=np.dot(e,e)
    pairs=np.flatnonzero(np.diff(years)==1)
    result=dict(lag1_adjacent_year_pairs=len(pairs),lag1_cov_ratio=np.nan,
        lag1_residual_permutation_p=np.nan,lag1_flag=False,acf_status='too_few_pairs')
    if denom<=1e-24*max(np.dot(x,x),1.):result['acf_status']='zero_detrended_variance';return result
    if len(pairs)<5:return result
    r=np.dot(e[pairs],e[pairs+1])/denom
    rng=np.random.default_rng(seed);perms=rng.random((4999,len(e))).argsort(axis=1)
    ep=e[perms];rp=np.sum(ep[:,pairs]*ep[:,pairs+1],axis=1)/denom
    p=(1+np.count_nonzero(np.abs(rp)>=abs(r)))/5000
    result.update(lag1_cov_ratio=float(r),lag1_residual_permutation_p=float(p),lag1_flag=bool(p<.05),acf_status='exploratory_residual_permutation')
    return result

def main():
    OUT.mkdir(exist_ok=True)
    selected=pd.read_csv(COHORT,dtype={'station_id':str})
    a=pd.read_csv(SOURCE,dtype={'station_id':str})
    b=a[a.station_id.isin(selected.station_id)&a.year.between(1961,2020)&(a.hour_frac>=.95)&(a.win_frac>=.95)&(a.ams_mm>0)].copy()
    count=b.groupby(['station_id','year']).duration_h.nunique()
    keys=count[count==10].index
    b=b.set_index(['station_id','year']).loc[keys].reset_index()
    assert not b.duplicated(['station_id','year','duration_h']).any()
    b.to_csv(OUT/'input_AMS_selected_common_years.csv',index=False)
    results=[]
    for station in selected.itertuples():
        for begin,end,expected in [(1961,1990,station.valid_1961_1990),(1991,2020,station.valid_1991_2020)]:
            period=b[(b.station_id==station.station_id)&b.year.between(begin,end)]
            assert period.year.nunique()==expected
            for duration,d in period.groupby('duration_h'):
                d=d.sort_values('year');years=d.year.to_numpy();x=d.ams_mm.to_numpy()
                assert len(x)>=15
                ss,tau,z,p,var=mk(x);rs,rz,rp=reference_mk(x)
                assert ss==rs and np.allclose([z,p],[rz,rp])
                if np.ptp(x)>0:assert np.isclose(tau,kendalltau(years,x).statistic)
                slope=theilslopes(x,years,alpha=.95)
                direct=np.array([(x[j]-x[i])/(years[j]-years[i]) for i in range(len(x)-1) for j in range(i+1,len(x))])
                assert np.isclose(np.median(direct),slope.slope)
                r=dict(station_id=station.station_id,station_name=station.station_name,start_year=begin,end_year=end,
                    duration_h=int(duration),n_years=len(x),valid_years=';'.join(map(str,years)),
                    missing_years=';'.join(str(y) for y in range(begin,end+1) if y not in years),
                    mk_s=ss,mk_tau_b=tau,mk_z=z,mk_p=p,mk_direction=int(np.sign(ss)) if p<.05 else 0,
                    sen_mm_per_year=slope.slope,sen_ci95_low_iid=slope.low_slope,sen_ci95_high_iid=slope.high_slope,
                    sen_mm_per_decade=slope.slope*10,sen_pct_per_decade=slope.slope*1000/np.median(x))
                r.update(serial_check(x,years,923000+int(station.station_id)+begin))
                results.append(r)
    df=pd.DataFrame(results);df.to_csv(OUT/'station_duration_results.csv',index=False)
    summaries=[]
    for scope,durations in [('main_1_to_12h',[1,2,3,4,5,6,12]),('all_10_durations',[1,2,3,4,5,6,12,24,48,72])]:
        for begin,g in df[df.duration_h.isin(durations)].groupby('start_year'):
            summaries.append(dict(scope=scope,start_year=int(begin),end_year=int(g.end_year.iloc[0]),series=len(g),
                significant_increase=int((g.mk_direction==1).sum()),significant_decrease=int((g.mk_direction==-1).sum()),
                no_detected_trend=int((g.mk_direction==0).sum()),
                lag1_testable=int(g.lag1_residual_permutation_p.notna().sum()),lag1_exploratory_flags=int(g.lag1_flag.sum())))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    report=dict(status='complete',stations=7,series=len(df),periods=[[1961,1990],[1991,2020]],
        year_definition='Existing AMS year labels retained; no DecNov re-extraction or relabelling',
        quality_rule='Existing hour_frac>=0.95 and win_frac>=0.95 and positive AMS; ten durations share retained years',
        mk='Two-sided normal approximation, ties and continuity correction; local alpha=.05, no FDR',
        sen='Actual calendar-year differences; IID rank confidence interval',
        autocorrelation='Sen-detrended covariance ratio on adjacent calendar years only. 4999 residual permutations with missing-year pattern fixed; exploratory, conditional on estimated trend, not exact stationarity or independence proof',
        no_reconstruction_of_missing_years=True,raw_QC_discrepancies_not_resolved=True,
        whole_day_AMS_included=False,GEV_or_transfer_run=False,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),summaries=summaries)
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
