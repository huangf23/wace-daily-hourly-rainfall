"""40-year AMS: MK, Sen slope and detrended serial-dependence diagnostics.
No century-long tests, change-point tests, GEV refitting or FDR correction.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import math
import json
import csv
import time
from pathlib import Path
import numpy as np
import netCDF4 as nc
from numba import njit,prange,set_num_threads
from scipy.stats import theilslopes,kendalltau,chi2,norm
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'ukcp18_ams_trends_40yr'
NAMES=['mk_s','mk_tau_b','mk_z','mk_p','sen_slope','sen_ci95_low_iid','sen_ci95_high_iid',
       'sample_median','lag1_r','lag1_p_approx','ljung_box_q5','mk_z_tfpw','mk_p_tfpw',
       'tfpw_applied','diagnostic_status']

@njit(cache=True)
def mk(x):
    n=len(x);s=0
    for i in range(n-1):
        for j in range(i+1,n):
            s+=int(x[j]>x[i])-int(x[j]<x[i])
    ordered=np.sort(x);term=0.;tied_pairs=0.;i=0
    while i<n:
        j=i+1
        while j<n and ordered[j]==ordered[i]:j+=1
        t=j-i;term+=t*(t-1)*(2*t+5);tied_pairs+=t*(t-1)/2;i=j
    var=(n*(n-1)*(2*n+5)-term)/18
    z=0.
    if var>0:
        if s>0:z=(s-1)/math.sqrt(var)
        elif s<0:z=(s+1)/math.sqrt(var)
    pairs=n*(n-1)/2
    tau=s/math.sqrt(pairs*(pairs-tied_pairs)) if pairs>tied_pairs else 0.
    return s,tau,z,math.erfc(abs(z)/math.sqrt(2)),var

@njit(parallel=True,nogil=True,cache=True)
def kernel(samples):
    count,n=samples.shape;nt=n*(n-1)//2
    result=np.full((count,15),np.nan)
    acf=np.full((count,5),np.nan)
    for p in prange(count):
        x=samples[p]
        if not np.all(np.isfinite(x)):
            result[p,14]=1;continue
        s,tau,z,pvalue,var=mk(x)
        result[p,0]=s;result[p,1]=tau;result[p,2]=z;result[p,3]=pvalue
        slopes=np.empty(nt,np.float64);k=0
        for i in range(n-1):
            for j in range(i+1,n):slopes[k]=(x[j]-x[i])/(j-i);k+=1
        slopes.sort()
        slope=(slopes[(nt-1)//2]+slopes[nt//2])/2
        result[p,4]=slope
        sigma=math.sqrt(var);zcrit=1.959963984540054
        lo=max(int(np.round((nt-zcrit*sigma)/2))-1,0)
        hi=min(int(np.round((nt+zcrit*sigma)/2)),nt-1)
        result[p,5]=slopes[lo];result[p,6]=slopes[hi]
        sx=np.sort(x);result[p,7]=(sx[(n-1)//2]+sx[n//2])/2
        residual=np.empty(n,np.float64)
        for i in range(n):residual[i]=x[i]-slope*i
        mean=residual.mean();residual-=mean
        denominator=np.dot(residual,residual)
        result[p,11]=z;result[p,12]=pvalue;result[p,13]=0;result[p,14]=0
        if denominator<=1e-24*max(np.dot(x,x),1.):
            result[p,14]=2;continue
        q=0.
        for lag in range(1,6):
            cross=0.
            for i in range(lag,n):cross+=residual[i]*residual[i-lag]
            r=cross/denominator;acf[p,lag-1]=r
            q+=r*r/(n-lag)
        r1=acf[p,0]
        acp=math.erfc(abs(r1)*math.sqrt(n)/math.sqrt(2))
        result[p,8]=r1;result[p,9]=acp;result[p,10]=n*(n+2)*q
        # Trend-free prewhitening is an accompanying sensitivity result.
        # Only apply AR(1) removal when the detrended lag-1 diagnostic is significant.
        if acp<.05:
            whitened=np.empty(n-1,np.float64)
            for i in range(1,n):whitened[i-1]=residual[i]-r1*residual[i-1]+slope*i
            _,_,zz,pp,_=mk(whitened)
            result[p,11]=zz;result[p,12]=pp;result[p,13]=1
    return result,acf

def reference_mk(x):
    n=len(x)
    s=int(np.sign(x[None,:]-x[:,None])[np.triu_indices(n,1)].sum())
    _,counts=np.unique(x,return_counts=True)
    variance=(n*(n-1)*(2*n+5)-np.sum(counts*(counts-1)*(2*counts+5)))/18
    z=(s-np.sign(s))/np.sqrt(variance) if variance>0 else 0.
    return s,z,2*norm.sf(abs(z))

def validate(samples,results,acf,indices):
    for p in indices:
        x=samples[p];r=results[p]
        if not np.all(np.isfinite(x)):assert r[14]==1;continue
        reference=theilslopes(x,np.arange(len(x)),alpha=.95)
        assert np.allclose(r[4:7],[reference.slope,reference.low_slope,reference.high_slope],rtol=1e-10,atol=1e-10)
        ss,zz,pp=reference_mk(x)
        assert ss==r[0] and np.allclose([r[2],r[3]],[zz,pp],atol=1e-12)
        if np.ptp(x)>0:assert np.isclose(r[1],kendalltau(np.arange(len(x)),x).statistic,atol=1e-12)
        residual=x-reference.slope*np.arange(len(x));residual-=residual.mean()
        if r[14]==2:continue
        test=acorr_ljungbox(residual,lags=[5],model_df=0,return_df=True)
        assert np.isclose(r[10],test['lb_stat'].iloc[0],rtol=1e-10,atol=1e-10)
        rr=np.dot(residual[1:],residual[:-1])/np.dot(residual,residual)
        assert np.isclose(rr,r[8],atol=1e-12)
        if r[13]==1:
            white=residual[1:]-rr*residual[:-1]+reference.slope*np.arange(1,len(x))
            _,z2,p2=reference_mk(white)
            assert np.allclose(r[11:13],[z2,p2],atol=1e-10)

def tests():
    rng=np.random.default_rng(2809);t=np.arange(40)
    x=rng.normal(size=(160,40))
    x[:40]+=t*.15;x[40:80]-=t*.15;x[80:100]=np.round(x[80:100])
    for i in range(100,150):
        for j in range(1,40):x[i,j]+=.8*x[i,j-1]
    x[-3]=5;x[-2]=t*.2+1;x[-1,0]=np.nan
    result,acf=kernel(x);validate(x,result,acf,range(len(x)))
    assert result[-3,3]==1 and result[-3,4]==0
    assert result[-2,3]<.05 and result[-2,4]>.19
    print('TESTS PASSED: MK ties/continuity, Sen+CI vs SciPy, autocorrelation vs statsmodels, TFPW, constants and NaN',flush=True)

def main():
    OUT.mkdir(exist_ok=True);set_num_threads(20);tests()
    jobs=[]
    for member in ['01','04','07','08']:
        for start,end in [(1981,2020),(2041,2080)]:
            jobs.append((member,start,end,'hourly_sliding',ROOT/'ukcp18_gev_lmoments_40yr'/f'ukcp18_{member}_AMS_GEV_{start}-{end}_DecNov.nc'))
        jobs.append((member,2041,2080,'calendar_day',ROOT/'ukcp18_future_calendar_day_ams_gev'/f'ukcp18_{member}_AMS_GEV_2041-2080_calendar_day.nc'))
    started=time.perf_counter();summary=[];fit_seconds=0.;checked=0
    for member,start,end,kind,path in jobs:
        begin=time.perf_counter()
        with nc.Dataset(path) as src:
            years=np.asarray(src['year'][:]);duration=np.asarray(src['duration'][:])
            assert np.array_equal(years,np.arange(start,end+1)) and len(years)==40
            a=np.ma.filled(src['ams_depth'][:],np.nan)
            samples=np.ascontiguousarray(a.transpose(1,2,3,0).reshape(-1,40))
            t=time.perf_counter();r,acf=kernel(samples);seconds=time.perf_counter()-t;fit_seconds+=seconds
            indices=np.linspace(0,len(samples)-1,100,dtype=int)
            validate(samples,r,acf,indices);checked+=len(indices)
            lbp=chi2.sf(r[:,10],df=5)
            dest=OUT/f'trend_{kind}_{member}_{start}-{end}.nc';tmp=dest.with_suffix('.tmp.nc')
            nd=len(duration);shape=(nd,244,180)
            with nc.Dataset(tmp,'w') as ds:
                ds.title='40-year AMS: Mann-Kendall trend, Sen slope and detrended autocorrelation'
                ds.source_ams_file=str(path);ds.ensemble_member=member;ds.window_definition=kind
                ds.start_year=start;ds.end_year=end;ds.sample_length=40;ds.calendar='360_day'
                ds.statistical_year='December-November, labelled by ending year'
                ds.alpha=.05;ds.multiple_testing='None: all p-values are local, unadjusted; not field significance'
                ds.mk_method='Two-sided normal approximation; tie-corrected variance; continuity correction; tau-b saved'
                ds.sen_method='Median of all 780 pairwise slopes; mm/year; actual annual spacing'
                ds.sen_ci_note='95% Sen rank confidence interval assumes independent observations; not autocorrelation adjusted'
                ds.autocorrelation_method='Remove Sen linear trend and residual mean; ACF(k)=sum(e[t]*e[t-k])/sum(e[t]^2), lags 1..5'
                ds.lag1_p_note='Approximate two-sided normal white-noise diagnostic using SE=1/sqrt(40)'
                ds.ljung_box_note='Q=n(n+2)*sum(r[k]^2/(n-k)), lags 1..5, chi-square df=5; approximate diagnostic after estimated detrending'
                ds.tfpw_note='Sensitivity result: only when lag1_p_approx<0.05, remove residual AR(1), restore Sen trend, apply MK to 39 points; otherwise repeat original MK'
                ds.interpretation='Failure to reject trend is not proof of stationarity; no Pettitt, FDR, GEV refit or century-long tests'
                ds.history=time.strftime('%Y-%m-%dT%H:%M:%S')+' test_ams_trends_40yr.py'
                for name,size in [('duration',nd),('projection_y_coordinate',244),('projection_x_coordinate',180),('bnds',2),('year',40),('lag',5)]:ds.createDimension(name,size)
                for name in ['duration','year','projection_y_coordinate','projection_x_coordinate','projection_y_coordinate_bnds','projection_x_coordinate_bnds','latitude','longitude','transverse_mercator']:
                    v=src[name];out=ds.createVariable(name,v.dtype,v.dimensions);out[:]=v[:]
                    out.setncatts({attr:v.getncattr(attr) for attr in v.ncattrs() if attr!='_FillValue'})
                lag=ds.createVariable('lag','i4',('lag',));lag[:]=np.arange(1,6);lag.units='years'
                dims=('duration','projection_y_coordinate','projection_x_coordinate')
                fields={name:r[:,j] for j,name in enumerate(NAMES)}
                fields['ljung_box_p5']=lbp
                fields['sen_slope_mm_per_decade']=r[:,4]*10
                fields['sen_slope_pct_per_decade']=np.divide(r[:,4]*1000,r[:,7],out=np.full(len(r),np.nan),where=r[:,7]>0)
                fields['sen_period_change_pct']=np.divide(r[:,4]*(end-start)*100,r[:,7],out=np.full(len(r),np.nan),where=r[:,7]>0)
                fields['sample_size']=np.isfinite(samples).sum(axis=1)
                fields['mk_direction']=np.where(np.isfinite(r[:,3]),np.where(r[:,3]<.05,np.sign(r[:,0]),0),np.nan)
                fields['mk_direction_tfpw']=np.where(np.isfinite(r[:,12]),np.where(r[:,12]<.05,np.sign(r[:,11]),0),np.nan)
                for name,values in fields.items():
                    v=ds.createVariable(name,'f8',dims,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,61,180));v[:]=values.reshape(shape)
                    v.units='1';v.grid_mapping='transverse_mercator';v.coordinates='latitude longitude'
                    if name in ['sen_slope','sen_ci95_low_iid','sen_ci95_high_iid']:v.units='mm year-1'
                    if name=='sample_median':v.units='mm'
                    if name=='sen_slope_mm_per_decade':v.units='mm decade-1'
                    if name=='sen_slope_pct_per_decade':v.units='percent decade-1'
                    if name=='sen_period_change_pct':v.units='percent';v.description='Sen slope * 39 year endpoint separation / sample median * 100'
                    if name.startswith('mk_direction'):v.description='-1 significant decrease; 0 not significant; +1 significant increase; local alpha 0.05'
                    if name=='diagnostic_status':v.description='0 valid; 1 missing sample; 2 zero detrended residual variance (ACF undefined, MK/Sen retained)'
                v=ds.createVariable('detrended_acf','f8',('lag',)+dims,fill_value=np.nan,zlib=True,complevel=1,chunksizes=(1,1,61,180))
                v[:]=acf.T.reshape((5,)+shape)
            with nc.Dataset(tmp) as ds:
                assert np.allclose(np.ma.filled(ds['mk_p'][:],np.nan).reshape(-1),r[:,3],equal_nan=True)
                assert np.allclose(np.ma.filled(ds['sen_slope'][:],np.nan).reshape(-1),r[:,4],equal_nan=True)
                assert np.all(ds['sample_size'][:]==40)
            os.replace(tmp,dest)
            for j,d in enumerate(duration):
                sub=r[j*43920:(j+1)*43920];p5=lbp[j*43920:(j+1)*43920]
                summary.append(dict(member=member,window_definition=kind,start_year=start,end_year=end,duration_hours=int(d),series=43920,
                    valid_mk=int(np.isfinite(sub[:,3]).sum()),mk_increase_local_p05=int(((sub[:,3]<.05)&(sub[:,0]>0)).sum()),
                    mk_decrease_local_p05=int(((sub[:,3]<.05)&(sub[:,0]<0)).sum()),
                    lag1_significant_local_p05=int((sub[:,9]<.05).sum()),ljung_box_significant_local_p05=int((p5<.05).sum()),
                    tfpw_increase_local_p05=int(((sub[:,12]<.05)&(sub[:,11]>0)).sum()),
                    tfpw_decrease_local_p05=int(((sub[:,12]<.05)&(sub[:,11]<0)).sum()),
                    median_sen_mm_per_decade=float(np.nanmedian(sub[:,4]*10)),
                    residual_variance_zero=int((sub[:,14]==2).sum())))
            print(json.dumps(dict(member=member,period=[start,end],kind=kind,series=len(samples),kernel_seconds=seconds,total_seconds=time.perf_counter()-begin,file=str(dest))),flush=True)
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    report=dict(status='complete',files=12,total_series=sum(row['series'] for row in summary),combinations=len(summary),
                periods=[[1981,2020],[2041,2080]],century_tests=False,alpha_local=.05,FDR_applied=False,
                methods=['Mann-Kendall (ties and continuity correction)','Sen slope and IID 95% CI','detrended ACF lags 1..5','Ljung-Box lag 5 diagnostic','conditional trend-free prewhitening sensitivity'],
                invalid_series=sum(row['series']-row['valid_mk'] for row in summary),
                original_mk_increase_local=sum(row['mk_increase_local_p05'] for row in summary),original_mk_decrease_local=sum(row['mk_decrease_local_p05'] for row in summary),
                lag1_flagged=sum(row['lag1_significant_local_p05'] for row in summary),
                reference_checked_real_series=checked,synthetic_test_series=160,kernel_seconds=fit_seconds,elapsed_seconds=time.perf_counter()-started,
                interpretation='Counts are local unadjusted test results; overlapping periods/durations/members/grid cells are not independent replications.')
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINISHED '+json.dumps(report),flush=True)

if __name__=='__main__':main()
