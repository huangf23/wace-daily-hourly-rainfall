"""Complete historical whole-day screening; audit trend effects without changing main fits."""
import os,time,json
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.stats import chi2
from numba import set_num_threads
import test_ams_trends_40yr as trend
from research_v3_common import *
from compute_point_diagnostics_v3 import fit_period
from fast_scores_v3 import fast_scores

DEST=ROOT/'research_wace'/'trend_sensitivity';DEST.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'ukcp18_ams_trends_40yr'
THREADS=min(20,os.cpu_count() or 1)

def historical_daily():
    for member in MEMBERS:
        dest=DEST/f'trend_calendar_day_{member}_1981-2020.nc'
        if dest.exists():continue
        path=sources(member)['historical_daily'];started=time.time()
        with nc.Dataset(path) as src:
            a=np.asarray(src['ams_depth'][:],float);dur=np.asarray(src['duration'][:]);years=np.asarray(src['year'][:])
            assert a.shape==(40,3,244,180) and np.array_equal(years,np.arange(1981,2021))
            samples=np.ascontiguousarray(a.transpose(1,2,3,0).reshape(-1,40));assert np.isfinite(samples).all()
            r,acf=trend.kernel(samples);trend.validate(samples,r,acf,np.linspace(0,len(r)-1,100,dtype=int))
            fields={name:r[:,j] for j,name in enumerate(trend.NAMES)}
            fields.update(ljung_box_p5=chi2.sf(r[:,10],5),sen_slope_mm_per_decade=r[:,4]*10,
                sen_slope_pct_per_decade=r[:,4]*1000/r[:,7],sen_period_change_pct=r[:,4]*3900/r[:,7])
            with nc.Dataset(dest.with_suffix('.tmp.nc'),'w') as out:
                out.title='Historical midnight-aligned AMS trend screening; MK, Sen and detrended serial dependence'
                out.source_ams_file=str(path);out.member=member;out.period='1981-2020, December-November, 360_day'
                out.sample_size=40;out.alpha=.05;out.multiple_testing='Local unadjusted diagnostics; not field significance'
                out.methods='Same kernel as previous screening: ties and continuity-corrected MK, Sen, detrended lag1 and Ljung-Box lag5; conditional TFPW sensitivity'
                out.sen_ci_note='IID interval; not corrected for autocorrelation'
                out.created='2026-10-01'
                for name in ['duration','projection_y_coordinate','projection_x_coordinate','year','bnds']:
                    out.createDimension(name,len(src.dimensions[name]))
                out.createDimension('lag',5)
                for name in ['duration','year','latitude','longitude','projection_y_coordinate','projection_x_coordinate','transverse_mercator']:
                    v=src[name];w=out.createVariable(name,v.dtype,v.dimensions);w[:]=v[:];w.setncatts({k:v.getncattr(k) for k in v.ncattrs() if k!='_FillValue'})
                dims=('duration','projection_y_coordinate','projection_x_coordinate')
                for name,val in fields.items():
                    v=out.createVariable(name,'f8',dims,zlib=True,complevel=1,fill_value=np.nan);v[:]=val.reshape(3,244,180)
                v=out.createVariable('detrended_acf','f8',('lag',)+dims,zlib=True,complevel=1);v[:]=acf.T.reshape(5,3,244,180)
            os.replace(dest.with_suffix('.tmp.nc'),dest)
        print(f'HISTORICAL DAILY SCREEN {member}: {len(samples)} series, {time.time()-started:.1f}s',flush=True)

def read_screen(member,role,cells):
    period='2041-2080' if role.startswith('future') else '1981-2020'
    kind='hourly_sliding' if role.endswith('hourly') else 'calendar_day'
    folder=DEST if role=='historical_daily' else OLD
    with nc.Dataset(folder/f'trend_{kind}_{member}_{period}.nc') as ds:
        durations=np.asarray(ds['duration'][:]);wanted=DURATIONS if role.endswith('hourly') else [24,48,72]
        ids=[int(np.flatnonzero(durations==x)[0]) for x in wanted]
        fields={name:np.asarray(ds[name][ids],float).reshape(len(ids),-1)[:,cells] for name in
            ['mk_p','mk_s','mk_p_tfpw','mk_z_tfpw','sen_slope','sample_median','lag1_r','lag1_p_approx','ljung_box_p5','sen_period_change_pct']}
    return fields

def adjust(a,slope):
    centered=np.arange(a.shape[0])-(a.shape[0]-1)/2
    b=a-centered[:,None,None]*slope[None]
    assert np.allclose(a.mean(axis=0),b.mean(axis=0),rtol=1e-12,atol=1e-10)
    return np.ascontiguousarray(b)

def freeze(cells):
    for member in MEMBERS:
        marker=DEST/f'prediction_{member}.json'
        if marker.exists():continue
        data,audit=allowed_ams(member,cells);pars=[];counts={};statuses={}
        for role in ['historical_hourly','historical_daily','future_daily']:
            screen=read_screen(member,role,cells);a=adjust(data[role],screen['sen_slope'])
            p,st,support=fit_period(a);pars.append(p)
            counts[role]=dict(nonpositive_values=int((a<=0).sum()),nonpositive_series=int((a<=0).any(axis=0).sum()),fit_status_nonzero=int((st!=0).sum()),support_flag_series=int((support>0).sum()))
            statuses[role]=st
        h,hd,fd=pars;sc,_=scaling(np.stack([hd,fd]))
        dest=DEST/f'frozen_prediction_parameters_{member}.npz'
        np.savez_compressed(dest,h=h,daily=np.stack([hd,fd]),scale=sc,cells=cells,**{k+'_status':v for k,v in statuses.items()})
        save_json(marker,dict(status='complete',member=member,future_hourly_access_count=0,allowed_inputs=audit,parameter_sha256=sha(dest),counts=counts))
        print(f'TREND-ADJUSTED INPUT PARAMETERS FROZEN {member}: {counts}',flush=True)

def summary_and_evaluate(cells):
    trendrows=[];score_rows=[];value_rows=[];group_rows=[]
    old_scores=np.load(ROOT/'manuscript_elsevier'/'figures_v3'/'source_data.npz')['scores']
    for mi,member in enumerate(MEMBERS):
        screens={role:read_screen(member,role,cells) for role in ['historical_hourly','historical_daily','future_daily','future_hourly']}
        for role,screen in screens.items():
            ds=DURATIONS if role.endswith('hourly') else [24,48,72]
            for j,duration in enumerate(ds):
                p=screen['mk_p'][j];s=screen['mk_s'][j];b=screen['sen_slope'][j];med=screen['sample_median'][j];rel=1000*b/med;span=39*b*100/med
                trendrows.append(dict(member=member,role=role,duration_h=int(duration),series=len(cells),
                    mk_increase_pct=float(100*np.mean((p<.05)&(s>0))),mk_decrease_pct=float(100*np.mean((p<.05)&(s<0))),
                    tfpw_increase_pct=float(100*np.mean((screen['mk_p_tfpw'][j]<.05)&(screen['mk_z_tfpw'][j]>0))),tfpw_decrease_pct=float(100*np.mean((screen['mk_p_tfpw'][j]<.05)&(screen['mk_z_tfpw'][j]<0))),
                    lag1_flag_pct=float(100*np.mean(screen['lag1_p_approx'][j]<.05)),ljung_box_flag_pct=float(100*np.mean(screen['ljung_box_p5'][j]<.05)),
                    sen_mm_decade_median=float(np.median(b*10)),sen_pct_decade_p10=float(np.quantile(rel,.1)),sen_pct_decade_median=float(np.median(rel)),sen_pct_decade_p90=float(np.quantile(rel,.9)),
                    absolute_39yr_change_pct_median=float(np.median(abs(span))),absolute_39yr_change_pct_p90=float(np.quantile(abs(span),.9)),
                    large_slope_20pct_over_39yr_pct=float(100*np.mean(abs(span)>20))))
        file=DEST/f'frozen_prediction_parameters_{member}.npz';mark=json.loads((DEST/f'prediction_{member}.json').read_text());assert sha(file)==mark['parameter_sha256'] and mark['future_hourly_access_count']==0
        f=np.load(file);h=f['h'];daily=f['daily'];sc=f['scale']
        future=reference_ams(member,cells);refadj=adjust(future,screens['future_hourly']['sen_slope']);ref,refst,refsp=fit_period(refadj)
        orig_h,orig_daily,orig_sc=frozen_parameters(member,cells);orig_ref=reference_parameters(member,cells)
        score=np.full((4,7,3,len(cells)),np.nan);valid=np.zeros((7,len(cells)),bool)
        inputflags=np.zeros_like(valid);refflags=screens['future_hourly']['mk_p']<.05
        with np.load(OUT/'point_diagnostics'/f'allowed_ams_{member}.npz') as allowed:
            negative={role:(adjust(allowed[role],screens[role]['sen_slope'])<=0).any(axis=0)
                for role in ['historical_hourly','historical_daily','future_daily']}
        for j,duration in enumerate(DURATIONS):
            score[:,j],valid[j]=fast_scores(h[j],daily[0,0],daily[1,0],sc,ref[j],float(duration))
            # Reject physically invalid transformed AMS series explicitly; no clipping or replacement.
            negative_input=negative['historical_hourly'][j]|negative['historical_daily'].any(axis=0)|negative['future_daily'].any(axis=0)
            valid[j]&=~negative_input & ~(refadj[:,j]<=0).any(axis=0)
            score_j=score[:,j]
            score_j[:,:,~valid[j]]=np.nan
            inputflags[j]=(screens['historical_hourly']['mk_p'][j]<.05)|(screens['historical_daily']['mk_p']<.05).any(axis=0)|(screens['future_daily']['mk_p']<.05).any(axis=0)
            for group,mask in [('all',np.ones(len(cells),bool)),('no_predictor_mk_flag',~inputflags[j]),('any_predictor_mk_flag',inputflags[j]),('target_hourly_mk_flag',refflags[j]),('no_target_hourly_mk_flag',~refflags[j])]:
                mask=mask&valid[j]
                for m,method in enumerate(METHODS):
                    group_rows.append(dict(member=member,duration_h=int(duration),group=group,method=method,n=int(mask.sum()),original_D=float(old_scores[mi,m,j,0,mask].mean()),adjusted_D=float(score[m,j,0,mask].mean())))
            for m,method in enumerate(METHODS):
                ok=valid[j]
                score_rows.append(dict(member=member,duration_h=int(duration),method=method,n=int(ok.sum()),original_D=float(old_scores[mi,m,j,0,ok].mean()),adjusted_D=float(score[m,j,0,ok].mean()),original_B=float(old_scores[mi,m,j,1,ok].mean()),adjusted_B=float(score[m,j,1,ok].mean())))
            # Selected engineering endpoints from the same regularized curve engine.
            qp=prediction_curves(h[j],daily[0,0],daily[1,0],sc,duration)[0]
            qo=prediction_curves(orig_h[j],orig_daily[0,0],orig_daily[1,0],orig_sc,duration)[0]
            # Independent verification of fused scoring on the complete current duration.
            qref=gev_curves(ref[j],reduced_variate(T));check=score_curves(qp[:,IDS],qref)
            assert np.allclose(check[:,:,valid[j]],score_j[:,:,valid[j]],rtol=1e-10,atol=1e-10,equal_nan=True)
            for rp in [2,10,100]:
                ti=int(np.argmin(abs(BASE_T-rp)));assert np.isclose(BASE_T[ti],rp)
                ra=gev_curves(ref[j],reduced_variate([rp]))[0];ro=gev_curves(orig_ref[j],reduced_variate([rp]))[0]
                for k,label in enumerate(METHODS+['Reference']):
                    a,b=(qp[k,ti],qo[k,ti]) if k<4 else (ra,ro)
                    delta=100*(a[valid[j]]/b[valid[j]]-1)
                    value_rows.append(dict(member=member,duration_h=int(duration),return_period=rp,method=label,n=len(delta),median_pct=float(np.median(delta)),p10_pct=float(np.quantile(delta,.1)),p90_pct=float(np.quantile(delta,.9)),median_abs_pct=float(np.median(abs(delta))),p90_abs_pct=float(np.quantile(abs(delta),.9)),more_than_10pct_change_pct=float(100*np.mean(abs(delta)>10))))
        np.savez_compressed(DEST/f'evaluation_{member}.npz',scores=score,valid=valid,input_mk_flags=inputflags,reference_mk_flags=refflags,reference_parameters=ref,reference_status=refst,reference_support=refsp)
        print(f'EVALUATED TREND SENSITIVITY {member}; valid={valid.sum()}/{valid.size}',flush=True)
    for name,rows in [('trend_summary',trendrows),('scores',score_rows),('quantile_changes',value_rows),('score_groups',group_rows)]:pd.DataFrame(rows).to_csv(DEST/(name+'.csv'),index=False)
    overall=pd.DataFrame(score_rows).groupby('method',sort=False)[['original_D','adjusted_D']].mean();overall.to_csv(DEST/'overall_scores.csv')
    totals=pd.DataFrame(trendrows).groupby('role').mean(numeric_only=True);totals.to_csv(DEST/'trend_totals.csv')
    save_json(DEST/'complete.json',dict(status='complete',members=MEMBERS,land_cells=len(cells),historical_daily_full_grid_series=4*3*43920,historical_daily_land_series=4*3*len(cells),threads=THREADS,
        original_mean_D=overall.original_D.to_dict(),trend_adjusted_mean_D=overall.adjusted_D.to_dict(),
        interpretation='Location-trend removal sensitivity, not a fitted full nonstationary GEV. Each period is centered separately; period means preserved. No inference of stationarity from non-rejection. No new interval estimate.',
        exclusions='Any nonpositive adjusted required AMS or invalid curve removed only from this sensitivity; original result compared on the same common valid set. Original main cohort unchanged.'))

if __name__=='__main__':
    set_num_threads(THREADS)
    save_json(DEST/'protocol.json',dict(date='2026-10-01',status='specified_before_additional_computations',scope='Requested completion and sensitivity; post-hoc relative to original study',
        trend_tests='Unchanged local MK/Sen/detrended ACF/Ljung-Box and conditional TFPW; no century test or FDR',
        transform='X_adjusted(y)=X(y)-SenSlope*(y-mean(y)), separately in each period and marginal; applied to all series, not gated by significance',
        input_policy='Historical hourly + historical daily + future daily fitted and frozen for all members before future-hourly evaluation',
        endpoint_interpretation='Trend removal at period midpoint is a location-only sensitivity. Original stationary fit remains period-distribution summary; cannot infer a particular year risk or no effect of other nonstationarity.',
        grouping='All; any/no MK flag among the 7 permitted predictor series for each hourly target; target-hourly MK flags for evaluation-only diagnostics; no deletion from main analysis',
        score='Same original D_RL over log-return periods 2-100; paired common-valid comparison; quantile changes at 2/10/100 years'))
    trend.tests();historical_daily();cells=geometry()['cells'];freeze(cells);summary_and_evaluate(cells)
