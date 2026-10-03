"""Case parameter intervals and complementary station distribution/event checks."""
import pandas as pd
from scipy.stats import spearmanr
from numba import set_num_threads
from research_v3_common import *
from bootstrap_joint_v3 import fit_draws,draws

set_num_threads(2)
DEST=OUT/'supplementary';DEST.mkdir(parents=True,exist_ok=True)
geo=geometry();case_ids=geo['case_land_indices'];rows=[]
for member in MEMBERS:
    if (DEST/f'case_parameter_ci_{member}.npz').exists() and (DEST/'case_shape_change_ci.csv').exists():continue
    p=np.load(OUT/'joint_bootstrap'/member/'allowed_parameters.npy',mmap_mode='r')
    allowed=np.ascontiguousarray(p[:,:,:,case_ids])
    _,idx=draws(member);ref,_,support=fit_draws(reference_ams(member,geo['cells'][case_ids]),idx)
    pars=np.concatenate([allowed,ref],axis=0)
    ci=np.nanquantile(pars,[.025,.5,.975],axis=2)
    np.savez_compressed(DEST/f'case_parameter_ci_{member}.npz',ci=ci,case_indices=case_ids,reference_support=support)
    for k in range(3):
        for j,d in enumerate(DURATIONS):
            dif=ref[j,2,:,k]-allowed[j,2,:,k]
            band=np.quantile(dif,[.025,.5,.975])
            rows.append(dict(member=member,case_percentile=geo['cases'][k]['percentile'],duration_h=int(d),
                xi_change_low=band[0],xi_change_median=band[1],xi_change_high=band[2],replicates=500))
    print(f'CASE PARAMETER INTERVALS {member}',flush=True)
if rows:pd.DataFrame(rows).to_csv(DEST/'case_shape_change_ci.csv',index=False)

source=ROOT/'midas_station_validation_7'/'transfer_v1';whole=source.parent/'whole_day_AMS'
cohort=pd.read_csv(source/'cohort.csv',dtype={'station_id':str})
h=pd.read_csv(whole/'hourly_AMS_paired_common_years.csv',dtype={'station_id':str})
d=pd.read_csv(whole/'whole_day_AMS_paired_common_years.csv',dtype={'station_id':str})
d=d[d.duration_h==24]
dependency=[];lossrows=[]
for st in cohort.itertuples():
    for label,lo,hi in [('historical',1961,1990),('target',1991,2020)]:
        daily=d[(d.station_id==st.station_id)&d.year.between(lo,hi)]
        for dur in DURATIONS:
            hourly=h[(h.station_id==st.station_id)&h.year.between(lo,hi)&(h.duration_h==dur)]
            x=hourly.merge(daily,on=['station_id','year'],suffixes=('_hour','_day'))
            he=pd.to_datetime(x.window_end_hour);hs=he-pd.to_timedelta(int(dur),unit='h')
            ds=pd.to_datetime(x.event_start);de=pd.to_datetime(x.window_end_day)
            overlap=((he>ds)&(de>hs)).mean();inside=((hs>=ds)&(he<=de)).mean()
            dependency.append(dict(station_id=st.station_id,station_name=st.station_name,primary=st.primary,period=label,
                duration_h=int(dur),n=len(x),rank_rho=float(spearmanr(x.ams_mm_hour,x.ams_mm_day).statistic),
                window_overlap_fraction=float(overlap),hourly_window_contained_fraction=float(inside)))
late=pd.read_csv(source/'heldout_late_hourly.csv',dtype={'station_id':str})
point=np.load(OUT/'stations'/'point_predictions.npz');q=point['q']
for k,st in enumerate(cohort.itertuples()):
    for j,dur in enumerate(DURATIONS):
        y=late[(late.station_id==st.station_id)&(late.duration_h==dur)].ams_mm.to_numpy()
        for period in [2,5,10]:
            tid=int(np.argmin(abs(BASE_T-period)));p=1-1/period
            for m,method in enumerate(METHODS):
                error=y-q[m,j,tid,k];loss=np.mean((p-(error<0))*error)/q[0,j,0,k]
                lossrows.append(dict(station_id=st.station_id,station_name=st.station_name,primary=st.primary,
                    duration_h=int(dur),T=period,method=method,normalized_QS=float(loss),n=len(y)))
pd.DataFrame(dependency).to_csv(DEST/'station_rank_window_relationship.csv',index=False)
pd.DataFrame(lossrows).to_csv(DEST/'station_raw_quantile_scores.csv',index=False)
save_json(DEST/'complete.json',dict(status='complete',case_intervals='4 members x 3 fixed cases x all parameters/durations; original 500 paired draws',
    station_raw_scores='T=2/5/10, fixed historical median normalization, same-period descriptive score',
    station_dependency='Same-year Spearman association and overlapping accumulation windows; not storm identity'))
