"""NC, probability relations, raw-AMS scores, fit and event diagnostics."""
import argparse,time
import pandas as pd
from scipy.stats import genextreme,rankdata
from numba import set_num_threads
from gev_lmoments_kernel import fit_batch
from research_v3_common import *

DEST=OUT/'point_diagnostics'

def fit_period(a):
    ny,nd,ng=a.shape
    p,_,_,status,support=fit_batch(np.ascontiguousarray(a.transpose(1,2,0).reshape(nd*ng,ny)),np.empty(0))
    return p.reshape(nd,ng,3).transpose(0,2,1),status.reshape(nd,ng),support.reshape(nd,ng)

def predict():
    g=geometry();cells=g['cells'];n=len(cells)
    for member in MEMBERS:
        marker=DEST/f'prediction_{member}.json'
        if marker.exists():continue
        st=time.time();h,daily,sc=frozen_parameters(member,cells)
        data,audit=allowed_ams(member,cells)
        np.savez(DEST/f'allowed_ams_{member}.npz',**data)
        hm,_,_=fit_period(data['historical_hourly'][1:])
        hdm,_,_=fit_period(data['historical_daily'][1:])
        scm,_=scaling(np.stack([hdm,daily[1]]))
        pred=np.empty((4,7,len(T),n));raw=np.empty((3,7,len(T),n))
        adjusted=np.zeros((4,7,n),bool);clipped=None;boundary=None
        crossings=[];previous=None
        for j,d in enumerate(DURATIONS):
            q,good,adj,clip,bd=prediction_curves(h[j],daily[0,0],daily[1,0],sc,d)
            assert good.all()
            pred[:,j]=q[:,IDS];adjusted[:,j]=adj
            raw[:,j],_,_,bdr=raw_predict(h[j],daily[0,0],daily[1,0],sc,float(d),reduced_variate(T))
            if j==0:clipped=clip[IDS];boundary=bdr
            if previous is not None:
                for m,method in enumerate(METHODS):
                    depth_cross=(q[m]<previous[m][0]-1e-9)
                    intensity_cross=(q[m]/d>previous[m][0]/previous[m][1]+1e-9)
                    crossings.append(dict(member=member,method=method,shorter_h=int(DURATIONS[j-1]),longer_h=int(d),
                        depth_crossing_cell_fraction=float(depth_cross.any(axis=0).mean()),
                        intensity_crossing_cell_fraction=float(intensity_cross.any(axis=0).mean()),
                        probability_grid='fixed BASE_T, including engineering nodes'))
            previous=[(q[m].copy(),float(d)) for m in range(4)]
        np.savez(DEST/f'predictions_{member}.npz',pred=pred,raw=raw,h=h,daily=daily,scale=sc,
            omitted_first_h=hm,omitted_first_hd=hdm,omitted_first_scale=scm,
            adjusted=adjusted,hqt_clipped=clipped,hqt_boundary=boundary,T=T,cells=cells)
        pd.DataFrame(crossings).to_csv(DEST/f'cross_duration_{member}.csv',index=False)
        save_json(marker,dict(status='complete',member=member,future_hourly_access_count=0,
            allowed_input_audit=audit,prediction_sha256=sha(DEST/f'predictions_{member}.npz'),seconds=time.time()-st,
            omitted_first_year='Historical 1981 omitted from hourly and whole-day fits together; target daily unchanged'))
        print(f'POINT PREDICTION FROZEN {member}, {time.time()-st:.1f}s',flush=True)

def fit_diagnostics(a,p,member,role,durations):
    rows=[];ny=a.shape[0];sorted_x=np.sort(a,axis=0)
    F=genextreme.cdf(sorted_x,c=-p[:,2][None],loc=p[:,0][None],scale=p[:,1][None])
    rank=np.arange(1,ny+1)[:,None,None]/ny
    ks=np.maximum((rank-F).max(axis=0),(F-(rank-1/ny)).max(axis=0))
    support=(1+p[:,2][None]*(a-p[:,0][None])/p[:,1][None]<=0).sum(axis=0)
    for j,d in enumerate(durations):
        rows.append(dict(member=member,role=role,duration_h=int(d),n_years=ny,
            cells=a.shape[-1],support_violation_cells=int((support[j]>0).sum()),
            support_violation_observations=int(support[j].sum()),
            ks_median=float(np.median(ks[j])),ks_P90=float(np.quantile(ks[j],.9)),
            xi_P10=float(np.quantile(p[j,2],.1)),xi_median=float(np.median(p[j,2])),xi_P90=float(np.quantile(p[j,2],.9)),
            ks_role='descriptive fitted-CDF discrepancy; not an unadjusted KS hypothesis test'))
    return rows,ks,support

def events(path,durations,cells):
    with nc.Dataset(path) as ds:
        ids=[int(np.flatnonzero(ds['duration'][:]==d)[0]) for d in durations]
        ny=len(ds.dimensions['year'])
        start=np.asarray(ds['event_start'][:,ids],np.int64).reshape(ny,len(ids),-1)[:,:,cells]
        end=np.asarray(ds['event_end'][:,ids],np.int64).reshape(ny,len(ids),-1)[:,:,cells]
        years=np.asarray(ds['year'][:]);units=ds['event_end'].units;cal=ds['event_end'].calendar
        dates=nc.num2date(end.ravel()-.0001,units,calendar=cal,only_use_cftime_datetimes=True)
        assigned=np.array([x.year+(x.month==12) for x in dates]).reshape(end.shape)
    assert np.array_equal(end-start,np.broadcast_to(np.asarray(durations)[None,:,None],end.shape))
    assert np.array_equal(assigned,np.broadcast_to(years[:,None,None],assigned.shape))
    return start,end,dict(duration_checks=int(end.size),year_assignment_checks=int(end.size),units=units,calendar=cal)

def evaluate():
    geo=geometry();cells=geo['cells'];n=len(cells);allsummary=[];allfit=[];allaudit=[];dep_summary=[];qsrows=[];omitrows=[]
    for member in MEMBERS:
        marker=DEST/f'evaluation_{member}.json'
        if marker.exists():continue
        st=time.time();pmark=json.loads((DEST/f'prediction_{member}.json').read_text())
        assert pmark['future_hourly_access_count']==0 and sha(DEST/f'predictions_{member}.npz')==pmark['prediction_sha256']
        pr=np.load(DEST/f'predictions_{member}.npz');pred=pr['pred'];raw=pr['raw'];h=pr['h'];daily=pr['daily'];sc=pr['scale']
        truthp=reference_parameters(member,cells);future=reference_ams(member,cells)
        allowed=np.load(DEST/f'allowed_ams_{member}.npz')
        truth=np.stack([gev_curves(x,reduced_variate(T)) for x in truthp])
        score=np.empty((4,7,3,n));relations=np.full((3,7,len(T),n),np.nan,np.float32)
        probability=np.empty_like(pred,dtype=np.float32);qs=np.empty((4,7,3,n))
        omission=np.empty_like(score);maxerr=0.;identity_rows=[];periodscores=[]
        hday=gev_curves(daily[0,0],reduced_variate(T));fday=gev_curves(daily[1,0],reduced_variate(T))
        mapped_sf=cf_survival(fday,daily[0,0]);mapped_T=np.divide(1.,mapped_sf,out=np.full_like(mapped_sf,np.inf),where=mapped_sf>0)
        for j,d in enumerate(DURATIONS):
            score[:,j]=score_curves(pred[:,j],truth[j])
            for m in range(4):probability[m,j]=T[:,None]*cf_survival(pred[m,j],truthp[j])
            # Relations are calculated from definitions, then checked against raw error identities.
            rel0=np.log((truth[j]/fday)/(pred[0,j]/hday))
            rel1=np.full_like(rel0,np.nan)
            valid=(pr['hqt_boundary']==0)&np.isfinite(raw[1,j])&(raw[1,j]>0)
            rel1[valid]=np.log(truth[j][valid]/raw[1,j][valid])
            theory=[]
            for s in range(2):
                pars=np.stack([sc[s,0]*(d/24.)**(-sc[s,3])*d,
                    sc[s,1]*(d/24.)**(-sc[s,4])*d,sc[s,2]])
                theory.append(gev_curves(pars,reduced_variate(T)))
            rel2=np.log((truth[j]/theory[1])/(pred[0,j]/theory[0]))
            relations[:,j]=[rel0,rel1,rel2]
            for k,rel in enumerate([rel0,rel1,rel2]):
                ok=np.isfinite(rel)&np.isfinite(raw[k,j])&(raw[k,j]>0)
                difference=np.max(np.abs(rel[ok]+np.log(raw[k,j][ok]/truth[j][ok])))
                assert difference<1e-9
                identity_rows.append(dict(member=member,duration_h=int(d),method=METHODS[k+1],max_identity_error=float(difference),
                    raw_valid_nodes=int(ok.sum()),excluded_nodes=int((~ok).sum()),
                    regularized_changed_cells=int(pr['adjusted'][k+1,j].sum()),probability_clipped_nodes=int(pr['hqt_clipped'].sum()) if k==1 else 0))
            for k,period in enumerate([2,5,10]):
                index=int(np.argmin(abs(BASE_T-period)))
                qbase=prediction_curves(h[j],daily[0,0],daily[1,0],sc,d)[0][:,index]
                pp=1-1/period
                error=future[:,j][None]-qbase[:,None,:]
                loss=np.mean((pp-(error<0))*error,axis=1)/pred[0,j,0]
                qs[:,j,k]=loss
                for m,method in enumerate(METHODS):
                    qsrows.append(dict(member=member,method=method,duration_h=int(d),T=period,normalized_QS=float(loss[m].mean()),
                        role='same-period raw-AMS check; not independent out-of-sample annual validation'))
            omitted=prediction_curves(pr['omitted_first_h'][j],pr['omitted_first_hd'][0],daily[1,0],pr['omitted_first_scale'],d)[0]
            omission[:,j]=score_curves(omitted[:,IDS],truth[j])
            for m,method in enumerate(METHODS):
                allsummary.append(dict(member=member,method=method,duration_h=int(d),D_RL=float(score[m,j,0].mean()),
                    B=float(score[m,j,1].mean()),S_Q=float(score[m,j,2].mean()),median_D=float(np.median(score[m,j,0])),
                    D_P10=float(np.quantile(score[m,j,0],.1)),D_P90=float(np.quantile(score[m,j,0],.9))))
                omitrows.append(dict(member=member,method=method,duration_h=int(d),original_D=float(score[m,j,0].mean()),
                    omit_1981_D=float(omission[m,j,0].mean()),mean_change=float((omission[m,j,0]-score[m,j,0]).mean()),
                    cell_change_P10=float(np.quantile(omission[m,j,0]-score[m,j,0],.1)),cell_change_P90=float(np.quantile(omission[m,j,0]-score[m,j,0],.9))))
            for lo,hi in [(2,10),(10,25),(25,100)]:
                subset=(T>=lo)&(T<=hi)
                # Boundary nodes are explicitly interpolated in log-T for band diagnostics.
                xx=np.unique(np.r_[np.log(lo),np.log(T[subset]),np.log(hi)])
                for m,method in enumerate(METHODS):
                    y=np.log(pred[m,j]/truth[j]);vals=np.stack([np.interp(xx,np.log(T),y[:,g]) for g in range(n)],axis=1)
                    band=np.trapezoid(abs(vals),x=xx,axis=0)/(np.log(hi/lo))
                    periodscores.append(dict(member=member,method=method,duration_h=int(d),T_low=lo,T_high=hi,mean_D=float(band.mean())))
        with nc.Dataset(SRC/f'evaluation_{member}.nc') as ev:
            old=np.stack([np.asarray(ev[k][:]).reshape(3,7,-1)[:,:,cells] for k in ['D_RL','B_log_bias','S_shape_error']],axis=2)
            maxerr=float(np.max(abs(old-score[1:])))
            assert maxerr<2e-10,('Existing score mismatch',maxerr)
        fitdata=[];fitks=[];fitsupport=[]
        for role,arr,pars,durs in [('historical_hourly',allowed['historical_hourly'],h,DURATIONS),
              ('historical_daily',allowed['historical_daily'],daily[0],[24,48,72]),
              ('future_daily',allowed['future_daily'],daily[1],[24,48,72]),('future_hourly_reference',future,truthp,DURATIONS)]:
            rows,ks,sup=fit_diagnostics(arr,pars,member,role,durs);fitdata+=rows;fitks.append(ks);fitsupport.append(sup)
        allfit+=fitdata
        depend=np.empty((2,7,n));overlap=np.empty_like(depend);contained=np.empty_like(depend)
        event_audits=[];event_cases={}
        for si,period in enumerate(['historical','future']):
            hp=sources(member)['historical_hourly'] if si==0 else ROOT/'ukcp18_gev_lmoments_40yr'/f'ukcp18_{member}_AMS_GEV_2041-2080_DecNov.nc'
            dp=sources(member)['historical_daily' if si==0 else 'future_daily']
            hs,he,ha=events(hp,DURATIONS,cells);ds,de,da=events(dp,[24,48,72],cells)
            assert np.all(ds%24==0) and np.all(de%24==0)
            event_audits.append(dict(member=member,period=period,hourly=ha,daily=da,midnight_alignment=True))
            hourly=allowed['historical_hourly'] if si==0 else future
            dailyams=allowed['historical_daily' if si==0 else 'future_daily'][:,0]
            rd=rankdata(dailyams,axis=0);rd-=rd.mean(axis=0)
            for j,d in enumerate(DURATIONS):
                rh=rankdata(hourly[:,j],axis=0);rh-=rh.mean(axis=0)
                depend[si,j]=(rh*rd).sum(axis=0)/np.sqrt((rh*rh).sum(axis=0)*(rd*rd).sum(axis=0))
                overlap[si,j]=(np.minimum(he[:,j],de[:,0])>np.maximum(hs[:,j],ds[:,0])).mean(axis=0)
                contained[si,j]=((hs[:,j]>=ds[:,0])&(he[:,j]<=de[:,0])).mean(axis=0)
                dep_summary.append(dict(member=member,period=period,duration_h=int(d),rank_rho_median=float(np.median(depend[si,j])),
                    rank_rho_P10=float(np.quantile(depend[si,j],.1)),rank_rho_P90=float(np.quantile(depend[si,j],.9)),
                    overlapping_window_fraction_mean=float(overlap[si,j].mean()),contained_window_fraction_mean=float(contained[si,j].mean()),
                    interpretation='same-year association and window overlap; not independent storm attribution'))
            event_cases[period+'_start']=hs[:,:,geo['case_land_indices']]
            event_cases[period+'_end']=he[:,:,geo['case_land_indices']]
        caseams=dict(historical_hourly=allowed['historical_hourly'][:,:,geo['case_land_indices']],
                     historical_daily=allowed['historical_daily'][:,:,geo['case_land_indices']],
                     future_daily=allowed['future_daily'][:,:,geo['case_land_indices']],future_hourly=future[:,:,geo['case_land_indices']])
        np.savez(DEST/f'diagnostics_{member}.npz',scores=score,truth_parameters=truthp,truth=truth,
            log_relations=relations,A=probability,hqt_mapped_T=mapped_T,normalized_QS=qs,omit_first_scores=omission,
            fit_ks=np.concatenate(fitks),fit_support=np.concatenate(fitsupport),
            rank_correlation=depend,window_overlap=overlap,window_contained=contained,T=T,cells=cells)
        np.savez(DEST/f'case_ams_{member}.npz',**caseams,**event_cases)
        pd.DataFrame(fitdata).to_csv(DEST/f'fit_diagnostics_{member}.csv',index=False)
        pd.DataFrame(identity_rows).to_csv(DEST/f'relation_identity_{member}.csv',index=False)
        pd.DataFrame([r for r in allsummary if r['member']==member]).to_csv(DEST/f'summary_{member}.csv',index=False)
        pd.DataFrame([r for r in qsrows if r['member']==member]).to_csv(DEST/f'raw_scores_{member}.csv',index=False)
        pd.DataFrame([r for r in dep_summary if r['member']==member]).to_csv(DEST/f'dependence_{member}.csv',index=False)
        pd.DataFrame([r for r in omitrows if r['member']==member]).to_csv(DEST/f'omit_first_{member}.csv',index=False)
        pd.DataFrame(periodscores).to_csv(DEST/f'T_bands_{member}.csv',index=False)
        save_json(DEST/f'event_audit_{member}.json',event_audits)
        save_json(marker,dict(status='complete',member=member,old_score_max_difference=maxerr,land_cells=n,
            methods=METHODS,seconds=time.time()-st,probability_zero_nodes=int((probability==0).sum()),
            historical_mapping_infinite_return_period_nodes=int(np.isinf(mapped_T).sum()),
            raw_relation_identities='checked within 1e-9 on valid raw support; not independent validation'))
        print(f'POINT DIAGNOSTICS COMPLETE {member}, {time.time()-st:.1f}s',flush=True)
    for prefix in ['summary','fit_diagnostics','raw_scores','dependence','omit_first','T_bands','relation_identity','cross_duration']:
        pd.concat([pd.read_csv(DEST/f'{prefix}_{m}.csv',dtype={'member':str}) for m in MEMBERS]).to_csv(DEST/f'{prefix}_all.csv',index=False)
    save_json(DEST/'complete.json',dict(status='complete',members=MEMBERS,experiments=['E01_grid','E02_grid','E03_point_fit','E06_grid','E08_grid','E09_grid'],
        detail='All land cells; original point scores checked. Boundary audit covers stored windows/year allocation plus first-year omission sensitivity; no re-extraction.'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['predict','evaluate','all']);ap.add_argument('--threads',type=int,default=3)
    args=ap.parse_args();set_num_threads(args.threads);DEST.mkdir(parents=True,exist_ok=True)
    if args.stage in ['predict','all']:predict()
    if args.stage in ['evaluate','all']:evaluate()
