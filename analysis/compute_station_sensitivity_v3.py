"""Station NC baseline and calendar-block/support sensitivity, prediction frozen first."""
import argparse,time
import pandas as pd
from numba import set_num_threads
from gev_lmoments_kernel import fit_batch
from research_v3_common import *
from fast_scores_v3 import fast_scores

SOURCE=ROOT/'midas_station_validation_7'/'transfer_v1'
DEST=OUT/'stations'
B=2000

def calendar_draws(seed,length):
    rng=np.random.default_rng(seed)
    start=rng.integers(0,30,(B,int(np.ceil(30/length))))
    return ((start[:,:,None]+np.arange(length))%30).reshape(B,-1)[:,:30]

def refit(frame,sid,durations,start,index):
    p=frame[frame.station_id==sid].pivot(index='year',columns='duration_h',values='ams_mm')
    arr=p.reindex(np.arange(start,start+30))[list(durations)].to_numpy()
    present=np.isfinite(arr).all(axis=1)
    assert np.all(np.isfinite(arr).any(axis=1)==present)
    sampled=arr[index];mask=present[index];n=mask.sum(axis=1)
    pars=np.full((len(durations),3,len(index)),np.nan)
    status=np.full((len(durations),len(index)),1,np.int16);support=np.zeros_like(status)
    for count in np.unique(n):
        if count<3:continue
        select=np.flatnonzero(n==count)
        x=np.stack([sampled[r,mask[r]].T for r in select])
        pp,_,_,ss,sp=fit_batch(np.ascontiguousarray(x.reshape(-1,count)),np.empty(0))
        pars[:,:,select]=pp.reshape(len(select),len(durations),3).transpose(1,2,0)
        status[:,select]=ss.reshape(len(select),len(durations)).T
        support[:,select]=sp.reshape(len(select),len(durations)).T
    return pars,n,status,support

def freeze():
    cohort=pd.read_csv(SOURCE/'cohort.csv',dtype={'station_id':str})
    old=np.load(SOURCE/'frozen_predictions.npz')
    h=old['early_hourly'][:7];daily=old['daily'];sc=old['tps_coefficients']
    q=np.stack([prediction_curves(h[j],daily[0,0],daily[1,0],sc,d)[0] for j,d in enumerate(DURATIONS)],axis=1)
    assert np.allclose(q[1:,:,IDS],old['curves401'][:,:,::2],rtol=1e-12,atol=1e-12)
    np.savez(DEST/'point_predictions.npz',q=q,h=h,daily=daily,sc=sc,T=BASE_T)
    save_json(DEST/'point_prediction_complete.json',dict(status='complete',future_hourly_access_count=0,sha256=sha(DEST/'point_predictions.npz')))
    data={r:pd.read_csv(SOURCE/f'{r}.csv',dtype={'station_id':str}) for r in ['early_hourly','early_daily','late_daily']}
    for length in [1,3,5]:
        folder=DEST/f'block_{length}';folder.mkdir(exist_ok=True)
        if (folder/'prediction_complete.json').exists():continue
        ih=calendar_draws(930061,length);iff=calendar_draws(930091,length)
        if length==3:
            legacy=np.load(SOURCE/'uncertainty_2000'/'calendar_draws.npz')
            assert np.array_equal(ih,legacy['early_indices']) and np.array_equal(iff,legacy['late_indices'])
        np.savez(folder/'draws.npz',historical=ih,future=iff)
        hashes={}
        for st in cohort.itertuples():
            if length==3:
                f=np.load(SOURCE/'uncertainty_2000'/f'frozen_parameters_{st.station_id}.npz')
                values={k:f[k] for k in f.files}
            else:
                h,nh,hs,hsp=refit(data['early_hourly'],st.station_id,DURATIONS,1961,ih)
                hd,nd,hds,hdsp=refit(data['early_daily'],st.station_id,[24,48,72],1961,ih)
                fd,nf,fds,fdsp=refit(data['late_daily'],st.station_id,[24,48,72],1991,iff)
                assert np.array_equal(nh,nd)
                sc,_=scaling(np.stack([hd,fd]))
                values=dict(h=h,hd=hd,fd=fd,sc=sc,nh=nh,nf=nf,hs=hs,hds=hds,fds=fds,hsp=hsp,hdsp=hdsp,fdsp=fdsp)
            path=folder/f'parameters_{st.station_id}.npz';np.savez_compressed(path,**values);hashes[st.station_id]=sha(path)
        save_json(folder/'prediction_complete.json',dict(status='complete',block_length=length,replicates=B,future_hourly_access_count=0,hashes=hashes))
        print(f'STATION PREDICTIONS FROZEN block={length}',flush=True)

def evaluate():
    cohort=pd.read_csv(SOURCE/'cohort.csv',dtype={'station_id':str});primary=cohort.primary.to_numpy()
    data=pd.read_csv(SOURCE/'heldout_late_hourly.csv',dtype={'station_id':str})
    truthp=np.load(SOURCE/'evaluation_reference.npz')['parameters'][:7]
    point=np.load(DEST/'point_predictions.npz');q=point['q']
    pt=np.stack([score_curves(q[:,j,IDS],gev_curves(truthp[j],BASE_A)[IDS]) for j in range(7)],axis=1)
    # station,duration,method,metric
    pts=pt.transpose(3,1,0,2)
    pd.DataFrame([dict(station_id=st.station_id,station_name=st.station_name,primary=st.primary,
        duration_h=int(d),method=method,D_RL=pts[k,j,m,0],B=pts[k,j,m,1],S_Q=pts[k,j,m,2])
        for k,st in enumerate(cohort.itertuples()) for j,d in enumerate(DURATIONS) for m,method in enumerate(METHODS)]).to_csv(DEST/'point_scores.csv',index=False)
    prob=np.stack([T[None,:,None]*cf_survival(q[:,j,IDS],truthp[j]) for j in range(7)],axis=1)
    np.savez(DEST/'point_evaluation.npz',scores=pts,truthp=truthp,A=prob)
    aggregates=[];deltas=[];individual=[];diagnostics=[]
    for length in [1,3,5]:
        folder=DEST/f'block_{length}';marker=json.loads((folder/'prediction_complete.json').read_text())
        assert marker['future_hourly_access_count']==0
        iff=np.load(folder/'draws.npz')['future']
        scores=np.full((7,7,4,3,B),np.nan);support_ok=np.zeros((7,7,B),bool)
        pred_band=np.empty((3,7,4,len(T),7));truth_band=np.empty((3,7,len(T),7));counts=[]
        for k,st in enumerate(cohort.itertuples()):
            fpath=folder/f'parameters_{st.station_id}.npz';assert sha(fpath)==marker['hashes'][st.station_id]
            f=np.load(fpath);truth,nn,status,support=refit(data,st.station_id,DURATIONS,1991,iff)
            assert np.array_equal(nn,f['nf']);counts.append(np.stack([f['nh'],f['nf']]))
            for j,d in enumerate(DURATIONS):
                ss,valid=fast_scores(f['h'][j],f['hd'][0],f['fd'][0],f['sc'],truth[j],float(d))
                clean=(f['hsp'][j]==0)&(f['hdsp']==0).all(axis=0)&(f['fdsp']==0).all(axis=0)&(support[j]==0)
                scores[k,j]=ss;support_ok[k,j]=valid&clean
                if length==3:
                    curves=prediction_curves(f['h'][j],f['hd'][0],f['fd'][0],f['sc'],d)[0]
                    tr=gev_curves(truth[j],BASE_A)
                    pred_band[:,j,:,:,k]=np.nanquantile(curves[:,IDS],[.025,.5,.975],axis=2)
                    truth_band[:,j,:,k]=np.nanquantile(tr[IDS],[.025,.5,.975],axis=1)
                diagnostics.append(dict(block_length=length,station_id=st.station_id,station_name=st.station_name,
                    duration_h=int(d),valid=int(valid.sum()),support_clean_valid=int((valid&clean).sum()),
                    historical_n_min=int(f['nh'].min()),future_n_min=int(f['nf'].min()),
                    support_violation_replicates=int((~clean).sum()),reference_fit_failures=int((status[j]!=0).sum())))
            print(f'STATION EVALUATED block={length} {st.station_id}',flush=True)
        if length==3:
            original=np.load(SOURCE/'uncertainty_2000'/'score_replicates.npz')['scores']
            assert np.array_equal(np.isnan(original),np.isnan(scores[:,:,1:]))
            diff=float(np.nanmax(abs(original-scores[:,:,1:])));assert diff<2e-12,diff
            np.savez_compressed(DEST/'bands_block3.npz',prediction=pred_band,reference=truth_band,T=T)
            save_json(DEST/'legacy_verification.json',dict(status='passed',maximum_score_difference=diff,
                same_missing_mask=True,same_year_draws=True,replicates=B))
        np.savez_compressed(folder/'evaluation.npz',scores=scores,support_ok=support_ok,sample_counts=np.stack(counts))
        masks=[('primary_6',primary),('all_7_exploratory',np.ones(7,bool)),
               ('no_nominal_trend_3_exploratory',cohort.station_id.isin(['00708','01145','01198']).to_numpy())]
        for group,mask in masks:
            for policy in ['original_admissible','support_clean']:
                arr=scores[mask];valid=np.isfinite(arr).all(axis=(0,1,2,3))
                if policy=='support_clean':valid&=support_ok[mask].all(axis=(0,1))
                ave=arr[...,valid].mean(axis=(0,1));orig=pts[mask].mean(axis=(0,1));nv=int(valid.sum())
                for m,method in enumerate(METHODS):
                    for v,metric in enumerate(['D_RL','B','S_Q']):
                        band=np.quantile(ave[m,v],[.025,.5,.975]) if nv else [np.nan]*3
                        aggregates.append(dict(block_length=length,group=group,policy=policy,method=method,metric=metric,
                            point=orig[m,v],low=band[0],median=band[1],high=band[2],valid_replicates=nv))
                for m,b0 in [(1,0),(2,1),(3,1)]:
                    values=ave[m,0]-ave[b0,0];band=np.quantile(values,[.025,.5,.975]) if nv else [np.nan]*3
                    deltas.append(dict(block_length=length,group=group,policy=policy,comparison=f'{METHODS[m]} minus {METHODS[b0]}',
                        point=orig[m,0]-orig[b0,0],low=band[0],median=band[1],high=band[2],valid_replicates=nv,
                        fraction_first_lower=float((values<0).mean()) if nv else np.nan))
        for k,st in enumerate(cohort.itertuples()):
            for policy in ['original_admissible','support_clean']:
                valid=np.isfinite(scores[k]).all(axis=(0,1,2))
                if policy=='support_clean':valid&=support_ok[k].all(axis=0)
                ave=scores[k][:,:,:,valid].mean(axis=0)
                orig=pts[k].mean(axis=0)
                for m,method in enumerate(METHODS):
                    band=np.quantile(ave[m,0],[.025,.5,.975]) if valid.any() else [np.nan]*3
                    individual.append(dict(block_length=length,policy=policy,station_id=st.station_id,station_name=st.station_name,
                        primary=st.primary,method=method,point=orig[m,0],low=band[0],median=band[1],high=band[2],valid_replicates=int(valid.sum())))
    for name,rows in [('aggregate_metrics',aggregates),('aggregate_differences',deltas),('individual_D',individual),('replicate_diagnostics',diagnostics)]:
        pd.DataFrame(rows).to_csv(DEST/f'{name}.csv',index=False)
    save_json(DEST/'complete.json',dict(status='complete',replicates=B,block_lengths=[1,3,5],methods=METHODS,
        support_sensitivity='Original admissibility vs zero fitted-support violations on all needed samples, jointly for comparisons',
        missing='Original calendar masks retained, effective sample count varies; no redrawing failures',
        scope='Six primary stations; Boulmer and no-nominal-trend subset remain exploratory; no point predictions altered'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=3);args=ap.parse_args();set_num_threads(args.threads)
    DEST.mkdir(parents=True,exist_ok=True);freeze();evaluate()
