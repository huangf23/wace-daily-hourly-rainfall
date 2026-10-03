"""Post-hoc, frozen-design robustness checks for the WACE manuscript.

Allowed-input strata and HQT candidates are saved before evaluation access.
Coverage uses known synthetic populations, not empirical 'truth'.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,math,time,json,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import genextreme
from numba import njit,prange,set_num_threads
from research_v3_common import *
from transfer_core import quantile
from fast_scores_v3 import pava_one,fast_scores
from gev_lmoments_kernel import fit_batch
from compute_controls_v3 import theoretical

DEST=ROOT/'research_wace'/'submission_checks_20261002'
EPS=[1e-4,1e-5,1e-6,1e-7,1e-8]
BANDS=[(2,10),(10,25),(25,100),(2,100)]
SIM_SCENARIOS=[0,1,4];RHOS=[0.,.3];OUTER=500;INNER=499;BATCH=20
SIM_DURS=[0,2,6];PAIRS=[(1,0),(2,1),(3,1)]
QIDS=np.abs(BASE_T[:,None]-np.array([10.,100.])[None]).argmin(axis=0)

def local_geometry():
    f=DEST/'input_snapshot_geometry.npz'
    return dict(np.load(f)) if f.exists() else geometry()

def local_parameters(member):
    f=DEST/f'input_snapshot_{member}.npz'
    return np.load(f) if f.exists() else np.load(OUT/'point_diagnostics'/f'predictions_{member}.npz')

def design():
    DEST.mkdir(parents=True,exist_ok=True)
    d=dict(status='frozen_before_new_outcomes',date='2026-10-02',analysis_status='post_hoc_extension_of_existing_study',
      hqt_epsilon=EPS,baseline_epsilon=1e-6,projection='unchanged fixed 401-node plus engineering-node log-PAVA',
      return_period_bands=BANDS,strata=['signed daily log-change averaged over log T=2..100','absolute historical hourly-minus-daily log tail growth Q100/Q2','historical hourly AMS midpoint circular seasonality R'],
      strata_definition='Within-member terciles; duration-specific for tail relation and seasonality. Joint 3x3 daily-change/tail strata. All features and cutpoints frozen using allowed inputs before new outcomes.',
      spatial_uncertainty='100-km square native-coordinate blocks; 1000 paired spatial bootstrap draws with fixed strata, common blocks across members; conditional descriptive intervals, not climate-model uncertainty or independent prediction validation.',
      coverage=dict(scenarios=SIM_SCENARIOS,latent_year_AR1=RHOS,outer=OUTER,inner=INNER,years=40,block_length=3,durations=[1,3,12],seed=10202026,
        estimands=['Population transfer-minus-reference integrated error contrasts; percentile and basic intervals','Method-specific population-prediction quantiles: nominal estimator coverage','True future quantiles: secondary inclusion rate, confounds structural transfer error'],
        dependence='Gaussian latent AR1; same-year shared fraction .8 across 10 duration marginals; periods independent; iid and stationary AR1(.3) cases; not nonstationary or physical storm models',
        denominator='All outer samples; unusable intervals count as noncoverage, with conditional coverage and validity also reported; no redrawing',
        limitations='Local three-duration aggregate contrasts, not spatial aggregate coverage; finite set of distributions; no empirically calibrated climate model'),
      selection_policy='No epsilon or method selected by evaluation; no retrospective optimized applicability classifier',future_hourly_role='evaluation only')
    path=DEST/'design.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==json.loads(json.dumps(d))
    else:save_json(path,d)
    return d

@njit(parallel=True,cache=True,nogil=True)
def hqt_curves(h,hd,fd,epsilon):
    n=h.shape[1];q=np.empty((len(BASE_A),n));flags=np.zeros((len(BASE_A),n),np.uint8)
    low=-math.log(-math.log(epsilon));high=-math.log(-math.log1p(-epsilon))
    for g in prange(n):
        values=np.empty(len(BASE_A))
        for j in range(len(BASE_A)):
            df=quantile(fd[0,g],fd[1,g],fd[2,g],BASE_A[j]);z=(df-hd[0,g])/hd[1,g];xi=hd[2,g];v=xi*z
            a=(math.inf if xi<0 else -math.inf) if abs(xi)>=1e-10 and 1+v<=0 else (z if abs(xi)<1e-10 else math.log1p(v)/xi)
            astar=min(high,max(low,a));flags[j,g]=astar!=a;values[j]=quantile(h[0,g],h[1,g],h[2,g],astar)
        if np.all(np.isfinite(values)&(values>0)):pava_one(values)
        q[:,g]=values
    return q,flags

def band_score(logerror,lo,hi):
    # Exact band endpoints interpolated in log T, matching the earlier audit.
    x=np.log(T);xx=np.r_[np.log(lo),x[(T>lo)&(T<hi)],np.log(hi)]
    ids=np.searchsorted(x,xx,side='right')-1;ids=np.clip(ids,0,len(T)-2);frac=(xx-x[ids])/(x[ids+1]-x[ids])
    vals=logerror[...,ids,:]*(1-frac)[...,None]+logerror[...,ids+1,:]*frac[...,None]
    return np.trapezoid(abs(vals),x=xx,axis=-2)/np.log(hi/lo)

def tercile(x):
    cuts=np.quantile(x,[1/3,2/3],axis=-1)
    groups=(x>cuts[0,...,None]).astype(np.int8)+(x>cuts[1,...,None]).astype(np.int8)
    return groups,cuts

def freeze_member(member):
    set_num_threads(2);geo=local_geometry();n=len(geo['cells']);fn=DEST/f'allowed_design_{member}.npz'
    if (DEST/f'frozen_{member}.json').exists():return
    p=local_parameters(member);h=p['h'];daily=p['daily'];scale=p['scale']
    hd=gev_curves(daily[0,0],reduced_variate(T));fd=gev_curves(daily[1,0],reduced_variate(T));daily_signal=np.trapezoid(np.log(fd/hd),x=np.log(T),axis=0)/np.log(50)
    tail=np.empty((7,n))
    for j in range(7):
        hh=gev_curves(h[j],reduced_variate(np.array([2.,100.])));tail[j]=abs(np.log(hh[1]/hh[0])-np.log(hd[-1]/hd[0]))
    if 'historical_midpoints' in p:
        season=abs(np.exp(2j*np.pi*(p['historical_midpoints']%8640)/8640).mean(axis=0))
    else:
        with nc.Dataset(sources(member)['historical_hourly']) as ds:
            jj=[int(np.flatnonzero(ds['duration'][:]==t)[0]) for t in DURATIONS]
            assert ds['event_start'].calendar=='360_day' and ds['event_start'].units.startswith('hours since ')
            starts=np.asarray(ds['event_start'][:,jj],float).reshape(40,7,-1)[:,:,geo['cells']]
            ends=np.asarray(ds['event_end'][:,jj],float).reshape(40,7,-1)[:,:,geo['cells']]
            season=abs(np.exp(2j*np.pi*((starts+ends)/2%8640)/8640).mean(axis=0))
    gd,cd=tercile(daily_signal);gt,ct=tercile(tail);gs,cs=tercile(season)
    np.savez(fn,daily_signal=daily_signal,tail=tail,season=season,gd=gd,gt=gt,gs=gs,cd=cd,ct=ct,cs=cs,cells=geo['cells'])
    curves=np.empty((len(EPS),7,len(T),n));clip=np.empty((len(EPS),7,n));clip100=np.empty_like(clip)
    for ie,ep in enumerate(EPS):
        for j in range(7):
            q,fl=hqt_curves(h[j],daily[0,0],daily[1,0],ep);curves[ie,j]=q[IDS];clip[ie,j]=fl.mean(axis=0);clip100[ie,j]=fl[-1]
    baseline=p['pred'][2] if 'pred' in p else np.stack([prediction_curves(h[j],daily[0,0],daily[1,0],scale,dur)[0][2,IDS] for j,dur in enumerate(DURATIONS)])
    assert np.allclose(curves[2],baseline,rtol=2e-12,atol=1e-10)
    assert np.isfinite(curves).all() and (curves>0).all()
    np.save(DEST/f'hqt_candidates_{member}.npy',curves);np.savez(DEST/f'hqt_flags_{member}.npz',fraction=clip,at100=clip100)
    save_json(DEST/f'frozen_{member}.json',dict(status='complete',member=member,future_hourly_access_count=0,features_sha=sha(fn),candidate_sha=sha(DEST/f'hqt_candidates_{member}.npy'),baseline_reproduced=True))
    print('FROZEN features and 5 HQT candidates '+member,flush=True)

def evaluate_member(member):
    set_num_threads(2);assert all((DEST/f'frozen_{m}.json').exists() for m in MEMBERS)
    f=local_parameters(member)
    if 'pred' in f:
        pred=f['pred'];diag=np.load(OUT/'point_diagnostics'/f'diagnostics_{member}.npz');truth=diag['truth']
    else:
        h,daily,scale=f['h'],f['daily'],f['scale'];pred=np.stack([prediction_curves(h[j],daily[0,0],daily[1,0],scale,dur)[0][:,IDS] for j,dur in enumerate(DURATIONS)],axis=1)
        ref=f['evaluation_reference_parameters'];truth=np.stack([gev_curves(ref[j],reduced_variate(T)) for j in range(7)])
    err=np.log(pred/truth[None]);score=np.stack([band_score(err,lo,hi) for lo,hi in BANDS]);rows=[]
    for ib,(lo,hi) in enumerate(BANDS):
        for m in range(4):
            for j,dur in enumerate(DURATIONS):rows.append(dict(member=member,method=METHODS[m],duration_h=int(dur),T_low=lo,T_high=hi,mean_D=float(score[ib,m,j].mean())))
    pd.DataFrame(rows).to_csv(DEST/f'bands_{member}.csv',index=False)
    # Verify independent reproduction of previously retained band summaries.
    prior=pd.read_csv(OUT/'point_diagnostics'/f'T_bands_{member}.csv')
    got=pd.DataFrame(rows);check=got.merge(prior,on=['member','method','duration_h','T_low','T_high'],suffixes=('_new','_old')) if prior.member.dtype==object else None
    prior['member']=prior.member.astype(str).str.zfill(2);check=got.merge(prior,on=['member','method','duration_h','T_low','T_high'],suffixes=('_new','_old'));assert np.allclose(check.mean_D_new,check.mean_D_old,atol=1e-10)
    q=np.load(DEST/f'hqt_candidates_{member}.npy',mmap_mode='r');flags=np.load(DEST/f'hqt_flags_{member}.npz');es=[];rows=[]
    for ie,ep in enumerate(EPS):
        ee=np.log(q[ie]/truth);ss=np.stack([band_score(ee,lo,hi) for lo,hi in BANDS]);es.append(ss)
        for ib,(lo,hi) in enumerate(BANDS):
            for j,dur in enumerate(DURATIONS):rows.append(dict(member=member,epsilon=ep,duration_h=int(dur),T_low=lo,T_high=hi,mean_D=float(ss[ib,j].mean()),improves_UCF=float((ss[ib,j]<score[ib,1,j]).mean()),clip_fraction=float(flags['fraction'][ie,j].mean()),clip_100=float(flags['at100'][ie,j].mean()),max_Q100=float(q[ie,j,-1].max()),median_Q100=float(np.median(q[ie,j,-1]))))
    pd.DataFrame(rows).to_csv(DEST/f'epsilon_{member}.csv',index=False)
    np.savez(DEST/f'grid_evaluation_{member}.npz',scores=score,epsilon_scores=np.array(es),Q100=q[:,:,-1],baseline_valid=np.isfinite(pred).all(axis=(0,2)))
    print('EVALUATED epsilon and bands '+member,flush=True)

def stratify():
    geo=local_geometry();xx,yy=np.meshgrid(geo['x'] if 'x' in geo else geo['projection_x_coordinate']/1000,geo['y'] if 'y' in geo else geo['projection_y_coordinate']/1000)
    bxy=np.stack([np.floor(xx.ravel()[geo['cells']]/100),np.floor(yy.ravel()[geo['cells']]/100)],axis=1);blocks,bi=np.unique(bxy,axis=0,return_inverse=True);nb=len(blocks)
    rng=np.random.default_rng(10202027);weights=rng.multinomial(nb,np.full(nb,1/nb),size=1000)
    matrices={};rows=[];counts=[];features=[]
    for mi,member in enumerate(MEMBERS):
        f=np.load(DEST/f'allowed_design_{member}.npz');score=np.load(DEST/f'grid_evaluation_{member}.npz')['scores'][-1];dif=np.array([score[m]-score[b] for m,b in PAIRS]);groups={'daily_change':np.broadcast_to(f['gd'],(7,len(bi))),'historical_tail':f['gt'],'historical_seasonality':f['gs'],'daily_x_tail':np.broadcast_to(f['gd'],(7,len(bi)))*3+f['gt']}
        for name,key in [('daily_change','cd'),('historical_tail','ct'),('historical_seasonality','cs')]:features.append(dict(member=member,feature=name,cutpoints=f[key].tolist()))
        for name,g in groups.items():
            ng=9 if name=='daily_x_tail' else 3;sums=np.zeros((ng,3,nb));den=np.zeros((ng,nb));win=np.zeros_like(sums)
            for c in range(ng):
                mask=g==c
                for j in range(7):
                    den[c]+=np.bincount(bi,weights=mask[j],minlength=nb)
                    for k in range(3):
                        sums[c,k]+=np.bincount(bi,weights=dif[k,j]*mask[j],minlength=nb);win[c,k]+=np.bincount(bi,weights=(dif[k,j]<0)*mask[j],minlength=nb)
                for k in range(3):rows.append(dict(member=member,feature=name,group=c,comparison=['UCF-NC','HQT-UCF','TPS-UCF'][k],mean_delta=sums[c,k].sum()/den[c].sum(),improve_fraction=win[c,k].sum()/den[c].sum(),n_local=int(den[c].sum()),n_blocks=int((den[c]>0).sum())))
            matrices[(member,name)]=(sums,den,win)
    aggregate=[]
    for name in ['daily_change','historical_tail','historical_seasonality','daily_x_tail']:
        ng=9 if name=='daily_x_tail' else 3;draws=[];points=[];wins=[]
        for member in MEMBERS:
            sums,den,win=matrices[(member,name)];denboot=weights@den.T
            val=np.stack([(weights@sums[:,k].T)/denboot for k in range(3)],axis=-1)
            draws.append(val);points.append(sums.sum(axis=-1)/den.sum(axis=-1)[:,None]);wins.append(win.sum(axis=-1)/den.sum(axis=-1)[:,None])
        samples=np.mean(draws,axis=0);pts=np.mean(points,axis=0);wi=np.mean(wins,axis=0);interval=np.quantile(samples,[.025,.975],axis=0)
        for c in range(ng):
            for k in range(3):aggregate.append(dict(feature=name,group=c,comparison=['UCF-NC','HQT-UCF','TPS-UCF'][k],mean_delta=pts[c,k],low=interval[0,c,k],high=interval[1,c,k],improve_fraction=wi[c,k],valid_spatial_draws=int(np.isfinite(samples[:,c,k]).sum()),blocks=nb))
    pd.DataFrame(rows).to_csv(DEST/'strata_member.csv',index=False);pd.DataFrame(aggregate).to_csv(DEST/'strata_aggregate.csv',index=False);save_json(DEST/'strata_cutpoints.json',features)
    np.savez(DEST/'spatial_bootstrap_design.npz',blocks=blocks,cell_block=bi,weights=weights)
    print('STRATIFICATION COMPLETE '+str(nb)+' spatial blocks',flush=True)

def samples(pars,rho,nouter,seed):
    rng=np.random.default_rng(seed);shared=rng.normal(size=(nouter,40,1));specific=rng.normal(size=(nouter,40,10));z=np.sqrt(.8)*shared+np.sqrt(.2)*specific
    for i in range(1,40):z[:,i]=rho*z[:,i-1]+np.sqrt(1-rho*rho)*z[:,i]
    u=np.clip(ndtr(z),np.finfo(float).eps,1-np.finfo(float).eps)
    return genextreme.ppf(u,c=-pars[:,2],loc=pars[:,0],scale=pars[:,1])

def fit_samples(a):
    # a [replicate,year,duration], output [duration,parameter,replicate]
    r,ny,nd=a.shape;p,_,_,status,support=fit_batch(np.ascontiguousarray(a.transpose(2,0,1).reshape(nd*r,ny)),np.empty(0))
    return p.reshape(nd,r,3).transpose(0,2,1),status.reshape(nd,r),support.reshape(nd,r)

def prediction_levels(h,hd,fd,sc,d):
    q,good,_,_,_=prediction_curves(h,hd,fd,sc,d)
    return q[:,QIDS],good.all(axis=0)

def coverage_task(config):
    sid,rho,offset,nouter=config;set_num_threads(1);path=DEST/'coverage_chunks'/f's{sid}_r{rho:.1f}_{offset:04d}.npz'
    if path.exists():return str(path)
    known=np.load(OUT/'controls'/'synthetic_results.npz')['known_parameters'];hp,fp=known[sid];seed=10202026+sid*100000+int(rho*10)*10000+offset
    hs=samples(hp,rho,nouter,seed);fs=samples(fp,rho,nouter,seed+5000000)
    # Fit prediction inputs before any target-hourly reference fitting.
    h,_,_=fit_samples(hs[:,:,:7]);hd,_,_=fit_samples(hs[:,:,7:]);fd,_,_=fit_samples(fs[:,:,7:]);sc,_=scaling(np.stack([hd,fd]))
    pointq=[]
    for j in SIM_DURS:pointq.append(prediction_levels(h[j],hd[0],fd[0],sc,float(DURATIONS[j]))[0])
    pointq=np.array(pointq) # duration, method, T, outer
    rng=np.random.default_rng(seed+9000000);indices=[]
    for period in range(2):
        starts=rng.integers(0,40,(nouter,INNER,14));indices.append(((starts[...,None]+np.arange(3))%40).reshape(nouter,INNER,-1)[...,:40])
    hboot=hs[np.arange(nouter)[:,None,None],indices[0]].reshape(nouter*INNER,40,10)
    fboot=fs[np.arange(nouter)[:,None,None],indices[1]].reshape(nouter*INNER,40,10)
    bh,_,_=fit_samples(hboot[:,:,:7]);bhd,_,_=fit_samples(hboot[:,:,7:]);bfd,_,_=fit_samples(fboot[:,:,7:]);bsc,_=scaling(np.stack([bhd,bfd]))
    bootq=[]
    for j in SIM_DURS:bootq.append(prediction_levels(bh[j],bhd[0],bfd[0],bsc,float(DURATIONS[j]))[0].reshape(4,2,nouter,INNER))
    bootq=np.array(bootq)
    # Evaluation-only fitting now follows frozen prediction inputs and curves.
    ref,_,_=fit_samples(fs[:,:,:7]);bref,_,_=fit_samples(fboot[:,:,:7]);ps=[];bs=[]
    for j in SIM_DURS:
        ps.append(fast_scores(h[j],hd[0],fd[0],sc,ref[j],float(DURATIONS[j]))[0][:,0]);bs.append(fast_scores(bh[j],bhd[0],bfd[0],bsc,bref[j],float(DURATIONS[j]))[0][:,0].reshape(4,nouter,INNER))
    ps=np.mean(ps,axis=0);bs=np.mean(bs,axis=0);delta=np.array([ps[m]-ps[b] for m,b in PAIRS]);bd=np.array([bs[m]-bs[b] for m,b in PAIRS]);valid=np.isfinite(bd).all(axis=0)
    low=np.full((3,nouter),np.nan);high=low.copy();ql=np.full(pointq.shape,np.nan);qh=ql.copy();qvalid=np.isfinite(bootq).all(axis=(0,1,2))
    for o in range(nouter):
        # At least 95% of inner draws must yield common-valid contrasts/levels.
        if valid[o].sum()>=math.ceil(.95*INNER):low[:,o],high[:,o]=np.quantile(bd[:,o,valid[o]],[.025,.975],axis=-1)
        if qvalid[o].sum()>=math.ceil(.95*INNER):
            ql[...,o],qh[...,o]=np.quantile(bootq[:,:,:,o,qvalid[o]],[.025,.975],axis=-1)
    np.savez(path,point_delta=delta,low=low,high=high,basic_low=2*delta-high,basic_high=2*delta-low,point_quantiles=pointq,quantile_low=ql,quantile_high=qh,valid_draws=valid.sum(axis=-1),valid_quantile_draws=qvalid.sum(axis=-1),scenario=sid,rho=rho,offset=offset)
    return str(path)

def merge_coverage():
    known=np.load(OUT/'controls'/'synthetic_results.npz')['known_parameters'];rows=[];qrows=[]
    for sid in SIM_SCENARIOS:
        hp,fp=known[sid];sc,_=scaling(np.stack([hp[7:],fp[7:]])[:,:,:,None]);population=[];pq=[];tq=[]
        for j in SIM_DURS:
            q=prediction_curves(hp[j,:,None],hp[7,:,None],fp[7,:,None],sc,float(DURATIONS[j]))[0];tr=gev_curves(fp[j,:,None],BASE_A);population.append(score_curves(q[:,IDS],tr[IDS])[:,0,0]);pq.append(q[:,QIDS,0]);tq.append(tr[QIDS,0])
        pop=np.mean(population,axis=0);contrast=np.array([pop[m]-pop[b] for m,b in PAIRS]);pq=np.array(pq);tq=np.array(tq)
        for rho in RHOS:
            chunks=[np.load(p) for p in sorted((DEST/'coverage_chunks').glob(f's{sid}_r{rho:.1f}_*.npz'))];assert sum(c['point_delta'].shape[-1] for c in chunks)==OUTER
            arrays={k:np.concatenate([c[k] for c in chunks],axis=-1) for k in ['point_delta','low','high','basic_low','basic_high','quantile_low','quantile_high','valid_draws','valid_quantile_draws']}
            for typ in ['percentile','basic']:
                lo=arrays['low' if typ=='percentile' else 'basic_low'];hi=arrays['high' if typ=='percentile' else 'basic_high']
                for k in range(3):
                    good=np.isfinite(lo[k])&np.isfinite(hi[k]);covered=good&(lo[k]<=contrast[k]+1e-12)&(hi[k]>=contrast[k]-1e-12);rate=covered.mean()
                    rows.append(dict(scenario=sid,rho=rho,comparison=['UCF-NC','HQT-UCF','TPS-UCF'][k],interval=typ,population_contrast=contrast[k],outer=OUTER,valid_outer=int(good.sum()),coverage=rate,conditional_coverage=float(covered[good].mean()) if good.any() else np.nan,mcse=math.sqrt(rate*(1-rate)/OUTER),median_width=float(np.nanmedian(hi[k]-lo[k])),mean_estimated_contrast=float(np.nanmean(arrays['point_delta'][k])),minimum_inner_valid=int(arrays['valid_draws'].min())))
            lo=arrays['quantile_low'];hi=arrays['quantile_high']
            for j,durj in enumerate(SIM_DURS):
                for m in range(4):
                    for it,rt in enumerate([10,100]):
                        good=np.isfinite(lo[j,m,it])&np.isfinite(hi[j,m,it]);cover=good&(lo[j,m,it]<=pq[j,m,it])&(hi[j,m,it]>=pq[j,m,it]);target=good&(lo[j,m,it]<=tq[j,it])&(hi[j,m,it]>=tq[j,it]);qrows.append(dict(scenario=sid,rho=rho,duration_h=int(DURATIONS[durj]),method=METHODS[m],T=rt,outer=OUTER,valid_outer=int(good.sum()),estimator_coverage=float(cover.mean()),true_target_inclusion=float(target.mean()),population_prediction=pq[j,m,it],true_target_quantile=tq[j,it],mcse=math.sqrt(cover.mean()*(1-cover.mean())/OUTER)))
    pd.DataFrame(rows).to_csv(DEST/'coverage_contrasts.csv',index=False);pd.DataFrame(qrows).to_csv(DEST/'coverage_quantiles.csv',index=False)
    save_json(DEST/'coverage_complete.json',dict(status='complete',configurations=6,outer_per_configuration=OUTER,inner=INNER,total_outer=6*OUTER,total_inner=6*OUTER*INNER,denominator='All outer draws; joint inner-valid counts disclosed',coverage_scope='Specified local synthetic contrasts and pointwise quantiles; does not establish spatial aggregate or nonstationary coverage'))

def run_grid():
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(freeze_member,MEMBERS))
    save_json(DEST/'all_inputs_frozen.json',dict(status='complete',future_hourly_access_count=0,files=[f'frozen_{m}.json' for m in MEMBERS]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(evaluate_member,MEMBERS))
    for prefix in ['bands','epsilon']:
        pd.concat([pd.read_csv(DEST/f'{prefix}_{m}.csv',dtype={'member':str}) for m in MEMBERS],ignore_index=True).to_csv(DEST/f'{prefix}_all.csv',index=False)
    stratify();save_json(DEST/'grid_complete.json',dict(status='complete',members=MEMBERS,land_cells=10397,baseline_reproduced=True,band_checks_against_previous=True,future_hourly_role='Evaluation after all four member feature and candidate freezes'))

def run_coverage(workers):
    (DEST/'coverage_chunks').mkdir(exist_ok=True)
    jobs=[(sid,rho,o,min(BATCH,OUTER-o)) for sid in SIM_SCENARIOS for rho in RHOS for o in range(0,OUTER,BATCH)];start=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(coverage_task,j) for j in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures)):
            f.result();save_json(DEST/'coverage_progress.json',dict(completed=i+1,total=len(jobs),seconds=time.time()-start));print(f'COVERAGE {i+1}/{len(jobs)} elapsed={time.time()-start:.0f}s',flush=True)
    merge_coverage()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['design','grid','coverage']);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args();design()
    if args.stage=='grid':run_grid()
    if args.stage=='coverage':run_coverage(args.workers)
