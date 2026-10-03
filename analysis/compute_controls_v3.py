"""Fixed sample-length and known-distribution controls; no evaluation-based tuning."""
import argparse,time
import pandas as pd
from scipy.special import ndtr
from scipy.stats import genextreme
from numba import set_num_threads
from gev_lmoments_kernel import fit_batch
from research_v3_common import *
from fast_scores_v3 import fast_scores

DEST=OUT/'controls'
REPS=200;SUBSET=128;LENGTHS=[15,20,30,40]

def fit_index(a,index):
    ny=index.shape[1];b=len(index);n=a.shape[2];pars=[];status=[];support=[]
    for j in range(a.shape[1]):
        p,_,_,s,sp=fit_batch(np.ascontiguousarray(a[:,j][index].transpose(0,2,1).reshape(b*n,ny)),np.empty(0))
        pars.append(p.T);status.append(s);support.append(sp)
    return np.stack(pars),np.stack(status),np.stack(support)

def freeze_lengths():
    geo=geometry()
    # Fixed before new sample-length outcomes: historical intensity strata, identical coordinates for all members.
    h,_,_=frozen_parameters('01',geo['cells'])
    q=gev_curves(h[2],reduced_variate(np.array([10.])))[0]
    ids=np.argsort(q)[np.floor((np.arange(SUBSET)+.5)*len(q)/SUBSET).astype(int)]
    save_json(DEST/'design.json',dict(status='frozen',repetitions=REPS,subset_count=SUBSET,lengths=LENGTHS,
        subset_land_indices=ids.tolist(),subset_full_grid_indices=geo['cells'][ids].tolist(),
        selection='One fixed grid point from each equal-count historical member01 3h Q10 stratum; no future-hourly information',
        sampling='Independent historical/future circular 3-year draws of n years; same indices across durations/cells/methods',
        reference='Fixed full40-year future-hourly GEV, accessed only for evaluation; this isolates input sample-length effects',
        interval='Input resampling spread, conditional on the fixed reference; differs from full joint reference bootstrap'))
    for member in MEMBERS:
        allowed=np.load(OUT/'point_diagnostics'/f'allowed_ams_{member}.npz')
        for n in LENGTHS:
            path=DEST/f'length_parameters_{member}_{n}.npz'
            if path.exists():continue
            draws=[]
            for seed in [930300+int(member),930400+int(member)]:
                rng=np.random.default_rng(seed+n)
                start=rng.integers(0,40,(REPS,int(np.ceil(n/3))))
                draws.append(((start[:,:,None]+np.arange(3))%40).reshape(REPS,-1)[:,:n])
            pa=[];status=[];support=[]
            for role,index in zip(['historical_hourly','historical_daily','future_daily'],[draws[0],draws[0],draws[1]]):
                p,s,sp=fit_index(allowed[role][:,:,ids],index);pa.append(p);status.append(s);support.append(sp)
            sc,_=scaling(np.stack(pa[1:]))
            np.savez_compressed(path,h=pa[0],hd=pa[1],fd=pa[2],sc=sc,subset=ids,ih=draws[0],iff=draws[1],
                status=np.concatenate(status),support=np.concatenate(support))
            print(f'LENGTH INPUTS FROZEN {member} n={n}',flush=True)
    save_json(DEST/'length_prediction_fits_frozen.json',dict(status='complete',future_hourly_access_count=0,
        files={p.name:sha(p) for p in DEST.glob('length_parameters_*.npz')}))

def evaluate_lengths():
    design=json.loads((DEST/'design.json').read_text());ids=np.array(design['subset_land_indices']);cells=geometry()['cells'][ids]
    rows=[];widthrows=[]
    for member in MEMBERS:
        ref=reference_parameters(member,cells)
        for n in LENGTHS:
            fn=DEST/f'length_results_{member}_{n}.npz'
            if fn.exists():
                r=np.load(fn);scores=r['scores'];q100=r['q100'];good=r['good']
            else:
                f=np.load(DEST/f'length_parameters_{member}_{n}.npz')
                scores=np.full((7,4,3,REPS,SUBSET),np.nan);q100=np.full((7,4,REPS,SUBSET),np.nan);good=np.zeros((7,REPS,SUBSET),bool)
                for j,d in enumerate(DURATIONS):
                    tr=np.tile(ref[j],(1,REPS))
                    ss,valid=fast_scores(f['h'][j],f['hd'][0],f['fd'][0],f['sc'],tr,float(d))
                    scores[j]=ss.reshape(4,3,REPS,SUBSET);good[j]=valid.reshape(REPS,SUBSET)
                    q=prediction_curves(f['h'][j],f['hd'][0],f['fd'][0],f['sc'],d)[0]
                    q100[j]=q[:,-1].reshape(4,REPS,SUBSET)
                np.savez_compressed(fn,scores=scores.astype(np.float32),q100=q100.astype(np.float32),good=good)
            for j,d in enumerate(DURATIONS):
                for m,method in enumerate(METHODS):
                    vals=scores[j,m,0];complete=np.isfinite(scores[j]).all(axis=(0,1,3));ave=vals[complete].mean(axis=1)
                    band=np.quantile(ave,[.025,.5,.975]) if len(ave) else [np.nan]*3
                    rows.append(dict(member=member,n_years=n,duration_h=int(d),method=method,mean_D=float(np.nanmean(vals)),
                        low=band[0],median=band[1],high=band[2],complete_repetitions=int(complete.sum()),total_repetitions=REPS))
                    lo,med,hi=np.nanquantile(q100[j,m],[.025,.5,.975],axis=0);w=np.log(hi/lo)
                    widthrows.append(dict(member=member,n_years=n,duration_h=int(d),method=method,T=100,
                        W_median=float(np.nanmedian(w)),W_P10=float(np.nanquantile(w,.1)),W_P90=float(np.nanquantile(w,.9))))
            print(f'LENGTH EVALUATED {member} n={n}',flush=True)
    pd.DataFrame(rows).to_csv(DEST/'sample_length_scores.csv',index=False)
    pd.DataFrame(widthrows).to_csv(DEST/'sample_length_widths.csv',index=False)

def theoretical(eta_mu=.65,eta_sigma=.5,amp=1.,xi=.1):
    durations=np.r_[DURATIONS,[24,48,72]].astype(float)
    return np.stack([amp*40*(durations/24)**(1-eta_mu),amp*9*(durations/24)**(1-eta_sigma),np.full(10,xi)],axis=1)

def synthetic():
    names=['No change','Common amplitude','Extra subdaily change','Hourly tail change','Changed temporal scaling']
    scenarios=[]
    for sid,name in enumerate(names):
        historical=theoretical();future=historical.copy()
        if sid==1:future=theoretical(amp=1.2)
        if sid in [2,3]:future=theoretical(amp=1.15)
        if sid==2:future[:7,:2]*=((24/DURATIONS)**.08)[:,None]
        if sid==3:future[:7,2]+=.15
        if sid==4:future=theoretical(eta_mu=.72,eta_sigma=.62,amp=1.2,xi=.14)
        scenarios.append((historical,future))
    design=dict(status='design_fixed_before_simulation',scenarios=names,years=40,replicates=1000,
        latent_normal_shared_fraction=.8,seed=931700,
        rationale='Population transfer discrepancy contrasted with estimation error under the same scenarios; includes no-change, common amplitude, unobserved subdaily changes and an exact TPS scaling family.',
        limits='Correlated GEV marginal samples, not a physically coherent storm simulator. Dependence is fixed, not estimated; no event or copula inference. Population identities are calibration checks, not empirical validation.')
    save_json(DEST/'synthetic_design.json',design)
    pop=[];mc=[];records=[];rng=np.random.default_rng(931700);Bsim=1000;n=40
    for si,(historical,future) in enumerate(scenarios):
        hp=historical[:7,:,None]
        daily=np.stack([historical[7:],future[7:]])[:,:,:,None];sc,_=scaling(daily)
        prediction=np.stack([prediction_curves(hp[j],daily[0,0],daily[1,0],sc,d)[0][:,IDS,0] for j,d in enumerate(DURATIONS)])
        # Allowed pseudo-inputs are drawn/fitted before the target-hourly population reference is evaluated.
        samples=[]
        for pars in [historical,future]:
            shared=rng.normal(size=(Bsim,n,1));specific=rng.normal(size=(Bsim,n,10))
            u=ndtr(np.sqrt(.8)*shared+np.sqrt(.2)*specific)
            sample=genextreme.ppf(u,c=-pars[:,2],loc=pars[:,0],scale=pars[:,1])
            samples.append(sample)
        fitted=[]
        for sample,select in [(samples[0],slice(0,7)),(samples[0],slice(7,10)),(samples[1],slice(7,10))]:
            values=sample[:,:,select];nd=values.shape[-1]
            p,_,_,status,support=fit_batch(np.ascontiguousarray(values.transpose(2,0,1).reshape(nd*Bsim,n)),np.empty(0))
            fitted.append(p.reshape(nd,Bsim,3).transpose(0,2,1))
        sf,_=scaling(np.stack(fitted[1:]))
        population=np.empty((7,4,3));estimated=np.full((7,4,3,Bsim),np.nan)
        for j,d in enumerate(DURATIONS):
            truth=gev_curves(future[j,:,None],reduced_variate(T))
            population[j]=score_curves(prediction[j,:,:,None],truth)[:,:,0]
            estimated[j]=fast_scores(fitted[0][j],fitted[1][0],fitted[2][0],sf,np.tile(future[j,:,None],(1,Bsim)),float(d))[0]
        pop.append(population);mc.append(estimated)
        for m,method in enumerate(METHODS):
            good=np.isfinite(estimated).all(axis=(0,1,2));vals=estimated[:,m,0][:,good].mean(axis=0)
            band=np.quantile(vals,[.025,.5,.975])
            records.append(dict(scenario=names[si],method=method,population_D=float(population[:,m,0].mean()),
                finite_sample_mean_D=float(vals.mean()),finite_sample_median_D=band[1],finite_sample_low=band[0],finite_sample_high=band[2],
                valid_replicates=int(good.sum()),replicates=Bsim,years=n))
        print(f'SYNTHETIC COMPLETE {names[si]}',flush=True)
    np.savez_compressed(DEST/'synthetic_results.npz',population=np.stack(pop),sampled=np.stack(mc),scenarios=np.array(names),known_parameters=np.array(scenarios))
    pd.DataFrame(records).to_csv(DEST/'synthetic_summary.csv',index=False)
    save_json(DEST/'complete.json',dict(status='complete',sample_lengths=LENGTHS,subset_cells=SUBSET,
        sample_length_repetitions=REPS,synthetic_scenarios=len(names),synthetic_repetitions=Bsim,
        evidence='Finite-sample sensitivity conditional on fixed full-period reference; known distributions illustrate assumed conditions, not future-climate truth'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=2);args=ap.parse_args();set_num_threads(args.threads)
    DEST.mkdir(parents=True,exist_ok=True);freeze_lengths();evaluate_lengths();synthetic()
