"""Land-grid joint score bootstrap; freeze allowed-input fits before evaluation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,time,json
from pathlib import Path
import numpy as np
import netCDF4 as nc
from numba import set_num_threads
from gev_lmoments_kernel import fit_batch
from research_v3_common import *
from fast_scores_v3 import fast_scores

DEST=OUT/'joint_bootstrap'

def fit_draws(a,ind):
    b=len(ind);n=a.shape[2];pp=[];st=[];sp=[]
    for d in range(a.shape[1]):
        smp=np.ascontiguousarray(a[:,d][ind].transpose(0,2,1).reshape(b*n,40))
        p,_,_,s,support=fit_batch(smp,np.empty(0))
        pp.append(p.T.reshape(3,b,n));st.append(s.reshape(b,n));sp.append(support.reshape(b,n))
    return np.stack(pp),np.stack(st),np.stack(sp)

def draws(member):
    with nc.Dataset(ROOT/'transfer_uncertainty_v1'/f'bootstrap_{member}.nc') as ds:
        return np.asarray(ds['historical_year_indices'][:],int),np.asarray(ds['future_year_indices'][:],int)

def freeze(args):
    cells=geometry()['cells'];n=len(cells);b=500;chunks=list(range(0,n,args.chunk))
    for member in MEMBERS:
        folder=DEST/member;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'prediction_fits_frozen.json').exists():continue
        data,audit=allowed_ams(member,cells);ih,iff=draws(member)
        meta=dict(member=member,replicates=b,chunk=args.chunk,cells=n,input_audit=audit,
                  pairing='original 500 circular 3-year draws; spatial and duration pairing retained',future_hourly_access_count=0)
        progress_path=folder/'freeze_progress.json'
        if progress_path.exists():
            old=json.loads(progress_path.read_text());assert old['config']==meta
            done=old['done'];mode='r+'
        else:done=[False]*len(chunks);mode='w+'
        # Keep the fitted arrays in RAM. Strided memmap flushes cause random I/O
        # on the local HDD; contiguous chunk checkpoints and one final write are faster.
        pars=np.empty((13,3,b,n),np.float64)
        status=np.empty((13,b,n),np.uint8)
        support=np.empty((13,b,n),np.uint16)
        legacy=None
        if mode=='r+' and (folder/'allowed_parameters.npy').exists():
            legacy=[np.load(folder/name,mmap_mode='r') for name in ['allowed_parameters.npy','allowed_status.npy','allowed_support.npy']]
        started=time.time()
        for c,s in enumerate(chunks):
            e=min(s+args.chunk,n);offset=0;checkpoint=folder/f'fits_{s:05d}_{e:05d}.npz'
            if checkpoint.exists():
                with np.load(checkpoint) as f:
                    pars[:,:,:,s:e]=f['p'];status[:,:,s:e]=f['st'];support[:,:,s:e]=f['sp']
                done[c]=True;continue
            if done[c] and legacy is not None:
                pars[:,:,:,s:e]=legacy[0][:,:,:,s:e];status[:,:,s:e]=legacy[1][:,:,s:e];support[:,:,s:e]=legacy[2][:,:,s:e]
                np.savez(checkpoint,p=pars[:,:,:,s:e],st=status[:,:,s:e],sp=support[:,:,s:e]);continue
            for role in ['historical_hourly','historical_daily','future_daily']:
                p,st,sp=fit_draws(data[role][:,:,s:e],iff if role=='future_daily' else ih)
                k=len(p);pars[offset:offset+k,:,:,s:e]=p;status[offset:offset+k,:,s:e]=st;support[offset:offset+k,:,s:e]=sp;offset+=k
            np.savez(checkpoint,p=pars[:,:,:,s:e],st=status[:,:,s:e],sp=support[:,:,s:e]);done[c]=True
            save_json(progress_path,dict(config=meta,done=done))
            print(f'FREEZE {member} {c+1}/{len(chunks)} elapsed={time.time()-started:.1f}s',flush=True)
        del legacy
        np.save(folder/'allowed_parameters.npy',pars);np.save(folder/'allowed_status.npy',status);np.save(folder/'allowed_support.npy',support)
        save_json(folder/'prediction_fits_frozen.json',dict(status='complete',config=meta,
            parameter_sha256=sha(folder/'allowed_parameters.npy'),elapsed_seconds=time.time()-started))
        del pars,status,support,data

def evaluate(args):
    geo=geometry();cells=geo['cells'];n=len(cells);b=500
    assert all((DEST/m/'prediction_fits_frozen.json').exists() for m in MEMBERS)
    for member in MEMBERS:
        folder=DEST/member
        if (folder/'evaluation_complete.json').exists():continue
        frozen=json.loads((folder/'prediction_fits_frozen.json').read_text())
        assert frozen['config']['future_hourly_access_count']==0
        assert sha(folder/'allowed_parameters.npy')==frozen['parameter_sha256']
        pars=np.load(folder/'allowed_parameters.npy')
        support=np.load(folder/'allowed_support.npy')
        truth_ams=reference_ams(member,cells);_,iff=draws(member)
        started=time.time();chunks=list(range(0,n,args.eval_chunk))
        # Independent chunk files allow resume without modifying already computed chunks.
        for c,s in enumerate(chunks):
            e=min(s+args.eval_chunk,n);fn=folder/f'eval_{s:05d}_{e:05d}.npz'
            if fn.exists():continue
            nb=e-s;tp,ts,tsp=fit_draws(truth_ams[:,:,s:e],iff)
            ph=np.ascontiguousarray(pars[:7,:,:,s:e]).reshape(7,3,b*nb)
            pd=np.ascontiguousarray(pars[7:,:,:,s:e]).reshape(2,3,3,b*nb)
            sc,_=scaling(pd)
            local=np.full((3,7,4,3,nb),np.nan,np.float32)
            delta=np.full((3,7,3,nb),np.nan,np.float32)
            sums=np.zeros((7,4,3,b));counts=np.zeros((7,b),np.int32)
            support_counts=np.zeros((7,nb),np.int32);valid_counts=np.zeros((7,nb),np.int32)
            idx=np.flatnonzero((geo['case_land_indices']>=s)&(geo['case_land_indices']<e))
            caseq=np.full((3,7,4,len(T),len(idx)),np.nan,np.float32)
            casetr=np.full((3,7,len(T),len(idx)),np.nan,np.float32)
            for j,d in enumerate(DURATIONS):
                scores,ok=fast_scores(ph[j],pd[0,0],pd[1,0],sc,np.ascontiguousarray(tp[j].reshape(3,b*nb)),float(d))
                val=scores.reshape(4,3,b,nb)
                local[:,j]=np.nanquantile(val,[.025,.5,.975],axis=2)
                dif=np.stack([val[1,0]-val[0,0],val[2,0]-val[1,0],val[3,0]-val[1,0]])
                delta[:,j]=np.nanquantile(dif,[.025,.5,.975],axis=1)
                sums[j]=np.nansum(val,axis=3);counts[j]=ok.reshape(b,nb).sum(axis=1)
                valid_counts[j]=ok.reshape(b,nb).sum(axis=0)
                badsupport=(support[j,:,s:e]>0)|(support[7:,:,s:e]>0).any(axis=0)|(tsp[j]>0)
                support_counts[j]=badsupport.sum(axis=0)
                if len(idx):
                    ci=geo['case_land_indices'][idx]-s
                    cp=np.ascontiguousarray(pars[:,:,:,s:e][:,:,:,ci]).reshape(13,3,b*len(ci))
                    csc,_=scaling(cp[7:].reshape(2,3,3,-1))
                    q=prediction_curves(cp[j],cp[7],cp[10],csc,d)[0]
                    tr=gev_curves(np.ascontiguousarray(tp[j][:,:,ci]).reshape(3,-1),BASE_A)
                    samples=q[:,IDS].reshape(4,len(T),b,len(ci))
                    caseq[:,j]=np.nanquantile(samples,[.025,.5,.975],axis=2)
                    trsam=tr[IDS].reshape(len(T),b,len(ci))
                    casetr[:,j]=np.nanquantile(trsam,[.025,.5,.975],axis=1)
                    del q,tr
                del scores,val
            np.savez_compressed(fn,local_ci=local,delta_ci=delta,sums=sums,counts=counts,valid_counts=valid_counts,
                support_counts=support_counts,case_indices=idx,case_prediction_ci=caseq,case_reference_ci=casetr,
                truth_fit_failures=int((ts!=0).sum()),start=s,end=e)
            save_json(DEST/'progress.json',dict(stage='evaluation',member=member,chunk=c+1,total_chunks=len(chunks),elapsed_seconds=time.time()-started))
            print(f'EVALUATE {member} {c+1}/{len(chunks)} elapsed={time.time()-started:.1f}s',flush=True)
        merge(member,chunks,args.eval_chunk,n,b)

def merge(member,chunks,size,n,b):
    folder=DEST/member;sums=np.zeros((7,4,3,b));counts=np.zeros((7,b),int)
    loc=np.full((3,7,4,3,n),np.nan,np.float32);delta=np.full((3,7,3,n),np.nan,np.float32)
    support=np.zeros((7,n),np.int32);valid=support.copy()
    cq=np.full((3,7,4,len(T),3),np.nan,np.float32);ct=np.full((3,7,len(T),3),np.nan,np.float32)
    for s in chunks:
        e=min(s+size,n)
        with np.load(folder/f'eval_{s:05d}_{e:05d}.npz') as f:
            sums+=f['sums'];counts+=f['counts'];loc[...,s:e]=f['local_ci'];delta[...,s:e]=f['delta_ci']
            valid[:,s:e]=f['valid_counts'];support[:,s:e]=f['support_counts']
            for k,index in enumerate(f['case_indices']):
                cq[...,index]=f['case_prediction_ci'][...,k];ct[...,index]=f['case_reference_ci'][...,k]
    means=sums/n;complete=counts==n
    for j in range(7):means[j,:,:,~complete[j]]=np.nan
    aggregate=means.mean(axis=0);allvalid=complete.all(axis=0)
    np.savez_compressed(folder/'joint_results.npz',means=means,aggregate=aggregate,counts=counts,complete=complete,
        local_ci=loc,delta_ci=delta,valid_counts=valid,support_counts=support,T=T,
        case_prediction_ci=cq,case_reference_ci=ct)
    save_json(folder/'evaluation_complete.json',dict(status='complete',replicates=b,land_cells=n,
        complete_spatial_replicates_by_duration=complete.sum(axis=1).tolist(),complete_all_duration_replicates=int(allvalid.sum()),
        minimum_local_valid=int(valid.min()),max_support_violation_replicates=int(support.max()),
        future_hourly_role='Joint evaluation reference only; same future draws as allowed daily; prediction fits frozen first',
        interval='Pointwise percentile 95%; original fit admissibility with support violations disclosed'))

def pilot(args):
    cells=geometry()['cells'][:32];data,_=allowed_ams('01',cells);ih,iff=draws('01')
    started=time.time();result=[]
    for role in ['historical_hourly','historical_daily','future_daily']:
        p,_,_=fit_draws(data[role],iff if role=='future_daily' else ih);result.append(p)
    h,hd,fd=[x.reshape(len(x),3,-1) for x in result];sc,_=scaling(np.stack([hd,fd]))
    freeze_seconds=time.time()-started
    tr,_,_=fit_draws(reference_ams('01',cells),iff)
    for j,d in enumerate(DURATIONS):
        q,good,_,_,_=prediction_curves(h[j],hd[0],fd[0],sc,d)
        true=gev_curves(tr[j].reshape(3,-1),BASE_A)
        v=score_curves(q[:,IDS],true[IDS]);assert np.isfinite(v).all()
    save_json(DEST/'pilot.json',dict(cells=32,replicates=500,threads=args.threads,
        fit_seconds=freeze_seconds,total_seconds=time.time()-started,estimated_full_seconds=(time.time()-started)*10397/32*4))
    print((DEST/'pilot.json').read_text(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','freeze','evaluate','all'])
    ap.add_argument('--threads',type=int,default=16);ap.add_argument('--chunk',type=int,default=256);ap.add_argument('--eval-chunk',type=int,default=64)
    args=ap.parse_args();set_num_threads(args.threads);DEST.mkdir(parents=True,exist_ok=True)
    if args.stage=='pilot':pilot(args)
    if args.stage in ['freeze','all']:freeze(args)
    if args.stage in ['evaluate','all']:evaluate(args)
    if args.stage=='all':save_json(DEST/'complete.json',dict(status='complete',members=MEMBERS,replicates=500))
