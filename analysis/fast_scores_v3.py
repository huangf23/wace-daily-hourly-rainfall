"""Fused per-series evaluation of the existing transfer + log-PAVA + score definitions."""
import math
import numpy as np
from numba import njit,prange
from transfer_core import quantile
from transfer_core_regularized import BASE_A,WEIGHTS,LOW_A,HIGH_A
from research_v3_common import IDS

@njit(cache=True,inline='always')
def pava_one(values):
    n=len(values);monotone=True
    for j in range(1,n):
        if values[j]<values[j-1]:monotone=False;break
    if monotone:return
    means=np.empty(n);mass=np.empty(n);ends=np.empty(n,np.int64);k=0
    for j in range(n):
        means[k]=math.log(values[j]);mass[k]=WEIGHTS[j];ends[k]=j;k+=1
        while k>1 and means[k-2]>means[k-1]:
            total=mass[k-2]+mass[k-1]
            means[k-2]=(mass[k-2]*means[k-2]+mass[k-1]*means[k-1])/total
            mass[k-2]=total;ends[k-2]=ends[k-1];k-=1
    start=0
    for block in range(k):
        value=math.exp(means[block])
        for j in range(start,ends[block]+1):values[j]=value
        start=ends[block]+1

@njit(parallel=True,cache=True,nogil=True)
def fast_scores(h,hd,fd,sc,ref,d):
    n=h.shape[1];out=np.full((4,3,n),np.nan);valid=np.zeros(n,np.bool_)
    for g in prange(n):
        if not (np.isfinite(h[:,g]).all() and np.isfinite(hd[:,g]).all() and np.isfinite(fd[:,g]).all()
                and np.isfinite(sc[:,:,g]).all() and np.isfinite(ref[:,g]).all()):continue
        if min(h[1,g],hd[1,g],fd[1,g],ref[1,g])<=0:continue
        ah=sc[0,0,g]*(d/24.)**(-sc[0,3,g]);bh=sc[0,1,g]*(d/24.)**(-sc[0,4,g])
        af=sc[1,0,g]*(d/24.)**(-sc[1,3,g]);bf=sc[1,1,g]*(d/24.)**(-sc[1,4,g])
        q=np.empty((4,len(BASE_A)));ok=True
        for j in range(len(BASE_A)):
            a=BASE_A[j]
            hh=quantile(h[0,g],h[1,g],h[2,g],a)
            dh=quantile(hd[0,g],hd[1,g],hd[2,g],a)
            df=quantile(fd[0,g],fd[1,g],fd[2,g],a)
            th=quantile(ah,bh,sc[0,2,g],a);tf=quantile(af,bf,sc[1,2,g],a)
            if dh<=0 or th<=0:ok=False;break
            q[0,j]=hh;q[1,j]=hh*df/dh;q[3,j]=hh*tf/th
            z=(df-hd[0,g])/hd[1,g];xi=hd[2,g];v=xi*z
            if abs(xi)>=1e-10 and 1+v<=0:
                astar=HIGH_A if xi<0 else LOW_A
            else:
                astar=z if abs(xi)<1e-10 else math.log1p(v)/xi
                astar=min(HIGH_A,max(LOW_A,astar))
            q[2,j]=quantile(h[0,g],h[1,g],h[2,g],astar)
            for m in range(4):
                if not (math.isfinite(q[m,j]) and q[m,j]>0):ok=False
            if not ok:break
        if not ok:continue
        for m in range(1,4):pava_one(q[m])
        logs=np.empty((4,len(IDS)))
        for j in range(len(IDS)):
            y=quantile(ref[0,g],ref[1,g],ref[2,g],BASE_A[IDS[j]])
            if not (math.isfinite(y) and y>0):ok=False;break
            for m in range(4):logs[m,j]=math.log(q[m,IDS[j]])-math.log(y)
        if not ok:continue
        valid[g]=True
        for m in range(4):
            total=0.;absolute=0.
            for j in range(len(IDS)):
                w=.5 if j==0 or j==len(IDS)-1 else 1.
                total+=w*logs[m,j];absolute+=w*abs(logs[m,j])
            bias=total/(len(IDS)-1);spread=0.
            for j in range(len(IDS)):
                w=.5 if j==0 or j==len(IDS)-1 else 1.
                spread+=w*(logs[m,j]-bias)**2
            out[m,0,g]=absolute/(len(IDS)-1);out[m,1,g]=bias;out[m,2,g]=math.sqrt(spread/(len(IDS)-1))
    return out,valid

def verify_and_time():
    import time
    from numba import set_num_threads
    from research_v3_common import geometry,frozen_parameters,reference_parameters,prediction_curves,gev_curves,BASE_A,IDS,score_curves
    set_num_threads(3)
    cells=geometry()['cells'][:256]
    h,daily,sc=frozen_parameters('01',cells);ref=reference_parameters('01',cells)
    errors=[]
    for j,d in enumerate([1,2,3,4,5,6,12]):
        q,good,_,_,_=prediction_curves(h[j],daily[0,0],daily[1,0],sc,d)
        tr=gev_curves(ref[j],BASE_A)
        expected=score_curves(q[:,IDS],tr[IDS]);actual,ok=fast_scores(h[j],daily[0,0],daily[1,0],sc,ref[j],float(d))
        error=float(np.nanmax(abs(actual-expected)));errors.append(error)
        assert ok.all() and error<1e-10
    hh=np.tile(h[0],(1,250));ddh=np.tile(daily[0,0],(1,250));ddf=np.tile(daily[1,0],(1,250))
    ss=np.tile(sc,(1,1,250));rr=np.tile(ref[0],(1,250))
    tic=time.time();actual,ok=fast_scores(hh,ddh,ddf,ss,rr,1.);secs=time.time()-tic
    print(dict(max_error=max(errors),series=len(ok),seconds=secs,threads=3),flush=True)

if __name__=='__main__':verify_and_time()
