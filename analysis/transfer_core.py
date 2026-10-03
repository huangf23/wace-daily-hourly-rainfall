"""Pure transfer functions. No file access and no future-hourly data dependency."""
import math
import numpy as np
from numba import njit,prange

METHODS=['UCF','HQT','TPS']
DURATIONS=np.array([1,2,3,4,5,6,12],dtype=np.int32)
T_ENGINEERING=np.array([2,5,10,20,25,50,100],dtype=np.float64)

def reduced_variate(T):return -np.log(-np.log1p(-1/np.asarray(T,dtype=np.float64)))

@njit(cache=True)
def quantile(mu,sigma,xi,a):
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma>0 and np.isfinite(xi)):return np.nan
    if abs(xi)<1e-10:return mu+sigma*a
    return mu+sigma*math.expm1(xi*a)/xi

@njit(parallel=True,nogil=True,cache=True)
def gev_curves(par,a):
    out=np.empty((len(a),par.shape[1]),np.float64)
    for g in prange(par.shape[1]):
        for j in range(len(a)):out[j,g]=quantile(par[0,g],par[1,g],par[2,g],a[j])
    return out

def scaling(daily):
    # daily: period (H/F), day duration (1/2/3), parameter (mu/sigma/xi), pixel.
    logk=np.log(np.array([1.,2.,3.]))
    out=np.full((2,5,daily.shape[-1]),np.nan)
    residual=np.full((2,2,daily.shape[-1]),np.nan)
    for s in range(2):
        intensity=daily[s,:2+1,:2]/np.array([24.,48.,72.])[:,None,None]
        out[s,0]=intensity[0,0];out[s,1]=intensity[0,1];out[s,2]=daily[s,0,2]
        for p in range(2):
            good=np.all(np.isfinite(intensity[:,p])&(intensity[:,p]>0),axis=0)
            ratios=np.full((3,daily.shape[-1]),np.nan)
            # Use explicit slicing to preserve day x pixel ordering.
            values=intensity[:,p,:]
            ratios[:,good]=np.log(values[:,good]/values[0,good])
            eta=-np.sum(logk[:,None]*ratios,axis=0)/np.sum(logk**2)
            out[s,3+p]=eta
            residual[s,p]=np.sqrt(np.mean((ratios+logk[:,None]*eta)**2,axis=0))
    return out,residual

@njit(parallel=True,nogil=True,cache=True)
def predict_curves(h,hd,fd,scale,t,a):
    n=h.shape[1];q=np.full((3,len(a),n),np.nan)
    u=np.full((len(a),n),np.nan);logtail=np.full_like(u,np.nan)
    boundary=np.zeros((len(a),n),np.uint8)
    for g in prange(n):
        ah=scale[0,0,g]*(t/24.)**(-scale[0,3,g]);bh=scale[0,1,g]*(t/24.)**(-scale[0,4,g])
        af=scale[1,0,g]*(t/24.)**(-scale[1,3,g]);bf=scale[1,1,g]*(t/24.)**(-scale[1,4,g])
        for j in range(len(a)):
            hourly=quantile(h[0,g],h[1,g],h[2,g],a[j])
            histday=quantile(hd[0,g],hd[1,g],hd[2,g],a[j])
            futureday=quantile(fd[0,g],fd[1,g],fd[2,g],a[j])
            if histday>0:q[0,j,g]=hourly*futureday/histday
            z=(futureday-hd[0,g])/hd[1,g];xi=hd[2,g]
            v=xi*z
            if abs(xi)>=1e-10 and 1+v<=0:
                if xi<0:
                    u[j,g]=1.;logtail[j,g]=-np.inf;boundary[j,g]=1
                    q[1,j,g]=h[0,g]-h[1,g]/h[2,g] if h[2,g]<0 else np.inf
                else:
                    u[j,g]=0.;logtail[j,g]=np.inf;boundary[j,g]=2
                    q[1,j,g]=h[0,g]-h[1,g]/h[2,g] if h[2,g]>0 else -np.inf
            else:
                logv=-z if abs(xi)<1e-10 else -math.log1p(v)/xi
                logtail[j,g]=logv
                u[j,g]=0. if logv>709 else math.exp(-math.exp(logv))
                # Avoid rounding an interior CDF to 1 before calculating its quantile.
                q[1,j,g]=quantile(h[0,g],h[1,g],h[2,g],-logv)
            qh=quantile(ah,bh,scale[0,2,g],a[j]);qf=quantile(af,bf,scale[1,2,g],a[j])
            if qh>0:q[2,j,g]=hourly*qf/qh
    return q,u,logtail,boundary

def valid_curves(q):
    positive=np.all(np.isfinite(q)&(q>0),axis=-2)
    with np.errstate(invalid='ignore'):difference=np.diff(q,axis=-2)
    tolerance=1e-10*np.maximum(1,np.max(np.abs(q),axis=-2))
    monotone=np.all(difference>=-tolerance[...,None,:],axis=-2)
    return positive,monotone,positive&monotone

@njit(parallel=True,nogil=True,cache=True)
def score_curves(pred,true):
    nm,nt,ng=pred.shape;out=np.full((nm,3,ng),np.nan)
    for g in prange(ng):
        for m in range(nm):
            total=0.;absolute=0.;ok=True
            for j in range(nt):
                if not (np.isfinite(pred[m,j,g]) and pred[m,j,g]>0 and np.isfinite(true[j,g]) and true[j,g]>0):ok=False;break
                e=math.log(pred[m,j,g])-math.log(true[j,g]);w=.5 if j==0 or j==nt-1 else 1.
                total+=w*e;absolute+=w*abs(e)
            if not ok:continue
            bias=total/(nt-1);spread=0.
            for j in range(nt):
                e=math.log(pred[m,j,g])-math.log(true[j,g]);w=.5 if j==0 or j==nt-1 else 1.
                spread+=w*(e-bias)**2
            out[m,0,g]=absolute/(nt-1);out[m,1,g]=bias;out[m,2,g]=math.sqrt(spread/(nt-1))
    return out

def tests():
    from scipy.stats import genextreme
    rng=np.random.default_rng(831);n=30
    h=np.vstack([rng.uniform(10,30,n),rng.uniform(3,8,n),rng.uniform(-.3,.3,n)])
    daily=np.empty((2,3,3,n))
    for s in range(2):
        for k in range(3):
            daily[s,k,0]=40*(k+1)**.6;daily[s,k,1]=10*(k+1)**.5;daily[s,k,2]=.1
    scale,res=scaling(daily)
    assert np.allclose(scale[:,3],.4) and np.allclose(scale[:,4],.5)
    T=np.geomspace(2,100,201);a=reduced_variate(T)
    q,u,_,_=predict_curves(h,daily[0,0],daily[1,0],scale,3.,a)
    ref=genextreme.ppf(1-1/T[:,None],c=-h[2],loc=h[0],scale=h[1])
    assert np.allclose(q,ref[None],rtol=1e-11,atol=1e-10)
    zero=score_curves(q,ref);assert np.nanmax(abs(zero))<1e-10
    shifted=score_curves(q*1.2,ref)
    assert np.allclose(shifted[:,0],np.log(1.2)) and np.allclose(shifted[:,1],np.log(1.2)) and np.max(shifted[:,2])<1e-10
    daily[1,:,0:2]*=1.15;scale,_=scaling(daily)
    q,u,_,_=predict_curves(h,daily[0,0],daily[1,0],scale,6.,a)
    fd=genextreme.ppf(1-1/T[:,None],c=-daily[1,0,2],loc=daily[1,0,0],scale=daily[1,0,1])
    ur=genextreme.cdf(fd,c=-daily[0,0,2],loc=daily[0,0,0],scale=daily[0,0,1])
    qr=genextreme.ppf(ur,c=-h[2],loc=h[0],scale=h[1])
    assert np.allclose(u,ur,atol=1e-12) and np.allclose(q[1],qr,rtol=1e-10)
    assert np.allclose(q[0],ref*1.15,rtol=1e-11)
    assert np.allclose(q[2],ref*1.15,rtol=1e-11)
    atday=predict_curves(h,daily[0,0],daily[1,0],scale,24.,a)[0]
    assert np.allclose(atday[2],atday[0],rtol=1e-11)
    print('CORE TESTS PASSED: GEV/SciPy, no-change identity, TPS exponents, HQT, D/B/S',flush=True)
