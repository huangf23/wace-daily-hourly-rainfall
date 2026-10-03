"""Explicit finite-probability and monotone regularization; no file access.

EPSILON is a numerical convention, never estimated from future-hourly truth.
It does not establish physical credibility of extrapolated tail estimates.
"""
import numpy as np
from numba import njit, prange
from transfer_core import (METHODS,DURATIONS,T_ENGINEERING,reduced_variate,
    scaling,quantile,gev_curves,valid_curves,score_curves)
from transfer_core import predict_curves as raw_predict

EPSILON=1e-6
# A single fixed projection grid makes outputs independent of requested T grid.
BASE_T=np.unique(np.r_[np.geomspace(2,100,401),T_ENGINEERING])
BASE_A=reduced_variate(BASE_T)
BASE_X=np.log(BASE_T)
WEIGHTS=np.r_[np.diff(BASE_X)[0]/2,
              (BASE_X[2:]-BASE_X[:-2])/2,np.diff(BASE_X)[-1]/2]
LOW_A=-np.log(-np.log(EPSILON))
HIGH_A=-np.log(-np.log1p(-EPSILON))
LAST_DIAGNOSTICS={}

@njit(parallel=True,cache=True,nogil=True)
def bounded_hqt(h,logtail,q,u):
    flags=np.zeros(logtail.shape,np.uint8)
    for g in prange(h.shape[1]):
        for j in range(logtail.shape[0]):
            original=-logtail[j,g]
            a=min(HIGH_A,max(LOW_A,original))
            flags[j,g]=original!=a
            logtail[j,g]=-a
            u[j,g]=np.exp(-np.exp(-a))
            q[1,j,g]=quantile(h[0,g],h[1,g],h[2,g],a)
    return flags

@njit(parallel=True,cache=True,nogil=True)
def project_log_curves(q,weights):
    out=q.copy();changed=np.zeros((q.shape[0],q.shape[2]),np.uint8)
    for g in prange(q.shape[2]):
        for m in range(q.shape[0]):
            # Leave already monotone curves exactly unchanged.
            if np.all(q[m,1:,g]>=q[m,:-1,g]):continue
            means=np.empty(q.shape[1]);mass=np.empty(q.shape[1]);ends=np.empty(q.shape[1],np.int64)
            k=0
            for j in range(q.shape[1]):
                means[k]=np.log(q[m,j,g]);mass[k]=weights[j];ends[k]=j;k+=1
                while k>1 and means[k-2]>means[k-1]:
                    total=mass[k-2]+mass[k-1]
                    means[k-2]=(mass[k-2]*means[k-2]+mass[k-1]*means[k-1])/total
                    mass[k-2]=total;ends[k-2]=ends[k-1];k-=1
            start=0
            for b in range(k):
                for j in range(start,ends[b]+1):out[m,j,g]=np.exp(means[b])
                start=ends[b]+1
            changed[m,g]=1
    return out,changed

def calculate_base(h,hd,fd,scale,t):
    q,u,lt,boundary=raw_predict(h,hd,fd,scale,t,BASE_A)
    clipped=bounded_hqt(h,lt,q,u)
    # Invalid inputs must never be concealed by zero filling or a method swap.
    assert np.all(np.isfinite(q)&(q>0)), 'Nonpositive/nonfinite curve remains after probability regularization'
    q,adjusted=project_log_curves(q,WEIGHTS)
    assert np.all(valid_curves(q)[2])
    return q,u,lt,boundary,clipped,adjusted

def predict_curves(h,hd,fd,scale,t,a):
    # All production queries are exact nodes of the shared projection grid.
    ids=np.abs(BASE_A[:,None]-np.asarray(a)[None,:]).argmin(axis=0)
    assert np.allclose(BASE_A[ids],a,rtol=0,atol=1e-12),'Unsupported projection-grid query'
    q,u,lt,b,clipped,adjusted=calculate_base(h,hd,fd,scale,t)
    LAST_DIAGNOSTICS.update(clipped=clipped[ids],adjusted=adjusted)
    return q[:,ids],u[ids],lt[ids],b[ids]

def tests():
    from scipy.optimize import isotonic_regression
    from scipy.stats import genextreme
    from transfer_core import tests as original_tests
    original_tests()
    rng=np.random.default_rng(52)
    q=np.exp(rng.normal(size=(3,len(BASE_A),6)))
    fitted,_=project_log_curves(q,WEIGHTS)
    for m in range(3):
        for g in range(6):
            ref=isotonic_regression(np.log(q[m,:,g]),weights=WEIGHTS).x
            assert np.allclose(np.log(fitted[m,:,g]),ref,atol=1e-12)
    h=np.array([[10.,20.],[3.,4.],[.1,-.1]])
    lt=np.array([[-np.inf,np.inf],[-20.,-1.]])
    out=np.zeros((3,2,2));u=np.empty((2,2));flags=bounded_hqt(h,lt,out,u)
    assert np.isfinite(out).all() and np.isfinite(lt).all()
    assert np.allclose(out[1],genextreme.ppf(u,c=-h[2],loc=h[0],scale=h[1]),rtol=1e-9)
    assert flags.sum()==3
    assert np.allclose(project_log_curves(fitted,WEIGHTS)[0],fitted)
    print('REGULARIZATION TESTS PASSED: weighted PAVA/SciPy, finite endpoint mapping, idempotence',flush=True)
