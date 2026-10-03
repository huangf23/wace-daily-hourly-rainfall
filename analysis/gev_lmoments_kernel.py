"""Batch, unbiased sample L-moments and conventional-xi GEV fitting.

Shape solves the population L-skewness equation numerically, without a
per-pixel Python optimizer. SciPy's shape c is the negative of xi.
"""
import math
import numpy as np
from numba import njit, prange

@njit(cache=True)
def gev_tau(xi):
    if abs(xi)<1e-8:
        a=math.log(3.);b=math.log(2.)
        return 2*a/b*(1+(a-b)*xi/2)-3
    return 2*math.expm1(xi*math.log(3.))/math.expm1(xi*math.log(2.))-3

@njit(parallel=True,nogil=True,cache=True)
def fit_batch(samples, return_periods):
    # samples: [series, year]. Require every supplied annual observation.
    nseries,n=samples.shape
    pars=np.full((nseries,3),np.nan)  # location, scale, conventional xi
    moments=np.full((nseries,3),np.nan) # l1,l2,tau3
    levels=np.full((nseries,len(return_periods)),np.nan)
    status=np.zeros(nseries,np.int16)
    support=np.zeros(nseries,np.int16)
    for p in prange(nseries):
        x=np.sort(samples[p])
        if n<3 or not np.all(np.isfinite(x)):
            status[p]=1;continue
        b0=0.;b1=0.;b2=0.
        for i in range(n):
            b0+=x[i]/n
            b1+=x[i]*i/(n*(n-1))
            b2+=x[i]*i*(i-1)/(n*(n-1)*(n-2))
        l1=b0;l2=2*b1-b0;l3=6*b2-6*b1+b0
        if l2<=1e-12*max(abs(l1),1.):
            status[p]=2;continue
        tau=l3/l2
        moments[p,0]=l1;moments[p,1]=l2;moments[p,2]=tau
        if not (-1<tau<1):status[p]=3;continue
        lo=-1.;hi=1.
        while gev_tau(lo)>tau and lo>-128:lo*=2
        if gev_tau(lo)>tau:status[p]=3;continue
        for _ in range(48):
            mid=(lo+hi)/2
            if gev_tau(mid)<tau:lo=mid
            else:hi=mid
        xi=(lo+hi)/2
        if abs(xi)<1e-7:
            xi=0.;scale=l2/math.log(2.)
            loc=l1-0.5772156649015329*scale
        else:
            lg=math.lgamma(1-xi)
            scale=l2*xi/(math.expm1(xi*math.log(2.))*math.exp(lg))
            loc=l1-scale*math.expm1(lg)/xi
        if not (math.isfinite(loc) and math.isfinite(scale) and scale>0):
            status[p]=4;continue
        pars[p,0]=loc;pars[p,1]=scale;pars[p,2]=xi
        for i in range(n):
            if 1+xi*(x[i]-loc)/scale<=0:support[p]+=1
        for j in range(len(return_periods)):
            a=-math.log(-math.log1p(-1/return_periods[j]))
            levels[p,j]=loc+scale*(a if xi==0 else math.expm1(xi*a)/xi)
    return pars,moments,levels,status,support

def self_test():
    from scipy.stats import genextreme
    from lmoments3 import distr,lmom_ratios
    rng=np.random.default_rng(9221)
    periods=np.array([2,5,10,25,50,100],dtype=np.float64)
    samples=np.vstack([genextreme.rvs(c=-xi,loc=25,scale=8,size=(100,40),random_state=rng) for xi in [-.4,-.1,0,.1,.4]])
    pars,moments,levels,status,support=fit_batch(samples,periods)
    assert np.all(status==0)
    differences=[]
    for i,x in enumerate(samples):
        ref=distr.gev.lmom_fit(x)
        refp=np.array([ref['loc'],ref['scale'],-ref['c']])
        refl=np.array(lmom_ratios(x,nmom=3))
        assert np.allclose(moments[i],refl,rtol=1e-10,atol=1e-10)
        # Hosking rational approximations and zero-shape tolerance in lmoments3
        # differ slightly from this direct numerical equation solution.
        assert np.allclose(pars[i],refp,rtol=3e-4,atol=3e-4),(pars[i],refp)
        q=genextreme.ppf(1-1/periods,c=-pars[i,2],loc=pars[i,0],scale=pars[i,1])
        assert np.allclose(levels[i],q,rtol=1e-10,atol=1e-10)
        differences.append(np.max(abs(pars[i]-refp)))
    bad=np.zeros((2,40));bad[1,0]=np.nan
    out=fit_batch(bad,periods)
    assert np.array_equal(out[3],[2,1])
    return {'reference_series':len(samples),'maximum_parameter_absolute_difference_vs_lmoments3':float(max(differences)),
            'return_levels_verified_against':'scipy.stats.genextreme.ppf','shape_convention':'xi=-scipy_c'}

if __name__=='__main__':
    print(self_test())
