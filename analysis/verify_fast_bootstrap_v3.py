"""Check the fused evaluator on actual resamples, including regularization."""
import time
from numba import set_num_threads
from research_v3_common import *
from bootstrap_joint_v3 import fit_draws,draws
from fast_scores_v3 import fast_scores

set_num_threads(3)
geo=geometry();member='01';ids=np.arange(0,128,8);b=500
folder=OUT/'joint_bootstrap'/member
pa=np.load(folder/'allowed_parameters.npy',mmap_mode='r')
p=np.ascontiguousarray(pa[:,:,:,ids]).reshape(13,3,-1)
sc,_=scaling(p[7:].reshape(2,3,3,-1))
_,index=draws(member)
truth,_,_=fit_draws(reference_ams(member,geo['cells'][ids]),index)
errors=[];adjustments=0;clips=0
for j,d in enumerate(DURATIONS):
    q,good,adj,clip,_=prediction_curves(p[j],p[7],p[10],sc,d)
    tr=gev_curves(truth[j].reshape(3,-1),BASE_A)
    valid=good.all(axis=0)&np.all(np.isfinite(tr)&(tr>0),axis=0)
    expected=score_curves(q[:,IDS],tr[IDS]);expected[:,:,~valid]=np.nan
    actual,ok=fast_scores(p[j],p[7],p[10],sc,truth[j].reshape(3,-1),float(d))
    assert np.array_equal(valid,ok)
    error=float(np.nanmax(abs(actual-expected)));assert error<2e-12,error
    errors.append(error);adjustments+=int(adj.sum());clips+=int(clip.sum())
report=dict(status='passed',member=member,real_resampled_series=500*len(ids)*7,max_score_difference=max(errors),
            adjusted_curves_tested=adjustments,clipped_probability_nodes_tested=clips,
            check='Fused evaluator equals original GEV/epsilon/PAVA/score pipeline and common valid mask')
save_json(OUT/'fast_score_verification.json',report)
print(report,flush=True)
