"""Independent scan of serialized numerical results and correction flags."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc

ROOT=Path(__file__).resolve().parent/'transfer_experiment_v2_regularized'

def main():
    checks=[];corrections=[]
    for member in ['01','04','07','08']:
        for kind in ['predictions','evaluation']:
            path=ROOT/f'{kind}_{member}.nc'
            with nc.Dataset(path) as ds:
                ds.set_auto_mask(False)
                for name,v in ds.variables.items():
                    if v.dtype is str or np.dtype(v.dtype).kind!='f':continue
                    count=0;bad=0;low=np.inf;high=-np.inf
                    for i in (range(v.shape[0]) if v.ndim else [None]):
                        a=np.asarray(v[i] if i is not None else v[...])
                        count+=a.size;bad+=int((~np.isfinite(a)).sum())
                        low=min(low,float(a.min()));high=max(high,float(a.max()))
                    checks.append(dict(file=path.name,variable=name,count=count,nonfinite=bad,min=low,max=high))
                    assert bad==0,(path.name,name,bad)
                if kind=='predictions':
                    clipped=ds['hqt_probability_clipped'][:].reshape(201,-1).astype(bool)
                    adjusted=ds['monotonicity_adjusted'][:].reshape(3,7,-1).astype(bool)
                    corrections.append(dict(member=member,hqt_clipped_grid_cells=int(clipped.any(axis=0).sum()),
                        hqt_clipped_probability_grid_pairs=int(clipped.sum()),
                        monotonicity_adjusted_duration_grid_pairs=adjusted.sum(axis=(1,2)).tolist()))
                else:
                    assert np.all(ds['common_valid'][:]==1)
    frame=pd.read_csv(ROOT/'evaluation_summary.csv')
    rows=[]
    for (domain,method),g in frame.groupby(['domain','method'],sort=False):
        n=int(g.common_valid_cells.sum())
        rows.append(dict(domain=domain,method=method,common_valid_pairs=n,
            mean_D_RL=float(np.average(g.mean_D_RL,weights=g.common_valid_cells)),
            best_percent=float(100*g.best_method_cells.sum()/n),
            integration_diff_over_1e4=int(g.integration_diff_over_1e4.sum())))
    (ROOT/'overall_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report=dict(status='passed',nonfinite_stored_float_values=sum(r['nonfinite'] for r in checks),
        stored_float_values_checked=sum(r['count'] for r in checks),corrections=corrections,variables=checks)
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',values_checked=report['stored_float_values_checked'],nonfinite=0,corrections=corrections,summary=rows)),flush=True)

if __name__=='__main__':main()
