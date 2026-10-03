"""Import only the compact reviewer inputs; never copy raw archives or credentials."""
from pathlib import Path
import argparse,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True,help='02_experiments folder from the compact review package')
    args=ap.parse_args();source=args.source.expanduser().resolve();dest=ROOT/'analysis'
    if not source.is_dir():ap.error('Source folder does not exist')
    members=['01','04','07','08'];base='research_wace/submission_checks_20261002'
    names=[f'{base}/input_snapshot_{m}.npz' for m in members]
    names += [f'{base}/input_snapshot_geometry.npz',f'{base}/software_environment.json',f'{base}/input_snapshot_manifest.json',
      'research_v3/controls/synthetic_results.npz','expected_results/grid_summary.csv','expected_results/station_point_scores.csv']
    names += [f'research_v3/point_diagnostics/T_bands_{m}.csv' for m in members]
    station='midas_station_validation_7/transfer_v1'
    names += [f'{station}/{name}' for name in ['cohort.csv','early_hourly.csv','early_daily.csv','late_daily.csv',
      'heldout_late_hourly.csv','prediction_GEV_parameters.csv','evaluation_GEV_parameters.csv','config.json']]
    missing=[name for name in names if not (source/name).is_file()]
    if missing:ap.error('Required inputs missing: '+', '.join(missing))
    for filename in ['station_duration_scores.csv','group_summary.csv']:
        if (source/'expected_results'/filename).is_file():names.append('expected_results/'+filename)
    names += [p.relative_to(source).as_posix() for p in sorted((source/'expected_results/submission_checks').glob('*.csv'))]
    checks=[]
    # Check all conflicts before making changes; rerunning with identical data is safe.
    for name in names:
        src=(source/name).resolve();target=(dest/name).resolve()
        if not src.is_relative_to(source) or not target.is_relative_to(dest.resolve()):ap.error('Unexpected path outside input/output folder')
        value=digest(src)
        if target.exists() and digest(target)!=value:ap.error(f'Existing input differs: {name}; use a fresh checkout to compare packages')
        checks.append((src,target,value,name))
    for src,target,value,name in checks:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(src,target)
        assert digest(target)==value
    manifest=dest/'reviewer_output/import_manifest.json';manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(json.dumps({'files':[{'path':name,'sha256':value} for _,_,value,name in checks]},indent=2),encoding='utf-8')
    print(f'Imported and verified {len(checks)} compact inputs. Data remain excluded from Git.')
    print('Next: python reproduce.py --threads 4')

if __name__=='__main__':main()
