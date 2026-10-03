"""Run the scientific reviewer entry point without changing its calculations."""
from pathlib import Path
import runpy,sys

def main():
    analysis=Path(__file__).resolve().parent/'analysis'
    required=['research_wace/submission_checks_20261002/input_snapshot_01.npz',
      'expected_results/grid_summary.csv','expected_results/station_point_scores.csv',
      'midas_station_validation_7/transfer_v1/cohort.csv']
    if '--help' not in sys.argv and '-h' not in sys.argv and any(not (analysis/p).is_file() for p in required):
        raise SystemExit('Reviewer data are not bundled in Git. First run:\n'
          '  python tools/prepare_review_data.py --source /path/to/02_experiments\n'
          'See docs/DATA_REQUIREMENTS.md for the data scope.')
    script=analysis/'review_reproduce.py'
    sys.path.insert(0,str(analysis));sys.argv[0]=str(script)
    runpy.run_path(str(script),run_name='__main__')

if __name__=='__main__':main()
