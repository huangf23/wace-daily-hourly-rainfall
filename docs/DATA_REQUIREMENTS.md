# Reviewer-package inputs and external data

The following compact inputs are supplied separately in the review package, not in Git. Import them with tools/prepare_review_data.py; paths below are relative to analysis/.

## Compact review inputs: no raw archive required

- research_wace/submission_checks_20261002/input_snapshot_01/04/07/08.npz:
  h = historical hourly GEV (7,3,10397); daily = historical/target daily GEV
  (2,3,3,10397); scale = TPS coefficients (2,5,10397); historical_midpoints =
  hourly AMS event midpoints in 360-day calendar hours; evaluation_reference_parameters
  = target hourly GEV used exclusively for evaluation. Parameters are location,
  scale, xi (the opposite sign of SciPy genextreme c); rainfall depths in mm.
- input_snapshot_geometry.npz: land-cell indices and native/geographic coordinates.
- research_v3/controls/synthetic_results.npz: known synthetic population parameters.
- midas_station_validation_7/transfer_v1: station AMS tables, fitted parameters,
  cohort definition and settings. The late hourly table is evaluation-only.
- expected_results: original saved values for numerical comparison.

## Required only for the full original pipeline

Place these folders next to the original scripts, matching their study-relative paths:

- source/ukcp18_pr: original UKCP18 hourly files, four members 01/04/07/08.
- ukcp18_ams_dec_nov: extracted 100-year hourly moving-window AMS.
- ukcp18_gev_lmoments_40yr: AMS/GEV NetCDF for 1981–2020 and 2041–2080.
- ukcp18_historical_calendar_day_ams_gev and ukcp18_future_calendar_day_ams_gev:
  whole-day AMS and GEV files for 1/2/3 days.
- transfer_experiment_v2_regularized: full predictions and evaluation NetCDF.
- research_v3: full diagnostics, controls, bootstrap outputs and auxiliary data.
- paper_figures_by_member/cases.json: historical case selection.
- MIDAS original hourly files and original station AMS/quality-control inputs:
  adjust the SOURCE path in the station extraction scripts to your local archive.

The original extraction scripts retain their original machine-specific settings.
Inspect their path constants and CLI options before a raw-data rerun. No absolute
source path is needed by review_reproduce.py or the snapshot-based extension.
The compact package does not independently rerun original AMS extraction or the
full bootstrap calculation. Its validation report describes exactly what was run.
