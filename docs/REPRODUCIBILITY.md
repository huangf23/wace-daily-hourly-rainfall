# Reproduction scope

The compact-review validation performed on 2026-10-03 recalculated the complete
four-member grid summary from fitted GEV parameters. All 112 rows agreed with
the study's tables to a maximum absolute difference of 9.72e-17. Refitting the
supplied station AMS and recalculating transfer scores checked 196 rows, with a
maximum absolute difference of 1.12e-16. Floating-point differences may vary by platform.

The repository preserves 41 original scripts and the portable reviewer entry
point. Original code checksums are recorded in `analysis/original_code_checksums.json`.
The root wrapper and local data importer are packaging additions, not new methods.

The quick-start command does not independently validate raw archive completeness,
rerun raw-hourly extraction, reproduce all figure layouts, or rerun the complete
grid bootstrap and coverage simulation. Those original scripts are retained for
review and use with their documented external inputs. Saved outputs are not
substituted for new calculations in the numerical reviewer entry point.

Do not use target-hourly reference parameters to select predictor groups,
regularization bounds or transfer coefficients. The reviewer entry point constructs
predictions before accessing the corresponding target-hourly evaluation inputs.
