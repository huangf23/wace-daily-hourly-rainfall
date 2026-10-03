# Daily-to-hourly design rainfall

Research code accompanying **Updating hourly design rainfall from daily projections:
How much complexity is needed?**

The study compares an unchanged historical baseline (NC), uniform change-factor
transfer (UCF), historical quantile transfer (HQT), and temporal parameter scaling
(TPS). UKCP18 members are fitted separately at each grid cell. Historical hourly
and historical/target whole-day extremes supply the predictors; target hourly
extremes are used only for evaluation.

## Repository layout

| Location | Contents |
|---|---|
| `analysis/` | Original AMS, L-moment GEV, transfer, uncertainty and plotting scripts |
| `analysis/script_groups.json` | Scripts grouped by experiment |
| `tools/prepare_review_data.py` | Import compact reviewer inputs from a local review package |
| `reproduce.py` | Recalculate and verify the principal grid and station comparisons |
| `docs/` | Data requirements, experiment map and reproduction scope |

This repository contains code and documentation. Rainfall data, model parameter
snapshots, manuscript files and generated results are not committed.

## Quick start

Python 3.12 is the recorded study environment. Create a virtual environment and
install the pinned numerical dependencies:

```sh
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.txt
```

Import the `02_experiments` folder from the author's compact review package:

```sh
python tools/prepare_review_data.py --source /path/to/review_package/02_experiments
python reproduce.py --threads 4
```

The data importer uses an explicit file list and copies only the compact inputs
and expected comparison tables. Data and outputs are excluded by `.gitignore`.
No automatic download or cloud service is required. See
[data requirements](docs/DATA_REQUIREMENTS.md) for input shapes and units.

The reviewer computation covers all four members, 10,397 land cells and seven
hourly durations. It regenerates predictions from supplied grid GEV fits and
refits the supplied station AMS before transfer and evaluation. Outputs appear
in `analysis/reviewer_output/`, including a machine-readable verification report.
First execution includes Numba compilation time.

## Other experiments

From `analysis/`, the supplied compact inputs also support:

```sh
python complete_submission_checks_wace.py design
python complete_submission_checks_wace.py grid
python complete_submission_checks_wace.py coverage --workers 4
python draw_submission_checks_wace.py
```

These commands reproduce HQT probability-bound sensitivity, return-period bands,
input-defined groups and synthetic bootstrap coverage. The full coverage run is
substantially slower: 500 outer replicates and 499 inner resamples per configuration.

For raw extraction, complete grid bootstrap or other original scripts, install
`requirements-optional.txt` and supply the larger inputs listed in the data guide.
The original scripts retain their historical path settings and filenames. They
are grouped in one directory to preserve local imports; helper modules with older
names are still dependencies of the current methods. Inspect their arguments and
path constants before running the full study pipeline.

## Scope and verification

- Historical model period: 1981–2020; target: 2041–2080; December–November statistical years on a 360-day calendar.
- Hourly targets: 1, 2, 3, 4, 5, 6 and 12 h. Daily predictors: midnight-aligned 1, 2 and 3 days.
- Four members: 01, 04, 07 and 08. Results are not fitted to pooled members.
- Station analyses distinguish six primary stations from one exploratory station.
- The portable reviewer entry point checks 112 grid-summary rows and 196 station-duration-method rows against the saved results.

See [reproduction scope](docs/REPRODUCIBILITY.md) and the
[Chinese experiment map](docs/EXPERIMENT_MAP.zh-CN.md). The full raw-hourly extraction
and all bootstrap experiments are not rerun by the quick-start command.

## Data and use

UKCP18 and MIDAS source-data conditions remain with their respective providers.
This repository does not redistribute those archives or grant rights to them.
It currently supports author and reviewer access; no open-source licence has yet
been selected by the authors. Manuscript submission materials are maintained separately.
