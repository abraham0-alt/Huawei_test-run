# TechArena 2026 Phase 1 — click-probability prediction

Predict the probability that a user clicks a recommended item in an impression list. Each event exposes 10 candidate items; the model scores every (event, item) row. The official metric is **ROC AUC, pooled over all rows** (not averaged per event).

## Evaluation metric

**ROC AUC** on the `clicked` label, computed on the concatenated set of all scored rows. Local validation should use the same pooled definition (`sklearn.metrics.roc_auc_score` over every row).

## Folder structure

| Path | Role |
| --- | --- |
| `data/raw/` | Original competition CSVs (git-ignored). |
| `data/parquet/` | Converted Parquet tables (git-ignored). |
| `data/folds/` | Time-based event fold maps (git-ignored). |
| `data/features/` | Generated feature tables (git-ignored). |
| `notebooks/` | Exploratory Jupyter notebooks. |
| `src/techarena/` | Importable package: paths, I/O, folds, features, models. |
| `scripts/` | Numbered CLI steps (`00` … `03`). |
| `reports/` | Small EDA outputs (markdown, figures, tiny CSVs) — **committed**. |
| `submissions/` | Generated submission CSVs (git-ignored). |
| `models/` | Saved model files (git-ignored). |
| `tests/` | Pytest suite, including a submission validator. |

Large files (`*.csv`, `*.parquet`, `*.pkl`, `*.joblib`, `*.lgb`, `*.cbm`, `*.npy`, `*.npz`) are ignored globally, with exceptions for `reports/` and `tests/fixtures/`.

## Setup

```bash
cd /path/to/Huawei_test-run
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

Run tests once the environment is ready:

```bash
pytest
```

## Data placement, Parquet conversion, and folds

1. Copy the competition CSVs into `data/raw/` (typically `train.csv` and `test.csv`). Train is ~7 GB as CSV; do **not** commit it.
2. Convert to Parquet (streaming via Polars):

   ```bash
   python scripts/00_csv_to_parquet.py
   ```

   Outputs land in `data/parquet/` (`train.parquet`, `test.parquet`, plus any other `*.csv` in `data/raw/`).
3. Build time-based event folds from train:

   ```bash
   python scripts/01_make_folds.py --n-folds 5
   ```

   Writes `data/folds/folds.parquet` (`event_id`, `fold`, `event_time`). All 10 impression rows of an event share one fold.

If Parquet or fold files already exist on disk, leave them where they are. This repo ignores those paths so Git will not pick them up.

## Run scripts in order

| Step | Command | Writes |
| --- | --- | --- |
| 0 | `python scripts/00_csv_to_parquet.py` | `data/parquet/*.parquet` |
| 1 | `python scripts/01_make_folds.py` | `data/folds/folds.parquet` |
| 2 | `python scripts/02_eda.py` | `reports/eda.md` |
| 3 | `python scripts/03_baselines.py` | `reports/baselines.md` (item-CTR OOF AUC) |

Later experiments should continue the numbering (`04_…`, `05_…`).

## Validation philosophy

- **Time-based splits.** Events are ordered by their earliest `timestamp`; fold 0 is the earliest slice and later folds are the future. Do not shuffle rows independently of time.
- **Events stay together.** An impression list has 10 rows that share `event_id`. They are assigned the same fold so the model never trains on one candidate from an event and validates on another.
- **No target leakage.** Label-based aggregates (item CTR, user CTR, counts, embeddings of clicked history, etc.) are estimated on the **fit** partition only, then joined onto the apply partition. Global fill values also come from fit data only.
- **Metric matches the leaderboard.** Report pooled ROC AUC over all scored rows, not a mean of per-event AUCs.

## Results log

| date | experiment | features | model | local AUC | public AUC |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
