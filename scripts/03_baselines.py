#!/usr/bin/env python3
"""Fit a leak-safe mean-CTR baseline and write a local AUC report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from techarena.config import (  # noqa: E402
    COL_CLICKED,
    COL_ITEM_ID,
    FOLDS_PARQUET,
    REPORTS_DIR,
    TRAIN_PARQUET,
)
from techarena.features import apply_target_means, fit_target_means  # noqa: E402
from techarena.io import read_parquet  # noqa: E402
from techarena.models import pooled_roc_auc  # noqa: E402
from techarena.validation import iter_fold_splits  # noqa: E402


def main() -> None:
    if not TRAIN_PARQUET.exists():
        raise SystemExit(
            f"Missing {TRAIN_PARQUET}. Run scripts/00_csv_to_parquet.py first."
        )
    if not FOLDS_PARQUET.exists():
        raise SystemExit(
            f"Missing {FOLDS_PARQUET}. Run scripts/01_make_folds.py first."
        )

    train = read_parquet(TRAIN_PARQUET)
    folds = read_parquet(FOLDS_PARQUET)
    frame = train.join(folds, on="event_id", how="left")
    if frame["fold"].null_count() > 0:
        raise SystemExit("Some rows did not match a fold; check event_id alignment.")

    rows = []
    oof_true = []
    oof_pred = []
    for fold_id, fit, valid in iter_fold_splits(frame):
        stats = fit_target_means(fit, [COL_ITEM_ID])
        scored = apply_target_means(valid, stats, [COL_ITEM_ID])
        global_mean = float(fit[COL_CLICKED].mean())
        pred = scored[f"{COL_ITEM_ID}_ctr"].fill_null(global_mean).to_numpy()
        y = scored[COL_CLICKED].to_numpy()
        auc = pooled_roc_auc(y, pred)
        rows.append({"fold": fold_id, "n_valid": scored.height, "auc": auc})
        oof_true.append(scored[COL_CLICKED])
        oof_pred.append(pl.Series("pred", pred))

    oof_y = pl.concat(oof_true).to_numpy()
    oof_p = pl.concat(oof_pred).to_numpy()
    overall = pooled_roc_auc(oof_y, oof_p)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "baselines.md"
    lines = [
        "# Item-CTR baseline",
        "",
        "Smoothed item click-through rate estimated on the **fit** fold only.",
        "",
        f"Pooled OOF ROC AUC: **{overall:.6f}**",
        "",
        "| fold | n_valid | auc |",
        "| ---: | ------: | --: |",
    ]
    for row in rows:
        lines.append(f"| {row['fold']} | {row['n_valid']} | {row['auc']:.6f} |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(f"pooled OOF AUC: {overall:.6f}")


if __name__ == "__main__":
    main()
