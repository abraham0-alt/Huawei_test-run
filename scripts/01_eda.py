#!/usr/bin/env python3
"""Full competition EDA. Writes reports/01_eda.md, figures, and small tables.

Usage:
    python scripts/01_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from techarena.config import REPORTS_DIR, resolve_competition_paths  # noqa: E402
from techarena.eda import SizeTracker, run_eda  # noqa: E402


def main() -> None:
    paths = resolve_competition_paths()
    print("Using parquet files:")
    for key, path in paths.items():
        print(f"  {key}: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    train = pl.scan_parquet(paths["train"])
    eval_lf = pl.scan_parquet(paths["eval"])
    folds = {
        "fold_2025_01_25": pl.scan_parquet(paths["fold_2025_01_25"]),
        "fold_2025_01_26": pl.scan_parquet(paths["fold_2025_01_26"]),
    }
    run_eda(train, eval_lf, folds, REPORTS_DIR, tracker=SizeTracker())


if __name__ == "__main__":
    main()
