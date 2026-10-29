#!/usr/bin/env python3
"""Build time-based event folds from train.parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techarena.config import FOLDS_PARQUET, TRAIN_PARQUET  # noqa: E402
from techarena.io import read_parquet  # noqa: E402
from techarena.validation import assign_time_folds, write_folds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-parquet", type=Path, default=TRAIN_PARQUET)
    parser.add_argument("--out", type=Path, default=FOLDS_PARQUET)
    args = parser.parse_args()

    if not args.train_parquet.exists():
        raise SystemExit(
            f"Missing {args.train_parquet}. Run scripts/00_csv_to_parquet.py first."
        )

    train = read_parquet(args.train_parquet)
    folded = assign_time_folds(train, n_folds=args.n_folds)
    out = write_folds(folded, args.out)
    summary = (
        folded.select("fold")
        .group_by("fold")
        .len()
        .sort("fold")
    )
    print(f"wrote {out}")
    print(summary)


if __name__ == "__main__":
    main()
