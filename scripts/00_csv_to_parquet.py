#!/usr/bin/env python3
"""Convert CSVs in data/raw/ to Parquet in data/parquet/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techarena.config import PARQUET_DIR, RAW_DIR  # noqa: E402
from techarena.io import convert_raw_csvs  # noqa: E402


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise SystemExit(
            f"No CSV files found in {RAW_DIR}. "
            "Copy train.csv / test.csv (and any other tables) there first."
        )
    written = convert_raw_csvs(RAW_DIR, PARQUET_DIR)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
