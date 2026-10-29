#!/usr/bin/env python3
"""Lightweight EDA on the Parquet train table; writes reports/eda.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from techarena.config import (  # noqa: E402
    COL_CLICKED,
    COL_EVENT_ID,
    COL_ITEM_ID,
    COL_USER_ID,
    REPORTS_DIR,
    TRAIN_PARQUET,
)
from techarena.io import scan_parquet  # noqa: E402


def main() -> None:
    if not TRAIN_PARQUET.exists():
        raise SystemExit(
            f"Missing {TRAIN_PARQUET}. Run scripts/00_csv_to_parquet.py first."
        )

    lf = scan_parquet(TRAIN_PARQUET)
    schema = lf.collect_schema()
    aggs: list[pl.Expr] = [pl.len().alias("n_rows")]
    if COL_CLICKED in schema:
        aggs.append(pl.col(COL_CLICKED).mean().alias("click_rate"))
    if COL_EVENT_ID in schema:
        aggs.append(pl.col(COL_EVENT_ID).n_unique().alias("n_events"))
    if COL_USER_ID in schema:
        aggs.append(pl.col(COL_USER_ID).n_unique().alias("n_users"))
    if COL_ITEM_ID in schema:
        aggs.append(pl.col(COL_ITEM_ID).n_unique().alias("n_items"))
    stats = lf.select(aggs).collect().to_dicts()[0]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "eda.md"
    lines = [
        "# EDA snapshot",
        "",
        f"- rows: `{stats.get('n_rows')}`",
        f"- unique events: `{stats.get('n_events')}`",
        f"- unique users: `{stats.get('n_users')}`",
        f"- unique items: `{stats.get('n_items')}`",
        f"- click rate: `{stats.get('click_rate')}`",
        f"- columns: `{', '.join(schema.names())}`",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
