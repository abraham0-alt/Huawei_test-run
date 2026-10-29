"""CSV/Parquet I/O and submission checks."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from techarena.config import (
    COL_CLICKED,
    COL_ROWID,
    PARQUET_DIR,
    RAW_DIR,
    SUBMISSION_COLUMNS,
)


def csv_to_parquet(
    csv_path: Path,
    parquet_path: Path | None = None,
    *,
    compression: str = "zstd",
) -> Path:
    """Convert a CSV to Parquet without loading the full file into pandas."""
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = PARQUET_DIR / f"{csv_path.stem}.parquet"
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_csv(csv_path, infer_schema_length=10_000)
        .sink_parquet(parquet_path, compression=compression)
    )
    return parquet_path


def convert_raw_csvs(raw_dir: Path = RAW_DIR, parquet_dir: Path = PARQUET_DIR) -> list[Path]:
    """Convert every CSV in data/raw/ into data/parquet/."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for csv_path in sorted(Path(raw_dir).glob("*.csv")):
        written.append(csv_to_parquet(csv_path, parquet_dir / f"{csv_path.stem}.parquet"))
    return written


def scan_parquet(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_submission(frame: pl.DataFrame, path: Path) -> Path:
    """Write a two-column submission (rowid, clicked)."""
    missing = [c for c in SUBMISSION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"submission is missing columns: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.select(list(SUBMISSION_COLUMNS))
    validate_submission_frame(out)
    out.write_csv(path)
    return path


def validate_submission_frame(
    frame: pl.DataFrame,
    *,
    expected_rowids: pl.Series | None = None,
) -> None:
    """Raise ValueError if a submission table is malformed."""
    missing = [c for c in SUBMISSION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"submission is missing columns: {missing}")

    extra = [c for c in frame.columns if c not in SUBMISSION_COLUMNS]
    if extra:
        raise ValueError(f"submission has unexpected columns: {extra}")

    if frame.height == 0:
        raise ValueError("submission is empty")

    if frame[COL_ROWID].null_count() > 0:
        raise ValueError("submission rowid contains nulls")
    if frame[COL_CLICKED].null_count() > 0:
        raise ValueError("submission clicked contains nulls")

    n_unique = frame[COL_ROWID].n_unique()
    if n_unique != frame.height:
        raise ValueError("submission rowid values are not unique")

    clicked = frame[COL_CLICKED]
    if clicked.dtype not in (
        pl.Float32,
        pl.Float64,
        pl.Decimal,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        raise ValueError(f"submission clicked has unsupported dtype: {clicked.dtype}")

    min_v = clicked.min()
    max_v = clicked.max()
    if min_v is None or max_v is None or min_v < 0 or max_v > 1:
        raise ValueError("submission clicked values must be in [0, 1]")

    if expected_rowids is not None:
        expected = expected_rowids.unique().sort()
        got = frame[COL_ROWID].unique().sort()
        if expected.len() != got.len() or not expected.equals(got):
            raise ValueError("submission rowids do not match the expected set")
