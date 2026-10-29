"""Time-based validation that keeps all impression rows of an event together."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from techarena.config import COL_EVENT_ID, COL_TIMESTAMP, FOLDS_DIR, RANDOM_SEED


def event_time_keys(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per event, using the event's earliest timestamp as its time."""
    if COL_EVENT_ID not in frame.columns or COL_TIMESTAMP not in frame.columns:
        raise ValueError("frame must contain event_id and timestamp")
    return (
        frame.group_by(COL_EVENT_ID)
        .agg(pl.col(COL_TIMESTAMP).min().alias("event_time"))
        .sort("event_time")
    )


def assign_time_folds(
    frame: pl.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """Assign fold ids by event time quantiles.

    All 10 impression rows that share an event_id receive the same fold.
    Fold 0 is the earliest time slice; fold n_folds-1 is the latest.
    ``seed`` is accepted for API stability; the split itself is deterministic
    given timestamps.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    _ = seed

    keys = event_time_keys(frame)
    n_events = keys.height
    if n_events < n_folds:
        raise ValueError(f"need at least {n_folds} events, found {n_events}")

    ranked = keys.with_columns(
        pl.col("event_time").rank("ordinal").alias("_rank"),
    )
    # Map rank in [1, n] to fold in [0, n_folds-1] without splitting an event.
    fold_of_event = ranked.with_columns(
        (((pl.col("_rank") - 1) * n_folds) // n_events).cast(pl.Int32).alias("fold")
    ).select(COL_EVENT_ID, "fold", "event_time")

    return frame.join(fold_of_event, on=COL_EVENT_ID, how="left")


def write_folds(frame: pl.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else FOLDS_DIR / "folds.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in (COL_EVENT_ID, "fold", "event_time") if c in frame.columns]
    if COL_EVENT_ID not in keep or "fold" not in keep:
        raise ValueError("frame must contain event_id and fold")
    (
        frame.select(keep)
        .unique(subset=[COL_EVENT_ID])
        .sort("fold", COL_EVENT_ID)
        .write_parquet(path)
    )
    return path


def iter_fold_splits(
    frame: pl.DataFrame,
    fold_col: str = "fold",
):
    """Yield (fold_id, train_df, valid_df). Valid is that fold; train is the rest.

    Prefer using earlier folds as train when you want a strict future-holdout:
    pass a frame already filtered so that train times precede valid times.
    """
    if fold_col not in frame.columns:
        raise ValueError(f"missing fold column: {fold_col}")
    folds = sorted(frame[fold_col].unique().to_list())
    for fold_id in folds:
        valid = frame.filter(pl.col(fold_col) == fold_id)
        train = frame.filter(pl.col(fold_col) != fold_id)
        yield fold_id, train, valid
