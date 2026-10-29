"""Leak-safe feature helpers.

Label-based aggregates (CTR, click counts, etc.) must be computed on the
**fit** partition only, then joined onto the apply partition. Never compute
target statistics on the full train+valid table.
"""

from __future__ import annotations

import polars as pl

from techarena.config import COL_CLICKED, COL_ITEM_ID, COL_USER_ID


def fit_target_means(
    train: pl.DataFrame,
    group_cols: list[str],
    *,
    target: str = COL_CLICKED,
    suffix: str = "_ctr",
    smoothing: float = 20.0,
) -> pl.DataFrame:
    """Smoothed mean target per group, estimated on ``train`` only."""
    global_mean = float(train[target].mean())
    stats = train.group_by(group_cols).agg(
        pl.col(target).sum().alias("_sum"),
        pl.len().alias("_n"),
    )
    name = "_".join(group_cols) + suffix
    return stats.with_columns(
        ((pl.col("_sum") + smoothing * global_mean) / (pl.col("_n") + smoothing)).alias(name)
    ).drop("_sum", "_n")


def apply_target_means(
    frame: pl.DataFrame,
    group_stats: pl.DataFrame,
    group_cols: list[str],
) -> pl.DataFrame:
    return frame.join(group_stats, on=group_cols, how="left")


def basic_context_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Cheap, non-leaky columns derived from a single row."""
    out = frame
    if COL_USER_ID in out.columns:
        out = out.with_columns(pl.col(COL_USER_ID).cast(pl.Utf8).alias("user_id_str"))
    if COL_ITEM_ID in out.columns:
        out = out.with_columns(pl.col(COL_ITEM_ID).cast(pl.Utf8).alias("item_id_str"))
    return out
