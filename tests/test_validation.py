"""Event-grouped time folds."""

from __future__ import annotations

import polars as pl

from techarena.validation import assign_time_folds


def test_all_rows_of_an_event_share_a_fold() -> None:
    frame = pl.DataFrame(
        {
            "event_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "timestamp": [10, 10, 20, 20, 30, 30, 40, 40],
            "clicked": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    folded = assign_time_folds(frame, n_folds=2)
    per_event = folded.group_by("event_id").agg(pl.col("fold").n_unique().alias("n"))
    assert per_event["n"].max() == 1
    assert set(folded["fold"].unique().to_list()) == {0, 1}


def test_earlier_events_get_lower_fold_ids() -> None:
    frame = pl.DataFrame(
        {
            "event_id": [10, 20, 30, 40],
            "timestamp": [100, 200, 300, 400],
        }
    )
    folded = assign_time_folds(frame, n_folds=2).sort("timestamp")
    assert folded["fold"].to_list() == [0, 0, 1, 1]
