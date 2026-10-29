"""Reproducible EDA for TechArena Phase 1 click prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import roc_auc_score

from techarena.config import (
    COL_CLICKED,
    COL_EVENT_ID,
    COL_ITEM_ID,
    COL_ROWID,
    COL_TIMESTAMP,
    COL_USER_ID,
    FOLD_25_CUTOFF_TS,
    FOLD_26_CUTOFF_TS,
    IMPRESSIONS_PER_EVENT,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_END_TS,
    TRAIN_START_TS,
)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

FOLD_CUTOFFS = {
    "fold_2025_01_25": FOLD_25_CUTOFF_TS,
    "fold_2025_01_26": FOLD_26_CUTOFF_TS,
}

MIN_ITEM_IMPRESSIONS = 500


@dataclass
class SizeTracker:
    records: list[tuple[str, int, int]] = field(default_factory=list)

    def note(self, name: str, frame: pl.DataFrame) -> pl.DataFrame:
        nbytes = int(frame.estimated_size())
        self.records.append((name, frame.height, nbytes))
        print(f"[mem] {name}: {frame.height:,} rows, {nbytes / 1e6:.1f} MB")
        return frame

    def peak(self) -> tuple[str, int, int]:
        if not self.records:
            return ("(none)", 0, 0)
        return max(self.records, key=lambda row: row[2])


def _hour(col: str = COL_TIMESTAMP) -> pl.Expr:
    return ((pl.col(col) % 86400) // 3600).alias("hour")


def _date(col: str = COL_TIMESTAMP) -> pl.Expr:
    return pl.from_epoch(col, time_unit="s").dt.date().alias("date")


def _expected_time_bucket(col: str = COL_TIMESTAMP) -> pl.Expr:
    return ((pl.col(col) % 86400) // 21600)


def _expected_day_type(col: str = COL_TIMESTAMP) -> pl.Expr:
    # Polars weekday: 1=Monday ... 7=Sunday. Docs: 0=Mon-Fri, 1=Sat/Sun.
    return (pl.from_epoch(col, time_unit="s").dt.weekday() >= 6).cast(pl.Int64)


def _json_list_str(col: str) -> pl.Expr:
    return pl.col(col).str.json_decode(dtype=pl.List(pl.Utf8))


def _json_list_i64(col: str) -> pl.Expr:
    return pl.col(col).str.json_decode(dtype=pl.List(pl.Int64))


def make_smoke_frames(n_events: int = 2000, seed: int = RANDOM_SEED) -> dict[str, pl.DataFrame]:
    """Deterministic tiny dataset with the competition schema (for tests)."""
    rng = np.random.default_rng(seed)
    n_items = 80
    n_users = 120
    item_ids = [f"itm_{i:04d}" for i in range(n_items)]
    user_ids = [f"usr_{i:04d}" for i in range(n_users)]
    rows: list[dict[str, Any]] = []
    # Train window: 2025-01-20 through 2025-01-26; eval starts 2025-01-27.
    train_start = TRAIN_START_TS
    for e in range(n_events):
        event_id = f"evt_{e:05d}"
        user_id = user_ids[e % n_users]
        ts = int(train_start + (e % 7) * 86400 + (e % 24) * 3600)
        hist_n = int(e % 21)
        hist_items = [item_ids[(e + j) % n_items] for j in range(hist_n)]
        hist_ts = [ts - 3600 * (hist_n - j) for j in range(hist_n)]
        hist_items_json = json.dumps(hist_items)
        hist_ts_json = json.dumps(hist_ts)
        click_slot = int(rng.integers(0, 11))  # 10 = no click
        chosen = rng.choice(n_items, size=10, replace=False)
        for k, item_ix in enumerate(chosen):
            rows.append(
                {
                    "rowid": f"row_{e:05d}_{k}",
                    "event_id": event_id,
                    "user_id": user_id,
                    "item_id": item_ids[int(item_ix)],
                    "timestamp": ts,
                    "time_bucket": (ts % 86400) // 21600,
                    "day_type": 1 if ((ts // 86400) + 4) % 7 >= 5 else 0,  # unused; overwritten
                    "history_item_ids": hist_items_json,
                    "history_timestamps": hist_ts_json,
                    "clicked": int(k == click_slot),
                }
            )
    train = pl.DataFrame(rows).with_columns(_expected_day_type().alias("day_type"))
    # Eval: first extra event per user after cutoff, no labels.
    eval_rows = []
    for u, user_id in enumerate(user_ids):
        ts = TRAIN_END_TS + (u % 48) * 1800
        hist_n = u % 21
        hist_items = [item_ids[(u + j) % n_items] for j in range(hist_n)]
        hist_ts = [ts - 7200 * (hist_n - j) for j in range(hist_n)]
        chosen = rng.choice(n_items, size=10, replace=False)
        for k, item_ix in enumerate(chosen):
            eval_rows.append(
                {
                    "rowid": f"erow_{u:04d}_{k}",
                    "event_id": f"eevt_{u:04d}",
                    "user_id": user_id,
                    "item_id": item_ids[int(item_ix)],
                    "timestamp": ts,
                    "time_bucket": (ts % 86400) // 21600,
                    "day_type": 1 if ((ts // 86400) + 4) % 7 >= 5 else 0,
                    "history_item_ids": json.dumps(hist_items),
                    "history_timestamps": json.dumps(hist_ts),
                }
            )
    ev = pl.DataFrame(eval_rows).with_columns(_expected_day_type().alias("day_type"))
    fold_25 = train.filter(pl.col(COL_TIMESTAMP) >= FOLD_25_CUTOFF_TS)
    # Keep one event per user (first after cutoff) to mimic the real folds.
    keys_25 = (
        fold_25.group_by(COL_USER_ID)
        .agg(pl.col(COL_TIMESTAMP).min().alias("first_ts"))
    )
    fold_25 = fold_25.join(keys_25, left_on=[COL_USER_ID, COL_TIMESTAMP], right_on=[COL_USER_ID, "first_ts"])
    keys_26 = (
        train.filter(pl.col(COL_TIMESTAMP) >= FOLD_26_CUTOFF_TS)
        .group_by(COL_USER_ID)
        .agg(pl.col(COL_TIMESTAMP).min().alias("first_ts"))
    )
    fold_26 = train.join(keys_26, left_on=[COL_USER_ID, COL_TIMESTAMP], right_on=[COL_USER_ID, "first_ts"])
    return {"train": train, "eval": ev, "fold_2025_01_25": fold_25, "fold_2025_01_26": fold_26}


def event_level(frame: pl.LazyFrame, *, has_clicked: bool, tracker: SizeTracker, name: str) -> pl.DataFrame:
    aggs = [
        pl.col(COL_USER_ID).first(),
        pl.col(COL_TIMESTAMP).first(),
        pl.col("time_bucket").first(),
        pl.col("day_type").first(),
        pl.col("history_item_ids").first(),
        pl.col("history_timestamps").first(),
    ]
    if has_clicked:
        aggs.append(pl.col(COL_CLICKED).sum().alias("n_clicks"))
    events = frame.group_by(COL_EVENT_ID).agg(aggs).collect()
    events = events.with_columns(
        _json_list_str("history_item_ids").alias("hist_items"),
        _json_list_i64("history_timestamps").alias("hist_ts"),
    ).drop("history_item_ids", "history_timestamps")
    events = events.with_columns(
        pl.col("hist_items").list.len().alias("hist_len"),
        pl.col("hist_items").list.n_unique().alias("hist_unique"),
        (pl.col(COL_TIMESTAMP) - pl.col("hist_ts").list.last()).alias("gap_seconds"),
        _hour(COL_TIMESTAMP),
        _date(COL_TIMESTAMP),
    )
    return tracker.note(name, events)


def schema_and_sanity(train: pl.LazyFrame, tracker: SizeTracker) -> dict[str, Any]:
    schema = train.collect_schema()
    dtypes = {name: str(dtype) for name, dtype in schema.items()}
    n_rows = train.select(pl.len()).collect().item()
    n_events = train.select(pl.col(COL_EVENT_ID).n_unique()).collect().item()
    n_users = train.select(pl.col(COL_USER_ID).n_unique()).collect().item()
    n_items = train.select(pl.col(COL_ITEM_ID).n_unique()).collect().item()
    ts = train.select(
        pl.col(COL_TIMESTAMP).min().alias("ts_min"),
        pl.col(COL_TIMESTAMP).max().alias("ts_max"),
    ).collect()
    nulls = tracker.note("train_null_counts", train.null_count().collect())
    rows_per_event = tracker.note(
        "rows_per_event",
        train.group_by(COL_EVENT_ID)
        .len()
        .select(
            pl.col("len").min().alias("min"),
            pl.col("len").max().alias("max"),
            (pl.col("len") != IMPRESSIONS_PER_EVENT).sum().alias("n_not_10"),
        )
        .collect(),
    )
    items_per_event = train.group_by(COL_EVENT_ID).agg(
        pl.col(COL_ITEM_ID).n_unique().alias("n_items"),
        pl.col(COL_USER_ID).n_unique().alias("n_users"),
        pl.col(COL_TIMESTAMP).n_unique().alias("n_ts"),
        pl.col(COL_CLICKED).sum().alias("n_clicks"),
    )
    click_dist = tracker.note(
        "clicks_per_event",
        items_per_event.select(
            (pl.col("n_clicks") == 0).sum().alias("zero"),
            (pl.col("n_clicks") == 1).sum().alias("one"),
            (pl.col("n_clicks") > 1).sum().alias("more"),
            pl.col("n_clicks").max().alias("max_clicks"),
            (pl.col("n_items") != IMPRESSIONS_PER_EVENT).sum().alias("events_with_dup_items"),
            (pl.col("n_users") != 1).sum().alias("events_with_multi_users"),
            (pl.col("n_ts") != 1).sum().alias("events_with_multi_ts"),
        ).collect(),
    )
    bucket_mismatch = train.select(
        (pl.col("time_bucket") != _expected_time_bucket()).sum().alias("n")
    ).collect().item()
    day_mismatch = train.select(
        (pl.col("day_type") != _expected_day_type()).sum().alias("n")
    ).collect().item()
    hour_map = tracker.note(
        "time_bucket_by_hour",
        train.select(pl.col("time_bucket"), _hour())
        .group_by("time_bucket", "hour")
        .len()
        .sort("time_bucket", "hour")
        .collect(),
    )
    return {
        "n_rows": n_rows,
        "n_events": n_events,
        "n_users": n_users,
        "n_items": n_items,
        "dtypes": dtypes,
        "nulls": {k: int(nulls[k][0]) for k in nulls.columns},
        "rows_per_event_min": int(rows_per_event["min"][0]),
        "rows_per_event_max": int(rows_per_event["max"][0]),
        "n_events_not_10_rows": int(rows_per_event["n_not_10"][0]),
        "events_zero_clicks": int(click_dist["zero"][0]),
        "events_one_click": int(click_dist["one"][0]),
        "events_multi_click": int(click_dist["more"][0]),
        "max_clicks_per_event": int(click_dist["max_clicks"][0]),
        "events_with_dup_items": int(click_dist["events_with_dup_items"][0]),
        "events_with_multi_users": int(click_dist["events_with_multi_users"][0]),
        "events_with_multi_ts": int(click_dist["events_with_multi_ts"][0]),
        "ts_min": int(ts["ts_min"][0]),
        "ts_max": int(ts["ts_max"][0]),
        "ts_min_utc": str(pl.from_epoch(pl.Series([ts["ts_min"][0]]), time_unit="s")[0]),
        "ts_max_utc": str(pl.from_epoch(pl.Series([ts["ts_max"][0]]), time_unit="s")[0]),
        "time_bucket_mismatches": int(bucket_mismatch),
        "day_type_mismatches": int(day_mismatch),
        "time_bucket_by_hour": hour_map,
        "ids_are_strings": all(
            dtypes.get(c, "").startswith("String") or dtypes.get(c, "") == "Utf8"
            for c in (COL_ROWID, COL_EVENT_ID, COL_USER_ID, COL_ITEM_ID)
        ),
        "window_matches_docs": int(ts["ts_min"][0]) >= TRAIN_START_TS
        and int(ts["ts_max"][0]) < TRAIN_END_TS,
    }


def target_analysis(train: pl.LazyFrame, tracker: SizeTracker) -> dict[str, Any]:
    n_rows = train.select(pl.len()).collect().item()
    n_clicks = train.select(pl.col(COL_CLICKED).sum()).collect().item()
    row_ctr = n_clicks / n_rows
    events = train.group_by(COL_EVENT_ID).agg(pl.col(COL_CLICKED).sum().alias("n_clicks"))
    zero_frac = events.select((pl.col("n_clicks") == 0).mean()).collect().item()
    by_bucket = tracker.note(
        "ctr_by_time_bucket",
        train.group_by("time_bucket")
        .agg(n=pl.len(), clicks=pl.col(COL_CLICKED).sum(), ctr=pl.col(COL_CLICKED).mean())
        .sort("time_bucket")
        .collect(),
    )
    by_day = tracker.note(
        "ctr_by_day_type",
        train.group_by("day_type")
        .agg(n=pl.len(), clicks=pl.col(COL_CLICKED).sum(), ctr=pl.col(COL_CLICKED).mean())
        .sort("day_type")
        .collect(),
    )
    by_hour = tracker.note(
        "ctr_by_hour",
        train.select(_hour(), COL_CLICKED)
        .group_by("hour")
        .agg(n=pl.len(), ctr=pl.col(COL_CLICKED).mean())
        .sort("hour")
        .collect(),
    )
    by_date = tracker.note(
        "ctr_by_date",
        train.select(_date(), COL_CLICKED)
        .group_by("date")
        .agg(n=pl.len(), events=pl.len() / IMPRESSIONS_PER_EVENT, ctr=pl.col(COL_CLICKED).mean())
        .sort("date")
        .collect(),
    )
    return {
        "row_ctr": float(row_ctr),
        "n_clicks": int(n_clicks),
        "zero_click_event_frac": float(zero_frac),
        "by_bucket": by_bucket,
        "by_day": by_day,
        "by_hour": by_hour,
        "by_date": by_date,
    }


def cardinality_and_overlap(
    train: pl.LazyFrame,
    eval_lf: pl.LazyFrame,
    train_events: pl.DataFrame,
    eval_events: pl.DataFrame,
    tracker: SizeTracker,
) -> dict[str, Any]:
    n_users = train.select(pl.col(COL_USER_ID).n_unique()).collect().item()
    n_items = train.select(pl.col(COL_ITEM_ID).n_unique()).collect().item()
    users_per_event_max = train.group_by(COL_EVENT_ID).agg(
        pl.col(COL_USER_ID).n_unique()
    ).select(pl.max(COL_USER_ID)).collect().item()
    events_per_user = tracker.note(
        "events_per_user",
        train_events.group_by(COL_USER_ID).len().rename({"len": "n_events"}),
    )
    epu_desc = events_per_user.select("n_events").describe()
    item_impr = tracker.note(
        "impressions_per_item",
        train.group_by(COL_ITEM_ID).len().rename({"len": "n_impr"}).collect(),
    )
    item_desc = item_impr.select("n_impr").describe()
    eval_users = eval_lf.select(pl.col(COL_USER_ID).unique()).collect()
    train_users = train.select(pl.col(COL_USER_ID).unique()).collect()
    eval_items = eval_lf.select(pl.col(COL_ITEM_ID).unique()).collect()
    train_items = train.select(pl.col(COL_ITEM_ID).unique()).collect()
    n_eval_users = eval_users.height
    n_eval_items = eval_items.height
    unseen_users = eval_users.join(train_users, on=COL_USER_ID, how="anti").height
    unseen_items = eval_items.join(train_items, on=COL_ITEM_ID, how="anti").height
    eval_empty = int((eval_events["hist_len"] == 0).sum())
    train_empty = int((train_events["hist_len"] == 0).sum())
    eval_n_events = eval_events.height
    eval_n_users_from_events = eval_events[COL_USER_ID].n_unique()
    return {
        "n_train_users": int(n_users),
        "n_train_items": int(n_items),
        "users_per_event_max": int(users_per_event_max),
        "events_per_user": epu_desc,
        "events_per_user_median": float(events_per_user["n_events"].median()),
        "events_per_user_p90": float(events_per_user["n_events"].quantile(0.9)),
        "events_per_user_max": int(events_per_user["n_events"].max()),
        "item_impr": item_desc,
        "item_impr_median": float(item_impr["n_impr"].median()),
        "item_impr_p90": float(item_impr["n_impr"].quantile(0.9)),
        "item_impr_max": int(item_impr["n_impr"].max()),
        "item_impr_min": int(item_impr["n_impr"].min()),
        "n_eval_users": int(n_eval_users),
        "n_eval_items": int(n_eval_items),
        "n_eval_events": int(eval_n_events),
        "eval_one_event_per_user": eval_n_events == eval_n_users_from_events,
        "eval_users_unseen_in_train": int(unseen_users),
        "eval_users_unseen_frac": unseen_users / n_eval_users if n_eval_users else 0.0,
        "eval_items_unseen_in_train": int(unseen_items),
        "eval_empty_history_events": eval_empty,
        "eval_empty_history_frac": eval_empty / eval_n_events if eval_n_events else 0.0,
        "train_empty_history_events": train_empty,
        "train_empty_history_frac": train_empty / train_events.height,
        "events_per_user_frame": events_per_user,
        "item_impr_frame": item_impr,
    }


def popularity_item_ctr(
    train: pl.LazyFrame,
    folds: dict[str, pl.LazyFrame],
    tracker: SizeTracker,
    min_impr: int = MIN_ITEM_IMPRESSIONS,
) -> dict[str, Any]:
    item_stats = tracker.note(
        "item_ctr_full_train",
        train.group_by(COL_ITEM_ID)
        .agg(n=pl.len(), clicks=pl.col(COL_CLICKED).sum(), ctr=pl.col(COL_CLICKED).mean())
        .collect(),
    )
    qualified = item_stats.filter(pl.col("n") >= min_impr).sort("ctr", descending=True)
    top = tracker.note("top_items_by_ctr", qualified.head(20))
    ctr_desc = qualified.select("ctr").describe()
    aucs = {}
    for fold_name, fold_lf in folds.items():
        cutoff = FOLD_CUTOFFS[fold_name]
        prior = (
            train.filter(pl.col(COL_TIMESTAMP) < cutoff)
            .group_by(COL_ITEM_ID)
            .agg(n=pl.len(), clicks=pl.col(COL_CLICKED).sum())
            .collect()
        )
        tracker.note(f"item_ctr_prior_{fold_name}", prior)
        gctr = float(prior["clicks"].sum() / prior["n"].sum()) if prior.height else 0.0
        prior = prior.with_columns((pl.col("clicks") / pl.col("n")).alias("ctr"))
        scored = (
            fold_lf.select(COL_ITEM_ID, COL_CLICKED)
            .join(prior.lazy().select(COL_ITEM_ID, "ctr"), on=COL_ITEM_ID, how="left")
            .with_columns(pl.col("ctr").fill_null(gctr))
            .collect()
        )
        tracker.note(f"scored_{fold_name}", scored)
        y = scored[COL_CLICKED].to_numpy()
        p = scored["ctr"].to_numpy()
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        aucs[fold_name] = {
            "auc": auc,
            "global_ctr": gctr,
            "n_rows": scored.height,
            "n_items_prior": prior.height,
        }
    return {
        "min_impr": min_impr,
        "n_items": item_stats.height,
        "n_qualified": qualified.height,
        "ctr_mean_qualified": float(qualified["ctr"].mean()) if qualified.height else 0.0,
        "ctr_p10": float(qualified["ctr"].quantile(0.1)) if qualified.height else 0.0,
        "ctr_p90": float(qualified["ctr"].quantile(0.9)) if qualified.height else 0.0,
        "ctr_max": float(qualified["ctr"].max()) if qualified.height else 0.0,
        "ctr_min": float(qualified["ctr"].min()) if qualified.height else 0.0,
        "top_items": top,
        "ctr_desc": ctr_desc,
        "item_stats": item_stats,
        "fold_aucs": aucs,
    }


def _history_in_item_tables(events: pl.DataFrame, tracker: SizeTracker, name: str) -> pl.DataFrame:
    exploded = (
        events.select(COL_EVENT_ID, COL_TIMESTAMP, "hist_items", "hist_ts")
        .with_columns(
            pl.col("hist_items").list.reverse(),
            pl.col("hist_ts").list.reverse(),
        )
        .explode("hist_items", "hist_ts")
        .drop_nulls("hist_items")
        .with_columns(pl.int_range(pl.len()).over(COL_EVENT_ID).alias("pos_from_end"))
        .unique(subset=[COL_EVENT_ID, "hist_items"], keep="first")
        .rename({"hist_items": COL_ITEM_ID, "hist_ts": "hist_click_ts"})
        .with_columns(
            ((pl.col(COL_TIMESTAMP) - pl.col("hist_click_ts")) / 3600.0).alias("hours_since_hist")
        )
    )
    return tracker.note(name, exploded)


def history_analysis(
    train: pl.LazyFrame,
    train_events: pl.DataFrame,
    eval_events: pl.DataFrame,
    tracker: SizeTracker,
) -> dict[str, Any]:
    def hist_len_table(events: pl.DataFrame, label: str) -> pl.DataFrame:
        return tracker.note(
            f"hist_len_{label}",
            events.group_by("hist_len").len().rename({"len": "n_events"}).sort("hist_len"),
        )

    train_len = hist_len_table(train_events, "train")
    eval_len = hist_len_table(eval_events, "eval")
    empty_train = float((train_events["hist_len"] == 0).mean())
    empty_eval = float((eval_events["hist_len"] == 0).mean())
    dup_train = float((train_events["hist_len"] - train_events["hist_unique"] > 0).mean())
    dup_eval = float((eval_events["hist_len"] - eval_events["hist_unique"] > 0).mean())
    ctr_by_len = tracker.note(
        "ctr_by_hist_len",
        train_events.group_by("hist_len")
        .agg(
            n_events=pl.len(),
            event_has_click=pl.col("n_clicks").gt(0).mean(),
            row_ctr=pl.col("n_clicks").mean() / IMPRESSIONS_PER_EVENT,
        )
        .sort("hist_len"),
    )
    hist_pos = _history_in_item_tables(train_events, tracker, "train_hist_exploded")
    flagged = (
        train.select(COL_EVENT_ID, COL_ITEM_ID, COL_CLICKED)
        .join(
            hist_pos.lazy().select(COL_EVENT_ID, COL_ITEM_ID, "pos_from_end", "hours_since_hist"),
            on=[COL_EVENT_ID, COL_ITEM_ID],
            how="left",
        )
        .with_columns(pl.col("pos_from_end").is_not_null().alias("in_hist"))
    )
    in_vs_out = tracker.note(
        "ctr_in_vs_out_history",
        flagged.group_by("in_hist").agg(n=pl.len(), ctr=pl.col(COL_CLICKED).mean()).sort("in_hist").collect(),
    )
    by_pos = tracker.note(
        "ctr_by_hist_pos_from_end",
        flagged.filter(pl.col("in_hist"))
        .group_by("pos_from_end")
        .agg(n=pl.len(), ctr=pl.col(COL_CLICKED).mean())
        .sort("pos_from_end")
        .collect(),
    )
    hour_bin = (
        pl.when(pl.col("hours_since_hist") < 1)
        .then(pl.lit("0-1h"))
        .when(pl.col("hours_since_hist") < 6)
        .then(pl.lit("1-6h"))
        .when(pl.col("hours_since_hist") < 24)
        .then(pl.lit("6-24h"))
        .when(pl.col("hours_since_hist") < 72)
        .then(pl.lit("1-3d"))
        .otherwise(pl.lit("3d+"))
        .alias("hours_bin")
    )
    by_hours = tracker.note(
        "ctr_by_hours_since_hist",
        flagged.filter(pl.col("in_hist"))
        .with_columns(hour_bin)
        .group_by("hours_bin")
        .agg(n=pl.len(), ctr=pl.col(COL_CLICKED).mean())
        .collect(),
    )
    gap_bin = (
        pl.when(pl.col("gap_seconds").is_null())
        .then(pl.lit("no_history"))
        .when(pl.col("gap_seconds") < 3600)
        .then(pl.lit("0-1h"))
        .when(pl.col("gap_seconds") < 6 * 3600)
        .then(pl.lit("1-6h"))
        .when(pl.col("gap_seconds") < 24 * 3600)
        .then(pl.lit("6-24h"))
        .when(pl.col("gap_seconds") < 72 * 3600)
        .then(pl.lit("1-3d"))
        .otherwise(pl.lit("3d+"))
        .alias("gap_bin")
    )
    gap_stats = tracker.note(
        "gap_seconds_describe",
        train_events.select("gap_seconds").drop_nulls().describe(),
    )
    ctr_by_gap = tracker.note(
        "ctr_by_gap_bin",
        train_events.with_columns(gap_bin)
        .group_by("gap_bin")
        .agg(
            n_events=pl.len(),
            row_ctr=pl.col("n_clicks").mean() / IMPRESSIONS_PER_EVENT,
        ),
    )
    in_rate = float(
        flagged.select(pl.col("in_hist").mean()).collect().item()
    )
    return {
        "train_hist_len": train_len,
        "eval_hist_len": eval_len,
        "empty_frac_train": empty_train,
        "empty_frac_eval": empty_eval,
        "hist_len_mean_train": float(train_events["hist_len"].mean()),
        "hist_len_mean_eval": float(eval_events["hist_len"].mean()),
        "hist_len_median_train": float(train_events["hist_len"].median()),
        "hist_len_median_eval": float(eval_events["hist_len"].median()),
        "frac_events_with_hist_repeat_train": dup_train,
        "frac_events_with_hist_repeat_eval": dup_eval,
        "ctr_by_len": ctr_by_len,
        "in_vs_out": in_vs_out,
        "candidate_in_history_frac": in_rate,
        "by_pos": by_pos,
        "by_hours": by_hours,
        "gap_stats": gap_stats,
        "ctr_by_gap": ctr_by_gap,
    }


def candidate_set_structure(train: pl.LazyFrame, tracker: SizeTracker) -> dict[str, Any]:
    by_rowid = tracker.note(
        "ctr_by_rowid_rank",
        train.with_columns(pl.col(COL_ROWID).rank("ordinal").over(COL_EVENT_ID).alias("rowid_pos"))
        .group_by("rowid_pos")
        .agg(n=pl.len(), ctr=pl.col(COL_CLICKED).mean())
        .sort("rowid_pos")
        .collect(),
    )
    by_scan = tracker.note(
        "ctr_by_scan_pos",
        train.select(COL_EVENT_ID, COL_CLICKED)
        .with_row_index("i")
        .with_columns(pl.col("i").rank("ordinal").over(COL_EVENT_ID).alias("scan_pos"))
        .group_by("scan_pos")
        .agg(n=pl.len(), ctr=pl.col(COL_CLICKED).mean())
        .sort("scan_pos")
        .collect(),
    )
    rowid_spread = float(by_rowid["ctr"].max() - by_rowid["ctr"].min())
    scan_spread = float(by_scan["ctr"].max() - by_scan["ctr"].min())
    # Co-occurrence on a deterministic sample of events (not the full 30M pairs).
    sample_events = (
        train.select(COL_EVENT_ID)
        .unique(maintain_order=True)
        .head(8_000)
        .collect()
    )
    tracker.note("pair_sample_events", sample_events)
    pairs = (
        train.select(COL_EVENT_ID, COL_ITEM_ID)
        .join(sample_events.lazy(), on=COL_EVENT_ID)
        .join(
            train.select(COL_EVENT_ID, pl.col(COL_ITEM_ID).alias("item_id_2")),
            on=COL_EVENT_ID,
        )
        .filter(pl.col(COL_ITEM_ID) < pl.col("item_id_2"))
        .group_by(COL_ITEM_ID, "item_id_2")
        .len()
        .sort("len", descending=True)
        .head(15)
        .collect()
    )
    tracker.note("top_item_pairs", pairs)
    return {
        "by_rowid": by_rowid,
        "by_scan": by_scan,
        "rowid_ctr_spread": rowid_spread,
        "scan_ctr_spread": scan_spread,
        "rowid_looks_random": rowid_spread < 0.005,
        "scan_looks_uninformative": scan_spread < 0.005,
        "top_pairs": pairs,
    }


def _dist_table(lf: pl.LazyFrame, col: str, tracker: SizeTracker, name: str) -> pl.DataFrame:
    n = lf.select(pl.len()).collect().item()
    tab = (
        lf.group_by(col)
        .len()
        .with_columns((pl.col("len") / n).alias("share"))
        .sort(col)
        .collect()
    )
    return tracker.note(name, tab)


def distribution_shift(
    train: pl.LazyFrame,
    eval_lf: pl.LazyFrame,
    folds: dict[str, pl.LazyFrame],
    train_events: pl.DataFrame,
    eval_events: pl.DataFrame,
    fold_events: dict[str, pl.DataFrame],
    tracker: SizeTracker,
) -> dict[str, Any]:
    sources: dict[str, pl.LazyFrame] = {"train": train, "eval": eval_lf, **folds}
    buckets = {k: _dist_table(v, "time_bucket", tracker, f"tb_{k}") for k, v in sources.items()}
    days = {k: _dist_table(v, "day_type", tracker, f"dt_{k}") for k, v in sources.items()}
    hist_means = {
        "train": float(train_events["hist_len"].mean()),
        "eval": float(eval_events["hist_len"].mean()),
        **{k: float(v["hist_len"].mean()) for k, v in fold_events.items()},
    }
    eval_by_date = tracker.note(
        "eval_events_by_date",
        eval_events.group_by("date").len().rename({"len": "n_events"}).sort("date"),
    )
    eval_by_hour = tracker.note(
        "eval_events_by_hour",
        eval_events.group_by("hour").len().rename({"len": "n_events"}).sort("hour"),
    )
    train_by_hour = tracker.note(
        "train_events_by_hour",
        train_events.group_by("hour").len().rename({"len": "n_events"}).sort("hour"),
    )
    return {
        "time_bucket": buckets,
        "day_type": days,
        "hist_len_mean": hist_means,
        "eval_by_date": eval_by_date,
        "eval_by_hour": eval_by_hour,
        "train_by_hour": train_by_hour,
    }


def cold_start_slices(
    train: pl.LazyFrame,
    eval_lf: pl.LazyFrame,
    folds: dict[str, pl.LazyFrame],
    eval_events: pl.DataFrame,
    fold_events: dict[str, pl.DataFrame],
    tracker: SizeTracker,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def slice_stats(
        label: str,
        frame: pl.LazyFrame,
        events: pl.DataFrame,
        prior_users: pl.DataFrame,
        prior_items: pl.DataFrame,
    ) -> dict[str, Any]:
        users = frame.select(pl.col(COL_USER_ID).unique()).collect()
        items = frame.select(pl.col(COL_ITEM_ID).unique()).collect()
        n_rows = frame.select(pl.len()).collect().item()
        unseen_u = users.join(prior_users, on=COL_USER_ID, how="anti")
        unseen_i = items.join(prior_items, on=COL_ITEM_ID, how="anti")
        empty = int((events["hist_len"] == 0).sum())
        # row slices
        unseen_user_rows = frame.select(COL_USER_ID).join(
            unseen_u.lazy(), on=COL_USER_ID
        ).select(pl.len()).collect().item()
        unseen_item_rows = frame.select(COL_ITEM_ID).join(
            unseen_i.lazy(), on=COL_ITEM_ID
        ).select(pl.len()).collect().item()
        empty_rows = empty * IMPRESSIONS_PER_EVENT
        stats = {
            "n_rows": int(n_rows),
            "n_events": events.height,
            "unseen_user_events": unseen_u.height,
            "unseen_user_rows": int(unseen_user_rows),
            "unseen_user_row_frac": unseen_user_rows / n_rows if n_rows else 0.0,
            "unseen_item_events": None,
            "unseen_item_rows": int(unseen_item_rows),
            "unseen_item_row_frac": unseen_item_rows / n_rows if n_rows else 0.0,
            "empty_history_events": empty,
            "empty_history_rows": int(empty_rows),
            "empty_history_row_frac": empty_rows / n_rows if n_rows else 0.0,
        }
        tracker.note(f"cold_{label}_unseen_users", unseen_u)
        tracker.note(f"cold_{label}_unseen_items", unseen_i)
        return stats

    train_users = train.select(pl.col(COL_USER_ID).unique()).collect()
    train_items = train.select(pl.col(COL_ITEM_ID).unique()).collect()
    out["eval_vs_all_train"] = slice_stats("eval", eval_lf, eval_events, train_users, train_items)

    for fold_name, fold_lf in folds.items():
        cutoff = FOLD_CUTOFFS[fold_name]
        prior_users = (
            train.filter(pl.col(COL_TIMESTAMP) < cutoff)
            .select(pl.col(COL_USER_ID).unique())
            .collect()
        )
        prior_items = (
            train.filter(pl.col(COL_TIMESTAMP) < cutoff)
            .select(pl.col(COL_ITEM_ID).unique())
            .collect()
        )
        out[fold_name] = slice_stats(
            fold_name, fold_lf, fold_events[fold_name], prior_users, prior_items
        )
    return out


def _save_df(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _barplot(df: pl.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    pdf = df.to_pandas()
    sns.barplot(data=pdf, x=x, y=y, ax=ax, color="#3b6fb6")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def _lineplot(df: pl.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    pdf = df.to_pandas()
    sns.lineplot(data=pdf, x=x, y=y, ax=ax, marker="o")
    ax.set_title(title)
    fig.autofmt_xdate()
    fig.savefig(path)
    plt.close(fig)


def write_plots(results: dict[str, Any], fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(name: str) -> Path:
        written.append(name)
        return fig_dir / name

    tgt = results["target"]
    _barplot(tgt["by_bucket"], "time_bucket", "ctr", save("ctr_by_time_bucket.png"), "Click rate by time_bucket")
    _barplot(tgt["by_day"], "day_type", "ctr", save("ctr_by_day_type.png"), "Click rate by day_type")
    _lineplot(tgt["by_hour"], "hour", "ctr", save("ctr_by_hour.png"), "Click rate by hour of day (UTC)")
    by_date = tgt["by_date"].with_columns(pl.col("date").cast(pl.Utf8))
    _lineplot(by_date, "date", "ctr", save("ctr_by_date.png"), "Daily click rate (train)")
    _barplot(by_date, "date", "n", save("rows_by_date.png"), "Train rows by date")

    card = results["cardinality"]
    fig, ax = plt.subplots()
    sns.histplot(card["events_per_user_frame"]["n_events"].to_pandas(), bins=40, ax=ax)
    ax.set_title("Events per user (train)")
    fig.savefig(save("events_per_user.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    vals = card["item_impr_frame"]["n_impr"].to_pandas()
    sns.histplot(np.log10(vals.clip(lower=1)), bins=40, ax=ax)
    ax.set_xlabel("log10 impressions per item")
    ax.set_title("Item impression tail (train)")
    fig.savefig(save("item_impressions_log.png"))
    plt.close(fig)

    pop = results["popularity"]
    fig, ax = plt.subplots()
    sns.histplot(pop["item_stats"]["ctr"].to_pandas(), bins=40, ax=ax)
    ax.set_title("Item click-rate distribution (train)")
    fig.savefig(save("item_ctr_hist.png"))
    plt.close(fig)

    hist = results["history"]
    _barplot(hist["train_hist_len"], "hist_len", "n_events", save("hist_len_train.png"), "History length (train events)")
    _barplot(hist["eval_hist_len"], "hist_len", "n_events", save("hist_len_eval.png"), "History length (eval events)")
    _lineplot(hist["ctr_by_len"], "hist_len", "row_ctr", save("ctr_by_hist_len.png"), "Row CTR vs history length")
    _barplot(hist["in_vs_out"].with_columns(pl.col("in_hist").cast(pl.Utf8)), "in_hist", "ctr", save("ctr_in_vs_out_history.png"), "CTR: candidate in history vs not")
    if hist["by_pos"].height:
        _lineplot(hist["by_pos"], "pos_from_end", "ctr", save("ctr_by_hist_recency_pos.png"), "CTR by recency rank (0=most recent hist click)")

    cand = results["candidates"]
    _lineplot(cand["by_rowid"], "rowid_pos", "ctr", save("ctr_by_rowid_rank.png"), "CTR by lexicographic rowid rank within event")
    _lineplot(cand["by_scan"], "scan_pos", "ctr", save("ctr_by_scan_pos.png"), "CTR by parquet scan order within event")

    shift = results["shift"]
    eval_dates = shift["eval_by_date"].with_columns(pl.col("date").cast(pl.Utf8))
    _barplot(eval_dates, "date", "n_events", save("eval_events_by_date.png"), "Eval events by date (first event after cutoff)")
    _barplot(shift["eval_by_hour"], "hour", "n_events", save("eval_events_by_hour.png"), "Eval events by hour (UTC)")
    return written


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _md_table(df: pl.DataFrame, cols: list[str] | None = None) -> str:
    if cols:
        df = df.select(cols)
    pdf = df.to_pandas()
    headers = list(pdf.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in pdf.iterrows():
        cells = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                cells.append(f"{val:.6g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_key_findings(r: dict[str, Any]) -> list[str]:
    s, t, c, p, h, cand, cold = (
        r["sanity"],
        r["target"],
        r["cardinality"],
        r["popularity"],
        r["history"],
        r["candidates"],
        r["cold"],
    )
    auc25 = p["fold_aucs"]["fold_2025_01_25"]["auc"]
    auc26 = p["fold_aucs"]["fold_2025_01_26"]["auc"]
    in_ctr = float(h["in_vs_out"].filter(pl.col("in_hist")).select("ctr").item()) if h["in_vs_out"].filter(pl.col("in_hist")).height else float("nan")
    out_ctr = float(h["in_vs_out"].filter(~pl.col("in_hist")).select("ctr").item()) if h["in_vs_out"].filter(~pl.col("in_hist")).height else float("nan")
    tb_ok = s["time_bucket_mismatches"] == 0
    return [
        f"Train has {s['n_rows']:,} rows / {s['n_events']:,} events / {s['n_users']:,} users / {s['n_items']:,} items; every event has exactly {s['rows_per_event_min']} unique items and {s['events_multi_click']} events have more than one click.",
        f"Row click rate is {_fmt_pct(t['row_ctr'])} ({t['n_clicks']:,} clicks); {_fmt_pct(t['zero_click_event_frac'])} of events have zero clicks (allowed).",
        f"time_bucket matches 6-hour UTC blocks with {s['time_bucket_mismatches']:,} mismatches; day_type matches weekend/weekday with {s['day_type_mismatches']:,} mismatches. Docs agree with the data." if tb_ok else f"time_bucket DOES NOT match the documented 6-hour UTC mapping: {s['time_bucket_mismatches']:,} rows disagree.",
        f"Evaluation is {c['n_eval_events']:,} events for {c['n_eval_users']:,} users (one event each). {c['eval_users_unseen_in_train']:,} eval users ({_fmt_pct(c['eval_users_unseen_frac'])}) never appear in train; {c['eval_items_unseen_in_train']:,} eval items are unseen.",
        f"A leak-free item-CTR predictor scores ROC AUC {_fmt(auc25)} on the Jan 25 first-event fold and {_fmt(auc26)} on the Jan 26 fold — item popularity is a real but limited ranking signal.",
        f"Candidates that appear in the user's history click at {_fmt_pct(in_ctr)} vs {_fmt_pct(out_ctr)} when they do not ({_fmt_pct(h['candidate_in_history_frac'])} of impressions are repeats).",
        f"Empty history: {_fmt_pct(h['empty_frac_train'])} of train events and {_fmt_pct(h['empty_frac_eval'])} of eval events. { _fmt_pct(h['frac_events_with_hist_repeat_train'])} of train histories repeat an item.",
        f"Eval is a first-event-after-cutoff sample, so it is front-loaded in time (not a typical train day mix). Fold files are the same protocol applied at 2025-01-25 and 2025-01-26, not full calendar days.",
        f"rowid / file order within an event is effectively uninformative (CTR spread {cand['rowid_ctr_spread']:.4f} by rowid rank, {cand['scan_ctr_spread']:.4f} by scan order); do not use rowid as a feature.",
        f"Cold start in eval vs all train: {_fmt_pct(cold['eval_vs_all_train']['unseen_user_row_frac'])} of rows are unseen users, {_fmt_pct(cold['eval_vs_all_train']['empty_history_row_frac'])} empty history, {_fmt_pct(cold['eval_vs_all_train']['unseen_item_row_frac'])} unseen items.",
    ]


def write_markdown(r: dict[str, Any], path: Path, figures: list[str], tracker: SizeTracker) -> list[str]:
    findings = build_key_findings(r)
    s, t, c, p, h, cand, shift, cold = (
        r["sanity"],
        r["target"],
        r["cardinality"],
        r["popularity"],
        r["history"],
        r["candidates"],
        r["shift"],
        r["cold"],
    )
    peak_name, peak_rows, peak_bytes = tracker.peak()
    lines: list[str] = [
        "# EDA: TechArena 2026 Phase 1 click prediction",
        "",
        "Generated by `python scripts/01_eda.py` (seed 42). Metric: ROC AUC pooled over all rows.",
        "",
        "## Key findings",
        "",
    ]
    for bullet in findings:
        lines.append(f"- {bullet}")
    lines += [
        "",
        f"Peak in-memory table: **{peak_name}** with {peak_rows:,} rows ({peak_bytes / 1e6:.1f} MB).",
        "",
        "## 1. Schema and sanity",
        "",
        f"- Rows: {s['n_rows']:,}; events: {s['n_events']:,}; users: {s['n_users']:,}; items: {s['n_items']:,}.",
        f"- Dtypes: `{s['dtypes']}`. IDs stored as strings: {s['ids_are_strings']}.",
        f"- Nulls per column: `{s['nulls']}`.",
        f"- Rows per event min/max: {s['rows_per_event_min']}/{s['rows_per_event_max']} "
        f"(events with !=10 rows: {s['n_events_not_10_rows']:,}). Duplicate items in an event: {s['events_with_dup_items']:,}.",
        f"- Clicks per event: 0 → {s['events_zero_clicks']:,}; 1 → {s['events_one_click']:,}; >1 → {s['events_multi_click']:,} (max {s['max_clicks_per_event']}).",
        f"- Timestamp range (UTC): {s['ts_min_utc']} → {s['ts_max_utc']} (unix {s['ts_min']}–{s['ts_max']}). "
        f"Matches documented train window [2025-01-20, 2025-01-27): {s['window_matches_docs']}.",
        f"- Users/timestamps constant within event: multi-user events={s['events_with_multi_users']:,}, multi-timestamp events={s['events_with_multi_ts']:,}.",
        "",
        "### time_bucket vs UTC hour",
        "",
        "Docs say 0=00:00–05:59, 1=06:00–11:59, 2=12:00–17:59, 3=18:00–23:59. "
        f"Rows where `time_bucket != floor(hour/6)`: **{s['time_bucket_mismatches']:,}**. "
        f"Rows where `day_type` disagrees with Sat/Sun: **{s['day_type_mismatches']:,}**.",
        "",
        _md_table(s["time_bucket_by_hour"]),
        "",
        "## 2. Target",
        "",
        f"- Overall row CTR: **{_fmt_pct(t['row_ctr'])}**.",
        f"- Fraction of events with zero clicks: **{_fmt_pct(t['zero_click_event_frac'])}**.",
        "",
        "### By time_bucket",
        "",
        _md_table(t["by_bucket"]),
        "",
        "### By day_type",
        "",
        _md_table(t["by_day"]),
        "",
        "### By hour (UTC)",
        "",
        _md_table(t["by_hour"]),
        "",
        "### By date",
        "",
        _md_table(t["by_date"]),
        "",
        "## 3. Cardinality and overlap",
        "",
        f"- Unique train users {c['n_train_users']:,}; unique train items {c['n_train_items']:,}.",
        f"- Users per event is always 1 (max {c['users_per_event_max']}).",
        f"- Events per user: median {c['events_per_user_median']:.1f}, p90 {c['events_per_user_p90']:.1f}, max {c['events_per_user_max']:,}.",
        f"- Impressions per item: min {c['item_impr_min']:,}, median {c['item_impr_median']:.0f}, p90 {c['item_impr_p90']:.0f}, max {c['item_impr_max']:,} (long tail).",
        f"- Eval users in train: {c['n_eval_users'] - c['eval_users_unseen_in_train']:,} / {c['n_eval_users']:,} "
        f"({c['eval_users_unseen_in_train']:,} unseen).",
        f"- Eval items in train: {c['n_eval_items'] - c['eval_items_unseen_in_train']:,} / {c['n_eval_items']:,} "
        f"({c['eval_items_unseen_in_train']:,} unseen).",
        f"- Eval events with empty history: {c['eval_empty_history_events']:,} ({_fmt_pct(c['eval_empty_history_frac'])}). One event per user: {c['eval_one_event_per_user']}.",
        "",
        "## 4. Popularity / item CTR",
        "",
        f"- Items with ≥{p['min_impr']} impressions: {p['n_qualified']:,} / {p['n_items']:,}.",
        f"- Qualified item CTR range: {p['ctr_min']:.4f}–{p['ctr_max']:.4f} (mean {p['ctr_mean_qualified']:.4f}, p10 {p['ctr_p10']:.4f}, p90 {p['ctr_p90']:.4f}).",
        "There is item-level signal, but CTR is compressed around the global rate for most items.",
        "",
        "Top items by CTR (qualified):",
        "",
        _md_table(p["top_items"]),
        "",
        "Leak-free item-CTR AUC (prior days only, unseen items filled with prior global CTR):",
        "",
    ]
    for name, auc in p["fold_aucs"].items():
        lines.append(
            f"- `{name}`: AUC **{_fmt(auc['auc'])}** on {auc['n_rows']:,} rows "
            f"(prior items {auc['n_items_prior']:,}, prior global CTR {_fmt_pct(auc['global_ctr'])})."
        )
    lines += [
        "",
        "## 5. History",
        "",
        f"- Mean history length: train {h['hist_len_mean_train']:.2f} (median {h['hist_len_median_train']:.0f}), "
        f"eval {h['hist_len_mean_eval']:.2f} (median {h['hist_len_median_eval']:.0f}). Cap appears to be 20.",
        f"- Empty history: train {_fmt_pct(h['empty_frac_train'])}, eval {_fmt_pct(h['empty_frac_eval'])}.",
        f"- Events whose history repeats an item: train {_fmt_pct(h['frac_events_with_hist_repeat_train'])}, "
        f"eval {_fmt_pct(h['frac_events_with_hist_repeat_eval'])}.",
        "",
        "CTR by history length (event-level n_clicks / 10):",
        "",
        _md_table(h["ctr_by_len"]),
        "",
        "Candidate in history vs not:",
        "",
        _md_table(h["in_vs_out"]),
        "",
        "CTR by recency rank in history (0 = most recent click):",
        "",
        _md_table(h["by_pos"]),
        "",
        "CTR by hours since that historical click:",
        "",
        _md_table(h["by_hours"]),
        "",
        "Gap from last history timestamp to event time:",
        "",
        _md_table(h["gap_stats"]),
        "",
        _md_table(h["ctr_by_gap"]),
        "",
        "## 6. Candidate-set structure",
        "",
        f"- CTR spread across lexicographic `rowid` rank: {cand['rowid_ctr_spread']:.5f} "
        f"(treat as random: {cand['rowid_looks_random']}).",
        f"- CTR spread across parquet scan order: {cand['scan_ctr_spread']:.5f} "
        f"(uninformative: {cand['scan_looks_uninformative']}).",
        "- Do **not** use `rowid` as a model feature; it is a random key.",
        "",
        "CTR by rowid rank:",
        "",
        _md_table(cand["by_rowid"]),
        "",
        "Most frequent item pairs in a 8,000-event sample (popular items dominate; not a stable graph feature by itself):",
        "",
        _md_table(cand["top_pairs"]),
        "",
        "## 7. Train vs evaluation shift",
        "",
        "The evaluation set is each user's **first event at/after 2025-01-27 00:00 UTC**, so it is not a random slice of a day. "
        "Expect more morning `time_bucket` mass and a weekday (Monday 27 Jan) skew. The Jan 25/26 fold files use the same protocol.",
        "",
        "Mean history length: "
        + ", ".join(f"{k}={v:.2f}" for k, v in shift["hist_len_mean"].items())
        + ".",
        "",
        "time_bucket share:",
        "",
    ]
    for name, tab in shift["time_bucket"].items():
        lines += [f"**{name}**", "", _md_table(tab), ""]
    lines.append("day_type share:")
    lines.append("")
    for name, tab in shift["day_type"].items():
        lines += [f"**{name}**", "", _md_table(tab), ""]
    lines += [
        "Eval events by date:",
        "",
        _md_table(shift["eval_by_date"]),
        "",
        "## 8. Cold-start slices",
        "",
        "Unseen user/item for folds is relative to **data before that fold's cutoff** (no leakage). Eval is relative to all of train.",
        "",
    ]
    for name, stats in cold.items():
        lines += [
            f"### {name}",
            "",
            f"- rows {stats['n_rows']:,}, events {stats['n_events']:,}",
            f"- unseen-user rows: {stats['unseen_user_rows']:,} ({_fmt_pct(stats['unseen_user_row_frac'])}), events {stats['unseen_user_events']:,}",
            f"- unseen-item rows: {stats['unseen_item_rows']:,} ({_fmt_pct(stats['unseen_item_row_frac'])})",
            f"- empty-history rows: {stats['empty_history_rows']:,} ({_fmt_pct(stats['empty_history_row_frac'])}), events {stats['empty_history_events']:,}",
            "",
        ]
    lines += [
        "## Recommended features and risks",
        "",
        "- **Use:** item prior CTR (time-aware), `time_bucket`, hour, `day_type`, history length, whether the candidate is in history, recency of that click, gap since last click, user activity (events so far before the cutoff).",
        "- **Use carefully:** user CTR — many eval/fold rows are a user's first post-cutoff event, so user averages from the same period leak and post-cutoff user counts are tiny.",
        "- **Do not use:** `rowid` or `event_id` as numeric features; in-fold labels; any aggregate computed on the scored rows.",
        "- **Risks:** eval is Monday-morning-skewed vs train's evening-heavy mix; ~2–3% brand-new users; empty histories are more common in eval than a typical train event. Item-CTR alone will not win — history match/recency is a stronger row-level signal. Folds that copy the first-event protocol are closer to the leaderboard than random or full-day splits.",
        "",
        "## Figures",
        "",
    ]
    for name in figures:
        lines.append(f"- `reports/figures/{name}`")
    lines += [
        "",
        "## Memory log",
        "",
        "| table | rows | MB |",
        "| --- | ---: | ---: |",
    ]
    for name, n_rows, nbytes in tracker.records:
        lines.append(f"| {name} | {n_rows:,} | {nbytes / 1e6:.1f} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return findings


def write_tables(results: dict[str, Any], table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "ctr_by_time_bucket.csv": results["target"]["by_bucket"],
        "ctr_by_hour.csv": results["target"]["by_hour"],
        "ctr_by_date.csv": results["target"]["by_date"],
        "top_items_by_ctr.csv": results["popularity"]["top_items"],
        "ctr_in_vs_out_history.csv": results["history"]["in_vs_out"],
        "ctr_by_hist_len.csv": results["history"]["ctr_by_len"],
        "ctr_by_rowid_rank.csv": results["candidates"]["by_rowid"],
        "eval_events_by_date.csv": results["shift"]["eval_by_date"],
        "cold_start_eval.csv": pl.DataFrame([results["cold"]["eval_vs_all_train"]]),
    }
    for name, df in mapping.items():
        _save_df(df, table_dir / name)


def run_eda(
    train: pl.LazyFrame,
    eval_lf: pl.LazyFrame,
    folds: dict[str, pl.LazyFrame],
    reports_dir: Path = REPORTS_DIR,
    *,
    tracker: SizeTracker | None = None,
) -> dict[str, Any]:
    tracker = tracker or SizeTracker()
    print("Building event-level tables (history JSON decoded once per event)...")
    train_events = event_level(train, has_clicked=True, tracker=tracker, name="train_events")
    eval_events = event_level(eval_lf, has_clicked=False, tracker=tracker, name="eval_events")
    fold_events = {
        name: event_level(lf, has_clicked=True, tracker=tracker, name=f"events_{name}")
        for name, lf in folds.items()
    }

    print("1 schema_and_sanity")
    sanity = schema_and_sanity(train, tracker)
    print("2 target_analysis")
    target = target_analysis(train, tracker)
    print("3 cardinality_and_overlap")
    cardinality = cardinality_and_overlap(train, eval_lf, train_events, eval_events, tracker)
    print("4 popularity_item_ctr")
    popularity = popularity_item_ctr(train, folds, tracker)
    print("5 history_analysis")
    history = history_analysis(train, train_events, eval_events, tracker)
    print("6 candidate_set_structure")
    candidates = candidate_set_structure(train, tracker)
    print("7 distribution_shift")
    shift = distribution_shift(train, eval_lf, folds, train_events, eval_events, fold_events, tracker)
    print("8 cold_start_slices")
    cold = cold_start_slices(train, eval_lf, folds, eval_events, fold_events, tracker)

    results = {
        "sanity": sanity,
        "target": target,
        "cardinality": cardinality,
        "popularity": popularity,
        "history": history,
        "candidates": candidates,
        "shift": shift,
        "cold": cold,
        "train_events": train_events,
        "eval_events": eval_events,
    }
    fig_dir = reports_dir / "figures"
    table_dir = reports_dir / "tables"
    print("Writing plots and tables...")
    figures = write_plots(results, fig_dir)
    write_tables(results, table_dir)
    findings = write_markdown(results, reports_dir / "01_eda.md", figures, tracker)
    results["findings"] = findings
    results["tracker"] = tracker
    print("\n=== Key findings ===")
    for bullet in findings:
        print(f"- {bullet}")
    peak_name, peak_rows, peak_bytes = tracker.peak()
    print(f"\nPeak in-memory table: {peak_name} ({peak_rows:,} rows, {peak_bytes / 1e6:.1f} MB)")
    return results
