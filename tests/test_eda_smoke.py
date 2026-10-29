"""Smoke-test EDA functions on a tiny deterministic sample (2,000 events)."""

from __future__ import annotations

import polars as pl

from techarena.eda import (
    SizeTracker,
    candidate_set_structure,
    cardinality_and_overlap,
    cold_start_slices,
    distribution_shift,
    event_level,
    history_analysis,
    make_smoke_frames,
    popularity_item_ctr,
    run_eda,
    schema_and_sanity,
    target_analysis,
)


def _lazy_bundle():
    frames = make_smoke_frames(n_events=2000, seed=42)
    return {k: v.lazy() for k, v in frames.items()}


def test_schema_and_target_on_2000_events() -> None:
    frames = _lazy_bundle()
    tracker = SizeTracker()
    sanity = schema_and_sanity(frames["train"], tracker)
    assert sanity["n_events"] == 2000
    assert sanity["rows_per_event_min"] == 10
    assert sanity["rows_per_event_max"] == 10
    assert sanity["events_multi_click"] == 0
    target = target_analysis(frames["train"], tracker)
    assert 0.0 < target["row_ctr"] < 1.0


def test_remaining_analyses_and_full_run(tmp_path) -> None:
    frames = _lazy_bundle()
    tracker = SizeTracker()
    train_events = event_level(frames["train"], has_clicked=True, tracker=tracker, name="train_events")
    eval_events = event_level(frames["eval"], has_clicked=False, tracker=tracker, name="eval_events")
    fold_events = {
        "fold_2025_01_25": event_level(
            frames["fold_2025_01_25"], has_clicked=True, tracker=tracker, name="e25"
        ),
        "fold_2025_01_26": event_level(
            frames["fold_2025_01_26"], has_clicked=True, tracker=tracker, name="e26"
        ),
    }
    folds = {
        "fold_2025_01_25": frames["fold_2025_01_25"],
        "fold_2025_01_26": frames["fold_2025_01_26"],
    }
    card = cardinality_and_overlap(frames["train"], frames["eval"], train_events, eval_events, tracker)
    assert card["n_eval_users"] > 0
    pop = popularity_item_ctr(frames["train"], folds, tracker, min_impr=10)
    assert "fold_2025_01_25" in pop["fold_aucs"]
    hist = history_analysis(frames["train"], train_events, eval_events, tracker)
    assert hist["in_vs_out"].height in (1, 2)
    cand = candidate_set_structure(frames["train"], tracker)
    assert cand["by_rowid"].height == 10
    shift = distribution_shift(
        frames["train"], frames["eval"], folds, train_events, eval_events, fold_events, tracker
    )
    assert "eval" in shift["time_bucket"]
    cold = cold_start_slices(frames["train"], frames["eval"], folds, eval_events, fold_events, tracker)
    assert "eval_vs_all_train" in cold

    out = tmp_path / "reports"
    results = run_eda(frames["train"], frames["eval"], folds, out, tracker=SizeTracker())
    assert (out / "01_eda.md").exists()
    assert len(results["findings"]) <= 10
    assert (out / "figures").exists()
    md = (out / "01_eda.md").read_text(encoding="utf-8")
    assert "Key findings" in md
    assert "Recommended features" in md
