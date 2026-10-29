"""Submission file / frame validation."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from techarena.io import validate_submission_frame, write_submission

FIXTURE = Path(__file__).parent / "fixtures" / "sample_submission.csv"


def test_fixture_submission_is_valid() -> None:
    frame = pl.read_csv(FIXTURE)
    validate_submission_frame(frame)


def test_rejects_out_of_range_probabilities() -> None:
    frame = pl.DataFrame({"rowid": [1, 2], "clicked": [0.1, 1.2]})
    with pytest.raises(ValueError, match="clicked values must be in"):
        validate_submission_frame(frame)


def test_rejects_duplicate_rowids() -> None:
    frame = pl.DataFrame({"rowid": [1, 1], "clicked": [0.1, 0.2]})
    with pytest.raises(ValueError, match="not unique"):
        validate_submission_frame(frame)


def test_rejects_nulls() -> None:
    frame = pl.DataFrame({"rowid": [1, 2], "clicked": [0.1, None]})
    with pytest.raises(ValueError, match="nulls"):
        validate_submission_frame(frame)


def test_write_submission_roundtrip(tmp_path: Path) -> None:
    frame = pl.read_csv(FIXTURE)
    path = tmp_path / "sub.csv"
    write_submission(frame, path)
    again = pl.read_csv(path)
    validate_submission_frame(again)


def test_expected_rowids_mismatch() -> None:
    frame = pl.read_csv(FIXTURE)
    with pytest.raises(ValueError, match="rowids do not match"):
        validate_submission_frame(frame, expected_rowids=pl.Series("rowid", [999]))
