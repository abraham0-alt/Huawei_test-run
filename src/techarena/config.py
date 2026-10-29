"""Project paths, column names, and global constants."""

from datetime import datetime, timezone
from pathlib import Path

RANDOM_SEED = 42

# --- paths (all derived from the repository root) ---
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
FOLDS_DIR = DATA_DIR / "folds"
FEATURES_DIR = DATA_DIR / "features"

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
REPORTS_DIR = REPO_ROOT / "reports"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
MODELS_DIR = REPO_ROOT / "models"
TESTS_DIR = REPO_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"

# Expected local filenames (place originals under data/raw/)
TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
TRAIN_PARQUET = PARQUET_DIR / "train.parquet"
TEST_PARQUET = PARQUET_DIR / "test.parquet"
EVAL_PARQUET = PARQUET_DIR / "eval.parquet"
FOLDS_PARQUET = FOLDS_DIR / "folds.parquet"
FOLD_25_PARQUET = FOLDS_DIR / "fold_2025_01_25.parquet"
FOLD_26_PARQUET = FOLDS_DIR / "fold_2025_01_26.parquet"

# Local copies used for this machine (never committed)
DOWNLOADS_DIR = Path.home() / "Downloads"
EXTERNAL_TRAIN_PARQUET = DOWNLOADS_DIR / "train.parquet"
EXTERNAL_EVAL_PARQUET = DOWNLOADS_DIR / "eval.parquet"
EXTERNAL_FOLD_25_PARQUET = DOWNLOADS_DIR / "fold_2025_01_25.parquet"
EXTERNAL_FOLD_26_PARQUET = DOWNLOADS_DIR / "fold_2025_01_26.parquet"

# Competition time window (UTC). Train is [start, end).
TRAIN_START_UTC = datetime(2025, 1, 20, tzinfo=timezone.utc)
TRAIN_END_UTC = datetime(2025, 1, 27, tzinfo=timezone.utc)
FOLD_25_CUTOFF_UTC = datetime(2025, 1, 25, tzinfo=timezone.utc)
FOLD_26_CUTOFF_UTC = datetime(2025, 1, 26, tzinfo=timezone.utc)
TRAIN_START_TS = int(TRAIN_START_UTC.timestamp())
TRAIN_END_TS = int(TRAIN_END_UTC.timestamp())
FOLD_25_CUTOFF_TS = int(FOLD_25_CUTOFF_UTC.timestamp())
FOLD_26_CUTOFF_TS = int(FOLD_26_CUTOFF_UTC.timestamp())


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def resolve_competition_paths() -> dict[str, Path]:
    """Prefer repo-local parquet, then ~/Downloads copies."""
    train = first_existing(TRAIN_PARQUET, EXTERNAL_TRAIN_PARQUET)
    eval_path = first_existing(EVAL_PARQUET, EXTERNAL_EVAL_PARQUET)
    fold_25 = first_existing(FOLD_25_PARQUET, EXTERNAL_FOLD_25_PARQUET)
    fold_26 = first_existing(FOLD_26_PARQUET, EXTERNAL_FOLD_26_PARQUET)
    missing = [
        name
        for name, path in [
            ("train", train),
            ("eval", eval_path),
            ("fold_2025_01_25", fold_25),
            ("fold_2025_01_26", fold_26),
        ]
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            "Missing parquet files: "
            + ", ".join(missing)
            + ". Place them under data/parquet and data/folds, or in ~/Downloads."
        )
    return {
        "train": train,
        "eval": eval_path,
        "fold_2025_01_25": fold_25,
        "fold_2025_01_26": fold_26,
    }

# --- column names ---
COL_ROWID = "rowid"
COL_EVENT_ID = "event_id"
COL_USER_ID = "user_id"
COL_ITEM_ID = "item_id"
COL_TIMESTAMP = "timestamp"
COL_TIME_BUCKET = "time_bucket"
COL_DAY_TYPE = "day_type"
COL_HISTORY_ITEM_IDS = "history_item_ids"
COL_HISTORY_TIMESTAMPS = "history_timestamps"
COL_CLICKED = "clicked"

ID_COLUMNS = (
    COL_ROWID,
    COL_EVENT_ID,
    COL_USER_ID,
    COL_ITEM_ID,
)

CONTEXT_COLUMNS = (
    COL_TIMESTAMP,
    COL_TIME_BUCKET,
    COL_DAY_TYPE,
    COL_HISTORY_ITEM_IDS,
    COL_HISTORY_TIMESTAMPS,
)

TARGET_COLUMN = COL_CLICKED

SUBMISSION_COLUMNS = (COL_ROWID, COL_CLICKED)

# Impression lists are 10 candidate items per event.
IMPRESSIONS_PER_EVENT = 10
