"""Project paths, column names, and global constants."""

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
FOLDS_PARQUET = FOLDS_DIR / "folds.parquet"

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
