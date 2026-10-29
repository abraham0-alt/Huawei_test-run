"""Thin wrappers around gradient-boosted classifiers for click probability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from techarena.config import MODELS_DIR, RANDOM_SEED


def pooled_roc_auc(y_true, y_score) -> float:
    """ROC AUC over all rows (the competition metric)."""
    return float(roc_auc_score(y_true, y_score))


def default_lightgbm_params(**overrides: Any) -> dict[str, Any]:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
    }
    params.update(overrides)
    return params


def default_catboost_params(**overrides: Any) -> dict[str, Any]:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": RANDOM_SEED,
        "verbose": False,
    }
    params.update(overrides)
    return params


def save_prediction_array(values: np.ndarray, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else MODELS_DIR / "oof.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    return path
