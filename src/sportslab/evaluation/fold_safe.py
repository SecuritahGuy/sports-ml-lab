"""Reusable fold-safe evaluation helpers for rolling-origin experiments.

Provides a consistent interface for:
    - Loading and building the standard feature spine
    - Iterating rolling-origin folds with train/val masks
    - Running fold-safe cross-validation with a caller-supplied model function
    - Scoring the 2025 holdout
    - Checking promotion against the v3.0.0 champion

Typical usage::
    from sportslab.evaluation.fold_safe import (
        fold_safe_cv,
        score_holdout,
        check_promotion,
        INCUMBENT_VAL_LL,
        INCUMBENT_HOLDOUT_LL,
    )

    df = load_feature_table()
    df = build_base_features(df)
    results = fold_safe_cv(df, model_fn)
    val_ll = results["avg_log_loss"]
    holdout = score_holdout(df, model_fn)
    holdout_ll = holdout["log_loss"]
    verdict = check_promotion(val_ll, holdout_ll)
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import compute_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

INCUMBENT_VAL_LL = 0.6305
INCUMBENT_HOLDOUT_LL = 0.6200
MIN_PROMOTION_DELTA = 0.001
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2

ELO_PARAMS = {
    "k_factor": BEST_K,
    "home_advantage": BEST_HFA,
    "preseason_regression": BEST_REG,
    "decay_half_life": BEST_DECAY,
}


def load_feature_table(path: str = FEATURE_TABLE_PATH) -> pd.DataFrame:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    return pd.read_parquet(fp)


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    overrides = build_team_regression_overrides(
        df,
        preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df,
        team_regression_overrides=overrides,
        **ELO_PARAMS,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)
    return df


def fold_masks(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    masks = []
    for train_seasons, val_season in ROLLING_FOLDS:
        train_mask = (
            df["season"].isin(train_seasons).values
            & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
        )
        val_mask = (
            (df["season"] == val_season).values
            & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
        )
        masks.append((train_mask, val_mask))
    return masks


def fold_safe_cv(
    df: pd.DataFrame,
    model_fn: Callable,
    score_fn: Callable = compute_metrics,
) -> Dict[str, float]:
    fold_metrics = []
    for train_mask, val_mask in fold_masks(df):
        val_preds = model_fn(df, train_mask, val_mask)
        y_true = df.loc[val_mask, "home_win"].values
        m = score_fn(y_true, val_preds)
        fold_metrics.append(m)
    avg_ll = float(np.mean([m["log_loss"] for m in fold_metrics if "log_loss" in m]))
    return {
        "fold_metrics": fold_metrics,
        "avg_log_loss": round(avg_ll, 4),
    }


def score_holdout(
    df: pd.DataFrame,
    model_fn: Callable,
    score_fn: Callable = compute_metrics,
) -> Dict[str, float]:
    hold_mask = (
        (df["season"] == HOLDOUT_SEASON).values
        & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    )
    val_preds = model_fn(df, hold_mask, hold_mask)
    y_true = df.loc[hold_mask, "home_win"].values
    return score_fn(y_true, val_preds)


def check_promotion(
    val_ll: float,
    holdout_ll: float,
    incumbent_val: float = INCUMBENT_VAL_LL,
    incumbent_holdout: float = INCUMBENT_HOLDOUT_LL,
    delta: float = MIN_PROMOTION_DELTA,
) -> Dict[str, object]:
    beats_val = val_ll <= incumbent_val - delta
    beats_holdout = holdout_ll <= incumbent_holdout - delta
    promoted = beats_val and beats_holdout
    return {
        "promoted": promoted,
        "beats_val": beats_val,
        "beats_holdout": beats_holdout,
        "val_delta": round(val_ll - incumbent_val, 4),
        "holdout_delta": round(holdout_ll - incumbent_holdout, 4),
    }
