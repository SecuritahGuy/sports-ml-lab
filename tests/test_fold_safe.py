"""Tests for fold_safe experiment helper."""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.fold_safe import (
    check_promotion,
    fold_masks,
    fold_safe_cv,
    load_feature_table,
    score_holdout,
)


@pytest.fixture
def tiny_df():
    """Tiny synthetic dataset: 12 games across 2021-2025, random features."""
    rng = np.random.default_rng(42)
    rows = []
    for i, season in enumerate([2021, 2021, 2021, 2022, 2022, 2022,
                                2023, 2023, 2023, 2024, 2024, 2024,
                                2025, 2025]):
        rows.append({
            "game_id": f"g{i:02d}",
            "season": season,
            "week": (i % 3) + 1,
            "home_team": f"Team{(i % 4) + 1}",
            "away_team": f"Team{((i + 2) % 4) + 1}",
            "home_win": 1.0 if rng.random() > 0.5 else 0.0,
            "home_score": 24.0 + rng.random() * 10,
            "away_score": 17.0 + rng.random() * 10,
            "elo_prob": 0.5 + rng.random() * 0.1,
            "model_eligible": True,
        })
    df = pd.DataFrame(rows)
    df["home_qb_changed"] = 0
    df["away_qb_changed"] = 0
    df["home_rolling_mov_3"] = 0.0
    df["away_rolling_mov_3"] = 0.0
    return df


def test_fold_masks_excludes_holdout():
    df = pd.DataFrame({
        "season": [2021, 2022, 2023, 2024, 2025],
        "home_win": [1.0, 0.0, 1.0, 0.0, 1.0],
        "home_score": [24.0, 17.0, 31.0, 14.0, 27.0],
        "away_score": [10.0, 31.0, 24.0, 20.0, 24.0],
        "model_eligible": [True] * 5,
        "home_qb_changed": [0] * 5,
        "away_qb_changed": [0] * 5,
        "home_rolling_mov_3": [0.0] * 5,
        "away_rolling_mov_3": [0.0] * 5,
    })
    masks = fold_masks(df)
    for train_mask, val_mask in masks:
        assert not val_mask[df["season"] == HOLDOUT_SEASON].any(), (
            "Holdout season must not appear in any validation fold"
        )


def test_fold_masks_covers_all_seasons(tiny_df):
    masks = fold_masks(tiny_df)
    assert len(masks) == len(ROLLING_FOLDS)
    for (train_ms, val_ms), (_, val_season) in zip(masks, ROLLING_FOLDS):
        val_in_mask = tiny_df.loc[val_ms, "season"].unique()
        assert set(val_in_mask) == {val_season}


def test_fold_safe_cv_const_model(tiny_df):
    def const_model(df, train_mask, val_mask):
        return np.full(val_mask.sum(), 0.5)

    results = fold_safe_cv(tiny_df, const_model)
    assert "avg_log_loss" in results
    assert "fold_metrics" in results
    assert len(results["fold_metrics"]) == len(ROLLING_FOLDS)
    for fm in results["fold_metrics"]:
        assert "log_loss" in fm
        assert "brier" in fm
        assert "accuracy" in fm


def test_fold_safe_cv_train_val_separation(tiny_df):
    """Verify model_fn receives only train rows for fitting."""
    def isolation_model(df, train_mask, val_mask):
        train_seasons = set(df.loc[train_mask, "season"])
        val_seasons = set(df.loc[val_mask, "season"])
        assert train_seasons.isdisjoint(val_seasons), (
            "Train and validation seasons must not overlap"
        )
        return np.full(val_mask.sum(), 0.6)

    results = fold_safe_cv(tiny_df, isolation_model)
    assert results["avg_log_loss"] > 0


def test_score_holdout_isolated(tiny_df):
    def const_model(df, train_mask, val_mask):
        return np.full(val_mask.sum(), 0.5)

    hold = score_holdout(tiny_df, const_model)
    assert "log_loss" in hold
    assert hold["log_loss"] > 0


def test_check_promotion_both_required():
    r = check_promotion(val_ll=0.6200, holdout_ll=0.6100,
                        incumbent_val=0.6305, incumbent_holdout=0.6200)
    assert r["beats_val"]
    assert r["beats_holdout"]
    assert r["promoted"]


def test_check_promotion_val_only():
    r = check_promotion(val_ll=0.6200, holdout_ll=0.6250,
                        incumbent_val=0.6305, incumbent_holdout=0.6200)
    assert r["beats_val"]
    assert not r["beats_holdout"]
    assert not r["promoted"]


def test_check_promotion_holdout_only():
    r = check_promotion(val_ll=0.6350, holdout_ll=0.6100,
                        incumbent_val=0.6305, incumbent_holdout=0.6200)
    assert not r["beats_val"]
    assert r["beats_holdout"]
    assert not r["promoted"]


def test_check_promotion_neither():
    r = check_promotion(val_ll=0.6400, holdout_ll=0.6300,
                        incumbent_val=0.6305, incumbent_holdout=0.6200)
    assert not r["beats_val"]
    assert not r["beats_holdout"]
    assert not r["promoted"]


def test_check_promotion_delta_threshold():
    r = check_promotion(val_ll=0.6296, holdout_ll=0.6191,
                        incumbent_val=0.6305, incumbent_holdout=0.6200,
                        delta=0.001)
    assert not r["beats_val"]
    assert not r["beats_holdout"]
    assert not r["promoted"]


def test_check_promotion_delta_exact():
    r = check_promotion(val_ll=0.6295, holdout_ll=0.6190,
                        incumbent_val=0.6305, incumbent_holdout=0.6200,
                        delta=0.001)
    assert r["beats_val"]
    assert r["beats_holdout"]
    assert r["promoted"]


def test_check_promotion_deltas_correct():
    r = check_promotion(val_ll=0.6250, holdout_ll=0.6150,
                        incumbent_val=0.6305, incumbent_holdout=0.6200)
    assert r["val_delta"] == -0.0055
    assert r["holdout_delta"] == -0.0050


def test_load_feature_table_exists():
    df = load_feature_table()
    assert len(df) > 0
    assert "season" in df.columns
    assert "home_team" in df.columns
    assert "away_team" in df.columns


def test_load_feature_table_min_season():
    df = load_feature_table()
    assert df["season"].min() >= 2000
