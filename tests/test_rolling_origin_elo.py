"""Tests for rolling-origin Elo validation module."""

import numpy as np
import pandas as pd

from sportslab.evaluation.rolling_origin_elo_validation import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    _filter_df,
    run_rolling_origin_grid_search,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)

_TEAMS = [
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LV",
    "LAC",
    "LAR",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
]


def _build_synthetic_table(n_weeks: int = 6, seed: int = 42) -> str:
    """Build a synthetic feature table parquet with enough rows for both classes."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in [2021, 2022, 2023, 2024, 2025]:
        for week in range(1, n_weeks + 1):
            for _ in range(4):
                ht, at = rng.choice(_TEAMS, size=2, replace=False)
                home_win = int(rng.random() > 0.45)
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "gameday": f"{season}-{9 + week // 4:02d}-{1 + (week % 4) * 7:02d}",
                        "home_team": ht,
                        "away_team": at,
                        "home_score": int(rng.integers(10, 35)),
                        "away_score": int(rng.integers(10, 35)),
                        "home_win": home_win,
                    }
                )
    df = pd.DataFrame(rows)
    df["home_win"] = df["home_win"].astype(float)
    df[TARGET_COLUMN] = df["home_win"]
    df[MODEL_ELIGIBLE_COLUMN] = True
    df[NEUTRAL_COLUMN] = False
    df["is_neutral"] = False
    df["model_eligible"] = True
    df["location"] = "Home"
    df["rest_diff"] = 0
    df["is_tie"] = False
    df["roof"] = "outdoors"
    df["surface"] = "grass"

    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    df.to_parquet(tmp.name, index=False)
    return tmp.name


class TestRollingFolds:
    def test_three_folds(self):
        assert len(ROLLING_FOLDS) == 3

    def test_fold1_correct(self):
        train, val = ROLLING_FOLDS[0]
        assert train == [2021]
        assert val == 2022

    def test_fold2_correct(self):
        train, val = ROLLING_FOLDS[1]
        assert train == [2021, 2022]
        assert val == 2023

    def test_fold3_correct(self):
        train, val = ROLLING_FOLDS[2]
        assert train == [2021, 2022, 2023]
        assert val == 2024

    def test_no_seasons_before_2021(self):
        for train_seasons, val_season in ROLLING_FOLDS:
            for s in train_seasons + [val_season]:
                assert s >= 2021

    def test_holdout_not_in_folds(self):
        fold_seasons = set()
        for train_seasons, val_season in ROLLING_FOLDS:
            fold_seasons.update(train_seasons)
            fold_seasons.add(val_season)
        assert HOLDOUT_SEASON not in fold_seasons


class TestFilterDf:
    def test_removes_ties_and_neutral(self):
        df = pd.DataFrame(
            {
                "model_eligible": [True, True, False],
                "is_neutral": [False, True, False],
                "season": [2022, 2022, 2022],
            }
        )
        result = _filter_df(df)
        assert len(result) == 1

    def test_all_eligible_non_neutral_passes(self):
        df = pd.DataFrame(
            {
                "model_eligible": [True, True],
                "is_neutral": [False, False],
                "season": [2022, 2022],
            }
        )
        result = _filter_df(df)
        assert len(result) == 2


class TestGridSearchNoHoldout:
    def test_no_holdout_in_all_results(self):
        """Grid search entries should not contain holdout_log_loss by default."""
        tmp = _build_synthetic_table(n_weeks=6, seed=42)
        try:
            best, all_results = run_rolling_origin_grid_search(feature_table_path=tmp)
            assert "params" in best
            assert "k_factor" in best["params"]
            assert "home_advantage" in best["params"]
            assert "preseason_regression" in best["params"]
            assert "avg_val_log_loss" in best
            assert "holdout_log_loss" not in best
            assert len(all_results) > 0
            for entry in all_results:
                assert "holdout_log_loss" not in entry
                assert "avg_val_log_loss" in entry
                assert "fold_details" in entry
                assert len(entry["fold_details"]) == 3
            assert best["avg_val_log_loss"] <= all_results[0]["avg_val_log_loss"] + 1e-5
        finally:
            import os

            os.unlink(tmp)

    def test_no_holdout_accessed_in_grid_search(self):
        """Verify that 2025 holdout season is not used in fold computation."""
        tmp = _build_synthetic_table(n_weeks=4, seed=99)
        try:
            best, all_results = run_rolling_origin_grid_search(feature_table_path=tmp)
            for entry in all_results:
                for fd in entry["fold_details"]:
                    assert fd["val_season"] != 2025
        finally:
            import os

            os.unlink(tmp)


class TestOldGridSearchBackwardCompat:
    def test_compute_holdout_false_omits_holdout(self):
        from sportslab.evaluation.elo_tuning import run_elo_grid_search

        tmp = _build_synthetic_table(n_weeks=4, seed=42)
        try:
            best, all_results = run_elo_grid_search(feature_table_path=tmp, compute_holdout=False)
            assert "holdout_log_loss" not in best
            for entry in all_results:
                assert "holdout_log_loss" not in entry
        finally:
            import os

            os.unlink(tmp)

    def test_compute_holdout_true_includes_holdout(self):
        from sportslab.evaluation.elo_tuning import run_elo_grid_search

        tmp = _build_synthetic_table(n_weeks=4, seed=42)
        try:
            best, all_results = run_elo_grid_search(feature_table_path=tmp, compute_holdout=True)
            assert "holdout_log_loss" in best
            for entry in all_results:
                assert "holdout_log_loss" in entry
        finally:
            import os

            os.unlink(tmp)
