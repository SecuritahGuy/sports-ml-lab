"""Tests for the Elo tuning and calibration module."""

import numpy as np
import pandas as pd

from sportslab.evaluation.elo_tuning import (
    _filter_df,
    _fit_isotonic,
    _fit_platt,
    _minimal_logistic_features,
    run_elo_grid_search,
)


class TestFilterDf:
    def test_removes_ties_and_neutral(self):
        df = pd.DataFrame(
            {
                "model_eligible": [True, True, False],
                "is_neutral": [False, True, False],
                "season": [2024, 2024, 2024],
            }
        )
        result = _filter_df(df)
        assert len(result) == 1

    def test_all_eligible_non_neutral_passes(self):
        df = pd.DataFrame(
            {
                "model_eligible": [True, True],
                "is_neutral": [False, False],
                "season": [2024, 2024],
            }
        )
        result = _filter_df(df)
        assert len(result) == 2


class TestPlattScaling:
    def test_fits_on_train_only(self):
        np.random.seed(42)
        train_prob = np.array([0.4, 0.5, 0.6, 0.45, 0.55])
        train_y = np.array([0, 0, 1, 0, 1])
        platt = _fit_platt(train_prob, train_y)
        preds = platt.predict_proba(train_prob.reshape(-1, 1))[:, 1]
        assert preds.shape == (5,)
        assert all(0 <= p <= 1 for p in preds)

    def test_monotonic(self):
        np.random.seed(0)
        train_prob = np.linspace(0.3, 0.7, 50)
        train_y = (np.random.rand(50) < train_prob).astype(int)
        platt = _fit_platt(train_prob, train_y)
        test_prob = np.linspace(0.3, 0.7, 10)
        preds = platt.predict_proba(test_prob.reshape(-1, 1))[:, 1]
        # Should be monotonic (higher input → same or higher output)
        diffs = np.diff(preds)
        assert np.all(diffs >= -1e-6)


class TestIsotonicCalibration:
    def test_fits_on_train_only(self):
        np.random.seed(42)
        train_prob = np.array([0.4, 0.5, 0.6, 0.45, 0.55])
        train_y = np.array([0, 0, 1, 0, 1])
        iso = _fit_isotonic(train_prob, train_y)
        preds = iso.predict(np.array([0.3, 0.5, 0.7]))
        assert preds.shape == (3,)
        assert all(0 <= p <= 1 for p in preds)

    def test_monotonic(self):
        np.random.seed(0)
        train_prob = np.linspace(0.3, 0.7, 50)
        train_y = (np.random.rand(50) < train_prob).astype(int)
        iso = _fit_isotonic(train_prob, train_y)
        test_prob = np.linspace(0.3, 0.7, 10)
        preds = iso.predict(test_prob)
        diffs = np.diff(preds)
        assert np.all(diffs >= -1e-6)


class TestMinimalLogisticFeatures:
    def test_selects_correct_columns(self):
        df = pd.DataFrame(
            {
                "elo_diff": [10.0, 20.0],
                "elo_prob": [0.55, 0.60],
                "rest_diff": [2, -1],
                "is_neutral": [False, False],
                "week": [1, 2],
                "other_col": ["a", "b"],
            }
        )
        result = _minimal_logistic_features(df)
        assert list(result.columns) == ["elo_diff", "elo_prob", "rest_diff", "is_neutral", "week"]
        assert "other_col" not in result.columns

    def test_handles_missing_columns(self):
        df = pd.DataFrame(
            {
                "elo_diff": [10.0],
                "week": [1],
            }
        )
        result = _minimal_logistic_features(df)
        assert "elo_diff" in result.columns
        assert "week" in result.columns
        assert "elo_prob" not in result.columns


class TestEloGridSearch:
    def test_returns_best_and_all_results(self):
        """Integration test on a tiny synthetic schedule."""
        np.random.seed(42)
        rows = []
        for season in [2021, 2022, 2023, 2024, 2025]:
            for week in range(1, 5):
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "gameday": f"{season}-09-0{week}",
                        "home_team": "TEAM_A" if np.random.rand() > 0.5 else "TEAM_B",
                        "away_team": "TEAM_B" if np.random.rand() > 0.5 else "TEAM_A",
                        "home_score": np.random.randint(10, 30),
                        "away_score": np.random.randint(10, 30),
                        "home_win": np.random.choice([0, 1]),
                    }
                )
        df = pd.DataFrame(rows)
        # Add target and flag columns like the real feature table
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

        # Save to temp parquet
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        df.to_parquet(tmp.name, index=False)

        best, all_results = run_elo_grid_search(feature_table_path=tmp.name)

        assert "params" in best
        assert "k_factor" in best["params"]
        assert "home_advantage" in best["params"]
        assert "preseason_regression" in best["params"]
        assert len(all_results) > 0
        # Best should have the lowest val log loss (allow float tolerance)
        assert best["val_log_loss"] <= all_results[0]["val_log_loss"] + 1e-5

        import os

        os.unlink(tmp.name)


# Import constants needed by test
from sportslab.features.build_features import (  # noqa: E402
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
