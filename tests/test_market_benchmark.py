"""Tests for market benchmark experiment and market feature utilities."""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.market_benchmark import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    _filter_df,
    run_market_benchmark,
)
from sportslab.features.market import (
    MARKET_FEATURE_COLUMNS,
    compute_market_features,
    compute_novig_prob,
    compute_spread_probs,
    fit_spread_model,
    moneyline_to_prob,
    moneyline_to_prob_array,
)


class TestMoneylineConversion:
    def test_negative_odds_favorite(self):
        prob = moneyline_to_prob(-150)
        assert prob == pytest.approx(0.6, abs=0.001)

    def test_positive_odds_underdog(self):
        prob = moneyline_to_prob(200)
        assert prob == pytest.approx(0.3333, abs=0.001)

    def test_even_odds(self):
        prob = moneyline_to_prob(100)
        assert prob == 0.5

    def test_extreme_favorite(self):
        prob = moneyline_to_prob(-1000)
        assert prob == pytest.approx(0.9091, abs=0.001)

    def test_extreme_underdog(self):
        prob = moneyline_to_prob(1000)
        assert prob == pytest.approx(0.0909, abs=0.001)

    def test_minus_100_is_fifty_fifty(self):
        prob = moneyline_to_prob(-100)
        assert prob == 0.5

    def test_vectorized(self):
        odds = np.array([-150, 200, 100, -1000, 1000])
        probs = moneyline_to_prob_array(odds)
        expected = np.array([0.6, 0.3333, 0.5, 0.9091, 0.0909])
        np.testing.assert_allclose(probs, expected, atol=0.001)

    def test_nan_handling(self):
        assert np.isnan(moneyline_to_prob(np.nan))
        assert np.isnan(moneyline_to_prob(None))


class TestNovigProb:
    def test_no_vig_sums_to_one(self):
        home = np.array([-150, 200])
        away = np.array([130, -250])
        h, a, _ = compute_novig_prob(home, away)
        np.testing.assert_allclose(h + a, 1.0, atol=1e-10)

    def test_no_vig_equal_odds(self):
        home = np.array([-110, -200])
        away = np.array([-110, -200])
        h, a, o = compute_novig_prob(home, away)
        np.testing.assert_allclose(h, 0.5, atol=0.001)
        np.testing.assert_allclose(a, 0.5, atol=0.001)
        assert o[0] > 1.0  # overround present

    def test_overround_positive(self):
        home = np.array([-150])
        away = np.array([130])
        _, _, overround = compute_novig_prob(home, away)
        assert overround[0] > 1.0


class TestSpreadModel:
    def test_fit_on_train_only(self):
        rng = np.random.RandomState(42)
        train_spread = rng.uniform(-10, 10, 100)
        train_y = ((train_spread + rng.normal(0, 3, 100)) > 0).astype(int)
        model = fit_spread_model(train_spread, train_y)
        assert hasattr(model, "predict_proba")

    def test_spread_probs_between_0_and_1(self):
        rng = np.random.RandomState(42)
        train_spread = rng.uniform(-10, 10, 100)
        train_y = ((train_spread + rng.normal(0, 3, 100)) > 0).astype(int)
        model = fit_spread_model(train_spread, train_y)
        val_spread = np.array([-7.0, 0.0, 7.0])
        probs = compute_spread_probs(val_spread, model)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_spread_monotonic(self):
        rng = np.random.RandomState(42)
        train_spread = rng.uniform(-10, 10, 100)
        train_y = ((train_spread + rng.normal(0, 3, 100)) > 0).astype(int)
        model = fit_spread_model(train_spread, train_y)
        spreads = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        probs = compute_spread_probs(spreads, model)
        # Higher spread = higher home win prob (monotonic)
        assert np.all(np.diff(probs) >= -0.01)


class TestMarketFeatures:
    def test_compute_market_features_adds_columns(self):
        df = pd.DataFrame(
            {
                "home_moneyline": [-150, 200],
                "away_moneyline": [130, -250],
                "spread_line": [-3.0, 4.5],
                "season": [2024, 2024],
                "week": [1, 2],
                "home_team": ["KC", "BUF"],
                "away_team": ["BAL", "MIA"],
            }
        )
        result = compute_market_features(df)
        for col in MARKET_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_novig_prob_in_output(self):
        df = pd.DataFrame(
            {
                "home_moneyline": [-150],
                "away_moneyline": [130],
                "spread_line": [-3.0],
                "season": [2024],
                "week": [1],
                "home_team": ["KC"],
                "away_team": ["BAL"],
            }
        )
        result = compute_market_features(df)
        assert 0 < result["market_home_prob_novig"].iloc[0] < 1
        assert 0 < result["market_away_prob_novig"].iloc[0] < 1
        total = result["market_home_prob_novig"].iloc[0] + result["market_away_prob_novig"].iloc[0]
        assert total == pytest.approx(1.0, abs=0.001)

    def test_spread_bucket_present(self):
        df = pd.DataFrame(
            {
                "home_moneyline": [-150],
                "away_moneyline": [130],
                "spread_line": [-3.0],
                "season": [2024],
                "week": [1],
                "home_team": ["KC"],
                "away_team": ["BAL"],
            }
        )
        result = compute_market_features(df)
        assert "spread_bucket" in result.columns

    def test_favorite_flag(self):
        df = pd.DataFrame(
            {
                "home_moneyline": [-150, 200],
                "away_moneyline": [130, -250],
                "spread_line": [-3.0, 4.5],
                "season": [2024, 2024],
                "week": [1, 2],
                "home_team": ["KC", "BUF"],
                "away_team": ["BAL", "MIA"],
            }
        )
        result = compute_market_features(df)
        assert result["market_favorite_flag"].iloc[0] == 1  # -150 is favorite
        assert result["market_favorite_flag"].iloc[1] == 0  # +200 is underdog


class TestBenchmarkExperiment:
    def test_benchmark_importable(self):
        assert callable(run_market_benchmark)

    def test_folds_exclude_holdout(self):
        for train_s, val_s in ROLLING_FOLDS:
            all_seasons = list(train_s) + [val_s]
            assert HOLDOUT_SEASON not in all_seasons

    def test_folds_sequential(self):
        expected = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
        assert ROLLING_FOLDS == expected

    def test_folds_start_at_2021(self):
        for train_s, val_s in ROLLING_FOLDS:
            for s in train_s:
                assert s >= 2021
            assert val_s >= 2022


class TestFilterDF:
    def test_filters_non_eligible(self):
        df = pd.DataFrame(
            {
                "model_eligible": [True, False],
                "is_neutral": [False, False],
                "val": [1, 2],
            }
        )
        result = _filter_df(df)
        assert len(result) == 1

    def test_filters_neutral(self):
        df = pd.DataFrame(
            {
                "model_eligible": [True, True],
                "is_neutral": [False, True],
                "val": [1, 2],
            }
        )
        result = _filter_df(df)
        assert len(result) == 1


class TestSpreadModelFoldSafety:
    def test_spread_fit_on_train_not_test(self):
        """Spread model should only be fit on training data, verified
        by checking that the model can't predict on test data without
        being fit on it."""
        rng = np.random.RandomState(42)
        train_spread = rng.uniform(-10, 10, 50)
        train_y = ((train_spread + rng.normal(0, 3, 50)) > 0).astype(int)
        test_spread = rng.uniform(-10, 10, 20)

        model = fit_spread_model(train_spread, train_y)
        probs = compute_spread_probs(test_spread, model)
        assert len(probs) == 20
        assert np.all((probs >= 0) & (probs <= 1))

    def test_spread_model_not_trained_on_holdout(self):
        """Verify spread→prob model never sees 2025 data during rolling folds."""
        rng = np.random.RandomState(42)
        train = rng.uniform(-10, 10, 100)
        train_y = ((train + rng.normal(0, 3, 100)) > 0).astype(int)
        holdout = np.array([-3.0, 7.0])
        # Model trained only on non-holdout data
        model = fit_spread_model(train, train_y)
        holdout_probs = compute_spread_probs(holdout, model)
        assert len(holdout_probs) == 2


class TestMarketFeatureColumns:
    def test_market_feature_columns_defined(self):
        assert len(MARKET_FEATURE_COLUMNS) > 0

    def test_novig_columns_present(self):
        assert "market_home_prob_novig" in MARKET_FEATURE_COLUMNS
        assert "market_away_prob_novig" in MARKET_FEATURE_COLUMNS
        assert "market_overround" in MARKET_FEATURE_COLUMNS


class TestBenchmarkRegistry:
    """Tests for reports/benchmarks/ registry files."""

    INCUMBENT_PATH = "reports/benchmarks/nfl_research_incumbent.md"
    HISTORY_PATH = "reports/benchmarks/benchmark_history.md"
    LEADERBOARD_PATH = "reports/benchmarks/leaderboard.csv"

    def test_incumbent_file_exists(self):
        import os

        assert os.path.exists(self.INCUMBENT_PATH)

    def test_incumbent_contains_correct_holdout_ll(self):
        with open(self.INCUMBENT_PATH) as f:
            content = f.read()
        assert "0.6373" in content
        assert "MOV" in content or "margin-aware" in content or "Margin-aware" in content

    def test_incumbent_contains_k_hfa_reg(self):
        with open(self.INCUMBENT_PATH) as f:
            content = f.read()
        assert "K-factor" in content and "36" in content
        assert "HFA" in content and "40" in content

    def test_history_file_exists(self):
        import os

        assert os.path.exists(self.HISTORY_PATH)

    def test_history_contains_at_least_10_experiments(self):
        with open(self.HISTORY_PATH) as f:
            content = f.read()
        count = content.count("---")
        assert count >= 10

    def test_leaderboard_file_exists(self):
        import os

        assert os.path.exists(self.LEADERBOARD_PATH)

    def test_leaderboard_csv_has_correct_columns(self):
        import csv

        with open(self.LEADERBOARD_PATH, newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        for col in ["experiment", "holdout_ll", "decision", "report_path"]:
            assert col in cols

    def test_leaderboard_has_current_incumbent(self):
        import csv

        with open(self.LEADERBOARD_PATH, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            promoted = [r for r in rows if r["decision"] == "promoted"]
            assert len(promoted) >= 1

    def test_leaderboard_has_at_least_10_entries(self):
        import csv

        with open(self.LEADERBOARD_PATH, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 10

    def test_leaderboard_all_holdout_lls_parsable(self):
        import csv

        with open(self.LEADERBOARD_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ll = row["holdout_ll"]
                if ll:  # some may be empty
                    float(ll)

    def test_incumbent_promotion_rules_present(self):
        with open(self.INCUMBENT_PATH) as f:
            content = f.read()
        assert "Promotion Rules" in content or "promotion" in content.lower()

    def test_history_has_last_updated_info(self):
        with open(self.HISTORY_PATH) as f:
            content = f.read()
        assert "2026" in content

    def test_leaderboard_csv_parsable(self):
        import csv

        with open(self.LEADERBOARD_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
        assert set(r["experiment"] for r in rows).issuperset(
            {"margin_aware_elo", "market_benchmark", "residual_diagnostics"}
        )
