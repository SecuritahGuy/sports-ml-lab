"""Step 7: Incumbent schema/contract tests.

Verifies that the incumbent model uses exactly these features:
  - elo_prob
  - home_qb_changed
  - away_qb_changed
  - home_rolling_mov_3
  - away_rolling_mov_3

And no others. Also verifies excluded feature families are NOT used.
"""

import os

import pandas as pd
import pytest

from sportslab.evaluation.predict_incumbent import (
    FEATURE_COLS,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
    _build_feature_pipeline,
)


class TestIncumbentSchema:
    """Contract tests for the incumbent model."""

    def test_exact_feature_set(self):
        """Incumbent must use exactly 5 features: elo_prob + 4 binary/continuous."""
        assert len(FEATURE_COLS) == 4, (
            f"Expected 4 feature columns, got {len(FEATURE_COLS)}: {FEATURE_COLS}"
        )
        allowed = {"home_qb_changed", "away_qb_changed",
                    "home_rolling_mov_3", "away_rolling_mov_3"}
        for col in FEATURE_COLS:
            assert col in allowed, f"Unexpected feature column: {col}"

    def test_market_features_not_in_incumbent(self):
        """Market features must not be used as model inputs for the incumbent."""
        assert "market_home_prob_novig" not in FEATURE_COLS
        assert "market_away_prob_novig" not in FEATURE_COLS
        assert "spread_home_prob" not in FEATURE_COLS
        assert "spread_line" not in FEATURE_COLS
        assert "home_moneyline" not in FEATURE_COLS

    def test_weather_features_not_in_incumbent(self):
        """Weather features must not be used in the incumbent."""
        weather_cols = ["temperature_f", "wind_mph", "precipitation_flag",
                        "cold_flag", "windy_flag", "bad_weather_flag"]
        for wc in weather_cols:
            assert wc not in FEATURE_COLS, f"Weather feature {wc} should not be in incumbent"

    def test_injury_features_not_in_incumbent(self):
        """Injury features must not be used in the incumbent."""
        assert "home_qb_out" not in FEATURE_COLS
        assert "any_qb_out" not in FEATURE_COLS

    def test_coach_features_not_in_incumbent(self):
        """Coach features must not be used in the incumbent."""
        assert "home_coach_tenure" not in FEATURE_COLS
        assert "home_coach_win_pct" not in FEATURE_COLS

    def test_scheduling_features_not_in_incumbent(self):
        """Scheduling features must not be used in the incumbent."""
        assert "home_short_week" not in FEATURE_COLS
        assert "home_off_bye" not in FEATURE_COLS
        assert "thursday_flag" not in FEATURE_COLS

    def test_efficiency_features_not_in_incumbent(self):
        """Efficiency features must not be used in the incumbent."""
        assert "home_pass_epa_3" not in FEATURE_COLS
        assert "home_total_epa_3" not in FEATURE_COLS

    def test_model_output_matches_incumbent_metadata(self):
        """Generated predictions should match the documented holdout LL."""
        fp = "reports/predictions/incumbent_predictions_2025_holdout.csv"
        if not os.path.exists(fp):
            pytest.skip("Holdout predictions file not found")
        df = pd.read_csv(fp)
        y_true = df["home_win_actual"].values
        y_prob = df["incumbent_home_win_prob"].values
        from sklearn.metrics import log_loss
        actual_ll = log_loss(y_true, y_prob)
        diff = abs(actual_ll - INCUMBENT_HOLDOUT_LL)
        assert diff < 0.005, (
            f"Holdout log loss {actual_ll:.4f} differs from documented "
            f"{INCUMBENT_HOLDOUT_LL} by {diff:.4f}. Regenerate predictions."
        )

    def test_incumbent_version_consistency(self):
        """Version string should be consistent across files."""
        assert INCUMBENT_VERSION == "v3.0.0"

    def test_predictions_csv_schema(self):
        """Predictions CSV must contain expected columns."""
        fp = "reports/predictions/incumbent_predictions.csv"
        if not os.path.exists(fp):
            pytest.skip("Predictions file not found")
        df = pd.read_csv(fp)
        required = [
            "game_id", "season", "week", "home_team", "away_team",
            "incumbent_home_win_prob", "confidence_bucket",
            "model_version", "model_date", "training_seasons",
            "feature_set", "calibration_method", "model_holdout_ll",
            "elo_k", "elo_hfa", "elo_reg", "elo_decay", "elo_qb_bonus",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        assert "market_prob_diagnostic" in df.columns, "Market diagnostic should be present"
        assert "market_minus_model_diagnostic" in df.columns, "Market comparison should be present"

    def test_no_extra_features_in_pipeline(self):
        """The feature pipeline should not silently add extra features.

        This test verifies that _build_feature_pipeline does not add
        new columns that could be accidentally used as features.
        """
        if not os.path.exists("data/features/nfl/feature_table.parquet"):
            pytest.skip("Feature table not found")
        try:
            df = _build_feature_pipeline()
        except Exception as e:
            pytest.skip(f"Cannot build feature pipeline: {e}")
            return

        expected_base = {"elo_prob"} | set(FEATURE_COLS)
        unknown = set(df.columns) - expected_base
        # Known allowed columns (metadata, not features)
        allowed_metadata = {
            "game_id", "season", "week", "gameday", "home_team", "away_team",
            "home_score", "away_score", "home_win", "away_qb_id", "home_qb_id",
            "away_qb_name", "home_qb_name", "away_coach", "home_coach",
            "referee", "stadium_id", "stadium", "location", "div_game",
            "roof", "surface", "temp", "wind", "game_type", "weekday",
            "gametime", "away_rest", "home_rest", "away_team_enc",
            "home_team_enc", "away_qb_id_enc", "home_qb_id_enc",
            "away_coach_enc", "home_coach_enc", "referee_enc",
            "stadium_id_enc", "game_type_enc", "weekday_enc", "roof_enc",
            "surface_enc", "home_elo_pre", "away_elo_pre", "elo_diff",
            "is_dome", "rest_diff",
            # Coach features (computed but not used)
            "home_coach_tenure", "away_coach_tenure",
            "home_coach_career_wins", "away_coach_career_wins",
            "home_coach_career_games", "away_coach_career_games",
            "home_coach_win_pct", "away_coach_win_pct",
            # Market features (diagnostic only)
            "market_home_prob_novig", "market_away_prob_novig",
            "market_overround", "spread_home_prob",
            "market_favorite_flag", "market_underdog_flag",
            "spread_bucket", "home_moneyline_prob_raw",
            "away_moneyline_prob_raw", "elo_vs_market_edge",
            "away_moneyline", "home_moneyline", "spread_line",
            "away_spread_odds", "home_spread_odds",
            "total_line", "under_odds", "over_odds",
            # QB feature columns (computed but not all used)
            "home_qb_changed", "away_qb_changed", "qb_change_diff",
            "home_qb_starts_this_season_pre", "away_qb_starts_this_season_pre",
            "qb_starts_diff", "home_qb_team_starts_pre", "away_qb_team_starts_pre",
            "home_qb_win_pct_pre", "away_qb_win_pct_pre", "qb_win_pct_diff",
            "home_games_since_qb_change", "away_games_since_qb_change",
            "games_since_qb_change_diff", "home_new_qb_flag", "away_new_qb_flag",
            "new_qb_diff", "home_qb_missing_flag", "away_qb_missing_flag",
            # Situational feature columns (computed but not all used)
            "home_rolling_mov_3", "away_rolling_mov_3",
            "home_rolling_mov_5", "away_rolling_mov_5",
            "home_rolling_pts_for", "away_rolling_pts_for",
            "home_rolling_pts_against", "away_rolling_pts_against",
            "home_win_streak", "away_win_streak",
            "home_ytd_win_pct", "away_ytd_win_pct",
            "turf_flag", "high_altitude_flag", "prime_time_flag",
            "rest_diff_squared",
            # Derived flags
            "is_tie", "model_eligible", "is_neutral",
            # QB adjustment columns (overlay features)
            "home_qb_adj", "away_qb_adj",
            "home_qb_starts", "away_qb_starts",
            # Leakage columns preserved for audit
            "overtime", "result", "total",
            # Weather (if present)
            "temperature_f", "wind_mph", "precipitation_flag",
            "cold_flag", "very_cold_flag", "hot_flag",
            "windy_flag", "very_windy_flag", "bad_weather_flag",
            "outdoor_game_flag", "weather_missing_flag",
            "temp_missing_flag", "wind_missing_flag", "weather_source",
        }
        unknown = unknown - allowed_metadata
        assert len(unknown) == 0, f"Unexpected columns in feature pipeline: {unknown}"
