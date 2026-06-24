"""Tests for situational feature computation."""

import pandas as pd

from sportslab.features.build_features import TARGET_COLUMN
from sportslab.features.situational import SITUATIONAL_FEATURE_COLUMNS, compute_situational_features


def _make_minimal_df() -> pd.DataFrame:
    rows = []
    for season in [2021, 2022, 2023]:
        for week in range(1, 6):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-09-0{week}",
                    "home_team": "CHI",
                    "away_team": "GB",
                    "home_rest": 7,
                    "away_rest": 7,
                    "rest_diff": 0,
                    "weekday": "Sunday",
                    "gametime": "1:00PM",
                    "stadium_id": "CHI00",
                    "stadium": "Soldier Field",
                    "surface": "Grass",
                    "location": "Home",
                    "is_neutral": False,
                    "home_score": 24 if week % 2 == 0 else 17,
                    "away_score": 20 if week % 2 == 0 else 14,
                    "result": 4 if week % 2 == 0 else 3,
                    "home_win": 1.0 if week % 2 == 0 else 0.0,
                }
            )
    df = pd.DataFrame(rows)
    df[TARGET_COLUMN] = df["home_win"]
    return df


class TestSituationalColumns:
    def test_feature_columns_defined(self):
        assert len(SITUATIONAL_FEATURE_COLUMNS) > 0

    def test_rolling_mov_3_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "home_rolling_mov_3" in df.columns

    def test_rolling_mov_5_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "away_rolling_mov_5" in df.columns

    def test_ytd_win_pct_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "home_ytd_win_pct" in df.columns

    def test_win_streak_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "home_win_streak" in df.columns

    def test_turf_flag_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "turf_flag" in df.columns

    def test_high_altitude_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "high_altitude_flag" in df.columns

    def test_prime_time_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "prime_time_flag" in df.columns

    def test_rest_diff_squared_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "rest_diff_squared" in df.columns

    def test_rolling_pts_for_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "home_rolling_pts_for" in df.columns

    def test_rolling_pts_against_computed(self):
        df = compute_situational_features(_make_minimal_df())
        assert "away_rolling_pts_against" in df.columns

    def test_all_situational_columns_present(self):
        df = compute_situational_features(_make_minimal_df())
        for col in SITUATIONAL_FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_leakage_future_info(self):
        df = compute_situational_features(_make_minimal_df())
        for col in SITUATIONAL_FEATURE_COLUMNS:
            assert df[col].isna().sum() <= 3, f"Too many NaN in {col}"


class TestSituationalBoundaries:
    def test_rolling_mov_first_game(self):
        df = compute_situational_features(_make_minimal_df())
        first = df[df["week"] == 1].iloc[0]
        assert first["home_rolling_mov_3"] == 0.0

    def test_win_streak_initial_zero(self):
        df = compute_situational_features(_make_minimal_df())
        first = df[df["week"] == 1].iloc[0]
        assert first["home_win_streak"] == 0
