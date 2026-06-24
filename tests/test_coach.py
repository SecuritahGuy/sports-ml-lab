"""Tests for coach feature computation."""

import pandas as pd

from sportslab.features.build_features import TARGET_COLUMN
from sportslab.features.coach import COACH_FEATURE_COLUMNS, compute_coach_features


def _make_minimal_df() -> pd.DataFrame:
    rows = []
    for season in [2021, 2022]:
        for week in range(1, 4):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-09-0{week}",
                    "home_team": "CHI",
                    "away_team": "GB",
                    "home_coach": "Matt Eberflus",
                    "away_coach": "Matt LaFleur",
                    "home_win": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    df[TARGET_COLUMN] = df["home_win"]
    return df


class TestCoachColumns:
    def test_feature_columns_defined(self):
        assert len(COACH_FEATURE_COLUMNS) > 0

    def test_home_coach_tenure_computed(self):
        df = compute_coach_features(_make_minimal_df())
        assert "home_coach_tenure" in df.columns

    def test_away_coach_tenure_computed(self):
        df = compute_coach_features(_make_minimal_df())
        assert "away_coach_tenure" in df.columns

    def test_home_coach_wins_computed(self):
        df = compute_coach_features(_make_minimal_df())
        assert "home_coach_career_wins" in df.columns

    def test_home_coach_games_computed(self):
        df = compute_coach_features(_make_minimal_df())
        assert "home_coach_career_games" in df.columns

    def test_home_coach_win_pct_computed(self):
        df = compute_coach_features(_make_minimal_df())
        assert "home_coach_win_pct" in df.columns

    def test_away_coach_win_pct_computed(self):
        df = compute_coach_features(_make_minimal_df())
        assert "away_coach_win_pct" in df.columns

    def test_all_coach_columns_present(self):
        df = compute_coach_features(_make_minimal_df())
        for col in COACH_FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_coach_tenure_increments_over_time(self):
        df = compute_coach_features(_make_minimal_df())
        df_home = df[df["week"] == 2]
        df_home_prev = df[df["week"] == 1]
        assert df_home["home_coach_tenure"].iloc[0] >= df_home_prev["home_coach_tenure"].iloc[0]

    def test_first_game_tenure(self):
        df = compute_coach_features(_make_minimal_df())
        first = df.iloc[0]
        assert first["home_coach_tenure"] >= 0
