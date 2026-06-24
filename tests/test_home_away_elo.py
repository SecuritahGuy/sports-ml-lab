"""Tests for home/away separate Elo ratings."""

import pandas as pd

from sportslab.features.build_features import TARGET_COLUMN
from sportslab.features.home_away_elo import compute_home_away_elo


def _make_minimal_df() -> pd.DataFrame:
    rows = []
    for season in [2021, 2022]:
        for week in range(1, 5):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-09-0{week}",
                    "home_team": "CHI",
                    "away_team": "GB",
                    "home_score": 24,
                    "away_score": 20,
                    "result": 4,
                    "home_win": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    df[TARGET_COLUMN] = df["home_win"]
    return df


class TestHomeAwayElo:
    def test_ha_prob_column_present(self):
        df = compute_home_away_elo(_make_minimal_df())
        assert "elo_prob" in df.columns

    def test_ha_diff_column_present(self):
        df = compute_home_away_elo(_make_minimal_df())
        assert "elo_diff" in df.columns

    def test_ha_prob_within_bounds(self):
        df = compute_home_away_elo(_make_minimal_df())
        assert df["elo_prob"].between(0, 1).all()

    def test_ha_elo_diff_within_bounds(self):
        df = compute_home_away_elo(_make_minimal_df())
        assert df["elo_diff"].between(-800, 800).all()
