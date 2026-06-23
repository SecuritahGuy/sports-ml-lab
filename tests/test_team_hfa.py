"""Tests for team-specific HFA computation."""

import pandas as pd

from sportslab.evaluation.team_hfa_experiment import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    run_team_hfa_experiment,
)
from sportslab.features.hfa import compute_team_hfa, margin_to_elo_hfa
from sportslab.features.ratings import (
    compute_elo_features,
)


def _mini_df() -> pd.DataFrame:
    rows = []
    for season in [2021, 2022]:
        for week in [1, 2, 3]:
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-09-{week:02d}",
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_win": 1,
                    "home_score": 31,
                    "away_score": 17,
                    "away_rest": 7,
                    "home_rest": 7,
                    "location": "Home",
                    "roof": "dome",
                    "surface": "turf",
                    "model_eligible": True,
                    "is_neutral": False,
                    "is_tie": False,
                }
            )
    return pd.DataFrame(rows)


class TestTeamHFA:
    def test_compute_team_hfa_returns_dict(self):
        df = _mini_df()
        hfa = compute_team_hfa(df, [2021])
        assert isinstance(hfa, dict)
        assert "KC" in hfa

    def test_team_hfa_home_advantage(self):
        """Team playing at home should have positive margin advantage."""
        df = _mini_df()
        hfa = compute_team_hfa(df, [2021])
        assert hfa.get("KC", -999) >= 0  # KC won big at home

    def test_team_hfa_away_team_negative_margin(self):
        """Visiting team should have lower or negative margin advantage."""
        df = _mini_df()
        hfa = compute_team_hfa(df, [2021])
        assert hfa.get("BUF", 999) <= 0  # BUF lost on the road

    def test_empty_seasons_returns_empty(self):
        df = _mini_df()
        hfa = compute_team_hfa(df, [])
        assert hfa == {}

    def test_margin_to_elo_hfa_capped(self):
        assert margin_to_elo_hfa(100.0) <= 30.0
        assert margin_to_elo_hfa(-100.0) >= -30.0

    def test_margin_to_elo_hfa_zero(self):
        assert margin_to_elo_hfa(0.0) == 0.0

    def test_team_hfa_modifies_elo_prob(self):
        df = _mini_df()
        team_hfa_dict = {"KC": 10.0, "BUF": -10.0}
        base = compute_elo_features(df, k_factor=20)
        adjusted = compute_elo_features(df, k_factor=20, team_hfa=team_hfa_dict)
        # KC home game should have higher win prob with positive HFA offset
        assert adjusted["elo_prob"].iloc[0] > base["elo_prob"].iloc[0]

    def test_team_hfa_none_matches_default(self):
        df = _mini_df()
        default = compute_elo_features(df, k_factor=20)
        explicit = compute_elo_features(df, k_factor=20, team_hfa=None)
        pd.testing.assert_frame_equal(default, explicit)


class TestExperiment:
    def test_importable(self):
        assert callable(run_team_hfa_experiment)

    def test_folds_exclude_holdout(self):
        for train_s, val_s in ROLLING_FOLDS:
            all_seasons = list(train_s) + [val_s]
            assert HOLDOUT_SEASON not in all_seasons

    def test_folds_sequential(self):
        expected = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
        assert ROLLING_FOLDS == expected
