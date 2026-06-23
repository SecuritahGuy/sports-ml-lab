"""Tests for the Elo rating feature module — no leakage, no network."""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.ratings import (
    DEFAULT_ELO,
    MOV_CAPPED_LINEAR,
    MOV_CAPPED_LOG,
    MOV_LOG,
    MOV_NONE,
    MOV_SQRT,
    _mov_multiplier,
    compute_elo_features,
    elo_diff_to_win_prob,
)


class TestEloConversion:
    def test_equal_ratings(self):
        prob = elo_diff_to_win_prob(1500, 1500)
        assert prob == pytest.approx(0.5, abs=1e-6)

    def test_higher_home_rating(self):
        prob = elo_diff_to_win_prob(1600, 1400)
        assert prob > 0.5

    def test_lower_home_rating(self):
        prob = elo_diff_to_win_prob(1400, 1600)
        assert prob < 0.5

    def test_symmetric(self):
        p1 = elo_diff_to_win_prob(1500, 1400)
        p2 = elo_diff_to_win_prob(1400, 1500)
        assert p1 + p2 == pytest.approx(1.0, abs=1e-6)

    def test_home_advantage_increases_home_prob(self):
        """HFA should increase home win probability for equal ratings."""
        no_hfa = elo_diff_to_win_prob(1500, 1500, home_advantage=0)
        with_hfa = elo_diff_to_win_prob(1500, 1500, home_advantage=50)
        assert with_hfa > no_hfa
        assert with_hfa > 0.5

    def test_hfa_equivalent_to_higher_rating(self):
        """HFA of 50 should be similar to home team having 50 more Elo."""
        p_hfa = elo_diff_to_win_prob(1500, 1500, home_advantage=50)
        p_elo = elo_diff_to_win_prob(1550, 1500, home_advantage=0)
        assert p_hfa == pytest.approx(p_elo, abs=1e-6)


class TestComputeEloFeatures:
    def test_starts_at_default(self):
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_win": [1],
            }
        )
        result = compute_elo_features(df)
        assert result["home_elo_pre"].iloc[0] == DEFAULT_ELO
        assert result["away_elo_pre"].iloc[0] == DEFAULT_ELO
        assert result["elo_diff"].iloc[0] == 0.0
        assert "elo_prob" in result.columns

    def test_features_assigned_before_update(self):
        """Elo features for game N must use ratings from games before N."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_win": [1, 1],
            }
        )
        result = compute_elo_features(df)
        assert result["home_elo_pre"].iloc[1] > DEFAULT_ELO
        assert result["away_elo_pre"].iloc[1] == DEFAULT_ELO

    def test_tie_handled(self):
        """Ties should update ratings (0.5 for each team)."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_win": [pd.NA, 1],
            }
        )
        result = compute_elo_features(df)
        assert result["home_elo_pre"].iloc[0] == DEFAULT_ELO
        assert result["home_elo_pre"].iloc[1] == DEFAULT_ELO

    def test_elo_diff_equals_difference(self):
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_win": [1],
            }
        )
        result = compute_elo_features(df)
        assert result["elo_diff"].iloc[0] == pytest.approx(
            result["home_elo_pre"].iloc[0] - result["away_elo_pre"].iloc[0]
        )

    def test_sequential_processing(self):
        """Games should be processed in chronological order."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [2, 1],
                "gameday": ["2024-09-12", "2024-09-05"],
                "home_team": ["KC", "BAL"],
                "away_team": ["LV", "KC"],
                "home_win": [1, 0],
            }
        )
        result = compute_elo_features(df)
        kc_elo_week2 = result.loc[result["week"] == 2, "home_elo_pre"].iloc[0]
        assert kc_elo_week2 > DEFAULT_ELO

    def test_k_factor_impact(self):
        """Higher K-factor should produce larger rating changes."""
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_win": [1],
            }
        )
        # After the game, elo_diff only exists as pre-game; check second game
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "season": [2024],
                        "week": [2],
                        "gameday": ["2024-09-12"],
                        "home_team": ["KC"],
                        "away_team": ["LV"],
                        "home_win": [1],
                    }
                ),
            ],
            ignore_index=True,
        )
        r_low = compute_elo_features(df, k_factor=4)
        r_high = compute_elo_features(df, k_factor=32)
        # KC should have higher Elo in game 2 with higher K (more reward for win)
        assert r_high["home_elo_pre"].iloc[1] > r_low["home_elo_pre"].iloc[1]

    def test_home_advantage_in_elo_prob(self):
        """HFA should affect elo_prob but not elo_diff."""
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_win": [1],
            }
        )
        no_hfa = compute_elo_features(df, home_advantage=0)
        with_hfa = compute_elo_features(df, home_advantage=50)
        # elo_diff should be the same
        assert no_hfa["elo_diff"].iloc[0] == with_hfa["elo_diff"].iloc[0]
        # elo_prob should be higher with HFA
        assert with_hfa["elo_prob"].iloc[0] > no_hfa["elo_prob"].iloc[0]
        assert with_hfa["elo_prob"].iloc[0] > 0.5

    def test_preseason_regression(self):
        """Ratings should regress toward default_elo at season boundaries."""
        df = pd.DataFrame(
            {
                "season": [2023, 2024],
                "week": [18, 1],
                "gameday": ["2024-01-07", "2024-09-05"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_win": [1, 1],
            }
        )
        no_reg = compute_elo_features(df, preseason_regression=0.0)
        with_reg = compute_elo_features(df, preseason_regression=0.5)
        # KC wins in 2023 week 18, so KC's Elo goes up
        # With 50% regression, KC starts 2024 closer to 1500
        kc_2024_no = no_reg.loc[no_reg["season"] == 2024, "home_elo_pre"].iloc[0]
        kc_2024_reg = with_reg.loc[with_reg["season"] == 2024, "home_elo_pre"].iloc[0]
        # With regression, KC's 2024 starting Elo should be closer to 1500
        assert abs(kc_2024_reg - DEFAULT_ELO) < abs(kc_2024_no - DEFAULT_ELO)

    def test_default_elo_parameter(self):
        """Custom default_elo should be used for new teams."""
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_win": [1],
            }
        )
        result = compute_elo_features(df, default_elo=1600.0)
        assert result["home_elo_pre"].iloc[0] == 1600.0
        assert result["away_elo_pre"].iloc[0] == 1600.0


class TestMovMultiplier:
    def test_none_default(self):
        assert _mov_multiplier(28, 14, mov_type=MOV_NONE) == 1.0

    def test_none_by_default_param(self):
        assert _mov_multiplier(28, 14) == 1.0

    def test_log_formula(self):
        mult = _mov_multiplier(35, 14, mov_type=MOV_LOG, mov_scale=0.10)
        assert mult > 1.0
        expected = 1.0 + 0.10 * np.log(1.0 + 21.0)
        assert mult == pytest.approx(expected)

    def test_sqrt_formula(self):
        mult = _mov_multiplier(35, 14, mov_type=MOV_SQRT, mov_scale=0.10)
        assert mult > 1.0
        expected = 1.0 + 0.10 * np.sqrt(21.0)
        assert mult == pytest.approx(expected)

    def test_capped_log_applies_cap(self):
        mult = _mov_multiplier(70, 0, mov_type=MOV_CAPPED_LOG, mov_scale=0.20, mov_cap=2.0)
        # Without cap: 1 + 0.20 * ln(1+70) = 1 + 0.20*4.26 = 1.85
        # But 70 point diff with scale 0.20 -> 1 + 0.20*ln(71) = 1.85, so should be < 2.0
        # Use a case where cap would be needed
        assert mult <= 2.0

    def test_capped_linear_formula(self):
        mult = _mov_multiplier(35, 14, mov_type=MOV_CAPPED_LINEAR, mov_scale=0.05, mov_cap=3.0)
        expected = 1.0 + 0.05 * 21.0  # 2.05
        assert mult == pytest.approx(expected)
        assert mult <= 3.0

    def test_capped_linear_hit_cap(self):
        mult = _mov_multiplier(70, 0, mov_type=MOV_CAPPED_LINEAR, mov_scale=0.20, mov_cap=2.0)
        # 1 + 0.20 * 70 = 15.0, capped at 2.0
        assert mult == pytest.approx(2.0)

    def test_log_capped_by_cap(self):
        mult = _mov_multiplier(70, 0, mov_type=MOV_CAPPED_LOG, mov_scale=0.50, mov_cap=1.5)
        # 1 + 0.50 * ln(71) = 1 + 0.50*4.26 = 3.13, capped at 1.5
        assert mult == pytest.approx(1.5)

    def test_home_win_away_win_same_mult(self):
        """MOV multiplier should be symmetric regardless of which team won."""
        hw = _mov_multiplier(35, 14, mov_type=MOV_LOG, mov_scale=0.10)
        aw = _mov_multiplier(14, 35, mov_type=MOV_LOG, mov_scale=0.10)
        assert hw == aw

    def test_tie_returns_one(self):
        mult = _mov_multiplier(14, 14, mov_type=MOV_LOG, mov_scale=0.10)
        assert mult == 1.0

    def test_pregame_prob_unaffected_by_mov(self):
        """MOV parameters must not affect pregame probabilities (only update)."""
        df_before = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_win": [1, 1],
                "home_score": [27, 20],
                "away_score": [20, 17],
            }
        )
        df_no_mov = compute_elo_features(df_before, k_factor=20)
        df_mov = compute_elo_features(df_before, k_factor=20, mov_type=MOV_LOG, mov_scale=0.10)
        # Pregame prob for game 1 should be the same (both start at 1500)
        assert df_no_mov["elo_prob"].iloc[0] == df_mov["elo_prob"].iloc[0]
        # Game 1 pregame h2h_elo_pre should be same
        assert df_no_mov["home_elo_pre"].iloc[0] == df_mov["home_elo_pre"].iloc[0]
        # After game 1, MOV should affect post-update ratings for game 2
        assert df_no_mov["home_elo_pre"].iloc[1] != df_mov["home_elo_pre"].iloc[1]
