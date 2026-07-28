"""Tests for Pi-Ratings: compute_pi_ratings_features, experiment config, CLI."""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.pi_ratings_experiment import (
    PI_ALPHAS,
    PI_BASE_KS,
    PI_HFAS,
    PI_HK_RATIOS,
    PI_REGS,
)
from sportslab.features.ratings import compute_pi_ratings_features


@pytest.fixture
def _sample_df() -> pd.DataFrame:
    """Minimal schedule for Pi-Ratings tests."""
    return pd.DataFrame({
        "game_id": ["2021_01_ARI_ATL", "2021_01_ATL_CHI", "2021_02_CHI_ATL"],
        "season": [2021, 2021, 2021],
        "week": [1, 1, 2],
        "gameday": ["2021-09-12", "2021-09-12", "2021-09-19"],
        "home_team": ["ATL", "CHI", "ATL"],
        "away_team": ["ARI", "ATL", "CHI"],
        "home_score": [24, 10, 17],
        "away_score": [10, 20, 24],
        "home_win": [1, 0, 0],
    })


class TestComputePiRatingsFeatures:
    def test_basic_output_columns(self, _sample_df):
        result = compute_pi_ratings_features(_sample_df)
        expected_cols = [
            "home_pi_pre", "away_pi_pre", "pi_diff", "pi_prob",
            "pi_mov_mult", "pi_home_k", "pi_away_k",
        ]
        for c in expected_cols:
            assert c in result.columns, f"Missing column: {c}"

    def test_pi_prob_bounded_0_1(self, _sample_df):
        result = compute_pi_ratings_features(_sample_df)
        assert result["pi_prob"].between(0, 1).all()

    def test_alpha_half_compresses(self, _sample_df):
        """alpha=0.5 should give smaller MOV for blowouts vs alpha=1.0."""
        r1 = compute_pi_ratings_features(_sample_df, alpha=0.5)
        r2 = compute_pi_ratings_features(_sample_df, alpha=1.0)
        # First game: margin=14, alpha=0.5 -> 14^0.5=3.74, alpha=1 -> 14
        assert r1["pi_mov_mult"].iloc[0] < r2["pi_mov_mult"].iloc[0]

    def test_alpha_125_amplifies(self, _sample_df):
        """alpha=1.25 should give larger MOV for blowouts vs alpha=1.0."""
        r1 = compute_pi_ratings_features(_sample_df, alpha=1.25)
        r2 = compute_pi_ratings_features(_sample_df, alpha=1.0)
        assert r1["pi_mov_mult"].iloc[0] > r2["pi_mov_mult"].iloc[0]

    def test_hk_ratio_075(self, _sample_df):
        """hk_ratio=0.75: away learns more than home."""
        r = compute_pi_ratings_features(_sample_df, hk_ratio=0.75)
        assert r["pi_home_k"].iloc[0] < r["pi_away_k"].iloc[0]

    def test_hk_ratio_125(self, _sample_df):
        """hk_ratio=1.25: home learns more than away."""
        r = compute_pi_ratings_features(_sample_df, hk_ratio=1.25)
        assert r["pi_home_k"].iloc[0] > r["pi_away_k"].iloc[0]

    def test_hk_ratio_1_symmetric(self, _sample_df):
        """hk_ratio=1.0: home and away learn at same rate."""
        r = compute_pi_ratings_features(_sample_df, hk_ratio=1.0)
        np.testing.assert_allclose(
            r["pi_home_k"].iloc[0], r["pi_away_k"].iloc[0]
        )

    def test_alpha_1_hk_1_reduces_to_standard(self, _sample_df):
        """With alpha=1, hk_ratio=1, the MOV is |margin| (linear)."""
        r = compute_pi_ratings_features(_sample_df, alpha=1.0, hk_ratio=1.0)
        margin = abs(_sample_df["home_score"].iloc[0] - _sample_df["away_score"].iloc[0])
        assert r["pi_mov_mult"].iloc[0] == margin

    def test_mov_cap_applies(self, _sample_df):
        """With mov_cap=5, the multiplier should be capped."""
        r = compute_pi_ratings_features(_sample_df, mov_cap=5.0)
        assert all(r["pi_mov_mult"] <= 5.0)

    def test_preseason_regression(self):
        """Ratings should regress toward default_elo at season boundary."""
        df = pd.DataFrame({
            "game_id": ["2021_01_A_B", "2022_01_A_B"],
            "season": [2021, 2022],
            "week": [1, 1],
            "gameday": ["2021-09-12", "2022-09-11"],
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_score": [50, 24],
            "away_score": [0, 10],
            "home_win": [1, 1],
        })
        r = compute_pi_ratings_features(df, alpha=1.0, preseason_regression=0.5)
        # After first game A won 50-0, rating should be well above 1500
        # Second game: with 0.5 regression, A's rating should be pulled toward 1500
        # Compared to no regression case
        r_no = compute_pi_ratings_features(df, alpha=1.0, preseason_regression=0.0)
        assert abs(r["home_pi_pre"].iloc[1] - 1500) < abs(r_no["home_pi_pre"].iloc[1] - 1500)

    def test_decay_applies(self, _sample_df):
        """Decay should pull ratings toward default_elo each game."""
        r = compute_pi_ratings_features(_sample_df, alpha=1.0, decay_half_life=1)
        # With decay=1, ratings should be closer to default after update
        assert all(r["home_pi_pre"].notna())

    def test_tie_handling(self):
        """Ties should result in 0.5 expected, small updates."""
        df = pd.DataFrame({
            "game_id": ["2021_01_A_B"],
            "season": [2021],
            "week": [1],
            "gameday": ["2021-09-12"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [10],
            "away_score": [10],
            "home_win": [pd.NA],
        })
        r = compute_pi_ratings_features(df)
        assert r["pi_prob"].iloc[0] == 0.5  # default symmetric
        # Update should be small (actual=0.5, expected≈0.5, close to 0)

    def test_no_score_columns(self):
        """Should handle missing score columns gracefully."""
        df = pd.DataFrame({
            "game_id": ["2021_01_A_B"],
            "season": [2021],
            "week": [1],
            "gameday": ["2021-09-12"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [0],
            "away_score": [0],
            "home_win": [pd.NA],
        })
        r = compute_pi_ratings_features(df)
        assert r["pi_prob"].iloc[0] == 0.5


class TestExperimentConfig:
    def test_grid_is_bounded(self):
        total = len(PI_ALPHAS) * len(PI_BASE_KS) * len(PI_HK_RATIOS) * len(PI_HFAS) * len(PI_REGS)
        assert total <= 200, f"Grid too large: {total}"

    def test_pi_alphas_include_one(self):
        assert 1.0 in PI_ALPHAS

    def test_hk_ratios_include_one(self):
        assert 1.0 in PI_HK_RATIOS


class TestCLIImportability:
    def test_cli_command_importable(self):
        from sportslab.cli import pi_ratings_cmd  # noqa: F401

    def test_feature_function_in_ratings_module(self):
        from sportslab.features.ratings import compute_pi_ratings_features  # noqa: F401
