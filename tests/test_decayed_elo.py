"""Tests for decayed Elo — exponential momentum features."""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.decayed_elo_experiment import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    run_decayed_elo_experiment,
)
from sportslab.features.ratings import (
    DEFAULT_ELO,
    MOV_CAPPED_LINEAR,
    compute_elo_features,
)


def _mini_df() -> pd.DataFrame:
    """Create a minimal 2-season, 2-team DataFrame for testing."""
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
                    "home_win": 1 if (season + week) % 2 == 0 else 0,
                    "home_score": 28,
                    "away_score": 21,
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
    df = pd.DataFrame(rows)
    df["div_game"] = 0
    return df


class TestDecayFunction:
    def test_no_decay_produces_same_as_standard(self):
        df = _mini_df()
        std = compute_elo_features(
            df, k_factor=20, mov_type=MOV_CAPPED_LINEAR, mov_scale=0.05, mov_cap=2.0
        )
        dcd = compute_elo_features(
            df,
            k_factor=20,
            mov_type=MOV_CAPPED_LINEAR,
            mov_scale=0.05,
            mov_cap=2.0,
            decay_half_life=None,
        )
        pd.testing.assert_frame_equal(std, dcd)

    def test_decay_pulls_ratings_toward_mean(self):
        df = _mini_df()
        no_decay = compute_elo_features(
            df, k_factor=20, mov_type=MOV_CAPPED_LINEAR, mov_scale=0.05, mov_cap=2.0
        )
        decay = compute_elo_features(
            df,
            k_factor=20,
            mov_type=MOV_CAPPED_LINEAR,
            mov_scale=0.05,
            mov_cap=2.0,
            decay_half_life=2.0,
        )
        # After several games, decayed ratings should be closer to 1500
        no_decay_dev = abs(no_decay["home_elo_pre"].iloc[-1] - DEFAULT_ELO)
        decay_dev = abs(decay["home_elo_pre"].iloc[-1] - DEFAULT_ELO)
        assert decay_dev < no_decay_dev

    def test_decay_faster_with_shorter_half_life(self):
        df = _mini_df()
        slow = compute_elo_features(
            df,
            k_factor=20,
            mov_type=MOV_CAPPED_LINEAR,
            mov_scale=0.05,
            mov_cap=2.0,
            decay_half_life=1000,
        )
        fast = compute_elo_features(
            df,
            k_factor=20,
            mov_type=MOV_CAPPED_LINEAR,
            mov_scale=0.05,
            mov_cap=2.0,
            decay_half_life=1.0,
        )
        slow_dev = abs(slow["home_elo_pre"].iloc[-1] - DEFAULT_ELO)
        fast_dev = abs(fast["home_elo_pre"].iloc[-1] - DEFAULT_ELO)
        assert fast_dev < slow_dev

    def test_decay_zero_half_life_does_not_crash(self):
        df = _mini_df()
        result = compute_elo_features(df, k_factor=20, decay_half_life=0.001)
        assert "elo_prob" in result.columns

    def test_decay_very_long_half_life_approaches_no_decay(self):
        df = _mini_df()
        no_decay = compute_elo_features(
            df, k_factor=20, mov_type=MOV_CAPPED_LINEAR, mov_scale=0.05, mov_cap=2.0
        )
        long_decay = compute_elo_features(
            df,
            k_factor=20,
            mov_type=MOV_CAPPED_LINEAR,
            mov_scale=0.05,
            mov_cap=2.0,
            decay_half_life=1e6,
        )
        np.testing.assert_allclose(
            no_decay["home_elo_pre"].values,
            long_decay["home_elo_pre"].values,
            atol=0.1,
        )

    def test_pregame_prob_unchanged_by_future_decay(self):
        df = _mini_df()
        result = compute_elo_features(df, k_factor=20, decay_half_life=4.0)
        assert result["elo_prob"].iloc[0] == pytest.approx(0.5, abs=0.01)

    def test_decay_does_not_affect_column_names(self):
        df = _mini_df()
        result = compute_elo_features(df, k_factor=20, decay_half_life=8.0)
        for col in ["home_elo_pre", "away_elo_pre", "elo_diff", "elo_prob"]:
            assert col in result.columns


class TestExperiment:
    def test_importable(self):
        assert callable(run_decayed_elo_experiment)

    def test_folds_exclude_holdout(self):
        for train_s, val_s in ROLLING_FOLDS:
            all_seasons = list(train_s) + [val_s]
            assert HOLDOUT_SEASON not in all_seasons

    def test_folds_sequential(self):
        expected = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
        assert ROLLING_FOLDS == expected
