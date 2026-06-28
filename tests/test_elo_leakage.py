"""Step 2: Leakage tests for compute_elo_features().

Verifies that Elo outputs are emitted BEFORE the current game result
updates ratings, and that the Elo formula is correctly implemented.
"""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.ratings import compute_elo_features


def test_elo_output_order(sample_schedule):
    """Verify elo_prob uses PRE-game ratings, not post-update values.

    Game 1: ATL vs ARI, both start at 1500.
    Game 2: ARI vs ATL. After G1, ATL gained ~15.9 Elo and ARI lost ~15.9.
    So G2 elo_diff should be ~(1484 - 1516) = -32, not 0.
    """
    df = compute_elo_features(
        sample_schedule,
        k_factor=36,
        home_advantage=40,
        preseason_regression=0.0,
        mov_type="none",
        decay_half_life=None,
    )

    # Game 1: both at 1500
    np.testing.assert_almost_equal(df.loc[0, "home_elo_pre"], 1500.0, decimal=4)
    np.testing.assert_almost_equal(df.loc[0, "away_elo_pre"], 1500.0, decimal=4)
    np.testing.assert_almost_equal(df.loc[0, "elo_diff"], 0.0, decimal=4)

    # Game 2: values reflect Game 1 outcome
    # ATL won G1 at home 24-10 → ATL gained Elo, ARI lost Elo
    assert df.loc[1, "home_elo_pre"] < 1500, "ARI (home in G2) should have lost Elo after G1"
    assert df.loc[1, "away_elo_pre"] > 1500, "ATL (away in G2) should have gained Elo after G1"
    assert df.loc[1, "elo_diff"] < 0, "ARI after loss should be rated lower than ATL after win"

    # elo_prob should be based on pre-game ratings, not post-game
    expected_prob = 1.0 / (1.0 + 10.0 ** (
        -(df.loc[1, "home_elo_pre"] - df.loc[1, "away_elo_pre"] + 40) / 400.0
    ))
    np.testing.assert_almost_equal(df.loc[1, "elo_prob"], expected_prob, decimal=6)


def test_elo_formula_first_game(sample_schedule):
    """Verify the Elo formula for the first meeting of two 1500-rated teams.

    home=ATL (1500), away=ARI (1500), HFA=40.
    elo_prob = 1 / (1 + 10^(-(1500 - 1500 + 40) / 400))
             = 1 / (1 + 10^(-0.1))
             = 1 / 1.794328 = 0.5573
    """
    df = compute_elo_features(
        sample_schedule,
        k_factor=36,
        home_advantage=40,
        mov_type="none",
        decay_half_life=None,
    )

    expected = 1.0 / (1.0 + 10.0 ** (-40.0 / 400.0))
    np.testing.assert_almost_equal(df.loc[0, "elo_prob"], expected, decimal=6)
    assert 0.55 < df.loc[0, "elo_prob"] < 0.56


def test_elo_update_uses_current_game_result(sample_schedule):
    """Verify the second game's elo_pre differs from the first game's.

    This proves that Game 1 scores updated ratings before Game 2.
    If Game 2 scores leaked into Game 2's elo_pre, the effect would differ.
    """
    df = compute_elo_features(
        sample_schedule,
        k_factor=36,
        home_advantage=40,
        mov_type="none",
        decay_half_life=None,
    )

    # Game 2 ratings should reflect Game 1 (ATL won, so ATL > 1500)
    game2_home = df.loc[1, "home_elo_pre"]  # ARI (home in G2)
    game2_away = df.loc[1, "away_elo_pre"]  # ATL (away in G2)

    assert game2_home < 1500, "ARI lost G1, should be below 1500"
    assert game2_away > 1500, "ATL won G1, should be above 1500"


def test_elo_third_game_independent(sample_schedule):
    """Game 3 introduces CHI at 1500. Verify CHI starts at default Elo."""
    df = compute_elo_features(
        sample_schedule,
        k_factor=36,
        home_advantage=40,
        mov_type="none",
        decay_half_life=None,
    )

    # Game 3: CHI @ ATL. ATL has played 2 games, CHI none.
    np.testing.assert_almost_equal(df.loc[2, "away_elo_pre"], 1500.0, decimal=4, err_msg="CHI should start at 1500")
    assert df.loc[2, "home_elo_pre"] != 1500, "ATL should not be 1500 after 2 games"


def test_elo_with_mov(sample_schedule):
    """Verify MOV multiplier affects rating updates (visible in Game 2)."""
    df_none = compute_elo_features(
        sample_schedule,
        k_factor=36,
        home_advantage=40,
        mov_type="none",
        decay_half_life=None,
    )
    df_mov = compute_elo_features(
        sample_schedule,
        k_factor=36,
        home_advantage=40,
        mov_type="capped_linear",
        mov_scale=0.05,
        mov_cap=2.0,
        decay_half_life=None,
    )

    # MOV makes updates larger, so Game 2 elo_diff should differ
    assert df_none.loc[1, "elo_diff"] != df_mov.loc[1, "elo_diff"], (
        "MOV should change the Elo update magnitude"
    )


def test_elo_no_future_data_leakage():
    """Verify that compute_elo_features never uses future game results.

    Create schedule where G1 result is a blowout and G2 is close.
    G1 elo_pre should be 1500 for both (no prior data).
    """
    df = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "season": [2021, 2021],
        "week": [1, 2],
        "gameday": ["2021-09-12", "2021-09-19"],
        "home_team": ["ATL", "ARI"],
        "away_team": ["ARI", "ATL"],
        "home_score": [100, 3],
        "away_score": [0, 0],
        "home_win": [1, 1],
    })
    result = compute_elo_features(df, k_factor=36, home_advantage=40, mov_type="none", decay_half_life=None)
    assert result.loc[0, "home_elo_pre"] == 1500.0
    assert result.loc[0, "away_elo_pre"] == 1500.0


def test_preseason_regression_affects_first_game_of_new_season(sample_schedule_multi_season):
    """With reg=0.1, the 2022-01 game should have ratings pulled toward 1500."""
    df_reg = compute_elo_features(
        sample_schedule_multi_season,
        k_factor=36,
        home_advantage=40,
        preseason_regression=0.1,
        mov_type="none",
        decay_half_life=None,
    )
    df_no_reg = compute_elo_features(
        sample_schedule_multi_season,
        k_factor=36,
        home_advantage=40,
        preseason_regression=0.0,
        mov_type="none",
        decay_half_life=None,
    )

    # Game 4 (2022-01 index 3): with regression, ratings should be closer to 1500
    game4_idx = 3
    reg_diff = abs(df_reg.loc[game4_idx, "elo_diff"])
    no_reg_diff = abs(df_no_reg.loc[game4_idx, "elo_diff"])
    assert reg_diff < no_reg_diff, "Preseason regression should pull ratings toward 1500"


def test_decay_affects_subsequent_game(sample_schedule):
    """Verify decay half-life reduces rating deviation between games."""
    df_decay = compute_elo_features(
        sample_schedule,
        k_factor=36,
        home_advantage=40,
        mov_type="none",
        decay_half_life=2,  # very fast decay
    )
    df_no_decay = compute_elo_features(
        sample_schedule,
        k_factor=36,
        home_advantage=40,
        mov_type="none",
        decay_half_life=None,
    )

    # With fast decay, Game 2 ratings are closer to 1500
    decay_diff = abs(df_decay.loc[1, "elo_diff"])
    no_decay_diff = abs(df_no_decay.loc[1, "elo_diff"])
    assert decay_diff < no_decay_diff, "Decay should reduce rating deviation"
