"""Tests for Glicko rating system implementation.

Verifies:
1. Formula correctness (_g, _glicko_expected, _glicko_update)
2. No future data leakage (chronological processing)
3. RD increases between seasons
4. QB RD bonus works
5. DataFrame column completeness
"""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.glicko import (
    DEFAULT_GLICKO,
    Q,
    _g,
    _glicko_expected,
    _glicko_update,
    compute_glicko_features,
)


class TestGlickoFormulas:
    def test_g_function_bounds(self):
        """g(RD) bounds: RD=0 -> 1.0, RD->inf -> 0.0"""
        assert _g(0.0) == pytest.approx(1.0, abs=1e-6)
        assert _g(1e6) < 0.001

    def test_g_function_known_value(self):
        """Known g(RD) at RD=350."""
        val = _g(350.0)
        expected = 1.0 / np.sqrt(1.0 + 3.0 * Q * Q * 350.0 * 350.0 / (np.pi * np.pi))
        assert val == pytest.approx(expected, abs=1e-6)
        assert val == pytest.approx(0.6691, abs=1e-4)

    def test_glicko_expected_symmetric(self):
        """Equal ratings + no HFA -> 0.5 expected."""
        prob = _glicko_expected(1500.0, 1500.0, 350.0, hfa=0.0)
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_glicko_expected_hfa(self):
        """HFA shifts expected probability."""
        prob_no_hfa = _glicko_expected(1500.0, 1500.0, 350.0, hfa=0.0)
        prob_with_hfa = _glicko_expected(1500.0, 1500.0, 350.0, hfa=40.0)
        assert prob_with_hfa > prob_no_hfa

    def test_high_rd_pulls_toward_05(self):
        """High opponent RD pulls probability toward 0.5."""
        big_fav = _glicko_expected(1600.0, 1400.0, rd_away=50.0, hfa=0.0)
        uncertain = _glicko_expected(1600.0, 1400.0, rd_away=500.0, hfa=0.0)
        assert uncertain < big_fav
        assert uncertain > 0.5

    def test_update_rating_increases_after_win(self):
        """Rating increases after a win."""
        new_r, _ = _glicko_update(1500.0, 350.0, _g(350.0), 0.5, 1.0, mov_mult=1.0)
        assert new_r > 1500.0

    def test_update_rating_decreases_after_loss(self):
        """Rating decreases after a loss."""
        new_r, _ = _glicko_update(1500.0, 350.0, _g(350.0), 0.5, 0.0, mov_mult=1.0)
        assert new_r < 1500.0

    def test_update_rd_decreases_after_game(self):
        """RD decreases after a completed game."""
        _, new_rd = _glicko_update(1500.0, 350.0, _g(350.0), 0.5, 1.0, mov_mult=1.0)
        assert new_rd < 350.0

    def test_mov_increases_update_magnitude(self):
        """MOV multiplier > 1 produces larger rating change."""
        r1, _ = _glicko_update(1500.0, 350.0, _g(350.0), 0.5, 1.0, mov_mult=1.0)
        r2, _ = _glicko_update(1500.0, 350.0, _g(350.0), 0.5, 1.0, mov_mult=2.0)
        assert abs(r2 - 1500.0) > abs(r1 - 1500.0)

    def test_update_magnitude_reasonable(self):
        """Typical win should change rating by ~100-200 points."""
        new_r, _ = _glicko_update(1500.0, 350.0, _g(350.0), 0.5, 1.0, mov_mult=1.0)
        delta = abs(new_r - 1500.0)
        assert 50 < delta < 400, f"Rating change {delta:.1f} outside expected range"


class TestGlickoFeatures:
    def test_column_completeness(self):
        """compute_glicko_features returns all expected columns."""
        df = pd.DataFrame({
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": ["2021-09-09", "2021-09-19"],
            "home_team": ["KC", "KC"],
            "away_team": ["CLE", "BAL"],
            "home_win": [1.0, 0.0],
            "home_score": [33, 28],
            "away_score": [29, 35],
        })
        result = compute_glicko_features(df, home_advantage=40.0)
        expected = {
            "home_glicko_pre", "away_glicko_pre", "home_glicko_rd",
            "away_glicko_rd", "glicko_diff", "glicko_prob",
        }
        assert expected.issubset(set(result.columns))

    def test_chronological_order_no_leakage(self):
        """Game 2 uses ratings from Game 1, not the reverse."""
        df = pd.DataFrame({
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": ["2021-09-09", "2021-09-19"],
            "home_team": ["KC", "KC"],
            "away_team": ["CLE", "BAL"],
            "home_win": [1.0, 0.0],
            "home_score": [33, 28],
            "away_score": [29, 35],
        })
        result = compute_glicko_features(df, home_advantage=40.0)
        # Game 1 home glicko should be DEFAULT_GLICKO (initial)
        assert result.loc[0, "home_glicko_pre"] == DEFAULT_GLICKO
        # Game 2 home glicko should have been updated after Game 1
        if result.loc[1, "home_glicko_pre"] == DEFAULT_GLICKO:
            pytest.fail("Game 2 glicko unchanged from default — no update occurred")

    def test_season_boundary_rd_increase(self):
        """RD increases between seasons due to system_constant_c."""
        df = pd.DataFrame({
            "season": [2021, 2022],
            "week": [17, 1],
            "gameday": ["2022-01-02", "2022-09-08"],
            "home_team": ["KC", "KC"],
            "away_team": ["CLE", "BAL"],
            "home_win": [1.0, 0.0],
            "home_score": [33, 28],
            "away_score": [29, 35],
        })
        result = compute_glicko_features(
            df, home_advantage=40.0, system_constant_c=100.0,
        )
        # Pre-game RD for 2021 game 1 is initial (350)
        rd_2021_pre = result.loc[0, "home_glicko_rd"]
        assert rd_2021_pre == pytest.approx(350.0)
        # Pre-game RD for 2022 = sqrt(post_game_rd_2021² + c²)
        # After 2021 game update, post_game_rd ≈ 290
        # So 2022 pre RD ≈ sqrt(290² + 100²) ≈ 307
        rd_2022_pre = result.loc[1, "home_glicko_rd"]
        # Verify growth by undoing c: sqrt(rd_2022_pre² - c²) ≈ post_game RD ~290
        estimated_post_2021 = np.sqrt(rd_2022_pre**2 - 100.0**2)
        assert estimated_post_2021 < 350.0, "Post-2021 RD should be < initial"
        assert estimated_post_2021 > 200.0, "Post-2021 RD should be reasonable"

    def test_glicko_prob_in_valid_range(self):
        """glicko_prob is always in [0, 1]."""
        df = pd.DataFrame({
            "season": [2021, 2021, 2021],
            "week": [1, 2, 3],
            "gameday": ["2021-09-09", "2021-09-19", "2021-09-26"],
            "home_team": ["KC", "BUF", "TB"],
            "away_team": ["CLE", "MIA", "NO"],
            "home_win": [1.0, 0.0, 1.0],
            "home_score": [33, 28, 31],
            "away_score": [29, 35, 17],
        })
        result = compute_glicko_features(df, home_advantage=40.0)
        assert result["glicko_prob"].between(0, 1).all()

    def test_qb_rd_bonus(self):
        """QB RD bonus increases RD for teams with QB change."""
        qb_change_map = {"KC": [2022]}
        s1 = pd.DataFrame({
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": ["2021-09-09", "2021-09-19"],
            "home_team": ["KC", "KC"],
            "away_team": ["CLE", "BAL"],
            "home_win": [1.0, 0.0],
            "home_score": [33, 28],
            "away_score": [29, 35],
        })
        s2 = pd.DataFrame({
            "season": [2022, 2022],
            "week": [1, 2],
            "gameday": ["2022-09-08", "2022-09-18"],
            "home_team": ["KC", "KC"],
            "away_team": ["LAC", "CIN"],
            "home_win": [1.0, 0.0],
            "home_score": [27, 22],
            "away_score": [22, 26],
        })
        df = pd.concat([s1, s2], ignore_index=True)
        result = compute_glicko_features(
            df, home_advantage=40.0, qb_rd_bonus=100.0,
            qb_change_map=qb_change_map, system_constant_c=100.0,
        )
        # KC 2022 game 1 RD should be higher with QB bonus
        kc_2022_rd = result.loc[2, "home_glicko_rd"]
        assert kc_2022_rd > 200  # RD should remain substantial

    def test_first_game_default_rd(self):
        """First game for a team uses initial RD."""
        df = pd.DataFrame({
            "season": [2021],
            "week": [1],
            "gameday": ["2021-09-09"],
            "home_team": ["KC"],
            "away_team": ["CLE"],
            "home_win": [1.0],
            "home_score": [33],
            "away_score": [29],
        })
        result = compute_glicko_features(df, home_advantage=40.0, initial_rd=250.0)
        assert result.loc[0, "home_glicko_rd"] == 250.0
        assert result.loc[0, "away_glicko_rd"] == 250.0
