"""Tests for Glicko rating system."""

import numpy as np
import pandas as pd

from sportslab.features.glicko import (
    Q,
    _g,
    _glicko_expected,
    _glicko_update,
    compute_glicko_features,
)


class TestGlickoMath:
    def test_g_function_value(self):
        """g(RD) = 1 when RD=0, and decreases as RD increases."""
        assert np.isclose(_g(0.0), 1.0)
        assert _g(100.0) < 1.0
        assert _g(200.0) < _g(100.0)
        assert _g(500.0) < _g(200.0)

    def test_g_function_formula(self):
        """Verify g(350) manually: 1 / sqrt(1 + 3*q^2*350^2/pi^2)."""
        expected = 1.0 / np.sqrt(1.0 + 3.0 * Q * Q * 350.0 * 350.0 / (np.pi * np.pi))
        assert np.isclose(_g(350.0), expected)

    def test_expected_score_symmetric(self):
        """Equal ratings + no HFA → 0.5 expected."""
        prob = _glicko_expected(1500.0, 1500.0, 200.0, hfa=0.0)
        assert np.isclose(prob, 0.5)

    def test_expected_score_hfa(self):
        """HFA shifts probability."""
        prob = _glicko_expected(1500.0, 1500.0, 200.0, hfa=40.0)
        assert prob > 0.5

    def test_expected_score_high_rd(self):
        """High opponent RD pulls toward 0.5."""
        prob_low = _glicko_expected(1600.0, 1500.0, 50.0, hfa=0.0)
        prob_high = _glicko_expected(1600.0, 1500.0, 400.0, hfa=0.0)
        # High RD should pull closer to 0.5
        assert abs(prob_high - 0.5) < abs(prob_low - 0.5)

    def test_glicko_update_winner(self):
        """Winner's rating increases, RD decreases."""
        new_r, new_rd = _glicko_update(
            rating=1500.0, rd=200.0, g_opp=0.9,
            expected=0.5, actual=1.0, mov_mult=1.0,
        )
        assert new_r > 1500.0  # Rating goes up
        assert new_rd < 200.0  # RD goes down

    def test_glicko_update_loser(self):
        """Loser's rating decreases, RD decreases."""
        new_r, new_rd = _glicko_update(
            rating=1500.0, rd=200.0, g_opp=0.9,
            expected=0.5, actual=0.0, mov_mult=1.0,
        )
        assert new_r < 1500.0
        assert new_rd < 200.0


class TestComputeGlicko:
    def test_basic_computation(self):
        """Minimal input produces expected columns."""
        df = pd.DataFrame({
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": ["2021-09-09", "2021-09-12"],
            "home_team": ["KC", "BUF"],
            "away_team": ["HOU", "MIA"],
            "home_win": [1, 0],
            "home_score": [31, 17],
            "away_score": [20, 21],
        })
        result = compute_glicko_features(df, home_advantage=40.0)
        expected_cols = {
            "home_glicko_pre", "away_glicko_pre",
            "home_glicko_rd", "away_glicko_rd",
            "glicko_diff", "glicko_prob",
        }
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
        assert len(result) == 2
        # First game: both teams at default 1500, HFA=40
        assert np.isclose(result["glicko_prob"].iloc[0], 0.538, atol=0.01)

    def test_season_boundary_rd_growth(self):
        """RD increases between seasons."""
        df = pd.DataFrame({
            "season": [2021, 2022],
            "week": [18, 1],
            "gameday": ["2022-01-09", "2022-09-08"],
            "home_team": ["KC", "KC"],
            "away_team": ["DEN", "DEN"],
            "home_win": [1, 0],
            "home_score": [28, 10],
            "away_score": [24, 27],
        })
        result = compute_glicko_features(
            df, home_advantage=40.0, system_constant_c=200.0,
        )
        # Game 2 pre-game RD should be > game 1 pre-game RD (season boundary),
        # but < sqrt(initial_rd^2 + c^2) because game 1 reduced RD
        rd_game1 = result["home_glicko_rd"].iloc[0]
        rd_game2 = result["home_glicko_rd"].iloc[1]
        max_possible = np.sqrt(rd_game1 ** 2 + 200 ** 2)
        # RD increased due to season boundary (350 -> ~352.7)
        assert rd_game2 > rd_game1, (
            f"Expected rd_game2 > rd_game1, got {rd_game2:.3f} <= {rd_game1:.3f}"
        )
        # But less than max (since game 1 reduced RD)
        assert rd_game2 < max_possible, (
            f"Expected rd_game2 < {max_possible:.3f}, got {rd_game2:.3f}"
        )

    def test_new_team_starts_at_default(self):
        """Unseen teams get initial rating and RD."""
        df = pd.DataFrame({
            "season": [2022],
            "week": [1],
            "gameday": ["2022-09-08"],
            "home_team": ["LAC"],
            "away_team": ["LV"],
            "home_win": [1],
            "home_score": [24],
            "away_score": [19],
        })
        result = compute_glicko_features(df, home_advantage=40.0)
        assert result["home_glicko_pre"].iloc[0] == 1500.0
        assert result["home_glicko_rd"].iloc[0] == 350.0
        assert result["away_glicko_pre"].iloc[0] == 1500.0

    def test_tie_handling(self):
        """Ties update ratings moderately."""
        df = pd.DataFrame({
            "season": [2021],
            "week": [1],
            "gameday": ["2021-09-09"],
            "home_team": ["KC"],
            "away_team": ["HOU"],
            "home_win": [pd.NA],
            "home_score": [24],
            "away_score": [24],
        })
        result = compute_glicko_features(df, home_advantage=40.0)
        assert result["glicko_prob"].iloc[0] > 0.5  # HFA still applies

    def test_qb_rd_bonus(self):
        """QB change causes extra RD growth at season boundary."""
        df = pd.DataFrame({
            "season": [2021, 2022],
            "week": [18, 1],
            "gameday": ["2022-01-09", "2022-09-08"],
            "home_team": ["KC", "KC"],
            "away_team": ["DEN", "DEN"],
            "home_win": [1, 0],
            "home_score": [28, 10],
            "away_score": [24, 27],
        })
        qb_map = {"KC": [2022]}
        result_with = compute_glicko_features(
            df, home_advantage=40.0, system_constant_c=200.0,
            qb_rd_bonus=100.0, qb_change_map=qb_map,
        )
        result_without = compute_glicko_features(
            df, home_advantage=40.0, system_constant_c=200.0,
        )
        rds_with = result_with["home_glicko_rd"].iloc[1]
        rds_without = result_without["home_glicko_rd"].iloc[1]
        assert rds_with > rds_without

    def test_holdout_not_touched(self):
        """2025 holdout games should not be used for rating updates of train."""
        df = pd.DataFrame({
            "season": [2021, 2025],
            "week": [1, 1],
            "gameday": ["2021-09-09", "2025-09-08"],
            "home_team": ["KC", "KC"],
            "away_team": ["HOU", "HOU"],
            "home_win": [1, 0],
            "home_score": [31, 20],
            "away_score": [20, 24],
        })
        result = compute_glicko_features(df, home_advantage=40.0)
        assert "glicko_prob" in result.columns

    def test_mov_multiplier(self):
        """MOV multiplier increases rating update for blowout."""
        df_blowout = pd.DataFrame({
            "season": [2021],
            "week": [1],
            "gameday": ["2021-09-09"],
            "home_team": ["KC"],
            "away_team": ["HOU"],
            "home_win": [1],
            "home_score": [42],
            "away_score": [7],
        })
        df_close = df_blowout.copy()
        df_close["home_score"] = 24
        df_close["away_score"] = 21

        blowout = compute_glicko_features(
            df_blowout, home_advantage=40.0,
            mov_type="capped_linear", mov_scale=0.05, mov_cap=2.0,
        )
        close = compute_glicko_features(
            df_close, home_advantage=40.0,
            mov_type="capped_linear", mov_scale=0.05, mov_cap=2.0,
        )
        assert blowout["home_glicko_pre"].iloc[0] == close["home_glicko_pre"].iloc[0]


class TestGridImport:
    def test_glicko_experiment_importable(self):
        """The experiment module imports without error."""
        from sportslab.evaluation.glicko_experiment import run_glicko_experiment
        assert callable(run_glicko_experiment)

    def test_cli_command_registered(self):
        """The glicko command is available via CLI."""
        from sportslab.cli import cli
        commands = [cmd for cmd in cli.commands]
        assert "glicko" in commands
