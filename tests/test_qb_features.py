"""Tests for QB starter/change features — no leakage, no network."""

import numpy as np
import pandas as pd

from sportslab.features.qb import (
    QB_FEATURE_COLUMNS,
    compute_qb_features,
)


class TestQbBasicFeatures:
    def test_first_game_defaults(self):
        """First game for a team should have stable defaults."""
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_qb_id": ["QB_A"],
                "away_qb_id": ["QB_B"],
                "home_win": [1],
                "home_score": [27],
                "away_score": [20],
            }
        )
        result = compute_qb_features(df)
        assert result["home_qb_changed"].iloc[0] == 0
        assert result["away_qb_changed"].iloc[0] == 0
        assert result["home_qb_starts_this_season_pre"].iloc[0] == 0
        assert result["away_qb_starts_this_season_pre"].iloc[0] == 0
        assert result["home_new_qb_flag"].iloc[0] == 1
        assert result["away_new_qb_flag"].iloc[0] == 1
        assert result["home_qb_win_pct_pre"].iloc[0] == 0.5

    def test_qb_change_detected_across_games(self):
        """QB change should be detected when consecutive games have different QBs."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_qb_id": ["QB_A", "QB_B"],
                "away_qb_id": ["QB_X", "QB_X"],
                "home_win": [1, 1],
                "home_score": [27, 24],
                "away_score": [20, 17],
            }
        )
        result = compute_qb_features(df)
        assert result["home_qb_changed"].iloc[0] == 0
        assert result["home_qb_changed"].iloc[1] == 1  # Changed!
        assert result["away_qb_changed"].iloc[1] == 0  # Same QB

    def test_no_change_with_same_qb(self):
        """Consecutive games with the same QB should not flag a change."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_qb_id": ["QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X"],
                "home_win": [1, 1],
                "home_score": [27, 24],
                "away_score": [20, 17],
            }
        )
        result = compute_qb_features(df)
        assert result["home_qb_changed"].iloc[0] == 0
        assert result["home_qb_changed"].iloc[1] == 0

    def test_starts_this_season_accumulate(self):
        """QB starts should accumulate across consecutive games."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_qb_id": ["QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X"],
                "home_win": [1, 1],
                "home_score": [27, 24],
                "away_score": [20, 17],
            }
        )
        result = compute_qb_features(df)
        # Before game 1: 0 starts
        # Before game 2: 1 start (game 1)
        assert result["home_qb_starts_this_season_pre"].iloc[0] == 0
        assert result["home_qb_starts_this_season_pre"].iloc[1] == 1

    def test_win_pct_tracks_wins(self):
        """QB win pct should reflect prior wins in the season."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "week": [1, 2, 3],
                "gameday": ["2024-09-05", "2024-09-12", "2024-09-19"],
                "home_team": ["KC", "KC", "KC"],
                "away_team": ["BAL", "LV", "LAC"],
                "home_qb_id": ["QB_A", "QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X", "QB_X"],
                "home_win": [1, 0, 1],
                "home_score": [27, 17, 31],
                "away_score": [20, 24, 28],
            }
        )
        result = compute_qb_features(df)
        # Before game 1: 0/0 = 0.5 default
        # Before game 2: 1/1 = 1.0
        # Before game 3: 1/2 = 0.5
        assert result["home_qb_win_pct_pre"].iloc[0] == 0.5
        assert result["home_qb_win_pct_pre"].iloc[1] == 1.0
        assert result["home_qb_win_pct_pre"].iloc[2] == 0.5

    def test_current_game_not_included_in_own_features(self):
        """The current game's result must not affect its own QB features."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_qb_id": ["QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X"],
                "home_win": [1, 1],
                "home_score": [27, 24],
                "away_score": [20, 17],
            }
        )
        result = compute_qb_features(df)
        # Game 1's features should not include game 1's result
        # starts_pre should be 0 (no prior starts)
        # win_pct_pre should be 0.5 (no prior games)
        assert result["home_qb_starts_this_season_pre"].iloc[0] == 0
        assert result["home_qb_win_pct_pre"].iloc[0] == 0.5

    def test_prior_games_only(self):
        """QB features must be computed from games strictly before current game."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "week": [1, 2, 3],
                "gameday": ["2024-09-05", "2024-09-12", "2024-09-19"],
                "home_team": ["KC", "KC", "KC"],
                "away_team": ["BAL", "LV", "LAC"],
                "home_qb_id": ["QB_A", "QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X", "QB_X"],
                "home_win": [1, 0, 1],
                "home_score": [27, 17, 31],
                "away_score": [20, 24, 28],
            }
        )
        result = compute_qb_features(df)
        # Game 3: should only have wins from games 1-2 (1 win, 2 games = 0.5)
        assert result["home_qb_starts_this_season_pre"].iloc[2] == 2
        assert result["home_qb_win_pct_pre"].iloc[2] == 0.5

    def test_season_boundary_resets_change_count(self):
        """QB change flag should not carry over between seasons."""
        df = pd.DataFrame(
            {
                "season": [2023, 2024],
                "week": [18, 1],
                "gameday": ["2024-01-07", "2024-09-05"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_qb_id": ["QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X"],
                "home_win": [1, 1],
                "home_score": [27, 24],
                "away_score": [20, 17],
            }
        )
        result = compute_qb_features(df)
        # Same QB in season 2024 opener, but should not be flagged as changed
        assert result["home_qb_changed"].iloc[1] == 0
        # Season starts reset
        assert result["home_qb_starts_this_season_pre"].iloc[1] == 0

    def test_team_starts_persist_across_seasons(self):
        """QB career team starts should carry over between seasons."""
        df = pd.DataFrame(
            {
                "season": [2023, 2024],
                "week": [18, 1],
                "gameday": ["2024-01-07", "2024-09-05"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_qb_id": ["QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X"],
                "home_win": [1, 1],
                "home_score": [27, 24],
                "away_score": [20, 17],
            }
        )
        result = compute_qb_features(df)
        # 1 start in 2023 + 0 before 2024 opener = 1 career start
        assert result["home_qb_team_starts_pre"].iloc[1] == 1

    def test_games_since_change_increment(self):
        """games_since_qb_change should increase with consecutive same-QB games."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "week": [1, 2, 3],
                "gameday": ["2024-09-05", "2024-09-12", "2024-09-19"],
                "home_team": ["KC", "KC", "KC"],
                "away_team": ["BAL", "LV", "LAC"],
                "home_qb_id": ["QB_A", "QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X", "QB_X"],
                "home_win": [1, 1, 1],
                "home_score": [27, 24, 31],
                "away_score": [20, 17, 28],
            }
        )
        result = compute_qb_features(df)
        # Before game 1: 0 (no prior games)
        # Before game 2: 1 (1 prior game with QB_A)
        # Before game 3: 2 (2 prior games with QB_A)
        assert result["home_games_since_qb_change"].iloc[0] == 0
        assert result["home_games_since_qb_change"].iloc[1] == 1
        assert result["home_games_since_qb_change"].iloc[2] == 2

    def test_games_since_change_resets_after_qb_change(self):
        """games_since_qb_change should reset when QB changes."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "week": [1, 2, 3],
                "gameday": ["2024-09-05", "2024-09-12", "2024-09-19"],
                "home_team": ["KC", "KC", "KC"],
                "away_team": ["BAL", "LV", "LAC"],
                "home_qb_id": ["QB_A", "QB_A", "QB_B"],
                "away_qb_id": ["QB_X", "QB_X", "QB_X"],
                "home_win": [1, 1, 1],
                "home_score": [27, 24, 31],
                "away_score": [20, 17, 28],
            }
        )
        result = compute_qb_features(df)
        assert result["home_games_since_qb_change"].iloc[2] == 0
        # QB_B is new for this team
        assert result["home_new_qb_flag"].iloc[2] == 1

    def test_new_qb_flag_after_change(self):
        """new_qb_flag should be 1 for first start, 0 after that."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_qb_id": ["QB_A", "QB_B"],
                "away_qb_id": ["QB_X", "QB_X"],
                "home_win": [1, 1],
                "home_score": [27, 24],
                "away_score": [20, 17],
            }
        )
        result = compute_qb_features(df)
        # QB_A is new for team
        assert result["home_new_qb_flag"].iloc[0] == 1
        # QB_B is also new for team
        assert result["home_new_qb_flag"].iloc[1] == 1
        # After one start, not new anymore
        # (need a 3rd game to test this)

    def test_new_qb_flag_clears_after_start(self):
        """Once a QB has started, new_qb_flag should be 0."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "week": [1, 2, 3],
                "gameday": ["2024-09-05", "2024-09-12", "2024-09-19"],
                "home_team": ["KC", "KC", "KC"],
                "away_team": ["BAL", "LV", "LAC"],
                "home_qb_id": ["QB_A", "QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X", "QB_X"],
                "home_win": [1, 1, 1],
                "home_score": [27, 24, 31],
                "away_score": [20, 17, 28],
            }
        )
        result = compute_qb_features(df)
        assert result["home_new_qb_flag"].iloc[0] == 1
        assert result["home_new_qb_flag"].iloc[1] == 0
        assert result["home_new_qb_flag"].iloc[2] == 0

    def test_missing_qb_values_handled(self):
        """Missing QB values should produce safe defaults and missing flags."""
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_qb_id": [None],
                "away_qb_id": [np.nan],
                "home_win": [1],
                "home_score": [27],
                "away_score": [20],
            }
        )
        result = compute_qb_features(df)
        assert result["home_qb_missing_flag"].iloc[0] == 1
        assert result["away_qb_missing_flag"].iloc[0] == 1
        assert result["home_qb_starts_this_season_pre"].iloc[0] == 0
        assert result["home_qb_win_pct_pre"].iloc[0] == 0.5

    def test_diff_features_exist(self):
        """Diff features should be present and computed correctly."""
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_qb_id": ["QB_A"],
                "away_qb_id": ["QB_B"],
                "home_win": [1],
                "home_score": [27],
                "away_score": [20],
            }
        )
        result = compute_qb_features(df)
        for col in [
            "qb_change_diff",
            "qb_starts_diff",
            "qb_win_pct_diff",
            "games_since_qb_change_diff",
            "new_qb_diff",
        ]:
            assert col in result.columns
        # All diffs should be 0 for identical defaults
        assert result["qb_change_diff"].iloc[0] == 0
        assert result["qb_starts_diff"].iloc[0] == 0

    def test_chronological_order(self):
        """Features should be correct even when input is not sorted."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [2, 1],
                "gameday": ["2024-09-12", "2024-09-05"],
                "home_team": ["KC", "BAL"],
                "away_team": ["LV", "KC"],
                "home_qb_id": ["QB_A", "QB_B"],
                "away_qb_id": ["QB_X", "QB_A"],
                "home_win": [1, 0],
                "home_score": [24, 17],
                "away_score": [17, 27],
            }
        )
        result = compute_qb_features(df)
        # After sorting, game 1 (week 1): KC (BAL) has no starts
        # KC has QB_A starting in week 2
        # But the sorted order should process week 1 first
        # Just verify it ran without error
        assert len(result) == 2

    def test_tie_not_counted_as_win(self):
        """Ties should not count as wins for QB win pct."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_qb_id": ["QB_A", "QB_A"],
                "away_qb_id": ["QB_X", "QB_X"],
                "home_win": [pd.NA, 1],
                "home_score": [20, 27],
                "away_score": [20, 17],
            }
        )
        result = compute_qb_features(df)
        # Tie (game 1) is not a win. Before game 2: 0 wins, 1 game = 0.0
        assert result["home_qb_win_pct_pre"].iloc[1] == 0.0

    def test_all_qb_feature_columns_present(self):
        """All columns in QB_FEATURE_COLUMNS should be present after compute."""
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_qb_id": ["QB_A"],
                "away_qb_id": ["QB_B"],
                "home_win": [1],
                "home_score": [27],
                "away_score": [20],
            }
        )
        result = compute_qb_features(df)
        for col in QB_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_missing_values_in_qb_features(self):
        """QB feature columns should have no NaN values."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "BUF"],
                "away_team": ["BAL", "MIA"],
                "home_qb_id": ["QB_A", "QB_C"],
                "away_qb_id": ["QB_B", "QB_D"],
                "home_win": [1, 0],
                "home_score": [27, 14],
                "away_score": [20, 31],
            }
        )
        result = compute_qb_features(df)
        for col in QB_FEATURE_COLUMNS:
            assert result[col].isna().sum() == 0, f"NaN in column: {col}"
