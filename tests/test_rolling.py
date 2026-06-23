"""Tests for the rolling pregame feature module — no leakage, no network."""

import pandas as pd
import pytest

from sportslab.features.rolling import compute_rolling_features


class TestComputeRollingFeatures:
    def test_first_game_default_values(self):
        """First game for a team should get default imputed values."""
        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "gameday": ["2024-09-05"],
                "home_team": ["KC"],
                "away_team": ["BAL"],
                "home_score": [27],
                "away_score": [20],
                "home_win": [1],
            }
        )
        result = compute_rolling_features(df)
        # No prior games → 0.5 win pct (imputed), 0.0 point diff
        assert result["home_rolling_win_pct"].iloc[0] == 0.5
        assert result["away_rolling_win_pct"].iloc[0] == 0.5
        assert result["home_rolling_point_diff"].iloc[0] == 0.0
        assert result["away_rolling_point_diff"].iloc[0] == 0.0

    def test_current_game_excluded(self):
        """Result of the current game must not be in its own rolling features."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_score": [27, 30],
                "away_score": [20, 24],
                "home_win": [1, 1],
            }
        )
        result = compute_rolling_features(df)

        # Game 2 features should only include game 1
        assert result["home_rolling_win_pct"].iloc[1] == 1.0  # 1/1 from game 1
        assert result["home_rolling_point_diff"].iloc[1] == 7.0  # 27-20

    def test_tie_in_rolling(self):
        """Ties should be counted as non-wins in rolling stats."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_score": [16, 30],
                "away_score": [16, 24],
                "home_win": [pd.NA, 1],
            }
        )
        result = compute_rolling_features(df)
        # KC drew game 1 → win pct for game 2 should be 0/1 = 0.0
        assert result["home_rolling_win_pct"].iloc[1] == 0.0
        # Point diff for KC from game 1: 16-16 = 0
        assert result["home_rolling_point_diff"].iloc[1] == 0.0

    def test_window_size(self):
        """Rolling window should respect the configured window size."""
        df_list = []
        for i in range(10):
            df_list.append(
                {
                    "season": 2024,
                    "week": i + 1,
                    "gameday": f"2024-09-{5 + i:02d}",
                    "home_team": ["KC"],
                    "away_team": [f"OPP_{i}"],
                    "home_score": [24],
                    "away_score": [10],
                    "home_win": [1],
                }
            )
        df = pd.DataFrame(
            {
                "season": [2024] * 10,
                "week": list(range(1, 11)),
                "gameday": [f"2024-09-{5 + i:02d}" for i in range(10)],
                "home_team": ["KC"] * 10,
                "away_team": [f"OPP_{i}" for i in range(10)],
                "home_score": [24] * 10,
                "away_score": [10] * 10,
                "home_win": [1] * 10,
            }
        )
        result = compute_rolling_features(df, window=5)
        # Game 6 (index 5) should use games 0-4 → 5 games, all won
        assert result["home_rolling_win_pct"].iloc[5] == 1.0
        assert result["home_rolling_point_diff"].iloc[5] == 14.0

        # Game 2 (index 1) should use only game 0 → 1 game
        assert result["home_rolling_win_pct"].iloc[1] == 1.0

    def test_diff_columns(self):
        """Win pct diff and point diff diff should be correctly computed."""
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "gameday": ["2024-09-05", "2024-09-12"],
                "home_team": ["KC", "KC"],
                "away_team": ["BAL", "LV"],
                "home_score": [27, 30],
                "away_score": [20, 24],
                "home_win": [1, 1],
            }
        )
        result = compute_rolling_features(df)
        # Game 2: KC rolling = 1.0, LV first game = 0.5
        assert result["rolling_win_pct_diff"].iloc[1] == pytest.approx(0.5, abs=1e-6)
        # KC rolling pt diff = 7, LV = 0
        assert result["rolling_point_diff_diff"].iloc[1] == 7.0
