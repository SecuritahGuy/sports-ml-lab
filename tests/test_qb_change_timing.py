"""Step 4: QB-change feature timing audit.

The `home_qb_changed` and `away_qb_changed` features are derived from
the FINAL actual starter data from nflreadpy schedules. This is a
research backtest feature, not a live-pregame feature.

Timing assumption:
  - In backtesting, `home_qb_id` / `away_qb_id` reflect who actually
    started the game, which is only known post-game or shortly before
    kickoff (when inactives are announced).
  - For live prediction, the feature would need a pregame-announced
    starter source (e.g., injury reports, depth charts).
  - The feature compares current QB vs that team's PRIOR game starter
    (chronological, no look-ahead). This test verifies no future data
    is used.
"""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.qb import compute_qb_features


class TestQBChangeTiming:
    """QB change detection must be chronological and no look-ahead."""

    def test_first_game_same_qb_not_changed(self):
        """First game of a season for a team has no prior QB → qb_changed=0."""
        df = pd.DataFrame({
            "game_id": ["g1"],
            "season": [2021],
            "week": [1],
            "gameday": ["2021-09-12"],
            "home_team": ["ATL"],
            "away_team": ["ARI"],
            "home_qb_id": ["MATT01"],
            "away_qb_id": ["KYLE01"],
            "home_win": [1],
        })
        result = compute_qb_features(df)
        assert result.loc[0, "home_qb_changed"] == 0
        assert result.loc[0, "away_qb_changed"] == 0

    def test_qb_change_detected_across_games(self):
        """Same team, different QB in Game 2 → qb_changed=1 for Game 2."""
        df = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": ["2021-09-12", "2021-09-19"],
            "home_team": ["ATL", "ATL"],
            "away_team": ["ARI", "CHI"],
            "home_qb_id": ["MATT01", "DESM01"],  # QB changed
            "away_qb_id": ["KYLE01", "JUST01"],
            "home_win": [1, 0],
        })
        result = compute_qb_features(df)
        assert result.loc[0, "home_qb_changed"] == 0, "G1: first game, no prior QB"
        assert result.loc[1, "home_qb_changed"] == 1, "G2: QB changed from G1"
        assert result.loc[0, "away_qb_changed"] == 0, "G1: first game for away"

    def test_no_look_ahead(self):
        """QB change feature must not use future game data.

        Create 3 games for the same team alternating QBs A, B, C.
        Game 1: QB=A, no prior → changed=0
        Game 2: QB=B, prior=A → changed=1
        Game 3: QB=C, prior=B → changed=1 (depends only on G2, not G4)
        """
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "season": [2021, 2021, 2021],
            "week": [1, 2, 3],
            "gameday": ["2021-09-12", "2021-09-19", "2021-09-26"],
            "home_team": ["ATL", "ATL", "ATL"],
            "away_team": ["ARI", "CHI", "DET"],
            "home_qb_id": ["QB_A", "QB_B", "QB_C"],
            "away_qb_id": ["QB_X", "QB_Y", "QB_Z"],
            "home_win": [1, 0, 1],
        })
        result = compute_qb_features(df)
        assert result.loc[0, "home_qb_changed"] == 0, "First game: no prior QB"
        assert result.loc[1, "home_qb_changed"] == 1, "Second game: QB changed from A→B"
        assert result.loc[2, "home_qb_changed"] == 1, "Third game: QB changed from B→C"

    def test_qb_unchanged_across_games(self):
        """Same QB across games → qb_changed=0."""
        df = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": ["2021-09-12", "2021-09-19"],
            "home_team": ["ATL", "ATL"],
            "away_team": ["ARI", "CHI"],
            "home_qb_id": ["MATT01", "MATT01"],
            "away_qb_id": ["KYLE01", "KYLE01"],
            "home_win": [1, 0],
        })
        result = compute_qb_features(df)
        assert result.loc[1, "home_qb_changed"] == 0
        assert result.loc[1, "away_qb_changed"] == 0

    def test_qb_missing_does_not_crash(self):
        """Missing QB data sets missing_flag=1 and changed=0."""
        df = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": ["2021-09-12", "2021-09-19"],
            "home_team": ["ATL", "ATL"],
            "away_team": ["ARI", "CHI"],
            "home_qb_id": ["MATT01", None],
            "away_qb_id": ["KYLE01", "KYLE01"],
            "home_win": [1, 0],
        })
        result = compute_qb_features(df)
        assert result.loc[1, "home_qb_missing_flag"] == 1
        assert result.loc[1, "home_qb_changed"] == 0

    def test_season_boundary_resets_qb_state(self):
        """QB tracking state should reset at season boundary."""
        df = pd.DataFrame({
            "game_id": ["2021_17", "2022_01"],
            "season": [2021, 2022],
            "week": [17, 1],
            "gameday": ["2022-01-09", "2022-09-11"],
            "home_team": ["ATL", "ATL"],
            "away_team": ["ARI", "CHI"],
            "home_qb_id": ["MATT01", "DESM01"],
            "away_qb_id": ["KYLE01", "JUST01"],
            "home_win": [1, 1],
        })
        result = compute_qb_features(df)
        # 2022 first game → new season, no prior QB → changed=0
        assert result.loc[1, "home_qb_changed"] == 0, (
            "Season boundary resets QB state: first game of new season should not count as a change"
        )

    def test_qb_change_vs_prior_team_game_only(self):
        """QB change detection only compares vs the same team's prior game.

        QB_A plays for ATL in G1, then QB_A plays for CHI in G2.
        ATL G2 has a different QB. CHI G2 has QB_A for the first time.
        This should not trigger a false change for CHI (no prior CHI game).
        """
        df = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": ["2021-09-12", "2021-09-19"],
            "home_team": ["ATL", "CHI"],
            "away_team": ["ARI", "ATL"],
            "home_qb_id": ["QB_A", "QB_A"],  # QB_A now with CHI
            "away_qb_id": ["QB_X", "QB_B"],  # ATL now has QB_B
            "home_win": [1, 1],
        })
        result = compute_qb_features(df)
        # G2: CHI home, QB_A. First CHI game → no prior QB → changed=0
        assert result.loc[1, "home_qb_changed"] == 0
        # G2: ATL away, QB_B. Prior ATL QB was QB_A → changed=1
        assert result.loc[1, "away_qb_changed"] == 1
