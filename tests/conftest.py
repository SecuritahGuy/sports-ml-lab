"""Shared test fixtures for sportslab tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_schedule() -> pd.DataFrame:
    """Minimal 4-game schedule with 3 teams, no ties, no neutrals.

    Two teams over 2 weeks (games 1-2), then a third team introduced (games 3-4).
    All rows are model-eligible (home_win is 0 or 1).
    """
    return pd.DataFrame(
        {
            "game_id": ["2021_01_ARI_ATL", "2021_01_ATL_ARI", "2021_02_CHI_ATL", "2021_02_ARI_CHI"],
            "season": [2021, 2021, 2021, 2021],
            "week": [1, 1, 2, 2],
            "gameday": ["2021-09-12", "2021-09-12", "2021-09-19", "2021-09-19"],
            "home_team": ["ATL", "ARI", "ATL", "CHI"],
            "away_team": ["ARI", "ATL", "CHI", "ARI"],
            "home_score": [24, 17, 10, 27],
            "away_score": [10, 24, 20, 10],
            "home_win": [1, 0, 0, 1],
            "location": ["", "", "", ""],
            "roof": ["outdoors", "outdoors", "dome", "outdoors"],
            "surface": ["grass", "grass", "grass", "grass"],
        }
    )


@pytest.fixture
def sample_schedule_with_tie() -> pd.DataFrame:
    """Schedule with one tie game and one non-tie game."""
    return pd.DataFrame(
        {
            "game_id": ["2021_01_ARI_ATL", "2021_01_ATL_CHI"],
            "season": [2021, 2021],
            "week": [1, 1],
            "gameday": ["2021-09-12", "2021-09-12"],
            "home_team": ["ATL", "CHI"],
            "away_team": ["ARI", "ATL"],
            "home_score": [24, 10],
            "away_score": [10, 10],
            "home_win": [1, pd.NA],  # second row is a tie
            "location": ["", ""],
            "roof": ["outdoors", "dome"],
            "surface": ["grass", "grass"],
        }
    )


@pytest.fixture
def sample_schedule_multi_season() -> pd.DataFrame:
    """5 games across 2021 and 2022 with same teams."""
    return pd.DataFrame(
        {
            "game_id": [
                "2021_01_ARI_ATL",
                "2021_01_ATL_ARI",
                "2021_02_ARI_ATL",
                "2022_01_ARI_ATL",
                "2022_01_ATL_ARI",
            ],
            "season": [2021, 2021, 2021, 2022, 2022],
            "week": [1, 1, 2, 1, 1],
            "gameday": [
                "2021-09-12",
                "2021-09-12",
                "2021-09-19",
                "2022-09-11",
                "2022-09-11",
            ],
            "home_team": ["ATL", "ARI", "ATL", "ARI", "ATL"],
            "away_team": ["ARI", "ATL", "ARI", "ATL", "ARI"],
            "home_score": [24, 10, 17, 20, 30],
            "away_score": [10, 24, 24, 17, 10],
            "home_win": [1, 0, 0, 1, 1],
            "location": ["", "", "", "", ""],
            "roof": ["outdoors"] * 5,
            "surface": ["grass"] * 5,
        }
    )


@pytest.fixture
def small_elo_result(sample_schedule) -> dict:
    """Precomputed Elo results for the first 2 games in sample_schedule.

    Parameters: K=36, HFA=40, reg=0, mov_type='none', decay=None.

    Game 1: ATL (home, 1500) vs ARI (away, 1500)
      elo_prob = 1 / (1 + 10^(-(1500 - 1500 + 40) / 400))
               = 1 / (1 + 10^(-0.1))
               = 1 / (1 + 0.794328)
               = 1 / 1.794328
               = 0.5573
      update = 36 * (1 - 0.5573) * 1  (no MOV)
             = 36 * 0.4427
             = 15.937
      ATL: 1500 + 15.937 = 1515.937
      ARI: 1500 - 15.937 = 1484.063

    Game 2: ARI (home, 1484.063) vs ATL (away, 1515.937)
      elo_prob = 1 / (1 + 10^(-(1484.063 - 1515.937 + 40) / 400))
               = 1 / (1 + 10^(-(8.126) / 400))
               = 1 / (1 + 10^(-0.020315))
               = 1 / (1 + 0.9543)
               = 0.5116
    """
    return {
        "game1_home_elo_pre": 1500.0,
        "game1_away_elo_pre": 1500.0,
        "game1_elo_prob": 0.5573,
        "game2_home_elo_pre": 1484.063,
        "game2_away_elo_pre": 1515.937,
        "game2_elo_prob": 0.5116,
    }
