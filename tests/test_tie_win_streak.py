"""Tests for tie win-streak fix in situational.py.

Verifies that ties reset win_streak to 0 (rather than counting as
a loss for both teams as before).
"""

import pandas as pd

from sportslab.features.situational import compute_situational_features


def _make_df(home_wins):
    """Build a minimal DataFrame for testing."""
    rows = []
    for i, hw in enumerate(home_wins):
        is_tie = pd.isna(hw)
        gid = f"g{i+1}"
        if is_tie:
            hs, aw, res = 17, 17, 0.0
        elif hw == 1:
            hs, aw, res = 24, 10, 14.0
        else:
            hs, aw, res = 10, 24, -14.0
        rows.append({
            "game_id": gid,
            "season": 2025,
            "week": i + 1,
            "gameday": f"2025-09-{7 + i}",
            "home_team": "ATL",
            "away_team": "ARI",
            "home_score": hs,
            "away_score": aw,
            "result": res,
            "home_win": hw,
            "location": "",
            "roof": "outdoors",
            "surface": "grass",
            "stadium": "Stadium",
            "gametime": "13:00",
            "weekday": "Sunday",
            "home_rest": 7,
            "away_rest": 7,
            "rest_diff": 0,
        })
    return pd.DataFrame(rows)


def test_win_streak_positive():
    """Consecutive wins produce positive win_streak."""
    df = _make_df([1, 1, 1])
    result = compute_situational_features(df)
    assert result.loc[0, "home_win_streak"] == 0, "First game: no prior streak"
    assert result.loc[1, "home_win_streak"] == 1, "Second game: 1-game streak"
    assert result.loc[2, "home_win_streak"] == 2, "Third game: 2-game streak"


def test_win_streak_negative():
    """Consecutive losses produce negative win_streak."""
    df = _make_df([0, 0, 0])
    result = compute_situational_features(df)
    assert result.loc[0, "home_win_streak"] == 0, "G1 pregame: no prior"
    assert result.loc[1, "home_win_streak"] == -1, "G2 pregame: after 1 loss"
    assert result.loc[2, "home_win_streak"] == -2, "G3 pregame: after 2 consecutive losses"


def test_tie_resets_streak():
    """Tie sets win_streak to 0 (doesn't count as win or loss)."""
    df = _make_df([1, pd.NA, 1])
    result = compute_situational_features(df)
    assert result.loc[0, "home_win_streak"] == 0
    assert result.loc[1, "home_win_streak"] == 1, "G2 pregame: after 1 win, streak=1"
    assert result.loc[2, "home_win_streak"] == 0, "G3 pregame: tie reset streak to 0"


def test_tie_after_win():
    """Tie after win resets to 0, then next win starts at 1."""
    df = _make_df([1, pd.NA, 1])
    result = compute_situational_features(df)
    assert result.loc[0, "home_win_streak"] == 0, "G1 pregame: no prior"
    assert result.loc[1, "home_win_streak"] == 1, "G2 pregame: after 1 win"
    assert result.loc[2, "home_win_streak"] == 0, "G3 pregame: tie reset to 0"


def test_tie_after_loss():
    """Tie after loss resets to 0."""
    df = _make_df([0, pd.NA, 0])
    result = compute_situational_features(df)
    assert result.loc[1, "home_win_streak"] == -1, "G2 pregame: after 1 loss"
    assert result.loc[2, "home_win_streak"] == 0, "G3 pregame: tie reset to 0"


def test_tie_does_not_increment_wins_or_losses():
    """Ties don't increment W or L counts."""
    df = _make_df([1, pd.NA, 0])
    result = compute_situational_features(df)
    # G1 pre: 0. ATL wins. Post: 1 win
    # G2 pre: 1. Tie. Post: 0 (tie reset)
    # G3 pre: 0. ATL loses. Post: 1 loss
    assert result.loc[0, "home_win_streak"] == 0, "G1 pregame"
    assert result.loc[1, "home_win_streak"] == 1, "G2 pregame: after 1 win"
    assert result.loc[2, "home_win_streak"] == 0, "G3 pregame: tie reset"


def test_away_team_streak():
    """Away team streaks work symmetrically."""
    df = _make_df([1, 0, pd.NA, 0])
    result = compute_situational_features(df)
    # Away team (ARI) lost G1, won G2, tied G3, won G4
    # G1 pre: 0. ARI loses. Post: streak=-1
    # G2 pre: -1. ARI wins. Post: streak=1
    # G3 pre: 1. Tie. Post: streak=0 (reset by tie)
    # G4 pre: 0. ARI wins. Post: streak=1
    assert result.loc[2, "away_win_streak"] == 1, "G3 pregame: after win, away streak=1"
    assert result.loc[3, "away_win_streak"] == 0, "G4 pregame: tie reset to 0"
