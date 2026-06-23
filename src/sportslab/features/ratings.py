"""Elo rating system for NFL team strength — purely pregame, no leakage."""

import pandas as pd

DEFAULT_ELO = 1500.0
K_FACTOR = 20
ELO_DIVISOR = 400.0


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for team A against team B (logistic Elo formula)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_DIVISOR))


def elo_diff_to_win_prob(elo_home: float, elo_away: float) -> float:
    """Convert pregame Elo difference to home win probability.

    Uses the standard logistic Elo formula.
    Returns a probability in [0, 1].
    """
    return _expected_score(elo_home, elo_away)


def compute_elo_features(df: pd.DataFrame, k_factor: int = K_FACTOR) -> pd.DataFrame:
    """Add pregame Elo features to a game-level DataFrame.

    Sorts by (season, week, gameday) and processes games chronologically.
    For each game:
      1. Records pregame Elo for both teams.
      2. Updates ratings based on the game result (ties = 0.5).

    Args:
        df: DataFrame with columns season, week, gameday, home_team, away_team,
            home_win (0, 1, or NA for ties).
        k_factor: Elo K-factor controlling update magnitude (default 20).

    Returns:
        DataFrame with additional columns: home_elo_pre, away_elo_pre, elo_diff.
    """
    out = df.copy()
    sorted_df = out.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    ratings: dict[str, float] = {}

    home_elo = []
    away_elo = []
    elo_diff = []

    for _, row in sorted_df.iterrows():
        home_team: str = row["home_team"]
        away_team: str = row["away_team"]

        h_elo = ratings.get(home_team, DEFAULT_ELO)
        a_elo = ratings.get(away_team, DEFAULT_ELO)

        home_elo.append(h_elo)
        away_elo.append(a_elo)
        elo_diff.append(h_elo - a_elo)

        # Update ratings (ties count as 0.5 for each team)
        home_won = row["home_win"]
        if pd.isna(home_won):
            actual_home = 0.5
        else:
            actual_home = float(home_won)

        expected_home = _expected_score(h_elo, a_elo)
        update = k_factor * (actual_home - expected_home)
        ratings[home_team] = h_elo + update
        ratings[away_team] = a_elo - update

    out["home_elo_pre"] = home_elo
    out["away_elo_pre"] = away_elo
    out["elo_diff"] = elo_diff

    return out
