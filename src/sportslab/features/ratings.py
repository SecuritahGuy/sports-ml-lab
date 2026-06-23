"""Elo rating system for NFL team strength — purely pregame, no leakage.

Supports configurable K-factor, home-field advantage, preseason
regression toward the mean, and margin-of-victory (MOV) multipliers
on rating updates.
"""

import numpy as np
import pandas as pd

DEFAULT_ELO = 1500.0
DEFAULT_K = 20
ELO_DIVISOR = 400.0

# Supported MOV types
MOV_NONE = "none"
MOV_LOG = "log"
MOV_SQRT = "sqrt"
MOV_CAPPED_LOG = "capped_log"
MOV_CAPPED_LINEAR = "capped_linear"


def _effective_expected(rating_a: float, rating_b: float, hfa: float = 0.0) -> float:
    """Expected score for team A (home) against team B (away), with HFA."""
    diff = (rating_a - rating_b + hfa) / ELO_DIVISOR
    return 1.0 / (1.0 + 10.0 ** (-diff))


def elo_diff_to_win_prob(elo_home: float, elo_away: float, home_advantage: float = 0.0) -> float:
    """Convert pregame Elo difference to home win probability."""
    return _effective_expected(elo_home, elo_away, hfa=home_advantage)


def _mov_multiplier(
    home_score: float,
    away_score: float,
    mov_type: str = MOV_NONE,
    mov_scale: float = 0.0,
    mov_cap: float | None = None,
) -> float:
    """Compute margin-of-victory multiplier for Elo rating update.

    Always >= 1.0 (a win is always at least a standard update).
    Only meaningful when home_score != away_score (non-tie).

    Args:
        home_score: Home team's score.
        away_score: Away team's score.
        mov_type: One of MOV_NONE, MOV_LOG, MOV_SQRT,
            MOV_CAPPED_LOG, MOV_CAPPED_LINEAR.
        mov_scale: Scaling factor for the MOV term.
        mov_cap: Maximum multiplier value (None = no cap).

    Returns:
        Multiplier >= 1.0.
    """
    if mov_type == MOV_NONE or mov_scale <= 0:
        return 1.0

    diff = abs(float(home_score) - float(away_score))
    if diff < 1:
        return 1.0

    if mov_type in (MOV_LOG, MOV_CAPPED_LOG):
        base = 1.0 + mov_scale * np.log(1.0 + diff)
    elif mov_type in (MOV_SQRT, MOV_CAPPED_LINEAR):
        if mov_type == MOV_SQRT:
            base = 1.0 + mov_scale * np.sqrt(diff)
        else:
            base = 1.0 + mov_scale * diff
    else:
        return 1.0

    if mov_cap is not None and mov_type in (MOV_CAPPED_LOG, MOV_CAPPED_LINEAR):
        return min(base, float(mov_cap))
    return base


def compute_elo_features(
    df: pd.DataFrame,
    k_factor: float = DEFAULT_K,
    home_advantage: float = 0.0,
    default_elo: float = DEFAULT_ELO,
    preseason_regression: float = 0.0,
    mov_type: str = MOV_NONE,
    mov_scale: float = 0.0,
    mov_cap: float | None = None,
) -> pd.DataFrame:
    """Add pregame Elo features to a game-level DataFrame.

    Sorts by (season, week, gameday) and processes games chronologically.

    Args:
        df: DataFrame with columns season, week, gameday, home_team, away_team,
            home_win, home_score, away_score.
        k_factor: Elo K-factor controlling update magnitude.
        home_advantage: HFA added to home team's effective rating for expected
            score calculation only.
        default_elo: Starting Elo for new teams.
        preseason_regression: Fraction (0–1) of regression toward default_elo
            at each season boundary.
        mov_type: Margin-of-victory multiplier type.
        mov_scale: Scaling factor for MOV multiplier.
        mov_cap: Maximum multiplier value (None = no cap).

    Returns:
        DataFrame with additional columns: home_elo_pre, away_elo_pre, elo_diff,
        elo_prob.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    prev_season: int | None = None

    home_elo = []
    away_elo = []
    elo_diff = []
    elo_prob = []

    for _, row in out.iterrows():
        home_team: str = row["home_team"]
        away_team: str = row["away_team"]
        season: int = row["season"]

        # Preseason regression at season boundary
        if prev_season is not None and season > prev_season and preseason_regression > 0:
            for team in ratings:
                r = ratings[team]
                ratings[team] = default_elo + (1.0 - preseason_regression) * (r - default_elo)
        prev_season = season

        h_elo = ratings.get(home_team, default_elo)
        a_elo = ratings.get(away_team, default_elo)

        home_elo.append(h_elo)
        away_elo.append(a_elo)
        elo_diff.append(h_elo - a_elo)
        elo_prob.append(_effective_expected(h_elo, a_elo, hfa=home_advantage))

        # Update ratings (ties count as 0.5 for each team)
        home_won = row["home_win"]
        if pd.isna(home_won):
            actual_home = 0.5
        else:
            actual_home = float(home_won)

        expected_home = _effective_expected(h_elo, a_elo, hfa=home_advantage)
        mov_mult = _mov_multiplier(
            row.get("home_score", 0),
            row.get("away_score", 0),
            mov_type=mov_type,
            mov_scale=mov_scale,
            mov_cap=mov_cap,
        )
        update = k_factor * (actual_home - expected_home) * mov_mult
        ratings[home_team] = h_elo + update
        ratings[away_team] = a_elo - update

    out["home_elo_pre"] = home_elo
    out["away_elo_pre"] = away_elo
    out["elo_diff"] = elo_diff
    out["elo_prob"] = elo_prob

    return out
