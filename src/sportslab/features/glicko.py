"""Glicko rating system for NFL team strength — purely pregame, no leakage.

Glicko-1 extends Elo with a rating deviation (RD) that captures uncertainty.
High RD → smaller g(RD) → predicted probability pulled toward 0.5 → more
conservative predictions for uncertain teams (new QB, early season, new coach).

Reference: Glickman (1995) "A Comprehensive Guide to Chess Ratings"
"""

import numpy as np
import pandas as pd

DEFAULT_GLICKO = 1500.0
DEFAULT_INITIAL_RD = 350.0
LN10 = np.log(10)
Q = LN10 / 400.0  # ≈ 0.005756

# Supported MOV types (reuse same logic as Elo)
MOV_NONE = "none"
MOV_CAPPED_LINEAR = "capped_linear"


def _g(rd: float) -> float:
    """Glicko g(RD) — scales prediction conservatism based on uncertainty."""
    return 1.0 / np.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (np.pi * np.pi))


def _glicko_expected(
    rating_home: float,
    rating_away: float,
    rd_away: float,
    hfa: float = 0.0,
) -> float:
    """Expected probability that home team wins, using Glicko rating + opponent RD.

    Uses only opponent's RD (standard Glicko-1).  High RD_away → lower g(RD_away)
    → predicted probability closer to 0.5.
    """
    g_rd = _g(rd_away)
    diff = (rating_home - rating_away + hfa) / 400.0
    return 1.0 / (1.0 + 10.0 ** (-g_rd * diff))


def _glicko_update(
    rating: float,
    rd: float,
    g_opp: float,
    expected: float,
    actual: float,
    mov_mult: float = 1.0,
) -> tuple[float, float]:
    """Glicko-1 rating update for a single team.

    Args:
        rating: Pre-game rating.
        rd: Pre-game rating deviation.
        g_opp: g(RD) of the opponent (controls how much opponent's uncertainty
            affects the update).
        expected: Pre-game expected score.
        actual: Actual outcome (1.0 = win, 0.0 = loss, 0.5 = tie).
        mov_mult: Margin-of-victory multiplier (>= 1.0, defaults to 1.0).

    Returns:
        (new_rating, new_rd).
    """
    d2 = 1.0 / (Q * Q * g_opp * g_opp * expected * (1.0 - expected) + 1e-15)
    update = (Q / (1.0 / rd + 1.0 / d2)) * g_opp * (actual - expected) * mov_mult
    new_rating = rating + update
    new_rd = np.sqrt(1.0 / (1.0 / (rd * rd) + 1.0 / d2))
    return new_rating, new_rd


def compute_glicko_features(
    df: pd.DataFrame,
    home_advantage: float = 0.0,
    initial_rating: float = DEFAULT_GLICKO,
    initial_rd: float = DEFAULT_INITIAL_RD,
    system_constant_c: float = 200.0,
    mov_type: str = MOV_NONE,
    mov_scale: float = 0.0,
    mov_cap: float | None = None,
    qb_rd_bonus: float = 0.0,
    qb_change_map: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Add pregame Glicko features to a game-level DataFrame.

    Sorts by (season, week, gameday) and processes games chronologically.
    Between seasons, RD increases by sqrt(RD^2 + c^2).
    If qb_rd_bonus > 0, teams with QB change get an additional RD boost at the
    start of that season.

    Args:
        df: DataFrame with columns season, week, gameday, home_team, away_team,
            home_win, home_score, away_score.
        home_advantage: Rating points added to home team for expected score.
        initial_rating: Starting rating for new teams.
        initial_rd: Starting rating deviation for new teams.
        system_constant_c: RD increase between seasons.
            RD_new = sqrt(RD^2 + c^2).
        mov_type: Margin-of-victory multiplier type.
        mov_scale: Scaling factor for MOV multiplier.
        mov_cap: Maximum MOV multiplier value.
        qb_rd_bonus: Additional RD increase for teams with QB change at season
            boundary.
        qb_change_map: Dict mapping team -> list of seasons where QB changed.
            Only used when qb_rd_bonus > 0.

    Returns:
        DataFrame with columns: home_glicko_pre, away_glicko_pre,
        home_glicko_rd, away_glicko_rd, glicko_diff, glicko_prob.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    rds: dict[str, float] = {}
    prev_season: int | None = None

    home_glicko = []
    away_glicko = []
    home_rd = []
    away_rd = []
    glicko_diff = []
    glicko_prob = []

    for _, row in out.iterrows():
        home: str = row["home_team"]
        away: str = row["away_team"]
        season: int = row["season"]

        # Season boundary: increase RD for all existing teams
        if prev_season is not None and season > prev_season:
            for team in rds:
                rds[team] = np.sqrt(rds[team] ** 2 + system_constant_c**2)
                # Additional RD boost for QB change
                if qb_rd_bonus > 0 and qb_change_map is not None:
                    changes = qb_change_map.get(team, [])
                    if season in changes:
                        rds[team] = np.sqrt(rds[team] ** 2 + qb_rd_bonus**2)
        prev_season = season

        h_rating = ratings.get(home, initial_rating)
        a_rating = ratings.get(away, initial_rating)
        h_rd_val = rds.get(home, initial_rd)
        a_rd_val = rds.get(away, initial_rd)

        # Expected probability (home uses away's RD, away uses home's RD)
        exp_home = _glicko_expected(h_rating, a_rating, a_rd_val, hfa=home_advantage)

        home_glicko.append(h_rating)
        away_glicko.append(a_rating)
        home_rd.append(h_rd_val)
        away_rd.append(a_rd_val)
        glicko_diff.append(h_rating - a_rating)
        glicko_prob.append(exp_home)

        # Update ratings
        home_won = row["home_win"]
        if pd.isna(home_won):
            actual_home = 0.5
        else:
            actual_home = float(home_won)

        actual_away = 1.0 - actual_home

        g_away = _g(a_rd_val)
        g_home = _g(h_rd_val)

        # MOV multiplier
        mov_mult = 1.0
        if mov_type != MOV_NONE and mov_scale > 0:
            diff = abs(float(row.get("home_score", 0)) - float(row.get("away_score", 0)))
            if diff >= 1:
                if mov_type == MOV_CAPPED_LINEAR:
                    base = 1.0 + mov_scale * diff
                    if mov_cap is not None:
                        mov_mult = min(base, float(mov_cap))
                    else:
                        mov_mult = base
                else:
                    from sportslab.features.ratings import _mov_multiplier

                    mov_mult = _mov_multiplier(
                        row.get("home_score", 0),
                        row.get("away_score", 0),
                        mov_type=mov_type,
                        mov_scale=mov_scale,
                        mov_cap=mov_cap,
                    )

        # Home update uses away's g(RD), away update uses home's g(RD)
        if not pd.isna(row.get("home_win")):
            new_h_rating, new_h_rd = _glicko_update(
                h_rating,
                h_rd_val,
                g_away,
                exp_home,
                actual_home,
                mov_mult,
            )
            exp_away = 1.0 - exp_home
            new_a_rating, new_a_rd = _glicko_update(
                a_rating,
                a_rd_val,
                g_home,
                exp_away,
                actual_away,
                mov_mult,
            )
        else:
            new_h_rating, new_h_rd = h_rating, h_rd_val
            new_a_rating, new_a_rd = a_rating, a_rd_val

        ratings[home] = new_h_rating
        rds[home] = new_h_rd
        ratings[away] = new_a_rating
        rds[away] = new_a_rd

    out["home_glicko_pre"] = home_glicko
    out["away_glicko_pre"] = away_glicko
    out["home_glicko_rd"] = home_rd
    out["away_glicko_rd"] = away_rd
    out["glicko_diff"] = glicko_diff
    out["glicko_prob"] = glicko_prob

    return out
