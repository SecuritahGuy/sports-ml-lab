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


def compute_od_elo_features(
    df: pd.DataFrame,
    k_factor: float = DEFAULT_K,
    home_advantage: float = 0.0,
    default_elo: float = DEFAULT_ELO,
    preseason_regression: float = 0.0,
    mov_type: str = MOV_NONE,
    mov_scale: float = 0.0,
    mov_cap: float | None = None,
    decay_half_life: float | None = None,
    k_off: float | None = None,
    k_def: float | None = None,
    team_regression_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Separate offensive/defensive Elo ratings.

    Each team maintains independent off_elo and def_elo ratings (both 1500).
    For prediction, ratings are combined (off + def) into a total rating,
    matching standard Elo's expected score calculation.

    For updates, the K-factor for offense and defense can differ:
    - off_update = k_off * (actual - expected) * mov * pt_share
    - def_update = k_def * (actual - expected) * mov * (1 - pt_share)
    where pt_share = fraction of total points scored by the team.

    When k_off == k_def (both equal to k_factor), the total update always
    equals the standard Elo update.  When they differ, lopsided scoring
    distributions cause asymmetric total updates even for the same result.

    Args:
        As compute_elo_features, plus:
        k_off: K-factor for offense updates.  If None, uses k_factor.
        k_def: K-factor for defense updates.  If None, uses k_factor.

    Returns:
        DataFrame with columns: home_elo_pre, away_elo_pre, elo_diff, elo_prob,
        home_off_elo, home_def_elo, away_off_elo, away_def_elo.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    if k_off is None:
        k_off = k_factor
    if k_def is None:
        k_def = k_factor

    off_ratings: dict[str, float] = {}
    def_ratings: dict[str, float] = {}
    prev_season: int | None = None

    if decay_half_life is not None and decay_half_life > 0:
        decay_factor = 2.0 ** (-1.0 / decay_half_life)
    else:
        decay_factor = None

    home_total_elo: list[float] = []
    away_total_elo: list[float] = []
    elo_diff: list[float] = []
    elo_prob: list[float] = []
    home_off_pre: list[float] = []
    home_def_pre: list[float] = []
    away_off_pre: list[float] = []
    away_def_pre: list[float] = []

    for _, row in out.iterrows():
        home: str = row["home_team"]
        away: str = row["away_team"]
        season: int = row["season"]

        # Preseason regression at season boundary
        if prev_season is not None and season > prev_season and preseason_regression > 0:
            for team in set(list(off_ratings.keys()) + list(def_ratings.keys())):
                reg_frac = (
                    team_regression_overrides.get(team, preseason_regression)
                    if team_regression_overrides is not None
                    else preseason_regression
                )
                if reg_frac > 0:
                    off_r = off_ratings.get(team, default_elo)
                    def_r = def_ratings.get(team, default_elo)
                    off_ratings[team] = default_elo + (1.0 - reg_frac) * (off_r - default_elo)
                    def_ratings[team] = default_elo + (1.0 - reg_frac) * (def_r - default_elo)
        prev_season = season

        h_off = off_ratings.get(home, default_elo)
        h_def = def_ratings.get(home, default_elo)
        a_off = off_ratings.get(away, default_elo)
        a_def = def_ratings.get(away, default_elo)

        h_total = h_off + h_def
        a_total = a_off + a_def

        home_total_elo.append(h_total)
        away_total_elo.append(a_total)
        elo_diff.append(h_total - a_total)
        elo_prob.append(_effective_expected(h_total, a_total, hfa=home_advantage))

        home_off_pre.append(h_off)
        home_def_pre.append(h_def)
        away_off_pre.append(a_off)
        away_def_pre.append(a_def)

        # Combined expected score (same as standard Elo)
        expected_home = _effective_expected(h_total, a_total, hfa=home_advantage)
        home_won = row["home_win"]
        if pd.isna(home_won):
            actual_home = 0.5
        else:
            actual_home = float(home_won)

        mov_mult = _mov_multiplier(
            row.get("home_score", 0), row.get("away_score", 0),
            mov_type=mov_type, mov_scale=mov_scale, mov_cap=mov_cap,
        )

        # Points-scored share for offensive weighting
        hs = float(row.get("home_score", 0))
        as_ = float(row.get("away_score", 0))
        total_pts = hs + as_
        if total_pts > 0:
            home_pt_share = hs / total_pts
            away_pt_share = as_ / total_pts
        else:
            home_pt_share = 0.5
            away_pt_share = 0.5

        base = (actual_home - expected_home) * mov_mult

        # Home: off_update uses k_off * pt_share, def_update uses k_def * (1-pt_share)
        home_off_update = k_off * base * home_pt_share
        home_def_update = k_def * base * (1.0 - home_pt_share)

        # Away
        actual_away = 1.0 - actual_home
        expected_away = 1.0 - expected_home
        base_away = (actual_away - expected_away) * mov_mult

        away_off_update = k_off * base_away * away_pt_share
        away_def_update = k_def * base_away * (1.0 - away_pt_share)

        off_ratings[home] = h_off + home_off_update
        def_ratings[home] = h_def + home_def_update
        off_ratings[away] = a_off + away_off_update
        def_ratings[away] = a_def + away_def_update

        # Decay toward mean
        if decay_factor is not None:
            off_ratings[home] = default_elo + (off_ratings[home] - default_elo) * decay_factor
            def_ratings[home] = default_elo + (def_ratings[home] - default_elo) * decay_factor
            off_ratings[away] = default_elo + (off_ratings[away] - default_elo) * decay_factor
            def_ratings[away] = default_elo + (def_ratings[away] - default_elo) * decay_factor

    out["home_elo_pre"] = home_total_elo
    out["away_elo_pre"] = away_total_elo
    out["elo_diff"] = elo_diff
    out["elo_prob"] = elo_prob
    out["home_off_elo"] = home_off_pre
    out["home_def_elo"] = home_def_pre
    out["away_off_elo"] = away_off_pre
    out["away_def_elo"] = away_def_pre

    return out


def compute_elo_features(
    df: pd.DataFrame,
    k_factor: float = DEFAULT_K,
    home_advantage: float = 0.0,
    default_elo: float = DEFAULT_ELO,
    preseason_regression: float = 0.0,
    mov_type: str = MOV_NONE,
    mov_scale: float = 0.0,
    mov_cap: float | None = None,
    decay_half_life: float | None = None,
    team_hfa: dict[str, float] | None = None,
    team_regression_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add pregame Elo features to a game-level DataFrame.

    Sorts by (season, week, gameday) and processes games chronologically.

    Args:
        df: DataFrame with columns season, week, gameday, home_team, away_team,
            home_win, home_score, away_score.
        k_factor: Elo K-factor controlling update magnitude.
        home_advantage: Global HFA added to home team's effective rating for
            expected score calculation.
        default_elo: Starting Elo for new teams.
        preseason_regression: Fraction (0–1) of regression toward default_elo
            at each season boundary.
        mov_type: Margin-of-victory multiplier type.
        mov_scale: Scaling factor for MOV multiplier.
        mov_cap: Maximum multiplier value (None = no cap).
        decay_half_life: Number of games to halve rating deviation from mean.
            None = no decay. Lower = faster decay toward mean.
        team_hfa: Per-team HFA offsets. If provided, each team's effective HFA
            is home_advantage + team_hfa.get(team, 0.0).

    Returns:
        DataFrame with additional columns: home_elo_pre, away_elo_pre, elo_diff,
        elo_prob.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    prev_season: int | None = None

    if decay_half_life is not None and decay_half_life > 0:
        decay_factor = 2.0 ** (-1.0 / decay_half_life)
    else:
        decay_factor = None

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
                reg_frac = (
                    team_regression_overrides.get(team, preseason_regression)
                    if team_regression_overrides is not None
                    else preseason_regression
                )
                if reg_frac > 0:
                    r = ratings[team]
                    ratings[team] = default_elo + (1.0 - reg_frac) * (r - default_elo)
        prev_season = season

        h_elo = ratings.get(home_team, default_elo)
        a_elo = ratings.get(away_team, default_elo)

        # Effective HFA: global + per-team offset
        if team_hfa is not None:
            effective_hfa = home_advantage + team_hfa.get(home_team, 0.0)
        else:
            effective_hfa = home_advantage

        home_elo.append(h_elo)
        away_elo.append(a_elo)
        elo_diff.append(h_elo - a_elo)
        elo_prob.append(_effective_expected(h_elo, a_elo, hfa=effective_hfa))

        # Update ratings (ties count as 0.5 for each team)
        home_won = row["home_win"]
        if pd.isna(home_won):
            actual_home = 0.5
        else:
            actual_home = float(home_won)

        expected_home = _effective_expected(h_elo, a_elo, hfa=effective_hfa)
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

        # Exponential decay toward mean after each game
        if decay_factor is not None:
            ratings[home_team] = default_elo + (ratings[home_team] - default_elo) * decay_factor
            ratings[away_team] = default_elo + (ratings[away_team] - default_elo) * decay_factor

    out["home_elo_pre"] = home_elo
    out["away_elo_pre"] = away_elo
    out["elo_diff"] = elo_diff
    out["elo_prob"] = elo_prob

    return out
