"""Per-team home field advantage computation.

Computes team-specific HFA from historical game margins — purely
pregame-safe, computed only from past games.
"""

import numpy as np
import pandas as pd


def compute_team_hfa(
    df: pd.DataFrame,
    seasons: list[int],
    min_games: int = 4,
) -> dict[str, float]:
    """Compute per-team HFA offset using point differential.

    For each team, the HFA offset is the difference between their
    average home margin and average away margin over the given seasons.
    A positive offset means the team has a stronger-than-average home
    field; a negative offset means weaker.

    Args:
        df: Game-level DataFrame with home_team, away_team, home_score,
            away_score, season columns.
        seasons: List of seasons to include for estimation.
        min_games: Minimum home games required per team.  Teams below
            this threshold get 0 offset.

    Returns:
        Dict mapping team name to HFA offset (Elo points).
    """
    sub = df[df["season"].isin(seasons)].copy()
    if sub.empty:
        return {}

    # Home margins
    home = (
        sub.groupby("home_team")
        .apply(
            lambda g: (g["home_score"] - g["away_score"]).mean(),
            include_groups=False,
        )
        .to_dict()
    )

    # Away margins
    away_margins = (
        sub.groupby("away_team")
        .apply(
            lambda g: -(g["home_score"] - g["away_score"]).mean(),
            include_groups=False,
        )
        .to_dict()
    )

    # Home game counts
    home_counts = sub.groupby("home_team").size().to_dict()

    all_teams = set(home.keys()) | set(away_margins.keys())
    hfa: dict[str, float] = {}
    for team in all_teams:
        h_margin = home.get(team, 0.0)
        a_margin = away_margins.get(team, 0.0)
        count = home_counts.get(team, 0)
        if count < min_games:
            hfa[team] = 0.0
        else:
            # HFA offset = home margin advantage over away margin
            # Scale to Elo units (roughly 25 Elo points per point of margin)
            advantage = h_margin - a_margin
            hfa[team] = advantage

    # Center so mean is 0
    values = [v for v in hfa.values()]
    if values:
        center = float(np.mean(values))
        for team in hfa:
            hfa[team] -= center

    return hfa


def margin_to_elo_hfa(margin_advantage: float) -> float:
    """Convert a point-differential advantage to Elo HFA offset.

    Rough calibration: 1 point of margin ≈ 25 Elo points.
    Cap at ±30 to prevent extreme values from small samples.
    """
    raw = margin_advantage * 25.0
    return float(np.clip(raw, -30.0, 30.0))
