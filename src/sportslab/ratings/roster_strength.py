"""V1 scaffold: position-group roster strength rating system.

Target design:
    team_roster_points = qb_points
                        + ol_points
                        + skill_points
                        + defensive_front_points
                        + lb_points
                        + coverage_points
                        + special_teams_points
                        + injury_adjustment

Current implementation (V0):
    - QB points fully populated using compute_qb_adjustments()
    - All other position groups return 0 (placeholder)
    - Injury adjustment: 0 (placeholder)

This scaffold establishes the interface, tables, and data contracts
so that future position-group modules can be plugged in without
refactoring the experiment pipeline.
"""

from typing import Dict

import numpy as np
import pandas as pd

from sportslab.features.qb_adjustment import compute_qb_adjustments

ROSTER_STRENGTH_COLUMNS = [
    "roster_qb_points",
    "roster_ol_points",
    "roster_skill_points",
    "roster_front_points",
    "roster_lb_points",
    "roster_coverage_points",
    "roster_st_points",
    "roster_injury_adj",
    "roster_total_adjustment",
]

# Weight (in Elo points) per unit of each position group rating.
# These are placeholder defaults — to be tuned in future experiments.
POSITION_WEIGHTS: Dict[str, float] = {
    "qb": 1.0,
    "ol": 0.5,
    "skill": 0.4,
    "front": 0.3,
    "lb": 0.2,
    "coverage": 0.2,
    "st": 0.1,
}

# Column name constants for downstream merging
HOME_PREFIX = "home_"
AWAY_PREFIX = "away_"

HOME_ROSTER_COLUMNS = [f"{HOME_PREFIX}{c}" for c in ROSTER_STRENGTH_COLUMNS]
AWAY_ROSTER_COLUMNS = [f"{AWAY_PREFIX}{c}" for c in ROSTER_STRENGTH_COLUMNS]
ALL_ROSTER_COLUMNS = HOME_ROSTER_COLUMNS + AWAY_ROSTER_COLUMNS


def compute_roster_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Compute roster-strength features for each game.

    V0: Only QB points are populated (via compute_qb_adjustments).
    All other position groups return 0.

    Args:
        df: DataFrame with columns from compute_elo_features()
            (season, week, gameday, home_team, away_team,
             home_elo_pre, away_elo_pre, home_qb_id, away_qb_id, etc.)

    Returns:
        DataFrame with added roster strength columns.
        Non-QB columns are 0 — ready for future V1+ expansion.
    """
    out = df.copy()

    # QB points from the QB adjustment system
    out = compute_qb_adjustments(out)

    # Map QB adj (in Elo points) to roster_qb_points
    out[f"{HOME_PREFIX}roster_qb_points"] = out["home_qb_adj"].fillna(0.0)
    out[f"{AWAY_PREFIX}roster_qb_points"] = out["away_qb_adj"].fillna(0.0)

    # Placeholder position groups (V1+)
    for prefix in [HOME_PREFIX, AWAY_PREFIX]:
        out[f"{prefix}roster_ol_points"] = 0.0
        out[f"{prefix}roster_skill_points"] = 0.0
        out[f"{prefix}roster_front_points"] = 0.0
        out[f"{prefix}roster_lb_points"] = 0.0
        out[f"{prefix}roster_coverage_points"] = 0.0
        out[f"{prefix}roster_st_points"] = 0.0
        out[f"{prefix}roster_injury_adj"] = 0.0

    # Total roster adjustment (weighted sum of position groups)
    # V0: only QB points contribute
    for prefix in [HOME_PREFIX, AWAY_PREFIX]:
        total = np.zeros(len(out), dtype=float)
        for group, weight in POSITION_WEIGHTS.items():
            col = f"{prefix}roster_{group}_points"
            if col in out.columns:
                total += out[col].values * weight
        total += out[f"{prefix}roster_injury_adj"].values
        out[f"{prefix}roster_total_adjustment"] = total

    return out


def compute_roster_adjusted_elo_prob(
    home_elo: np.ndarray,
    away_elo: np.ndarray,
    home_roster_total: np.ndarray,
    away_roster_total: np.ndarray,
    hfa: float = 40.0,
) -> np.ndarray:
    """Compute home win probability with roster-adjusted Elo ratings.

    team_effective_elo = team_elo + roster_total_adjustment
    """
    home_effective = home_elo + home_roster_total
    away_effective = away_elo + away_roster_total
    diff = (home_effective - away_effective + hfa) / 400.0
    return 1.0 / (1.0 + 10.0 ** (-diff))
