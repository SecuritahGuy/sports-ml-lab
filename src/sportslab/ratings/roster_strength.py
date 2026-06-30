"""V1 scaffold: position-group roster strength rating system.

Architecture:
    team_roster_adjustment = qb_points * qb_weight
                           + ol_points * ol_weight
                           + skill_points * skill_weight
                           + defensive_front_points * front_weight
                           + lb_points * lb_weight
                           + coverage_points * coverage_weight
                           + injury_adjustment

    team_effective_elo = team_elo + team_roster_adjustment

Current implementation (V1):
    - Position-group points populated from injury-based availability scores
    - Each group's points = weight * (2 * availability - 1), mapping [0,1] to [-weight, weight]
    - Injury adjustment uses total_out count (placeholder weight)
"""

from typing import Dict

import numpy as np
import pandas as pd

from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.roster_availability import (
    POSITION_GROUP_DEPTH,
    compute_roster_availability,
)

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

POSITION_WEIGHTS: Dict[str, float] = {
    "qb": 1.0,
    "ol": 0.5,
    "skill": 0.4,
    "front": 0.3,
    "lb": 0.2,
    "coverage": 0.2,
    "st": 0.1,
}

HOME_PREFIX = "home_"
AWAY_PREFIX = "away_"

HOME_ROSTER_COLUMNS = [f"{HOME_PREFIX}{c}" for c in ROSTER_STRENGTH_COLUMNS]
AWAY_ROSTER_COLUMNS = [f"{AWAY_PREFIX}{c}" for c in ROSTER_STRENGTH_COLUMNS]
ALL_ROSTER_COLUMNS = HOME_ROSTER_COLUMNS + AWAY_ROSTER_COLUMNS

# Groups with injury-based availability (non-QB)
AVAILABILITY_GROUPS = ["ol", "skill", "front", "lb", "coverage", "st"]


def compute_roster_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Compute roster-strength features for each game.

    V1: All position groups are populated from injury report availability.
    Each group's points = weight * (2 * availability - 1), ranging from
    -weight (fully depleted) to +weight (fully healthy).

    Args:
        df: DataFrame with columns from compute_elo_features()
            (season, week, gameday, home_team, away_team,
             home_elo_pre, away_elo_pre, home_qb_id, away_qb_id, etc.)

    Returns:
        DataFrame with added roster strength columns.
    """
    out = df.copy()

    # QB points from the QB adjustment system
    out = compute_qb_adjustments(out)
    out[f"{HOME_PREFIX}roster_qb_points"] = out["home_qb_adj"].fillna(0.0)
    out[f"{AWAY_PREFIX}roster_qb_points"] = out["away_qb_adj"].fillna(0.0)

    # Availability scores for all position groups
    out = compute_roster_availability(out)

    # V1: populate non-QB groups from availability scores
    for prefix in [HOME_PREFIX, AWAY_PREFIX]:
        for group in AVAILABILITY_GROUPS:
            avail_col = f"{prefix}{group}_availability"
            pts_col = f"{prefix}roster_{group}_points"
            weight = POSITION_WEIGHTS.get(group, 0.0)
            avail = out.get(avail_col, pd.Series(1.0, index=out.index))
            out[pts_col] = (weight * (2 * avail - 1)).round(2)

        # Injury adjustment: sum of all position-group out counts * -0.1
        total_out = np.zeros(len(out), dtype=float)
        for group, depth in POSITION_GROUP_DEPTH.items():
            out_col = f"{prefix}{group}_out"
            if out_col in out.columns:
                total_out += out[out_col].fillna(0).astype(float).values
        out[f"{prefix}roster_injury_adj"] = (-0.1 * total_out).round(2)

    # Total roster adjustment (weighted sum of position groups + injury adj)
    for prefix in [HOME_PREFIX, AWAY_PREFIX]:
        total = np.zeros(len(out), dtype=float)
        for group, weight in POSITION_WEIGHTS.items():
            col = f"{prefix}roster_{group}_points"
            if col in out.columns:
                total += out[col].values * weight
        total += out[f"{prefix}roster_injury_adj"].values
        out[f"{prefix}roster_total_adjustment"] = total.round(2)

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
