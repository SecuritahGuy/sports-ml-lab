"""Position-group availability scores from injury data.

Computes per-game availability ratings for each position group
using injury report OUT counts. Availability is scored 0-1 where
1 = fully healthy (no starters OUT).
"""

from typing import Dict

import pandas as pd

from sportslab.features.injuries import compute_injury_features

ROSTER_AVAILABILITY_COLUMNS = [
    "home_qb_availability",
    "away_qb_availability",
    "home_ol_availability",
    "away_ol_availability",
    "home_skill_availability",
    "away_skill_availability",
    "home_front_availability",
    "away_front_availability",
    "home_lb_availability",
    "away_lb_availability",
    "home_coverage_availability",
    "away_coverage_availability",
    "home_st_availability",
    "away_st_availability",
    "home_overall_availability",
    "away_overall_availability",
]

POSITION_GROUP_DEPTH: Dict[str, int] = {
    "qb": 1,
    "ol": 5,
    "skill": 5,
    "front": 4,
    "lb": 3,
    "coverage": 4,
    "st": 3,
}

SIDE_PREFIXES = {"home": "home_", "away": "away_"}


def compute_roster_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game position-group availability scores.

    Adds ROSTER_AVAILABILITY_COLUMNS: each in [0, 1] where 1 = fully healthy.
    Requires injury feature columns (computed by compute_injury_features()).

    Args:
        df: DataFrame with schedule columns; injury features will be
            added automatically if missing.

    Returns:
        DataFrame with added availability columns.
    """
    out = df.copy()

    needed = {"home_qb_out", "home_ol_out", "home_front_out",
              "home_lb_out", "home_coverage_out"}
    if not needed.issubset(out.columns):
        out = compute_injury_features(out)

    for side, prefix in SIDE_PREFIXES.items():
        for group, depth in POSITION_GROUP_DEPTH.items():
            out_col = f"{prefix}{group}_out"
            avail_col = f"{prefix}{group}_availability"
            raw = out.get(out_col, pd.Series(0, index=out.index))
            raw_num = raw.fillna(0).astype(int)
            depletion = (raw_num / depth).clip(upper=1.0)
            out[avail_col] = (1.0 - depletion).round(4)

        group_cols = [f"{prefix}{g}_availability" for g in POSITION_GROUP_DEPTH]
        present = [c for c in group_cols if c in out.columns]
        out[f"{prefix}overall_availability"] = out[present].mean(axis=1).round(4) if present else 1.0

    return out
