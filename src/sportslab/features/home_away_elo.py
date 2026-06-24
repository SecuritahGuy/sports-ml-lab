"""Home/away separate Elo ratings — each team has a home rating and an away rating."""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _mov_mult(mov: float, mov_type: str, scale: float, cap: Optional[float]) -> float:
    if mov_type == "none" or scale <= 0 or mov <= 0:
        return 1.0
    if mov_type in ("log", "capped_log"):
        val = np.log1p(mov) * scale
    elif mov_type == "sqrt":
        val = np.sqrt(mov) * scale
    elif mov_type == "capped_linear":
        val = mov * scale
    else:
        return 1.0
    if cap is not None and mov_type in ("capped_log", "capped_linear"):
        val = min(val, cap)
    return max(val, 0.5) if scale > 0 else 1.0


def compute_home_away_elo(
    df: pd.DataFrame,
    k_factor: float = 36,
    home_advantage: float = 40,
    default_elo: float = 1500,
    preseason_regression: float = 0.1,
    mov_type: str = "capped_linear",
    mov_scale: float = 0.05,
    mov_cap: Optional[float] = 2.0,
) -> pd.DataFrame:
    """Compute separate home/away Elo per team."""
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    # home_elo[team], away_elo[team]
    team_home_elo: Dict[str, float] = {}
    team_away_elo: Dict[str, float] = {}
    elo_probs = []
    elo_diffs = []

    current_season: Optional[int] = None

    for _, row in out.iterrows():
        season = row["season"]
        if current_season is None or season != current_season:
            team_home_elo = {}
            team_away_elo = {}
            current_season = season

        home = row["home_team"]
        away = row["away_team"]
        result = row.get("result", 0)
        row.get("home_score", 0)
        row.get("away_score", 0)

        h_elo = team_home_elo.get(home, default_elo)
        a_elo = team_away_elo.get(away, default_elo)

        expected_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - home_advantage) / 400.0))
        expected_away = 1.0 - expected_home

        elo_probs.append(expected_home)
        elo_diffs.append(h_elo - a_elo)

        # Update Elo after game
        if not pd.isna(result):
            home_won = 1 if result > 0 else (0 if result < 0 else 0.5)
            away_won = 1.0 - home_won

            mov_mult = _mov_mult(abs(result), mov_type, mov_scale, mov_cap)
            k_home = k_factor * mov_mult
            k_away = k_factor * mov_mult

            team_home_elo[home] = h_elo + k_home * (home_won - expected_home)
            team_away_elo[away] = a_elo + k_away * (away_won - expected_away)

    out["elo_prob"] = elo_probs
    out["elo_diff"] = elo_diffs
    return out
