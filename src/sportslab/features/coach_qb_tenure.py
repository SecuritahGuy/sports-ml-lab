"""Coach-QB tenure features — how long the QB-coach pair has worked together."""

import numpy as np
import pandas as pd

COACH_QB_TENURE_COLUMNS = [
    "home_coach_qb_games",
    "away_coach_qb_games",
    "coach_qb_games_diff",
    "home_coach_qb_log_games",
    "away_coach_qb_log_games",
]


def compute_coach_qb_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add coach-QB tenure features.

    Tracks how many games the current QB-coach pair has worked together
    using a single chronological pass.

    Args:
        df: Must contain columns: season, week, gameday, home_team, away_team,
            home_qb_id, away_qb_id, home_coach, away_coach.

    Returns:
        DataFrame with added COACH_QB_TENURE_COLUMNS.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    season_pairs: dict[int, dict[str, int]] = {}

    features = {
        "home_coach_qb_games": [],
        "away_coach_qb_games": [],
        "home_coach_qb_log_games": [],
        "away_coach_qb_log_games": [],
    }

    for _, row in out.iterrows():
        for side, team_col, qb_col, coach_col in [
            ("home", "home_team", "home_qb_id", "home_coach"),
            ("away", "away_team", "away_qb_id", "away_coach"),
        ]:
            qb_id = row.get(qb_col)
            coach = row.get(coach_col)
            season = row["season"]

            if pd.isna(qb_id) or pd.isna(coach) or str(qb_id).strip() == "" or str(coach).strip() == "":
                features[f"{side}_coach_qb_games"].append(0)
                features[f"{side}_coach_qb_log_games"].append(0.0)
                continue

            pair_key = f"{qb_id}|{coach}"

            if season not in season_pairs:
                season_pairs[season] = {}

            games = season_pairs[season].get(pair_key, 0)
            features[f"{side}_coach_qb_games"].append(games)
            features[f"{side}_coach_qb_log_games"].append(np.log1p(games))

            # Post-game update
            season_pairs[season][pair_key] = games + 1

    for col_name, values in features.items():
        out[col_name] = np.array(values, dtype=np.float64)

    out["coach_qb_games_diff"] = out["home_coach_qb_games"] - out["away_coach_qb_games"]

    return out
