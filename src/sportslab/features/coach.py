"""Pregame coach tenure features — experience, record, continuity — computed chronologically."""

from typing import Dict, List

import pandas as pd

COACH_FEATURE_COLUMNS = [
    "home_coach_tenure",
    "away_coach_tenure",
    "home_coach_career_wins",
    "away_coach_career_wins",
    "home_coach_career_games",
    "away_coach_career_games",
    "home_coach_win_pct",
    "away_coach_win_pct",
]


def compute_coach_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pregame coach experience features computed chronologically.

    Requires columns: season, week, gameday, home_team, away_team,
    home_coach, away_coach, home_win.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    _coach_state: Dict[str, dict] = {}

    features: Dict[str, List] = {
        "home_coach_tenure": [],
        "away_coach_tenure": [],
        "home_coach_career_wins": [],
        "away_coach_career_wins": [],
        "home_coach_career_games": [],
        "away_coach_career_games": [],
        "home_coach_win_pct": [],
        "away_coach_win_pct": [],
    }

    for _, row in out.iterrows():
        row["season"]
        home = row["home_team"]
        away = row["away_team"]
        home_coach = row.get("home_coach")
        away_coach = row.get("away_coach")
        home_win = row.get("home_win")

        for team, coach, side in [
            (home, home_coach, "home"),
            (away, away_coach, "away"),
        ]:
            coach_str = str(coach).strip() if not pd.isna(coach) else ""
            if coach_str and coach_str != "nan":
                if coach_str not in _coach_state:
                    _coach_state[coach_str] = {
                        "games": 0,
                        "wins": 0,
                        "current_team": team,
                        "team_tenure": 0,
                    }
                cs = _coach_state[coach_str]
                # If coach switched teams, reset tenure
                if cs["current_team"] != team:
                    cs["current_team"] = team
                    cs["team_tenure"] = 0

                features[f"{side}_coach_tenure"].append(cs["team_tenure"])
                features[f"{side}_coach_career_wins"].append(cs["wins"])
                features[f"{side}_coach_career_games"].append(cs["games"])
                features[f"{side}_coach_win_pct"].append(
                    cs["wins"] / cs["games"] if cs["games"] > 0 else 0.5
                )
            else:
                features[f"{side}_coach_tenure"].append(0)
                features[f"{side}_coach_career_wins"].append(0)
                features[f"{side}_coach_career_games"].append(0)
                features[f"{side}_coach_win_pct"].append(0.5)

        # Post-game state update
        for team, coach, is_home in [
            (home, home_coach, True),
            (away, away_coach, False),
        ]:
            coach_str = str(coach).strip() if not pd.isna(coach) else ""
            if coach_str and coach_str != "nan":
                if coach_str not in _coach_state:
                    _coach_state[coach_str] = {
                        "games": 0,
                        "wins": 0,
                        "current_team": team,
                        "team_tenure": 0,
                    }
                cs = _coach_state[coach_str]
                if cs["current_team"] == team:
                    cs["team_tenure"] += 1
                else:
                    cs["current_team"] = team
                    cs["team_tenure"] = 1
                cs["games"] += 1
                if not pd.isna(home_win):
                    won = bool(home_win == 1) if is_home else bool(home_win == 0)
                    if won:
                        cs["wins"] += 1

    for col, vals in features.items():
        out[col] = vals

    return out
