"""Pregame situational features — rolling MOV, streaks, altitude, turf,
primetime, rest squared, YTD win percentage — all computed
chronologically with no leakage."""

from typing import Dict, List

import numpy as np
import pandas as pd

HIGH_ALTITUDE_STADIUMS = {
    "Empower Field at Mile High",  # Denver, 5280 ft
    "Azteca Stadium",  # Mexico City, 7200 ft
}

SITUATIONAL_FEATURE_COLUMNS = [
    "home_rolling_mov_3",
    "away_rolling_mov_3",
    "home_rolling_mov_5",
    "away_rolling_mov_5",
    "home_rolling_pts_for",
    "away_rolling_pts_for",
    "home_rolling_pts_against",
    "away_rolling_pts_against",
    "home_win_streak",
    "away_win_streak",
    "home_ytd_win_pct",
    "away_ytd_win_pct",
    "turf_flag",
    "high_altitude_flag",
    "prime_time_flag",
    "rest_diff_squared",
]


def _is_turf(surface: str) -> int:
    s = str(surface).strip().lower()
    return 0 if s in ("", "grass") else 1


def _is_prime_time(gametime, weekday) -> int:
    gt = str(gametime).strip()
    wd = str(weekday).strip()
    try:
        hour = int(gt.split(":")[0])
    except (ValueError, IndexError):
        return 0
    if wd in ("Monday", "Thursday"):
        return 1
    if wd == "Sunday" and hour >= 20:
        return 1
    if hour >= 20:
        return 1
    return 0


def compute_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pregame situational features computed chronologically.

    Requires columns: season, week, gameday, home_team, away_team,
    home_score, away_score, result, home_win, surface, stadium, gametime,
    weekday, home_rest, away_rest, rest_diff.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    # Static features
    out["turf_flag"] = out["surface"].apply(_is_turf)
    out["high_altitude_flag"] = out["stadium"].isin(HIGH_ALTITUDE_STADIUMS).astype(int)
    out["prime_time_flag"] = out.apply(
        lambda r: _is_prime_time(r.get("gametime"), r.get("weekday")), axis=1
    )
    out["rest_diff_squared"] = (out["rest_diff"] ** 2).fillna(0).astype(float)

    # Chronological team-level features
    _team_state: Dict[str, dict] = {}
    _keys = [
        "season",
        "last_movs_3",
        "last_movs_5",
        "pts_for_season",
        "pts_against_season",
        "games_played",
        "win_streak",
        "wins",
        "losses",
    ]

    def _reset_state(team: str, season: int) -> dict:
        _team_state[team] = {
            "season": season,
            "last_movs_3": [],
            "last_movs_5": [],
            "pts_for_season": 0.0,
            "pts_against_season": 0.0,
            "games_played": 0,
            "win_streak": 0,
            "wins": 0,
            "losses": 0,
        }
        return _team_state[team]

    def _get_state(team: str, season: int) -> dict:
        if team not in _team_state or _team_state[team]["season"] != season:
            return _reset_state(team, season)
        return _team_state[team]

    features: Dict[str, List] = {
        "home_rolling_mov_3": [],
        "away_rolling_mov_3": [],
        "home_rolling_mov_5": [],
        "away_rolling_mov_5": [],
        "home_rolling_pts_for": [],
        "away_rolling_pts_for": [],
        "home_rolling_pts_against": [],
        "away_rolling_pts_against": [],
        "home_win_streak": [],
        "away_win_streak": [],
        "home_ytd_win_pct": [],
        "away_ytd_win_pct": [],
    }

    for _, row in out.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]
        home_score = row.get("home_score", 0)
        away_score = row.get("away_score", 0)
        result = row.get("result", 0)
        home_win = row.get("home_win")

        for team, side in [(home, "home"), (away, "away")]:
            state = _get_state(team, season)

            # Rolling MOV
            mov_3 = float(np.mean(state["last_movs_3"])) if state["last_movs_3"] else 0.0
            mov_5 = float(np.mean(state["last_movs_5"])) if state["last_movs_5"] else 0.0
            features[f"{side}_rolling_mov_3"].append(mov_3)
            features[f"{side}_rolling_mov_5"].append(mov_5)

            # Rolling season averages
            gp = max(state["games_played"], 1)
            features[f"{side}_rolling_pts_for"].append(state["pts_for_season"] / gp)
            features[f"{side}_rolling_pts_against"].append(state["pts_against_season"] / gp)

            # Win streak
            features[f"{side}_win_streak"].append(state["win_streak"])

            # YTD win %
            ytd = state["wins"] / gp if state["games_played"] > 0 else 0.5
            features[f"{side}_ytd_win_pct"].append(ytd)

        # ── Post-game state update ──
        is_tie = pd.isna(home_win)
        for team, is_home in [(home, True), (away, False)]:
            state = _get_state(team, season)
            scored = home_score if is_home else away_score
            allowed = away_score if is_home else home_score
            team_mov = result if is_home else -result
            if is_tie:
                won = False
            else:
                won = bool(home_win == 1) if is_home else bool(home_win == 0)

            # Update rolling MOV windows (all games including ties)
            state["last_movs_3"].append(team_mov)
            if len(state["last_movs_3"]) > 3:
                state["last_movs_3"].pop(0)
            state["last_movs_5"].append(team_mov)
            if len(state["last_movs_5"]) > 5:
                state["last_movs_5"].pop(0)

            state["pts_for_season"] += scored
            state["pts_against_season"] += allowed
            state["games_played"] += 1

            if is_tie:
                state["win_streak"] = 0
            elif won:
                state["wins"] += 1
                state["win_streak"] = max(state["win_streak"], 0) + 1
            else:
                state["losses"] += 1
                state["win_streak"] = (
                    state["win_streak"] - 1 if state["win_streak"] <= 0 else -1
                )

    # Apply features
    for col, vals in features.items():
        out[col] = vals

    return out
