"""Pregame QB starter/change features — no leakage, chronological."""

import numpy as np
import pandas as pd


def compute_qb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pregame QB features computed from games BEFORE each game.

    Tracks QB identity, changes, starts, win records, and continuity
    per team using a single chronological pass.

    Args:
        df: Must contain columns: season, week, gameday, home_team, away_team,
            home_qb_id, away_qb_id, home_win.

    Returns:
        DataFrame with added QB feature columns.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    # State per team
    _reset_keys = [
        "last_qb_id",
        "qb_starts_this_season",
        "qb_wins_this_season",
        "qb_games_this_season",
        "games_since_change",
    ]

    team_state: dict[str, dict] = {}

    def _ensure_team(team: str, season: int) -> dict:
        if team not in team_state:
            team_state[team] = {
                "current_season": season,
                "last_qb_id": None,
                "qb_starts_this_season": {},
                "qb_team_starts": {},
                "qb_wins_this_season": {},
                "qb_games_this_season": {},
                "games_since_change": 0,
            }
        state = team_state[team]
        if season != state["current_season"]:
            for k in _reset_keys:
                is_dict = isinstance(state.get(k), dict)
                state[k] = {} if is_dict else (None if k == "last_qb_id" else 0)
            state["current_season"] = season
        return state

    # Pre-allocate feature lists
    features = {
        "home_qb_changed": [],
        "away_qb_changed": [],
        "home_qb_starts_this_season_pre": [],
        "away_qb_starts_this_season_pre": [],
        "home_qb_team_starts_pre": [],
        "away_qb_team_starts_pre": [],
        "home_qb_win_pct_pre": [],
        "away_qb_win_pct_pre": [],
        "home_games_since_qb_change": [],
        "away_games_since_qb_change": [],
        "home_new_qb_flag": [],
        "away_new_qb_flag": [],
        "home_qb_missing_flag": [],
        "away_qb_missing_flag": [],
    }

    for idx, row in out.iterrows():
        for side, team_col, qb_col in [
            ("home", "home_team", "home_qb_id"),
            ("away", "away_team", "away_qb_id"),
        ]:
            team = row[team_col]
            qb_id = row.get(qb_col)
            season = row["season"]
            state = _ensure_team(team, season)

            qb_missing = pd.isna(qb_id) or qb_id is None or str(qb_id).strip() == ""
            if qb_missing:
                features[f"{side}_qb_changed"].append(0)
                features[f"{side}_qb_starts_this_season_pre"].append(0)
                features[f"{side}_qb_team_starts_pre"].append(0)
                features[f"{side}_qb_win_pct_pre"].append(0.5)
                features[f"{side}_games_since_qb_change"].append(0)
                features[f"{side}_new_qb_flag"].append(0)
                features[f"{side}_qb_missing_flag"].append(1)
                continue

            changed = 1 if (state["last_qb_id"] is not None and state["last_qb_id"] != qb_id) else 0
            starts_this_season = state["qb_starts_this_season"].get(qb_id, 0)
            team_starts = state["qb_team_starts"].get(qb_id, 0)
            wins_this = state["qb_wins_this_season"].get(qb_id, 0)
            games_this = state["qb_games_this_season"].get(qb_id, 0)
            win_pct = wins_this / games_this if games_this > 0 else 0.5
            games_since = 0 if changed else state["games_since_change"]
            new_qb = 1 if team_starts == 0 else 0

            features[f"{side}_qb_changed"].append(changed)
            features[f"{side}_qb_starts_this_season_pre"].append(starts_this_season)
            features[f"{side}_qb_team_starts_pre"].append(team_starts)
            features[f"{side}_qb_win_pct_pre"].append(win_pct)
            features[f"{side}_games_since_qb_change"].append(games_since)
            features[f"{side}_new_qb_flag"].append(new_qb)
            features[f"{side}_qb_missing_flag"].append(0)

        # Post-game state update
        for team_col, qb_col, is_home in [
            ("home_team", "home_qb_id", True),
            ("away_team", "away_qb_id", False),
        ]:
            team = row[team_col]
            qb_id = row.get(qb_col)
            season = row["season"]
            qb_missing = pd.isna(qb_id) or qb_id is None or str(qb_id).strip() == ""
            if qb_missing:
                continue

            state = _ensure_team(team, season)

            home_won = row.get("home_win")
            if pd.isna(home_won):
                qb_won = False
            else:
                qb_won = bool(home_won == 1) if is_home else bool(home_won == 0)

            state["qb_starts_this_season"][qb_id] = state["qb_starts_this_season"].get(qb_id, 0) + 1
            state["qb_team_starts"][qb_id] = state["qb_team_starts"].get(qb_id, 0) + 1
            state["qb_games_this_season"][qb_id] = state["qb_games_this_season"].get(qb_id, 0) + 1
            if qb_won:
                state["qb_wins_this_season"][qb_id] = state["qb_wins_this_season"].get(qb_id, 0) + 1

            if state["last_qb_id"] is not None and state["last_qb_id"] != qb_id:
                state["games_since_change"] = 1
            else:
                state["games_since_change"] += 1
            state["last_qb_id"] = qb_id

    # Write features
    for col_name, values in features.items():
        out[col_name] = values

    # Diff features
    out["qb_change_diff"] = np.array(features["home_qb_changed"]) - np.array(
        features["away_qb_changed"]
    )
    out["qb_starts_diff"] = np.array(features["home_qb_starts_this_season_pre"]) - np.array(
        features["away_qb_starts_this_season_pre"]
    )
    out["qb_win_pct_diff"] = np.array(features["home_qb_win_pct_pre"]) - np.array(
        features["away_qb_win_pct_pre"]
    )
    out["games_since_qb_change_diff"] = np.array(features["home_games_since_qb_change"]) - np.array(
        features["away_games_since_qb_change"]
    )
    out["new_qb_diff"] = np.array(features["home_new_qb_flag"]) - np.array(
        features["away_new_qb_flag"]
    )

    return out


QB_FEATURE_COLUMNS = [
    "home_qb_changed",
    "away_qb_changed",
    "qb_change_diff",
    "home_qb_starts_this_season_pre",
    "away_qb_starts_this_season_pre",
    "qb_starts_diff",
    "home_qb_team_starts_pre",
    "away_qb_team_starts_pre",
    "home_qb_win_pct_pre",
    "away_qb_win_pct_pre",
    "qb_win_pct_diff",
    "home_games_since_qb_change",
    "away_games_since_qb_change",
    "games_since_qb_change_diff",
    "home_new_qb_flag",
    "away_new_qb_flag",
    "new_qb_diff",
    "home_qb_missing_flag",
    "away_qb_missing_flag",
]


QB_IDENTITY_COLUMNS = [
    "home_qb_id",
    "away_qb_id",
]
