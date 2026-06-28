"""Continuous QB change magnitude features using rolling EPA.

Computes per-QB rolling passing_epa averages and engineers magnitude
features that quantify how much QB quality drops when a change occurs.
"""

import numpy as np
import pandas as pd

import nflreadpy as nfl

QB_MAGNITUDE_COLUMNS = [
    "home_qb_rolling_epa",
    "away_qb_rolling_epa",
    "qb_epa_diff",
    "home_qb_change_magnitude",
    "away_qb_change_magnitude",
    "qb_change_magnitude_diff",
    "home_qb_change_magnitude_signed",
    "away_qb_change_magnitude_signed",
    "home_qb_rolling_epa_missing",
    "away_qb_rolling_epa_missing",
]


def _build_qb_rolling_epa(seasons: list[int]) -> pd.DataFrame:
    """Build per-QB rolling 5-game avg of passing_epa across given seasons.

    Returns a DataFrame keyed by (player_id, season, week) with a
    pre-game rolling_epa column (games before the current week only).
    """
    ps = nfl.load_player_stats(seasons=seasons).to_pandas()
    qb = ps[ps["position"] == "QB"].copy()
    qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    qb["passing_epa"] = qb["passing_epa"].fillna(0.0)

    roll = (
        qb.groupby("player_id")["passing_epa"]
        .apply(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    qb["rolling_epa"] = roll

    return qb[["player_id", "season", "week", "team", "rolling_epa"]]


def compute_qb_magnitude_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add continuous QB change magnitude features to the feature table.

    Args:
        df: Must contain columns: season, week, gameday, home_team,
            away_team, home_qb_id, away_qb_id, home_qb_changed,
            away_qb_changed.

    Returns:
        DataFrame with added QB_MAGNITUDE_COLUMNS columns.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    seasons = sorted(out["season"].unique())
    qb_roll = _build_qb_rolling_epa(seasons)

    epa_lookup = qb_roll.set_index(["player_id", "season", "week"])["rolling_epa"]

    team_state: dict[str, dict] = {}

    features = {
        "home_qb_rolling_epa": [],
        "away_qb_rolling_epa": [],
        "qb_epa_diff": [],
        "home_qb_change_magnitude": [],
        "away_qb_change_magnitude": [],
        "qb_change_magnitude_diff": [],
        "home_qb_change_magnitude_signed": [],
        "away_qb_change_magnitude_signed": [],
        "home_qb_rolling_epa_missing": [],
        "away_qb_rolling_epa_missing": [],
    }

    for idx, row in out.iterrows():
        for side, team_col, qb_col, changed_col in [
            ("home", "home_team", "home_qb_id", "home_qb_changed"),
            ("away", "away_team", "away_qb_id", "away_qb_changed"),
        ]:
            team = row[team_col]
            qb_id = row.get(qb_col)
            season = row["season"]
            week = int(row["week"])
            changed = row.get(changed_col, 0)

            if not team:
                team = f"__unknown_{side}__"
            if team not in team_state:
                team_state[team] = {"current_season": season, "last_qb_id": None, "last_epa": None}

            state = team_state[team]
            if season != state.get("current_season"):
                state["last_qb_id"] = None
                state["last_epa"] = None
                state["current_season"] = season

            qb_missing = pd.isna(qb_id) or qb_id is None or str(qb_id).strip() == ""
            if qb_missing:
                cur_epa = np.nan
            else:
                try:
                    cur_epa = float(epa_lookup.loc[(qb_id, season, week)])
                except (KeyError, TypeError):
                    cur_epa = np.nan

            magnitude = 0.0
            signed_mag = 0.0
            if changed and not qb_missing and state["last_epa"] is not None and not np.isnan(cur_epa):
                prev_epa = state["last_epa"]
                if not np.isnan(prev_epa):
                    magnitude = float(abs(prev_epa - cur_epa))
                    signed_mag = float(prev_epa - cur_epa)

            features[f"{side}_qb_rolling_epa"].append(cur_epa if not np.isnan(cur_epa) else 0.0)
            features[f"{side}_qb_change_magnitude"].append(magnitude)
            features[f"{side}_qb_change_magnitude_signed"].append(signed_mag)
            features[f"{side}_qb_rolling_epa_missing"].append(1 if qb_missing or np.isnan(cur_epa) else 0)

            state["last_qb_id"] = qb_id
            if not np.isnan(cur_epa if not qb_missing else np.nan):
                state["last_epa"] = cur_epa

    features["qb_epa_diff"] = np.array(features["home_qb_rolling_epa"]) - np.array(
        features["away_qb_rolling_epa"]
    )
    features["qb_change_magnitude_diff"] = np.array(features["home_qb_change_magnitude"]) - np.array(
        features["away_qb_change_magnitude"]
    )

    for col_name, values in features.items():
        out[col_name] = np.array(values, dtype=np.float64)

    return out
