"""Pregame injury report features — no leakage, chronological.

Uses nflreadpy injury data to create pregame team-level injury counts
and injury-driven QB change detection.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None

# Position groups for team-level injury counts
OL_POSITIONS = {"C", "G", "T"}
SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}
DEFENSE_POSITIONS = {"DE", "DT", "LB", "S", "CB", "NT", "EDGE", "ILB", "OLB", "MLB", "SS", "FS"}

INJURY_FEATURE_COLUMNS = [
    "home_qb_out",
    "away_qb_out",
    "home_qb_doubtful_or_out",
    "away_qb_doubtful_or_out",
    "home_total_out",
    "away_total_out",
    "home_total_doubtful_or_out",
    "away_total_doubtful_or_out",
    "home_skill_out",
    "away_skill_out",
    "home_ol_out",
    "away_ol_out",
    "home_def_out",
    "away_def_out",
    "any_qb_out",
    "net_injuries",
    "net_skill_out",
    "net_def_out",
    "home_qb_injury_change",
    "away_qb_injury_change",
]


def _load_injury_data(
    seasons: Optional[list] = None,
) -> pd.DataFrame:
    """Load injury data from nflreadpy and flatten to pregame snapshot.

    Returns one row per (season, week, team, player) — the latest
    report entry for that player-week.
    """
    if nfl is None:
        raise ImportError("nflreadpy is required")

    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025]

    injuries = nfl.load_injuries(seasons=seasons).to_pandas()

    # Drop rows with empty position strings
    injuries = injuries[injuries["position"].notna() & (injuries["position"].str.strip() != "")]

    return injuries


def _build_injury_summary(injuries: pd.DataFrame) -> pd.DataFrame:
    """Build team-week injury summary from raw injury data.

    Returns a DataFrame with team-week injury counts by position group.
    """
    out_rows = []
    for (season, week, team), group in injuries.groupby(["season", "week", "team"]):
        row = {"season": season, "week": week, "team": team}
        positions = group["position"].values
        statuses = group["report_status"].values

        def _count(positions_set):
            mask = np.isin(positions, list(positions_set))
            out_mask = (statuses == "Out") & mask
            out_do_mask = (np.isin(statuses, ["Out", "Doubtful"])) & mask
            return int(out_mask.sum()), int(out_do_mask.sum())

        row["qb_out"], row["qb_doubtful_or_out"] = _count({"QB"})
        row["total_out"], row["total_doubtful_or_out"] = _count(set(positions))
        row["skill_out"], _ = _count(SKILL_POSITIONS)
        row["ol_out"], _ = _count(OL_POSITIONS)
        row["def_out"], _ = _count(DEFENSE_POSITIONS)
        out_rows.append(row)

    summary = pd.DataFrame(out_rows)
    return summary


def compute_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pregame injury features to a game-level DataFrame.

    Requires columns: season, week, home_team, away_team, home_qb_id,
    away_qb_id, home_qb_name, away_qb_name, gameday.

    Adds INJURY_FEATURE_COLUMNS plus provides week-level injury counts
    and injury-driven QB change detection.
    """
    injuries = _load_injury_data()
    summary = _build_injury_summary(injuries)

    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    # Merge injury summary for home and away teams
    home_inj = summary.rename(
        columns={
            "team": "home_team",
            "qb_out": "home_qb_out",
            "qb_doubtful_or_out": "home_qb_doubtful_or_out",
            "total_out": "home_total_out",
            "total_doubtful_or_out": "home_total_doubtful_or_out",
            "skill_out": "home_skill_out",
            "ol_out": "home_ol_out",
            "def_out": "home_def_out",
        }
    )
    away_inj = summary.rename(
        columns={
            "team": "away_team",
            "qb_out": "away_qb_out",
            "qb_doubtful_or_out": "away_qb_doubtful_or_out",
            "total_out": "away_total_out",
            "total_doubtful_or_out": "away_total_doubtful_or_out",
            "skill_out": "away_skill_out",
            "ol_out": "away_ol_out",
            "def_out": "away_def_out",
        }
    )

    out = out.merge(home_inj, on=["season", "week", "home_team"], how="left")
    out = out.merge(away_inj, on=["season", "week", "away_team"], how="left")

    # Fill NaN (teams with no injury report entries that week)
    fill_cols = [
        "home_qb_out",
        "away_qb_out",
        "home_qb_doubtful_or_out",
        "away_qb_doubtful_or_out",
        "home_total_out",
        "away_total_out",
        "home_total_doubtful_or_out",
        "away_total_doubtful_or_out",
        "home_skill_out",
        "away_skill_out",
        "home_ol_out",
        "away_ol_out",
        "home_def_out",
        "away_def_out",
    ]
    for col in fill_cols:
        out[col] = out[col].fillna(0).astype(int)

    # Composite features
    out["any_qb_out"] = ((out["home_qb_out"] > 0) | (out["away_qb_out"] > 0)).astype(int)
    out["net_injuries"] = out["home_total_out"] - out["away_total_out"]
    out["net_skill_out"] = out["home_skill_out"] - out["away_skill_out"]
    out["net_def_out"] = out["home_def_out"] - out["away_def_out"]

    # Injury-driven QB change detection
    # Track previous QB per team chronologically
    _team_state: Dict[str, dict] = {}
    _season_keys = ["last_qb_id", "last_qb_name"]

    def _ensure_team(team: str, season: int) -> dict:
        if team not in _team_state:
            _team_state[team] = {"current_season": season}
            for k in _season_keys:
                _team_state[team][k] = None
        state = _team_state[team]
        if season != state["current_season"]:
            for k in _season_keys:
                state[k] = None
            state["current_season"] = season
        return state

    home_injury_changes = []
    away_injury_changes = []

    for _, row in out.iterrows():
        season = row["season"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_qb = row.get("home_qb_name")
        away_qb = row.get("away_qb_name")
        home_qb_id = row.get("home_qb_id")
        away_qb_id = row.get("away_qb_id")

        for side, team, qb_name, qb_id, change_list in [
            ("home", home_team, home_qb, home_qb_id, home_injury_changes),
            ("away", away_team, away_qb, away_qb_id, away_injury_changes),
        ]:
            state = _ensure_team(team, season)
            last_qb_id = state.get("last_qb_id")
            last_qb_name = state.get("last_qb_name")

            qb_missing = pd.isna(qb_name) or str(qb_name).strip() == ""
            last_missing = pd.isna(last_qb_name) or str(last_qb_name).strip() == ""

            if qb_missing or last_missing or last_qb_id is None:
                change_list.append(0)
            elif last_qb_id != qb_id:
                # QB changed — check if old QB was OUT this week
                _out = injuries
                if not isinstance(injuries.index, pd.RangeIndex):
                    pass
                # Check if old QB was OUT this week for this team
                old_qb_match = (
                    (injuries["season"] == season)
                    & (injuries["week"] == row["week"])
                    & (injuries["team"] == team)
                    & (injuries["full_name"] == last_qb_name)
                    & (injuries["report_status"] == "Out")
                )
                was_out = int(old_qb_match.any())
                change_list.append(was_out)
            else:
                change_list.append(0)

        # Update state post-game
        for team, qb_name, qb_id in [
            (home_team, home_qb, home_qb_id),
            (away_team, away_qb, away_qb_id),
        ]:
            qb_missing = pd.isna(qb_name) or str(qb_name).strip() == ""
            if not qb_missing:
                state = _ensure_team(team, season)
                state["last_qb_id"] = qb_id
                state["last_qb_name"] = qb_name

    out["home_qb_injury_change"] = home_injury_changes
    out["away_qb_injury_change"] = away_injury_changes

    return out
