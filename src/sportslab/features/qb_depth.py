"""QB depth features — backup/rust/first-season-start / experience depth.

Computed chronologically with no leakage. Adds features beyond what
`compute_qb_features()` provides: rust games (games since this QB's
last start for their team) and first-season-start flags.
"""

import pandas as pd

QB_DEPTH_COLUMNS = [
    "home_qb_rust_games",
    "away_qb_rust_games",
    "home_qb_first_season_start",
    "away_qb_first_season_start",
    "qb_rust_diff",
    "qb_first_season_start_diff",
]


def compute_qb_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add QB depth features from a chronological pass.

    Requires columns: season, week, gameday, home_team, away_team,
    home_qb_id, away_qb_id.  Must be called AFTER compute_qb_features
    so that qb_changed / starts features are available.

    Args:
        df: DataFrame sorted chronologically with QB identity columns.

    Returns:
        DataFrame with added QB_DEPTH_COLUMNS.
    """
    out = df.copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    team_state: dict[str, dict] = {}

    def _ensure(team: str, season: int) -> dict:
        if team not in team_state or team_state[team]["season"] != season:
            team_state[team] = {
                "season": season,
                "last_qb_id": None,
                "qb_last_seen": {},
                "total_games": 0,
            }
        return team_state[team]

    home_rust: list[int] = []
    away_rust: list[int] = []
    home_first: list[int] = []
    away_first: list[int] = []

    for _, row in out.iterrows():
        season = int(row["season"])
        for side, team_col, qb_col in [
            ("home", "home_team", "home_qb_id"),
            ("away", "away_team", "away_qb_id"),
        ]:
            team = row[team_col]
            qb_id = row.get(qb_col)
            state = _ensure(team, season)

            qb_missing = pd.isna(qb_id) or qb_id is None or str(qb_id).strip() == ""

            if qb_missing:
                if side == "home":
                    home_rust.append(0)
                    home_first.append(0)
                else:
                    away_rust.append(0)
                    away_first.append(0)
                continue

            qb_id = str(qb_id)

            changed = (
                1
                if (state["last_qb_id"] is not None and state["last_qb_id"] != qb_id)
                else 0
            )

            # Rust: games since this QB's last start for this team
            if changed:
                last_seen = state["qb_last_seen"].get(qb_id)
                if last_seen is not None:
                    rust = state["total_games"] - last_seen - 1
                else:
                    rust = 999
            else:
                rust = 0

            # First season start: starts_this_season_pre == 0
            # AND this QB has started for team before (team_starts_pre > 0)
            starts_col = f"{side}_qb_starts_this_season_pre"
            team_starts_col = f"{side}_qb_team_starts_pre"
            starts = row.get(starts_col, 0)
            team_starts = row.get(team_starts_col, 0)
            first_ss = 1 if (changed and starts == 0 and team_starts > 0) else 0

            if side == "home":
                home_rust.append(rust)
                home_first.append(first_ss)
            else:
                away_rust.append(rust)
                away_first.append(first_ss)

            # Post-game state update
            state["qb_last_seen"][qb_id] = state["total_games"]
            state["last_qb_id"] = qb_id
            state["total_games"] += 1

    out["home_qb_rust_games"] = home_rust
    out["away_qb_rust_games"] = away_rust
    out["home_qb_first_season_start"] = home_first
    out["away_qb_first_season_start"] = away_first
    out["qb_rust_diff"] = (
        pd.Series(home_rust, dtype=float) - pd.Series(away_rust, dtype=float)
    ).values
    out["qb_first_season_start_diff"] = (
        pd.Series(home_first, dtype=float) - pd.Series(away_first, dtype=float)
    ).values

    return out
