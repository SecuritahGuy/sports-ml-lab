"""Rolling pregame features for NFL teams — no leakage, chronologically safe."""

import numpy as np
import pandas as pd

ROLLING_WINDOW = 5


def compute_rolling_features(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Add rolling pregame features computed from games BEFORE each game.

    For every game, each team's stats are computed from its *previous* games
    only (within the rolling window).  The current game's result is never
    included.  Rolling stats carry over between seasons.

    Args:
        df: DataFrame with columns season, week, gameday, home_team, away_team,
            home_score, away_score, home_win (0, 1, or NA for ties).
        window: Number of most-recent games to include.

    Returns:
        DataFrame with added columns:
            home_rolling_win_pct, away_rolling_win_pct, rolling_win_pct_diff,
            home_rolling_point_diff, away_rolling_point_diff,
            rolling_point_diff_diff, home_rolling_points_for,
            away_rolling_points_for, home_rolling_points_against,
            away_rolling_points_against.
    """
    out = df.copy()
    sorted_df = out.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    # Build per-team game history (each game adds two entries: home + away)
    all_games: list[dict] = []
    for idx, row in sorted_df.iterrows():
        hw = row["home_win"]
        for side, team, pf, pa in [
            ("home", row["home_team"], row["home_score"], row["away_score"]),
            ("away", row["away_team"], row["away_score"], row["home_score"]),
        ]:
            if pd.isna(hw):
                won = False  # tie → not a win for either side
            elif side == "home":
                won = bool(hw == 1)
            else:
                won = bool(hw == 0)

            all_games.append(
                {
                    "team": team,
                    "season": row["season"],
                    "week": row["week"],
                    "gameday": row["gameday"],
                    "points_for": float(pf),
                    "points_against": float(pa),
                    "won": won,
                    "game_index": idx,
                }
            )

    # Group by team, sorted chronologically
    team_games: dict[str, list[dict]] = {}
    for g in all_games:
        team_games.setdefault(g["team"], []).append(g)
    for team in team_games:
        team_games[team].sort(key=lambda x: (x["season"], x["week"], str(x["gameday"])))

    def _rolling(team_name: str, current_idx: int) -> dict:
        """Compute rolling stats for team_name from games before current_idx."""
        games = team_games.get(team_name, [])
        prev = [g for g in games if g["game_index"] < current_idx][-window:]
        n = len(prev)

        if n == 0:
            return {"win_pct": 0.5, "pt_diff": 0.0, "pts_for": 0.0, "pts_against": 0.0}

        wins = sum(1 for g in prev if g["won"])
        pf_vals = [g["points_for"] for g in prev]
        pa_vals = [g["points_against"] for g in prev]

        return {
            "win_pct": wins / n,
            "pt_diff": np.mean([g["points_for"] - g["points_against"] for g in prev]),
            "pts_for": np.mean(pf_vals),
            "pts_against": np.mean(pa_vals),
        }

    h_wp, a_wp = [], []
    h_pd, a_pd = [], []
    h_pf, a_pf = [], []
    h_pa, a_pa = [], []

    for idx, row in sorted_df.iterrows():
        h = _rolling(row["home_team"], idx)
        a = _rolling(row["away_team"], idx)
        h_wp.append(h["win_pct"])
        a_wp.append(a["win_pct"])
        h_pd.append(h["pt_diff"])
        a_pd.append(a["pt_diff"])
        h_pf.append(h["pts_for"])
        a_pf.append(a["pts_for"])
        h_pa.append(h["pts_against"])
        a_pa.append(a["pts_against"])

    arr = lambda x: np.array(x)  # noqa: E731
    out["home_rolling_win_pct"] = h_wp
    out["away_rolling_win_pct"] = a_wp
    out["rolling_win_pct_diff"] = arr(h_wp) - arr(a_wp)
    out["home_rolling_point_diff"] = h_pd
    out["away_rolling_point_diff"] = a_pd
    out["rolling_point_diff_diff"] = arr(h_pd) - arr(a_pd)
    out["home_rolling_points_for"] = h_pf
    out["away_rolling_points_for"] = a_pf
    out["home_rolling_points_against"] = h_pa
    out["away_rolling_points_against"] = a_pa

    return out
