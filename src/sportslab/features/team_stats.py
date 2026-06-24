"""Pregame rolling team-stat features from nflreadpy weekly player stats.

Aggregates player-level stats to team-level offensive and defensive metrics,
then computes pregame rolling averages (shifted by 1 game).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SPORTSLAB_MIN_SEASON = 2021

TEAM_STATS_COLUMNS: list[str] = [
    "home_off_yds_rolling_3",
    "away_off_yds_rolling_3",
    "home_def_yds_allowed_rolling_3",
    "away_def_yds_allowed_rolling_3",
    "home_off_yds_rolling_5",
    "away_off_yds_rolling_5",
    "home_def_yds_allowed_rolling_5",
    "away_def_yds_allowed_rolling_5",
    "off_yds_net_3",
    "off_yds_net_5",
    "home_fantasy_pts_rolling_3",
    "away_fantasy_pts_rolling_3",
    "home_def_sacks_rolling_3",
    "away_def_sacks_rolling_3",
    "home_fantasy_pts_rolling_5",
    "away_fantasy_pts_rolling_5",
    "home_def_sacks_rolling_5",
    "away_def_sacks_rolling_5",
    "home_team_stats_missing",
    "away_team_stats_missing",
]


def load_weekly_player_stats(
    seasons: list[int],
    cache_dir: str = "data/interim/nfl",
) -> pd.DataFrame:
    """Load nflreadpy weekly player stats with local caching."""
    bad = [s for s in seasons if s < SPORTSLAB_MIN_SEASON]
    if bad:
        raise ValueError(f"Seasons before {SPORTSLAB_MIN_SEASON} not allowed: {bad}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fragments: list[pd.DataFrame] = []
    uncached: list[int] = []

    for s in seasons:
        cf = cache_path / f"player_stats_{s}.parquet"
        if cf.exists():
            fragments.append(pd.read_parquet(cf))
        else:
            uncached.append(s)

    if uncached:
        import nflreadpy as nfl

        for s in uncached:
            raw = nfl.load_player_stats(int(s))
            season_df = raw.to_pandas()
            cf = cache_path / f"player_stats_{s}.parquet"
            season_df.to_parquet(cf, index=False)
            fragments.append(season_df)

    return pd.concat(fragments, ignore_index=True) if fragments else pd.DataFrame()


def aggregate_team_stats(
    player_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate player-level stats to team-level per week.

    Returns DataFrame with columns: season, week, team, opponent_team,
    off_yds, def_yds_allowed, fantasy_pts, def_sacks.
    """
    # Offensive team totals
    off = (
        player_stats.groupby(["season", "week", "team"], observed=False)
        .agg(
            passing_yards=("passing_yards", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            fantasy_pts=("fantasy_points", "sum"),
            def_sacks=("def_sacks", "sum"),
            def_interceptions=("def_interceptions", "sum"),
        )
        .reset_index()
    )
    off["off_yds"] = off["passing_yards"] + off["rushing_yards"]

    # Per-team: yards allowed = opponent's offensive yards
    opp_lookup = (
        player_stats[["season", "week", "team", "opponent_team"]]
        .drop_duplicates()
        .dropna(subset=["opponent_team"])
    )
    off_with_opp = off.merge(opp_lookup, on=["season", "week", "team"], how="left")

    # For each team, yards allowed = opponent's total offensive yards
    opp_off = off_with_opp[["season", "week", "team", "off_yds"]].rename(
        columns={"team": "opponent_team", "off_yds": "opp_off_yds"}
    )
    merged = off_with_opp.merge(opp_off, on=["season", "week", "opponent_team"], how="left")
    merged["def_yds_allowed"] = merged["opp_off_yds"]

    out = merged[
        ["season", "week", "team", "off_yds", "def_yds_allowed", "fantasy_pts", "def_sacks"]
    ].copy()
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def _compute_rolling(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def compute_rolling_team_stats(
    team_stats: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling averages for team stats by team+season."""
    if windows is None:
        windows = [3, 5]
    out = team_stats.copy().sort_values(["season", "week", "team"])

    metrics = ["off_yds", "def_yds_allowed", "fantasy_pts", "def_sacks"]

    for (team, season), grp in out.groupby(["team", "season"], observed=False):
        grp_idx = grp.index
        for w in windows:
            for metric in metrics:
                col = f"{metric}_rolling_{w}"
                out.loc[grp_idx, col] = _compute_rolling(grp[metric].astype(float), w)

    return out


def compute_team_stats_features(
    df_games: pd.DataFrame,
    player_stats: pd.DataFrame | None = None,
    cache_dir: str = "data/interim/nfl",
) -> pd.DataFrame:
    """Compute pregame rolling team-stat features for each game."""
    out = df_games.copy()

    if player_stats is None:
        seasons_needed = sorted(out["season"].unique())
        player_stats = load_weekly_player_stats(seasons_needed, cache_dir=cache_dir)

    if player_stats.empty:
        for c in TEAM_STATS_COLUMNS:
            out[c] = 0
        return out

    tg = aggregate_team_stats(player_stats)
    if tg.empty:
        for c in TEAM_STATS_COLUMNS:
            out[c] = 0
        return out

    tg_roll = compute_rolling_team_stats(tg)

    # Per-game lookup — join on (season, week, team)
    home_off_3: list[float] = []
    home_off_5: list[float] = []
    home_def_3: list[float] = []
    home_def_5: list[float] = []
    away_off_3: list[float] = []
    away_off_5: list[float] = []
    away_def_3: list[float] = []
    away_def_5: list[float] = []
    home_fp_3: list[float] = []
    home_fp_5: list[float] = []
    away_fp_3: list[float] = []
    away_fp_5: list[float] = []
    home_sacks_3: list[float] = []
    home_sacks_5: list[float] = []
    away_sacks_3: list[float] = []
    away_sacks_5: list[float] = []
    home_missing: list[float] = []
    away_missing: list[float] = []

    for _, row in out.iterrows():
        season = row["season"]
        week = row["week"]
        home = row["home_team"]
        away = row["away_team"]

        h_match = (
            (tg_roll["team"] == home) & (tg_roll["season"] == season) & (tg_roll["week"] == week)
        )
        a_match = (
            (tg_roll["team"] == away) & (tg_roll["season"] == season) & (tg_roll["week"] == week)
        )
        h_row = tg_roll[h_match]
        a_row = tg_roll[a_match]

        home_off_3.append(h_row["off_yds_rolling_3"].iloc[0] if not h_row.empty else np.nan)
        home_off_5.append(h_row["off_yds_rolling_5"].iloc[0] if not h_row.empty else np.nan)
        home_def_3.append(h_row["def_yds_allowed_rolling_3"].iloc[0] if not h_row.empty else np.nan)
        home_def_5.append(h_row["def_yds_allowed_rolling_5"].iloc[0] if not h_row.empty else np.nan)
        home_fp_3.append(h_row["fantasy_pts_rolling_3"].iloc[0] if not h_row.empty else np.nan)
        home_fp_5.append(h_row["fantasy_pts_rolling_5"].iloc[0] if not h_row.empty else np.nan)
        home_sacks_3.append(h_row["def_sacks_rolling_3"].iloc[0] if not h_row.empty else np.nan)
        home_sacks_5.append(h_row["def_sacks_rolling_5"].iloc[0] if not h_row.empty else np.nan)

        away_off_3.append(a_row["off_yds_rolling_3"].iloc[0] if not a_row.empty else np.nan)
        away_off_5.append(a_row["off_yds_rolling_5"].iloc[0] if not a_row.empty else np.nan)
        away_def_3.append(a_row["def_yds_allowed_rolling_3"].iloc[0] if not a_row.empty else np.nan)
        away_def_5.append(a_row["def_yds_allowed_rolling_5"].iloc[0] if not a_row.empty else np.nan)
        away_fp_3.append(a_row["fantasy_pts_rolling_3"].iloc[0] if not a_row.empty else np.nan)
        away_fp_5.append(a_row["fantasy_pts_rolling_5"].iloc[0] if not a_row.empty else np.nan)
        away_sacks_3.append(a_row["def_sacks_rolling_3"].iloc[0] if not a_row.empty else np.nan)
        away_sacks_5.append(a_row["def_sacks_rolling_5"].iloc[0] if not a_row.empty else np.nan)

        home_missing.append(1.0 if h_row.empty else 0.0)
        away_missing.append(1.0 if a_row.empty else 0.0)

    out["home_off_yds_rolling_3"] = home_off_3
    out["away_off_yds_rolling_3"] = away_off_3
    out["home_def_yds_allowed_rolling_3"] = home_def_3
    out["away_def_yds_allowed_rolling_3"] = away_def_3
    out["home_off_yds_rolling_5"] = home_off_5
    out["away_off_yds_rolling_5"] = away_off_5
    out["home_def_yds_allowed_rolling_5"] = home_def_5
    out["away_def_yds_allowed_rolling_5"] = away_def_5
    out["off_yds_net_3"] = pd.Series(home_off_3) - pd.Series(away_def_3)
    out["off_yds_net_5"] = pd.Series(home_off_5) - pd.Series(away_def_5)
    out["home_fantasy_pts_rolling_3"] = home_fp_3
    out["away_fantasy_pts_rolling_3"] = away_fp_3
    out["home_def_sacks_rolling_3"] = home_sacks_3
    out["away_def_sacks_rolling_3"] = away_sacks_3
    out["home_fantasy_pts_rolling_5"] = home_fp_5
    out["away_fantasy_pts_rolling_5"] = away_fp_5
    out["home_def_sacks_rolling_5"] = home_sacks_5
    out["away_def_sacks_rolling_5"] = away_sacks_5
    out["home_team_stats_missing"] = home_missing
    out["away_team_stats_missing"] = away_missing

    # Fill NaN with 0 (neutral)
    for c in TEAM_STATS_COLUMNS:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    return out
