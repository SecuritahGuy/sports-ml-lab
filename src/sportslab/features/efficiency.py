"""Comprehensive pregame efficiency features from multiple nflreadpy sources.

Sources:
  1. Team Stats Total EPA (passing_epa, rushing_epa, receiving_epa) — game-level
  2. PFR Advanced Stats (pass pressure/bad-throw rates, rush YAC/broken-tackles,
     rec drop rates, def passer-rating/missed-tackles) — player→team aggregated
  3. Snap Counts (OL snap%, top RB snap%) — playing-time indicators

All features are computed chronologically, shifted (current game excluded),
and reset at season boundaries.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None

SPORTSLAB_MIN_SEASON = 2021
ROLLING_WINDOWS = [3, 5]

# ── Column lists ──────────────────────────────────────────────────────────

TEAM_EPA_COLUMNS = [
    "home_pass_epa_3",
    "away_pass_epa_3",
    "home_pass_epa_5",
    "away_pass_epa_5",
    "home_rush_epa_3",
    "away_rush_epa_3",
    "home_rush_epa_5",
    "away_rush_epa_5",
    "home_rec_epa_3",
    "away_rec_epa_3",
    "home_rec_epa_5",
    "away_rec_epa_5",
    "home_total_epa_3",
    "away_total_epa_3",
    "home_total_epa_5",
    "away_total_epa_5",
    "epa_net_3",
    "epa_net_5",
]

PFR_COLUMNS = [
    # Pass efficiency
    "home_pressure_rate_3",
    "away_pressure_rate_3",
    "home_pressure_rate_5",
    "away_pressure_rate_5",
    "home_bad_throw_rate_3",
    "away_bad_throw_rate_3",
    "home_bad_throw_rate_5",
    "away_bad_throw_rate_5",
    "pressure_rate_net_3",
    "pressure_rate_net_5",
    # Rush efficiency
    "home_yac_per_rush_3",
    "away_yac_per_rush_3",
    "home_yac_per_rush_5",
    "away_yac_per_rush_5",
    "home_broken_tackles_per_rush_3",
    "away_broken_tackles_per_rush_3",
    "home_broken_tackles_per_rush_5",
    "away_broken_tackles_per_rush_5",
    "yac_net_3",
    "yac_net_5",
    # Defensive efficiency
    "home_def_passer_rating_3",
    "away_def_passer_rating_3",
    "home_def_passer_rating_5",
    "away_def_passer_rating_5",
    "home_def_missed_tackle_pct_3",
    "away_def_missed_tackle_pct_3",
    "home_def_missed_tackle_pct_5",
    "away_def_missed_tackle_pct_5",
    "def_passer_rating_net_3",
    "def_passer_rating_net_5",
]

SNAP_COLUMNS = [
    "home_ol_snap_pct_3",
    "away_ol_snap_pct_3",
    "home_ol_snap_pct_5",
    "away_ol_snap_pct_5",
    "home_top_rb_snap_pct_3",
    "away_top_rb_snap_pct_3",
    "home_top_rb_snap_pct_5",
    "away_top_rb_snap_pct_5",
    "ol_snap_net_3",
    "ol_snap_net_5",
]

COMPREHENSIVE_EFFICIENCY_COLUMNS = TEAM_EPA_COLUMNS + PFR_COLUMNS + SNAP_COLUMNS

# OL positions in snap counts
OL_POSITIONS = {"T", "G", "C"}

# ── Shared helpers ────────────────────────────────────────────────────────


def _compute_rolling(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over `window` prior games, shifted to exclude current."""
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _build_team_rolling(
    team_game: pd.DataFrame,
    metrics: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling columns per team-season.

    Args:
        team_game: Must have columns team, season, week, game_id, + metrics.
        metrics: Column names to build rolling averages of.
        windows: Rolling window sizes.

    Returns:
        team_game with added ``{metric}_rolling_{w}`` columns.
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    out = team_game.copy().sort_values(["season", "week", "game_id"])
    for (_team, season), grp in out.groupby(["team", "season"], observed=False):
        idx = grp.index
        for w in windows:
            w_str = str(w)
            for m in metrics:
                col = f"{m}_rolling_{w_str}"
                out.loc[idx, col] = _compute_rolling(grp[m].astype(float), w)
    return out


def _lookup_home_away(
    df_games: pd.DataFrame,
    team_game_roll: pd.DataFrame,
    source_prefix: str,
    metrics: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Attach home/away rolling features and net differentials.

    For each game in ``df_games``, looks up the rolling features for the
    home and away teams from ``team_game_roll``.

    Args:
        df_games: Must have game_id, home_team, away_team, season, week.
        team_game_roll: Output of ``_build_team_rolling``.
        source_prefix: Prefix for home/away column names (e.g. "pass", "rush").
        metrics: Metric column names (the ``{metric}_rolling_{w}`` pattern).
        windows: Rolling windows.

    Returns:
        df_games with added home_*, away_*, and *_net columns.
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    out = df_games.copy()
    out = out.reset_index(drop=True)

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        for w in windows:
            w_str = str(w)
            col_map = {}
            for m in metrics:
                col_map[f"{side}_{m}_{w_str}"] = f"{m}_rolling_{w_str}"
            prefix = f"{side}_{source_prefix}_" if source_prefix else f"{side}_"
            _ = prefix  # unused but kept for clarity
            for new_col, src_col in col_map.items():
                vals = []
                for _, row in out.iterrows():
                    match = team_game_roll[
                        (team_game_roll["team"] == row[team_col])
                        & (team_game_roll["game_id"] == row["game_id"])
                    ]
                    vals.append(
                        match[src_col].iloc[0]
                        if not match.empty and src_col in match.columns
                        else np.nan
                    )
                out[new_col] = vals

    # Compute net differentials
    for w in windows:
        w_str = str(w)
        for m in metrics:
            home_col = f"home_{source_prefix}_{m}_{w_str}" if source_prefix else f"home_{m}_{w_str}"
            away_col = f"away_{source_prefix}_{m}_{w_str}" if source_prefix else f"away_{m}_{w_str}"
            net_col = f"{source_prefix}_{m}_net_{w_str}" if source_prefix else f"{m}_net_{w_str}"
            if home_col in out.columns and away_col in out.columns:
                out[net_col] = out[home_col] - out[away_col]

    return out


# ── Team Stats Total EPA ──────────────────────────────────────────────────


def _load_team_stats(seasons: list[int], cache_dir: str = "data/interim/nfl") -> pd.DataFrame:
    """Load and cache team-level stats from nflreadpy."""
    if nfl is None:
        raise ImportError("nflreadpy is required")
    seasons = [int(s) for s in seasons]
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cf = cache_path / "team_stats_all.parquet"
    if cf.exists():
        df = pd.read_parquet(cf)
        loaded_seasons = sorted(df["season"].unique())
        if all(s in loaded_seasons for s in seasons):
            return df[df["season"].isin(seasons)].reset_index(drop=True)

    ts = nfl.load_team_stats(seasons=seasons).to_pandas()
    ts["game_id"] = ts.apply(
        lambda r: f"{r['season']:d}_{r['week']:02d}_{r['team']}_{r['opponent_team']}", axis=1
    )
    ts.to_parquet(cf, index=False)
    return ts


def _compute_team_epa_features(
    df_games: pd.DataFrame,
    ts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rolling total EPA features from team_stats.

    Args:
        df_games: Feature table (needs season, week, home_team, away_team, game_id).
        ts_df: Team stats DataFrame (from ``_load_team_stats``).

    Returns:
        df_games with added TEAM_EPA_COLUMNS.
    """
    tg = ts_df[
        [
            "game_id",
            "season",
            "week",
            "team",
            "opponent_team",
            "passing_epa",
            "rushing_epa",
            "receiving_epa",
        ]
    ].copy()
    tg["total_epa"] = tg["passing_epa"] + tg["rushing_epa"] + tg["receiving_epa"]

    # Defensive EPA = opponent's offensive EPA (join on opponent)
    opp = tg[["game_id", "season", "week", "team", "total_epa"]].rename(
        columns={"team": "opponent_team", "total_epa": "opp_total_epa"}
    )
    tg = tg.merge(opp, on=["game_id", "season", "week", "opponent_team"], how="left")

    epa_metrics = ["passing_epa", "rushing_epa", "receiving_epa", "total_epa"]
    tg_roll = _build_team_rolling(tg, epa_metrics)
    out = _lookup_home_away(df_games, tg_roll, "", epa_metrics)

    # Map source metric names to short output names
    short_map = {
        "passing_epa": "pass_epa",
        "rushing_epa": "rush_epa",
        "receiving_epa": "rec_epa",
        "total_epa": "total_epa",
    }
    for side in ["home", "away"]:
        for w in ROLLING_WINDOWS:
            w_str = str(w)
            for full_name, short_name in short_map.items():
                src_col = f"{side}_{full_name}_{w_str}"
                dst_col = f"{side}_{short_name}_{w_str}"
                if src_col in out.columns and src_col != dst_col:
                    out[dst_col] = out[src_col]
                    out = out.drop(columns=[src_col], errors="ignore")

    # Drop any remaining _rolling_ columns from intermediate processing
    out = out.drop(
        columns=[c for c in out.columns if c.endswith("_rolling_3") or c.endswith("_rolling_5")],
        errors="ignore",
    )

    # Compute net differentials
    for w in ROLLING_WINDOWS:
        w_str = str(w)
        home_total = f"home_total_epa_{w_str}"
        away_total = f"away_total_epa_{w_str}"
        if home_total in out.columns and away_total in out.columns:
            out[f"epa_net_{w_str}"] = (out[home_total] - out[away_total]).fillna(0)

    # Fill remaining NaN
    for c in TEAM_EPA_COLUMNS:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    return out


# ── PFR Advanced Stats ────────────────────────────────────────────────────


def _load_pfr_data(
    seasons: list[int],
    cache_dir: str = "data/interim/nfl",
) -> dict[str, pd.DataFrame]:
    """Load and cache all 4 PFR advanced stat types."""
    if nfl is None:
        raise ImportError("nflreadpy is required")
    seasons = [int(s) for s in seasons]
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    result = {}
    for stype in ["pass", "rush", "rec", "def"]:
        cf = cache_path / f"pfr_{stype}.parquet"
        if cf.exists():
            df = pd.read_parquet(cf)
            loaded = sorted(df["season"].unique())
            if all(s in loaded for s in seasons):
                result[stype] = df[df["season"].isin(seasons)].reset_index(drop=True)
                continue
        raw = nfl.load_pfr_advstats(seasons=seasons, stat_type=stype).to_pandas()
        raw.to_parquet(cf, index=False)
        result[stype] = raw
    return result


def _aggregate_pfr_pass(pfr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PFR passing stats to team-game level.

    Pressure rate = times_pressured / estimated dropbacks.
    Dropbacks estimated as: blitzed + hit + hurried + sacked + pressured.
    """
    df = pfr_df.copy()
    # Estimate dropbacks as sum of pressure-related events
    df["dropbacks_est"] = (
        df["times_pressured"].fillna(0)
        + df["times_blitzed"].fillna(0)
        + df["times_hit"].fillna(0)
        + df["times_hurried"].fillna(0)
        + df["times_sacked"].fillna(0)
        + 1
    )

    team_game = (
        df.groupby(["game_id", "season", "week", "team", "opponent"], observed=False)
        .agg(
            total_pressures=("times_pressured", "sum"),
            total_blitzed=("times_blitzed", "sum"),
            total_hit=("times_hit", "sum"),
            total_hurried=("times_hurried", "sum"),
            total_sacked=("times_sacked", "sum"),
            total_bad_throws=("passing_bad_throws", "sum"),
            total_dropbacks=("dropbacks_est", "sum"),
        )
        .reset_index()
    )
    team_game["pressure_rate"] = (
        team_game["total_pressures"] / team_game["total_dropbacks"]
    ).fillna(0)
    team_game["bad_throw_rate"] = (
        team_game["total_bad_throws"] / team_game["total_dropbacks"]
    ).fillna(0)
    # Blitz rate from defensive perspective (stored as def_times_blitzed)
    # Actually times_blitzed is QB-facing; keep as blitz_rate_saw
    return team_game


def _aggregate_pfr_rush(pfr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PFR rushing stats to team-game level (carry-weighted)."""
    df = pfr_df.copy()
    # Weighted average of YBC, YAC per carry + total broken tackles
    team_game = (
        df.groupby(["game_id", "season", "week", "team", "opponent"], observed=False)
        .agg(
            total_carries=("carries", "sum"),
            total_ybc=("rushing_yards_before_contact", "sum"),
            total_yac=("rushing_yards_after_contact", "sum"),
            total_broken_tackles=("rushing_broken_tackles", "sum"),
        )
        .reset_index()
    )
    team_game["yac_per_rush"] = (
        (team_game["total_yac"] / team_game["total_carries"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    team_game["broken_tackles_per_rush"] = (
        team_game["total_broken_tackles"] / team_game["total_carries"]
    ).fillna(0)
    return team_game


def _aggregate_pfr_rec(pfr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PFR receiving stats to team-game level (target-weighted)."""
    df = pfr_df.copy()
    team_game = (
        df.groupby(["game_id", "season", "week", "team", "opponent"], observed=False)
        .agg(
            total_targets=("receiving_drop", "count"),
            total_drops=("receiving_drop", "sum"),
            total_rec_broken_tackles=("receiving_broken_tackles", "sum"),
        )
        .reset_index()
    )
    team_game["drop_rate"] = (team_game["total_drops"] / team_game["total_targets"]).fillna(0)
    team_game["broken_tackles_per_rec"] = (
        (team_game["total_rec_broken_tackles"] / team_game["total_targets"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return team_game


def _aggregate_pfr_def(pfr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PFR defensive stats to team-game level."""
    df = pfr_df.copy()
    team_game = (
        df.groupby(["game_id", "season", "week", "team", "opponent"], observed=False)
        .agg(
            def_targets=("def_targets", "sum"),
            def_completions_allowed=("def_completions_allowed", "sum"),
            def_yards_allowed=("def_yards_allowed", "sum"),
            def_missed_tackles=("def_missed_tackles", "sum"),
            def_tackles_combined=("def_tackles_combined", "sum"),
            def_passer_rating=("def_passer_rating_allowed", "mean"),
        )
        .reset_index()
    )
    team_game["def_completion_pct"] = (
        team_game["def_completions_allowed"] / team_game["def_targets"]
    ).fillna(0)
    team_game["def_missed_tackle_pct"] = (
        team_game["def_missed_tackles"]
        / (team_game["def_missed_tackles"] + team_game["def_tackles_combined"])
    ).fillna(0)
    team_game["def_yards_per_target"] = (
        (team_game["def_yards_allowed"] / team_game["def_targets"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return team_game


def _compute_pfr_features(
    df_games: pd.DataFrame,
    pfr_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute rolling PFR features from all 4 stat types.

    Aggregates:
      - Pass: pressure_rate, bad_throw_rate
      - Rush: yac_per_rush, broken_tackles_per_rush
      - Def: def_passer_rating, def_missed_tackle_pct

    Returns:
        df_games with added PFR_COLUMNS.
    """
    out = df_games.copy()

    # ── Pass ──
    pass_tg = _aggregate_pfr_pass(pfr_data.get("pass", pd.DataFrame()))
    if not pass_tg.empty:
        pass_roll = _build_team_rolling(pass_tg, ["pressure_rate", "bad_throw_rate"])
        pfr_pass_metrics = ["pressure_rate", "bad_throw_rate"]
        for side, team_col in [("home", "home_team"), ("away", "away_team")]:
            for w in ROLLING_WINDOWS:
                w_str = str(w)
                for m in pfr_pass_metrics:
                    src = f"{m}_rolling_{w_str}"
                    col = f"{side}_{m}_{w_str}"
                    vals = []
                    for _, row in out.iterrows():
                        match = pass_roll[
                            (pass_roll["team"] == row[team_col])
                            & (pass_roll["game_id"] == row["game_id"])
                        ]
                        vals.append(match[src].iloc[0] if not match.empty else np.nan)
                    out[col] = vals
        # Net differentials
        for w in ROLLING_WINDOWS:
            w_str = str(w)
            for m in pfr_pass_metrics:
                out[f"{m}_net_{w_str}"] = out[f"home_{m}_{w_str}"] - out[f"away_{m}_{w_str}"]

    # ── Rush ──
    rush_tg = _aggregate_pfr_rush(pfr_data.get("rush", pd.DataFrame()))
    if not rush_tg.empty:
        rush_roll = _build_team_rolling(rush_tg, ["yac_per_rush", "broken_tackles_per_rush"])
        for side, team_col in [("home", "home_team"), ("away", "away_team")]:
            for w in ROLLING_WINDOWS:
                w_str = str(w)
                for m in ["yac_per_rush", "broken_tackles_per_rush"]:
                    src = f"{m}_rolling_{w_str}"
                    col = f"{side}_{m}_{w_str}"
                    vals = []
                    for _, row in out.iterrows():
                        match = rush_roll[
                            (rush_roll["team"] == row[team_col])
                            & (rush_roll["game_id"] == row["game_id"])
                        ]
                        vals.append(match[src].iloc[0] if not match.empty else np.nan)
                    out[col] = vals
        for w in ROLLING_WINDOWS:
            w_str = str(w)
            out[f"yac_net_{w_str}"] = (
                out[f"home_yac_per_rush_{w_str}"] - out[f"away_yac_per_rush_{w_str}"]
            )
            out[f"broken_tackles_net_{w_str}"] = (
                out[f"home_broken_tackles_per_rush_{w_str}"]
                - out[f"away_broken_tackles_per_rush_{w_str}"]
            )

    # ── Defense ──
    def_tg = _aggregate_pfr_def(pfr_data.get("def", pd.DataFrame()))
    if not def_tg.empty:
        def_roll = _build_team_rolling(def_tg, ["def_passer_rating", "def_missed_tackle_pct"])
        for side, team_col in [("home", "home_team"), ("away", "away_team")]:
            for w in ROLLING_WINDOWS:
                w_str = str(w)
                for m in ["def_passer_rating", "def_missed_tackle_pct"]:
                    src = f"{m}_rolling_{w_str}"
                    col = f"{side}_{m}_{w_str}"
                    vals = []
                    for _, row in out.iterrows():
                        match = def_roll[
                            (def_roll["team"] == row[team_col])
                            & (def_roll["game_id"] == row["game_id"])
                        ]
                        vals.append(match[src].iloc[0] if not match.empty else np.nan)
                    out[col] = vals
        for w in ROLLING_WINDOWS:
            w_str = str(w)
            out[f"def_passer_rating_net_{w_str}"] = (
                out[f"home_def_passer_rating_{w_str}"] - out[f"away_def_passer_rating_{w_str}"]
            )
            out[f"def_missed_tackle_net_{w_str}"] = (
                out[f"home_def_missed_tackle_pct_{w_str}"]
                - out[f"away_def_missed_tackle_pct_{w_str}"]
            )

    # Fill NaN
    for c in PFR_COLUMNS:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    return out


# ── Snap Counts ───────────────────────────────────────────────────────────


def _load_snap_counts(
    seasons: list[int],
    cache_dir: str = "data/interim/nfl",
) -> pd.DataFrame:
    """Load and cache snap count data."""
    if nfl is None:
        raise ImportError("nflreadpy is required")
    seasons = [int(s) for s in seasons]
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cf = cache_path / "snap_counts_all.parquet"
    if cf.exists():
        df = pd.read_parquet(cf)
        loaded = sorted(df["season"].unique())
        if all(s in loaded for s in seasons):
            return df[df["season"].isin(seasons)].reset_index(drop=True)
    sc = nfl.load_snap_counts(seasons=seasons).to_pandas()
    sc.to_parquet(cf, index=False)
    return sc


def _compute_snap_features(
    df_games: pd.DataFrame,
    snap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rolling OL snap% and top RB snap% features.

    OL snap% = average snap share of T+G+C position players.
    Top RB snap% = highest snap share among RB-position players.

    Returns:
        df_games with added SNAP_COLUMNS.
    """
    out = df_games.copy()

    # Build team-game position snap percentages
    rows = []
    for (game_id, season, week, team), grp in snap_df.groupby(
        ["game_id", "season", "week", "team"], observed=False
    ):
        total_off_snaps = grp["offense_snaps"].max() if not grp["offense_snaps"].isna().all() else 0
        if total_off_snaps == 0:
            continue
        # OL avg snap %
        ol_players = grp[grp["position"].isin(OL_POSITIONS)]
        ol_snap_pct = (
            ol_players["offense_snaps"].sum() / (total_off_snaps * len(ol_players))
            if len(ol_players) > 0
            else 0.0
        )
        # Top RB snap %
        rb_players = grp[grp["position"] == "RB"]
        top_rb_pct = (
            rb_players["offense_snaps"].max() / total_off_snaps if len(rb_players) > 0 else 0.0
        )
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "team": team,
                "ol_snap_pct": ol_snap_pct,
                "top_rb_snap_pct": top_rb_pct,
            }
        )

    tg = pd.DataFrame(rows)
    if tg.empty:
        for c in SNAP_COLUMNS:
            out[c] = 0
        return out

    snap_metrics = ["ol_snap_pct", "top_rb_snap_pct"]
    tg_roll = _build_team_rolling(tg, snap_metrics)

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        for w in ROLLING_WINDOWS:
            w_str = str(w)
            for m in snap_metrics:
                src = f"{m}_rolling_{w_str}"
                col = f"{side}_{m}_{w_str}"
                vals = []
                for _, row in out.iterrows():
                    match = tg_roll[
                        (tg_roll["team"] == row[team_col]) & (tg_roll["game_id"] == row["game_id"])
                    ]
                    vals.append(match[src].iloc[0] if not match.empty else np.nan)
                out[col] = vals
    for w in ROLLING_WINDOWS:
        w_str = str(w)
        out[f"ol_snap_net_{w_str}"] = (
            out[f"home_ol_snap_pct_{w_str}"] - out[f"away_ol_snap_pct_{w_str}"]
        )

    for c in SNAP_COLUMNS:
        if c in out.columns:
            out[c] = out[c].fillna(0.5)

    return out


# ── Main entry point ──────────────────────────────────────────────────────


def compute_comprehensive_efficiency_features(
    df_games: pd.DataFrame,
    cache_dir: str = "data/interim/nfl",
) -> pd.DataFrame:
    """Compute all comprehensive efficiency features.

    Loads and computes:
      1. Team Stats Total EPA (pass/rush/rec game totals, rolling 3/5)
      2. PFR Advanced Stats (pressure rate, bad throw rate, YAC/rush,
         broken tackles, def passer rating, def missed tackle %)
      3. Snap Counts (OL snap%, top RB snap%)

    All features are pregame-safe: rolling windows are shifted, current
    game excluded, and reset at season boundaries.

    Args:
        df_games: Feature table with game_id, season, week, home_team,
                  away_team, home_score, away_score.
        cache_dir: Directory for caching raw data.

    Returns:
        df_games with added comprehensive efficiency feature columns.
    """
    out = df_games.copy()
    seasons_needed = [int(s) for s in sorted(out["season"].unique())]

    # 1. Team Stats Total EPA
    print("\n=== Team Stats Total EPA ===")
    ts = _load_team_stats(seasons_needed, cache_dir=cache_dir)
    if not ts.empty:
        out = _compute_team_epa_features(out, ts)
        team_epa_added = [c for c in TEAM_EPA_COLUMNS if c in out.columns]
        print(f"  Added {len(team_epa_added)} team EPA columns")

    # 2. PFR Advanced Stats
    print("\n=== PFR Advanced Stats ===")
    pfr_raw = _load_pfr_data(seasons_needed, cache_dir=cache_dir)
    out = _compute_pfr_features(out, pfr_raw)
    pfr_added = [c for c in PFR_COLUMNS if c in out.columns]
    print(f"  Added {len(pfr_added)} PFR advanced stat columns")

    # 3. Snap Counts
    print("\n=== Snap Counts ===")
    snap_raw = _load_snap_counts(seasons_needed, cache_dir=cache_dir)
    out = _compute_snap_features(out, snap_raw)
    snap_added = [c for c in SNAP_COLUMNS if c in out.columns]
    print(f"  Added {len(snap_added)} snap count columns")

    total_added = len(team_epa_added) + len(pfr_added) + len(snap_added)
    print(f"\n  Total efficiency feature columns added: {total_added}")
    return out
