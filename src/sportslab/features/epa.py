"""Pregame rolling EPA (Expected Points Added) features.

Computes team-level rolling EPA metrics from nflverse/nflfastR play-by-play
data. All features are computed chronologically using only games played
before the current game.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SPORTSLAB_MIN_SEASON = 2021

ROLLING_WINDOWS = [3, 5]

EPA_FEATURE_COLUMNS: list[str] = [
    # Offense overall
    "home_off_epa_per_play_rolling_3",
    "away_off_epa_per_play_rolling_3",
    "home_off_success_rate_rolling_3",
    "away_off_success_rate_rolling_3",
    "home_off_epa_per_play_rolling_5",
    "away_off_epa_per_play_rolling_5",
    "home_off_success_rate_rolling_5",
    "away_off_success_rate_rolling_5",
    # Defense overall
    "home_def_epa_per_play_rolling_3",
    "away_def_epa_per_play_rolling_3",
    "home_def_success_rate_rolling_3",
    "away_def_success_rate_rolling_3",
    "home_def_epa_per_play_rolling_5",
    "away_def_epa_per_play_rolling_5",
    "home_def_success_rate_rolling_5",
    "away_def_success_rate_rolling_5",
    # Net differentials
    "epa_net_per_play_3",
    "epa_net_per_play_5",
    "success_rate_net_3",
    "success_rate_net_5",
    # Passing splits
    "home_off_pass_epa_rolling_3",
    "away_off_pass_epa_rolling_3",
    "home_off_pass_success_rolling_3",
    "away_off_pass_success_rolling_3",
    "home_off_pass_epa_rolling_5",
    "away_off_pass_epa_rolling_5",
    "home_off_pass_success_rolling_5",
    "away_off_pass_success_rolling_5",
    # Rushing splits
    "home_off_rush_epa_rolling_3",
    "away_off_rush_epa_rolling_3",
    "home_off_rush_success_rolling_3",
    "away_off_rush_success_rolling_3",
    "home_off_rush_epa_rolling_5",
    "away_off_rush_epa_rolling_5",
    "home_off_rush_success_rolling_5",
    "away_off_rush_success_rolling_5",
    # Passing defense splits
    "home_def_pass_epa_rolling_3",
    "away_def_pass_epa_rolling_3",
    "home_def_pass_success_rolling_3",
    "away_def_pass_success_rolling_3",
    "home_def_pass_epa_rolling_5",
    "away_def_pass_epa_rolling_5",
    "home_def_pass_success_rolling_5",
    "away_def_pass_success_rolling_5",
    # Rushing defense splits
    "home_def_rush_epa_rolling_3",
    "away_def_rush_epa_rolling_3",
    "home_def_rush_success_rolling_3",
    "away_def_rush_success_rolling_3",
    "home_def_rush_epa_rolling_5",
    "away_def_rush_epa_rolling_5",
    "home_def_rush_success_rolling_5",
    "away_def_rush_success_rolling_5",
    # Missingness
    "home_epa_games_available",
    "away_epa_games_available",
    "home_epa_missing",
    "away_epa_missing",
]


def load_pbp_data(
    seasons: list[int],
    cache_dir: str = "data/interim/nfl",
) -> pd.DataFrame:
    """Load NFL play-by-play data for given seasons, with local caching.

    Uses nflreadpy to pull from nflverse. If a cached parquet exists
    for a season, loads from cache instead of re-downloading.

    Args:
        seasons: List of season years (min 2021).
        cache_dir: Directory for cached parquet files.

    Returns:
        DataFrame with columns game_id, season, week, posteam, defteam,
        epa, success, pass_attempt, rush_attempt, play_type.
    """
    bad = [s for s in seasons if s < SPORTSLAB_MIN_SEASON]
    if bad:
        raise ValueError(f"Seasons before {SPORTSLAB_MIN_SEASON} not allowed: {bad}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    fragments: list[pd.DataFrame] = []
    uncached: list[int] = []

    for s in seasons:
        cf = cache_path / f"pbp_{s}.parquet"
        if cf.exists():
            fragments.append(pd.read_parquet(cf))
        else:
            uncached.append(s)

    if uncached:
        import nflreadpy as nfl

        for s in uncached:
            raw = nfl.load_pbp(int(s))
            season_df = raw.to_pandas()
            cf = cache_path / f"pbp_{s}.parquet"
            season_df.to_parquet(cf, index=False)
            fragments.append(season_df)

    return pd.concat(fragments, ignore_index=True) if fragments else pd.DataFrame()


def compute_team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play to team-game-level offense and defense stats.

    Args:
        pbp: Play-by-play DataFrame with game_id, season, week, posteam,
             defteam, epa, success, pass_attempt, rush_attempt.

    Returns:
        DataFrame with one row per team per game, containing offensive and
        defensive aggregated EPA stats.
    """
    valid = pbp["epa"].notna() & pbp["posteam"].notna()
    plays = pbp[valid].copy()

    if plays.empty:
        return pd.DataFrame()

    # Offensive stats
    off = (
        plays.groupby(["game_id", "season", "week", "posteam"], observed=False)
        .agg(
            off_plays=("epa", "count"),
            off_epa_total=("epa", "sum"),
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Passing offense
    pass_plays = plays[plays["pass_attempt"] == 1]
    if not pass_plays.empty:
        p_off = (
            pass_plays.groupby(["game_id", "posteam"], observed=False)
            .agg(
                off_pass_epa=("epa", "mean"),
                off_pass_success=("success", "mean"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        off = off.merge(p_off, on=["game_id", "team"], how="left")
    else:
        off["off_pass_epa"] = np.nan
        off["off_pass_success"] = np.nan

    # Rushing offense
    rush_plays = plays[plays["rush_attempt"] == 1]
    if not rush_plays.empty:
        r_off = (
            rush_plays.groupby(["game_id", "posteam"], observed=False)
            .agg(
                off_rush_epa=("epa", "mean"),
                off_rush_success=("success", "mean"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        off = off.merge(r_off, on=["game_id", "team"], how="left")
    else:
        off["off_rush_epa"] = np.nan
        off["off_rush_success"] = np.nan

    # Defensive stats (opponent's offense)
    def_ = (
        plays.groupby(["game_id", "season", "week", "defteam"], observed=False)
        .agg(
            def_plays=("epa", "count"),
            def_epa_total=("epa", "sum"),
            def_epa_per_play=("epa", "mean"),
            def_success_rate=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # Passing defense
    if not pass_plays.empty:
        p_def = (
            pass_plays.groupby(["game_id", "defteam"], observed=False)
            .agg(
                def_pass_epa=("epa", "mean"),
                def_pass_success=("success", "mean"),
            )
            .reset_index()
            .rename(columns={"defteam": "team"})
        )
        def_ = def_.merge(p_def, on=["game_id", "team"], how="left")
    else:
        def_["def_pass_epa"] = np.nan
        def_["def_pass_success"] = np.nan

    # Rushing defense
    if not rush_plays.empty:
        r_def = (
            rush_plays.groupby(["game_id", "defteam"], observed=False)
            .agg(
                def_rush_epa=("epa", "mean"),
                def_rush_success=("success", "mean"),
            )
            .reset_index()
            .rename(columns={"defteam": "team"})
        )
        def_ = def_.merge(r_def, on=["game_id", "team"], how="left")
    else:
        def_["def_rush_epa"] = np.nan
        def_["def_rush_success"] = np.nan

    # Merge offense + defense per team per game
    merged = off.merge(def_, on=["game_id", "season", "week", "team"], how="outer")
    return merged.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _compute_rolling(series: pd.Series, window: int) -> pd.Series:
    """Compute expanding mean over rolling window, shifted so current game
    is excluded. Uses at most `window` prior games.
    """
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def compute_rolling_epa_features(
    team_game: pd.DataFrame,
    windows: list[int] | None = None,
    reset_season: bool = True,
) -> pd.DataFrame:
    """Compute rolling EPA features per team, chronologically.

    For each team, computes rolling averages of offensive and defensive
    EPA/success rate over the last N games, excluding the current game.

    By default, rolling stats reset at season boundaries (no carry-over
    from prior seasons). Set reset_season=False to carry over.

    Args:
        team_game: Output of compute_team_game_epa().
        windows: List of rolling window sizes (default [3, 5]).
        reset_season: If True, reset rolling stats at each season boundary.

    Returns:
        team_game with additional rolling columns:
        (off|def)_(epa_per_play|success_rate)_(pass|rush)?_(WINDOW)
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    out = team_game.copy().sort_values(["season", "week", "game_id"])

    metrics = [
        "off_epa_per_play",
        "off_success_rate",
        "off_pass_epa",
        "off_pass_success",
        "off_rush_epa",
        "off_rush_success",
        "def_epa_per_play",
        "def_success_rate",
        "def_pass_epa",
        "def_pass_success",
        "def_rush_epa",
        "def_rush_success",
    ]

    group_cols = ["team"] if not reset_season else ["team", "season"]

    for team, grp in out.groupby(group_cols, observed=False):
        grp_idx = grp.index
        for w in windows:
            w_str = str(w)
            for metric in metrics:
                col = f"{metric}_rolling_{w_str}"
                out.loc[grp_idx, col] = _compute_rolling(grp[metric].astype(float), w)

    return out


def compute_epa_features(
    df_games: pd.DataFrame,
    pbp: pd.DataFrame | None = None,
    cache_dir: str = "data/interim/nfl",
) -> pd.DataFrame:
    """Compute pregame rolling EPA features for each game.

    For each game in df_games, attaches rolling EPA stats computed from
    prior games for both home and away teams.

    Args:
        df_games: Game-level DataFrame with game_id, season, week,
                  home_team, away_team.
        pbp: Pre-loaded PBP DataFrame. If None, loads via load_pbp_data().
        cache_dir: Cache directory for PBP data.

    Returns:
        df_games with added EPA feature columns.
    """
    out = df_games.copy()

    if pbp is None:
        seasons_needed = sorted(out["season"].unique())
        pbp = load_pbp_data(seasons_needed, cache_dir=cache_dir)

    if pbp.empty:
        for c in EPA_FEATURE_COLUMNS:
            out[c] = 0
        return out

    # ── Aggregate PBP to team-game level ──
    tg = compute_team_game_epa(pbp)
    if tg.empty:
        for c in EPA_FEATURE_COLUMNS:
            out[c] = 0
        return out

    # ── Compute rolling features per team ──
    tg_roll = compute_rolling_epa_features(tg)

    # ── For each game, look up home/away team rolling features ──
    epa_cols_map: dict[str, str] = {}
    for w in ROLLING_WINDOWS:
        w_str = str(w)
        for base in [
            "off_epa_per_play",
            "off_success_rate",
            "off_pass_epa",
            "off_pass_success",
            "off_rush_epa",
            "off_rush_success",
            "def_epa_per_play",
            "def_success_rate",
            "def_pass_epa",
            "def_pass_success",
            "def_rush_epa",
            "def_rush_success",
        ]:
            epa_cols_map[f"{base}_rolling_{w_str}"] = f"{base}_rolling_{w_str}"

    home_features: dict[str, list[float]] = {f"home_{k}": [] for k in epa_cols_map}
    away_features: dict[str, list[float]] = {f"away_{k}": [] for k in epa_cols_map}
    home_games_avail: list[int] = []
    away_games_avail: list[int] = []
    home_missing: list[int] = []
    away_missing: list[int] = []

    for _, row in out.iterrows():
        gid = row["game_id"]
        home = row["home_team"]
        away = row["away_team"]

        # Look up rolling stats for home and away teams
        home_row = tg_roll[(tg_roll["team"] == home) & (tg_roll["game_id"] == gid)]
        away_row = tg_roll[(tg_roll["team"] == away) & (tg_roll["game_id"] == gid)]

        h_avail = 0
        a_avail = 0
        h_miss = 1 if home_row.empty else 0
        a_miss = 1 if away_row.empty else 0

        for col_key, epa_col in epa_cols_map.items():
            h_val = (
                home_row[epa_col].iloc[0]
                if not home_row.empty and epa_col in home_row.columns
                else np.nan
            )
            a_val = (
                away_row[epa_col].iloc[0]
                if not away_row.empty and epa_col in away_row.columns
                else np.nan
            )
            # Track games available as the number of prior games for the team
            if not home_row.empty:
                # Count how many prior games this team has played in the season
                prior_count = tg_roll[
                    (tg_roll["team"] == home)
                    & (
                        (tg_roll["season"] < row["season"])
                        | ((tg_roll["season"] == row["season"]) & (tg_roll["week"] < row["week"]))
                    )
                ].shape[0]
                h_avail = min(prior_count, max(ROLLING_WINDOWS))
            if not away_row.empty:
                prior_count = tg_roll[
                    (tg_roll["team"] == away)
                    & (
                        (tg_roll["season"] < row["season"])
                        | ((tg_roll["season"] == row["season"]) & (tg_roll["week"] < row["week"]))
                    )
                ].shape[0]
                a_avail = min(prior_count, max(ROLLING_WINDOWS))

            home_features[f"home_{col_key}"].append(h_val)
            away_features[f"away_{col_key}"].append(a_val)

        home_games_avail.append(h_avail)
        away_games_avail.append(a_avail)
        home_missing.append(h_miss)
        away_missing.append(a_miss)

    for col, vals in home_features.items():
        out[col] = vals
    for col, vals in away_features.items():
        out[col] = vals
    out["home_epa_games_available"] = home_games_avail
    out["away_epa_games_available"] = away_games_avail
    out["home_epa_missing"] = home_missing
    out["away_epa_missing"] = away_missing

    # ── Impute NaN with 0 (neutral EPA) and fill all missing ──
    epa_cols = [
        c
        for c in out.columns
        if any(pat in c for pat in ["_epa_", "_success_", "_epa_games", "_epa_missing"])
    ]
    for c in epa_cols:
        out[c] = out[c].fillna(0)

    # ── Compute differentials ──
    for w in ROLLING_WINDOWS:
        w_str = str(w)
        out[f"epa_net_per_play_{w_str}"] = (
            out[f"home_off_epa_per_play_rolling_{w_str}"]
            - out[f"away_def_epa_per_play_rolling_{w_str}"]
        )
        out[f"success_rate_net_{w_str}"] = (
            out[f"home_off_success_rate_rolling_{w_str}"]
            - out[f"away_def_success_rate_rolling_{w_str}"]
        )

    return out
