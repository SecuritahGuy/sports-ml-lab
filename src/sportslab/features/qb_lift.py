"""QB Lift: rolling QB efficiency features from play-by-play data.

Computes per-game QB EPA/dropback, CPOE, and success rate from PBP,
then constructs rolling-window averages (3-game, 5-game) for pregame-safe
features. Filters QBs with >= 10 dropbacks per game.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PBP_DIR = "data/interim/nfl"
MIN_DROPBACKS = 10
ROLLING_WINDOWS = [3, 5]

QB_LIFT_COLUMNS = [
    "home_qb_epa_3",
    "away_qb_epa_3",
    "home_qb_epa_5",
    "away_qb_epa_5",
    "home_qb_cpoe_3",
    "away_qb_cpoe_3",
    "net_qb_epa_3",
    "net_qb_epa_5",
]


def _short_to_long(name: str) -> str:
    """Convert 'P.Mahomes' -> 'Patrick Mahomes' using PBP player name table."""
    return name


def _build_pbp_index(pbp_dir: str = PBP_DIR) -> pd.DataFrame:
    """Load and combine PBP from all available seasons."""
    pdir = Path(pbp_dir)
    frames = []
    for fpath in sorted(pdir.glob("pbp_*.parquet")):
        frames.append(pd.read_parquet(fpath))
    if not frames:
        raise FileNotFoundError(f"No PBP parquet files found in {pbp_dir}")
    return pd.concat(frames, ignore_index=True)


def compute_qb_game_stats(
    pbp: Optional[pd.DataFrame] = None,
    pbp_dir: str = PBP_DIR,
) -> pd.DataFrame:
    """Compute per-game QB stats from play-by-play data.

    Returns DataFrame with columns:
        game_id, passer_player_id, passer_player_name, posteam,
        dropbacks, total_epa, epa_per_db, avg_cpoe, success_rate
    Filtered to QBs with >= MIN_DROPBACKS.
    """
    if pbp is None:
        pbp = _build_pbp_index(pbp_dir)

    if pbp.empty or "passer_player_id" not in pbp.columns:
        return pd.DataFrame(columns=[
            "game_id", "passer_player_id", "passer_player_name", "posteam",
            "dropbacks", "total_epa", "avg_cpoe", "avg_success", "epa_per_db",
        ])

    qb_games = (
        pbp[pbp["passer_player_id"].notna()]
        .groupby(["game_id", "passer_player_id", "passer_player_name", "posteam"])
        .agg(
            dropbacks=("passer_player_id", "count"),
            total_epa=("epa", "sum"),
            avg_cpoe=("cpoe", "mean"),
            avg_success=("success", "mean"),
        )
        .reset_index()
    )

    qb_games["epa_per_db"] = qb_games["total_epa"] / qb_games["dropbacks"]
    qb_games = qb_games[qb_games["dropbacks"] >= MIN_DROPBACKS].copy()
    return qb_games.reset_index(drop=True)


def _map_qb_names_to_feature_format(qb_stats: pd.DataFrame) -> pd.DataFrame:
    """Build name mapping: PBP short names -> feature table long names.

    PBP uses 'P.Mahomes', feature table uses 'Patrick Mahomes'.
    We extract the mapping from the full PBP player name table.
    """
    pbp_full = _build_pbp_index(PBP_DIR)
    name_map = (
        pbp_full[pbp_full["passer_player_id"].notna()]
        .groupby("passer_player_id")
        .agg(
            short_name=("passer_player_name", "first"),
            full_name=("passer", lambda x: x.dropna().iloc[0] if x.notna().any() else ""),
        )
        .reset_index()
    )

    name_lookup = dict(zip(name_map["short_name"], name_map["full_name"]))
    qb_stats["qb_name_long"] = qb_stats["passer_player_name"].map(name_lookup)
    qb_stats["qb_name_long"] = qb_stats["qb_name_long"].fillna(qb_stats["passer_player_name"])
    return qb_stats


def _add_game_season_week(qb_stats: pd.DataFrame) -> pd.DataFrame:
    """Add season and week to QB stats from game info."""
    pbp = _build_pbp_index(PBP_DIR)
    game_info = pbp[["game_id", "season", "week"]].drop_duplicates("game_id")
    qb_stats = qb_stats.merge(game_info, on="game_id", how="left")
    return qb_stats


def compute_rolling_qb_features(
    qb_stats: pd.DataFrame,
    windows: list = None,
) -> pd.DataFrame:
    """Compute rolling-window QB features.

    For each QB, computes rolling averages of EPA/dropback and CPOE
    over prior games (excludes current game). Requires chronological
    ordering by season and week.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    result_rows = []
    for (pid,), grp in qb_stats.groupby(["passer_player_id"]):
        grp = grp.sort_values(["season", "week"]).reset_index(drop=True)
        for i in range(len(grp)):
            row = grp.iloc[i]
            row_dict = {
                "game_id": row["game_id"],
                "passer_player_id": pid,
            }
            for w in windows:
                start = max(0, i - w)
                prior = grp.iloc[start:i]
                if len(prior) > 0:
                    row_dict[f"rolling_epa_{w}"] = prior["epa_per_db"].mean()
                    row_dict[f"rolling_cpoe_{w}"] = prior["avg_cpoe"].mean()
                else:
                    row_dict[f"rolling_epa_{w}"] = np.nan
                    row_dict[f"rolling_cpoe_{w}"] = np.nan
            result_rows.append(row_dict)

    return pd.DataFrame(result_rows)


def compute_qb_lift_features(
    ft: pd.DataFrame,
    pbp_dir: str = PBP_DIR,
) -> pd.DataFrame:
    """Compute QB Lift rolling features and merge into feature table.

    Adds columns:
        home_qb_epa_{3,5}: home QB rolling EPA/dropback
        away_qb_epa_{3,5}: away QB rolling EPA/dropback
        home_qb_cpoe_{3,5}: home QB rolling CPOE
        away_qb_cpoe_{3,5}: away QB rolling CPOE
        net_qb_epa_{3,5}: home - away EPA/dropback difference
    """
    print("  Computing QB game stats from PBP...")
    qb_stats = compute_qb_game_stats(pbp_dir=pbp_dir)
    qb_stats = _add_game_season_week(qb_stats)
    qb_stats = _map_qb_names_to_feature_format(qb_stats)

    print(f"  QB-game rows with 10+ dropbacks: {len(qb_stats)}")

    print("  Computing rolling QB features...")
    rolling = compute_rolling_qb_features(qb_stats)
    print(f"  Rolling feature rows: {len(rolling)}")

    # Build lookup from (game_id, passer_player_id) -> rolling values
    epa_3_lookup = {}
    epa_5_lookup = {}
    cpoe_3_lookup = {}
    for _, r in rolling.iterrows():
        epa_3_lookup[(r["game_id"], r["passer_player_id"])] = r["rolling_epa_3"]
        epa_5_lookup[(r["game_id"], r["passer_player_id"])] = r["rolling_epa_5"]
        cpoe_3_lookup[(r["game_id"], r["passer_player_id"])] = r["rolling_cpoe_3"]

    # Load PBP for game info
    pbp = _build_pbp_index(pbp_dir)

    # For each QB in the QB stats, find their home/away designation
    qb_home_away = qb_stats[["game_id", "passer_player_id", "passer_player_name", "posteam"]].copy()

    # Merge with game info to get home/away team
    game_info = pbp[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    qb_home_away = qb_home_away.merge(game_info, on="game_id", how="left")
    qb_home_away["is_home"] = qb_home_away["posteam"] == qb_home_away["home_team"]

    print("  Merging QB lift features into feature table...")
    out = ft.copy()

    # Initialize columns
    for col in QB_LIFT_COLUMNS:
        out[col] = np.nan

    # Use the QB stats game-level data with rolling features pre-computed
    for _, r in qb_home_away.iterrows():
        gid = r["game_id"]
        pid = r["passer_player_id"]
        is_home = r["is_home"]

        # Get rolling values
        e3 = epa_3_lookup.get((gid, pid), np.nan)
        e5 = epa_5_lookup.get((gid, pid), np.nan)
        c3 = cpoe_3_lookup.get((gid, pid), np.nan)

        # Map to feature table rows
        ft_mask = out["game_id"] == gid
        if not ft_mask.any():
            continue

        if is_home:
            out.loc[ft_mask, "home_qb_epa_3"] = e3
            out.loc[ft_mask, "home_qb_epa_5"] = e5
            out.loc[ft_mask, "home_qb_cpoe_3"] = c3
        else:
            out.loc[ft_mask, "away_qb_epa_3"] = e3
            out.loc[ft_mask, "away_qb_epa_5"] = e5
            out.loc[ft_mask, "away_qb_cpoe_3"] = c3

    # Compute net differences
    out["net_qb_epa_3"] = out["home_qb_epa_3"] - out["away_qb_epa_3"]
    out["net_qb_epa_5"] = out["home_qb_epa_5"] - out["away_qb_epa_5"]

    filled = out[QB_LIFT_COLUMNS].notna().any(axis=1).sum()
    print(f"  Games with any QB lift feature: {filled}/{len(out)}")
    for col in QB_LIFT_COLUMNS:
        n = out[col].notna().sum()
        print(f"    {col}: {n}/{len(out)} non-null")

    return out
