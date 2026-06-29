"""Rolling turnover differential features — chronologically computed, leakage-safe."""

from pathlib import Path

import pandas as pd

TURNOVER_COLUMNS = [
    "home_to_net_3",
    "away_to_net_3",
    "to_net_diff_3",
    "home_to_net_5",
    "away_to_net_5",
    "to_net_diff_5",
]

CACHE_DIR = "data/interim/nfl"


def _load_team_stats(seasons: list[int]) -> pd.DataFrame:
    """Load team stats from nflreadpy with local cache."""
    import nflreadpy as nfl

    seasons = [int(s) for s in seasons]
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cache_path = Path(CACHE_DIR) / "team_stats_all.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        loaded = sorted(df["season"].unique())
        if all(s in loaded for s in seasons):
            return df[df["season"].isin(seasons)].reset_index(drop=True)

    ts = nfl.load_team_stats(seasons=seasons).to_pandas()
    ts.to_parquet(cache_path, index=False)
    return ts


def _compute_rolling(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over `window` prior games, shifted to exclude current."""
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def compute_turnover_features(
    df_games: pd.DataFrame,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Add rolling turnover differential features.

    Args:
        df_games: Feature table with season, week, home_team, away_team, game_id.
        cache_dir: Directory for caching nflreadpy data.

    Returns:
        df_games with added TURNOVER_COLUMNS.
    """
    seasons_needed = sorted(int(s) for s in df_games["season"].unique() if s >= 2021)
    ts = _load_team_stats(seasons_needed)

    # Compute turnover metrics per team-game
    tg = ts[["game_id", "season", "week", "team", "opponent_team"]].copy()
    tg["to_committed"] = (
        ts["passing_interceptions"].fillna(0).astype(int)
        + ts["sack_fumbles_lost"].fillna(0).astype(int)
        + ts["rushing_fumbles_lost"].fillna(0).astype(int)
        + ts["receiving_fumbles_lost"].fillna(0).astype(int)
    )
    tg["takeaways"] = (
        ts["def_interceptions"].fillna(0).astype(int)
        + ts["def_fumbles"].fillna(0).astype(int)
    )
    tg["to_net"] = (tg["takeaways"] - tg["to_committed"]).astype(int)

    # Build rolling averages per team-season
    tg = tg.sort_values(["season", "week"]).reset_index(drop=True)
    for (team, season), grp in tg.groupby(["team", "season"], observed=False):
        idx = grp.index
        for w in (3, 5):
            col = f"to_net_rolling_{w}"
            tg.loc[idx, col] = _compute_rolling(grp["to_net"].astype(float), w)

    # Look up home/away values for each game
    out = df_games.copy()
    rolling_cols = ["to_net_rolling_3", "to_net_rolling_5"]
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        for rc in rolling_cols:
            w = rc.split("_")[-1]
            col_name = f"{side}_to_net_{w}"
            vals = []
            for _, row in out.iterrows():
                match = tg[
                    (tg["team"] == row[team_col])
                    & (tg["game_id"] == row["game_id"])
                ]
                vals.append(
                    float(match[rc].iloc[0])
                    if not match.empty and rc in match.columns
                    else 0.0
                )
            out[col_name] = vals
            out[col_name] = out[col_name].fillna(0)

    # Net differentials
    out["to_net_diff_3"] = (out["home_to_net_3"] - out["away_to_net_3"]).fillna(0)
    out["to_net_diff_5"] = (out["home_to_net_5"] - out["away_to_net_5"]).fillna(0)

    return out
