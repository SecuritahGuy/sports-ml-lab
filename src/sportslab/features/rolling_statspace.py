"""Rolling-window StatSpace component features from per-game PBP data.

Computes per-game offensive/defensive components (EPA, success rate, explosive rate,
etc.) and rolls them over the last N completed games per team. The Platt model's
logistic regression learns the optimal weights — no manual compositing needed.
"""

from typing import Optional

import numpy as np
import pandas as pd

from sportslab.features.epa import load_pbp_data
from sportslab.features.statspace.nfl_chaos_rate import ChaosRateWeights
from sportslab.features.statspace.nfl_doba import DOBAWeights

ROLLING_WINDOWS = [3, 5]
SPORTSLAB_MIN_SEASON = 2021

# Columns that get computed per team per game for offense
OFF_COMPONENT_COLS = [
    "off_epa_per_play",
    "off_success_rate",
    "early_down_success",
    "third_fourth_down_success",
    "explosive_rate",
    "red_zone_td_rate",
    "negative_play_rate",
    "turnover_rate",
    "dependency_penalty",
    "pass_run_epa_gap",
]

# Columns for defense
DEF_COMPONENT_COLS = [
    "def_epa_per_play_allowed",
    "def_success_rate_allowed",
    "negative_epa_forced_rate",
    "sack_rate",
    "turnover_forced_rate",
    "explosive_rate_allowed",
    "third_fourth_down_stop_rate",
    "penalty_first_down_rate_allowed",
]


def _ensure_columns(pbp: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "epa": 0.0, "success": 0.0, "down": 0.0,
        "yards_gained": 0.0, "yardline_100": 100.0,
        "touchdown": 0.0, "interception": 0.0,
        "fumble_lost": 0.0, "sack": 0.0,
        "qb_dropback": 0.0, "rush_attempt": 0.0,
        "game_seconds_remaining": 3600.0,
        "score_differential": 0.0,
        "first_down_penalty": 0.0, "penalty_team": "",
    }
    for col, val in defaults.items():
        if col not in pbp.columns:
            pbp[col] = val
    return pbp


def _scrimmage_only(pbp: pd.DataFrame) -> pd.DataFrame:
    """Filter to scrimmage plays (dropbacks + rushes)."""
    scrim = (pbp["qb_dropback"].fillna(0) == 1) | (pbp["rush_attempt"].fillna(0) == 1)
    return pbp[scrim].copy()


def compute_per_game_offensive(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-game offensive components. One row per (game_id, team)."""
    pbp = _ensure_columns(pbp.copy())
    pbp = pbp[pbp["posteam"].notna()].copy()
    pbp = _scrimmage_only(pbp)

    pbp["is_early"] = pbp["down"].isin([1.0, 2.0])
    pbp["is_late"] = pbp["down"].isin([3.0, 4.0])
    pbp["explosive"] = pbp["yards_gained"].fillna(0) >= 20
    pbp["is_rz"] = pbp["yardline_100"].fillna(100) <= 20
    pbp["negative"] = (pbp["epa"].fillna(0) < 0) | (pbp["yards_gained"].fillna(0) < 0)
    pbp["turnover"] = pbp[["interception", "fumble_lost"]].fillna(0).max(axis=1)
    pbp["garbage"] = (
        (pbp["game_seconds_remaining"].fillna(3600) <= 900)
        & (pbp["score_differential"].fillna(0).abs() >= 17)
    )
    pbp["pepa"] = pbp["epa"].clip(lower=0)
    pbp["exp_pepa"] = pbp["pepa"] * pbp["explosive"].astype(float)
    pbp["gar_pepa"] = pbp["pepa"] * pbp["garbage"].astype(float)

    g = pbp.groupby(["game_id", "season", "week", "posteam"], as_index=False)
    off = g.agg(
        off_epa=("epa", "mean"),
        off_sr=("success", "mean"),
        early_sr=("success", lambda s: s[pbp.loc[s.index, "is_early"]].mean()),
        late_sr=("success", lambda s: s[pbp.loc[s.index, "is_late"]].mean()),
        expl_rate=("explosive", "mean"),
        rz_td=("touchdown", lambda s: s[pbp.loc[s.index, "is_rz"]].mean()),
        neg_rate=("negative", "mean"),
        to_rate=("turnover", "mean"),
        pass_epa=("epa", lambda s: s[pbp.loc[s.index, "qb_dropback"] == 1].mean()),
        rush_epa=("epa", lambda s: s[pbp.loc[s.index, "rush_attempt"] == 1].mean()),
        pepa_sum=("pepa", "sum"),
        exp_pepa_sum=("exp_pepa", "sum"),
        gar_pepa_sum=("gar_pepa", "sum"),
    ).rename(columns={"posteam": "team"})

    off["dep_penalty"] = (
        (off["pass_epa"].fillna(0) - off["rush_epa"].fillna(0)).abs()
        + (off["exp_pepa_sum"] / off["pepa_sum"].replace(0, pd.NA)).fillna(0)
        + (off["gar_pepa_sum"] / off["pepa_sum"].replace(0, pd.NA)).fillna(0)
    ) / 3.0

    cols = {
        "off_epa": "off_epa_per_play",
        "off_sr": "off_success_rate",
        "early_sr": "early_down_success",
        "late_sr": "third_fourth_down_success",
        "expl_rate": "explosive_rate",
        "rz_td": "red_zone_td_rate",
        "neg_rate": "negative_play_rate",
        "to_rate": "turnover_rate",
        "dep_penalty": "dependency_penalty",
    }
    gap = off["pass_epa"].fillna(0) - off["rush_epa"].fillna(0)
    off["pass_run_epa_gap"] = gap.abs()
    off["pass_epa"] = off["pass_epa"].fillna(0)
    off["rush_epa"] = off["rush_epa"].fillna(0)

    keep = ["game_id", "season", "week", "team", "pass_run_epa_gap"] + list(cols.keys())
    return off[keep].rename(columns=cols)


def compute_per_game_defensive(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-game defensive components. One row per (game_id, team)."""
    pbp = _ensure_columns(pbp.copy())
    pbp = pbp[pbp["defteam"].notna()].copy()
    pbp = _scrimmage_only(pbp)

    pbp["is_late"] = pbp["down"].isin([3.0, 4.0])
    pbp["neg_epa"] = pbp["epa"].fillna(0) < 0
    pbp["explosive"] = pbp["yards_gained"].fillna(0) >= 20
    pbp["turnover"] = pbp[["interception", "fumble_lost"]].fillna(0).max(axis=1)
    pbp["stop"] = pbp["is_late"] & (pbp["success"].fillna(0) < 1)
    pbp["pen_fd"] = (
        (pbp["first_down_penalty"].fillna(0) == 1)
        & (pbp["penalty_team"].fillna("") == pbp["defteam"].fillna(""))
    )

    g = pbp.groupby(["game_id", "season", "week", "defteam"], as_index=False)
    def_comp = g.agg(
        def_epa=("epa", "mean"),
        def_sr=("success", "mean"),
        neg_epa_rate=("neg_epa", "mean"),
        sack_rate=("sack", "mean"),
        to_forced_rate=("turnover", "mean"),
        expl_rate_allowed=("explosive", "mean"),
        stop_rate=("stop", lambda s: s[pbp.loc[s.index, "is_late"]].mean()),
        pen_fd_rate=("pen_fd", "mean"),
    ).rename(columns={"defteam": "team"})

    cols = {
        "def_epa": "def_epa_per_play_allowed",
        "def_sr": "def_success_rate_allowed",
        "neg_epa_rate": "negative_epa_forced_rate",
        "sack_rate": "sack_rate",
        "to_forced_rate": "turnover_forced_rate",
        "expl_rate_allowed": "explosive_rate_allowed",
        "stop_rate": "third_fourth_down_stop_rate",
        "pen_fd_rate": "penalty_first_down_rate_allowed",
    }
    return def_comp[list(cols.keys()) + ["game_id", "season", "week", "team"]].rename(columns=cols)


def _zscore_series(s: pd.Series) -> pd.Series:
    """Z-score normalize a series, returning 0 if constant."""
    num = pd.to_numeric(s, errors="coerce").fillna(0)
    std = float(num.std(ddof=0))
    if std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (num - float(num.mean())) / std


def compute_rolling_composites(
    ft: pd.DataFrame,
    pbp: Optional[pd.DataFrame] = None,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Compute rolling DOBA and Chaos Rate composites for each game in ft.

    For each window, computes DOBA and Chaos composite scores from the raw
    rolling component averages, z-scored against all teams in that season.

    Returns ft with additional columns:
      {side}_doba_composite_{window}
      {side}_chaos_composite_{window}
      (4 columns per window = 8 total for [3,5])
    """
    if pbp is None:
        all_seasons = sorted(ft["season"].unique())
        max_pbp = 2025
        seasons = [s for s in all_seasons if s <= max_pbp]
        if not seasons:
            raise ValueError(f"No PBP data available (need seasons <= {max_pbp})")
        print(f"  Loading PBP for seasons {seasons}...")
        pbp = load_pbp_data(seasons)
        print(f"  {len(pbp)} plays loaded")
        if seasons != all_seasons:
            print(f"  Note: no PBP for {set(all_seasons) - set(seasons)} (beyond nflreadpy range)")

    pbp = pbp[pbp["season"] >= SPORTSLAB_MIN_SEASON].copy()
    off = compute_per_game_offensive(pbp)
    def_comp = compute_per_game_defensive(pbp)
    print(f"  Per-game components: {len(off)} offensive, {len(def_comp)} defensive")

    out = ft.copy()

    doba_cols = [c for c in OFF_COMPONENT_COLS if c != "pass_run_epa_gap"]
    chaos_cols = DEF_COMPONENT_COLS

    for window in windows or ROLLING_WINDOWS:
        off_rolled = _compute_rolled(off, OFF_COMPONENT_COLS, window)
        def_rolled = _compute_rolled(def_comp, DEF_COMPONENT_COLS, window)

        for side, team_col in [("home", "home_team"), ("away", "away_team")]:
            doba_score = _composite_from_rolled(
                ft, off_rolled, team_col, doba_cols, _doba_from_z,
            )
            out[f"{side}_doba_composite_{window}"] = doba_score

            chaos_score = _composite_from_rolled(
                ft, def_rolled, team_col, chaos_cols, _chaos_from_z,
            )
            out[f"{side}_chaos_composite_{window}"] = chaos_score

    new_cols = [c for c in out.columns if c not in ft.columns]
    for c in new_cols:
        out[c] = out[c].fillna(0)

    return out


def _zscore_season_columns(
    raw: pd.DataFrame,
    comp_cols: list[str],
) -> pd.DataFrame:
    """Z-score each component column across teams within each season."""
    result = raw.copy()
    for col in comp_cols:
        col_z = f"{col}_z"
        result[col_z] = 0.0
        for season in result["season"].unique():
            mask = result["season"] == season
            result.loc[mask, col_z] = _zscore_series(result.loc[mask, col])
    return result


def _doba_from_z(row: pd.Series) -> float:
    """Compute DOBA score from z-scored components."""
    w = DOBAWeights()
    z_prefix = "_z"
    vals = np.array([
        row.get(f"off_epa_per_play{z_prefix}", 0),
        row.get(f"off_success_rate{z_prefix}", 0),
        row.get(f"early_down_success{z_prefix}", 0),
        row.get(f"explosive_rate{z_prefix}", 0),
        row.get(f"red_zone_td_rate{z_prefix}", 0),
        row.get(f"third_fourth_down_success{z_prefix}", 0),
        row.get(f"negative_play_rate{z_prefix}", 0),
        row.get(f"turnover_rate{z_prefix}", 0),
        row.get(f"dependency_penalty{z_prefix}", 0),
    ])
    weights = np.array([
        w.offensive_epa_per_play, w.offensive_success_rate,
        w.early_down_efficiency, w.explosive_rate, w.red_zone_efficiency,
        w.third_fourth_down_efficiency, w.negative_play_rate,
        w.turnover_rate, w.dependency_penalty,
    ])
    return float(np.dot(vals, weights))


def _chaos_from_z(row: pd.Series) -> float:
    """Compute Chaos Rate score from z-scored components."""
    w = ChaosRateWeights()
    z_prefix = "_z"
    vals = np.array([
        row.get(f"negative_epa_forced_rate{z_prefix}", 0),
        -row.get(f"def_epa_per_play_allowed{z_prefix}", 0),
        -row.get(f"def_success_rate_allowed{z_prefix}", 0),
        row.get(f"sack_rate{z_prefix}", 0),
        row.get(f"turnover_forced_rate{z_prefix}", 0),
        row.get(f"third_fourth_down_stop_rate{z_prefix}", 0),
        -row.get(f"explosive_rate_allowed{z_prefix}", 0),
        row.get(f"penalty_first_down_rate_allowed{z_prefix}", 0),
    ])
    weights = np.array([
        w.negative_epa_forced_rate,
        w.defensive_epa_per_play_allowed_inverted,
        w.success_rate_allowed_inverted,
        w.sack_rate,
        w.turnover_forced_rate,
        w.third_fourth_down_stop_rate,
        w.explosive_rate_allowed_inverted,
        w.penalty_first_down_rate_allowed,
    ])
    return float(np.dot(vals, weights)) / 8.0


def _composite_from_rolled(
    ft: pd.DataFrame,
    rolled: pd.DataFrame,
    team_col: str,
    comp_cols: list[str],
    composite_fn,
) -> pd.Series:
    """Compute composite scores from rolled + z-scored components."""
    merged = ft[["game_id", "season", "week", team_col]].merge(
        rolled,
        left_on=["season", "week", team_col],
        right_on=["season", "week", "team"],
        how="left",
    )
    for c in comp_cols:
        merged[c] = merged[c].fillna(0)
    zscored = _zscore_season_columns(merged, comp_cols)
    scores = zscored.apply(composite_fn, axis=1)
    return pd.Series(np.nan_to_num(scores.values, nan=0.0), index=ft.index)


# Keep the raw component version under a different name for API compatibility
compute_rolling_statspace_features = compute_rolling_composites


def _compute_rolled(
    comp_df: pd.DataFrame,
    comp_cols: list[str],
    window: int,
) -> pd.DataFrame:
    """Compute rolling averages of components for each team-game.

    shift(1) excludes the current game, then rolling mean over the window.
    Returns DataFrame with columns [team, season, week] + comp_cols.
    """
    ordered = comp_df.sort_values(["team", "season", "week"]).copy()
    rolled = ordered[["team", "season", "week"] + comp_cols].copy()
    for col in comp_cols:
        rolled[col] = (
            ordered.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return rolled
