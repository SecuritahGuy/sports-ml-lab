"""Auto-source QB starter data from nflreadpy depth charts.

Extracts starting QB per team per week from nflreadpy's depth_charts
data. Generates a QB-input CSV that can be fed to predict-future /
predict-week.

Handles two depth_chart schemas:
- Pre-2025: has `week`, `club_code`, `position`, `depth_team` columns
- 2025+:    has `pos_abb`, `pos_rank`, `team`, `dt` columns (no week)

For the 2025+ schema without weeks, the latest snapshot for each
team's QB1 is applied to all weeks.

Timing caveat:
    nflreadpy depth charts are updated during the week based on team
    depth charts and injury reports. They should reflect expected
    starters before game time, but may sometimes be updated after
    games. Always verify against official injury reports for critical
    use.

    The returned qb_source value is 'auto_qb' for traceability.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def _load_depth_chart_qbs(
    season: int,
    week: Optional[int] = None,
) -> pd.DataFrame:
    """Load depth_charts and extract starting QBs per team.

    Handles both pre-2025 (week/club_code/depth_team) and 2025+
    (pos_rank/team/dt) schemas.

    Returns a DataFrame with columns:
        team_code, gsis_id, [, week]

    When week column is not available (2025+), the returned DataFrame
    has no week column — the caller should apply the same starter to
    all weeks.
    """
    try:
        import nflreadpy as nfl
    except ImportError:
        raise ImportError(
            "nflreadpy is required for auto QB sourcing. "
            "Install with: pip install nflreadpy"
        )

    dc = nfl.load_depth_charts(seasons=season)

    if dc is None or len(dc) == 0:
        raise ValueError(f"No depth chart data available for season {season}")

    # Detect schema and extract starting QBs
    has_old_schema = "position" in dc.columns and "depth_team" in dc.columns

    if has_old_schema:
        # Pre-2025 schema: has week, club_code, position, depth_team
        starters = dc.filter(
            (dc["position"] == "QB") & (dc["depth_team"] == "1")
        )
        if len(starters) == 0:
            raise ValueError(f"No starting QBs found in depth chart for season {season}")

        starters_pd = starters.to_pandas()
        starters_pd = starters_pd.dropna(subset=["week"]).copy()
        starters_pd["week"] = starters_pd["week"].astype(int)
        result = pd.DataFrame({
            "team_code": starters_pd["club_code"].values,
            "gsis_id": starters_pd["gsis_id"].values,
            "week": starters_pd["week"].values,
        })
        if week is not None:
            result = result[result["week"] == week].reset_index(drop=True)
        return result
    else:
        # 2025+ schema: has pos_abb, pos_rank, team, dt (no week)
        if "pos_abb" not in dc.columns or "pos_rank" not in dc.columns:
            raise ValueError(
                f"Unrecognized depth chart schema for season {season}. "
                f"Columns: {dc.columns}"
            )

        starters = dc.filter(
            (dc["pos_abb"] == "QB") & (dc["pos_rank"] == 1)
        )
        if len(starters) == 0:
            raise ValueError(f"No starting QBs found in depth chart for season {season}")

        # Take the most recent snapshot per team
        starters_df = starters.to_pandas()
        starters_df["dt_parsed"] = pd.to_datetime(starters_df["dt"])
        latest = (
            starters_df.sort_values("dt_parsed")
            .groupby("team")
            .last()
            .reset_index()
        )
        result = pd.DataFrame({
            "team_code": latest["team"].values,
            "gsis_id": latest["gsis_id"].values,
        })
        # No week column -- caller applies to all weeks
        return result


def _map_team_code(code: str) -> str:
    """Map nflreadpy team code to feature table team code.

    Most codes match (BUF, ATL, etc.), but some differ:
    - 'LA' -> 'LAR' (Rams)
    - 'OAK' -> 'LV'  (Raiders; shouldn't appear post-2020)
    - 'STL' -> 'LA'  (Rams; shouldn't appear post-2020)
    - 'SD'  -> 'LAC' (Chargers; shouldn't appear post-2020)
    """
    mapping = {
        "LA": "LAR",
        "OAK": "LV",
        "STL": "LA",
        "SD": "LAC",
    }
    return mapping.get(code, code)


def build_auto_qb_csv(
    season: int,
    week: Optional[int] = None,
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    output_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Build a QB-input CSV from nflreadpy depth charts.

    Args:
        season: Season year (e.g., 2026).
        week: Week number (if None, all weeks).
        feature_table_path: Path to feature table for game_id lookup.
        output_path: Optional path to save CSV.

    Returns:
        Tuple of (DataFrame with QB input columns, qb_source label).
    """
    # Load depth chart QBs (handles both pre-2025 and 2025+ schema)
    dc_starters = _load_depth_chart_qbs(season, week=week)

    has_week_col = "week" in dc_starters.columns

    # Map team codes
    dc_starters["team_code"] = dc_starters["team_code"].apply(_map_team_code)

    # Load feature table to get game_ids for the season/week
    ft_path = Path(feature_table_path)
    if not ft_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table_path}")

    ft = pd.read_parquet(ft_path)
    ft_games = ft[ft["season"] == season].copy()
    if week is not None:
        ft_games = ft_games[ft_games["week"] == week].copy()

    if len(ft_games) == 0:
        raise ValueError(
            f"No games found in feature table for {season}"
            + (f" week {week}" if week is not None else "")
        )

    if has_week_col:
        # Pre-2025: depth chart has week column, merge by team+week
        dc_lookup = dc_starters[["team_code", "week", "gsis_id"]].copy()
        dc_lookup.columns = ["team", "week", "qb_id"]

        ft_games = ft_games.merge(
            dc_lookup,
            left_on=["home_team", "week"],
            right_on=["team", "week"],
            how="left",
        )
        ft_games.rename(columns={"qb_id": "home_qb_id_dc"}, inplace=True)
        ft_games.drop(columns=["team"], inplace=True, errors="ignore")

        ft_games = ft_games.merge(
            dc_lookup,
            left_on=["away_team", "week"],
            right_on=["team", "week"],
            how="left",
        )
        ft_games.rename(columns={"qb_id": "away_qb_id_dc"}, inplace=True)
        ft_games.drop(columns=["team"], inplace=True, errors="ignore")
    else:
        # 2025+: no week column -- same starter applies to all weeks
        dc_lookup = dc_starters[["team_code", "gsis_id"]].copy()
        dc_lookup.columns = ["team", "qb_id"]

        ft_games = ft_games.merge(
            dc_lookup,
            left_on="home_team",
            right_on="team",
            how="left",
        )
        ft_games.rename(columns={"qb_id": "home_qb_id_dc"}, inplace=True)
        ft_games.drop(columns=["team"], inplace=True, errors="ignore")

        ft_games = ft_games.merge(
            dc_lookup,
            left_on="away_team",
            right_on="team",
            how="left",
        )
        ft_games.rename(columns={"qb_id": "away_qb_id_dc"}, inplace=True)
        ft_games.drop(columns=["team"], inplace=True, errors="ignore")

    # Use depth_chart QB when available, fall back to oracle (schedule) QB
    home_oracle = ft_games["home_qb_id"].where(ft_games["home_qb_id"].notna(), None)
    away_oracle = ft_games["away_qb_id"].where(ft_games["away_qb_id"].notna(), None)
    home_dc = ft_games["home_qb_id_dc"].where(ft_games["home_qb_id_dc"].notna(), None)
    away_dc = ft_games["away_qb_id_dc"].where(ft_games["away_qb_id_dc"].notna(), None)

    ft_games["home_qb_id_out"] = home_dc.fillna(home_oracle)
    ft_games["away_qb_id_out"] = away_dc.fillna(away_oracle)

    # Report mismatches
    matched_home = ft_games["home_qb_id_dc"].notna().sum()
    matched_away = ft_games["away_qb_id_dc"].notna().sum()
    oracle_fallback_home = ft_games["home_qb_id_dc"].isna().sum()
    oracle_fallback_away = ft_games["away_qb_id_dc"].isna().sum()
    n_games = len(ft_games)
    print(
        f"  Auto QB: {matched_home}/{n_games} home, "
        f"{matched_away}/{n_games} away from depth charts"
    )
    if oracle_fallback_home > 0 or oracle_fallback_away > 0:
        print(f"  Fallback to oracle: {oracle_fallback_home} home, {oracle_fallback_away} away")

    # Build output columns
    out_df = pd.DataFrame({
        "game_id": ft_games["game_id"].values,
        "home_qb_id": ft_games["home_qb_id_out"].values,
        "away_qb_id": ft_games["away_qb_id_out"].values,
    })

    # Save if requested
    if output_path is not None:
        out_df.to_csv(output_path, index=False)
        print(f"  Auto QB CSV saved: {output_path}")

    qb_source = "auto_qb"
    n_found = ft_games["home_qb_id_dc"].notna().sum()
    print(f"  Auto QB source: {n_found}/{len(ft_games)} games have QB data from depth charts")

    return out_df, qb_source


def build_weekly_qb_csv(
    season: int,
    week: int,
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    output_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Build QB input CSV using week-over-week tracking.

    For each team playing in the target week, finds the actual QB
    starter from their most recent completed game prior to that week.
    Falls back to preseason depth chart snapshot for teams with no
    prior game data (e.g., week 1).

    This is more accurate than a single preseason snapshot because
    it catches mid-season QB changes (injuries, benchings, etc.)
    that the depth chart snapshot misses.

    Works for rehearsal mode immediately (feature table has full
    historical QB data). For live mode, requires backfilled feature
    table after each completed week.

    Args:
        season: Target season.
        week: Target week number.
        feature_table_path: Path to feature table parquet.
        output_path: Optional path to save CSV.

    Returns:
        Tuple of (DataFrame with home_qb_id/away_qb_id, qb_source label).
    """
    ft_path = Path(feature_table_path)
    if not ft_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table_path}")

    ft = pd.read_parquet(ft_path)

    # Get target week's games
    target = ft[(ft["season"] == season) & (ft["week"] == week)].copy()
    if len(target) == 0:
        raise ValueError(f"No games found in feature table for {season} week {week}")

    # Get prior completed games in same season
    prior = ft[(ft["season"] == season) & (ft["week"] < week) & (ft["home_win"].notna())].copy()

    # Build team→QB mapping from prior games using feature table's tracked QB ids
    home_df = prior[["home_team", "week", "home_qb_id"]].dropna(subset=["home_qb_id"])
    away_df = prior[["away_team", "week", "away_qb_id"]].dropna(subset=["away_qb_id"])
    home_df.columns = ["team", "week", "qb_id"]
    away_df.columns = ["team", "week", "qb_id"]

    if len(home_df) == 0 and len(away_df) == 0:
        # No prior data — always week 1
        prior_qbs: Dict[str, str] = {}
    else:
        all_prior = pd.concat([home_df, away_df]).sort_values("week")
        prior_qbs = all_prior.groupby("team").last()["qb_id"].to_dict()

    # Load depth chart snapshot for fallback
    dc_df = _load_depth_chart_qbs(season, week=week)
    dc_df["team_code"] = dc_df["team_code"].apply(_map_team_code)
    dc_qbs = dict(zip(dc_df["team_code"], dc_df["gsis_id"]))

    # Determine each team's QB and source
    all_teams = set(target["home_team"].tolist() + target["away_team"].tolist())

    team_qbs: Dict[str, str] = {}
    team_src: Dict[str, str] = {}
    for team in sorted(all_teams):
        if team in prior_qbs:
            team_qbs[team] = prior_qbs[team]
            team_src[team] = "prior_week"
        elif team in dc_qbs:
            team_qbs[team] = dc_qbs[team]
            team_src[team] = "depth_chart"
        else:
            team_qbs[team] = pd.NA
            team_src[team] = "missing"

    # Build output
    home_ids = [team_qbs[t] for t in target["home_team"]]
    away_ids = [team_qbs[t] for t in target["away_team"]]
    home_src = [team_src[t] for t in target["home_team"]]
    away_src = [team_src[t] for t in target["away_team"]]

    out_df = pd.DataFrame({
        "game_id": target["game_id"].values,
        "home_qb_id": home_ids,
        "away_qb_id": away_ids,
        "home_qb_source": home_src,
        "away_qb_source": away_src,
    })

    n_games = len(target)
    n_prior_home = sum(1 for s in home_src if s == "prior_week")
    n_dc_home = sum(1 for s in home_src if s == "depth_chart")
    n_missing_home = sum(1 for s in home_src if s == "missing")
    n_prior_away = sum(1 for s in away_src if s == "prior_week")
    n_dc_away = sum(1 for s in away_src if s == "depth_chart")

    print(f"  Weekly QB ({season} w{week}):")
    print(f"    Home: {n_prior_home}/{n_games} from prior week, {n_dc_home} depth chart, {n_missing_home} missing")
    print(f"    Away: {n_prior_away}/{n_games} from prior week, {n_dc_away} depth chart")

    # Save if requested
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        print(f"  Weekly QB CSV saved: {output_path}")

    qb_source = "weekly_qb"

    return out_df, qb_source


def build_auto_qb_csv_standalone(
    season: int,
    week: Optional[int] = None,
    output: Optional[str] = None,
) -> str:
    """Standalone entry point for CLI. Returns path to generated CSV."""
    df, source = build_auto_qb_csv(season, week, output_path=output)
    if output:
        return output
    tmp = Path(f"/tmp/auto_qb_{season}_w{week or 'all'}.csv")
    df.to_csv(tmp, index=False)
    return str(tmp)
