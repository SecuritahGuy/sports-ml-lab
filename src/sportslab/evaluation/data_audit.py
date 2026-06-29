"""Data audit — validate schedule and feature table health."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from sportslab.features.build_features import (
    MARKET_COLUMNS,
    MODEL_ELIGIBLE_COLUMN,
    TARGET_COLUMN,
)

SCHEDULES_PATH = "data/raw/nfl/schedules.parquet"
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
STALE_DAYS = 7


def _check(condition: bool, label: str, issues: list) -> None:
    status = "✓" if condition else "✗"
    print(f"  {status} {label}")
    if not condition:
        issues.append(label)


def _check_stale_data(df_ft: pd.DataFrame, ft_path: Path, issues: list) -> None:
    """Detect when the feature table or its source schedule data is stale.

    Warns if:
    - The feature table file is older than STALE_DAYS.
    - Completed past seasons have no target rows (all games played).
    - There are past-dated games that still lack scores.
    """
    now = datetime.now(timezone.utc)

    # File modification time check
    ft_mtime = datetime.fromtimestamp(ft_path.stat().st_mtime, tz=timezone.utc)
    ft_age_days = (now - ft_mtime).days
    _check(ft_age_days < STALE_DAYS,
           f"Feature table is fresh (modified {ft_age_days}d ago, threshold {STALE_DAYS}d)",
           issues)

    # Check if schedule file is newer than feature table (needs rebuild)
    sched_path = Path(SCHEDULES_PATH)
    if sched_path.exists():
        sched_mtime = datetime.fromtimestamp(sched_path.stat().st_mtime, tz=timezone.utc)
        if sched_mtime > ft_mtime:
            msg = (
                f"Schedule is newer than feature table"
                f" (schedule: {sched_mtime.date()}, ft: {ft_mtime.date()})"
                f" — rebuild features"
            )
            _check(False, msg, issues)
        else:
            _check(True, "Schedule not newer than feature table", issues)

    # Past-dated games without scores (non-tie)
    if "gameday" in df_ft.columns and "home_score" in df_ft.columns:
        gameday_dt = pd.to_datetime(df_ft["gameday"], errors="coerce")
        past_mask = gameday_dt < pd.Timestamp(now.date())
        past_games = df_ft[past_mask]
        if "is_tie" in past_games.columns:
            past_non_tie = past_games[~past_games["is_tie"].fillna(False)]
        else:
            past_non_tie = past_games
        missing_scores = past_non_tie[past_non_tie["home_score"].isna()]
        if len(missing_scores) > 0:
            msg = (
                f"Past-dated games without scores: {len(missing_scores)}"
                f" — may indicate partial ingest"
            )
            _check(False, msg, issues)
            for _, row in missing_scores.head(5).iterrows():
                gid = row.get("game_id", "?")
                s = row.get("season", "?")
                w = row.get("week", "?")
                print(f"    {gid} ({s} w{w})")
        else:
            _check(True, "All past-dated games have scores", issues)

        upcoming = df_ft[gameday_dt >= pd.Timestamp(now.date())]
        print(f"  Upcoming games (on or after today): {len(upcoming)}")


def _check_partial_ingest(df_sched: pd.DataFrame, df_ft: pd.DataFrame, issues: list) -> None:
    """Check if schedule and feature table row counts match per season.

    Flags seasons where the feature table is missing games that exist
    in the schedule, indicating a partial ingest or stale rebuild.
    """
    sched_counts = df_sched.groupby("season").size()
    ft_counts = df_ft.groupby("season").size()

    common_seasons = sorted(set(sched_counts.index) & set(ft_counts.index))
    for s in common_seasons:
        sched_n = int(sched_counts[s])
        ft_n = int(ft_counts[s])
        if sched_n != ft_n:
            msg = (
                f"Season {s}: schedule has {sched_n} rows"
                f" but feature table has {ft_n} (diff: {sched_n - ft_n})"
            )
            _check(False, msg, issues)

    # Total row count check
    if len(df_sched) != len(df_ft):
        diff = len(df_sched) - len(df_ft)
        msg = (
            f"Total rows: schedule {len(df_sched)}"
            f" vs feature table {len(df_ft)} (diff: {diff})"
        )
        _check(False, msg, issues)
    else:
        _check(
            True,
            f"Row counts match (schedule: {len(df_sched)}, ft: {len(df_ft)})",
            issues,
        )

    # Season coverage: missing seasons
    sched_seasons = set(df_sched["season"].unique())
    ft_seasons = set(df_ft["season"].unique())
    missing_in_ft = sched_seasons - ft_seasons
    if missing_in_ft:
        _check(
            False,
            f"Seasons in schedule but missing from feature table: {sorted(missing_in_ft)}",
            issues,
        )

    extra_in_sched = ft_seasons - sched_seasons
    if extra_in_sched:
        _check(
            False,
            f"Seasons in feature table but missing from schedule: {sorted(extra_in_sched)}",
            issues,
        )


def run_data_audit(seasons: Optional[List[int]] = None) -> List[str]:
    """Audit schedule and feature table health.

    Checks include: file existence, column presence, duplicate game_ids,
    season validity, stale data, partial ingest, and final-score completeness.

    Args:
        seasons: If provided, only check these seasons.

    Returns:
        List of issue descriptions. Empty if all checks pass.
    """
    issues: List[str] = []

    print("\n=== Data Audit ===\n")

    # ── Schedule file ──
    print("## Schedule File\n")
    sched_path = Path(SCHEDULES_PATH)
    _check(sched_path.exists(), "Schedule file exists", issues)
    if not sched_path.exists():
        return issues

    df_sched = pd.read_parquet(sched_path)
    _check(len(df_sched) > 0, "Schedule has rows", issues)
    print(f"  Rows: {len(df_sched)}, Columns: {len(df_sched.columns)}")
    all_seasons = sorted(df_sched["season"].unique())
    print(f"  Seasons: {all_seasons}")

    if seasons:
        missing = [s for s in seasons if s not in all_seasons]
        _check(len(missing) == 0, f"All requested seasons present: {seasons}", issues)
        if missing:
            print(f"    Missing: {missing}")

    # Required schedule columns
    required_sched = ["game_id", "season", "week", "gameday",
                      "away_team", "home_team", "away_score", "home_score"]
    missing_sched = [c for c in required_sched if c not in df_sched.columns]
    _check(len(missing_sched) == 0, "Required schedule columns exist", issues)
    if missing_sched:
        print(f"    Missing: {missing_sched}")

    # Duplicate game_ids
    dups = df_sched["game_id"].duplicated().sum()
    _check(dups == 0, "No duplicate game_ids", issues)
    if dups:
        print(f"    Found {dups} duplicates")

    # Seasons >= 2021
    bad_seasons = df_sched[df_sched["season"] < 2021]
    _check(len(bad_seasons) == 0, "All seasons >= 2021", issues)

    # ── Feature table ──
    print("\n## Feature Table\n")
    ft_path = Path(FEATURE_TABLE_PATH)
    _check(ft_path.exists(), "Feature table exists", issues)
    if not ft_path.exists():
        return issues

    df_ft = pd.read_parquet(ft_path)
    _check(len(df_ft) > 0, "Feature table has rows", issues)
    print(f"  Rows: {len(df_ft)}, Columns: {len(df_ft.columns)}")
    ft_seasons = sorted(df_ft["season"].unique())
    print(f"  Seasons: {ft_seasons}")
    _check(len(ft_seasons) > 0, "Feature table has seasons", issues)

    if seasons:
        ft_missing = [s for s in seasons if s not in ft_seasons]
        _check(len(ft_missing) == 0, f"All requested seasons in feature table: {seasons}", issues)
        if ft_missing:
            print(f"    Missing: {ft_missing}")

    # Required feature columns
    required_ft = ["game_id", "season", "week", TARGET_COLUMN, MODEL_ELIGIBLE_COLUMN]
    missing_ft = [c for c in required_ft if c not in df_ft.columns]
    _check(len(missing_ft) == 0, "Required feature columns exist", issues)
    if missing_ft:
        print(f"    Missing: {missing_ft}")

    # Duplicate game_ids in feature table
    ft_dups = df_ft["game_id"].duplicated().sum()
    _check(ft_dups == 0, "No duplicate game_ids in feature table", issues)

    # Baseline columns expected in feature table
    expected_base = ["game_id", "season", "week", "gameday",
                     "home_team", "away_team", "home_score", "away_score",
                     "location", TARGET_COLUMN, MODEL_ELIGIBLE_COLUMN]
    expected_in_ft = [c for c in expected_base if c in df_ft.columns]
    _check(len(expected_in_ft) == len(expected_base),
           f"Baseline columns present ({len(expected_in_ft)}/{len(expected_base)})",
           issues)
    if len(expected_in_ft) < len(expected_base):
        missing_base = [c for c in expected_base if c not in df_ft.columns]
        print(f"    Missing: {missing_base}")

    # Market columns preserved in feature table (for audit only)
    market_in_ft = [c for c in MARKET_COLUMNS if c in df_ft.columns]
    _check(len(market_in_ft) > 0, "Market columns preserved in feature table", issues)

    # ── Partial ingest check ──
    print("\n## Partial Ingest Check\n")
    _check_partial_ingest(df_sched, df_ft, issues)

    # ── Stale data check ──
    print("\n## Stale Data Check\n")
    _check_stale_data(df_ft, ft_path, issues)

    # ── Data integrity ──
    print("\n## Data Integrity\n")

    # Future/mode-ineligible games (no home_win) should not have scores
    # (except tie games which are model-ineligible but have scores)
    future = df_ft[df_ft[TARGET_COLUMN].isna()]
    if len(future) > 0:
        if "home_score" in df_ft.columns and "away_score" in df_ft.columns:
            if "is_tie" in df_ft.columns:
                future_non_tie = future[~future["is_tie"].fillna(False)]
            else:
                future_non_tie = future
            future_scores = future_non_tie[future_non_tie["home_score"].notna()]
            if len(future_scores) > 0:
                _check(False,
                       "Non-tie future games have scores",
                       issues)
                print(f"    Non-tie games with scores: {len(future_scores)}")
            else:
                _check(True, "Future games correctly lack scores", issues)
        print(f"  Future games (no target): {len(future)}")
        print(f"  Future seasons: {sorted(future['season'].unique())}")
    else:
        _check(True, "No future games (all have targets)", issues)

    # Completed games should have scores
    completed = df_ft[df_ft[TARGET_COLUMN].notna()]
    if len(completed) > 0:
        if "home_score" in df_ft.columns:
            completed_no_score = completed[completed["home_score"].isna()]
            _check(len(completed_no_score) == 0,
                   "Completed games have scores", issues)
        print(f"  Completed games: {len(completed)}")

    # Seasons 2021+
    _check(df_ft["season"].min() >= 2021,
           "Feature table includes only seasons 2021+", issues)

    # ── Summary ──
    print(f"\n{'='*40}")
    if issues:
        print(f"Audit found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("All checks passed.")
    print(f"{'='*40}\n")

    return issues
