#!/usr/bin/env python3
"""
NFL Ingestion Layer for sportslab project.
This module handles downloading and saving NFL schedule/game-level data using nflreadpy.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import nflreadpy

    NFLREADPY_AVAILABLE = True
except ImportError:
    NFLREADPY_AVAILABLE = False


def _normalize_table(obj) -> pd.DataFrame:
    """Normalize various table-like objects to pandas DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj
    elif hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    elif hasattr(obj, "collect") and hasattr(obj.collect(), "to_pandas"):
        return obj.collect().to_pandas()
    else:
        raise ValueError(f"Cannot normalize object of type {type(obj)} to pandas DataFrame")


def _build_metadata(
    source_function: str,
    seasons_requested: List[int],
    seasons_found: List[int],
    df: pd.DataFrame,
    output_path: str,
) -> Dict[str, Any]:
    """Build metadata dictionary for the ingested data."""
    return {
        "package": "nflreadpy",
        "nflreadpy_version": getattr(nflreadpy, "__version__", "unknown"),
        "seasons_requested": seasons_requested,
        "seasons_found": seasons_found,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "created_at": datetime.now().isoformat(),
        "output_path": output_path,
        "source_function": source_function,
    }


NFL_MIN_SEASON = 2021
_SOURCE_FUNCTION = "load_schedules"


def _discover_schedule_function():
    """Find the schedule/game-loading function in nflreadpy."""
    func_name = _SOURCE_FUNCTION
    if hasattr(nflreadpy, func_name):
        return func_name, getattr(nflreadpy, func_name)

    candidates = []
    for name in dir(nflreadpy):
        if name.startswith("_"):
            continue
        lower = name.lower()
        if any(token in lower for token in ["schedule", "game"]):
            candidates.append(name)

    if candidates:
        func_name = candidates[0]
        return func_name, getattr(nflreadpy, func_name)

    public = [n for n in dir(nflreadpy) if not n.startswith("_")]
    raise ValueError(
        f"No schedule/game-related function found in nflreadpy.\n"
        f"Available public names: {public}\n"
        f"Inspect nflreadpy documentation or package source."
    )


def _validate_seasons(seasons: List[int]) -> None:
    """Validate that all requested seasons are >= NFL_MIN_SEASON."""
    bad = [s for s in seasons if s < NFL_MIN_SEASON]
    if bad:
        raise ValueError(
            f"Season(s) {bad} are not allowed. "
            f"This project supports NFL seasons {NFL_MIN_SEASON}–current only. "
            f"Requested: {seasons}"
        )


def _load_existing_schedules(parquet_path: Path) -> Optional[pd.DataFrame]:
    """Load existing schedules parquet if it exists."""
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        if len(existing) > 0:
            return existing
    return None


def _merge_schedules(
    new_df: pd.DataFrame,
    existing_df: Optional[pd.DataFrame],
    replace_all: bool = False,
    replace_seasons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Merge new schedules with existing, deduplicating by game_id.

    Safe append mode (default): new seasons are appended, existing seasons
    are updated in-place (same season re-downloaded replaces old data).

    Args:
        new_df: Newly downloaded schedule data.
        existing_df: Previously saved schedule data, or None.
        replace_all: If True, discard existing data entirely.
        replace_seasons: If set, only these seasons are replaced in the
            existing data; all other existing seasons are preserved.

    Returns:
        Merged DataFrame.
    """
    if replace_all or existing_df is None:
        return new_df

    # Build merged set
    new_seasons = set(new_df["season"].unique())

    if replace_seasons:
        # Remove only the specified seasons from existing
        replace_set = set(replace_seasons)
        existing_kept = existing_df[~existing_df["season"].isin(replace_set)].copy()
        merged = pd.concat([existing_kept, new_df], ignore_index=True)
    else:
        # Default safe append: keep existing rows for seasons not in new data,
        # replace rows for seasons that overlap
        existing_kept = existing_df[~existing_df["season"].isin(new_seasons)].copy()
        merged = pd.concat([existing_kept, new_df], ignore_index=True)

    # Deduplicate by game_id (keep last = new data wins on conflict)
    merged = merged.drop_duplicates(subset=["game_id"], keep="last")
    merged = merged.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    return merged


def ingest_nfl(
    seasons: List[int],
    replace_all: bool = False,
    replace_seasons: Optional[List[int]] = None,
) -> None:
    """
    Ingest NFL schedule/game-level data for specified seasons.

    By default, appends new seasons to existing data without removing
    historical seasons. Use ``replace_all=True`` for a full destructive
    overwrite, or ``replace_seasons=[...]`` to replace specific seasons.

    Uses nflreadpy.load_schedules() to fetch data and saves locally
    as parquet with associated metadata.

    Args:
        seasons: List of season years to download data for.
        replace_all: Discard all existing data before saving.
        replace_seasons: If set, only these seasons are replaced in
            existing data; all other seasons are preserved.

    Raises:
        ImportError: If nflreadpy is not available.
        ValueError: If no suitable function is found, if function call fails,
                    or if any season is before {NFL_MIN_SEASON}.
    """
    if not NFLREADPY_AVAILABLE:
        raise ImportError(
            "nflreadpy not available.\n"
            "Run 'make install' or 'pip install -e .' to install dependencies."
        )

    _validate_seasons(seasons)

    func_name, func = _discover_schedule_function()

    try:
        print(f"Loading data using: nflreadpy.{func_name}(seasons={seasons})")
        result = func(seasons=seasons)
    except Exception as exc:
        print(f"Function: {func_name}")
        print(f"Seasons argument: {seasons}")
        print(f"Exception: {exc}")
        print(
            f'Debug command: python -c "import nflreadpy; '
            f"df = nflreadpy.{func_name}(seasons={seasons}); "
            f'print(type(df)); print(df)"'
        )
        raise ValueError(
            f"nflreadpy.{func_name} failed for seasons={seasons}.\n"
            f"Exception: {exc}\n"
            f"Run the debug command above for more details."
        )

    new_df = _normalize_table(result)

    data_dir = Path("data/raw/nfl")
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "schedules.parquet"
    metadata_path = data_dir / "schedules_metadata.json"

    # Safe merge with existing data
    existing_df = _load_existing_schedules(parquet_path)
    merged = _merge_schedules(new_df, existing_df, replace_all=replace_all,
                              replace_seasons=replace_seasons)

    n_new = len(new_df)
    n_existing = len(existing_df) if existing_df is not None else 0
    n_merged = len(merged)
    if existing_df is not None and not replace_all:
        print(f"  New rows: {n_new}, Existing rows before: {n_existing}, Merged total: {n_merged}")
        print(f"  Seasons before: {sorted(existing_df['season'].unique())}")
    print(f"  Seasons after: {sorted(merged['season'].unique())}")

    merged.to_parquet(parquet_path, index=False)

    metadata = _build_metadata(
        source_function=func_name,
        seasons_requested=seasons,
        seasons_found=seasons,
        df=merged,
        output_path=str(parquet_path),
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Source function used: {func_name}")
    print(f"Row count: {len(merged)}")
    print(f"Column count: {len(merged.columns)}")
    print(f"Seasons requested: {seasons}")
    print(f"Seasons found: {sorted(merged['season'].unique())}")
    print(f"Available columns: {merged.columns.tolist()}")
    print(f"Output parquet path: {parquet_path}")
    print(f"Metadata JSON path: {metadata_path}")
