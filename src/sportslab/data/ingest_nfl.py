#!/usr/bin/env python3
"""
NFL Ingestion Layer for sportslab project.
This module handles downloading and saving NFL schedule/game-level data using nflreadpy.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
        # Polars DataFrame or similar
        return obj.to_pandas()
    elif hasattr(obj, "collect") and hasattr(obj.collect(), "to_pandas"):
        # Lazy DataFrame with collect method
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
    """Find the schedule/game-loading function in nflreadpy.

    Returns:
        The discovered function object.

    Raises:
        ValueError: If no suitable function is found.
    """
    func_name = _SOURCE_FUNCTION
    if hasattr(nflreadpy, func_name):
        return func_name, getattr(nflreadpy, func_name)

    # Fallback: search for any schedule/game-related function
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
    """Validate that all requested seasons are >= NFL_MIN_SEASON.

    Args:
        seasons: List of season years to validate.

    Raises:
        ValueError: If any season is earlier than NFL_MIN_SEASON.
    """
    bad = [s for s in seasons if s < NFL_MIN_SEASON]
    if bad:
        raise ValueError(
            f"Season(s) {bad} are not allowed. "
            f"This project supports NFL seasons {NFL_MIN_SEASON}–current only. "
            f"Requested: {seasons}"
        )


def ingest_nfl(seasons: List[int]) -> None:
    """
    Ingest NFL schedule/game-level data for specified seasons.

    Uses nflreadpy.load_schedules() to fetch data and saves it locally
    as parquet with associated metadata.

    Args:
        seasons: List of season years to download data for.

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

    df = _normalize_table(result)

    data_dir = Path("data/raw/nfl")
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "schedules.parquet"
    metadata_path = data_dir / "schedules_metadata.json"

    df.to_parquet(parquet_path, index=False)

    metadata = _build_metadata(
        source_function=func_name,
        seasons_requested=seasons,
        seasons_found=seasons,
        df=df,
        output_path=str(parquet_path),
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Source function used: {func_name}")
    print(f"Row count: {len(df)}")
    print(f"Column count: {len(df.columns)}")
    print(f"Seasons requested: {seasons}")
    print(f"Seasons found: {seasons}")
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Output parquet path: {parquet_path}")
    print(f"Metadata JSON path: {metadata_path}")
