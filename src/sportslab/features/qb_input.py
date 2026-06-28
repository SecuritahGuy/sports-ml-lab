"""Live-safe QB input parsing for future predictions.

Allows users to provide pregame-announced QB starters via CSV,
overriding the oracle-based QB data from nflreadpy schedules.

Usage:
    python -c "
    from sportslab.features.qb_input import parse_qb_input_csv
    qb_df = parse_qb_input_csv('qb_input.csv')
    "
"""

import pandas as pd

QB_INPUT_COLUMNS = ["game_id", "home_qb_id", "away_qb_id"]


def parse_qb_input_csv(path: str) -> pd.DataFrame:
    """Parse a QB input CSV into a DataFrame.

    Expected columns:
        game_id: str — unique game identifier matching the feature table
        home_qb_id: str — home team's starting QB identifier (pregame-announced)
        away_qb_id: str — away team's starting QB identifier (pregame-announced)

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with columns: game_id, home_qb_id, away_qb_id.
        All values are strings; empty/missing mapped to pd.NA.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    df = pd.read_csv(path)
    missing = [c for c in QB_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"QB input CSV missing columns: {missing}. "
            f"Required: {QB_INPUT_COLUMNS}"
        )
    if df.empty:
        raise ValueError(f"QB input CSV is empty: {path}")
    for col in ["home_qb_id", "away_qb_id"]:
        df[col] = df[col].astype(str).replace(["nan", "", "None"], pd.NA)
    return df[QB_INPUT_COLUMNS]


def apply_qb_input(df: pd.DataFrame, qb_input_df: pd.DataFrame) -> pd.DataFrame:
    """Override oracle QB columns with live-safe QB input values.

    Replaces home_qb_id and away_qb_id in df with values from qb_input_df
    for matching game_ids. Non-matching rows keep their original values.

    Args:
        df: DataFrame with game_id, home_qb_id, away_qb_id columns.
        qb_input_df: DataFrame produced by parse_qb_input_csv.

    Returns:
        Copy of df with overridden QB columns for matching games.
    """
    out = df.copy()
    id_map = qb_input_df.set_index("game_id")
    match_mask = out["game_id"].isin(id_map.index)
    for col in ["home_qb_id", "away_qb_id"]:
        out.loc[match_mask, col] = out.loc[match_mask, "game_id"].map(
            id_map[col]
        )
    return out
