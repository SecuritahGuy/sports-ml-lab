"""Live-safe QB input parsing for future predictions.

Allows users to provide pregame-announced QB starters via CSV,
overriding the oracle-based QB data from nflreadpy schedules.

V1 format (backward compatible):
    game_id, home_qb_id, away_qb_id

V2 format (recommended):
    game_id, home_qb_id, away_qb_id, home_qb_name, away_qb_name,
    source, confidence, timestamp, notes
"""

import pandas as pd

QB_INPUT_COLUMNS_V1 = ["game_id", "home_qb_id", "away_qb_id"]

QB_INPUT_COLUMNS_V2 = QB_INPUT_COLUMNS_V1 + [
    "home_qb_name",
    "away_qb_name",
    "source",
    "confidence",
    "timestamp",
    "notes",
]

VALID_SOURCES = {
    "injury_report",
    "depth_chart",
    "coach_announcement",
    "roster_move",
    "beat_writer",
    "manual",
}

VALID_CONFIDENCE = {
    "confirmed",
    "probable",
    "questionable",
    "estimated",
}


def parse_qb_input_csv(path: str) -> pd.DataFrame:
    """Parse a QB input CSV into a DataFrame.

    Accepts both V1 (3-column) and V2 (9-column) formats.
    Missing optional v2 columns are filled with pd.NA.

    V1 columns:
        game_id: str
        home_qb_id: str
        away_qb_id: str

    V2 additional columns:
        home_qb_name: str — human-readable name
        away_qb_name: str — human-readable name
        source: str — one of injury_report, depth_chart, coach_announcement,
                      roster_move, beat_writer, manual
        confidence: str — one of confirmed, probable, questionable, estimated
        timestamp: str — ISO 8601 datetime when info was collected
        notes: str — optional free-text

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with all v2 columns.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    df = pd.read_csv(path)

    missing = [c for c in QB_INPUT_COLUMNS_V1 if c not in df.columns]
    if missing:
        raise ValueError(
            f"QB input CSV missing required columns: {missing}. "
            f"Required: {QB_INPUT_COLUMNS_V1}"
        )
    if df.empty:
        raise ValueError(f"QB input CSV is empty: {path}")

    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Check for duplicate game_ids
    dups = df["game_id"].duplicated()
    if dups.any():
        dup_ids = df.loc[dups, "game_id"].tolist()
        raise ValueError(
            f"Duplicate game_id(s) found in QB input CSV: {dup_ids}. "
            f"Each game_id must appear at most once."
        )

    for col in ["home_qb_id", "away_qb_id"]:
        df[col] = df[col].astype(str).replace(["nan", "", "None", "NaN", "N/A"], pd.NA)

    # Check for all-null QB ID columns
    for col in ["home_qb_id", "away_qb_id"]:
        if df[col].isna().all():
            raise ValueError(
                f"All {col} values are missing in QB input CSV. "
                f"Each game must have a valid QB identifier."
            )

    for col in QB_INPUT_COLUMNS_V2:
        if col not in df.columns:
            df[col] = pd.NA

    if "source" in df.columns:
        invalid = df["source"].dropna()[~df["source"].dropna().isin(VALID_SOURCES)]
        if not invalid.empty:
            raise ValueError(
                f"Invalid source values: {invalid.unique().tolist()}. "
                f"Valid: {sorted(VALID_SOURCES)}"
            )

    if "confidence" in df.columns:
        invalid = df["confidence"].dropna()[
            ~df["confidence"].dropna().isin(VALID_CONFIDENCE)
        ]
        if not invalid.empty:
            raise ValueError(
                f"Invalid confidence values: {invalid.unique().tolist()}. "
                f"Valid: {sorted(VALID_CONFIDENCE)}"
            )

    return df[QB_INPUT_COLUMNS_V2]


def apply_qb_input(df: pd.DataFrame, qb_input_df: pd.DataFrame) -> pd.DataFrame:
    """Override oracle QB columns with live-safe QB input values.

    Replaces home_qb_id, away_qb_id in df with values from qb_input_df
    for matching game_ids. Also overrides any v2 metadata columns present
    in qb_input_df. Non-matching rows keep their original values.

    Args:
        df: DataFrame with game_id, home_qb_id, away_qb_id columns.
        qb_input_df: DataFrame produced by parse_qb_input_csv.

    Returns:
        Copy of df with overridden QB columns for matching games.
    """
    out = df.copy()
    id_map = qb_input_df.set_index("game_id")
    match_mask = out["game_id"].isin(id_map.index)
    override_cols = [c for c in qb_input_df.columns if c != "game_id"]
    for col in override_cols:
        if col not in out.columns:
            out[col] = pd.NA
        out.loc[match_mask, col] = out.loc[match_mask, "game_id"].map(
            id_map[col]
        )
    return out
