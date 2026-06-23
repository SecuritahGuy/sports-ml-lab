"""Build the first NFL feature table from raw schedules data."""

from pathlib import Path

import pandas as pd

SPORTSLAB_MIN_SEASON = 2021

TARGET_COLUMN = "home_win"
MODEL_ELIGIBLE_COLUMN = "model_eligible"
TIE_COLUMN = "is_tie"
NEUTRAL_COLUMN = "is_neutral"

LEAKAGE_COLUMNS = [
    "away_score",
    "home_score",
    "result",
    "total",
    "overtime",
    TARGET_COLUMN,
    TIE_COLUMN,
]

MARKET_COLUMNS = [
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
]

WEATHER_COLUMNS = [
    "weather_temp",
    "weather_tmin",
    "weather_tmax",
    "weather_humidity",
    "weather_precip",
    "weather_wind_speed",
    "weather_pressure",
    "weather_cloud_cover",
]

SPARSE_ID_COLUMNS = [
    "old_game_id",
    "gsis",
    "nfl_detail_id",
    "pfr",
    "pff",
    "espn",
    "ftn",
]

PREGAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "location",
    "away_rest",
    "home_rest",
    "div_game",
    "roof",
    "surface",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
    "referee",
    "stadium_id",
    "stadium",
]

BASELINE_FEATURE_COLUMNS = [
    "season",
    "week",
    "away_team_enc",
    "home_team_enc",
    "away_rest",
    "home_rest",
    "rest_diff",
    "div_game",
    "is_dome",
    NEUTRAL_COLUMN,
    "away_qb_id_enc",
    "home_qb_id_enc",
    "away_coach_enc",
    "home_coach_enc",
    "stadium_id_enc",
    "game_type_enc",
    "weekday_enc",
    "roof_enc",
    "surface_enc",
]

SCHEDULING_FEATURE_COLUMNS = [
    "home_short_week",
    "away_short_week",
    "home_off_bye",
    "away_off_bye",
    "thursday_flag",
    "monday_flag",
    NEUTRAL_COLUMN,
    "is_international",
    "home_consecutive_road",
    "away_consecutive_road",
]

QB_FEATURE_COLUMNS = [
    "home_qb_changed",
    "away_qb_changed",
    "qb_change_diff",
    "home_qb_starts_this_season_pre",
    "away_qb_starts_this_season_pre",
    "qb_starts_diff",
    "home_qb_team_starts_pre",
    "away_qb_team_starts_pre",
    "home_qb_win_pct_pre",
    "away_qb_win_pct_pre",
    "qb_win_pct_diff",
    "home_games_since_qb_change",
    "away_games_since_qb_change",
    "games_since_qb_change_diff",
    "home_new_qb_flag",
    "away_new_qb_flag",
    "new_qb_diff",
    "home_qb_missing_flag",
    "away_qb_missing_flag",
]

QB_IDENTITY_COLUMNS = [
    "home_qb_id",
    "away_qb_id",
]

WEATHER_FEATURE_COLUMNS = [
    "temperature_f",
    "wind_mph",
    "precipitation_flag",
    "cold_flag",
    "very_cold_flag",
    "hot_flag",
    "windy_flag",
    "very_windy_flag",
    "bad_weather_flag",
    "outdoor_game_flag",
    "is_dome",
    "weather_missing_flag",
    "temp_missing_flag",
    "wind_missing_flag",
]

TEAM_STRENGTH_FEATURE_COLUMNS = [
    "season",
    "week",
    # Elo features
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff",
    # Rolling features
    "home_rolling_win_pct",
    "away_rolling_win_pct",
    "rolling_win_pct_diff",
    "home_rolling_point_diff",
    "away_rolling_point_diff",
    "rolling_point_diff_diff",
    "home_rolling_points_for",
    "away_rolling_points_for",
    "home_rolling_points_against",
    "away_rolling_points_against",
    # Rest
    "home_rest",
    "away_rest",
    "rest_diff",
    # Structural features (low-cardinality, OK label-encoded)
    "div_game",
    "is_dome",
    NEUTRAL_COLUMN,
    "game_type_enc",
    "weekday_enc",
    "roof_enc",
    "surface_enc",
]


def _validate_seasons(df: pd.DataFrame) -> None:
    """Ensure all rows have season >= SPORTSLAB_MIN_SEASON."""
    bad = df[df["season"] < SPORTSLAB_MIN_SEASON]
    if not bad.empty:
        raise ValueError(
            f"Found {len(bad)} rows with season < {SPORTSLAB_MIN_SEASON}. "
            "This project only supports 2021–current NFL seasons."
        )


def _is_dome(roof: str) -> int:
    """Return 1 if the roof type indicates an enclosed stadium."""
    return 1 if str(roof).lower() in ("dome", "closed") else 0


def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Simple integer label encoding for categorical identifiers."""
    out = df.copy()

    for col in [
        "away_team",
        "home_team",
        "away_qb_id",
        "home_qb_id",
        "away_coach",
        "home_coach",
        "referee",
        "stadium_id",
        "game_type",
        "weekday",
        "roof",
        "surface",
    ]:
        out[col + "_enc"] = out[col].astype("category").cat.codes

    return out


def build_feature_table(
    schedules_path: str = "data/raw/nfl/schedules.parquet",
    output_path: str = "data/features/nfl/feature_table.parquet",
    fetch_weather: bool = True,
) -> str:
    """Build the first feature table from raw NFL schedules.

    Creates a feature table that includes the target column (home_win),
    model eligibility flags, and both leakage and market columns preserved
    for audit.  Model training code should use BASELINE_FEATURE_COLUMNS
    and filter to model_eligible == True.

    Args:
        schedules_path: Path to the raw schedules parquet file.
        output_path: Where to save the feature table parquet.
        fetch_weather: If True, fetch real weather data via meteostat.

    Returns:
        The output path the feature table was written to.

    Raises:
        FileNotFoundError: If the schedules parquet does not exist.
        ValueError: If season validation fails.
    """
    schedules = Path(schedules_path)
    if not schedules.exists():
        raise FileNotFoundError(f"Schedules file not found: {schedules}")

    print(f"Reading schedules from: {schedules}")
    df = pd.read_parquet(schedules)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    _validate_seasons(df)
    print(f"  Seasons: {sorted(df['season'].unique())}")

    # --- Create target and flag columns from raw scores ---
    df[TARGET_COLUMN] = df.apply(
        lambda r: (
            1 if r.home_score > r.away_score else (0 if r.home_score < r.away_score else pd.NA)
        ),
        axis=1,
    )
    df[TIE_COLUMN] = df["home_score"] == df["away_score"]
    df[MODEL_ELIGIBLE_COLUMN] = df[TARGET_COLUMN].notna()
    df[NEUTRAL_COLUMN] = df["location"] == "Neutral"

    tie_count = df[TIE_COLUMN].sum()
    neutral_count = df[NEUTRAL_COLUMN].sum()
    print(f"  Ties: {tie_count}, Neutral-site: {neutral_count}")

    # --- Separate column groups from raw df ---
    pregame_cols = [c for c in PREGAME_COLUMNS if c in df.columns]
    leakage_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    market_cols = [c for c in MARKET_COLUMNS if c in df.columns]

    # --- Build features from pregame columns ---
    features = df[pregame_cols].copy()
    print(f"  Pregame columns retained: {len(pregame_cols)}")

    # Add derived columns (pregame only)
    features["is_dome"] = features["roof"].apply(_is_dome)
    features["rest_diff"] = features["home_rest"] - features["away_rest"]

    # Add target and flag columns
    for col in [TARGET_COLUMN, TIE_COLUMN, MODEL_ELIGIBLE_COLUMN, NEUTRAL_COLUMN]:
        features[col] = df[col]
    print("  Added target: home_win, flags: is_tie, model_eligible, is_neutral")

    # Optional: weather enrichment
    if fetch_weather:
        print("  Fetching weather data via meteostat...")
        from sportslab.features.weather import build_weather_features

        weather_df = build_weather_features(
            features["stadium_id"],
            pd.to_datetime(features["gameday"]),
        )
        features = pd.concat([features, weather_df], axis=1)
        print(f"  Weather columns added: {list(weather_df.columns)}")

    # Encode categoricals
    features = _encode_categorical(features)
    enc_count = len([c for c in features.columns if c.endswith("_enc")])
    print(f"  Encoded {enc_count} categorical columns")

    # Preserve leakage and market columns in the table (for audit)
    for col in leakage_cols + market_cols:
        if col not in features.columns:
            features[col] = df[col]

    actual_leakage = [c for c in leakage_cols if c in features.columns]
    actual_market = [c for c in market_cols if c in features.columns]
    print(f"  Leakage columns preserved in table: {actual_leakage}")
    print(f"  Market columns preserved in table: {actual_market}")

    baseline_available = [c for c in BASELINE_FEATURE_COLUMNS if c in features.columns]
    print(f"  Baseline feature columns available: {len(baseline_available)}")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    print(f"\nFeature table saved to: {out}")
    print(f"  Shape: {features.shape}")

    return str(out)
