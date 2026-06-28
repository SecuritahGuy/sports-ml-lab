"""Pregame weather features — no leakage, dome-aware.

Uses raw nflreadpy `temp` (°F) and `wind` (mph) columns when available,
with meteostat fallback for games that need external weather data.
"""

import numpy as np
import pandas as pd

DOME_ROOF_TYPES = {"dome", "closed"}
OUTDOOR_ROOF_TYPES = {"outdoors", "open"}
NEUTRAL_TEMP_F = 70.0
NEUTRAL_WIND_MPH = 0.0


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371


def _has_nflreadpy_cols(df: pd.DataFrame) -> bool:
    return "temp" in df.columns and "wind" in df.columns


def _has_meteostat_cols(df: pd.DataFrame) -> bool:
    needed = ["weather_tmin", "weather_wind_speed", "weather_precip"]
    return all(c in df.columns for c in needed)


def compute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pregame weather features from available weather data sources.

    Priority:
    1. nflreadpy `temp` (°F) and `wind` (mph) columns (already in raw schedules)
    2. Meteostat columns (weather_tmin, weather_tmax, weather_wind_speed,
       weather_precip) as fallback

    Handles dome/indoor games by neutralizing weather signal.
    Creates flags for cold, wind, precipitation, and bad weather.

    Args:
        df: Must contain columns: roof, one of (temp, wind) or
            (weather_tmin, weather_wind_speed, weather_precip).

    Returns:
        DataFrame with added weather feature columns.
    """
    out = df.copy()

    # Determine indoor/outdoor
    roof_str = out["roof"].astype(str).str.lower()
    out["is_dome"] = roof_str.isin(DOME_ROOF_TYPES).astype(int)
    out["outdoor_game_flag"] = roof_str.isin(OUTDOOR_ROOF_TYPES).astype(int)

    # ── Source temperature and wind ──
    if _has_nflreadpy_cols(out):
        out["temperature_f"] = out["temp"]
        out["wind_mph"] = out["wind"]
        out["precipitation_flag"] = 0
        out["temp_missing_flag"] = out["temp"].isna().astype(int)
        out["wind_missing_flag"] = out["wind"].isna().astype(int)
        out["weather_source"] = "nflreadpy"
    elif _has_meteostat_cols(out):
        temp_c = (out["weather_tmin"] + out.get("weather_tmax", out["weather_tmin"])) / 2.0
        out["temperature_f"] = temp_c.apply(lambda v: _c_to_f(v) if pd.notna(v) else None)
        out["wind_mph"] = out["weather_wind_speed"].apply(
            lambda v: _kmh_to_mph(v) if pd.notna(v) else None
        )
        out["precipitation_flag"] = (
            out["weather_precip"].notna() & (out["weather_precip"] > 0)
        ).astype(int)
        out["temp_missing_flag"] = out["weather_tmin"].isna().astype(int)
        out["wind_missing_flag"] = out["weather_wind_speed"].isna().astype(int)
        out["weather_source"] = "meteostat"
    else:
        out["temperature_f"] = np.nan
        out["wind_mph"] = np.nan
        out["precipitation_flag"] = 0
        out["temp_missing_flag"] = 1
        out["wind_missing_flag"] = 1
        out["weather_source"] = "none"

    out["weather_missing_flag"] = (
        out["temp_missing_flag"] | out["wind_missing_flag"]
    ).astype(int)

    # ── Dome/indoor: neutralize weather ──
    dome_mask = out["is_dome"] == 1
    out.loc[dome_mask, "temperature_f"] = NEUTRAL_TEMP_F
    out.loc[dome_mask, "wind_mph"] = NEUTRAL_WIND_MPH
    out.loc[dome_mask, "precipitation_flag"] = 0

    # ── Impute remaining NaN with dataset medians ──
    temp_med = out["temperature_f"].median(skipna=True)
    wind_med = out["wind_mph"].median(skipna=True)
    if pd.notna(temp_med):
        out["temperature_f"] = out["temperature_f"].fillna(temp_med)
    if pd.notna(wind_med):
        out["wind_mph"] = out["wind_mph"].fillna(wind_med)

    # ── Threshold flags ──
    temp_ok = out["temperature_f"].notna()
    out["cold_flag"] = (temp_ok & (out["temperature_f"] <= 32)).astype(int)
    out["very_cold_flag"] = (temp_ok & (out["temperature_f"] <= 20)).astype(int)
    out["hot_flag"] = (temp_ok & (out["temperature_f"] >= 85)).astype(int)
    wind_ok = out["wind_mph"].notna()
    out["windy_flag"] = (wind_ok & (out["wind_mph"] >= 15)).astype(int)
    out["very_windy_flag"] = (wind_ok & (out["wind_mph"] >= 20)).astype(int)

    # Bad weather: cold + wind + precip combined
    cold_or_windy = (out["cold_flag"] == 1) | (out["windy_flag"] == 1)
    has_precip = out["precipitation_flag"] == 1
    out["bad_weather_flag"] = (cold_or_windy | has_precip).astype(int)

    return out


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
    "weather_source",
]
