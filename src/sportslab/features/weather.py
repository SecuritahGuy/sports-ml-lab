"""Weather data enrichment using meteostat."""

from datetime import date, datetime
from typing import Optional

import pandas as pd
from meteostat import Daily, Stations

from sportslab.features.stadiums import STADIUM_COORDS


def _get_station_id(lat: float, lon: float) -> Optional[str]:
    """Find the nearest meteostat weather station for a coordinate pair."""
    nearby = Stations().nearby(lat, lon, radius=50000).fetch()
    if nearby.empty:
        return None
    return str(nearby.index[0])


def fetch_weather(
    stadium_id: str,
    game_date: date,
) -> Optional[pd.Series]:
    """Fetch daily weather for a stadium on a given date.

    Returns a Series with meteostat weather columns, or None if
    the stadium is not mapped or weather data is unavailable.
    """
    coords = STADIUM_COORDS.get(stadium_id)
    if coords is None:
        return None

    lat, lon, tz = coords
    station_id = _get_station_id(lat, lon)
    if station_id is None:
        return None

    start = datetime(game_date.year, game_date.month, game_date.day)
    end = start

    ts = Daily(station_id, start, end)
    df = ts.fetch()
    if df is None or df.empty:
        return None

    row = df.iloc[0]
    return pd.Series(
        {
            "weather_temp": row.get("temp"),
            "weather_tmin": row.get("tmin"),
            "weather_tmax": row.get("tmax"),
            "weather_humidity": float(row["rhum"]) if pd.notna(row.get("rhum")) else None,
            "weather_precip": row.get("prcp"),
            "weather_wind_speed": row.get("wspd"),
            "weather_pressure": row.get("pres"),
            "weather_cloud_cover": float(row["cldc"]) if pd.notna(row.get("cldc")) else None,
        }
    )


def build_weather_features(
    stadium_ids: pd.Series,
    game_dates: pd.Series,
) -> pd.DataFrame:
    """Build a weather feature DataFrame for a set of stadiums and dates.

    Returns a DataFrame indexed identically to the input Series,
    with meteostat weather columns added.
    """
    results = []
    errors = 0
    for idx in range(len(stadium_ids)):
        sid = stadium_ids.iloc[idx]
        gd = game_dates.iloc[idx]
        if isinstance(gd, pd.Timestamp):
            gd = gd.date()
        w = fetch_weather(sid, gd)
        if w is not None:
            results.append(w)
        else:
            errors += 1
            results.append(
                pd.Series(
                    {
                        "weather_temp": None,
                        "weather_tmin": None,
                        "weather_tmax": None,
                        "weather_humidity": None,
                        "weather_precip": None,
                        "weather_wind_speed": None,
                        "weather_pressure": None,
                        "weather_cloud_cover": None,
                    }
                )
            )

    if errors:
        print(f"  Weather fetch completed with {errors} missing entries")

    return pd.DataFrame(results, index=stadium_ids.index)
