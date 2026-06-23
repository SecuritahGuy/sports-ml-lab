"""Tests for weather features — no leakage, dome-aware."""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.weather import (
    WEATHER_FEATURE_COLUMNS,
    compute_weather_features,
)


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "gameday": ["2024-09-05", "2024-09-12"],
            "home_team": ["KC", "KC"],
            "away_team": ["BAL", "LV"],
            "home_win": [1, 1],
            "home_score": [27, 24],
            "away_score": [20, 17],
            "weather_tmin": [18.0, 25.0],
            "weather_tmax": [30.0, 35.0],
            "weather_wind_speed": [15.0, 5.0],
            "weather_precip": [0.0, 5.0],
            "roof": ["outdoors", "outdoors"],
        }
    )


class TestWeatherConversion:
    def test_temperature_fahrenheit(self):
        df = _base_df()
        result = compute_weather_features(df)
        # avg(18, 30) = 24°C = 75.2°F
        assert result["temperature_f"].iloc[0] == pytest.approx(75.2, abs=0.1)
        # avg(25, 35) = 30°C = 86°F
        assert result["temperature_f"].iloc[1] == pytest.approx(86.0, abs=0.1)

    def test_wind_conversion(self):
        df = _base_df()
        result = compute_weather_features(df)
        # 15 km/h = 9.32 mph
        assert result["wind_mph"].iloc[0] == pytest.approx(9.32, abs=0.1)
        # 5 km/h = 3.11 mph
        assert result["wind_mph"].iloc[1] == pytest.approx(3.11, abs=0.1)

    def test_precipitation_flag(self):
        df = _base_df()
        result = compute_weather_features(df)
        assert result["precipitation_flag"].iloc[0] == 0  # 0.0 precip
        assert result["precipitation_flag"].iloc[1] == 1  # 5.0 precip > 0

    def test_cold_flag(self):
        df = _base_df()
        result = compute_weather_features(df)
        # 24°C = 75.2°F, not cold
        assert result["cold_flag"].iloc[0] == 0
        # 86°F, not cold
        assert result["cold_flag"].iloc[1] == 0

    def test_cold_flag_below_freezing(self):
        df = _base_df()
        df["weather_tmin"] = [-10.0, 25.0]
        df["weather_tmax"] = [-2.0, 35.0]
        result = compute_weather_features(df)
        # avg(-10, -2) = -6°C = 21.2°F, not cold (not <= 32F? Wait 21.2 <= 32, so cold)
        # Actually 21.2 ≤ 32 → cold_flag = 1
        assert result["cold_flag"].iloc[0] == 1
        # 86°F, not cold
        assert result["cold_flag"].iloc[1] == 0

    def test_very_cold_flag(self):
        df = _base_df()
        df["weather_tmin"] = [-18.0, 25.0]
        df["weather_tmax"] = [-10.0, 35.0]
        result = compute_weather_features(df)
        # avg(-18, -10) = -14°C = 6.8°F, very cold
        assert result["very_cold_flag"].iloc[0] == 1
        assert result["very_cold_flag"].iloc[1] == 0

    def test_hot_flag(self):
        df = _base_df()
        result = compute_weather_features(df)
        # 86°F >= 85°F
        assert result["hot_flag"].iloc[1] == 1
        assert result["hot_flag"].iloc[0] == 0

    def test_windy_flag(self):
        df = _base_df()
        result = compute_weather_features(df)
        # 9.32 mph < 15
        assert result["windy_flag"].iloc[0] == 0
        assert result["windy_flag"].iloc[1] == 0

    def test_windy_flag_above_threshold(self):
        df = _base_df()
        df["weather_wind_speed"] = [30.0, 5.0]
        result = compute_weather_features(df)
        # 30 km/h = 18.64 mph >= 15
        assert result["windy_flag"].iloc[0] == 1
        assert result["very_windy_flag"].iloc[0] == 0  # < 20 mph


class TestDomeHandling:
    def test_dome_game_neutralizes_weather(self):
        df = _base_df()
        df["roof"] = ["dome", "outdoors"]
        df["weather_tmin"] = [-10.0, 25.0]
        df["weather_tmax"] = [-2.0, 35.0]
        df["weather_wind_speed"] = [40.0, 5.0]
        df["weather_precip"] = [25.0, 0.0]
        result = compute_weather_features(df)
        # Dome: temp=70, wind=0, precip=0
        assert result["temperature_f"].iloc[0] == 70.0
        assert result["wind_mph"].iloc[0] == 0.0
        assert result["precipitation_flag"].iloc[0] == 0
        # Outdoor: unaffected
        assert result["temperature_f"].iloc[1] > 70.0
        assert result["is_dome"].iloc[0] == 1
        assert result["is_dome"].iloc[1] == 0

    def test_closed_roof_neutralizes(self):
        df = _base_df()
        df["roof"] = ["closed", "open"]
        result = compute_weather_features(df)
        assert result["is_dome"].iloc[0] == 1
        assert result["is_dome"].iloc[1] == 0
        assert result["temperature_f"].iloc[0] == 70.0

    def test_outdoor_game_flag(self):
        df = _base_df()
        df["roof"] = ["dome", "outdoors"]
        result = compute_weather_features(df)
        assert result["outdoor_game_flag"].iloc[0] == 0
        assert result["outdoor_game_flag"].iloc[1] == 1

    def test_open_roof_is_outdoor(self):
        df = _base_df()
        df["roof"] = ["open", "outdoors"]
        result = compute_weather_features(df)
        assert result["outdoor_game_flag"].iloc[0] == 1
        assert result["outdoor_game_flag"].iloc[1] == 1


class TestMissingValues:
    def test_missing_tmin_sets_missing_flag(self):
        df = _base_df()
        df["weather_tmin"] = [np.nan, 25.0]
        result = compute_weather_features(df)
        assert result["temp_missing_flag"].iloc[0] == 1
        assert result["temp_missing_flag"].iloc[1] == 0
        assert result["weather_missing_flag"].iloc[0] == 1
        assert result["weather_missing_flag"].iloc[1] == 0

    def test_missing_wind_sets_wind_flag(self):
        df = _base_df()
        df["weather_wind_speed"] = [np.nan, 5.0]
        result = compute_weather_features(df)
        assert result["wind_missing_flag"].iloc[0] == 1
        assert result["wind_missing_flag"].iloc[1] == 0
        assert result["weather_missing_flag"].iloc[0] == 1

    def test_missing_weather_does_not_crash(self):
        df = _base_df()
        df["weather_tmin"] = [np.nan, np.nan]
        df["weather_tmax"] = [np.nan, np.nan]
        df["weather_wind_speed"] = [np.nan, np.nan]
        df["weather_precip"] = [np.nan, np.nan]
        result = compute_weather_features(df)
        assert result["weather_missing_flag"].sum() == 2
        assert result["temperature_f"].isna().sum() == 2

    def test_bad_weather_flag_combines_conditions(self):
        df = _base_df()
        # Game 1: cold + precip
        df.loc[0, "weather_tmin"] = -5.0
        df.loc[0, "weather_tmax"] = 0.0
        df.loc[0, "weather_precip"] = 10.0
        df.loc[0, "weather_wind_speed"] = 5.0
        # Game 2: normal (no precip, no cold, no wind)
        df.loc[1, "weather_precip"] = 0.0
        result = compute_weather_features(df)
        assert result["bad_weather_flag"].iloc[0] == 1
        assert result["bad_weather_flag"].iloc[1] == 0


class TestAllColumns:
    def test_all_weather_feature_columns_present(self):
        df = _base_df()
        result = compute_weather_features(df)
        for col in WEATHER_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_missing_values_in_flags(self):
        """Flag columns should never have NaN values."""
        df = _base_df()
        df["weather_tmin"] = [np.nan, 25.0]
        result = compute_weather_features(df)
        for col in WEATHER_FEATURE_COLUMNS:
            if col not in ("temperature_f", "wind_mph"):
                assert result[col].isna().sum() == 0, f"NaN in column: {col}"
