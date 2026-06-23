"""Tests for the feature engineering module — no network calls."""

import pandas as pd
import pytest

from sportslab.features.build_features import (
    BASELINE_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    MARKET_COLUMNS,
    NEUTRAL_COLUMN,
    PREGAME_COLUMNS,
    TARGET_COLUMN,
    WEATHER_COLUMNS,
    _encode_categorical,
    _is_dome,
    _validate_seasons,
)


class TestValidateSeasons:
    def test_invalid_seasons_raise(self):
        df = pd.DataFrame({"season": [2010, 2021]})
        with pytest.raises(ValueError, match="season < 2021"):
            _validate_seasons(df)

    def test_valid_seasons_pass(self):
        df = pd.DataFrame({"season": [2021, 2022, 2025]})
        _validate_seasons(df)  # should not raise


class TestIsDome:
    def test_dome(self):
        assert _is_dome("dome") == 1

    def test_closed(self):
        assert _is_dome("closed") == 1

    def test_outdoors(self):
        assert _is_dome("outdoors") == 0

    def test_open(self):
        assert _is_dome("open") == 0

    def test_case_insensitive(self):
        assert _is_dome("Dome") == 1
        assert _is_dome("Closed") == 1


class TestEncodeCategorical:
    def test_encoding_creates_enc_columns(self):
        df = pd.DataFrame(
            {
                "away_team": ["KC", "BUF", "KC"],
                "home_team": ["TB", "KC", "BUF"],
                "away_qb_id": ["qb1", "qb2", "qb1"],
                "home_qb_id": ["qb3", "qb1", "qb2"],
                "away_coach": ["c1", "c2", "c1"],
                "home_coach": ["c3", "c1", "c2"],
                "referee": ["r1", "r2", "r1"],
                "stadium_id": ["s1", "s2", "s1"],
                "game_type": ["REG", "REG", "WC"],
                "weekday": ["Sun", "Sun", "Mon"],
                "roof": ["outdoors", "dome", "outdoors"],
                "surface": ["grass", "turf", "grass"],
            }
        )
        result = _encode_categorical(df)
        assert "away_team_enc" in result.columns
        assert "home_team_enc" in result.columns
        # Same away_team value gets same code
        assert result["away_team_enc"].iloc[0] == result["away_team_enc"].iloc[2]


class TestColumnConstants:
    def test_leakage_columns(self):
        assert "home_score" in LEAKAGE_COLUMNS
        assert "away_score" in LEAKAGE_COLUMNS
        assert "result" in LEAKAGE_COLUMNS
        assert "total" in LEAKAGE_COLUMNS

    def test_market_columns(self):
        assert "home_moneyline" in MARKET_COLUMNS
        assert "spread_line" in MARKET_COLUMNS
        assert "total_line" in MARKET_COLUMNS

    def test_pregame_has_game_id(self):
        assert "game_id" in PREGAME_COLUMNS

    def test_leakage_not_in_pregame(self):
        for c in LEAKAGE_COLUMNS:
            assert c not in PREGAME_COLUMNS, f"{c} should not be in PREGAME_COLUMNS"

    def test_market_not_in_pregame(self):
        for c in MARKET_COLUMNS:
            assert c not in PREGAME_COLUMNS, f"{c} should not be in PREGAME_COLUMNS"


class TestTargetColumn:
    def test_home_win_exists(self):

        assert TARGET_COLUMN == "home_win"

    def test_home_win_one_when_home_wins(self):
        df = pd.DataFrame({"home_score": [24], "away_score": [10]})
        home_win = df.apply(
            lambda r: (
                1 if r.home_score > r.away_score else (0 if r.home_score < r.away_score else pd.NA)
            ),
            axis=1,
        )
        assert home_win.iloc[0] == 1

    def test_home_win_zero_when_away_wins(self):
        df = pd.DataFrame({"home_score": [10], "away_score": [24]})
        home_win = df.apply(
            lambda r: (
                1 if r.home_score > r.away_score else (0 if r.home_score < r.away_score else pd.NA)
            ),
            axis=1,
        )
        assert home_win.iloc[0] == 0

    def test_home_win_null_for_tie(self):
        df = pd.DataFrame({"home_score": [17], "away_score": [17]})
        home_win = df.apply(
            lambda r: (
                1 if r.home_score > r.away_score else (0 if r.home_score < r.away_score else pd.NA)
            ),
            axis=1,
        )
        assert home_win.isna().iloc[0]


class TestTieHandling:
    def test_is_tie_true_for_equal_scores(self):
        df = pd.DataFrame({"home_score": [17], "away_score": [17]})
        is_tie = df["home_score"] == df["away_score"]
        assert is_tie.iloc[0]

    def test_is_tie_false_for_different_scores(self):
        df = pd.DataFrame({"home_score": [24], "away_score": [10]})
        is_tie = df["home_score"] == df["away_score"]
        assert not is_tie.iloc[0]

    def test_model_eligible_false_for_tie(self):
        home_win = pd.Series([pd.NA])
        eligible = home_win.notna()
        assert not eligible.iloc[0]

    def test_model_eligible_true_for_non_tie(self):
        home_win = pd.Series([1])
        eligible = home_win.notna()
        assert eligible.iloc[0]


class TestNeutralHandling:
    def test_is_neutral_flag(self):
        assert NEUTRAL_COLUMN == "is_neutral"

    def test_pregame_has_location(self):
        assert "location" in PREGAME_COLUMNS


class TestColumnExclusions:
    def test_market_not_in_baseline(self):
        for c in MARKET_COLUMNS:
            assert c not in BASELINE_FEATURE_COLUMNS, (
                f"{c} should not be in BASELINE_FEATURE_COLUMNS"
            )

    def test_weather_not_in_baseline(self):
        for c in WEATHER_COLUMNS:
            assert c not in BASELINE_FEATURE_COLUMNS, (
                f"{c} should not be in BASELINE_FEATURE_COLUMNS"
            )

    def test_leakage_not_in_baseline(self):
        for c in LEAKAGE_COLUMNS:
            assert c not in BASELINE_FEATURE_COLUMNS, (
                f"{c} should not be in BASELINE_FEATURE_COLUMNS"
            )

    def test_baseline_has_encoded_team(self):
        assert "away_team_enc" in BASELINE_FEATURE_COLUMNS
        assert "home_team_enc" in BASELINE_FEATURE_COLUMNS


class TestBuildFeatureTable:
    def test_missing_file_raises(self):
        from sportslab.features.build_features import build_feature_table

        with pytest.raises(FileNotFoundError):
            build_feature_table(schedules_path="/nonexistent/path.parquet", fetch_weather=False)
