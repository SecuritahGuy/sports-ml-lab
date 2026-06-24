"""Tests for AutoGluon experiment module."""

import importlib

import pandas as pd


def test_import_runnable():
    mod = importlib.import_module("sportslab.evaluation.autogluon_experiment")
    assert hasattr(mod, "run_autogluon_experiment")
    assert hasattr(mod, "ROLLING_FOLDS")
    assert hasattr(mod, "HOLDOUT_SEASON")


def test_fold_structure():
    from sportslab.evaluation.autogluon_experiment import ROLLING_FOLDS

    assert len(ROLLING_FOLDS) == 3
    for train_seasons, val_season in ROLLING_FOLDS:
        assert isinstance(train_seasons, list)
        assert isinstance(val_season, int)
        assert val_season not in train_seasons
        for s in train_seasons:
            assert isinstance(s, int)
            assert s >= 2021


def test_holdout_unchanged():
    from sportslab.evaluation.autogluon_experiment import HOLDOUT_SEASON, ROLLING_FOLDS

    assert HOLDOUT_SEASON == 2025
    for train_seasons, val_season in ROLLING_FOLDS:
        assert HOLDOUT_SEASON not in train_seasons
        assert HOLDOUT_SEASON != val_season


def test_feature_lists_complete():
    from sportslab.evaluation.autogluon_experiment import (
        BASIC_FEATURES,
        ELO_ONLY_FEATURES,
        QB_FLAG_FEATURES,
        SCHEDULING_FEATURES,
    )

    assert len(ELO_ONLY_FEATURES) >= 4
    assert len(SCHEDULING_FEATURES) >= 5
    assert len(QB_FLAG_FEATURES) >= 10
    assert len(BASIC_FEATURES) >= 5


def test_available_features_returns_known():
    from sportslab.evaluation.autogluon_experiment import _available_features

    df = pd.DataFrame(
        {
            "home_off_elo": [1500.0],
            "away_off_elo": [1500.0],
            "home_def_elo": [1500.0],
            "away_def_elo": [1500.0],
            "elo_diff": [0.0],
            "elo_prob": [0.5],
            "home_short_week": [0],
            "away_short_week": [0],
            "home_off_bye": [0],
            "away_off_bye": [0],
            "thursday_flag": [0],
            "monday_flag": [0],
            "is_international": [0],
            "home_consecutive_road": [0],
            "away_consecutive_road": [0],
            "home_qb_changed": [0],
            "away_qb_changed": [0],
            "week": [1],
            "rest_diff": [0],
            "div_game": [0],
            "is_dome": [0],
            "game_type_enc": [0],
            "roof_enc": [0],
            "surface_enc": [0],
            "weekday_enc": [0],
            "home_team_enc": [0],
            "away_team_enc": [0],
            "home_coach_enc": [0],
            "away_coach_enc": [0],
        }
    )
    avail = _available_features(df)
    assert len(avail) >= 10
    assert "home_off_elo" in avail
    assert "elo_prob" in avail
    assert "week" in avail


def test_available_features_weather_conditional():
    from sportslab.evaluation.autogluon_experiment import (
        WEATHER_FEATURES,
        _available_features,
    )

    df = pd.DataFrame(
        {
            "home_off_elo": [1500.0],
            "away_off_elo": [1500.0],
            "home_def_elo": [1500.0],
            "away_def_elo": [1500.0],
            "elo_diff": [0.0],
            "elo_prob": [0.5],
            "week": [1],
            "rest_diff": [0],
            "div_game": [0],
            "is_dome": [0],
            "game_type_enc": [0],
            "roof_enc": [0],
            "surface_enc": [0],
            "weekday_enc": [0],
            "home_team_enc": [0],
            "away_team_enc": [0],
            "home_coach_enc": [0],
            "away_coach_enc": [0],
        }
    )
    avail = _available_features(df)
    for wf in WEATHER_FEATURES:
        assert wf not in avail


def test_cli_importability():
    from sportslab.cli import autogluon_cmd

    assert autogluon_cmd is not None
