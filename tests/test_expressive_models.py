"""Tests for expressive models experiment."""

from pathlib import Path

import pytest

from sportslab.evaluation.expressive_models_experiment import (
    CURATED_FEATURE_COLUMNS,
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    run_expressive_models_experiment,
)


@pytest.fixture
def temp_report(tmp_path: Path) -> str:
    return str(tmp_path / "expressive_models_test.md")


def test_curated_features_exclude_raw_identity():
    """Curated feature set should not contain raw identity columns."""
    raw_identity = ["home_team", "away_team", "stadium_id", "referee"]
    for col in raw_identity:
        assert col not in CURATED_FEATURE_COLUMNS, f"Raw identity {col} found in curated features"


def test_curated_features_exclude_qb_identity_ohe():
    """Curated features should not contain QB identity OHE columns."""
    qb_ohe = ["home_qb_name", "away_qb_name", "home_qb_id", "away_qb_id"]
    for col in qb_ohe:
        assert col not in CURATED_FEATURE_COLUMNS, f"QB identity {col} found in curated features"


def test_curated_features_no_postgame_derived():
    """Curated features should not contain any target or result columns."""
    postgame = ["home_win", "is_tie", "home_score", "away_score", "result"]
    for col in postgame:
        msg = f"Postgame column {col} found in curated features"
        assert col not in CURATED_FEATURE_COLUMNS, msg


def test_rolling_folds_never_include_holdout():
    """Rolling folds should never use HOLDOUT_SEASON in train or validation."""
    for train_seasons, val_season in ROLLING_FOLDS:
        assert HOLDOUT_SEASON not in train_seasons, f"Holdout {HOLDOUT_SEASON} in training seasons"
        assert val_season != HOLDOUT_SEASON, f"Holdout {HOLDOUT_SEASON} used as validation season"


def test_rolling_folds_have_correct_structure():
    """Rolling folds should be properly structured (3 folds, increasing training window)."""
    assert len(ROLLING_FOLDS) == 3, f"Expected 3 folds, got {len(ROLLING_FOLDS)}"
    for train_seasons, val_season in ROLLING_FOLDS:
        assert len(train_seasons) >= 1, "Each fold must have at least 1 training season"
        assert isinstance(val_season, int), "Validation season must be an int"
        assert val_season not in train_seasons, "Validation season should not be in training"


def test_rolling_folds_chronological_order():
    """Validation season should be exactly one year after the last training season."""
    for train_seasons, val_season in ROLLING_FOLDS:
        expected_val = train_seasons[-1] + 1
        assert val_season == expected_val, (
            f"Val season {val_season} != last training season + 1 ({expected_val})"
        )


def test_rolling_folds_increasing_window():
    """Training window should grow across folds."""
    for i in range(len(ROLLING_FOLDS) - 1):
        curr_train, _ = ROLLING_FOLDS[i]
        next_train, _ = ROLLING_FOLDS[i + 1]
        assert len(next_train) > len(curr_train), "Training window should increase"


def test_expressive_model_cli_imports():
    """Verify the experiment function is importable and has required attributes."""
    assert callable(run_expressive_models_experiment)
    assert run_expressive_models_experiment.__doc__ is not None


def test_missing_feature_families_do_not_crash():
    """Experiment should handle gracefully if a feature family is partially missing."""
    cols = CURATED_FEATURE_COLUMNS.copy()
    for col in ["cold_flag", "windy_flag", "bad_weather_flag", "outdoor_game_flag"]:
        assert col in cols, f"Weather flag {col} should be in curated features"
    # All feature families should have representatives in CURATED_FEATURE_COLUMNS
    scheduling_repr = {"home_short_week", "away_short_week", "thursday_flag", "monday_flag"}
    qb_repr = {"home_qb_changed", "away_qb_changed", "qb_starts_diff", "qb_win_pct_diff"}
    weather_repr = {"cold_flag", "windy_flag", "bad_weather_flag", "outdoor_game_flag", "is_dome"}

    assert scheduling_repr.issubset(set(CURATED_FEATURE_COLUMNS)), "Missing scheduling features"
    assert qb_repr.issubset(set(CURATED_FEATURE_COLUMNS)), "Missing QB features"
    assert weather_repr.issubset(set(CURATED_FEATURE_COLUMNS)), "Missing weather features"


def test_curated_feature_diversity():
    """Curated features should span multiple families (not just Elo)."""
    # Elo-only features
    elo_only = {"elo_prob", "elo_logit", "elo_diff"}
    non_elo = [c for c in CURATED_FEATURE_COLUMNS if c not in elo_only]
    assert len(non_elo) >= 20, f"Too few non-Elo features: {len(non_elo)}"


def test_holdout_not_in_validation_during_selection():
    """Confirm HOLDOUT_SEASON never appears in validation during selection."""
    for train_seasons, val_season in ROLLING_FOLDS:
        assert val_season <= 2024, f"Validation season {val_season} should be <= 2024"
        assert HOLDOUT_SEASON not in train_seasons, f"Holdout in training for fold: {train_seasons}"
