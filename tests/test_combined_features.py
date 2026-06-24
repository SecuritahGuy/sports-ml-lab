"""Tests for combined features experiment."""

from sportslab.evaluation.combined_features_experiment import (
    COACH_FEATURE_COLUMNS,
    ROLLING_FOLDS,
)


class TestImport:
    def test_module_importable(self):
        from sportslab.evaluation import combined_features_experiment

        assert hasattr(combined_features_experiment, "run_combined_experiment")

    def test_fold_structure(self):
        assert len(ROLLING_FOLDS) == 3

    def test_fold_seasons_valid(self):
        for train_seasons, val_season in ROLLING_FOLDS:
            for ts in train_seasons:
                assert ts >= 2021
            assert val_season >= 2022

    def test_holdout_not_in_folds(self):
        hold = 2025
        for train_seasons, val_season in ROLLING_FOLDS:
            assert val_season < hold
            for ts in train_seasons:
                assert ts < hold

    def test_coach_columns_present(self):
        assert len(COACH_FEATURE_COLUMNS) > 0
