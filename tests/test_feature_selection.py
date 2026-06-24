"""Tests for forward feature selection experiment."""

from sportslab.evaluation.feature_selection_experiment import (
    CANDIDATE_FEATURES,
    ROLLING_FOLDS,
)


class TestImport:
    def test_module_importable(self):
        from sportslab.evaluation import feature_selection_experiment

        assert hasattr(feature_selection_experiment, "run_feature_selection_experiment")

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

    def test_candidate_features_defined(self):
        assert len(CANDIDATE_FEATURES) > 0

    def test_candidate_feature_columns(self):
        for name, cols in CANDIDATE_FEATURES.items():
            assert len(cols) > 0
            assert isinstance(name, str)

    def test_fold_no_duplicates(self):
        all_val = [v for _, v in ROLLING_FOLDS]
        assert len(all_val) == len(set(all_val))

    def test_candidate_groups_distinct(self):
        names = list(CANDIDATE_FEATURES.keys())
        assert len(names) == len(set(names))
