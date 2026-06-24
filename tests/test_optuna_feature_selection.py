"""Tests for Optuna feature selection experiment."""

from sportslab.evaluation.optuna_feature_selection_experiment import (
    FEATURE_GROUPS,
    ROLLING_FOLDS,
)


class TestImport:
    def test_module_importable(self):
        from sportslab.evaluation import optuna_feature_selection_experiment

        assert hasattr(optuna_feature_selection_experiment, "run_optuna_feature_selection")

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

    def test_feature_groups_defined(self):
        assert len(FEATURE_GROUPS) >= 14

    def test_feature_groups_include_known(self):
        assert "qb_changed" in FEATURE_GROUPS
        assert "rolling_mov_3" in FEATURE_GROUPS
        assert "rolling_mov_5" in FEATURE_GROUPS
        assert "coach_tenure" in FEATURE_GROUPS
        assert "turf_flag" in FEATURE_GROUPS

    def test_each_group_has_columns(self):
        for name, cols in FEATURE_GROUPS.items():
            assert len(cols) > 0, f"Group {name} has no columns"
            assert isinstance(name, str)

    def test_group_count_matches(self):
        assert len(FEATURE_GROUPS) == 16

    def test_no_duplicate_groups(self):
        names = list(FEATURE_GROUPS.keys())
        assert len(names) == len(set(names))
