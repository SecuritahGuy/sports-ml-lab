"""Tests for residual blending experiment."""

from sportslab.evaluation.residual_blending_experiment import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    run_residual_blending_experiment,
)


class TestExperiment:
    def test_importable(self):
        assert callable(run_residual_blending_experiment)

    def test_folds_exclude_holdout(self):
        for train_s, val_s in ROLLING_FOLDS:
            all_seasons = list(train_s) + [val_s]
            assert HOLDOUT_SEASON not in all_seasons

    def test_folds_sequential(self):
        expected = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
        assert ROLLING_FOLDS == expected
