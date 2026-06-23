"""Tests for confidence calibration and probability shrinkage module."""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.confidence_calibration_experiment import (
    CLIP_BOUNDS,
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    SHRINK_STRENGTHS,
    TEMPERATURES,
    _filter_df,
    clip_probabilities,
    early_season_shrink,
    high_confidence_shrink,
    run_confidence_calibration_experiment,
    run_grid_search,
    shrink_to_prior,
    temperature_scale,
)


class TestClipProbabilities:
    def test_clip_lower_bound(self):
        p = np.array([-0.1, 0.0, 0.01, 0.5, 1.0, 1.1])
        result = clip_probabilities(p, lo=0.01, hi=0.99)
        assert result.min() >= 0.01
        assert result.max() <= 0.99

    def test_clip_respects_bounds(self):
        p = np.array([0.0, 0.5, 1.0])
        for lo, hi in CLIP_BOUNDS:
            result = clip_probabilities(p, lo=lo, hi=hi)
            assert result[0] == lo
            assert result[1] == 0.5
            assert result[2] == hi

    def test_clip_midpoint_unchanged(self):
        p = np.array([0.3, 0.5, 0.7])
        result = clip_probabilities(p, lo=0.01, hi=0.99)
        np.testing.assert_array_equal(result, p)


class TestTemperatureScaling:
    def test_temperature_one_no_change(self):
        p = np.array([0.1, 0.5, 0.9])
        result = temperature_scale(p, temperature=1.0)
        np.testing.assert_array_almost_equal(result, p)

    def test_temperature_greater_one_softens(self):
        p = np.array([0.1, 0.9])
        result = temperature_scale(p, temperature=1.5)
        assert result[0] > 0.1  # low prob moves up
        assert result[1] < 0.9  # high prob moves down

    def test_temperature_returns_valid_probs(self):
        p = np.array([0.01, 0.99])
        for t in TEMPERATURES:
            result = temperature_scale(p, temperature=t)
            assert result.min() >= 0.0
            assert result.max() <= 1.0

    def test_temperature_zero_raises(self):
        p = np.array([0.5])
        with pytest.raises(ValueError, match="must be > 0"):
            temperature_scale(p, temperature=0)

    def test_temperature_negative_raises(self):
        p = np.array([0.5])
        with pytest.raises(ValueError, match="must be > 0"):
            temperature_scale(p, temperature=-1)


class TestShrinkToPrior:
    def test_shrink_moves_toward_prior(self):
        p = np.array([0.9, 0.5, 0.1])
        result = shrink_to_prior(p, alpha=0.1, prior=0.5)
        assert result[0] < 0.9  # moved down
        assert result[1] == 0.5  # unchanged
        assert result[2] > 0.1  # moved up

    def test_alpha_zero_no_change(self):
        p = np.array([0.1, 0.9])
        result = shrink_to_prior(p, alpha=0.0, prior=0.5)
        np.testing.assert_array_equal(result, p)

    def test_alpha_one_equals_prior(self):
        p = np.array([0.1, 0.5, 0.9])
        result = shrink_to_prior(p, alpha=1.0, prior=0.5)
        np.testing.assert_array_equal(result, [0.5, 0.5, 0.5])

    def test_alpha_out_of_range_raises(self):
        p = np.array([0.5])
        with pytest.raises(ValueError):
            shrink_to_prior(p, alpha=-0.1, prior=0.5)
        with pytest.raises(ValueError):
            shrink_to_prior(p, alpha=1.1, prior=0.5)

    def test_shrink_with_home_prior(self):
        p = np.array([0.8, 0.3])
        result = shrink_to_prior(p, alpha=0.1, prior=0.548)
        expected = (1 - 0.1) * p + 0.1 * 0.548
        np.testing.assert_array_almost_equal(result, expected)

    def test_shrink_applies_correctly(self):
        p = np.array([0.8, 0.5, 0.2])
        for a in SHRINK_STRENGTHS:
            result = shrink_to_prior(p, alpha=a, prior=0.5)
            expected = (1 - a) * p + a * 0.5
            np.testing.assert_array_almost_equal(result, expected)


class TestHighConfidenceShrink:
    def test_extreme_probs_shrunk(self):
        p = np.array([0.02, 0.5, 0.98])
        result = high_confidence_shrink(
            p, alpha=0.1, threshold_lo=0.10, threshold_hi=0.90, prior=0.5
        )
        assert result[0] > 0.02  # low extreme moved up
        assert result[1] == 0.5  # mid unchanged
        assert result[2] < 0.98  # high extreme moved down

    def test_mid_probs_unchanged(self):
        p = np.array([0.2, 0.5, 0.8])
        result = high_confidence_shrink(
            p, alpha=0.1, threshold_lo=0.10, threshold_hi=0.90, prior=0.5
        )
        np.testing.assert_array_equal(result, p)

    def test_boundary_values(self):
        p = np.array([0.10, 0.90])
        result = high_confidence_shrink(
            p, alpha=0.1, threshold_lo=0.10, threshold_hi=0.90, prior=0.5
        )
        # Exactly at boundary should be shrunk (<= lo or >= hi)
        # Actually, the mask is: p <= 0.10 OR p >= 0.90
        assert result[0] != 0.10  # was shrunk
        assert result[1] != 0.90  # was shrunk

    def test_different_thresholds(self):
        p = np.array([0.06, 0.09, 0.5, 0.91, 0.94])
        for lo, hi in [(0.10, 0.90), (0.08, 0.92), (0.05, 0.95)]:
            result = high_confidence_shrink(
                p, alpha=0.1, threshold_lo=lo, threshold_hi=hi, prior=0.5
            )
            extreme_mask = (p <= lo) | (p >= hi)
            mid_mask = ~extreme_mask
            if extreme_mask.sum() > 0:
                assert not np.allclose(result[extreme_mask], p[extreme_mask])
            if mid_mask.sum() > 0:
                np.testing.assert_array_equal(result[mid_mask], p[mid_mask])


class TestEarlySeasonShrink:
    def test_early_weeks_shrunk(self):
        p = np.array([0.9, 0.1, 0.5])
        weeks = np.array([1, 3, 8])
        result = early_season_shrink(p, weeks, alpha_early=0.1, alpha_late=0.0, prior=0.5)
        assert result[0] < 0.9  # week 1, shrunk
        assert result[1] > 0.1  # week 3, shrunk
        assert result[2] == 0.5  # week 8, not shrunk

    def test_late_weeks_unchanged(self):
        p = np.array([0.9, 0.1])
        weeks = np.array([5, 8])
        result = early_season_shrink(p, weeks, alpha_early=0.1, alpha_late=0.0, prior=0.5)
        np.testing.assert_array_equal(result, p)

    def test_all_weeks_early(self):
        p = np.array([0.9, 0.1])
        weeks = np.array([1, 4])
        result = early_season_shrink(p, weeks, alpha_early=0.2, alpha_late=0.0, prior=0.5)
        expected = (1 - 0.2) * p + 0.2 * 0.5
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_weeks(self):
        p = np.array([0.9, 0.5, 0.1])
        weeks = np.array([1, 5, 4])
        result = early_season_shrink(p, weeks, alpha_early=0.1, alpha_late=0.0, prior=0.5)
        assert result[0] < 0.9  # week 1 early
        assert result[1] == 0.5  # week 5 late
        assert result[2] > 0.1  # week 4 early


class TestRollingFolds:
    def test_folds_start_at_2021(self):
        for train_seasons, val_season in ROLLING_FOLDS:
            for s in train_seasons:
                assert s >= 2021
            assert val_season >= 2022

    def test_holdout_is_2025(self):
        assert HOLDOUT_SEASON == 2025

    def test_folds_sequential(self):
        expected = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
        assert ROLLING_FOLDS == expected

    def test_holdout_not_in_folds(self):
        all_fold_seasons = set()
        for train_s, val_s in ROLLING_FOLDS:
            all_fold_seasons.update(train_s)
            all_fold_seasons.add(val_s)
        assert HOLDOUT_SEASON not in all_fold_seasons


class TestFilterDF:
    def test_filters_non_eligible(self):
        df = pd.DataFrame(
            {
                "model_eligible": [True, False],
                "is_neutral": [False, False],
                "val": [1, 2],
            }
        )
        result = _filter_df(df)
        assert len(result) == 1

    def test_filters_neutral(self):
        df = pd.DataFrame(
            {
                "model_eligible": [True, True],
                "is_neutral": [False, True],
                "val": [1, 2],
            }
        )
        result = _filter_df(df)
        assert len(result) == 1


class TestGridSearch:
    def test_grid_search_returns_keys(self):
        rng = np.random.RandomState(42)
        n = 200
        p = rng.uniform(0.3, 0.7, n)
        y = (rng.uniform(0, 1, n) < p).astype(float)
        weeks = rng.randint(1, 18, n)
        seasons = np.array([2021] * 50 + [2022] * 50 + [2023] * 50 + [2024] * 50)
        result = run_grid_search(p, y, weeks=weeks, seasons=seasons)
        assert "best_key" in result
        assert "best_avg_val_ll" in result
        assert "all_avg" in result
        assert "all_results" in result

    def test_best_is_minimum(self):
        rng = np.random.RandomState(42)
        n = 200
        p = rng.uniform(0.3, 0.7, n)
        y = (rng.uniform(0, 1, n) < p).astype(float)
        weeks = rng.randint(1, 18, n)
        seasons = np.array([2021] * 50 + [2022] * 50 + [2023] * 50 + [2024] * 50)
        result = run_grid_search(p, y, weeks=weeks, seasons=seasons)
        best_avg = result["best_avg_val_ll"]
        all_avgs = result["all_avg"]["avg_val_log_loss"].values
        assert best_avg <= all_avgs.min() + 1e-10

    def test_no_holdout_in_results(self):
        rng = np.random.RandomState(42)
        n = 200
        p = rng.uniform(0.3, 0.7, n)
        y = (rng.uniform(0, 1, n) < p).astype(float)
        weeks = rng.randint(1, 18, n)
        seasons = np.array([2021] * 50 + [2022] * 50 + [2023] * 50 + [2024] * 50)
        result = run_grid_search(p, y, weeks=weeks, seasons=seasons)
        folds = result["all_results"]["fold"].unique()
        assert HOLDOUT_SEASON not in folds

    def test_baseline_included(self):
        rng = np.random.RandomState(42)
        n = 200
        p = rng.uniform(0.3, 0.7, n)
        y = (rng.uniform(0, 1, n) < p).astype(float)
        seasons = np.array([2021] * 50 + [2022] * 50 + [2023] * 50 + [2024] * 50)
        result = run_grid_search(p, y, weeks=None, seasons=seasons)
        assert "baseline_raw_elo" in result["all_results"]["method"].values


class TestExperimentImport:
    def test_run_experiment_importable(self):
        assert callable(run_confidence_calibration_experiment)

    def test_module_has_constants(self):
        assert len(CLIP_BOUNDS) > 0
        assert len(TEMPERATURES) > 0
        assert len(SHRINK_STRENGTHS) > 0
