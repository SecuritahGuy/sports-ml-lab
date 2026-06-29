"""Tests for QB ablation experiment."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sportslab.evaluation.qb_ablation import (
    ALL_SEASONS,
    HOLDOUT_SEASON,
    INCUMBENT_FEATURE_COLS,
    LIVE_QB_FEATURE_COLS,
    N_BOOTSTRAP,
    NO_QB_FEATURE_COLS,
    TRAIN_SEASONS,
    _bootstrap_delta,
    _build_feature_matrix,
    _compute_metrics,
    _fit_platt,
    _qb_change_subset_analysis,
)


class TestConstants:
    def test_incumbent_has_qb_features(self):
        assert any("qb" in c for c in INCUMBENT_FEATURE_COLS)

    def test_no_qb_has_no_qb_features(self):
        assert all("qb" not in c for c in NO_QB_FEATURE_COLS)

    def test_no_qb_subset_of_incumbent(self):
        assert set(NO_QB_FEATURE_COLS).issubset(set(INCUMBENT_FEATURE_COLS))

    def test_live_qb_matches_incumbent(self):
        assert LIVE_QB_FEATURE_COLS == INCUMBENT_FEATURE_COLS

    def test_all_seasons_within_range(self):
        assert all(2021 <= s <= 2025 for s in ALL_SEASONS)
        assert sorted(ALL_SEASONS) == ALL_SEASONS

    def test_train_seasons_no_holdout(self):
        assert HOLDOUT_SEASON not in TRAIN_SEASONS

    def test_bootstrap_iterations(self):
        assert N_BOOTSTRAP == 1000


class TestComputeMetrics:
    def test_all_correct(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        m = _compute_metrics(y_true, y_prob)
        assert m["log_loss"] < 0.3
        assert m["accuracy"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.1, 0.9])
        m = _compute_metrics(y_true, y_prob)
        assert m["accuracy"] == 0.0

    def test_empty(self):
        m = _compute_metrics(np.array([]), np.array([]))
        assert m == {}

    def test_handles_nan(self):
        y_true = np.array([1, np.nan, 0])
        y_prob = np.array([0.9, 0.5, 0.2])
        m = _compute_metrics(y_true, y_prob)
        assert m["accuracy"] == 1.0

    def test_clipped_extremes(self):
        y_true = np.array([1, 0])
        y_prob = np.array([0.0, 1.0])
        m = _compute_metrics(y_true, y_prob)
        assert np.isfinite(m["log_loss"])


class TestBuildFeatureMatrix:
    def test_elo_only(self):
        df = pd.DataFrame({"elo_prob": [0.5, 0.6, 0.7]})
        mat = _build_feature_matrix(df, [])
        assert mat.shape == (3, 1)
        assert np.allclose(mat.flatten(), [0.5, 0.6, 0.7])

    def test_with_features(self):
        df = pd.DataFrame({
            "elo_prob": [0.5, 0.6],
            "home_qb_changed": [0, 1],
            "away_qb_changed": [0, 0],
        })
        mat = _build_feature_matrix(df, ["home_qb_changed", "away_qb_changed"])
        assert mat.shape == (2, 3)

    def test_missing_features_ignored(self):
        df = pd.DataFrame({"elo_prob": [0.5]})
        mat = _build_feature_matrix(df, ["home_qb_changed", "home_rolling_mov_3"])
        assert mat.shape == (1, 1)


class TestFitPlatt:
    def test_basic_fit(self):
        x = np.array([[0.5], [0.6], [0.7], [0.3]])
        y = np.array([1, 1, 0, 0])
        pipe = _fit_platt(x, y)
        assert isinstance(pipe, Pipeline)
        prob = pipe.predict_proba(x)[:, 1]
        assert np.all((prob >= 0) & (prob <= 1))


class TestBootstrapDelta:
    def test_identical_models(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=100)
        p = rng.uniform(0.3, 0.7, size=100)
        mean_d, ci_l, ci_h = _bootstrap_delta(y, p, p, n_iter=100)
        assert abs(mean_d) < 0.01
        assert ci_l <= mean_d <= ci_h

    def test_perfect_vs_random(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200)
        perfect = y.astype(float)
        perfect = np.where(perfect == 1, 0.99, 0.01)
        random = np.full(200, 0.5)
        mean_d, ci_l, ci_h = _bootstrap_delta(y, perfect, random, n_iter=100)
        assert mean_d > 0.5  # random is worse, so delta > 0
        assert ci_l <= mean_d <= ci_h


class TestQBChangeSubsetAnalysis:
    def test_empty_results(self):
        results = {
            "incumbent": {"df": pd.DataFrame(), "overall": {}},
        }
        subset = _qb_change_subset_analysis(results)
        assert isinstance(subset, dict)

    def test_no_qb_change_games(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "home_win_actual": [1, 0],
            "incumbent_prob": [0.6, 0.4],
            "home_qb_changed": [0, 0],
            "away_qb_changed": [0, 0],
        })
        results = {"incumbent": {"df": df, "overall": {}}}
        subset = _qb_change_subset_analysis(results)
        assert subset["incumbent"]["num_games"] == 0

    def test_with_qb_changes(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "home_win_actual": [1, 0, 1],
            "incumbent_prob": [0.6, 0.4, 0.7],
            "home_qb_changed": [1, 0, 1],
            "away_qb_changed": [0, 1, 0],
        })
        results = {"incumbent": {"df": df, "overall": {}}}
        subset = _qb_change_subset_analysis(results)
        assert subset["incumbent"]["num_games"] == 3  # all have at least one QB change

    def test_partial_qb_changes(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4"],
            "home_win_actual": [1, 0, 1, 0],
            "incumbent_prob": [0.6, 0.4, 0.7, 0.3],
            "home_qb_changed": [1, 0, 0, 1],
            "away_qb_changed": [0, 0, 1, 0],
        })
        results = {"incumbent": {"df": df, "overall": {}}}
        subset = _qb_change_subset_analysis(results)
        assert subset["incumbent"]["num_games"] == 3  # g1, g3, g4


# Importability tests


def test_run_qb_ablation_importable():
    from sportslab.evaluation.qb_ablation import run_qb_ablation
    assert callable(run_qb_ablation)


def test_cli_importable():
    from sportslab.cli import qb_ablation_cmd
    assert callable(qb_ablation_cmd)


def test_rolling_simulation_importable():
    from sportslab.evaluation.qb_ablation import _run_rolling_ablation
    assert callable(_run_rolling_ablation)


def test_fitted_once_importable():
    from sportslab.evaluation.qb_ablation import _run_fitted_once
    assert callable(_run_fitted_once)
