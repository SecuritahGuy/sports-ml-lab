"""Tests for QB-continuity refinement experiment."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sportslab.evaluation.qb_continuity import (
    ALL_SEASONS,
    FULL_CONTINUITY_COLS,
    HOLDOUT_SEASON,
    LIVE_SAFE_LABELS,
    MODEL_VARIANTS,
    MOV_COLS,
    N_BOOTSTRAP,
    NEW_QB_COLS,
    QB_CHANGE_COLS,
    RECOVERY_COLS,
    STARTS_COLS,
    TRAIN_SEASONS,
    _bootstrap_delta,
    _build_feature_matrix,
    _calibration_buckets,
    _compute_metrics,
    _confidence_buckets,
    _fit_platt,
    _qb_subset_analysis,
    _run_fitted_once,
    _run_rolling_simulation,
    _worst_predictions,
)


class TestConstants:
    def test_all_seasons_within_range(self):
        assert all(2021 <= s <= 2025 for s in ALL_SEASONS)
        assert sorted(ALL_SEASONS) == ALL_SEASONS

    def test_train_seasons_no_holdout(self):
        assert HOLDOUT_SEASON not in TRAIN_SEASONS

    def test_six_model_variants(self):
        assert len(MODEL_VARIANTS) == 6
        names = [v[0] for v in MODEL_VARIANTS]
        assert names == ["incumbent", "no_qb", "qb_minimal",
                         "qb_experience", "qb_recovery", "qb_full_small"]

    def test_incumbent_feature_cols(self):
        _, cols, _, _ = MODEL_VARIANTS[0]
        assert "home_qb_changed" in cols
        assert "away_qb_changed" in cols
        assert "home_rolling_mov_3" in cols
        assert "away_rolling_mov_3" in cols

    def test_no_qb_has_no_qb_features(self):
        _, cols, _, _ = MODEL_VARIANTS[1]
        assert all("qb" not in c for c in cols)

    def test_no_qb_has_mov_only(self):
        _, cols, _, _ = MODEL_VARIANTS[1]
        assert cols == MOV_COLS

    def test_qb_minimal_has_new_qb_flag(self):
        _, cols, _, _ = MODEL_VARIANTS[2]
        assert all(c in cols for c in NEW_QB_COLS)

    def test_qb_experience_has_starts(self):
        _, cols, _, _ = MODEL_VARIANTS[3]
        assert all(c in cols for c in STARTS_COLS)

    def test_qb_recovery_has_games_since(self):
        _, cols, _, _ = MODEL_VARIANTS[4]
        assert all(c in cols for c in RECOVERY_COLS)

    def test_qb_full_small_has_all(self):
        _, cols, _, _ = MODEL_VARIANTS[5]
        assert all(c in cols for c in FULL_CONTINUITY_COLS)

    def test_live_safety_labels(self):
        assert LIVE_SAFE_LABELS["incumbent"] == "research_oracle"
        assert LIVE_SAFE_LABELS["no_qb"] == "live_safe_no_qb"
        assert LIVE_SAFE_LABELS["qb_minimal"] == "research_oracle"
        assert LIVE_SAFE_LABELS["qb_experience"] == "research_oracle"
        assert LIVE_SAFE_LABELS["qb_recovery"] == "research_oracle"
        assert LIVE_SAFE_LABELS["qb_full_small"] == "research_oracle"

    def test_bootstrap_iterations(self):
        assert N_BOOTSTRAP == 1000

    def test_model_variant_structure(self):
        for name, cols, live_safety, desc in MODEL_VARIANTS:
            assert isinstance(name, str)
            assert isinstance(cols, list)
            assert isinstance(live_safety, str)
            assert isinstance(desc, str)
            assert len(cols) >= 2

    def test_qb_change_cols(self):
        assert "home_qb_changed" in QB_CHANGE_COLS
        assert "away_qb_changed" in QB_CHANGE_COLS


class TestComputeMetrics:
    def test_all_correct(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        m = _compute_metrics(y_true, y_prob)
        assert m["log_loss"] < 0.3
        assert m["accuracy"] == 1.0
        assert "brier" in m

    def test_all_wrong(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.1, 0.9])
        m = _compute_metrics(y_true, y_prob)
        assert m["accuracy"] == 0.0
        assert m["log_loss"] > 0.5

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
        assert mean_d > 0.5

    def test_sign_convention(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200)
        better = np.where(y == 1, 0.9, 0.1)
        worse = np.full(200, 0.5)
        mean_d, _, _ = _bootstrap_delta(y, better, worse, n_iter=100)
        assert mean_d > 0

    def test_sign_convention_reverse(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200)
        better = np.where(y == 1, 0.9, 0.1)
        worse = np.full(200, 0.5)
        mean_d, _, _ = _bootstrap_delta(y, worse, better, n_iter=100)
        assert mean_d < 0


class TestCalibrationBuckets:
    def test_basic_structure(self):
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.55, 0.45, 0.65, 0.35, 0.75,
                           0.85, 0.25, 0.95, 0.15, 0.05])
        buckets = _calibration_buckets(y_true, y_prob)
        assert isinstance(buckets, list)
        assert len(buckets) > 0
        for b in buckets:
            assert "bucket" in b
            assert "n" in b
            assert "mean_pred" in b
            assert "mean_actual" in b
            assert "cal_error" in b

    def test_nan_handling(self):
        y_true = np.array([1, np.nan, 0])
        y_prob = np.array([0.9, 0.5, 0.2])
        buckets = _calibration_buckets(y_true, y_prob)
        n = sum(b["n"] for b in buckets)
        assert n == 2


class TestConfidenceBuckets:
    def test_basic_structure(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.6, 0.4])
        buckets = _confidence_buckets(y_true, y_prob)
        assert isinstance(buckets, list)
        assert len(buckets) > 0
        for b in buckets:
            assert "bucket" in b
            assert "n" in b
            assert "log_loss" in b


class TestWorstPredictions:
    def test_returns_n_results(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.9, 0.1, 0.5])
        game_ids = np.array(["g1", "g2", "g3", "g4", "g5"])
        teams = np.array(["ATL", "ARI", "CHI", "DET", "GB"])
        worst = _worst_predictions(y_true, y_prob, game_ids, teams, n=3)
        assert len(worst) == 3

    def test_sorted_by_descending_contrib(self):
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.1, 0.1, 0.9])
        game_ids = np.array(["g1", "g2", "g3"])
        teams = np.array(["ATL", "ARI", "CHI"])
        worst = _worst_predictions(y_true, y_prob, game_ids, teams, n=3)
        contribs = [w["log_loss_contrib"] for w in worst]
        assert all(contribs[i] >= contribs[i + 1] for i in range(len(contribs) - 1))

    def test_nan_filtered(self):
        y_true = np.array([1, np.nan, 0])
        y_prob = np.array([0.1, 0.5, 0.9])
        game_ids = np.array(["g1", "g2", "g3"])
        teams = np.array(["ATL", "ARI", "CHI"])
        worst = _worst_predictions(y_true, y_prob, game_ids, teams, n=5)
        assert len(worst) == 2


class TestQBSubsetAnalysis:
    def test_empty_results(self):
        results = {"incumbent": {"df": pd.DataFrame(), "overall": {}}}
        subset = _qb_subset_analysis(results, qb_changed=True)
        assert isinstance(subset, dict)

    def test_with_qb_changes(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "home_win_actual": [1, 0, 1],
            "incumbent_prob": [0.6, 0.4, 0.7],
            "home_qb_changed": [1, 0, 1],
            "away_qb_changed": [0, 1, 0],
        })
        results = {"incumbent": {"df": df, "overall": {}}}
        subset = _qb_subset_analysis(results, qb_changed=True)
        assert subset["incumbent"]["num_games"] == 3

    def test_no_qb_change_games(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "home_win_actual": [1, 0, 1],
            "incumbent_prob": [0.6, 0.4, 0.7],
            "home_qb_changed": [1, 0, 1],
            "away_qb_changed": [0, 1, 0],
        })
        results = {"incumbent": {"df": df, "overall": {}}}
        subset = _qb_subset_analysis(results, qb_changed=False)
        assert subset["incumbent"]["num_games"] == 0

    def test_partial_qb_changes(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4"],
            "home_win_actual": [1, 0, 1, 0],
            "incumbent_prob": [0.6, 0.4, 0.7, 0.3],
            "home_qb_changed": [1, 0, 0, 1],
            "away_qb_changed": [0, 0, 1, 0],
        })
        results = {"incumbent": {"df": df, "overall": {}}}
        subset = _qb_subset_analysis(results, qb_changed=True)
        assert subset["incumbent"]["num_games"] == 3

    def test_non_qb_change_games(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4"],
            "home_win_actual": [1, 0, 1, 0],
            "incumbent_prob": [0.6, 0.4, 0.7, 0.3],
            "home_qb_changed": [1, 0, 0, 1],
            "away_qb_changed": [0, 0, 1, 0],
        })
        results = {"incumbent": {"df": df, "overall": {}}}
        subset = _qb_subset_analysis(results, qb_changed=False)
        assert subset["incumbent"]["num_games"] == 1  # g2 only (no QB changed)


class TestRollingSimulation:
    def test_no_eligible_games(self):
        df = pd.DataFrame({
            "season": [2021],
            "week": [1],
            "gameday": ["2021-09-12"],
            "model_eligible": [False],
            "is_neutral": [False],
        })
        result = _run_rolling_simulation(df, ["home_qb_changed"], label="test")
        assert result["num_games"] == 0
        assert result["overall"] == {}


class TestFittedOnce:
    def test_is_callable(self):
        assert callable(_run_fitted_once)


# --- Importability tests ---


def test_run_qb_continuity_importable():
    from sportslab.evaluation.qb_continuity import run_qb_continuity
    assert callable(run_qb_continuity)


def test_cli_importable():
    from sportslab.cli import qb_continuity_cmd
    assert callable(qb_continuity_cmd)


def test_rolling_simulation_importable():
    from sportslab.evaluation.qb_continuity import _run_rolling_simulation
    assert callable(_run_rolling_simulation)


def test_fitted_once_importable():
    from sportslab.evaluation.qb_continuity import _run_fitted_once
    assert callable(_run_fitted_once)
