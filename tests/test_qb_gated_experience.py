"""Tests for gated QB-experience experiment."""

import numpy as np
import pandas as pd

from sportslab.evaluation.experiment_utils import (
    bootstrap_delta,
    compute_metrics,
)
from sportslab.evaluation.qb_gated_experience import (
    ALL_SEASONS,
    FEATURE_COLS,
    GATED_AWAY_STARTS,
    GATED_HOME_STARTS,
    GATED_STARTS_COLS,
    GATED_STARTS_DIFF,
    HOLDOUT_SEASON,
    LIVE_SAFE_LABELS,
    MODEL_VARIANTS,
    MOV_COLS,
    QB_CHANGE_COLS,
    STARTS_COLS,
    TRAIN_SEASONS,
    _coefficient_diagnostics,
    _qb_side_breakdown,
    _qb_subset_analysis,
    _run_fitted_once,
    _run_rolling_simulation,
    _season_week_breakdown,
    compute_gated_columns,
)


class TestConstants:
    def test_all_seasons_within_range(self):
        assert all(2021 <= s <= 2025 for s in ALL_SEASONS)
        assert sorted(ALL_SEASONS) == ALL_SEASONS

    def test_train_seasons_no_holdout(self):
        assert HOLDOUT_SEASON not in TRAIN_SEASONS

    def test_five_model_variants(self):
        assert len(MODEL_VARIANTS) == 5
        names = [v[0] for v in MODEL_VARIANTS]
        assert names == [
            "incumbent",
            "qb_experience_global",
            "qb_experience_gated_binary",
            "qb_experience_gated_team_specific",
            "qb_experience_gated_simple_diff",
        ]

    def test_incumbent_feature_cols_unchanged(self):
        _, cols, _, _ = MODEL_VARIANTS[0]
        assert cols == FEATURE_COLS

    def test_incumbent_live_safety(self):
        assert LIVE_SAFE_LABELS["incumbent"] == "research_oracle"

    def test_no_market_features(self):
        for _, cols, _, _ in MODEL_VARIANTS:
            assert all("market" not in c for c in cols)

    def test_gated_starts_cols_defined(self):
        assert GATED_HOME_STARTS in GATED_STARTS_COLS
        assert GATED_AWAY_STARTS in GATED_STARTS_COLS
        assert GATED_STARTS_DIFF in GATED_STARTS_COLS

    def test_qb_experience_global_has_starts(self):
        _, cols, _, _ = MODEL_VARIANTS[1]
        assert all(c in cols for c in STARTS_COLS)
        assert all(c in cols for c in QB_CHANGE_COLS)
        assert all(c in cols for c in MOV_COLS)

    def test_gated_binary_has_gated_cols(self):
        _, cols, _, _ = MODEL_VARIANTS[2]
        assert all(c in cols for c in GATED_STARTS_COLS)
        assert "home_qb_starts_this_season_pre" not in cols
        assert "away_qb_starts_this_season_pre" not in cols

    def test_gated_team_specific_has_gated_cols(self):
        _, cols, _, _ = MODEL_VARIANTS[3]
        assert all(c in cols for c in GATED_STARTS_COLS)

    def test_gated_simple_diff_has_only_diff(self):
        _, cols, _, _ = MODEL_VARIANTS[4]
        assert GATED_STARTS_DIFF in cols
        assert GATED_HOME_STARTS not in cols
        assert GATED_AWAY_STARTS not in cols


class TestComputeMetrics:
    def test_all_correct(self):
        m = compute_metrics(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])
        )
        assert m["accuracy"] == 1.0
        assert m["log_loss"] < 0.3

    def test_all_wrong(self):
        m = compute_metrics(
            np.array([1, 0, 1, 0]), np.array([0.1, 0.9, 0.1, 0.9])
        )
        assert m["accuracy"] == 0.0

    def test_empty(self):
        assert compute_metrics(np.array([]), np.array([])) == {}

    def test_handles_nan(self):
        m = compute_metrics(np.array([1, np.nan, 0]), np.array([0.9, 0.5, 0.2]))
        assert m["accuracy"] == 1.0


class TestGatedColumns:
    def test_gated_binary_zeros_when_no_change(self):
        df = pd.DataFrame({
            "home_qb_changed": [0, 0],
            "away_qb_changed": [0, 0],
            "home_qb_starts_this_season_pre": [5, 0],
            "away_qb_starts_this_season_pre": [3, 2],
            "qb_starts_diff": [2, -2],
        })
        result = compute_gated_columns(df, "gated_binary")
        assert (result[GATED_HOME_STARTS] == 0).all()
        assert (result[GATED_AWAY_STARTS] == 0).all()
        assert (result[GATED_STARTS_DIFF] == 0).all()

    def test_gated_binary_preserves_when_change(self):
        df = pd.DataFrame({
            "home_qb_changed": [1, 0],
            "away_qb_changed": [0, 1],
            "home_qb_starts_this_season_pre": [5, 3],
            "away_qb_starts_this_season_pre": [3, 2],
            "qb_starts_diff": [2, 1],
        })
        result = compute_gated_columns(df, "gated_binary")
        assert result[GATED_HOME_STARTS].iloc[0] == 5
        assert result[GATED_AWAY_STARTS].iloc[0] == 3
        assert result[GATED_STARTS_DIFF].iloc[0] == 2
        # Second row: away changed, so gate is global
        assert result[GATED_HOME_STARTS].iloc[1] == 3
        assert result[GATED_AWAY_STARTS].iloc[1] == 2

    def test_team_specific_home_only(self):
        df = pd.DataFrame({
            "home_qb_changed": [1, 0],
            "away_qb_changed": [0, 0],
            "home_qb_starts_this_season_pre": [5, 3],
            "away_qb_starts_this_season_pre": [3, 2],
            "qb_starts_diff": [2, 1],
        })
        result = compute_gated_columns(df, "gated_team_specific")
        # Row 0: home changed, away unchanged
        assert result[GATED_HOME_STARTS].iloc[0] == 5  # home gate active
        assert result[GATED_AWAY_STARTS].iloc[0] == 0  # away gate inactive
        assert result[GATED_STARTS_DIFF].iloc[0] == 2  # global gate active
        # Row 1: no changes at all
        assert result[GATED_HOME_STARTS].iloc[1] == 0
        assert result[GATED_AWAY_STARTS].iloc[1] == 0
        assert result[GATED_STARTS_DIFF].iloc[1] == 0

    def test_team_specific_away_only(self):
        df = pd.DataFrame({
            "home_qb_changed": [0, 1],
            "away_qb_changed": [1, 0],
            "home_qb_starts_this_season_pre": [5, 3],
            "away_qb_starts_this_season_pre": [3, 2],
            "qb_starts_diff": [2, 1],
        })
        result = compute_gated_columns(df, "gated_team_specific")
        # First row: away only
        assert result[GATED_HOME_STARTS].iloc[0] == 0  # home gate inactive
        assert result[GATED_AWAY_STARTS].iloc[0] == 3  # away gate active
        # Second row: home only
        assert result[GATED_HOME_STARTS].iloc[1] == 3  # home gate active
        assert result[GATED_AWAY_STARTS].iloc[1] == 0  # away gate inactive

    def test_team_specific_both_changed(self):
        df = pd.DataFrame({
            "home_qb_changed": [1],
            "away_qb_changed": [1],
            "home_qb_starts_this_season_pre": [5],
            "away_qb_starts_this_season_pre": [3],
            "qb_starts_diff": [2],
        })
        result = compute_gated_columns(df, "gated_team_specific")
        assert result[GATED_HOME_STARTS].iloc[0] == 5
        assert result[GATED_AWAY_STARTS].iloc[0] == 3

    def test_gated_simple_diff_only(self):
        df = pd.DataFrame({
            "home_qb_changed": [1, 0],
            "away_qb_changed": [0, 1],
            "home_qb_starts_this_season_pre": [5, 3],
            "away_qb_starts_this_season_pre": [3, 2],
            "qb_starts_diff": [2, 1],
        })
        result = compute_gated_columns(df, "gated_simple_diff")
        # gated_simple_diff creates all gated columns; model variant only uses GATED_STARTS_DIFF
        assert result[GATED_STARTS_DIFF].iloc[0] == 2  # diff gated by global gate
        assert result[GATED_STARTS_DIFF].iloc[1] == 1  # away change → global gate active

    def test_gate_zero_when_no_qb_columns(self):
        df = pd.DataFrame({
            "home_qb_starts_this_season_pre": [5],
            "away_qb_starts_this_season_pre": [3],
            "qb_starts_diff": [2],
        })
        result = compute_gated_columns(df, "gated_binary")
        assert (result[GATED_HOME_STARTS] == 0).all()  # no gate columns → gate=0


class TestBootstrapDelta:
    def test_identical_models(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=100)
        p = rng.uniform(0.3, 0.7, size=100)
        mean_d, ci_l, ci_h = bootstrap_delta(y, p, p, n_iter=100)
        assert abs(mean_d) < 0.01
        assert ci_l <= mean_d <= ci_h

    def test_sign_convention(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200)
        better = np.where(y == 1, 0.9, 0.1)
        worse = np.full(200, 0.5)
        mean_d, _, _ = bootstrap_delta(y, better, worse, n_iter=100)
        assert mean_d > 0  # worse - better > 0


class TestQBSubsetAnalysis:
    def test_empty_results(self):
        subset = _qb_subset_analysis({"incumbent": {"df": pd.DataFrame()}}, True)
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
        subset = _qb_subset_analysis(results, True)
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
        subset = _qb_subset_analysis(results, False)
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
        subset = _qb_subset_analysis(results, False)
        assert subset["incumbent"]["num_games"] == 1  # g2 only


class TestQBSideBreakdown:
    def test_empty_df(self):
        result = _qb_side_breakdown(pd.DataFrame(), "prob")
        assert result == []

    def test_basic_breakdown(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4"],
            "home_win_actual": [1, 0, 1, 0],
            "incumbent_prob": [0.6, 0.4, 0.7, 0.3],
            "home_qb_changed": [1, 0, 0, 1],
            "away_qb_changed": [0, 1, 0, 0],
        })
        result = _qb_side_breakdown(df, "incumbent_prob")
        segments = {r["segment"]: r for r in result}
        assert "Neither QB changed" in segments
        assert segments["Neither QB changed"]["n"] == 1  # g3 only
        assert "Home QB changed" in segments
        assert segments["Home QB changed"]["n"] == 2  # g1, g4


class TestSeasonWeekBreakdown:
    def test_empty_df(self):
        result = _season_week_breakdown(pd.DataFrame(), "prob")
        assert result == []

    def test_basic(self):
        df = pd.DataFrame({
            "season": [2021, 2021, 2022],
            "week": [2, 3, 1],
            "home_win_actual": [1, 0, 1],
            "incumbent_prob": [0.6, 0.4, 0.7],
        })
        result = _season_week_breakdown(df, "incumbent_prob")
        assert len(result) >= 2  # at least 2 seasons


class TestCoefficientDiagnostics:
    def test_empty_results(self):
        diag = _coefficient_diagnostics({})
        assert diag == {}

    def test_basic(self):
        results = {
            "test": {
                "coef": np.array([0.5, -0.3, 0.1]),
                "feature_names": ["elo_prob", "feat_a", "feat_b"],
                "coefs": [],
            }
        }
        diag = _coefficient_diagnostics(results)
        assert "test" in diag
        assert len(diag["test"]["features"]) == 3
        assert diag["test"]["features"][0]["feature"] == "elo_prob"

    def test_with_rolling_coefs(self):
        results = {
            "test": {
                "coef": np.array([0.5, -0.3]),
                "feature_names": ["elo_prob", "feat_a"],
                "coefs": [
                    {"season": 2021, "week": 1, "coef": np.array([0.5, -0.3])},
                    {"season": 2021, "week": 2, "coef": np.array([0.6, -0.2])},
                ],
            }
        }
        diag = _coefficient_diagnostics(results)
        assert diag["test"]["n_rolling_models"] == 2


class TestRollingSimulation:
    def test_no_eligible_games(self):
        df = pd.DataFrame({
            "season": [2021], "week": [1],
            "gameday": ["2021-09-12"],
            "model_eligible": [False],
            "is_neutral": [False],
        })
        result = _run_rolling_simulation(df, ["home_qb_changed"], label="test")
        assert result["num_games"] == 0
        assert result["overall"] == {}

    def test_returns_coefs_key(self):
        df = pd.DataFrame({
            "season": [2021], "week": [1],
            "gameday": ["2021-09-12"],
            "model_eligible": [False],
            "is_neutral": [False],
        })
        result = _run_rolling_simulation(df, ["home_qb_changed"], label="test")
        assert "coefs" in result
        assert "feature_names" in result


class TestFittedOnce:
    def test_is_callable(self):
        assert callable(_run_fitted_once)


# --- Importability tests ---


def test_run_qb_gated_experience_importable():
    from sportslab.evaluation.qb_gated_experience import run_qb_gated_experience
    assert callable(run_qb_gated_experience)


def test_cli_importable():
    from sportslab.cli import qb_gated_experience_cmd
    assert callable(qb_gated_experience_cmd)


def test_rolling_simulation_importable():
    from sportslab.evaluation.qb_gated_experience import _run_rolling_simulation
    assert callable(_run_rolling_simulation)


def test_fitted_once_importable():
    from sportslab.evaluation.qb_gated_experience import _run_fitted_once
    assert callable(_run_fitted_once)


def test_compute_gated_columns_importable():
    from sportslab.evaluation.qb_gated_experience import compute_gated_columns
    assert callable(compute_gated_columns)


def test_no_promotion_effects():
    """Verify that importing/using the gated experiment doesn't
    alter incumbent constants or import side effects."""
    from sportslab.evaluation.predict_incumbent import FEATURE_COLS as INC_FEATURES
    from sportslab.evaluation.predict_incumbent import INCUMBENT_HOLDOUT_LL
    from sportslab.evaluation.qb_gated_experience import FEATURE_COLS as GATED_FEATURES
    assert list(INC_FEATURES) == list(GATED_FEATURES)
    assert INCUMBENT_HOLDOUT_LL == 0.6262


class TestDeterministic:
    def test_bootstrap_seed(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=50)
        p = np.full(50, 0.5)
        d1, _, _ = bootstrap_delta(y, p, p, n_iter=100, seed=42)
        d2, _, _ = bootstrap_delta(y, p, p, n_iter=100, seed=42)
        assert d1 == d2

    def test_metrics_deterministic(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.1, 0.8, 0.2])
        m1 = compute_metrics(y, p)
        m2 = compute_metrics(y, p)
        assert m1 == m2


class TestTieExclusion:
    def test_metrics_ignores_ties(self):
        y_true = np.array([1, np.nan, 0, np.nan])
        y_prob = np.array([0.9, 0.5, 0.2, 0.5])
        m = compute_metrics(y_true, y_prob)
        assert m["accuracy"] == 1.0  # only non-NaN games count
