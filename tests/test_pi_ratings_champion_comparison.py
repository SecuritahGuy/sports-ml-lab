"""Tests for Pi-Ratings champion comparison module."""

import numpy as np
import pandas as pd

from sportslab.evaluation.pi_ratings_champion_comparison import (
    MIN_PROMOTION_DELTA,
    PI_PARAMS,
    QB_GATE_CAP,
    QB_GATE_GAMMA,
    V3_PARAMS,
    _apply_qb_overlay,
    _build_qb_gate_mask,
    _logit,
    _score_model,
    _sigmoid,
)


class TestMathHelpers:
    def test_sigmoid_zero(self):
        assert _sigmoid(np.array([0.0]))[0] == 0.5

    def test_sigmoid_symmetry(self):
        x = np.array([-1.0, 0.5, 2.0])
        assert np.allclose(_sigmoid(x), 1 - _sigmoid(-x))

    def test_logit_sigmoid_inverse(self):
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(_sigmoid(_logit(p)), p)

    def test_logit_extremes(self):
        p = np.array([1e-7, 0.5, 1 - 1e-7])
        result = _logit(p)
        assert np.all(np.isfinite(result))
        assert result[0] < result[1] < result[2]


class TestQBGateMask:
    def test_no_changed_no_starts(self):
        df = pd.DataFrame({
            "home_qb_changed": [0],
            "away_qb_changed": [0],
        })
        mask = _build_qb_gate_mask(df)
        assert not mask[0]

    def test_home_changed(self):
        df = pd.DataFrame({
            "home_qb_changed": [1],
            "away_qb_changed": [0],
        })
        mask = _build_qb_gate_mask(df)
        assert mask[0]

    def test_away_changed(self):
        df = pd.DataFrame({
            "home_qb_changed": [0],
            "away_qb_changed": [1],
        })
        mask = _build_qb_gate_mask(df)
        assert mask[0]

    def test_low_starts(self):
        df = pd.DataFrame({
            "home_qb_changed": [0],
            "away_qb_changed": [0],
            "home_qb_team_starts_pre": [5.0],
            "away_qb_team_starts_pre": [50.0],
        })
        mask = _build_qb_gate_mask(df)
        assert mask[0]

    def test_sufficient_starts_no_change(self):
        df = pd.DataFrame({
            "home_qb_changed": [0],
            "away_qb_changed": [0],
            "home_qb_team_starts_pre": [50.0],
            "away_qb_team_starts_pre": [100.0],
        })
        mask = _build_qb_gate_mask(df)
        assert not mask[0]

    def test_missing_start_data(self):
        df = pd.DataFrame({
            "home_qb_changed": [0],
            "away_qb_changed": [0],
            "home_qb_team_starts_pre": [None],
            "away_qb_team_starts_pre": [None],
        })
        mask = _build_qb_gate_mask(df)
        assert not mask[0]


class TestApplyQBOverlay:
    def test_no_overlay_when_gate_off(self):
        base = np.array([0.0])
        result = _apply_qb_overlay(base, np.array([False]), np.array([0.0]), np.array([0.0]))
        assert result[0] == 0.0

    def test_overlay_applied_when_gate_on(self):
        base = np.array([0.0])
        result = _apply_qb_overlay(base, np.array([True]), np.array([10.0]), np.array([0.0]))
        assert result[0] != 0.0

    def test_home_away_adj_cancel(self):
        base = np.array([0.0])
        result = _apply_qb_overlay(base, np.array([True]), np.array([5.0]), np.array([5.0]))
        assert result[0] == 0.0

    def test_cap_applies(self):
        base = np.array([0.0])
        adj = QB_GATE_CAP + 100.0
        result = _apply_qb_overlay(base, np.array([True]), np.array([adj]), np.array([0.0]))
        h_gamma = QB_GATE_GAMMA * QB_GATE_CAP * np.log(10) / 400.0
        assert abs(result[0] - h_gamma) < 1e-10


class TestConstants:
    def test_v3_params_has_k_factor(self):
        assert "k_factor" in V3_PARAMS

    def test_v3_params_has_home_advantage(self):
        assert "home_advantage" in V3_PARAMS

    def test_pi_params_has_alpha(self):
        assert "alpha" in PI_PARAMS

    def test_pi_params_has_base_k(self):
        assert "base_k" in PI_PARAMS

    def test_pi_params_has_hk_ratio(self):
        assert "hk_ratio" in PI_PARAMS

    def test_promotion_delta_is_positive(self):
        assert MIN_PROMOTION_DELTA > 0

    def test_qb_gate_cap_positive(self):
        assert QB_GATE_CAP > 0

    def test_qb_gate_gamma_positive(self):
        assert QB_GATE_GAMMA > 0


class TestScoreModel:
    def test_returns_dict_with_keys(self):
        n = 20
        rng = np.random.default_rng(42)
        prob = rng.uniform(0.4, 0.6, n)
        y = (prob + 0.1 * rng.normal(size=n) > 0.5).astype(float)
        feat = np.empty((n, 0))
        mask = np.zeros(n, dtype=bool)
        train_mask = np.array([i < 14 for i in range(n)])
        val_mask = np.array([i >= 14 for i in range(n)])
        folds = [(train_mask, val_mask)]
        result = _score_model(prob, y, feat, mask, np.zeros(n), np.zeros(n), folds)
        assert "val_ll" in result
        assert "hold_ll" in result


class TestCLIImportability:
    def test_cli_command_importable(self):
        from sportslab.cli import pi_ratings_compare_cmd
        assert callable(pi_ratings_compare_cmd)

    def test_run_function_importable(self):
        from sportslab.evaluation.pi_ratings_champion_comparison import run_champion_comparison
        assert callable(run_champion_comparison)
