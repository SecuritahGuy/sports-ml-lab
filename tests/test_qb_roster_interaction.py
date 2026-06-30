"""Tests for QB × roster interaction experiment module."""

import numpy as np
import pandas as pd

from sportslab.evaluation.qb_roster_interaction_experiment import (
    ELO_TO_LOGIT,
    MIN_PROMOTION_DELTA,
    POSITION_GROUPS,
    QB_GATE_CAP,
    QB_GATE_GAMMA,
    R_CAP_VALUES,
    R_GAMMA_VALUES,
    R_THRESHOLD_VALUES,
    _apply_qb_overlay,
    _apply_roster_overlay,
    _build_depletion_masks,
    _get_features,
    _logit,
    _sigmoid,
)


class TestSigmoidLogit:
    def test_sigmoid_zero(self):
        assert _sigmoid(np.array([0.0]))[0] == 0.5

    def test_sigmoid_symmetry(self):
        x = np.array([-2.0, 0.5, 3.0])
        s = _sigmoid(x)
        assert np.allclose(s, 1 - _sigmoid(-x))

    def test_logit_sigmoid_inverse(self):
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(_sigmoid(_logit(p)), p)

    def test_logit_clips_extremes(self):
        p = np.array([0.0, 1.0])
        lp = _logit(p)
        assert np.all(np.isfinite(lp))


class TestBuildDepletionMasks:
    def test_depletion_single_group(self):
        df = pd.DataFrame({
            "home_ol_availability": [1.0, 0.6],
            "away_ol_availability": [0.8, 1.0],
        })
        masks = _build_depletion_masks(df)
        assert "ol_net_depletion" in masks
        assert np.all(masks["ol_home_depletion"] >= 0)
        assert np.all(masks["ol_home_depletion"] <= 1)

    def test_depletion_range(self):
        df = pd.DataFrame({
            "home_ol_availability": [0.0],
            "away_ol_availability": [1.0],
        })
        masks = _build_depletion_masks(df)
        assert masks["ol_home_depletion"][0] == 1.0
        assert masks["ol_away_depletion"][0] == 0.0
        assert masks["ol_net_depletion"][0] == 1.0

    def test_depletion_missing_defaults(self):
        df = pd.DataFrame({"home_score": [10]})
        masks = _build_depletion_masks(df)
        for group in POSITION_GROUPS:
            key = f"{group}_home_depletion"
            assert key in masks

    def test_all_position_groups(self):
        data: dict[str, list] = {}
        for group in POSITION_GROUPS:
            data[f"home_{group}_availability"] = [1.0, 0.5]
            data[f"away_{group}_availability"] = [0.8, 1.0]
        df = pd.DataFrame(data)
        masks = _build_depletion_masks(df)
        for group in POSITION_GROUPS:
            assert f"{group}_net_depletion" in masks
            assert f"{group}_home_depletion" in masks
            assert f"{group}_away_depletion" in masks


class TestGetFeatures:
    def test_returns_subset(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = _get_features(df, ["a", "c"])
        assert result.shape == (2, 2)
        assert np.array_equal(result[:, 0], [1, 2])
        assert np.array_equal(result[:, 1], [5, 6])

    def test_empty_cols(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _get_features(df, [])
        assert result.shape == (2, 0)

    def test_missing_cols_ignored(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _get_features(df, ["a", "b"])
        assert result.shape == (2, 1)


class TestApplyQbOverlay:
    def test_no_gate_no_change(self):
        base = np.array([0.0, 0.0])
        gate = np.array([False, False])
        result = _apply_qb_overlay(base, gate, np.zeros(2), np.zeros(2))
        assert np.array_equal(result, base)

    def test_gate_active_changes_prob(self):
        base = np.array([0.0])
        gate = np.array([True])
        home_adj = np.array([10.0])
        away_adj = np.array([0.0])
        result = _apply_qb_overlay(base, gate, home_adj, away_adj)
        assert not np.array_equal(result, base)

    def test_gate_active_home_advantage(self):
        base = np.array([0.0])
        gate = np.array([True])
        home_adj = np.array([QB_GATE_CAP])
        away_adj = np.array([0.0])
        result = _apply_qb_overlay(base, gate, home_adj, away_adj)
        assert result[0] > 0.0

    def test_gate_active_away_disadvantage(self):
        base = np.array([0.0])
        gate = np.array([True])
        home_adj = np.array([0.0])
        away_adj = np.array([QB_GATE_CAP])
        result = _apply_qb_overlay(base, gate, home_adj, away_adj)
        assert result[0] < 0.0

    def test_capping(self):
        base = np.array([0.0])
        gate = np.array([True])
        large_adj = np.array([100.0])
        result = _apply_qb_overlay(base, gate, large_adj, np.zeros(1))
        capped = _apply_qb_overlay(base, gate, np.array([QB_GATE_CAP]), np.zeros(1))
        assert np.array_equal(result, capped)


class TestApplyRosterOverlay:
    def test_gamma_zero_unchanged(self):
        l1 = np.array([0.0, 0.85])
        base = np.array([0.0, 0.85])
        result = _apply_roster_overlay(
            l1, base, np.zeros(2), np.zeros(2), np.zeros(2),
            np.array([True, True]), 0, 0.1, 20,
        )
        assert np.array_equal(result, _sigmoid(l1))

    def test_non_gated_unchanged(self):
        l1 = np.array([0.0])
        base = np.array([0.0])
        result = _apply_roster_overlay(
            l1, base, np.array([0.0]), np.array([0.0]), np.array([0.5]),
            np.array([False]), 20, 0.1, 40,
        )
        assert result[0] == _sigmoid(l1)[0]

    def test_gated_changes_prob(self):
        l1 = np.array([0.0])
        base = np.array([0.0])
        result = _apply_roster_overlay(
            l1, base, np.array([0.5]), np.array([0.0]), np.array([0.5]),
            np.array([True]), 20, 0.1, 40,
        )
        assert result[0] != _sigmoid(l1)[0]


class TestExperimentStructure:
    def test_position_groups_defined(self):
        for g in ["ol", "skill", "front", "lb", "coverage"]:
            assert g in POSITION_GROUPS

    def test_elo_to_logit_constant(self):
        assert ELO_TO_LOGIT == np.log(10) / 400.0

    def test_qb_gate_constants(self):
        assert QB_GATE_GAMMA == 1.0
        assert QB_GATE_CAP == 40

    def test_min_promotion_delta(self):
        assert MIN_PROMOTION_DELTA == 0.001

    def test_sweep_ranges(self):
        assert 0 in R_GAMMA_VALUES
        assert 0.1 in R_THRESHOLD_VALUES
        assert 20 in R_CAP_VALUES

    def test_module_importable(self):
        from sportslab.evaluation import qb_roster_interaction_experiment
        assert hasattr(qb_roster_interaction_experiment, "run_qb_roster_interaction")

    def test_cli_importable(self):
        from sportslab.cli import qb_roster_cmd
        assert callable(qb_roster_cmd)
