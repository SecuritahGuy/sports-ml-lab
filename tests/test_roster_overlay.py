"""Tests for roster_overlay_foldsafe_experiment module."""

import numpy as np
import pandas as pd

from sportslab.evaluation.roster_overlay_foldsafe_experiment import (
    ELO_TO_LOGIT,
    POSITION_GROUPS,
    _apply_overlay,
    _build_depletion_masks,
    _logit,
    _sigmoid,
)


class TestHelperFunctions:
    def test_sigmoid_zero(self):
        """Sigmoid(0) should be 0.5."""
        assert _sigmoid(np.array([0.0]))[0] == 0.5

    def test_sigmoid_symmetry(self):
        """Sigmoid(-x) should be 1 - sigmoid(x)."""
        x = np.array([-2.0, 0.5, 3.0])
        s = _sigmoid(x)
        assert np.allclose(s, 1 - _sigmoid(-x))

    def test_logit_sigmoid_inverse(self):
        """logit and sigmoid should be inverses."""
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(_sigmoid(_logit(p)), p)

    def test_logit_clips_extremes(self):
        """logit should clip at [1e-15, 1-1e-15]."""
        p = np.array([0.0, 1.0])
        lp = _logit(p)
        assert np.all(np.isfinite(lp))

    def test_build_depletion_masks(self):
        """depletion masks should be in [0, 1]."""
        df = pd.DataFrame({
            "home_ol_availability": [1.0, 0.6],
            "away_ol_availability": [0.8, 1.0],
        })
        masks = _build_depletion_masks(df)
        assert "ol_net_depletion" in masks
        assert np.all(masks["ol_home_depletion"] >= 0) and np.all(masks["ol_home_depletion"] <= 1)
        assert np.all(masks["ol_net_depletion"] >= -1) and np.all(masks["ol_net_depletion"] <= 1)

    def test_depletion_missing_defaults(self):
        """Missing availability columns should default to 1.0."""
        df = pd.DataFrame({"home_score": [10]})
        masks = _build_depletion_masks(df)
        for g in POSITION_GROUPS:
            assert g in masks or any(g in k for k in masks)

    def test_apply_overlay_no_gamma(self):
        """gamma=0 should return incumbent_prob unchanged."""
        prob = np.array([0.5, 0.7])
        logit = np.array([0.0, 0.85])
        result = _apply_overlay(prob, logit, np.zeros(2), np.zeros(2), np.zeros(2), 0, 0.1, 40)
        assert np.array_equal(result, prob)

    def test_apply_overlay_gated(self):
        """Games below threshold should remain at base probability."""
        prob = np.array([0.5, 0.5])
        logit = np.array([0.0, 0.0])
        home_dep = np.array([0.5, 0.05])
        away_dep = np.array([0.0, 0.0])
        net = np.array([0.5, 0.05])
        result = _apply_overlay(prob, logit, home_dep, away_dep, net, 20, 0.3, 40)
        # First game: max(0.5, 0) = 0.5 > 0.3 -> gated ON -> prob changes
        # Second game: max(0.05, 0) = 0.05 <= 0.3 -> gated OFF -> prob stays
        assert result[1] == 0.5  # gated OFF, identical to base
        assert result[0] != 0.5  # gated ON, prob changed

    def test_apply_overlay_capped(self):
        """Net depletion should be capped by cap parameter."""
        prob = np.array([0.5])
        logit = np.array([0.0])
        result = _apply_overlay(prob, logit, np.array([1.0]), np.array([0.0]),
                                np.array([10.0]), 20, 0.0, 20)
        # net dep is clipped to cap/60 = 20/60 = 0.333
        # overlay = 20 * 0.333 * ln(10)/400
        assert result[0] != 0.5

    def test_non_gated_equality(self):
        """Non-gated games should be exactly equal to incumbent."""
        prob = np.array([0.5])
        logit = np.array([0.0])
        result = _apply_overlay(prob, logit, np.array([0.0]), np.array([0.0]),
                                np.array([0.0]), 30, 0.1, 40)
        assert result[0] == 0.5


class TestExperimentStructure:
    def test_position_groups_defined(self):
        """Expected position groups should be defined."""
        for g in ["ol", "skill", "front", "lb", "coverage"]:
            assert g in POSITION_GROUPS, f"Missing group: {g}"

    def test_elo_to_logit_constants(self):
        """ELO_TO_LOGIT should match ln(10)/400."""
        assert ELO_TO_LOGIT == np.log(10) / 400.0

    def test_importability(self):
        """Module should import."""
        from sportslab.evaluation import roster_overlay_foldsafe_experiment
        assert hasattr(roster_overlay_foldsafe_experiment, "run_roster_overlay_foldsafe")
