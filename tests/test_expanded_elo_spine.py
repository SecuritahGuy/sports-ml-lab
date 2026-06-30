"""Tests for expanded Elo spine experiment module."""


import numpy as np

from sportslab.evaluation.expanded_elo_spine_experiment import (
    DECAY_VALUES,
    ELO_TO_LOGIT,
    HFA_VALUES,
    K_VALUES,
    MIN_PROMOTION_DELTA,
    QB_GATE_CAP,
    QB_GATE_GAMMA,
    REG_VALUES,
    V3_HOLDOUT_LL,
    V3_VAL_LL,
    _apply_qb_overlay,
    _build_qb_gate_mask,
    _generate_param_combos,
    _get_features,
    _logit,
    _sigmoid,
)


class TestSigmoidLogit:
    def test_sigmoid_zero(self):
        assert _sigmoid(np.array([0.0]))[0] == 0.5

    def test_sigmoid_symmetry(self):
        x = np.array([-1.0, 0.5, 2.0])
        s = _sigmoid(x)
        assert np.allclose(s, 1 - _sigmoid(-x))

    def test_logit_sigmoid_inverse(self):
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(_sigmoid(_logit(p)), p)

    def test_logit_clips_extremes(self):
        p = np.array([0.0, 1.0])
        lp = _logit(p)
        assert np.all(np.isfinite(lp))


class TestGetFeatures:
    def test_returns_subset(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _get_features(df, ["a"])
        assert result.shape == (2, 1)

    def test_empty_cols(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1]})
        result = _get_features(df, [])
        assert result.shape == (1, 0)


class TestApplyQbOverlay:
    def test_no_gate_no_change(self):
        base = np.array([0.0, 0.0])
        gate = np.array([False, False])
        result = _apply_qb_overlay(base, gate, np.zeros(2), np.zeros(2))
        assert np.array_equal(result, base)

    def test_gate_active(self):
        base = np.array([0.0])
        gate = np.array([True])
        result = _apply_qb_overlay(base, gate, np.array([10.0]), np.array([0.0]))
        assert not np.array_equal(result, base)

    def test_capping(self):
        base = np.array([0.0])
        gate = np.array([True])
        r1 = _apply_qb_overlay(base, gate, np.array([100.0]), np.zeros(1))
        r2 = _apply_qb_overlay(base, gate, np.array([QB_GATE_CAP]), np.zeros(1))
        assert np.array_equal(r1, r2)


class TestBuildQbGateMask:
    def test_no_changed_returns_false(self):
        import pandas as pd
        df = pd.DataFrame({
            "home_qb_changed": [0, 0],
            "away_qb_changed": [0, 0],
        })
        mask = _build_qb_gate_mask(df)
        assert not mask.any()

    def test_changed_returns_true(self):
        import pandas as pd
        df = pd.DataFrame({
            "home_qb_changed": [1, 0],
            "away_qb_changed": [0, 0],
        })
        mask = _build_qb_gate_mask(df)
        assert mask[0]
        assert not mask[1]

    def test_low_starts_triggers_gate(self):
        import pandas as pd
        df = pd.DataFrame({
            "home_qb_changed": [0],
            "away_qb_changed": [0],
            "home_qb_team_starts_pre": [5],
            "away_qb_team_starts_pre": [20],
        })
        mask = _build_qb_gate_mask(df)
        assert mask[0]


class TestGenerateParamCombos:
    def test_expected_count(self):
        combos = _generate_param_combos()
        expected = len(K_VALUES) * len(HFA_VALUES) * len(REG_VALUES) * len(DECAY_VALUES)
        assert len(combos) == expected

    def test_all_have_keys(self):
        combos = _generate_param_combos()
        for c in combos:
            assert "K" in c
            assert "HFA" in c
            assert "reg" in c
            assert "decay" in c

    def test_covers_grid_values(self):
        combos = _generate_param_combos()
        k_set = set(c["K"] for c in combos)
        for k in K_VALUES:
            assert k in k_set


class TestConstants:
    def test_v3_reference(self):
        assert V3_VAL_LL == 0.6305
        assert V3_HOLDOUT_LL == 0.6200

    def test_qb_gate(self):
        assert QB_GATE_GAMMA == 1.0
        assert QB_GATE_CAP == 40

    def test_elo_to_logit(self):
        assert ELO_TO_LOGIT == np.log(10) / 400.0

    def test_min_promotion_delta(self):
        assert MIN_PROMOTION_DELTA == 0.001

    def test_grid_values(self):
        assert 60 in K_VALUES
        assert 20 in K_VALUES
        assert 50 in HFA_VALUES
        assert 20 in HFA_VALUES
        assert 0.3 in REG_VALUES
        assert 0.0 in REG_VALUES
        assert None in DECAY_VALUES
        assert 64 in DECAY_VALUES

    def test_module_importable(self):
        from sportslab.evaluation import expanded_elo_spine_experiment
        assert hasattr(expanded_elo_spine_experiment, "run_expanded_elo_spine")

    def test_cli_importable(self):
        from sportslab.cli import expanded_elo_spine_cmd
        assert callable(expanded_elo_spine_cmd)
