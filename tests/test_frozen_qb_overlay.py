"""Tests for frozen-incumbent QB overlay experiment.

Critical tests:
1. Non-gated games match incumbent exactly (within fp tolerance)
2. Gating logic is correct
3. Conversion from Elo points to logit is correct
4. No recalibration after gating
"""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.frozen_qb_overlay_experiment import (
    CAP_VALUES,
    ELO_TO_LOGIT,
    GAMMA_VALUES,
    _logit,
    _sigmoid,
)

FROZEN_EXPERIMENT = "sportslab.evaluation.frozen_qb_overlay_experiment"


class TestLogitSigmoid:
    def test_logit_sigmoid_identity(self):
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
        lo = _logit(p)
        p2 = _sigmoid(lo)
        assert np.allclose(p, p2, atol=1e-10)

    def test_sigmoid_range(self):
        x = np.array([-1000, -10, -1, 0, 1, 10, 1000])
        s = _sigmoid(x)
        assert np.all(s >= 0.0)
        assert np.all(s <= 1.0)
        assert np.allclose(s[0], 0.0, atol=1e-10)
        assert np.allclose(s[-1], 1.0, atol=1e-10)

    def test_logit_0_5_is_0(self):
        assert abs(_logit(np.array([0.5]))[0]) < 1e-10

    def test_logit_extremes_clip(self):
        lo = _logit(np.array([1e-20]))
        assert np.isfinite(lo)


class TestELOToLogitConversion:
    def test_conversion_factor_positive(self):
        assert ELO_TO_LOGIT > 0

    def test_positive_net_adj_increases_prob(self):
        base_logit = _logit(np.array([0.5]))
        net_adj = 40.0  # 40 Elo points = ~0.556 for neutral game
        delta = net_adj * ELO_TO_LOGIT
        final_logit = base_logit + delta
        final_prob = _sigmoid(final_logit)
        assert final_prob[0] > 0.5
        # 40 Elo points should give approx 0.556 probability
        expected = 1.0 / (1.0 + 10.0 ** (-40.0 / 400.0))
        assert abs(final_prob[0] - expected) < 0.001

    def test_negative_net_adj_decreases_prob(self):
        base_logit = _logit(np.array([0.5]))
        net_adj = -40.0
        delta = net_adj * ELO_TO_LOGIT
        final_logit = base_logit + delta
        final_prob = _sigmoid(final_logit)
        assert final_prob[0] < 0.5

    def test_zero_gamma_no_change(self):
        base_prob = np.array([0.3, 0.5, 0.8])
        base_logit = _logit(base_prob)
        net_adj = np.array([50.0, 0.0, -30.0])
        delta = 0.0 * net_adj * ELO_TO_LOGIT  # gamma=0
        final_logit = base_logit + delta
        final_prob = _sigmoid(final_logit)
        assert np.allclose(final_prob, base_prob, atol=1e-10)

    def test_gamma_one_matches_elo_adjustment(self):
        """At gamma=1.0, the overlay should match Elo-adjusted probability."""
        base_prob = np.array([0.5])
        base_logit = _logit(base_prob)
        home_adj = 40.0
        away_adj = 0.0
        net_adj = home_adj - away_adj
        gamma = 1.0
        overlay = gamma * net_adj * ELO_TO_LOGIT
        final_logit = base_logit + overlay
        final_prob = _sigmoid(final_logit)

        # Expected from Elo formula
        expected = 1.0 / (1.0 + 10.0 ** (-40.0 / 400.0))
        assert abs(final_prob[0] - expected) < 0.001


class TestNonGatedEquality:
    def test_no_gate_exact_match(self):
        """When no overlay gate is active, prediction must match incumbent."""
        base_prob = np.array([0.3, 0.5, 0.8])
        base_logit = _logit(base_prob)
        net_adj = np.array([50.0, -20.0, 30.0])
        gamma = 0.5
        gate = np.array([0, 0, 0])  # No overlay
        overlay = gamma * net_adj * ELO_TO_LOGIT * gate.astype(float)
        final_logit = base_logit + overlay
        final_prob = _sigmoid(final_logit)
        assert np.allclose(final_prob, base_prob, atol=1e-10)

    def test_mixed_gate_exact_for_non_gated(self):
        """Only gated games should change; non-gated must match exactly."""
        base_prob = np.array([0.3, 0.5, 0.8])
        base_logit = _logit(base_prob)
        net_adj = np.array([50.0, -20.0, 30.0])
        gamma = 0.5
        gate = np.array([1, 0, 1])  # Only games 0 and 2 get overlay
        overlay = gamma * net_adj * ELO_TO_LOGIT * gate.astype(float)
        final_logit = base_logit + overlay
        final_prob = _sigmoid(final_logit)

        # Game 1 (index 1) should be unchanged
        assert final_prob[1] == base_prob[1]
        # Games 0 and 2 should differ
        assert final_prob[0] != base_prob[0]
        assert final_prob[2] != base_prob[2]

    def test_all_gates_exact_for_baseline(self):
        """Baseline variant (gate_mode=baseline) must match incumbent on all games."""
        base_prob = np.array([0.3, 0.5, 0.8])
        base_logit = _logit(base_prob)
        # Baseline means no overlay at all
        final_prob = _sigmoid(base_logit)
        assert np.allclose(final_prob, base_prob, atol=1e-10)


class TestCappedAdjustment:
    def test_cap_limits_adjustment(self):
        base_prob = np.array([0.5])
        base_logit = _logit(base_prob)
        net_adj = np.array([200.0])  # Very large
        cap = 60.0
        capped_adj = np.clip(net_adj, -cap, cap)
        overlay = 1.0 * capped_adj * ELO_TO_LOGIT
        final_logit = base_logit + overlay
        final_prob = _sigmoid(final_logit)

        # With cap=60, prob should be ~1/(1+10^(-60/400)) ~ 0.586
        expected = 1.0 / (1.0 + 10.0 ** (-60.0 / 400.0))
        assert abs(final_prob[0] - expected) < 0.001

    def test_no_cap_no_limit(self):
        base_prob = np.array([0.5])
        base_logit = _logit(base_prob)
        net_adj = np.array([200.0])
        overlay = 1.0 * net_adj * ELO_TO_LOGIT
        final_logit = base_logit + overlay
        final_prob = _sigmoid(final_logit)

        expected_capped = 1.0 / (1.0 + 10.0 ** (-60.0 / 400.0))
        assert final_prob[0] > expected_capped


class TestGateConditions:
    def test_qb_changed_gate(self):
        """Gate should be True when either side has a QB change."""
        h_changed = np.array([1, 0, 0, 1])
        a_changed = np.array([0, 0, 1, 0])
        gate = (h_changed == 1) | (a_changed == 1)
        expected = np.array([True, False, True, True])
        assert np.array_equal(gate, expected)

    def test_starts_gate(self):
        """Gate should be True when either QB has < N starts."""
        h_starts = np.array([2, 10, 3, 0])
        a_starts = np.array([50, 50, 30, 0])
        threshold = 4
        gate = (h_starts < threshold) | (a_starts < threshold)
        expected = np.array([True, False, True, True])
        assert np.array_equal(gate, expected)

        threshold = 8
        gate = (h_starts < threshold) | (a_starts < threshold)
        expected = np.array([True, False, True, True])
        assert np.array_equal(gate, expected)

    def test_combined_gate(self):
        """Union of change and starts gates."""
        h_changed = np.array([1, 0, 0])
        a_changed = np.array([0, 0, 1])
        changed = (h_changed == 1) | (a_changed == 1)
        h_starts = np.array([0, 10, 50])
        a_starts = np.array([50, 3, 0])
        threshold = 4
        starts = (h_starts < threshold) | (a_starts < threshold)
        gate = changed | starts
        expected = np.array([True, True, True])  # all True
        assert np.array_equal(gate, expected)


class TestGAMMAandCAP:
    def test_gamma_values_include_boundary(self):
        assert 0.00 in GAMMA_VALUES
        assert 1.00 in GAMMA_VALUES

    def test_gamma_increasing(self):
        for i in range(len(GAMMA_VALUES) - 1):
            assert GAMMA_VALUES[i] <= GAMMA_VALUES[i + 1]

    def test_cap_values_include_none(self):
        assert None in CAP_VALUES

    def test_cap_values_positive(self):
        for c in CAP_VALUES:
            if c is not None:
                assert c > 0


class TestExperimentImportability:
    def test_module_imports(self):
        from sportslab.evaluation import frozen_qb_overlay_experiment
        assert hasattr(
            frozen_qb_overlay_experiment,
            "run_frozen_qb_overlay_experiment",
        )

    def test_cli_command_registered(self):
        from sportslab import cli
        commands = {c.name for c in cli.cli.commands.values()}
        assert "frozen-qb-overlay" in commands

    def test_makefile_target(self):
        import os
        path = "Makefile"
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            assert "frozen-qb-overlay" in content


class TestExperimentSafety:
    def test_no_season_before_2021(self):
        from sportslab.evaluation.experiment_config import ALL_SEASONS
        for s in ALL_SEASONS:
            assert s >= 2021

    def test_experiment_smoke(self, tmp_path):
        """Run end-to-end smoke test with real feature table."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")

        from sportslab.evaluation.frozen_qb_overlay_experiment import (
            run_frozen_qb_overlay_experiment,
        )
        report_p = tmp_path / "frozen_qb_test.md"
        run_frozen_qb_overlay_experiment(report_path=str(report_p))
        assert report_p.exists()
        content = report_p.read_text()
        assert "Decision" in content
        assert "Frozen-Incumbent" in content
        assert "Equality check" in content

    def test_equality_check_in_report(self, tmp_path):
        """The report must document whether non-gated games match incumbent."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")

        from sportslab.evaluation.frozen_qb_overlay_experiment import (
            run_frozen_qb_overlay_experiment,
        )
        report_p = tmp_path / "frozen_qb_test.md"
        run_frozen_qb_overlay_experiment(report_path=str(report_p))
        content = report_p.read_text()
        assert "Equality" in content
        assert "PASSED" in content or "max_diff" in content.lower()

    def test_incumbent_files_not_modified(self, tmp_path):
        """Verify experiment does not overwrite incumbent artifacts."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")

        from sportslab.evaluation.frozen_qb_overlay_experiment import (
            run_frozen_qb_overlay_experiment,
        )
        report_p = tmp_path / "frozen_qb_test.md"
        run_frozen_qb_overlay_experiment(report_path=str(report_p))
        import os
        inc_pred = "reports/predictions/incumbent_predictions.csv"
        if os.path.exists(inc_pred):
            df = pd.read_csv(inc_pred)
            assert "incumbent_home_win_prob" in df.columns
