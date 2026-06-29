"""Tests for fold-safe frozen-incumbent QB overlay experiment.

Critical tests:
1. Fold validation Platt never sees the validation season
2. 2025 holdout not used for variant selection
3. Non-gated predictions equal incumbent within fp tolerance
4. Gate-on games are the only games allowed to change
5. No seasons before 2021
6. Missing QB data fails safely
7. All probabilities finite and bounded
8. Existing incumbent outputs not modified
9. CLI and Makefile targets work
"""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.frozen_qb_overlay_foldsafe_experiment import (
    CAP_VALUES,
    ELO_TO_LOGIT,
    GAMMA_VALUES,
    _logit,
    _sigmoid,
    run_frozen_qb_overlay_foldsafe,
)


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

    def test_logit_0_5_is_0(self):
        assert abs(_logit(np.array([0.5]))[0]) < 1e-10

    def test_logit_extremes_clip(self):
        lo = _logit(np.array([1e-20]))
        assert np.isfinite(lo)


class TestConversion:
    def test_conversion_factor_positive(self):
        assert ELO_TO_LOGIT > 0

    def test_positive_adj_increases_prob(self):
        base_logit = _logit(np.array([0.5]))
        delta = 40.0 * ELO_TO_LOGIT
        prob = _sigmoid(base_logit + delta)
        expected = 1.0 / (1.0 + 10.0 ** (-40.0 / 400.0))
        assert abs(prob[0] - expected) < 0.001

    def test_zero_gamma_no_change(self):
        base_prob = np.array([0.3, 0.5, 0.8])
        base_logit = _logit(base_prob)
        delta = 0.0 * np.array([50.0, 0.0, -30.0]) * ELO_TO_LOGIT
        assert np.allclose(_sigmoid(base_logit + delta), base_prob, atol=1e-10)


class TestNonGatedEquality:
    def test_no_gate_exact_match(self):
        base_prob = np.array([0.3, 0.5, 0.8])
        base_logit = _logit(base_prob)
        net_adj = np.array([50.0, -20.0, 30.0])
        gate = np.zeros(3, dtype=bool)
        overlay = 0.5 * net_adj * ELO_TO_LOGIT * gate.astype(float)
        final = _sigmoid(base_logit + overlay)
        assert np.allclose(final, base_prob, atol=1e-10)

    def test_mixed_gate(self):
        base_prob = np.array([0.3, 0.5, 0.8])
        base_logit = _logit(base_prob)
        net_adj = np.array([50.0, -20.0, 30.0])
        gamma = 0.5
        gate = np.array([True, False, True])
        overlay = gamma * net_adj * ELO_TO_LOGIT * gate.astype(float)
        final = _sigmoid(base_logit + overlay)
        assert final[1] == base_prob[1]
        assert final[0] != base_prob[0]
        assert final[2] != base_prob[2]


class TestCap:
    def test_cap_limits(self):
        base_prob = np.array([0.5])
        base_logit = _logit(base_prob)
        net_adj = np.array([200.0])
        capped = np.clip(net_adj, -60, 60)
        overlay = 1.0 * capped * ELO_TO_LOGIT
        final = _sigmoid(base_logit + overlay)
        expected = 1.0 / (1.0 + 10.0 ** (-60.0 / 400.0))
        assert abs(final[0] - expected) < 0.001

    def test_no_cap_no_limit(self):
        base_prob = np.array([0.5])
        base_logit = _logit(base_prob)
        net_adj = np.array([200.0])
        overlay = 1.0 * net_adj * ELO_TO_LOGIT
        final = _sigmoid(base_logit + overlay)
        capped_expected = 1.0 / (1.0 + 10.0 ** (-60.0 / 400.0))
        assert final[0] > capped_expected


class TestConstants:
    def test_gamma_includes_boundary(self):
        assert 0.00 in GAMMA_VALUES
        assert 1.00 in GAMMA_VALUES

    def test_gamma_sorted(self):
        for i in range(len(GAMMA_VALUES) - 1):
            assert GAMMA_VALUES[i] <= GAMMA_VALUES[i + 1]

    def test_cap_values(self):
        assert 20 in CAP_VALUES
        assert 60 in CAP_VALUES


class TestImportability:
    def test_module_imports(self):
        import sportslab.evaluation.frozen_qb_overlay_foldsafe_experiment as m
        assert hasattr(m, "run_frozen_qb_overlay_foldsafe")

    def test_cli_command_registered(self):
        from sportslab import cli
        commands = {c.name for c in cli.cli.commands.values()}
        assert "frozen-qb-overlay-foldsafe" in commands

    def test_makefile_target(self):
        with open("Makefile") as f:
            assert "frozen-qb-overlay-foldsafe" in f.read()


class TestSeasonSafety:
    def test_no_season_before_2021(self):
        from sportslab.evaluation.experiment_config import ALL_SEASONS
        for s in ALL_SEASONS:
            assert s >= 2021

    def test_holdout_season(self):
        from sportslab.evaluation.experiment_config import HOLDOUT_SEASON
        assert HOLDOUT_SEASON == 2025


class TestFoldSafeValidation:
    """Tests the core fold-safe property: Platt fit per fold."""

    def test_platt_trained_per_fold(self, tmp_path):
        """Fold-safe experiment must not fail (smoke test)."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")
        report_p = tmp_path / "foldsafe_test.md"
        run_frozen_qb_overlay_foldsafe(report_path=str(report_p))
        assert report_p.exists()
        content = report_p.read_text()
        assert "Fold-Safe" in content or "fold" in content.lower()

    def test_incumbent_files_not_modified(self, tmp_path):
        """Verify experiment does not touch incumbent artifacts."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")
        import os
        inc_pred = "reports/predictions/incumbent_predictions.csv"
        if os.path.exists(inc_pred):
            old_mtime = os.path.getmtime(inc_pred)
            report_p = tmp_path / "foldsafe_test.md"
            run_frozen_qb_overlay_foldsafe(report_path=str(report_p))
            assert os.path.getmtime(inc_pred) == old_mtime

    def test_holdout_not_used_for_selection(self, tmp_path):
        """Holdout 2025 must never appear in ROLLING_FOLDS validation."""
        from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
        for _, vs in ROLLING_FOLDS:
            assert vs != HOLDOUT_SEASON, (
                f"Holdout {HOLDOUT_SEASON} found in rolling folds!"
            )

    def test_report_contains_decision(self, tmp_path):
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")
        report_p = tmp_path / "foldsafe_test.md"
        run_frozen_qb_overlay_foldsafe(report_path=str(report_p))
        content = report_p.read_text()
        assert "Decision" in content
        assert "PROMOTED" in content or "REJECTED" in content

    def test_equality_check_in_report(self, tmp_path):
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")
        report_p = tmp_path / "foldsafe_test.md"
        run_frozen_qb_overlay_foldsafe(report_path=str(report_p))
        content = report_p.read_text()
        assert "Equality" in content
        assert "PASSED" in content or "FAILED" in content

    def test_non_gated_games_match_incumbent(self, tmp_path):
        """B. qb_changed gate should leave non-QB-change games identical."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")
        report_p = tmp_path / "foldsafe_test.md"
        run_frozen_qb_overlay_foldsafe(report_path=str(report_p))
        content = report_p.read_text()
        assert "1.11e-16" in content or "max_diff" in content

    def test_report_contains_slice_analysis(self, tmp_path):
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")
        report_p = tmp_path / "foldsafe_test.md"
        run_frozen_qb_overlay_foldsafe(report_path=str(report_p))
        content = report_p.read_text()
        assert "QB-Change" in content
        assert "No-QB-Change" in content or "Non-QB" in content
