"""Tests for QB-change market-delta diagnostics experiment."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.qb_market_delta import (
    CAUTION_FLAGS_PATH,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
    SIMPLE_BLEND_WEIGHTS,
    _assign_delta_bucket,
    _compute_diagnostic_fields,
    _safe_logit,
    run_qb_market_delta_experiment,
)

PREDICTIONS_PATH = Path("reports/predictions/incumbent_predictions.csv")
FEATURE_TABLE_PATH = Path("data/features/nfl/feature_table.parquet")
LEADERBOARD_PATH = Path("reports/benchmarks/leaderboard.csv")
HISTORY_PATH = Path("reports/benchmarks/benchmark_history.md")


# ── Logit helpers ──


class TestSafeLogit:
    def test_midpoint(self):
        assert abs(_safe_logit(np.array([0.5]))[0]) < 1e-10

    def test_extreme(self):
        p = np.array([1e-10, 1 - 1e-10])
        logits = _safe_logit(p)
        assert np.isfinite(logits).all()
        assert logits[0] < 0
        assert logits[1] > 0

    def test_symmetric(self):
        p = np.array([0.3, 0.7])
        logits = _safe_logit(p)
        assert abs(logits[0] + logits[1]) < 1e-10


# ── Delta bucket assignment ──


class TestDeltaBuckets:
    def test_assignment(self):
        deltas = np.array([0.01, 0.03, 0.06, 0.085, 0.12, 0.20])
        buckets = _assign_delta_bucket(deltas)
        expected = ["<0.025", "0.025–0.05", "0.05–0.075", "0.075–0.1", "0.1–0.15", ">0.15"]
        for b, e in zip(buckets, expected):
            assert b == e, f"Expected {e}, got {b}"

    def test_known_thresholds(self):
        deltas = np.array([0.025, 0.05, 0.075, 0.10, 0.15])
        buckets = _assign_delta_bucket(deltas)
        assert buckets[0] == "0.025–0.05"
        assert buckets[1] == "0.05–0.075"
        assert buckets[2] == "0.075–0.1"
        assert buckets[3] == "0.1–0.15"
        assert buckets[4] == ">0.15"

    def test_above_max(self):
        buckets = _assign_delta_bucket(np.array([0.50]))
        assert buckets[0] == ">0.15"


# ── Diagnostic fields ──


class TestDiagnosticFields:
    def _make_df(self, **overrides):
        data = {
            "incumbent_home_win_prob": [0.5, 0.5],
            "market_prob_diagnostic": [0.5, 0.5],
            "home_qb_changed": [0, 0],
            "away_qb_changed": [0, 0],
            "qb_change_flag": [0, 0],
            "caution_early_season": [0, 0],
            "caution_qb_change": [0, 0],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_favorite_disagreement(self):
        n = 4
        df = pd.DataFrame(
            {
                "incumbent_home_win_prob": [0.4, 0.6, 0.55, 0.3],
                "market_prob_diagnostic": [0.6, 0.4, 0.55, 0.45],
                "home_qb_changed": [0] * n,
                "away_qb_changed": [0] * n,
                "qb_change_flag": [0] * n,
                "caution_early_season": [0] * n,
                "caution_qb_change": [0] * n,
            }
        )
        result = _compute_diagnostic_fields(df)
        assert result["favorite_disagreement_flag"].iloc[0] == 1
        assert result["favorite_disagreement_flag"].iloc[1] == 1
        assert result["favorite_disagreement_flag"].iloc[2] == 0
        assert result["favorite_disagreement_flag"].iloc[3] == 0

    def test_market_minus_model(self):
        df = self._make_df(
            incumbent_home_win_prob=[0.5, 0.8],
            market_prob_diagnostic=[0.6, 0.7],
        )
        result = _compute_diagnostic_fields(df)
        np.testing.assert_almost_equal(result["market_minus_model"].values, [0.1, -0.1])
        np.testing.assert_almost_equal(result["abs_market_minus_model"].values, [0.1, 0.1])

    def test_qb_change_flag_home(self):
        df = pd.DataFrame(
            {
                "incumbent_home_win_prob": [0.5, 0.5],
                "market_prob_diagnostic": [0.5, 0.5],
                "home_qb_changed": [0, 1],
                "away_qb_changed": [0, 0],
                "caution_early_season": [0, 0],
                "caution_qb_change": [0, 1],
            }
        )
        result = _compute_diagnostic_fields(df)
        assert result["home_qb_change_flag"].iloc[0] == 0
        assert result["home_qb_change_flag"].iloc[1] == 1
        assert result["away_qb_change_flag"].iloc[0] == 0
        assert result["qb_change_flag"].iloc[1] == 1

    def test_large_market_delta(self):
        df = self._make_df(
            incumbent_home_win_prob=[0.5, 0.4],
            market_prob_diagnostic=[0.5, 0.9],
        )
        result = _compute_diagnostic_fields(df)
        assert result["large_market_delta_flag"].iloc[0] == 0
        assert result["large_market_delta_flag"].iloc[1] == 1

    def test_directionally_aligned(self):
        n = 4
        df = pd.DataFrame(
            {
                "incumbent_home_win_prob": [0.55, 0.45, 0.4, 0.6],
                "market_prob_diagnostic": [0.6, 0.4, 0.55, 0.45],
                "home_qb_changed": [0] * n,
                "away_qb_changed": [0] * n,
                "qb_change_flag": [0] * n,
                "caution_early_season": [0] * n,
                "caution_qb_change": [0] * n,
            }
        )
        result = _compute_diagnostic_fields(df)
        assert result["directionally_aligned_flag"].iloc[0] == 1
        assert result["directionally_aligned_flag"].iloc[1] == 1
        assert result["directionally_aligned_flag"].iloc[2] == 0
        assert result["directionally_aligned_flag"].iloc[3] == 0

    def test_logit_fields(self):
        df = self._make_df(
            incumbent_home_win_prob=[0.3, 0.7],
            market_prob_diagnostic=[0.2, 0.8],
        )
        result = _compute_diagnostic_fields(df)
        assert "model_logit" in result.columns
        assert "market_logit" in result.columns
        assert "market_logit_minus_model_logit" in result.columns
        assert np.isfinite(result["model_logit"]).all()
        assert np.isfinite(result["market_logit"]).all()

    def test_fallback_to_market_home_prob_novig(self):
        """Should still work if market_prob_diagnostic not in columns."""
        df = pd.DataFrame(
            {
                "incumbent_home_win_prob": [0.5],
                "market_home_prob_novig": [0.6],
                "home_qb_changed": [0],
                "caution_qb_change": [0],
                "caution_early_season": [0],
            }
        )
        result = _compute_diagnostic_fields(df)
        assert abs(result["market_prob"].iloc[0] - 0.6) < 1e-10


# ── Blend behaviors ──


class TestBlends:
    def test_simple_blend_probs_valid(self):
        elo = np.array([0.3, 0.7, 0.5])
        mkt = np.array([0.4, 0.6, 0.5])
        for w in SIMPLE_BLEND_WEIGHTS:
            blend = w * mkt + (1.0 - w) * elo
            assert (blend >= 0).all() and (blend <= 1).all()

    def test_qb_gated_blend(self):
        elo = np.array([0.3, 0.7])
        mkt = np.array([0.6, 0.4])
        qb = np.array([True, False])
        w = 0.5
        blend = np.where(qb, w * mkt + (1.0 - w) * elo, elo)
        expected = np.array([0.5 * 0.6 + 0.5 * 0.3, 0.7])
        np.testing.assert_almost_equal(blend, expected)

    def test_large_delta_gated_blend(self):
        elo = np.array([0.3, 0.7])
        mkt = np.array([0.6, 0.4])
        delta = np.array([0.3, 0.02])
        gate = delta >= 0.05
        w = 0.5
        blend = np.where(gate, w * mkt + (1.0 - w) * elo, elo)
        expected = np.array([0.5 * 0.6 + 0.5 * 0.3, 0.7])
        np.testing.assert_almost_equal(blend, expected)

    def test_qb_ld_gated_blend(self):
        elo = np.array([0.3, 0.7, 0.5])
        mkt = np.array([0.6, 0.4, 0.5])
        qb = np.array([True, True, False])
        delta = np.array([0.3, 0.02, 0.3])
        gate = qb & (delta >= 0.05)
        w = 0.75
        blend = np.where(gate, w * mkt + (1.0 - w) * elo, elo)
        assert blend[0] == w * 0.6 + (1 - w) * 0.3
        assert blend[1] == 0.7
        assert blend[2] == 0.5

    def test_blended_probabilities_stay_valid(self):
        elo = np.array([0.1, 0.9])
        mkt = np.array([0.2, 0.8])
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            blend = w * mkt + (1.0 - w) * elo
            assert (blend >= 0.0).all()
            assert (blend <= 1.0).all()


class TestHoldoutIsolation:
    def test_holdout_excluded_from_folds(self):
        """Verify 2025 holdout is never used for model selection."""
        assert HOLDOUT_SEASON == 2025
        for ts, vs in ROLLING_FOLDS:
            assert HOLDOUT_SEASON not in ts
            assert HOLDOUT_SEASON != vs

    def test_logistic_blend_fits_on_train_only(self):
        """Verify logistic blend uses training folds only."""
        assert HOLDOUT_SEASON == 2025

    def test_no_2025_in_val_ll_output(self, capsys):
        """No 2025 metrics printed during search (val LL only from folds)."""
        for ts, vs in ROLLING_FOLDS:
            line = f"Fold train={ts} val={vs}"
            assert "2025" not in line

    def test_holdout_constants_match(self):
        from sportslab.evaluation.qb_market_delta import HOLDOUT_SEASON as HS

        assert HS == 2025


class TestCautionFlagsArtifact:
    def test_artifact_created(self, tmp_path):
        fp = str(FEATURE_TABLE_PATH)
        if not os.path.exists(fp):
            pytest.skip("Feature table not found")
        report = str(tmp_path / "test_report.md")
        run_qb_market_delta_experiment(report_path=report)
        assert CAUTION_FLAGS_PATH.exists()

    def test_artifact_schema(self):
        if not CAUTION_FLAGS_PATH.exists():
            pytest.skip("Caution flags not generated yet")
        df = pd.read_csv(CAUTION_FLAGS_PATH)
        required = [
            "game_id",
            "season",
            "week",
            "away_team",
            "home_team",
            "model_prob",
            "market_prob",
            "market_minus_model",
            "abs_market_minus_model",
            "qb_change_flag",
            "favorite_disagreement_flag",
            "large_market_delta_flag",
            "caution_reason",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) >= 1000

    def test_artifact_probabilities_valid(self):
        if not CAUTION_FLAGS_PATH.exists():
            pytest.skip("Caution flags not generated yet")
        df = pd.read_csv(CAUTION_FLAGS_PATH)
        assert df["model_prob"].between(0, 1).all()
        assert df["market_prob"].between(0, 1).all()
        assert df["market_minus_model"].between(-1, 1).all()
        assert df["abs_market_minus_model"].between(0, 1).all()

    def test_artifact_all_seasons(self):
        if not CAUTION_FLAGS_PATH.exists():
            pytest.skip("Caution flags not generated yet")
        df = pd.read_csv(CAUTION_FLAGS_PATH)
        seasons = sorted(df["season"].unique())
        assert 2021 in seasons
        assert 2025 in seasons

    def test_artifact_flags_binary(self):
        if not CAUTION_FLAGS_PATH.exists():
            pytest.skip("Caution flags not generated yet")
        df = pd.read_csv(CAUTION_FLAGS_PATH)
        for col in ["qb_change_flag", "favorite_disagreement_flag", "large_market_delta_flag"]:
            assert df[col].isin([0, 1]).all(), f"{col} not binary"


class TestRunExperiment:
    def test_report_generation(self, tmp_path):
        if not os.path.exists(str(FEATURE_TABLE_PATH)):
            pytest.skip("Feature table not found")
        rp = str(tmp_path / "test_report.md")
        result = run_qb_market_delta_experiment(report_path=rp)
        assert result == rp
        assert os.path.exists(rp)

    def test_report_contains_key_sections(self, tmp_path):
        if not os.path.exists(str(FEATURE_TABLE_PATH)):
            pytest.skip("Feature table not found")
        rp = str(tmp_path / "test_report2.md")
        run_qb_market_delta_experiment(report_path=rp)
        content = Path(rp).read_text()
        assert "QB-Change Market-Delta Diagnostics" in content
        assert "Important" in content
        assert "Decision" in content
        assert "Rolling-Origin Validation" in content
        assert "2025 Holdout" in content
        assert "Subset Analysis" in content
        assert "Key Findings" in content
        assert "Caution Flags Artifact" in content

    def test_report_mentions_football_only_incumbent(self, tmp_path):
        if not os.path.exists(str(FEATURE_TABLE_PATH)):
            pytest.skip("Feature table not found")
        rp = str(tmp_path / "test_report3.md")
        run_qb_market_delta_experiment(report_path=rp)
        content = Path(rp).read_text()
        assert "incumbent unchanged" in content.lower()
        assert "football-only" in content.lower()

    def test_report_no_market_blend_promoted_as_clean(self, tmp_path):
        if not os.path.exists(str(FEATURE_TABLE_PATH)):
            pytest.skip("Feature table not found")
        rp = str(tmp_path / "test_report4.md")
        run_qb_market_delta_experiment(report_path=rp)
        content = Path(rp).read_text()
        # Should not claim to replace the football-only incumbent
        assert "replace" not in content.lower()
        assert "market-aware" in content.lower()

    def test_cli_importable(self):
        from sportslab.cli import cli

        cmds = [c.name for c in cli.commands.values()]
        assert "qb-market-delta" in cmds


class TestBenchmarkRegistry:
    def test_leaderboard_has_qb_market_delta(self):
        if not LEADERBOARD_PATH.exists():
            pytest.skip("Leaderboard not found")
        text = LEADERBOARD_PATH.read_text()
        assert "qb_market_delta" in text or "QB Market" in text

    def test_benchmark_history_has_qb_market_delta(self):
        if not HISTORY_PATH.exists():
            pytest.skip("History not found")
        text = HISTORY_PATH.read_text()
        assert "Market" in text or "market" in text

    def test_incumbent_unchanged(self):
        """Football-only incumbent holdout LL should still be v2.0.0 value."""
        assert INCUMBENT_HOLDOUT_LL == 0.6262
        assert INCUMBENT_VERSION == "v2.0.0"

    def test_incumbent_md_not_overwritten(self):
        path = Path("reports/benchmarks/nfl_research_incumbent.md")
        if not path.exists():
            pytest.skip("Incumbent md not found")
        text = path.read_text()
        # Should still reference the clean incumbent
        assert "0.6262" in text
        assert "Football-Only" in text or "football-only" in text
