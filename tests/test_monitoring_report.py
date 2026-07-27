"""Tests for weekly monitoring report generator."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.monitoring_report import (
    _check_drift_thresholds,
    _dome_mask,
    _fmt,
    _get_snapshot_info,
    _high_confidence_accuracy,
    _load_all_graded_games,
    _missing_weather_mask,
    _qb_change_mask,
    _small_sample_rule,
    _subgroup_log_loss,
    generate_monitoring_report,
)
from sportslab.evaluation.predict_incumbent import (
    INCUMBENT_VERSION,
)


class TestSmallSampleRule:
    def test_no_ll_zero_to_five(self):
        for n in range(0, 6):
            assert _small_sample_rule(n) == "no_ll"

    def test_low_sample_six_to_fifteen(self):
        for n in range(6, 16):
            assert _small_sample_rule(n) == "low_sample"

    def test_full_sixteen_plus(self):
        for n in [16, 20, 100, 1000]:
            assert _small_sample_rule(n) == "full"


class TestHighConfidenceAccuracy:
    def test_empty_df(self):
        df = pd.DataFrame()
        n, missed, acc = _high_confidence_accuracy(df, 0.80)
        assert n == 0
        assert missed == 0
        assert acc == 0.0

    def test_no_high_confidence(self):
        df = pd.DataFrame({
            "actual_home_win": [1, 0, 1],
            "incumbent_home_win_prob": [0.51, 0.49, 0.52],
        })
        n, missed, acc = _high_confidence_accuracy(df, 0.80)
        assert n == 0

    def test_all_correct(self):
        df = pd.DataFrame({
            "actual_home_win": [1, 0, 0],
            "incumbent_home_win_prob": [0.95, 0.05, 0.10],
        })
        n, missed, acc = _high_confidence_accuracy(df, 0.80)
        assert n == 3
        assert missed == 0
        assert acc == 1.0

    def test_some_wrong(self):
        df = pd.DataFrame({
            "actual_home_win": [0, 0],
            "incumbent_home_win_prob": [0.90, 0.10],
        })
        n, missed, acc = _high_confidence_accuracy(df, 0.80)
        assert n == 2
        assert missed == 1
        assert acc == 0.5


class TestSubgroupLogLoss:
    def test_empty_mask(self):
        df = pd.DataFrame({"actual_home_win": [1, 0], "incumbent_home_win_prob": [0.6, 0.4]})
        mask = np.zeros(len(df), dtype=bool)
        assert _subgroup_log_loss(df, mask) is None

    def test_small_sample_no_ll(self):
        df = pd.DataFrame({"actual_home_win": [1], "incumbent_home_win_prob": [0.9]})
        mask = np.array([True])
        assert _subgroup_log_loss(df, mask) is None

    def test_valid_log_loss(self):
        df = pd.DataFrame({
            "actual_home_win": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "incumbent_home_win_prob": [0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.55, 0.45,
                                        0.65, 0.35, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2,
                                        0.55, 0.45, 0.65, 0.35],
        })
        mask = np.ones(len(df), dtype=bool)
        ll = _subgroup_log_loss(df, mask)
        assert ll is not None
        assert isinstance(ll, float)
        assert ll > 0


class TestMaskFunctions:
    def test_missing_weather_mask_flag(self):
        df = pd.DataFrame({"weather_missing_flag": [True, False, None]})
        mask = _missing_weather_mask(df)
        assert mask.tolist() == [True, False, False]

    def test_missing_weather_mask_no_col(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        mask = _missing_weather_mask(df)
        assert mask.tolist() == [False, False, False]

    def test_dome_mask(self):
        df = pd.DataFrame({"roof": ["dome", "open", "retractable", None]})
        mask = _dome_mask(df)
        assert mask.tolist() == [True, False, False, False]

    def test_qb_change_mask_flag(self):
        df = pd.DataFrame({"qb_change_flag": [True, False, None]})
        mask = _qb_change_mask(df)
        assert mask.tolist() == [True, False, False]

    def test_qb_change_mask_home_away(self):
        df = pd.DataFrame({
            "home_qb_changed": [True, False],
            "away_qb_changed": [False, True],
        })
        mask = _qb_change_mask(df)
        assert mask.tolist() == [True, True]

    def test_qb_change_mask_no_cols(self):
        df = pd.DataFrame({"x": [1, 2]})
        mask = _qb_change_mask(df)
        assert mask.tolist() == [False, False]


class TestDriftThresholds:
    def test_all_pass_when_no_data(self):
        checks = _check_drift_thresholds(
            weekly_ll=None, rolling_4_ll=None,
            weekly_brier=None, weekly_acc=None,
            ece=None, rolling_4_ece=None,
            high_conf_miss_rate=None, missing_weather_rate=None,
            qb_change_ll_gap=None,
            n_pred=16, n_graded=16, snapshot_checksum_ok=True,
            market_gap_ll=None, schema_ok=True, stale_days=2,
        )
        failing = [c for c in checks if c["status"] != "✅"]
        assert not failing, f"Checks that failed: {failing}"

    def test_weekly_ll_warning(self):
        checks = _check_drift_thresholds(
            weekly_ll=0.70, rolling_4_ll=None,
            weekly_brier=0.25, weekly_acc=0.50,
            ece=0.05, rolling_4_ece=None,
            high_conf_miss_rate=0.10, missing_weather_rate=0.30,
            qb_change_ll_gap=0.01,
            n_pred=16, n_graded=16, snapshot_checksum_ok=True,
            market_gap_ll=0.04, schema_ok=True, stale_days=2,
        )
        ll_check = [c for c in checks if c["check"] == "Weekly LL"][0]
        assert ll_check["status"] == "⚠️"

    def test_weekly_ll_pass(self):
        checks = _check_drift_thresholds(
            weekly_ll=0.60, rolling_4_ll=0.59,
            weekly_brier=0.22, weekly_acc=0.65,
            ece=0.05, rolling_4_ece=0.04,
            high_conf_miss_rate=0.10, missing_weather_rate=0.30,
            qb_change_ll_gap=0.01,
            n_pred=16, n_graded=16, snapshot_checksum_ok=True,
            market_gap_ll=0.04, schema_ok=True, stale_days=2,
        )
        ll_check = [c for c in checks if c["check"] == "Weekly LL"][0]
        assert ll_check["status"] == "✅"

    def test_high_conf_miss_rate_warning(self):
        checks = _check_drift_thresholds(
            weekly_ll=0.60, rolling_4_ll=None,
            weekly_brier=0.22, weekly_acc=0.65,
            ece=0.05, rolling_4_ece=None,
            high_conf_miss_rate=0.30, missing_weather_rate=0.30,
            qb_change_ll_gap=0.01,
            n_pred=16, n_graded=16, snapshot_checksum_ok=True,
            market_gap_ll=0.04, schema_ok=True, stale_days=2,
        )
        miss_check = [c for c in checks if c["check"] == "High-confidence miss rate (p≥0.80)"][0]
        assert miss_check["status"] == "⚠️"

    def test_schema_unchanged_pass(self):
        checks = _check_drift_thresholds(
            weekly_ll=0.60, rolling_4_ll=None,
            weekly_brier=0.22, weekly_acc=0.65,
            ece=0.05, rolling_4_ece=None,
            high_conf_miss_rate=0.10, missing_weather_rate=0.30,
            qb_change_ll_gap=0.01,
            n_pred=16, n_graded=16, snapshot_checksum_ok=True,
            market_gap_ll=0.04, schema_ok=True, stale_days=2,
        )
        schema_check = [c for c in checks if c["check"] == "Schema changes"][0]
        assert schema_check["status"] == "✅"

    def test_no_games_found_warning(self):
        checks = _check_drift_thresholds(
            weekly_ll=None, rolling_4_ll=None,
            weekly_brier=None, weekly_acc=None,
            ece=None, rolling_4_ece=None,
            high_conf_miss_rate=None, missing_weather_rate=None,
            qb_change_ll_gap=None,
            n_pred=0, n_graded=0, snapshot_checksum_ok=True,
            market_gap_ll=None, schema_ok=True, stale_days=None,
        )
        games_check = [c for c in checks if c["check"] == "No games found"][0]
        assert games_check["status"] == "⚠️"

    def test_stale_data_warning(self):
        checks = _check_drift_thresholds(
            weekly_ll=0.60, rolling_4_ll=None,
            weekly_brier=0.22, weekly_acc=0.65,
            ece=0.05, rolling_4_ece=None,
            high_conf_miss_rate=0.10, missing_weather_rate=0.30,
            qb_change_ll_gap=0.01,
            n_pred=16, n_graded=16, snapshot_checksum_ok=True,
            market_gap_ll=0.04, schema_ok=True, stale_days=10,
        )
        stale_check = [c for c in checks if c["check"] == "Stale data"][0]
        assert stale_check["status"] == "⚠️"

    def test_prediction_count_mismatch_warning(self):
        checks = _check_drift_thresholds(
            weekly_ll=0.60, rolling_4_ll=None,
            weekly_brier=0.22, weekly_acc=0.65,
            ece=0.05, rolling_4_ece=None,
            high_conf_miss_rate=0.10, missing_weather_rate=0.30,
            qb_change_ll_gap=0.01,
            n_pred=16, n_graded=10, snapshot_checksum_ok=True,
            market_gap_ll=0.04, schema_ok=True, stale_days=2,
        )
        count_check = [c for c in checks if c["check"] == "Prediction count match"][0]
        assert count_check["status"] == "⚠️"

    def test_checksum_mismatch_warning(self):
        checks = _check_drift_thresholds(
            weekly_ll=0.60, rolling_4_ll=None,
            weekly_brier=0.22, weekly_acc=0.65,
            ece=0.05, rolling_4_ece=None,
            high_conf_miss_rate=0.10, missing_weather_rate=0.30,
            qb_change_ll_gap=0.01,
            n_pred=16, n_graded=16, snapshot_checksum_ok=False,
            market_gap_ll=0.04, schema_ok=True, stale_days=2,
        )
        cs_check = [c for c in checks if c["check"] == "Published file checksum match"][0]
        assert cs_check["status"] == "⚠️"


class TestGetSnapshotInfo:
    def test_finds_snapshot(self):
        manifest = {
            "snapshots": [
                {"season": 2026, "week": 1, "mode": "live", "path": "a.csv"},
                {"season": 2026, "week": 2, "mode": "live", "path": "b.csv"},
            ]
        }
        info = _get_snapshot_info(manifest, 2026, 1)
        assert info is not None
        assert info["path"] == "a.csv"

    def test_not_found(self):
        manifest = {"snapshots": []}
        info = _get_snapshot_info(manifest, 2026, 1)
        assert info is None

    def test_mode_filter(self):
        manifest = {
            "snapshots": [
                {"season": 2026, "week": 1, "mode": "dry_run", "path": "a.csv"},
            ]
        }
        info = _get_snapshot_info(manifest, 2026, 1, mode="live")
        assert info is None


class TestLoadAllGradedGames:
    def test_empty_manifest(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('{"snapshots": []}')
        with patch("sportslab.evaluation.weekly_pipeline.MANIFEST_PATH",
                   manifest_path):
            result = _load_all_graded_games(2026)
            assert result.empty

    def test_no_graded_snapshots(self, tmp_path):
        manifest = {
            "snapshots": [
                {"season": 2026, "week": 1, "mode": "live", "path": str(tmp_path / "a.csv"),
                 "graded": True},
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        df = pd.DataFrame({"game_id": ["1"], "actual_home_win": [1],
                           "incumbent_home_win_prob": [0.6]})
        df.to_csv(tmp_path / "a.csv", index=False)
        with patch("sportslab.evaluation.weekly_pipeline.MANIFEST_PATH", manifest_path):
            result = _load_all_graded_games(2026)
            assert not result.empty
            assert len(result) == 1


class TestGenerateMonitoringReport:
    def _patch_paths(self, tmp_path):
        """Patch weekly_pipeline paths and monitoring dir."""
        history_path = tmp_path / "history.csv"
        manifest_path = tmp_path / "manifest.json"
        monitoring_dir = tmp_path
        manifest_path.write_text('{"manifest_version": 1, "snapshots": []}')
        return (
            patch("sportslab.evaluation.weekly_pipeline.HISTORY_PATH", history_path),
            patch("sportslab.evaluation.weekly_pipeline.MANIFEST_PATH", manifest_path),
            patch("sportslab.evaluation.monitoring_report.MONITORING_DIR", monitoring_dir),
        )

    def test_generates_report_missing_data(self, tmp_path):
        p1, p2, p3 = self._patch_paths(tmp_path)
        with p1, p2, p3:
            history_path = tmp_path / "history.csv"
            pd.DataFrame(columns=["season", "week", "n", "log_loss",
                                  "brier", "accuracy", "auc",
                                  "model_version", "snapshot", "mode",
                                  "graded_at"]).to_csv(
                history_path, index=False)
            path = generate_monitoring_report(season=2026, week=1)
            assert path is not None
            report = Path(path).read_text()
            assert "Weekly Monitoring Report" in report
            assert "2026 Week 1" in report
            assert INCUMBENT_VERSION in report
            assert "Drift Check Summary" in report
            assert "## Overview" in report

    def test_generates_report_with_output_arg(self, tmp_path):
        p1, p2, p3 = self._patch_paths(tmp_path)
        with p1, p2, p3:
            history_path = tmp_path / "history.csv"
            pd.DataFrame(columns=["season", "week", "n", "log_loss",
                                  "brier", "accuracy", "auc",
                                  "model_version", "snapshot", "mode",
                                  "graded_at"]).to_csv(
                history_path, index=False)
            out = tmp_path / "custom_report.md"
            path = generate_monitoring_report(season=2026, week=1, output=str(out))
            assert path == str(out)
            assert out.exists()

    def test_invalid_season_raises(self):
        with pytest.raises(ValueError, match="not allowed"):
            generate_monitoring_report(season=2019, week=1)

    def test_report_contains_section_headers(self, tmp_path):
        p1, p2, p3 = self._patch_paths(tmp_path)
        with p1, p2, p3:
            history_path = tmp_path / "history.csv"
            pd.DataFrame(columns=["season", "week", "n", "log_loss",
                                  "brier", "accuracy", "auc",
                                  "model_version", "snapshot", "mode",
                                  "graded_at"]).to_csv(
                history_path, index=False)
            path = generate_monitoring_report(season=2026, week=1)
            report = Path(path).read_text()
            for section in ["Overview", "Core Metrics", "Calibration Buckets",
                            "High-Confidence Predictions", "Subgroup Performance",
                            "Model-vs-Market Disagreement", "Operator Notes",
                            "Drift Check Summary"]:
                assert section in report, f"Missing section: {section}"

    def test_report_with_history_data(self, tmp_path):
        p1, p2, p3 = self._patch_paths(tmp_path)
        with p1, p2, p3:
            history_path = tmp_path / "history.csv"
            history = pd.DataFrame([
                {"season": 2026, "week": 1, "n": 16, "log_loss": 0.52,
                 "brier": 0.21, "accuracy": 0.69, "auc": 0.72,
                 "model_version": INCUMBENT_VERSION, "snapshot": "",
                 "mode": "live", "graded_at": "2026-09-10"},
            ])
            history.to_csv(history_path, index=False)
            path = generate_monitoring_report(season=2026, week=1)
            report = Path(path).read_text()
            assert "0.5200" in report
            assert "0.6900" in report


class TestFmt:
    def test_none(self):
        assert _fmt(None) == "—"

    def test_float(self):
        assert _fmt(0.12345) == "0.1235"

    def test_int(self):
        assert _fmt(42) == "42"
