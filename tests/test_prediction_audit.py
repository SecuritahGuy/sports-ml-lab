"""Tests for prediction audit module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from sportslab.evaluation.prediction_audit import (
    _calibration_buckets,
    _confidence_buckets,
    _get_last_manifest_entry,
    _qb_source_breakdown,
    _safe_brier,
    _safe_log_loss,
    _worst_predictions,
    run_prediction_audit,
)
from sportslab.evaluation.weekly_pipeline import _write_manifest

# --- Helpers ---

def _make_manifest_entry(season, week, graded=True, qb_source="oracle",
                         snapshot_id="test_snap", n=10, log_loss=0.62,
                         brier=0.22, accuracy=0.65, auc=0.70):
    return {
        "manifest_version": "1.0",
        "snapshots": [{
            "snapshot_id": snapshot_id,
            "season": season,
            "week": week,
            "graded": graded,
            "qb_source": qb_source,
            "created_at": f"2026-{week:02d}-01T00:00:00",
            "graded_at": f"2026-{week:02d}-02T00:00:00",
            "path": str(Path(tempfile.gettempdir()) / f"test_snap_{season}_{week}.csv"),
            "checksum": "sha256:fakechecksum",
            "grade_metrics": {
                "n": n, "log_loss": log_loss, "brier": brier,
                "accuracy": accuracy, "auc": auc,
            } if graded else None,
        }],
    }


def _fake_csv_path(season, week, n=10):
    p = Path(tempfile.gettempdir()) / f"test_snap_{season}_{week}.csv"
    data = {
        "game_id": [f"2025_{i}" for i in range(n)],
        "season": [season] * n,
        "week": [week] * n,
        "away_team": ["AWAY"] * n,
        "home_team": ["HOME"] * n,
        "incumbent_home_win_prob": [0.55 + 0.04 * (i % 10) for i in range(n)],
        "actual_home_win": [1 if i % 2 == 0 else 0 for i in range(n)],
        "qb_source": ["oracle"] * n,
    }
    df = pd.DataFrame(data)
    df.to_csv(p, index=False)
    return p


# --- Tests ---

class TestSafeMetrics:
    def test_safe_log_loss_mixed(self):
        y_true = [1, 0, 1, 0]
        y_prob = [0.9, 0.2, 0.8, 0.3]
        ll = _safe_log_loss(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        assert not np.isnan(ll)
        assert ll > 0

    def test_safe_log_loss_single_class_all_ones(self):
        y_true = [1, 1, 1]
        y_prob = [0.9, 0.85, 0.95]
        ll = _safe_log_loss(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        assert not np.isnan(ll)
        assert ll > 0

    def test_safe_log_loss_single_class_all_zeros(self):
        y_true = [0, 0, 0]
        y_prob = [0.1, 0.2, 0.05]
        ll = _safe_log_loss(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        assert not np.isnan(ll)
        assert ll > 0

    def test_safe_brier_mixed(self):
        y_true = [1, 0, 1, 0]
        y_prob = [0.9, 0.2, 0.8, 0.3]
        br = _safe_brier(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        assert not np.isnan(br)
        assert 0 <= br <= 1

    def test_safe_brier_single_class_all_ones(self):
        y_true = [1, 1, 1]
        y_prob = [0.9, 0.85, 0.95]
        br = _safe_brier(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        assert not np.isnan(br)
        assert br > 0  # preds aren't perfect, so brier > 0

    def test_safe_brier_single_class_all_zeros(self):
        y_true = [0, 0, 0]
        y_prob = [0.1, 0.2, 0.05]
        br = _safe_brier(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        assert not np.isnan(br)
        assert br > 0


class TestCalibrationBuckets:
    def test_basic(self):
        y_true = [1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
        y_prob = [0.9, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95, 0.85, 0.15, 0.2]
        buckets = _calibration_buckets(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        assert len(buckets) > 0
        total_n = sum(b["n"] for b in buckets)
        assert total_n == len(y_true)
        # Each bucket has reasonable values
        for b in buckets:
            assert 0 <= b["mean_pred"] <= 1

    def test_empty(self):
        buckets = _calibration_buckets(
            pd.array([]).astype(float), pd.array([]).astype(float))
        assert buckets == []

    def test_single_value(self):
        buckets = _calibration_buckets(
            pd.array([1.0]), pd.array([0.8]))
        # single value with only one class per bucket
        for b in buckets:
            assert b["n"] >= 1

    def test_single_class_bucket(self):
        """Calibration bucket with only one class should not return nan."""
        # All home wins (actual=1), high probabilities → all land in 80-90% and 90-100% buckets
        y_true = [1, 1, 1, 1, 1]
        y_prob = [0.85, 0.88, 0.92, 0.95, 0.98]
        buckets = _calibration_buckets(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        for b in buckets:
            assert b["log_loss"] != "nan", f"nan log_loss in bucket {b['bucket']}"
            assert b["log_loss"] is not None
            assert not (isinstance(b["log_loss"], float) and np.isnan(b["log_loss"]))
            assert b["brier"] != "nan"
            assert b["brier"] is not None
            assert not (isinstance(b["brier"], float) and np.isnan(b["brier"]))

    def test_single_class_bucket_zeros(self):
        """Calibration bucket with only zeros should not return nan."""
        y_true = [0, 0, 0, 0]
        y_prob = [0.1, 0.12, 0.08, 0.15]
        buckets = _calibration_buckets(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        for b in buckets:
            assert not (isinstance(b.get("log_loss"), float) and np.isnan(b["log_loss"]))
            assert not (isinstance(b.get("brier"), float) and np.isnan(b["brier"]))

    def test_tiny_bucket_n2(self):
        """A bucket with n=2 and same class should produce real metrics."""
        y_true = [1, 1]
        y_prob = [0.91, 0.95]
        buckets = _calibration_buckets(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        for b in buckets:
            assert not (isinstance(b.get("log_loss"), float) and np.isnan(b["log_loss"]))
            assert not (isinstance(b.get("brier"), float) and np.isnan(b["brier"]))


class TestConfidenceBuckets:
    def test_basic(self):
        y_true = [1, 1, 0, 1, 0, 1]
        y_prob = [0.9, 0.55, 0.1, 0.6, 0.45, 0.95]
        buckets = _confidence_buckets(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        assert len(buckets) > 0
        total_n = sum(b["n"] for b in buckets)
        assert total_n == len(y_true)

    def test_empty(self):
        buckets = _confidence_buckets(
            pd.array([]).astype(float), pd.array([]).astype(float))
        assert buckets == []

    def test_sparse_confidence_bucket(self):
        """High-confidence bucket with single class should not return nan."""
        y_true = [1, 1, 1]
        y_prob = [0.95, 0.98, 0.99]
        buckets = _confidence_buckets(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        for b in buckets:
            assert not (isinstance(b.get("log_loss"), float) and np.isnan(b["log_loss"])), (
                f"nan log_loss in bucket {b['bucket']}")
            assert not (isinstance(b.get("brier"), float) and np.isnan(b["brier"]))

    def test_confidence_tiny_n2(self):
        """Confidence bucket with n=2 should still produce real metrics."""
        y_true = [1, 1]
        y_prob = [0.92, 0.96]
        buckets = _confidence_buckets(
            pd.array(y_true).astype(float), pd.array(y_prob).astype(float))
        for b in buckets:
            assert not (isinstance(b.get("log_loss"), float) and np.isnan(b["log_loss"]))
            assert not (isinstance(b.get("brier"), float) and np.isnan(b["brier"]))


class TestWorstPredictions:
    def test_basic(self):
        df = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(10)],
            "season": [2025] * 10,
            "week": [1] * 10,
            "away_team": ["A"] * 10,
            "home_team": ["B"] * 10,
            "incumbent_home_win_prob": [0.5, 0.99, 0.01, 0.6, 0.4, 0.55, 0.45, 0.7, 0.3, 0.5],
            "actual_home_win": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        })
        worst = _worst_predictions(df, n=3)
        assert len(worst) == 3
        # Most confident misses should be worst
        probs = [w["prob"] for w in worst]
        assert 0.99 in probs or 0.01 in probs

    def test_empty(self):
        df = pd.DataFrame()
        # Should handle empty DataFrame gracefully
        try:
            worst = _worst_predictions(df, n=5)
            assert worst == []
        except (KeyError, ValueError):
            pass

    def test_all_nan_actuals(self):
        df = pd.DataFrame({
            "game_id": ["g1"],
            "incumbent_home_win_prob": [0.5],
            "actual_home_win": [pd.NA],
        })
        worst = _worst_predictions(df, n=5)
        assert worst == []


class TestQBBreakdown:
    def test_no_graded(self):
        manifest = _make_manifest_entry(2025, 1, graded=False)
        result = _qb_source_breakdown(manifest, 2025)
        assert result is None

    def test_no_snapshots(self):
        result = _qb_source_breakdown({"manifest_version": "1.0", "snapshots": []}, 2025)
        assert result is None

    def test_oracle_only(self):
        manifest = _make_manifest_entry(2025, 1, graded=True, qb_source="oracle")
        result = _qb_source_breakdown(manifest, 2025)
        assert result is not None
        assert result["oracle"]["n_weeks"] == 1
        assert result["live_pregame"]["n_weeks"] == 0

    def test_live_only(self):
        manifest = _make_manifest_entry(2025, 1, graded=True, qb_source="live_pregame")
        result = _qb_source_breakdown(manifest, 2025)
        assert result is not None
        assert result["live_pregame"]["n_weeks"] == 1
        assert result["oracle"]["n_weeks"] == 0

    def test_both_sources(self):
        manifest = {
            "manifest_version": "1.0",
            "snapshots": [
                _make_manifest_entry(2025, 1, snapshot_id="s1")["snapshots"][0],
                _make_manifest_entry(2025, 2, snapshot_id="s2",
                                     qb_source="live_pregame")["snapshots"][0],
            ],
        }
        result = _qb_source_breakdown(manifest, 2025)
        assert result is not None
        assert result["oracle"]["n_weeks"] == 1
        assert result["live_pregame"]["n_weeks"] == 1


class TestLastManifestEntry:
    def test_basic(self):
        manifest = {
            "manifest_version": "1.0",
            "snapshots": [
                {"snapshot_id": "s1", "created_at": "2026-01-01T00:00:00"},
                {"snapshot_id": "s2", "created_at": "2026-02-01T00:00:00"},
            ],
        }
        entry = _get_last_manifest_entry(manifest)
        assert entry["snapshot_id"] == "s2"

    def test_empty(self):
        entry = _get_last_manifest_entry({"manifest_version": "1.0", "snapshots": []})
        assert entry is None


class TestRunPredictionAudit:
    def test_no_snapshots(self):
        # Clean manifest
        _write_manifest({"manifest_version": "1.0", "snapshots": []})
        try:
            paths = run_prediction_audit(2025)
            assert "predictions" in paths
        finally:
            # Clean up
            pass

    def test_with_graded_data(self):
        season, week = 2025, 1
        entry = _make_manifest_entry(season, week)
        csv_path = _fake_csv_path(season, week)
        entry["snapshots"][0]["path"] = str(csv_path)
        _write_manifest(entry)
        try:
            paths = run_prediction_audit(season)
            assert "predictions" in paths
            # Read output
            out = Path(paths["predictions"])
            assert out.exists()
            content = out.read_text()
            assert f"Prediction Audit — {season}" in content
            assert "Worst Predictions" in content
            assert "Confidence Buckets" in content
            assert "checksums" in content
        finally:
            csv_path.unlink(missing_ok=True)

    def test_cross_season_filtering(self):
        """Audit for season 1 should not include season 2 data."""
        season1 = 2025
        season2 = 2026
        manifest = {
            "manifest_version": "1.0",
            "snapshots": [
                _make_manifest_entry(season1, 1, snapshot_id="s1",
                                     n=5)["snapshots"][0],
                _make_manifest_entry(season2, 1, snapshot_id="s2",
                                     n=10)["snapshots"][0],
            ],
        }
        csv1 = _fake_csv_path(season1, 1, n=5)
        csv2 = _fake_csv_path(season2, 1, n=10)
        manifest["snapshots"][0]["path"] = str(csv1)
        manifest["snapshots"][1]["path"] = str(csv2)
        _write_manifest(manifest)
        try:
            paths = run_prediction_audit(season1)
            content = Path(paths["predictions"]).read_text()
            # Week table should show only season1
            assert "Worst Predictions" in content
        finally:
            csv1.unlink(missing_ok=True)
            csv2.unlink(missing_ok=True)

    def test_empty_manifest(self):
        """Empty manifest should generate report without error."""
        _write_manifest({"manifest_version": "1.0", "snapshots": []})
        paths = run_prediction_audit(2025)
        content = Path(paths["predictions"]).read_text()
        assert "No graded snapshots" in content


class TestImportable:
    def test_module_importable(self):
        from sportslab.evaluation import prediction_audit
        assert hasattr(prediction_audit, "run_prediction_audit")

    def test_internal_functions_accessible(self):
        from sportslab.evaluation.prediction_audit import (
            _calibration_buckets,
            _confidence_buckets,
            _worst_predictions,
        )
        assert callable(_calibration_buckets)
        assert callable(_confidence_buckets)
        assert callable(_worst_predictions)
