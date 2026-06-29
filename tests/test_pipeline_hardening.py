"""Tests for pipeline hardening: modes, lifecycle, Oracle blocking, data-audit."""

import os

import pandas as pd
import pytest

from sportslab.evaluation.data_audit import run_data_audit
from sportslab.evaluation.weekly_pipeline import (
    LIVE_MODES,
    VALID_MODES,
    _get_snapshot_from_manifest,
    _read_manifest,
    _register_snapshot,
    _snapshot_path,
    _update_manifest_grade,
    _validate_mode,
    grade_week,
    predict_week,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, TARGET_COLUMN

# ── Mode validation ──


class TestModeValidation:
    def test_valid_modes(self):
        for m in VALID_MODES:
            _validate_mode(m)  # should not raise

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            _validate_mode("invalid")

    def test_live_modes_defined(self):
        assert "live" in LIVE_MODES
        assert len(LIVE_MODES) >= 1


# ── Snapshot lifecycle ──


class TestSnapshotLifecycle:
    def test_register_sets_initial_status(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        import sportslab.evaluation.weekly_pipeline as wp

        orig = wp.MANIFEST_PATH
        wp.MANIFEST_PATH = manifest_path
        try:
            snap = tmp_path / "test_snap.csv"
            snap.write_text("game_id,prob\nG1,0.6")
            sid = _register_snapshot(snap, 2026, 1, "live_pregame", 1, mode="live")
            manifest = _read_manifest()
            entry = manifest["snapshots"][0]
            assert entry["status"] == "initial"
            assert entry["mode"] == "live"
            assert entry["snapshot_id"] == sid
        finally:
            wp.MANIFEST_PATH = orig

    def test_supersedes_previous(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        import sportslab.evaluation.weekly_pipeline as wp

        orig = wp.MANIFEST_PATH
        wp.MANIFEST_PATH = manifest_path
        try:
            snap1 = tmp_path / "snap1.csv"
            snap1.write_text("game_id,prob\nG1,0.6")
            _register_snapshot(snap1, 2026, 1, "live_pregame", 1, mode="live")

            snap2 = tmp_path / "snap2.csv"
            snap2.write_text("game_id,prob\nG1,0.7")
            _register_snapshot(snap2, 2026, 1, "live_pregame", 1, mode="live")

            manifest = _read_manifest()
            assert len(manifest["snapshots"]) == 2
            # First entry should be superseded
            first = [s for s in manifest["snapshots"] if s["snapshot_id"] == "snap1"][0]
            assert first["status"] == "superseded"
            # Should retrieve latest non-superseded
            entry = _get_snapshot_from_manifest(2026, 1, mode="live")
            assert entry["snapshot_id"] == "snap2"
        finally:
            wp.MANIFEST_PATH = orig

    def test_grade_sets_graded_status(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        import sportslab.evaluation.weekly_pipeline as wp

        orig = wp.MANIFEST_PATH
        wp.MANIFEST_PATH = manifest_path
        try:
            snap = tmp_path / "grade_snap.csv"
            snap.write_text("game_id,prob\nG1,0.6")
            _register_snapshot(snap, 2026, 1, "live_pregame", 1, mode="live")
            metrics = {"n": 1, "log_loss": 0.6, "brier": 0.2,
                       "accuracy": 0.5, "auc": 0.5}
            _update_manifest_grade(2026, 1, metrics, mode="live")

            entry = _get_snapshot_from_manifest(2026, 1, mode="live")
            assert entry["status"] == "graded"
            assert entry["graded"] is True
            assert entry["grade_metrics"]["log_loss"] == 0.6
        finally:
            wp.MANIFEST_PATH = orig

    def test_mode_isolation(self, tmp_path):
        """Snapshots in different modes should not interfere."""
        manifest_path = tmp_path / "manifest.json"
        import sportslab.evaluation.weekly_pipeline as wp

        orig = wp.MANIFEST_PATH
        wp.MANIFEST_PATH = manifest_path
        try:
            live_snap = tmp_path / "live.csv"
            live_snap.write_text("game_id,prob\nG1,0.6")
            _register_snapshot(live_snap, 2026, 1, "live_pregame", 1, mode="live")

            dry_snap = tmp_path / "dry.csv"
            dry_snap.write_text("game_id,prob\nG1,0.6")
            _register_snapshot(dry_snap, 2026, 1, "oracle", 1, mode="dry_run")

            # Getting live should NOT return dry_run
            live_entry = _get_snapshot_from_manifest(2026, 1, mode="live")
            assert live_entry["mode"] == "live"

            dry_entry = _get_snapshot_from_manifest(2026, 1, mode="dry_run")
            assert dry_entry["mode"] == "dry_run"
        finally:
            wp.MANIFEST_PATH = orig


# ── Snapshot path ──


class TestSnapshotPath:
    def test_includes_mode(self):
        p = _snapshot_path(2026, 1, mode="dry_run")
        assert "dry_run" in p.name

    def test_live_default(self):
        p = _snapshot_path(2026, 1)
        assert "live" in p.name


# ── predict_week mode blocking ──


class TestPredictWeekModeBlocking:
    def test_live_mode_rejects_no_qb(self):
        with pytest.raises(ValueError, match="Oracle QB data not allowed in live mode"):
            predict_week(2026, 1, qb_input=None, mode="live")

    def test_dry_run_allows_no_qb(self, tmp_path):
        """dry_run mode should accept oracle QB (no qb_input)."""
        import sportslab.evaluation.predict_future as pf
        import sportslab.evaluation.weekly_pipeline as wp

        orig_m = wp.MANIFEST_PATH
        orig_h = wp.HISTORY_PATH
        orig_f = wp.FEATURE_TABLE_PATH
        orig_pf_f = pf.FEATURE_TABLE_PATH
        wp.MANIFEST_PATH = tmp_path / "manifest.json"
        wp.HISTORY_PATH = tmp_path / "history.csv"
        faux_ft = tmp_path / "no_feature_table.parquet"
        wp.FEATURE_TABLE_PATH = str(faux_ft)
        pf.FEATURE_TABLE_PATH = str(faux_ft)

        try:
            # Will fail because no feature table — but should NOT fail on mode validation
            with pytest.raises(FileNotFoundError, match="feature table|Feature table"):
                predict_week(2026, 1, mode="dry_run")
        finally:
            wp.MANIFEST_PATH = orig_m
            wp.HISTORY_PATH = orig_h
            wp.FEATURE_TABLE_PATH = orig_f
            pf.FEATURE_TABLE_PATH = orig_pf_f


# ── grade_week mode filtering ──


class TestGradeWeekModeFiltering:
    def test_grade_live_ignores_dry_run(self, tmp_path):
        import sportslab.evaluation.weekly_pipeline as wp

        orig_m = wp.MANIFEST_PATH
        orig_h = wp.HISTORY_PATH
        orig_f = wp.FEATURE_TABLE_PATH

        wp.MANIFEST_PATH = tmp_path / "manifest.json"
        wp.HISTORY_PATH = tmp_path / "history.csv"
        wp.FEATURE_TABLE_PATH = str(tmp_path / "no_feature_table.parquet")

        try:
            # Create a dry_run snapshot only
            dry_snap = tmp_path / "dry_snap.csv"
            dry_snap.write_text("game_id,prob\nG1,0.6")
            _register_snapshot(dry_snap, 2026, 1, "oracle", 1, mode="dry_run")

            # Grading in live mode should not find the dry_run snapshot
            with pytest.raises(FileNotFoundError,
                               match="No live snapshot found for 2026 week 1"):
                grade_week(2026, 1, mode="live")
        finally:
            wp.MANIFEST_PATH = orig_m
            wp.HISTORY_PATH = orig_h
            wp.FEATURE_TABLE_PATH = orig_f


# ── Data audit ──


class TestDataAudit:
    def test_importable(self):
        import sportslab.evaluation.data_audit  # noqa: F401

        assert True

    def test_callable(self):
        assert callable(run_data_audit)

    def test_run_without_file(self, tmp_path):
        """Should fail gracefully when schedule file doesn't exist."""
        import sportslab.evaluation.data_audit as da

        orig_s = da.SCHEDULES_PATH
        orig_f = da.FEATURE_TABLE_PATH
        da.SCHEDULES_PATH = str(tmp_path / "no_such_file.parquet")
        da.FEATURE_TABLE_PATH = str(tmp_path / "no_ft.parquet")
        try:
            issues = run_data_audit()
            assert len(issues) > 0
            assert any("Schedule file exists" in i for i in issues)
        finally:
            da.SCHEDULES_PATH = orig_s
            da.FEATURE_TABLE_PATH = orig_f

    def test_check_minimal(self):
        """_check marks issues correctly."""
        from sportslab.evaluation.data_audit import _check

        issues = []
        _check(True, "Good", issues)
        assert len(issues) == 0
        _check(False, "Bad", issues)
        assert len(issues) == 1
        assert "Bad" in issues[0]

    def test_cli_importable(self):
        import sportslab.cli  # noqa: F401

        assert "data_audit_cmd" in dir(sportslab.cli)

    def test_stale_data_past_games_no_scores(self, tmp_path):
        """Past-dated games without scores should be flagged."""
        from sportslab.evaluation.data_audit import _check_stale_data

        ft_path = tmp_path / "feature_table.parquet"
        df = pd.DataFrame({
            "game_id": ["G1"],
            "season": [2024],
            "week": [18],
            "gameday": ["2024-01-01"],
            "home_score": [pd.NA],
            "away_score": [pd.NA],
            "is_tie": [False],
            TARGET_COLUMN: [pd.NA],
            MODEL_ELIGIBLE_COLUMN: [False],
        })
        df.to_parquet(ft_path, index=False)
        issues = []
        _check_stale_data(df, ft_path, issues)
        assert any("Past-dated" in i for i in issues)

    def test_stale_data_file_age(self, tmp_path):
        """A very old feature table should be flagged (age check)."""
        from sportslab.evaluation.data_audit import _check_stale_data

        # Create a file with old mtime
        ft_path = tmp_path / "old_ft.parquet"
        df = pd.DataFrame({"game_id": [], "season": [], "week": [],
                           TARGET_COLUMN: [], MODEL_ELIGIBLE_COLUMN: []})
        df.to_parquet(ft_path, index=False)

        # Set mtime far in the past
        import time
        old_time = time.time() - (365 * 86400)  # 365 days ago
        os.utime(str(ft_path), (old_time, old_time))

        issues = []
        _check_stale_data(df, ft_path, issues)
        assert any("stale" in i.lower() or "fresh" not in i.lower() for i in issues)

    def test_partial_ingest_detection(self, tmp_path):
        """Row count mismatch between schedule and feature table is flagged."""
        from sportslab.evaluation.data_audit import _check_partial_ingest

        df_sched = pd.DataFrame({
            "game_id": ["G1", "G2", "G3"],
            "season": [2025, 2025, 2025],
        })
        df_ft = pd.DataFrame({
            "game_id": ["G1", "G2"],
            "season": [2025, 2025],
        })
        issues = []
        _check_partial_ingest(df_sched, df_ft, issues)
        assert any("diff" in i for i in issues)

    def test_partial_ingest_matches(self, tmp_path):
        """Matching row counts should not flag."""
        from sportslab.evaluation.data_audit import _check_partial_ingest

        df_sched = pd.DataFrame({
            "game_id": ["G1", "G2"],
            "season": [2025, 2025],
        })
        df_ft = pd.DataFrame({
            "game_id": ["G1", "G2"],
            "season": [2025, 2025],
        })
        issues = []
        _check_partial_ingest(df_sched, df_ft, issues)
        assert not any("diff" in i for i in issues)


# ── CLI imports ──


class TestCLIImports:
    def test_predict_week_has_mode(self):
        import sportslab.cli

        cmd = sportslab.cli.predict_week_cmd
        # Check for --mode option in click params
        param_names = [p.name for p in cmd.params]
        assert "mode" in param_names

    def test_data_audit_importable(self):
        import sportslab.cli

        assert "data_audit_cmd" in dir(sportslab.cli)

    def test_data_audit_params(self):
        import sportslab.cli

        cmd = sportslab.cli.data_audit_cmd
        param_names = [p.name for p in cmd.params]
        assert "seasons" in param_names

    def test_live_preflight_importable(self):
        import sportslab.cli

        assert "live_preflight_cmd" in dir(sportslab.cli)

    def test_live_preflight_params(self):
        import sportslab.cli

        cmd = sportslab.cli.live_preflight_cmd
        param_names = [p.name for p in cmd.params]
        assert "qb_input" in param_names
        assert "seasons" in param_names

    def test_live_preflight_help(self):
        from click.testing import CliRunner

        from sportslab.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["live-preflight", "--help"])
        assert result.exit_code == 0
        assert "pre-week checklist" in result.output


# ── predict_future mode ──


class TestPredictFutureMode:
    def test_live_requires_qb_input(self):
        from sportslab.evaluation.predict_future import predict_future

        with pytest.raises(ValueError, match="Oracle QB data not allowed in live mode"):
            predict_future(season=2026, week=1, mode="live")

    def test_dry_run_allows_oracle(self):
        import sportslab.evaluation.predict_future as pf
        from sportslab.evaluation.predict_future import predict_future

        orig = pf.FEATURE_TABLE_PATH
        pf.FEATURE_TABLE_PATH = "/tmp/no_table.parquet"
        try:
            with pytest.raises(FileNotFoundError, match="[Ff]eature table"):
                predict_future(season=2026, week=1, mode="dry_run", qb_input_path=None)
        finally:
            pf.FEATURE_TABLE_PATH = orig

    def test_run_predict_future_passes_mode(self):
        from sportslab.evaluation.predict_future import run_predict_future

        with pytest.raises(ValueError, match="Oracle QB data not allowed in live mode"):
            run_predict_future(season=2026, week=1, mode="live")

    def test_live_modes_importable(self):
        from sportslab.evaluation.predict_future import LIVE_MODES as LM

        assert "live" in LM


# ── _snapshot_id format ──


class TestSnapshotId:
    def test_includes_mode(self):
        from sportslab.evaluation.weekly_pipeline import _snapshot_id

        sid = _snapshot_id(2026, 1, "123456", mode="dry_run")
        assert "dry_run" in sid

    def test_live_default(self):
        from sportslab.evaluation.weekly_pipeline import _snapshot_id

        sid = _snapshot_id(2026, 1, "123456")
        assert "live" in sid
