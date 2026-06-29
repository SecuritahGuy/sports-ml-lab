"""Tests for historical rehearsal season module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.rehearsal_season import (
    _load_feature_table,
    get_season_weeks,
    rehearsal_paths,
    rehearse_season,
)

# ── Import / callable tests ──


class TestImport:
    def test_module_importable(self):
        import sportslab.evaluation.rehearsal_season  # noqa: F401

    def test_rehearse_season_callable(self):
        assert callable(rehearse_season)

    def test_get_season_weeks_callable(self):
        assert callable(get_season_weeks)

    def test_rehearsal_paths_is_context_manager(self):
        with rehearsal_paths():
            pass  # context manager enters and exits without error

    def test_cli_importable(self):
        import sportslab.cli  # noqa: F401
        assert "rehearsal_season_cmd" in dir(sportslab.cli)


# ── Path context manager tests ──


class TestRehearsalPaths:
    def test_swaps_paths(self):
        """Context manager changes weekly_pipeline globals to rehearsal dir."""
        import sportslab.evaluation.prediction_audit as pa
        import sportslab.evaluation.weekly_pipeline as wp

        orig_manifest = wp.MANIFEST_PATH
        with rehearsal_paths() as rehearsal_dir:
            assert wp.MANIFEST_PATH == rehearsal_dir / "manifest.json"
            assert wp.HISTORY_PATH == rehearsal_dir / "prediction_history.csv"
            assert wp.SNAPSHOT_DIR == rehearsal_dir / "snapshots"
            assert wp.REPORT_DIR == rehearsal_dir
            assert pa.REPORT_DIR == rehearsal_dir
            assert pa.DOCS_DIR == rehearsal_dir
        assert wp.MANIFEST_PATH == orig_manifest

    def test_restores_after_exception(self):
        """Paths are restored even if an exception occurs inside context."""
        import sportslab.evaluation.weekly_pipeline as wp

        orig_manifest = wp.MANIFEST_PATH
        try:
            with rehearsal_paths():
                raise ValueError("test error")
        except ValueError:
            pass
        assert wp.MANIFEST_PATH == orig_manifest

    def test_isolated_from_live(self):
        """Rehearsal writes should not affect live manifest."""
        import json

        import sportslab.evaluation.weekly_pipeline as wp

        # Snapshot live manifest state
        live_manifest = wp.MANIFEST_PATH.read_text() if wp.MANIFEST_PATH.exists() else None
        live_history = wp.HISTORY_PATH.read_text() if wp.HISTORY_PATH.exists() else None

        with rehearsal_paths():
            wp.MANIFEST_PATH.write_text(json.dumps({
                "manifest_version": 1,
                "snapshots": [{"snapshot_id": "rehearsal_only"}],
            }))
            wp.HISTORY_PATH.write_text("season,week,n,log_loss\n2025,1,10,0.62\n")

        # Live manifest should be unchanged
        if live_manifest is None:
            assert not wp.MANIFEST_PATH.exists() or wp.MANIFEST_PATH.read_text().strip() == ""
        else:
            assert wp.MANIFEST_PATH.exists()
            assert wp.MANIFEST_PATH.read_text() == live_manifest
        if live_history is None:
            assert not wp.HISTORY_PATH.exists() or wp.HISTORY_PATH.read_text().strip() == ""
        else:
            assert wp.HISTORY_PATH.exists()
            assert wp.HISTORY_PATH.read_text() == live_history

    def test_rehearsal_dir_created(self):
        """Context manager creates the rehearsal directory."""
        with rehearsal_paths() as rehearsal_dir:
            assert rehearsal_dir.exists()

    def test_snapshot_dir_created(self):
        """Snapshot subdirectory is created."""
        with rehearsal_paths() as rehearsal_dir:
            assert (rehearsal_dir / "snapshots").exists()


# ── Week detection tests ──


class TestGetSeasonWeeks:
    def test_with_real_data(self):
        """Integration: detect weeks from actual feature table."""
        try:
            weeks = get_season_weeks(season=2025)
            assert len(weeks) > 0
            assert all(isinstance(w, (int, np.integer)) for w in weeks)
        except FileNotFoundError:
            pytest.skip("Feature table not found")

    def test_with_mock_data(self):
        df = pd.DataFrame({
            "season": [2025] * 6 + [2026] * 3,
            "week": [1, 1, 2, 2, 3, 3, 1, 2, 3],
            "model_eligible": [True] * 9,
        })
        weeks = get_season_weeks(df, season=2025)
        assert weeks == [1, 2, 3]

    def test_mock_filters_model_eligible(self):
        df = pd.DataFrame({
            "season": [2025] * 4,
            "week": [1, 1, 2, 2],
            "model_eligible": [True, True, False, False],
        })
        weeks = get_season_weeks(df, season=2025)
        assert weeks == [1]  # week 2 all ineligible


# ── Load feature table ──


class TestLoadFeatureTable:
    def test_real_table_exists(self):
        fp = Path("data/features/nfl/feature_table.parquet")
        if fp.exists():
            df = _load_feature_table()
            assert len(df) > 0
            assert "season" in df.columns
            assert "week" in df.columns
            assert "game_id" in df.columns
        else:
            pytest.skip("Feature table not found")


# ── Full rehearsal integration tests ──


class TestRehearseSeason:
    def test_get_season_weeks_returns_list(self):
        """Verify 2025 has weeks to rehearse."""
        try:
            weeks = get_season_weeks(season=2025)
            assert len(weeks) > 0
        except FileNotFoundError:
            pytest.skip("Feature table not found")

    def test_returns_dict_with_keys(self):
        """Verify return structure of rehearse_season."""
        try:
            result = rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        assert "rehearsal_dir" in result
        assert "manifest" in result
        assert "history" in result
        assert "overall_metrics" in result

    def test_writes_output_files(self):
        """Full rehearsal creates expected artifact files."""
        try:
            result = rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        assert Path(result["rehearsal_dir"]).exists()
        assert Path(result["manifest"]).exists()
        assert Path(result["history"]).exists()
        assert result["overall_metrics"]["n"] > 0

    def test_manifest_has_rehearsal_snapshots(self):
        """Rehearsal manifest contains snapshot entries."""
        try:
            result = rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        manifest = Path(result["manifest"]).read_text()
        assert "rehearsal" in manifest.lower() or "week_" in manifest
        assert "snapshot_id" in manifest

    def test_history_has_entries(self):
        """Rehearsal history CSV has data."""
        try:
            result = rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        history = pd.read_csv(result["history"])
        assert len(history) > 0
        assert "log_loss" in history.columns
        assert "season" in history.columns
        assert history["season"].iloc[0] == 2025

    def test_audit_report_nonempty(self):
        """Rehearsal audit report contains calibration data."""
        try:
            result = rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        audit_path = Path(result["rehearsal_dir"]) / "audit_2025.md"
        assert audit_path.exists()
        content = audit_path.read_text()
        assert "Historical Rehearsal" in content
        assert "Calibration" in content
        assert "Confidence" in content
        assert "Worst Predictions" in content
        assert "Season-Week Performance" in content
        assert "checksums" in content
        assert "v2.0.0" in content

    def test_audit_includes_calibration_buckets(self):
        """Audit report has calibration data (not empty)."""
        try:
            result = rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        audit_path = Path(result["rehearsal_dir"]) / "audit_2025.md"
        content = audit_path.read_text()
        assert "| Bucket | N | Mean Pred" in content
        assert "| Confidence | N | Log Loss" in content
        assert "# | Game ID" in content

    def test_audit_distinguishes_rehearsal_mode(self):
        """Audit report header says Historical Rehearsal, not Prediction Audit."""
        try:
            result = rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        audit_path = Path(result["rehearsal_dir"]) / "audit_2025.md"
        content = audit_path.read_text()
        assert "Historical Rehearsal" in content
        assert "[Rehearsal]" in content

    def test_live_artifacts_untouched(self):
        """Run rehearsal and verify live manifest/history are unchanged."""
        import sportslab.evaluation.weekly_pipeline as wp

        # Snapshot live state
        live_manifest_before = (
            wp.MANIFEST_PATH.read_text() if wp.MANIFEST_PATH.exists() else None
        )
        live_history_before = (
            wp.HISTORY_PATH.read_text() if wp.HISTORY_PATH.exists() else None
        )

        try:
            rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        # Verify live state unchanged
        live_manifest_after = (
            wp.MANIFEST_PATH.read_text() if wp.MANIFEST_PATH.exists() else None
        )
        live_history_after = (
            wp.HISTORY_PATH.read_text() if wp.HISTORY_PATH.exists() else None
        )
        assert live_manifest_before == live_manifest_after
        assert live_history_before == live_history_after

    def test_snapshot_files_exist(self):
        """Rehearsal snapshot CSV files are present."""
        try:
            result = rehearse_season(season=2025)
        except FileNotFoundError:
            pytest.skip("Feature table not found")
        except Exception as e:
            if "Feature table not found" in str(e):
                pytest.skip("Feature table not found")
            raise

        snap_dir = Path(result["rehearsal_dir"]) / "snapshots"
        snapshots = list(snap_dir.glob("*.csv"))
        assert len(snapshots) > 0
        # Each snapshot should have actual_home_win column
        for sp in snapshots:
            df = pd.read_csv(sp)
            assert "actual_home_win" in df.columns
            assert "incumbent_home_win_prob" in df.columns

    def test_model_unchanged(self):
        """Incumbent constants are not modified by rehearsal."""
        from sportslab.evaluation.predict_incumbent import (
            INCUMBENT_HOLDOUT_LL,
            INCUMBENT_VERSION,
        )

        assert INCUMBENT_HOLDOUT_LL == 0.6262
        assert INCUMBENT_VERSION == "v2.0.0"
