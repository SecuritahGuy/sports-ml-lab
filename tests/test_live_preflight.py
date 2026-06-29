"""Tests for live-preflight module."""

import os
import tempfile

from sportslab.evaluation.live_preflight import (
    dry_run_smoke_test,
    run_live_preflight,
    validate_qb_csv,
)


def _write_temp_csv(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    path = f.name
    f.close()
    return path


class TestValidateQbCsv:
    def test_missing_file(self):
        issues = validate_qb_csv("/nonexistent/file.csv")
        assert any("not found" in i for i in issues)

    def test_none_path(self):
        issues = validate_qb_csv(None)
        assert issues == []

    def test_valid_csv(self):
        content = "game_id,home_qb_id,away_qb_id\n2025_01_ARI_ATL,J.McCarthy,C.Williams\n"
        path = _write_temp_csv(content)
        try:
            issues = validate_qb_csv(path)
            assert issues == [] or all("error" not in i.lower() for i in issues)
        finally:
            os.unlink(path)

    def test_malformed_csv(self):
        content = "game_id,home_qb_id\n2025_01_ARI_ATL,J.McCarthy\n"
        path = _write_temp_csv(content)
        try:
            issues = validate_qb_csv(path)
            assert any("error" in i.lower() or "missing" in i.lower() for i in issues)
        finally:
            os.unlink(path)


class TestDryRunSmokeTest:
    def test_returns_list(self):
        issues = dry_run_smoke_test()
        assert isinstance(issues, list)


class TestRunLivePreflight:
    def test_returns_list(self):
        issues = run_live_preflight()
        assert isinstance(issues, list)

    def test_with_qb_input(self):
        content = "game_id,home_qb_id,away_qb_id\n2025_01_ARI_ATL,J.McCarthy,C.Williams\n"
        path = _write_temp_csv(content)
        try:
            issues = run_live_preflight(qb_input_path=path)
            assert isinstance(issues, list)
        finally:
            os.unlink(path)

    def test_seasons_filter(self):
        issues = run_live_preflight(seasons=[2026])
        assert isinstance(issues, list)

    def test_cli_importable(self):
        import sportslab.cli  # noqa: F401

        assert "live_preflight_cmd" in dir(sportslab.cli)
