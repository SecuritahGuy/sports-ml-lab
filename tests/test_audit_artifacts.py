import csv
import re
from pathlib import Path

import pytest


class TestAuditArtifacts:
    def test_module_importable(self):
        from sportslab.evaluation.audit_artifacts import gather_issues, run_audit

        assert run_audit is not None
        assert gather_issues is not None

    def test_gather_issues(self):
        from sportslab.evaluation.audit_artifacts import gather_issues

        issues = gather_issues()
        assert isinstance(issues, list)
        if issues:
            print(f"\nAudit issues found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

    def test_run_audit_creates_report(self):
        from sportslab.evaluation.audit_artifacts import BENCHMARKS, run_audit

        report_path = BENCHMARKS / "artifact_audit.md"
        if report_path.exists():
            report_path.unlink()
        run_audit()
        assert report_path.exists(), "Audit report was not created"
        text = report_path.read_text()
        assert "# Artifact Audit Report" in text

    def test_incumbent_holdout_constant_matches_leaderboard(self):
        from sportslab.evaluation.audit_artifacts import (
            INCUMBENT_HOLDOUT_LL,
            LEADERBOARD_PATH,
        )

        with open(LEADERBOARD_PATH) as f:
            rows = list(csv.DictReader(f))
        promoted = [r for r in rows if r["decision"] == "promoted"]
        assert any(
            abs(float(r["holdout_ll"]) - INCUMBENT_HOLDOUT_LL) < 0.001
            for r in promoted
            if r["holdout_ll"]
        )

    def test_comprehensive_efficiency_in_leaderboard(self):
        from sportslab.evaluation.audit_artifacts import LEADERBOARD_PATH

        text = LEADERBOARD_PATH.read_text()
        assert "comprehensive_efficiency" in text

    def test_history_summary_matches_entries(self):
        from sportslab.evaluation.audit_artifacts import HISTORY_PATH

        text = HISTORY_PATH.read_text()
        lines = text.splitlines()
        entry_count = sum(
            1
            for line in lines
            if (line.strip().startswith("### ") and not line.strip().startswith("### Summary"))
            or (
                line.strip().startswith("## ")
                and len(line.strip()) > 3
                and line.strip()[3].isdigit()
                and ". " in line.strip()
            )
        )
        total_line = [line for line in lines if "Total experiments" in line]
        assert total_line, "No summary table found in history"
        m = re.search(r"\b(\d+)\b", total_line[0])
        assert m, "Could not parse entry count from summary"
        parsed = int(m.group(1))
        assert parsed == entry_count, f"Summary says {parsed} entries, found {entry_count}"

    def test_leaderboard_no_diagnostic_promoted(self):
        from sportslab.evaluation.audit_artifacts import LEADERBOARD_PATH

        with open(LEADERBOARD_PATH) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            decision = (r.get("decision") or "").lower()
            experiment = (r.get("experiment") or "").lower()
            is_diag = "diagnostic" in decision or "holdout" in decision or "optuna" in experiment
            is_promoted = decision == "promoted"
            if is_diag and is_promoted:
                pytest.fail(f"Diagnostic '{r['experiment']}' labeled promoted")

    def test_all_referenced_reports_exist(self):
        from sportslab.evaluation.audit_artifacts import EXPERIMENTS, LEADERBOARD_PATH

        with open(LEADERBOARD_PATH) as f:
            rows = list(csv.DictReader(f))
        missing = []
        for r in rows:
            rp = r.get("report_path", "")
            if rp and rp != "nan" and rp.strip():
                ep = EXPERIMENTS / Path(rp).name
                if not ep.exists():
                    missing.append(rp)
        assert not missing, f"Missing reports: {missing}"
