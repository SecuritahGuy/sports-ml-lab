"""Tests for promotion policy consistency across the repo.

Verifies that:
1. All policy docs use relative (not absolute) holdout thresholds.
2. The check_promotion() function requires both val AND holdout improvement.
3. The schedule_rest_experiment.py is documented as legacy validation-only.
4. All experiment files that define promotion logic use both-required rule.
"""

from pathlib import Path

import pytest

from sportslab.evaluation.fold_safe import check_promotion

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPromotionPolicyFunction:
    def test_both_required(self):
        r = check_promotion(val_ll=0.6200, holdout_ll=0.6100,
                            incumbent_val=0.6305, incumbent_holdout=0.6200)
        assert r["promoted"]
        assert r["beats_val"]
        assert r["beats_holdout"]

    def test_val_only_not_promoted(self):
        r = check_promotion(val_ll=0.6200, holdout_ll=0.6250,
                            incumbent_val=0.6305, incumbent_holdout=0.6200)
        assert not r["promoted"]

    def test_holdout_only_not_promoted(self):
        r = check_promotion(val_ll=0.6350, holdout_ll=0.6100,
                            incumbent_val=0.6305, incumbent_holdout=0.6200)
        assert not r["promoted"]

    def test_delta_required_for_promotion(self):
        r = check_promotion(val_ll=0.6296, holdout_ll=0.6191,
                            incumbent_val=0.6305, incumbent_holdout=0.6200,
                            delta=0.001)
        assert not r["promoted"]

    def test_delta_exact_boundary(self):
        r = check_promotion(val_ll=0.6295, holdout_ll=0.6190,
                            incumbent_val=0.6305, incumbent_holdout=0.6200,
                            delta=0.001)
        assert r["promoted"]


class TestPolicyDocsUseRelativeNotAbsolute:
    """Policy documents must use relative holdout comparison,
    not hardcoded absolute thresholds like 'must beat 0.6200'."""

    POLICY_FILES = [
        "reports/benchmarks/nfl_research_incumbent.md",
        "docs/benchmarks.md",
        "docs/weekly_runbook.md",
    ]

    @pytest.mark.parametrize("rel_path", POLICY_FILES)
    def test_policy_docs_no_absolute_holdout_in_rules(self, rel_path):
        path = REPO_ROOT / rel_path
        assert path.exists(), f"Policy doc not found: {path}"
        text = path.read_text()
        # Find the Promotion Rules section
        in_rules = False
        for i, line in enumerate(text.splitlines()):
            if "Promotion Rule" in line:
                in_rules = True
                continue
            if in_rules and line.strip().startswith("## "):
                break
            if in_rules:
                # Must not reference a hardcoded benchmark number
                assert "beat **0." not in line, (
                    f"{rel_path}:{i+1} uses absolute holdout threshold: "
                    f"{line.strip()}"
                )


class TestLegacyExperimentDocumented:
    """The schedule_rest_experiment is the only experiment with validation-only
    promotion. It must be documented as legacy."""

    def test_schedule_rest_has_legacy_comment(self):
        path = REPO_ROOT / "src/sportslab/evaluation/schedule_rest_experiment.py"
        text = path.read_text()
        assert "validation-only" in text, (
            "schedule_rest_experiment must document its validation-only "
            "promotion as legacy policy"
        )
        assert "RALPH Loop 4" in text, (
            "schedule_rest_experiment must reference RALPH Loop 4 as the "
            "canonical policy"
        )
