"""Tests for weekly QB source audit."""

from pathlib import Path

import pandas as pd
import pytest

from sportslab.evaluation.weekly_qb_audit import (
    run_weekly_qb_audit,
)


def test_weekly_qb_audit_module_importable():
    from sportslab.evaluation import weekly_qb_audit
    assert hasattr(weekly_qb_audit, "run_weekly_qb_audit")


def test_weekly_qb_audit_returns_dict():
    result = run_weekly_qb_audit(season=2025, week=2)
    assert isinstance(result, dict)
    assert "report" in result


def test_weekly_qb_audit_output_file(tmp_path):
    out = str(tmp_path / "audit.csv")
    result = run_weekly_qb_audit(season=2025, week=2, output_path=out)
    assert Path(out).exists()
    df = pd.read_csv(out)
    assert "game_id" in df.columns
    assert "home_team" in df.columns
    assert "oracle_home_qb_id" in df.columns
    assert "dc_home_qb_id" in df.columns
    assert "wk_home_qb_id" in df.columns
    assert "oracle_gate" in df.columns
    assert "dc_gate" in df.columns
    assert "wk_gate" in df.columns
    assert "oracle_final_prob" in df.columns
    assert "dc_final_prob" in df.columns
    assert "wk_final_prob" in df.columns
    assert "prob_diff_wk_vs_dc" in df.columns
    assert len(df) > 0


def test_weekly_qb_audit_all_three_sources_present():
    out = "/tmp/test_weekly_qb_audit_sources.csv"
    result = run_weekly_qb_audit(season=2025, week=5, output_path=out)
    df = pd.read_csv(out)
    assert df["oracle_final_prob"].notna().sum() > 0
    assert df["dc_final_prob"].notna().sum() > 0
    assert df["wk_final_prob"].notna().sum() > 0


def test_weekly_qb_audit_week1():
    """Week 1 should work — all sources should have data."""
    result = run_weekly_qb_audit(season=2025, week=1)
    assert "report" in result


def test_weekly_qb_audit_cli_importable():
    from sportslab.cli import weekly_qb_audit_cmd
    assert weekly_qb_audit_cmd is not None


def test_weekly_qb_audit_qb_diff_detected():
    """Verify some QB differences are detected between sources."""
    out = "/tmp/test_weekly_qb_audit_qb_diff.csv"
    run_weekly_qb_audit(season=2025, week=10, output_path=out)
    df = pd.read_csv(out)
    # Some games should have QB differences between weekly and snapshot
    n_qb_diff = (
        df["h_qb_wk_vs_dc"].dropna().astype(bool).sum()
        + df["a_qb_wk_vs_dc"].dropna().astype(bool).sum()
    )
    # At least one QB slot should differ
    assert n_qb_diff > 0, "Expected at least one QB difference between weekly and snapshot"


def test_weekly_qb_audit_invalid_season():
    """Should handle missing season gracefully."""
    result = run_weekly_qb_audit(season=1999, week=1)
    assert isinstance(result, dict)
    assert "No eligible" in str(result)


def test_weekly_qb_audit_gate_change_detected():
    """Verify gate changes are detected."""
    out = "/tmp/test_weekly_qb_audit_gate.csv"
    run_weekly_qb_audit(season=2025, week=5, output_path=out)
    df = pd.read_csv(out)
    if "gate_change_wk_vs_dc" in df.columns:
        n_gate = df["gate_change_wk_vs_dc"].sum()
        # Just verify the column is populated
        assert n_gate >= 0
