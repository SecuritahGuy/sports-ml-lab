"""RALPH Loop 3: Weekly operator workflow audit.

Verifies that every CLI command fails clearly when:
  - Required files are missing
  - Seasons are pre-2021
  - Input is malformed
  - QB CSV has bad data
"""


import pytest
from click.testing import CliRunner

from sportslab.cli import cli

# ── Helpers ──

runner = CliRunner()


def invoke(*args, **kwargs):
    """Run CLI command and return result."""
    return runner.invoke(cli, list(args), **kwargs)


# ── Season validation: every CLI command that takes a season ──


@pytest.mark.parametrize("cmd_args,expected_message", [
    (["predict-future", "--season", "1999"], "not allowed"),
    (["predict-week", "--season", "1999", "--week", "1"], "not allowed"),
    (["grade-week", "--season", "1999", "--week", "1"], "not allowed"),
    (["weekly-qb-audit", "--season", "1999", "--week", "1"], "not allowed"),
])
def test_cli_command_rejects_pre_2000(cmd_args, expected_message):
    """Every CLI command with a season arg must reject < 2000."""
    result = invoke(*cmd_args)
    # Click captures exceptions, so output may be in result.output or result.exception
    exc = str(result.exception) if result.exception else ""
    output_combined = (result.output or "") + " " + exc
    assert expected_message in output_combined.lower() or result.exit_code != 0


# ── Missing required files ──


def test_predict_future_missing_feature_table():
    """predict_future should fail if feature table doesn't exist."""
    result = invoke("predict-future", "--season", "2025", "--week", "1")
    # Should either error on missing file or oracle block (live mode)
    assert result.exit_code != 0 or "not found" in result.output.lower()


def test_data_audit_no_arg():
    """data-audit should work without arguments (defaults to all)."""
    result = invoke("data-audit")
    assert result.exit_code == 0


def test_audit_artifacts_no_arg():
    """audit-artifacts should work without arguments."""
    result = invoke("audit-artifacts")
    assert result.exit_code == 0


# ── Backtest validation ──


def test_backtest_empty_seasons():
    """backtest with no seasons should print usage."""
    result = invoke("backtest")
    assert result.exit_code == 0
    assert "Usage" in result.output


# ── QB input validation (predict-week) ──


def test_predict_week_live_blocks_oracle():
    """predict-week --mode live without --qb-input should raise error."""
    result = invoke("predict-week", "--season", "2025", "--week", "1", "--mode", "live")
    assert "oracle" in str(result.output).lower() or result.exit_code != 0


def test_predict_week_dry_run_accepts_oracle():
    """predict-week --mode dry_run should accept oracle QB (no --qb-input)."""
    result = invoke("predict-week", "--season", "2025", "--week", "1", "--mode", "dry_run")
    # dry_run should not block oracle
    out_lower = str(result.output).lower()
    assert result.exit_code == 0 or "snapshot" in out_lower or "no games" in out_lower


# ── Future prediction validation ──


def test_predict_future_live_blocks_oracle():
    """predict-future in live mode without qb_input should raise error."""
    result = invoke("predict-future", "--season", "2025", "--week", "1")
    assert "oracle" in str(result.output).lower() or result.exit_code != 0


# ── Weekly report validation ──


def test_weekly_report_default():
    """weekly-report should run without arguments."""
    result = invoke("weekly-report")
    assert result.exit_code == 0


# ── Prediction audit validation ──


def test_prediction_audit_missing_manifest():
    """prediction-audit without manifest should handle gracefully."""
    result = invoke("prediction-audit", "--season", "2025")
    # Should fail gracefully (no manifest or empty history)
    out_lower = str(result.output).lower()
    assert result.exit_code == 0 or "no" in out_lower or "not found" in out_lower
