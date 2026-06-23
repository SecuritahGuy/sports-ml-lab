"""Tests for residual diagnostics module."""

from pathlib import Path

from sportslab.evaluation.residual_diagnostics import (
    run_residual_diagnostics,
)


def test_residual_diagnostics_importable():
    """Experiment function should be callable."""
    assert callable(run_residual_diagnostics)
    assert run_residual_diagnostics.__doc__ is not None


def test_residual_diagnostics_creates_report():
    """Running diagnostics should produce a report file."""
    tmp_report = str(Path("/tmp/test_residual_diagnostics.md"))
    result = run_residual_diagnostics(
        feature_table_path="data/features/nfl/feature_table.parquet",
        report_path=tmp_report,
    )
    assert result == tmp_report
    rp = Path(tmp_report)
    assert rp.exists()
    content = rp.read_text()
    assert "Overall Performance" in content
    assert "Calibration" in content
    assert "Residuals by Team" in content
    assert "Extreme Prediction Errors" in content
    rp.unlink(missing_ok=True)


def test_residual_diagnostics_references_incumbent():
    """Report should reference the incumbent holdout log loss."""
    tmp_report = str(Path("/tmp/test_residual_incumbent.md"))
    run_residual_diagnostics(report_path=tmp_report)
    content = Path(tmp_report).read_text()
    assert "0.6373" in content or "Incumbent" in content
    Path(tmp_report).unlink(missing_ok=True)
