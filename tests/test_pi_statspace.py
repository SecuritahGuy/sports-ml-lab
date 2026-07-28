"""Tests for pi_statspace experiment."""

from sportslab.evaluation.pi_statspace_experiment import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    compute_fdr_features,
    compute_metric_features,
    run_pi_statspace_experiment,
)


def test_rolling_folds_structure():
    """Rolling folds use 2021-2024, holdout 2025."""
    assert HOLDOUT_SEASON == 2025
    assert len(ROLLING_FOLDS) == 3
    # Fold 1: train 2021, val 2022
    assert ROLLING_FOLDS[0] == ([2021], 2022)
    # Fold 2: train 2021-2022, val 2023
    assert ROLLING_FOLDS[1] == ([2021, 2022], 2023)
    # Fold 3: train 2021-2023, val 2024
    assert ROLLING_FOLDS[2] == ([2021, 2022, 2023], 2024)


def test_compute_fdr_features_importable():
    """compute_fdr_features is importable."""
    assert callable(compute_fdr_features)


def test_compute_metric_features_importable():
    """compute_metric_features is importable."""
    assert callable(compute_metric_features)


def test_run_experiment_importable():
    """run_pi_statspace_experiment is importable."""
    assert callable(run_pi_statspace_experiment)


def test_run_pi_statspace_generates_report():
    """Report contains expected sections."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        report_path = run_pi_statspace_experiment(
            ft_path="data/features/nfl/feature_table.parquet",
            report_path=f.name,
        )
        content = Path(report_path).read_text()
        assert "## Validation (Rolling-Origin 3-Fold)" in content
        assert "## Holdout (2025)" in content
        assert "## Decision" in content
        assert "Promoted" in content or "No model beats" in content


def test_cli_importable():
    """Module is importable via CLI pattern."""
    from sportslab import cli
    assert hasattr(cli, "pi_statspace_cmd")
