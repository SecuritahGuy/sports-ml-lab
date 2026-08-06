"""Tests for return-from-injury experiment."""

from sportslab.evaluation.return_from_injury_experiment import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    RUST_COLUMNS,
    run_return_from_injury_experiment,
)


def test_rolling_folds_structure():
    """Rolling folds use 2021-2024, holdout 2025."""
    assert HOLDOUT_SEASON == 2025
    assert len(ROLLING_FOLDS) == 3
    assert ROLLING_FOLDS[0] == ([2021], 2022)
    assert ROLLING_FOLDS[1] == ([2021, 2022], 2023)
    assert ROLLING_FOLDS[2] == ([2021, 2022, 2023], 2024)


def test_rust_columns_defined():
    """All 8 rust columns are listed."""
    assert len(RUST_COLUMNS) == 8
    assert "home_rust_score" in RUST_COLUMNS
    assert "away_rust_score" in RUST_COLUMNS
    assert "home_rust_qb" in RUST_COLUMNS
    assert "away_rust_qb" in RUST_COLUMNS
    assert "home_rust_skill" in RUST_COLUMNS
    assert "away_rust_skill" in RUST_COLUMNS
    assert "home_rust_games_missed" in RUST_COLUMNS
    assert "away_rust_games_missed" in RUST_COLUMNS


def test_run_importable():
    """run_return_from_injury_experiment is callable."""
    assert callable(run_return_from_injury_experiment)


def test_run_experiment_generates_report():
    """Report contains expected sections."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        report_path = run_return_from_injury_experiment(
            ft_path="data/features/nfl/feature_table.parquet",
            report_path=f.name,
        )
        content = Path(report_path).read_text()
        assert "## Validation (Rolling-Origin 3-Fold)" in content
        assert "## Holdout (2025)" in content
        assert "## Decision" in content
        assert "No rust variant beats" in content


def test_cli_importable():
    """Module is importable via CLI pattern."""
    from sportslab import cli
    assert hasattr(cli, "return_from_injury_cmd")
