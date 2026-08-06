"""Tests for player recovery analysis and experiment."""

from sportslab.evaluation.recovery_experiment import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    run_recovery_experiment,
)
from sportslab.features.player_recovery import (
    BASELINE_GAMES,
    RECOVERY_MAX_WEEKS,
    build_player_game_table,
    identify_return_events,
    run_recovery_analysis,
)


def test_rolling_folds_structure():
    """Rolling folds use 2021-2024, holdout 2025."""
    assert HOLDOUT_SEASON == 2025
    assert len(ROLLING_FOLDS) == 3
    assert ROLLING_FOLDS[0] == ([2021], 2022)
    assert ROLLING_FOLDS[1] == ([2021, 2022], 2023)
    assert ROLLING_FOLDS[2] == ([2021, 2022, 2023], 2024)


def test_constants():
    """Recovery constants are reasonable."""
    assert BASELINE_GAMES == 4
    assert RECOVERY_MAX_WEEKS == 8


def test_build_player_game_table_importable():
    """build_player_game_table is callable."""
    assert callable(build_player_game_table)


def test_identify_return_events_importable():
    """identify_return_events is callable."""
    assert callable(identify_return_events)


def test_run_recovery_analysis_importable():
    """run_recovery_analysis is callable."""
    assert callable(run_recovery_analysis)


def test_run_recovery_experiment_importable():
    """run_recovery_experiment is callable."""
    assert callable(run_recovery_experiment)


def test_build_player_game_table_small():
    """Can build player-game table for 1 season."""
    pg = build_player_game_table([2024])
    assert len(pg) > 0
    assert "gsis_id" in pg.columns
    assert "fantasy_pts" in pg.columns
    assert "position" in pg.columns
    assert "total_yards" in pg.columns


def test_run_recovery_analysis_generates_report():
    """Report contains expected sections."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        report_path = run_recovery_analysis(
            seasons=[2024],
            min_games_out=2,
            report_path=f.name,
        )
        content = Path(report_path).read_text()
        assert "## Return Events by Position" in content
        assert "## Injury Type Breakdown" in content
        assert "## Recovery Curve Model" in content


def test_run_experiment_generates_report():
    """Experiment report contains expected sections."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        report_path = run_recovery_experiment(
            ft_path="data/features/nfl/feature_table.parquet",
            report_path=f.name,
        )
        content = Path(report_path).read_text()
        assert "## Validation (Rolling-Origin 3-Fold)" in content
        assert "## Holdout (2025)" in content
        assert "## Comparison vs Incumbent" in content
        assert "## Decision" in content


def test_cli_importable():
    """Module is importable via CLI pattern."""
    from sportslab import cli
    assert hasattr(cli, "player_recovery_cmd")
    assert hasattr(cli, "recovery_experiment_cmd")


def test_player_game_table_has_positions():
    """Player-game table has QB, RB, WR, TE positions."""
    pg = build_player_game_table([2024])
    for pos in ["QB", "RB", "WR", "TE"]:
        assert pos in pg["position"].values, f"Missing position: {pos}"
