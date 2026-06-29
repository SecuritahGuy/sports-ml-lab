"""Smoke tests for QB-adjusted Elo experiment."""

import pandas as pd
import pytest


class TestExperimentImportability:
    def test_module_imports(self):
        from sportslab.evaluation import qb_adjusted_elo_experiment
        assert hasattr(qb_adjusted_elo_experiment, "run_qb_adjusted_elo_experiment")

    def test_cli_importability(self):
        """Verify CLI commands are registered."""
        from sportslab import cli
        commands = {c.name for c in cli.cli.commands.values()}
        assert "build-qb-adjustments" in commands
        assert "qb-adjusted-elo" in commands
        assert "roster-strength" in commands

    def test_roster_strength_importability(self):
        from sportslab.ratings.roster_strength import (
            ALL_ROSTER_COLUMNS,
            ROSTER_STRENGTH_COLUMNS,
        )
        assert len(ROSTER_STRENGTH_COLUMNS) == 9
        assert "roster_qb_points" in ROSTER_STRENGTH_COLUMNS
        assert len(ALL_ROSTER_COLUMNS) == 18

    def test_qb_adjustment_columns_defined(self):
        from sportslab.features.qb_adjustment import QB_ADJUSTMENT_COLUMNS
        assert len(QB_ADJUSTMENT_COLUMNS) == 4
        assert "home_qb_adj" in QB_ADJUSTMENT_COLUMNS


class TestExperimentSafety:
    def test_no_season_before_2021(self):
        """Verify the experiment config does not use pre-2021 seasons."""
        from sportslab.evaluation.experiment_config import ALL_SEASONS, ROLLING_FOLDS
        for s in ALL_SEASONS:
            assert s >= 2021
        for train_seasons, val_season in ROLLING_FOLDS:
            for ts in train_seasons:
                assert ts >= 2021
            assert val_season >= 2022

    def test_feature_table_has_no_pre2021(self):
        """Run this test only if feature table exists."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            df = pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")
        assert df["season"].min() >= 2021

    def test_qb_adj_experiment_smoke(self, tmp_path):
        """Minimal smoke test — verify experiment runs end-to-end.

        Uses the real feature table.  Tests only that it completes
        without error, not the metric values.
        """
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found — cannot run smoke test")

        from sportslab.evaluation.qb_adjusted_elo_experiment import (
            run_qb_adjusted_elo_experiment,
        )
        report_p = tmp_path / "qb_adjusted_elo_test.md"
        run_qb_adjusted_elo_experiment(report_path=str(report_p))
        assert report_p.exists()
        content = report_p.read_text()
        assert "Decision" in content
        assert "Rolling-Origin Validation" in content
        assert "2025 Holdout" in content

    def test_cli_smoke_build_qb_adjustments(self, tmp_path):
        """Smoke test for build-qb-adjustments."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")

        from click.testing import CliRunner

        from sportslab.cli import cli

        runner = CliRunner()
        out_path = tmp_path / "qb_adj_test.parquet"
        result = runner.invoke(cli, ["build-qb-adjustments", "--output", str(out_path)])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        df = pd.read_parquet(out_path)
        assert "home_qb_adj" in df.columns
        assert "away_qb_adj" in df.columns
        assert len(df) > 0

    def test_cli_smoke_roster_strength(self, tmp_path):
        """Smoke test for roster-strength."""
        fp = "data/features/nfl/feature_table.parquet"
        try:
            pd.read_parquet(fp)
        except (FileNotFoundError, OSError):
            pytest.skip("Feature table not found")

        from click.testing import CliRunner

        from sportslab.cli import cli

        runner = CliRunner()
        out_path = tmp_path / "roster_test.parquet"
        result = runner.invoke(cli, ["roster-strength", "--output", str(out_path)])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        df = pd.read_parquet(out_path)
        assert "home_roster_qb_points" in df.columns
        assert "home_roster_ol_points" in df.columns
        assert len(df) > 0
