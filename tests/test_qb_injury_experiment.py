"""Tests for QB injury flag experiment."""

from sportslab.evaluation.qb_injury_experiment import (
    QB_OUT_FEATURE,
    run_qb_injury_experiment,
)


class TestConstants:
    def test_qb_out_feature_defined(self):
        assert QB_OUT_FEATURE == "home_injuries_qb_out"


class TestExperiment:
    def test_importable(self):
        from sportslab.evaluation import qb_injury_experiment

        assert hasattr(qb_injury_experiment, "run_qb_injury_experiment")

    def test_folds_exclude_holdout(self):
        from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS

        for _, val_season in ROLLING_FOLDS:
            assert val_season != HOLDOUT_SEASON

    def test_experiment_runs(self, tmp_path):
        """run_qb_injury_experiment completes without error."""
        report_path = tmp_path / "qb_injury_test.md"
        result = run_qb_injury_experiment(
            feature_table_path="data/features/nfl/feature_table.parquet",
            report_path=str(report_path),
        )
        assert report_path.exists()
        assert "qb_injury_test.md" in result

    def test_cli_registered(self):
        from sportslab.cli import cli

        found = any("qb-injury" in c or "qb_injury" in c for c in cli.commands)
        assert found

    def test_qb_out_column_present(self):
        """The QB OUT feature exists in the injury features module."""
        from sportslab.features.injuries import INJURY_FEATURE_COLUMNS

        assert QB_OUT_FEATURE in INJURY_FEATURE_COLUMNS

    def test_folds_sequential_and_distinct(self):
        from sportslab.evaluation.experiment_config import ROLLING_FOLDS

        val_seasons = [v for _, v in ROLLING_FOLDS]
        assert len(val_seasons) == len(set(val_seasons))
        assert 2025 not in val_seasons
