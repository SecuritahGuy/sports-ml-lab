"""Tests for Optuna joint Elo search experiment."""

from sportslab.evaluation.optuna_elo_search import MOV_TYPES, N_TRIALS, run_optuna_search


class TestConstants:
    def test_mov_types_defined(self):
        assert len(MOV_TYPES) >= 3
        assert "none" in MOV_TYPES
        assert "capped_linear" in MOV_TYPES

    def test_n_trials_positive(self):
        assert N_TRIALS >= 50


class TestExperiment:
    def test_importable(self):
        from sportslab.evaluation import optuna_elo_search

        assert hasattr(optuna_elo_search, "run_optuna_search")

    def test_folds_exclude_holdout(self):
        from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS

        for _, val_season in ROLLING_FOLDS:
            assert val_season != HOLDOUT_SEASON

    def test_folds_unique_val_seasons(self):
        from sportslab.evaluation.experiment_config import ROLLING_FOLDS

        val_seasons = [v for _, v in ROLLING_FOLDS]
        assert len(val_seasons) == len(set(val_seasons))
        assert 2025 not in val_seasons

    def test_short_search_works(self, tmp_path):
        """A short 3-trial search completes without error."""
        report_path = tmp_path / "optuna_test.md"
        result = run_optuna_search(
            feature_table_path="data/features/nfl/feature_table.parquet",
            report_path=str(report_path),
            n_trials=3,
        )
        assert report_path.exists()
        assert "optuna_test.md" in result

    def test_cli_registered(self):
        from sportslab.cli import cli

        assert any("optuna" in c for c in cli.commands)

    def test_best_trial_improves_over_random(self):
        """The objective function should find params that beat random (0.693)."""
        import optuna
        import pandas as pd

        from sportslab.evaluation.optuna_elo_search import _objective
        from sportslab.evaluation.season_regression_experiment import (
            build_team_regression_overrides,
        )

        df = pd.read_parquet("data/features/nfl/feature_table.parquet")
        team_overrides = build_team_regression_overrides(
            df,
            preseason_regression=0.1,
            qb_change_bonus=0.0,
        )

        # Try the incumbent params directly
        trial = optuna.trial.create_trial(
            params={
                "k": 36,
                "hfa": 40,
                "reg": 0.1,
                "decay": 32,
                "qb_bonus": 0.2,
                "k_off": 52,
                "k_def": 20,
                "mov_type": "capped_linear",
                "mov_scale": 0.05,
                "mov_cap": 2.0,
            },
            distributions={
                "k": optuna.distributions.IntDistribution(20, 60),
                "hfa": optuna.distributions.IntDistribution(10, 50),
                "reg": optuna.distributions.FloatDistribution(0.0, 0.5),
                "decay": optuna.distributions.IntDistribution(16, 64),
                "qb_bonus": optuna.distributions.FloatDistribution(0.0, 0.5),
                "k_off": optuna.distributions.IntDistribution(20, 80),
                "k_def": optuna.distributions.IntDistribution(10, 60),
                "mov_type": optuna.distributions.CategoricalDistribution(MOV_TYPES),
                "mov_scale": optuna.distributions.FloatDistribution(0.01, 0.15),
                "mov_cap": optuna.distributions.FloatDistribution(1.5, 5.0),
            },
            values=[0.6376],  # approximate incumbent val LL
        )
        val = _objective(trial, df, team_overrides)
        assert val < 0.66, f"Objective {val:.4f} should be well below random"
