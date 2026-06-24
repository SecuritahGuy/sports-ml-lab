"""Tests for injury features module and experiment."""

import pandas as pd
import pytest

from sportslab.evaluation.injury_features_experiment import (
    BEST_HFA,
    BEST_K,
    BEST_K_DEF,
    BEST_K_OFF,
    BEST_REG,
    run_injury_features_experiment,
)
from sportslab.features.injuries import (
    INJURY_FEATURE_COLUMNS,
    compute_injury_features,
    load_injury_data,
)


class TestInjuryFeatures:
    def test_feature_columns_defined(self):
        """INJURY_FEATURE_COLUMNS should have 19 entries."""
        assert len(INJURY_FEATURE_COLUMNS) == 19

    def test_columns_have_home_away(self):
        """Each feature type has home_ and away_ prefix variants."""
        assert any(c.startswith("home_") for c in INJURY_FEATURE_COLUMNS)
        assert any(c.startswith("away_") for c in INJURY_FEATURE_COLUMNS)

    def test_load_injury_data_returns_polars(self):
        """load_injury_data returns a polars DataFrame with expected columns."""
        data = load_injury_data(seasons=[2024], cache_dir="data/interim/nfl")
        import polars as pl
        assert isinstance(data, pl.DataFrame)
        assert "season" in data.columns
        assert "week" in data.columns
        assert "team" in data.columns
        assert "report_status" in data.columns

    def test_compute_features_adds_all_columns(self, sample_ft):
        """compute_injury_features adds all expected columns."""
        result = compute_injury_features(sample_ft)
        for c in INJURY_FEATURE_COLUMNS:
            assert c in result.columns, f"Missing column: {c}"

    def test_compute_features_non_negative(self, sample_ft):
        """Injury count features should be non-negative."""
        result = compute_injury_features(sample_ft)
        for c in INJURY_FEATURE_COLUMNS:
            if "diff" not in c:
                assert (result[c] >= 0).all(), f"Negative values in {c}"

    def test_compute_diff_columns(self, sample_ft):
        """Diff columns should be home minus away."""
        result = compute_injury_features(sample_ft)
        for prefix in ["injuries_out", "injuries_qb_out",
                        "injuries_skill_out", "injuries_ol_out",
                        "injuries_def_out"]:
            diff_col = f"{prefix}_diff"
            if diff_col in result.columns:
                home_col = f"home_{prefix}"
                away_col = f"away_{prefix}"
                pd.testing.assert_series_equal(
                    result[diff_col],
                    result[home_col] - result[away_col],
                    check_names=False,
                )

    def test_incumbent_params_frozen(self):
        """Incumbent params should match expected values."""
        assert BEST_K == 36
        assert BEST_HFA == 40
        assert BEST_REG == 0.1
        assert BEST_K_OFF == 52
        assert BEST_K_DEF == 20


class TestExperiment:
    def test_importable(self):
        """Module imports without error."""
        from sportslab.evaluation import injury_features_experiment
        assert hasattr(injury_features_experiment, "run_injury_features_experiment")

    def test_experiment_runs(self, tmp_path):
        """run_injury_features_experiment completes without error."""
        report_path = tmp_path / "injury_test.md"
        result = run_injury_features_experiment(
            feature_table_path="data/features/nfl/feature_table.parquet",
            report_path=str(report_path),
        )
        assert report_path.exists()
        assert "injury_test.md" in result

    def test_fold_holdout_safety(self):
        """Experiment config should exclude holdout from folds."""
        from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
        for _, val_season in ROLLING_FOLDS:
            assert val_season != HOLDOUT_SEASON

    def test_cli_importable(self):
        """CLI injury-features command group is registered."""
        from sportslab.cli import cli
        found = any(
            c.name == "injury-features_cmd"
            or "injury" in c.name
            for c in cli.commands.values()
        )
        if not found:
            found = any(
                "injury" in c.callback.__name__
                for c in cli.commands.values()
            )
        assert found, "injury-features command not found in CLI"


@pytest.fixture
def sample_ft():
    """Minimal feature table for injury computation tests."""
    data = {
        "season": [2024, 2024, 2024],
        "week": [1, 2, 3],
        "home_team": ["KC", "BUF", "SF"],
        "away_team": ["BAL", "MIA", "DAL"],
        "home_rest_days": [7, 7, 7],
        "away_rest_days": [7, 7, 7],
        "model_eligible": [True, True, True],
        "neutral": [False, False, False],
        "target": [1.0, 0.0, 1.0],
        "home_moneyline": [-200, -150, -300],
        "away_moneyline": [170, 130, 250],
        "spread_line": [-3.5, -2.5, -6.5],
        "home_spread_odds": [-110, -110, -110],
        "away_spread_odds": [-110, -110, -110],
        "total_line": [48.5, 45.5, 47.5],
        "over_odds": [-110, -110, -110],
        "under_odds": [-110, -110, -110],
    }
    return pd.DataFrame(data)
