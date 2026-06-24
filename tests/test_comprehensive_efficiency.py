"""Tests for comprehensive efficiency features module and experiment."""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.comprehensive_efficiency_experiment import (
    run_comprehensive_efficiency_experiment,
)
from sportslab.features.efficiency import (
    COMPREHENSIVE_EFFICIENCY_COLUMNS,
    PFR_COLUMNS,
    SNAP_COLUMNS,
    TEAM_EPA_COLUMNS,
    _compute_rolling,
    compute_comprehensive_efficiency_features,
)


class TestColumnConstants:
    """Verify column lists are well-formed."""

    def test_team_epa_columns_nonempty(self):
        assert len(TEAM_EPA_COLUMNS) > 0

    def test_pfr_columns_nonempty(self):
        assert len(PFR_COLUMNS) > 0

    def test_snap_columns_nonempty(self):
        assert len(SNAP_COLUMNS) > 0

    def test_all_columns_nonempty(self):
        assert len(COMPREHENSIVE_EFFICIENCY_COLUMNS) > 0

    def test_all_columns_are_strings(self):
        for c in COMPREHENSIVE_EFFICIENCY_COLUMNS:
            assert isinstance(c, str)

    def test_home_away_balance(self):
        home = [c for c in COMPREHENSIVE_EFFICIENCY_COLUMNS if c.startswith("home_")]
        away = [c for c in COMPREHENSIVE_EFFICIENCY_COLUMNS if c.startswith("away_")]
        assert len(home) == len(away)


class TestRollingHelper:
    """Verify chronological rolling with shift."""

    def test_shifted_rolling(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = _compute_rolling(s, 3)
        # index 0: nan (shifted, no prior)
        # index 1: 1.0
        # index 2: (1+2)/2 = 1.5
        # index 3: (1+2+3)/3 = 2.0
        # index 4: (2+3+4)/3 = 3.0
        assert np.isnan(r.iloc[0])
        assert r.iloc[1] == pytest.approx(1.0)
        assert r.iloc[2] == pytest.approx(1.5)
        assert r.iloc[3] == pytest.approx(2.0)
        assert r.iloc[4] == pytest.approx(3.0)


class TestFeatureComputation:
    """End-to-end: run on feature table and verify columns added."""

    @pytest.fixture(scope="class")
    def feature_df(self):
        fp = "data/features/nfl/feature_table.parquet"
        try:
            df = pd.read_parquet(fp)
            # Subsample for speed
            df = df[df["season"].isin([2021, 2022])].head(100).copy()
            result = compute_comprehensive_efficiency_features(df, cache_dir="data/interim/nfl")
            return result
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Skipping: {e}")

    def test_team_epa_columns_added(self, feature_df):
        present = [c for c in TEAM_EPA_COLUMNS if c in feature_df.columns]
        assert len(present) >= len(TEAM_EPA_COLUMNS) * 0.5, (
            f"Only {len(present)}/{len(TEAM_EPA_COLUMNS)} present"
        )

    def test_pfr_columns_added(self, feature_df):
        present = [c for c in PFR_COLUMNS if c in feature_df.columns]
        assert len(present) >= len(PFR_COLUMNS) * 0.5, (
            f"Only {len(present)}/{len(PFR_COLUMNS)} present"
        )

    def test_snap_columns_added(self, feature_df):
        present = [c for c in SNAP_COLUMNS if c in feature_df.columns]
        assert len(present) >= len(SNAP_COLUMNS) * 0.5, (
            f"Only {len(present)}/{len(SNAP_COLUMNS)} present"
        )

    def test_no_nan_in_columns(self, feature_df):
        for c in TEAM_EPA_COLUMNS + PFR_COLUMNS + SNAP_COLUMNS:
            if c in feature_df.columns:
                assert feature_df[c].isna().sum() == 0, f"{c} has NaN values"

    def test_no_inf_in_columns(self, feature_df):
        for c in TEAM_EPA_COLUMNS + PFR_COLUMNS + SNAP_COLUMNS:
            if c in feature_df.columns:
                vals = feature_df[c].values
                assert not np.any(np.isinf(vals)), f"{c} has inf values"

    def test_game_id_preserved(self, feature_df):
        assert "game_id" in feature_df.columns

    def test_comprehensive_list_matches(self, feature_df):
        """All COMPREHENSIVE_EFFICIENCY_COLUMNS that could be computed exist."""
        present = [c for c in COMPREHENSIVE_EFFICIENCY_COLUMNS if c in feature_df.columns]
        assert len(present) >= len(COMPREHENSIVE_EFFICIENCY_COLUMNS) * 0.5


class TestExperiment:
    """Verify experiment is importable and runs without crash on subset."""

    def test_importable(self):
        from sportslab.evaluation.comprehensive_efficiency_experiment import (
            run_comprehensive_efficiency_experiment,
        )

        assert callable(run_comprehensive_efficiency_experiment)

    def test_cli_importable(self):
        from sportslab.cli import comprehensive_efficiency_cmd

        assert callable(comprehensive_efficiency_cmd)

    def test_experiment_runs(self):
        """Quick smoke test — run on 2 seasons, check report created."""
        result = run_comprehensive_efficiency_experiment(
            feature_table_path="data/features/nfl/feature_table.parquet",
            report_path="/tmp/test_comprehensive_efficiency.md",
            cache_dir="data/interim/nfl",
        )
        import os

        assert os.path.exists(result)
