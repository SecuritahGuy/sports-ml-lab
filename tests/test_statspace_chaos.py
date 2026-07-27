"""Tests for the StatSpace Chaos Rate experiment module."""

import pandas as pd
import pytest

from sportslab.evaluation.statspace_chaos_experiment import (
    run_statspace_chaos_experiment,
)
from sportslab.features.statspace import (
    merge_team_season_metrics,
)
from sportslab.features.statspace.nfl_chaos_rate import (
    ChaosRateWeights,
    _zscore,
    build_chaos_rate,
)


@pytest.fixture
def dummy_pbp():
    return pd.DataFrame({
        "game_id": ["2021_01_GB_CHI"] * 8,
        "season": [2021] * 8,
        "week": [1] * 8,
        "posteam": ["GB"] * 4 + ["CHI"] * 4,
        "defteam": ["CHI"] * 4 + ["GB"] * 4,
        "epa": [1.2, 0.5, -0.3, 0.8, -0.6, 0.3, -0.4, 0.1],
        "qb_dropback": [1] * 8,
        "rush_attempt": [0] * 8,
        "success": [1, 1, 0, 1, 0, 1, 0, 1],
        "down": [1, 3, 2, 1, 1, 2, 3, 1],
        "ydstogo": [10, 5, 8, 10, 10, 7, 6, 10],
        "yardline_100": [75, 40, 60, 25, 75, 50, 45, 20],
        "yards_gained": [8, 4, -1, 15, 2, 5, 3, 12],
        "touchdown": [0, 0, 0, 1, 0, 0, 0, 1],
        "interception": [0] * 8,
        "fumble_lost": [0] * 8,
        "sack": [0] * 8,
        "play_type": ["pass"] * 8,
        "no_play": [0] * 8,
        "two_point_attempt": [0] * 8,
        "extra_point_attempt": [0] * 8,
        "half_seconds_remaining": [900] * 8,
        "qb_kneel": [0] * 8,
    })


class TestChaosRateCore:
    def test_chaos_rate_columns_and_empty(self):
        """Chaos Rate returns expected schema even with minimal data."""
        empty = pd.DataFrame(columns=[
            "game_id", "season", "week", "posteam", "defteam", "epa",
            "qb_dropback", "rush_attempt", "success", "down", "ydstogo",
            "yardline_100", "yards_gained", "touchdown", "interception",
            "fumble_lost", "sack", "play_type", "no_play",
        ])
        result, meta = build_chaos_rate(empty)
        assert result.empty
        expected = ["chaos_rate", "team",
                    "defensive_epa_per_play_allowed",
                    "negative_epa_forced_rate"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_chaos_rate_default_weights(self):
        w = ChaosRateWeights()
        assert abs(w.negative_epa_forced_rate - 0.25) < 1e-6
        assert abs(w.turnover_forced_rate - 0.10) < 1e-6
        assert abs(w.defensive_epa_per_play_allowed_inverted - 0.20) < 1e-6
        assert abs(w.sack_rate - 0.15) < 1e-6

    def test_chaos_rate_empty_pbp(self):
        empty = pd.DataFrame(columns=["game_id", "season", "posteam", "defteam", "epa"])
        result, meta = build_chaos_rate(empty)
        assert result.empty

    def test_zscore_single(self):
        result = _zscore(pd.Series([5.0]))
        assert abs(result.iloc[0]) < 1e-10


class TestMergeHelper:
    def test_chaos_merge(self):
        features = pd.DataFrame({
            "home_team": ["GB"], "away_team": ["CHI"], "season": [2021],
        })
        metric = pd.DataFrame({
            "team": ["GB"], "season": [2021], "chaos_rate": [1.5],
        })
        result = merge_team_season_metrics(
            features, metric, prefix="chaos",
            value_columns=["chaos_rate"],
        )
        assert "home_chaos_chaos_rate" in result.columns
        assert result["home_chaos_chaos_rate"].iloc[0] == 1.5


class TestExperiment:
    def test_experiment_importable(self):
        from sportslab.evaluation import statspace_chaos_experiment
        assert hasattr(statspace_chaos_experiment, "run_statspace_chaos_experiment")

    def test_experiment_runs_and_produces_report(self, tmp_path):
        report = tmp_path / "test_chaos_report.md"
        r = run_statspace_chaos_experiment(
            ft_path="data/features/nfl/feature_table.parquet",
            report_path=str(report),
        )
        assert r == str(report)
        assert report.exists()
        content = report.read_text()
        assert "Promoted" in content or "No model" in content
        assert "Chaos" in content
