"""Tests for the StatSpace FDR experiment module."""

import pandas as pd
import pytest

from sportslab.evaluation.statspace_fdr_experiment import (
    run_statspace_fdr_experiment,
)
from sportslab.features.statspace import (
    merge_team_season_metrics,
    schedule_to_nfl_historical_games,
)
from sportslab.features.statspace.nfl_branded_stats import (
    FraudDetectorWeights,
    _zscore,
    build_fraud_detector_rating,
)
from sportslab.features.statspace.nfl_elo import NFLEloEngine


@pytest.fixture
def dummy_schedule():
    return pd.DataFrame({
        "game_id": ["2021_01_GB_CHI"],
        "season": [2021],
        "week": [1],
        "gameday": ["2021-09-05"],
        "game_type": ["REG"],
        "away_team": ["CHI"],
        "home_team": ["GB"],
        "away_score": [10],
        "home_score": [24],
        "away_qb_id": ["00-0000001"],
        "home_qb_id": ["00-0000002"],
    })


@pytest.fixture
def dummy_pbp():
    return pd.DataFrame({
        "game_id": ["2021_01_GB_CHI"] * 10,
        "season": [2021] * 10,
        "week": [1] * 10,
        "posteam": ["GB"] * 5 + ["CHI"] * 5,
        "defteam": ["CHI"] * 5 + ["GB"] * 5,
        "epa": [0.5, -0.3, 0.8, -0.1, 0.2, -0.4, 0.3, -0.6, 0.1, -0.2],
        "qb_dropback": [1] * 5 + [1] * 5,
        "rush_attempt": [0] * 10,
        "success": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "interception": [0] * 10,
        "fumble_lost": [0] * 10,
    })


class TestFDRCore:
    def test_schedule_conversion(self, dummy_schedule):
        games = schedule_to_nfl_historical_games(dummy_schedule)
        assert len(games) == 1
        assert games[0].game_id == "2021_01_GB_CHI"
        assert games[0].home_team == "GB"
        assert games[0].away_team == "CHI"
        assert games[0].home_score == 24
        assert games[0].away_score == 10
        assert games[0].completed is True

    def test_schedule_conversion_missing_columns(self):
        bad = pd.DataFrame({"game_id": ["x"]})
        with pytest.raises(ValueError, match="missing required columns"):
            schedule_to_nfl_historical_games(bad)

    def test_fdr_produces_32_teams(self, dummy_schedule, dummy_pbp):
        games = schedule_to_nfl_historical_games(dummy_schedule)
        result, meta = build_fraud_detector_rating(
            games, pbp_df=dummy_pbp, season=2021,
            elo_engine=NFLEloEngine(),
        )
        assert not result.empty
        assert "fraud_detector_rating" in result.columns
        assert "team" in result.columns

    def test_fdr_default_weights(self):
        w = FraudDetectorWeights()
        assert abs(w.record_strength - 0.35) < 1e-6
        assert abs(w.underlying_quality + 0.85) < 1e-6

    def test_zscore_edge_cases(self):
        result_single = _zscore(pd.Series([0.0]))
        assert abs(result_single.iloc[0]) < 1e-10  # single value → 0.0 (ddof=0)
        result_empty = _zscore(pd.Series([], dtype=float))
        assert result_empty.empty
        result = _zscore(pd.Series([1.0, 2.0, 3.0]))
        assert abs(result.iloc[1]) < 1e-10

    def test_nfl_elo_engine_creatable(self):
        engine = NFLEloEngine()
        assert engine.config.initial_elo == 1505.0
        assert engine.config.k_factor == 20.0


class TestMergeHelper:
    def test_basic_merge(self):
        features = pd.DataFrame({
            "home_team": ["GB"],
            "away_team": ["CHI"],
            "season": [2021],
        })
        metric_df = pd.DataFrame({
            "team": ["GB"],
            "season": [2021],
            "fraud_detector_rating": [1.5],
        })
        result = merge_team_season_metrics(
            features, metric_df, prefix="fdr",
            value_columns=["fraud_detector_rating"],
        )
        assert "home_fdr_fraud_detector_rating" in result.columns
        assert result["home_fdr_fraud_detector_rating"].iloc[0] == 1.5

    def test_merge_no_prefix(self):
        features = pd.DataFrame({
            "home_team": ["GB"],
            "away_team": ["CHI"],
            "season": [2021],
        })
        metric = pd.DataFrame({
            "team": ["GB"],
            "season": [2021],
            "fdr": [2.0],
        })
        result = merge_team_season_metrics(
            features, metric, prefix="",
            value_columns=["fdr"],
        )
        assert "home_fdr" in result.columns
        assert result["home_fdr"].iloc[0] == 2.0

    def test_merge_default_value_columns(self):
        features = pd.DataFrame({
            "home_team": ["GB"],
            "away_team": ["CHI"],
            "season": [2021],
        })
        metric = pd.DataFrame({
            "team": ["GB"],
            "season": [2021],
            "fdr_score": [1.0],
        })
        result = merge_team_season_metrics(
            features, metric, prefix="test",
        )
        assert "home_test_fdr_score" in result.columns


class TestExperiment:
    def test_experiment_importable(self):
        from sportslab.evaluation import statspace_fdr_experiment
        assert hasattr(statspace_fdr_experiment, "run_statspace_fdr_experiment")

    def test_experiment_runs_and_produces_report(self, tmp_path):
        report = tmp_path / "test_fdr_report.md"
        r = run_statspace_fdr_experiment(
            ft_path="data/features/nfl/feature_table.parquet",
            report_path=str(report),
        )
        assert r == str(report)
        assert report.exists()
        content = report.read_text()
        assert "Promoted" in content or "No model" in content
        assert "Val LL" in content or "Validation" in content or "val" in content.lower()
        assert "FDR" in content
