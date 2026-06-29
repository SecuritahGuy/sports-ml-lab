"""Tests for weekly prediction pipeline (snapshot, grading, season report)."""

import numpy as np
import pandas as pd

from sportslab.evaluation.weekly_pipeline import (
    _compute_metrics,
    _read_history,
    _season_report_content,
    _write_history,
    grade_week,
    predict_week,
    season_report,
)


class TestComputeMetrics:
    def test_all_correct(self):
        df = pd.DataFrame(
            {
                "actual_home_win": [1, 1, 1, 0, 0],
                "incumbent_home_win_prob": [0.99, 0.95, 0.80, 0.10, 0.05],
            }
        )
        m = _compute_metrics(df)
        assert m["n"] == 5
        assert m["accuracy"] == 1.0
        assert m["log_loss"] < 0.1

    def test_all_wrong(self):
        df = pd.DataFrame(
            {
                "actual_home_win": [1, 1, 0, 0],
                "incumbent_home_win_prob": [0.05, 0.10, 0.95, 0.90],
            }
        )
        m = _compute_metrics(df)
        assert m["n"] == 4
        assert m["accuracy"] == 0.0

    def test_empty(self):
        df = pd.DataFrame(
            {
                "actual_home_win": [np.nan, np.nan],
                "incumbent_home_win_prob": [0.5, 0.5],
            }
        )
        m = _compute_metrics(df)
        assert m["n"] == 0

    def test_single_class(self):
        df = pd.DataFrame(
            {
                "actual_home_win": [1, 1, 1],
                "incumbent_home_win_prob": [0.7, 0.8, 0.9],
            }
        )
        m = _compute_metrics(df)
        assert m["n"] == 3
        assert np.isnan(m["log_loss"])
        assert np.isnan(m["brier"])
        assert np.isnan(m["auc"])
        assert m["accuracy"] == 1.0

    def test_missing_actuals(self):
        df = pd.DataFrame(
            {
                "actual_home_win": [np.nan, np.nan],
                "incumbent_home_win_prob": [0.6, 0.7],
            }
        )
        m = _compute_metrics(df)
        assert m["n"] == 0


class TestHistory:
    def test_read_write_roundtrip(self, tmp_path):

        import sportslab.evaluation.weekly_pipeline as wp

        orig = wp.HISTORY_PATH
        wp.HISTORY_PATH = tmp_path / "history.csv"
        try:
            df = pd.DataFrame(
                {
                    "season": [2026],
                    "week": [1],
                    "n": [12],
                    "log_loss": [0.62],
                    "brier": [0.22],
                    "accuracy": [0.65],
                    "auc": [0.70],
                    "model_version": ["v3.0.0"],
                    "snapshot": ["test.csv"],
                    "graded_at": ["20260101_000000"],
                }
            )
            _write_history(df)
            loaded = _read_history()
            assert len(loaded) == 1
            assert loaded["season"].iloc[0] == 2026
            assert loaded["week"].iloc[0] == 1
        finally:
            wp.HISTORY_PATH = orig

    def test_read_empty(self):
        df = _read_history()
        assert "season" in df.columns
        assert len(df) == 0


class TestSeasonReport:
    def test_content_generation(self):
        df = pd.DataFrame(
            {
                "season": [2026, 2026, 2026],
                "week": [1, 2, 3],
                "n": [12, 13, 14],
                "log_loss": [0.62, 0.65, 0.59],
                "brier": [0.22, 0.23, 0.21],
                "accuracy": [0.65, 0.62, 0.71],
                "auc": [0.70, 0.68, 0.73],
                "model_version": ["v3.0.0"] * 3,
                "snapshot": ["a.csv", "b.csv", "c.csv"],
                "graded_at": ["t1", "t2", "t3"],
            }
        )
        content = _season_report_content(df, 2026)
        assert "Season Report" in content
        assert "Week" in content
        assert "1" in content
        assert "2" in content
        assert "0.62" in content  # mean log loss

    def test_empty_season(self):
        df = pd.DataFrame(
            columns=[
                "season",
                "week",
                "n",
                "log_loss",
                "brier",
                "accuracy",
                "auc",
                "model_version",
                "snapshot",
                "graded_at",
            ]
        )
        content = _season_report_content(df, 2026)
        assert "No graded weeks" in content


class TestCLICommands:
    def test_importable(self):
        import sportslab.evaluation.weekly_pipeline  # noqa: F401

        assert True

    def test_cli_importable(self):
        import sportslab.cli  # noqa: F401

        assert "predict_week_cmd" in dir(sportslab.cli)
        assert "grade_week_cmd" in dir(sportslab.cli)
        assert "season_report_cmd" in dir(sportslab.cli)


class TestFunctionsExist:
    def test_predict_week_is_callable(self):
        assert callable(predict_week)

    def test_grade_week_is_callable(self):
        assert callable(grade_week)

    def test_season_report_is_callable(self):
        assert callable(season_report)
