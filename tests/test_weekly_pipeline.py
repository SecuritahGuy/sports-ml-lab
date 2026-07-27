"""Tests for weekly prediction pipeline (snapshot, grading, season report)."""

import numpy as np
import pandas as pd
import pytest

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


def _find_future_season_week() -> tuple:
    """Find (season, week) with future games (no result) in the feature table."""
    ft_path = "data/features/nfl/feature_table.parquet"
    ft = pd.read_parquet(ft_path)
    future = ft[ft["home_score"].isna() & ft["model_eligible"].fillna(False)]
    if not future.empty:
        row = future.iloc[0]
        return int(row["season"]), int(row["week"])
    pytest.skip("No future games in feature table")


class TestV3OverlayColumns:
    """Regression test: predict-week --mode dry_run output includes v3 overlay."""

    def test_dry_run_has_overlay_columns(self):
        """Verify dry_run snapshot contains all v3 overlay columns."""
        from pathlib import Path
        if not Path("data/features/nfl/feature_table.parquet").exists():
            pytest.skip("Feature table not available")
        season, week = _find_future_season_week()
        result = predict_week(season=season, week=week, mode="dry_run")
        assert "snapshot" in result, f"predict_week({season} w{week}) must return a snapshot path"
        snap_path = result["snapshot"]
        df = pd.read_csv(snap_path)

        expected_overlay_cols = [
            "overlay_gate_active",
            "overlay_gamma",
            "overlay_cap",
            "home_qb_adj",
            "away_qb_adj",
            "base_incumbent_prob",
        ]
        for col in expected_overlay_cols:
            assert col in df.columns, f"Missing v3 overlay column: {col}"

        assert df["overlay_gate_active"].dropna().isin([0, 1]).all(), \
            "overlay_gate_active must be 0 or 1"
        assert (df["overlay_gamma"] == 1.0).all(), \
            "overlay_gamma must be 1.0"
        assert (df["overlay_cap"] == 40).all(), \
            "overlay_cap must be 40"

    def test_overlay_changes_prob_when_gate_and_adj_nonzero(self):
        """When gate is active AND adjustments are non-zero, prob must differ."""
        from pathlib import Path
        if not Path("data/features/nfl/feature_table.parquet").exists():
            pytest.skip("Feature table not available")
        season, week = _find_future_season_week()
        result = predict_week(season=season, week=week, mode="dry_run")
        df = pd.read_csv(result["snapshot"])

        gated_with_adj = df[
            (df["overlay_gate_active"] == 1)
            & ((df["home_qb_adj"] != 0) | (df["away_qb_adj"] != 0))
        ]
        if len(gated_with_adj) == 0:
            return
        diff = (gated_with_adj["incumbent_home_win_prob"]
                - gated_with_adj["base_incumbent_prob"]).abs()
        assert (diff > 1e-10).all(), \
            f"All {len(gated_with_adj)} gated+adjusted games have identical probs"

    def test_overlay_preserves_prob_when_gate_inactive(self):
        """When gate is inactive, final prob must equal base prob."""
        from pathlib import Path
        if not Path("data/features/nfl/feature_table.parquet").exists():
            pytest.skip("Feature table not available")
        season, week = _find_future_season_week()
        result = predict_week(season=season, week=week, mode="dry_run")
        df = pd.read_csv(result["snapshot"])

        ungated = df[df["overlay_gate_active"] == 0]
        if len(ungated) == 0:
            return
        diff = (ungated["incumbent_home_win_prob"]
                - ungated["base_incumbent_prob"]).abs()
        assert (diff < 1e-10).all(), \
            "Ungated games should have identical base and final prob"
