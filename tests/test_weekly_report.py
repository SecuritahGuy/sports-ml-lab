"""Tests for weekly report generation."""

from pathlib import Path

import pandas as pd
import pytest

from sportslab.evaluation.weekly_report import (
    _cautions_for_row,
    _detect_last_season_week,
    generate_weekly_report,
)

FULL_PATH = Path("reports/predictions/incumbent_predictions.csv")


@pytest.fixture(scope="session")
def full_df():
    return pd.read_csv(FULL_PATH)


class TestHelpers:
    def test_detect_last_season_week(self, full_df):
        season, week = _detect_last_season_week(full_df)
        assert season >= 2021
        assert 1 <= week <= 22

    def test_cautions_for_row_empty(self):
        row = pd.Series({"caution_qb_change": 0, "caution_early_season": 0})
        assert _cautions_for_row(row) == []

    def test_cautions_for_row_qb_change(self):
        row = pd.Series({"caution_qb_change": 1, "caution_early_season": 0})
        assert "QB change" in _cautions_for_row(row)

    def test_cautions_for_row_multiple(self):
        row = pd.Series({"caution_qb_change": 1, "caution_early_season": 1, "caution_neutral": 0})
        c = _cautions_for_row(row)
        assert len(c) == 2
        assert "QB change" in c
        assert "Early season (W1-4)" in c


class TestGenerateWeeklyReport:
    def test_latest_week_creates_report(self, tmp_path):
        out = str(tmp_path / "report.md")
        result = generate_weekly_report(output=out)
        assert result == out
        assert Path(out).exists()

    def test_specific_season_week(self, tmp_path):
        out = str(tmp_path / "report2.md")
        generate_weekly_report(season=2024, week=1, output=out)
        assert Path(out).exists()
        text = Path(out).read_text()
        assert "Week 1" in text
        assert "2024" in text

    def test_report_contains_model_metadata(self, tmp_path):
        out = str(tmp_path / "report3.md")
        generate_weekly_report(season=2024, week=1, output=out)
        text = Path(out).read_text()
        assert "Model version" in text
        assert "Holdout LL" in text
        assert "Experiment report" in text

    def test_report_contains_games_table(self, tmp_path):
        out = str(tmp_path / "report4.md")
        generate_weekly_report(season=2024, week=1, output=out)
        text = Path(out).read_text()
        assert "## Games" in text
        assert "Predicted Winner" in text
        assert "Home Win Prob" in text
        assert "Bucket" in text

    def test_report_contains_highest_confidence_section(self, tmp_path):
        out = str(tmp_path / "report5.md")
        generate_weekly_report(season=2024, week=1, output=out)
        text = Path(out).read_text()
        assert "Highest-Confidence" in text

    def test_report_contains_caveats(self, tmp_path):
        out = str(tmp_path / "report6.md")
        generate_weekly_report(season=2024, week=1, output=out)
        text = Path(out).read_text()
        assert "research output" in text
        assert "not betting" in text or "not betting" in text

    def test_report_contains_disclaimer(self, tmp_path):
        out = str(tmp_path / "report7.md")
        generate_weekly_report(season=2024, week=1, output=out)
        text = Path(out).read_text()
        assert "Do not use this output for gambling" in text

    def test_invalid_week_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Week must be in"):
            generate_weekly_report(season=2024, week=99, output=str(tmp_path / "bad.md"))

    def test_invalid_week_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Week must be in"):
            generate_weekly_report(season=2024, week=0, output=str(tmp_path / "bad0.md"))

    def test_invalid_season_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Season must be in"):
            generate_weekly_report(season=2019, week=1, output=str(tmp_path / "bad_season.md"))

    def test_invalid_season_future_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Season must be in"):
            generate_weekly_report(season=2030, week=1, output=str(tmp_path / "bad_future.md"))

    def test_empty_week_generates_empty_report(self, tmp_path):
        out = str(tmp_path / "empty.md")
        # Use a week that doesn't exist in the data
        max_week = pd.read_csv(FULL_PATH)["week"].max()
        generate_weekly_report(season=2021, week=max_week + 1, output=out)
        text = Path(out).read_text()
        assert "No games found" in text

    def test_cli_importable(self):
        from sportslab.evaluation.weekly_report import main

        assert main is not None

    def test_sportslab_cli_weekly_report_importable(self):
        from sportslab.cli import cli

        cmds = [c.name for c in cli.commands.values()]
        assert "weekly-report" in cmds
