"""Tests for weekly report generation (incumbent and future predictions)."""

import os
import tempfile

import pandas as pd
import pytest

from sportslab.evaluation.weekly_report import (
    _cautions_for_row,
    _detect_last_season_week,
    generate_weekly_report,
)


def test_detect_last_season_week():
    """Detects correct latest season and week."""
    df = pd.DataFrame({
        "season": [2024, 2025, 2025],
        "week": [18, 1, 22],
    })
    s, w = _detect_last_season_week(df)
    assert s == 2025
    assert w == 22


def test_cautions_for_row_no_cautions():
    """No caution flags returns empty list."""
    row = pd.Series({"caution_qb_change": 0, "caution_early_season": 0})
    assert _cautions_for_row(row) == []


def test_cautions_for_row_with_flags():
    """Active caution flags are returned as labels."""
    row = pd.Series({
        "caution_qb_change": 1,
        "caution_early_season": 0,
        "caution_neutral": 0,
        "caution_missing_features": 0,
        "caution_model_market_disagreement": 1,
    })
    flags = _cautions_for_row(row)
    assert "QB change" in flags
    assert "Model-market disagreement" in flags
    assert len(flags) == 2


def test_cautions_for_row_missing_column():
    """Missing caution column is handled gracefully."""
    row = pd.Series({})
    assert _cautions_for_row(row) == []


def test_generate_weekly_report_incumbent():
    """Generates report from incumbent predictions CSV."""
    # Create temp prediction CSV spanning multiple seasons
    df = pd.DataFrame({
        "game_id": [f"2025_{i}" for i in range(16)],
        "season": [2025] * 16,
        "week": [1] * 16,
        "gameday": ["2025-09-04"] * 16,
        "away_team": ["DAL"] * 16,
        "home_team": ["PHI"] * 16,
        "incumbent_home_win_prob": [0.5 + i * 0.03 for i in range(16)],
        "predicted_winner": ["PHI"] * 16,
        "confidence_bucket": ["55-60"] * 16,
        "home_qb_id": ["QB1"] * 16,
        "away_qb_id": ["QB2"] * 16,
        "caution_qb_change": [0] * 16,
        "caution_early_season": [1] * 16,
    })
    fp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    fp.write(df.to_csv(index=False))
    input_path = fp.name
    fp.close()
    try:
        path = generate_weekly_report(season=2025, week=1,
                                      output="/tmp/test_wr_inc.md",
                                      input_path=input_path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "Incumbent" in content or "weekly-report" in content
        assert "Model Metadata" in content
        # Only check for oracle warning if the CSV had qb_source=oracle
        os.unlink(path)
    finally:
        os.unlink(input_path)


def test_generate_weekly_report_future_format():
    """Generates report from a CSV that looks like future predictions."""
    df = pd.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_01_KC_LAC"],
        "season": [2025, 2025],
        "week": [1, 1],
        "gameday": ["2025-09-04", "2025-09-07"],
        "away_team": ["DAL", "KC"],
        "home_team": ["PHI", "LAC"],
        "incumbent_home_win_prob": [0.76, 0.55],
        "predicted_winner": ["PHI", "LAC"],
        "confidence_bucket": ["70-80", "55-60"],
        "qb_source": ["live_pregame", "live_pregame"],
        "home_qb_id": ["00-0036389", "00-0036355"],
        "away_qb_id": ["00-0033077", "00-0033873"],
    })
    fp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    fp.write(df.to_csv(index=False))
    path = fp.name
    fp.close()
    try:
        result = generate_weekly_report(
            season=2025, week=1, output="/tmp/test_wr_future.md", input_path=path
        )
        assert os.path.exists(result)
        with open(result) as f:
            content = f.read()
            assert "QB Starters" in content
            assert "live_pregame" in content
            assert "QB Source" in content
        os.unlink(result)
    finally:
        os.unlink(path)


def test_generate_weekly_report_no_predictions():
    """Raises FileNotFoundError when no prediction file exists."""
    with pytest.raises(FileNotFoundError):
        generate_weekly_report(input_path="/nonexistent/path.csv")
