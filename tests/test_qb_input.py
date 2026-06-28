"""Tests for live-safe QB input parsing module."""

import os
import tempfile

import pandas as pd
import pytest

from sportslab.features.qb_input import apply_qb_input, parse_qb_input_csv


def test_parse_qb_input_csv_basic():
    """Basic CSV parsing produces correct columns."""
    content = "game_id,home_qb_id,away_qb_id\n2025_01_ARI_ATL,J.McCarthy,C.Williams\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        result = parse_qb_input_csv(path)
        assert list(result.columns) == ["game_id", "home_qb_id", "away_qb_id"]
        assert len(result) == 1
        assert result.iloc[0]["game_id"] == "2025_01_ARI_ATL"
        assert result.iloc[0]["home_qb_id"] == "J.McCarthy"
        assert result.iloc[0]["away_qb_id"] == "C.Williams"
    finally:
        os.unlink(path)


def test_parse_qb_input_csv_missing_columns():
    """Missing columns raise ValueError."""
    content = "game_id,home_qb_id\n2025_01_ARI_ATL,J.McCarthy\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        with pytest.raises(ValueError, match="missing columns"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_qb_input_csv_empty():
    """Empty CSV raises ValueError."""
    content = "game_id,home_qb_id,away_qb_id\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        with pytest.raises(ValueError, match="empty"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_qb_input_csv_not_found():
    """FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        parse_qb_input_csv("/nonexistent/path.csv")


def test_apply_qb_input_basic():
    """Apply override replaces oracle QB columns."""
    df = pd.DataFrame({
        "game_id": ["2025_01_ARI_ATL", "2025_02_ARI_ATL"],
        "home_qb_id": ["oracle_h1", "oracle_h2"],
        "away_qb_id": ["oracle_a1", "oracle_a2"],
    })
    qb_input = pd.DataFrame({
        "game_id": ["2025_01_ARI_ATL"],
        "home_qb_id": ["live_h1"],
        "away_qb_id": ["live_a1"],
    })
    result = apply_qb_input(df, qb_input)
    # First row overridden
    assert result.loc[0, "home_qb_id"] == "live_h1"
    assert result.loc[0, "away_qb_id"] == "live_a1"
    # Second row unchanged
    assert result.loc[1, "home_qb_id"] == "oracle_h2"
    assert result.loc[1, "away_qb_id"] == "oracle_a2"


def test_apply_qb_input_no_match():
    """No matching game_ids leaves all rows unchanged."""
    df = pd.DataFrame({
        "game_id": ["2025_01_ARI_ATL"],
        "home_qb_id": ["oracle_h"],
        "away_qb_id": ["oracle_a"],
    })
    qb_input = pd.DataFrame({
        "game_id": ["2025_99_XXX_YYY"],
        "home_qb_id": ["live_h"],
        "away_qb_id": ["live_a"],
    })
    result = apply_qb_input(df, qb_input)
    assert result.loc[0, "home_qb_id"] == "oracle_h"
    assert result.loc[0, "away_qb_id"] == "oracle_a"


def test_apply_qb_input_preserves_other_columns():
    """Non-QB columns are preserved after override."""
    df = pd.DataFrame({
        "game_id": ["2025_01_ARI_ATL"],
        "home_qb_id": ["oracle_h"],
        "away_qb_id": ["oracle_a"],
        "home_team": ["ATL"],
        "away_team": ["ARI"],
    })
    qb_input = pd.DataFrame({
        "game_id": ["2025_01_ARI_ATL"],
        "home_qb_id": ["live_h"],
        "away_qb_id": ["live_a"],
    })
    result = apply_qb_input(df, qb_input)
    assert result.loc[0, "home_team"] == "ATL"
    assert result.loc[0, "away_team"] == "ARI"


def test_parse_qb_input_csv_handles_nan():
    """NaN/empty QB values are mapped to pd.NA."""
    content = "game_id,home_qb_id,away_qb_id\n2025_01_ARI_ATL,nan,J.Doe\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        result = parse_qb_input_csv(path)
        assert pd.isna(result.iloc[0]["home_qb_id"]), "NaN should be mapped to NA"
        assert result.iloc[0]["away_qb_id"] == "J.Doe"
    finally:
        os.unlink(path)
