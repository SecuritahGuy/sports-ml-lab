"""Tests for live-safe QB input parsing module (v2 format)."""

import os
import tempfile

import pandas as pd
import pytest

from sportslab.features.qb_input import (
    QB_INPUT_COLUMNS_V2,
    apply_qb_input,
    parse_qb_input_csv,
)


def _write_temp_csv(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    path = f.name
    f.close()
    return path


def test_parse_v1_format():
    """V1 (3-column) format parses correctly, v2 columns filled with NA."""
    content = "game_id,home_qb_id,away_qb_id\n2025_01_ARI_ATL,J.McCarthy,C.Williams\n"
    path = _write_temp_csv(content)
    try:
        result = parse_qb_input_csv(path)
        assert list(result.columns) == QB_INPUT_COLUMNS_V2
        assert len(result) == 1
        assert result.iloc[0]["game_id"] == "2025_01_ARI_ATL"
        assert result.iloc[0]["home_qb_id"] == "J.McCarthy"
        assert result.iloc[0]["away_qb_id"] == "C.Williams"
        assert pd.isna(result.iloc[0]["home_qb_name"])
        assert pd.isna(result.iloc[0]["source"])
        assert pd.isna(result.iloc[0]["confidence"])
        assert pd.isna(result.iloc[0]["timestamp"])
        assert pd.isna(result.iloc[0]["notes"])
    finally:
        os.unlink(path)


def test_parse_v2_format():
    """V2 format with all columns parses correctly."""
    content = (
        "game_id,home_qb_id,away_qb_id,home_qb_name,away_qb_name,"
        "source,confidence,timestamp,notes\n"
        "2025_01_ARI_ATL,J.McCarthy,C.Williams,Kyler Murray,Caleb Williams,"
        "injury_report,confirmed,2025-09-05T12:00:00Z,QB cleared concussion protocol\n"
    )
    path = _write_temp_csv(content)
    try:
        result = parse_qb_input_csv(path)
        assert len(result) == 1
        assert result.iloc[0]["home_qb_name"] == "Kyler Murray"
        assert result.iloc[0]["away_qb_name"] == "Caleb Williams"
        assert result.iloc[0]["source"] == "injury_report"
        assert result.iloc[0]["confidence"] == "confirmed"
        assert result.iloc[0]["timestamp"] == "2025-09-05T12:00:00Z"
        assert result.iloc[0]["notes"] == "QB cleared concussion protocol"
    finally:
        os.unlink(path)


def test_parse_v2_partial():
    """V2 with only some metadata columns fills rest with NA."""
    content = (
        "game_id,home_qb_id,away_qb_id,source\n"
        "2025_01_ARI_ATL,J.McCarthy,C.Williams,manual\n"
    )
    path = _write_temp_csv(content)
    try:
        result = parse_qb_input_csv(path)
        assert result.iloc[0]["source"] == "manual"
        assert pd.isna(result.iloc[0]["confidence"])
        assert pd.isna(result.iloc[0]["timestamp"])
        assert pd.isna(result.iloc[0]["notes"])
    finally:
        os.unlink(path)


def test_parse_invalid_source():
    """Invalid source value raises ValueError."""
    content = (
        "game_id,home_qb_id,away_qb_id,source\n"
        "2025_01_ARI_ATL,J.McCarthy,C.Williams,crystal_ball\n"
    )
    path = _write_temp_csv(content)
    try:
        with pytest.raises(ValueError, match="Invalid source"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_invalid_confidence():
    """Invalid confidence value raises ValueError."""
    content = (
        "game_id,home_qb_id,away_qb_id,confidence\n"
        "2025_01_ARI_ATL,J.McCarthy,C.Williams,definitely\n"
    )
    path = _write_temp_csv(content)
    try:
        with pytest.raises(ValueError, match="Invalid confidence"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_missing_columns():
    """Missing required columns raise ValueError."""
    content = "game_id,home_qb_id\n2025_01_ARI_ATL,J.McCarthy\n"
    path = _write_temp_csv(content)
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_empty():
    """Empty CSV raises ValueError."""
    content = "game_id,home_qb_id,away_qb_id\n"
    path = _write_temp_csv(content)
    try:
        with pytest.raises(ValueError, match="empty"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_not_found():
    """FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        parse_qb_input_csv("/nonexistent/path.csv")


def test_parse_handles_nan():
    """NaN/empty QB values are mapped to pd.NA, valid values preserved."""
    content = ("game_id,home_qb_id,away_qb_id\n"
               "2025_01_ARI_ATL,nan,J.Doe\n"
               "2025_02_CHI_GB,A.Rodgers,nan\n")
    path = _write_temp_csv(content)
    try:
        result = parse_qb_input_csv(path)
        assert pd.isna(result.iloc[0]["home_qb_id"])
        assert result.iloc[0]["away_qb_id"] == "J.Doe"
        assert result.iloc[1]["home_qb_id"] == "A.Rodgers"
        assert pd.isna(result.iloc[1]["away_qb_id"])
    finally:
        os.unlink(path)


def test_apply_qb_input_v1():
    """Apply override replaces oracle QB columns (v1 parity)."""
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
    assert result.loc[0, "home_qb_id"] == "live_h1"
    assert result.loc[0, "away_qb_id"] == "live_a1"
    assert result.loc[1, "home_qb_id"] == "oracle_h2"
    assert result.loc[1, "away_qb_id"] == "oracle_a2"


def test_apply_qb_input_v2_metadata():
    """V2 metadata columns are carried through on override."""
    df = pd.DataFrame({
        "game_id": ["2025_01_ARI_ATL"],
        "home_qb_id": ["oracle_h"],
        "away_qb_id": ["oracle_a"],
        "home_qb_name": [pd.NA],
        "away_qb_name": [pd.NA],
        "source": [pd.NA],
    })
    qb_input = pd.DataFrame({
        "game_id": ["2025_01_ARI_ATL"],
        "home_qb_id": ["K.Murray"],
        "away_qb_id": ["C.Williams"],
        "home_qb_name": ["Kyler Murray"],
        "away_qb_name": ["Caleb Williams"],
        "source": ["injury_report"],
        "confidence": ["confirmed"],
        "timestamp": ["2025-09-05T12:00:00Z"],
    })
    result = apply_qb_input(df, qb_input)
    assert result.loc[0, "home_qb_name"] == "Kyler Murray"
    assert result.loc[0, "away_qb_name"] == "Caleb Williams"
    assert result.loc[0, "source"] == "injury_report"
    assert result.loc[0, "confidence"] == "confirmed"
    assert result.loc[0, "timestamp"] == "2025-09-05T12:00:00Z"


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


def test_sample_fixture_loads():
    """The sample QB input fixture parses correctly."""
    path = "data/samples/sample_qb_input_2025_w1.csv"
    result = parse_qb_input_csv(path)
    assert len(result) == 16
    assert list(result.columns) == QB_INPUT_COLUMNS_V2
    assert result["home_qb_id"].notna().all()
    assert result["away_qb_id"].notna().all()
    # Check v2 metadata populated
    assert result["home_qb_name"].notna().all()
    assert result["away_qb_name"].notna().all()
    assert result["source"].notna().all()
    assert result["confidence"].notna().all()


# ── Malformed QB Input Tests ──


def test_parse_duplicate_game_ids():
    """Duplicate game_ids raise ValueError."""
    content = (
        "game_id,home_qb_id,away_qb_id\n"
        "2025_01_ARI_ATL,J.McCarthy,C.Williams\n"
        "2025_01_ARI_ATL,K.Murray,J.Fields\n"
    )
    path = _write_temp_csv(content)
    try:
        with pytest.raises(ValueError, match="Duplicate game_id"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_extra_whitespace_in_ids():
    """Extra whitespace in QB IDs is stripped."""
    content = "game_id,home_qb_id,away_qb_id\n2025_01_ARI_ATL, J.McCarthy ,C.Williams\n"
    path = _write_temp_csv(content)
    try:
        result = parse_qb_input_csv(path)
        assert result.iloc[0]["home_qb_id"] == "J.McCarthy"
    finally:
        os.unlink(path)


def test_parse_wrong_column_case():
    """Wrong column casing raises ValueError for missing required columns."""
    content = "game_id,HOME_Qb_id,away_qb_id\n2025_01_ARI_ATL,J.McCarthy,C.Williams\n"
    path = _write_temp_csv(content)
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_extra_columns():
    """Extra columns are accepted and ignored (kept in output)."""
    content = (
        "game_id,home_qb_id,away_qb_id,team_color\n"
        "2025_01_ARI_ATL,J.McCarthy,C.Williams,red\n"
    )
    path = _write_temp_csv(content)
    try:
        result = parse_qb_input_csv(path)
        assert result.iloc[0]["home_qb_id"] == "J.McCarthy"
        assert "team_color" not in result.columns
    finally:
        os.unlink(path)


def test_parse_all_null_qb_ids():
    """All QB IDs null raises ValueError."""
    content = "game_id,home_qb_id,away_qb_id\n2025_01_ARI_ATL,nan,nan\n"
    path = _write_temp_csv(content)
    try:
        with pytest.raises(
            ValueError,
            match=r"All (home|away)_qb_id values are missing",
        ):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_apply_qb_input_wrong_type():
    """apply_qb_input handles type mismatches gracefully."""
    df = pd.DataFrame({
        "game_id": [12345],
        "home_qb_id": ["oracle_h"],
        "away_qb_id": ["oracle_a"],
    })
    qb_input = pd.DataFrame({
        "game_id": ["12345"],
        "home_qb_id": ["live_h"],
        "away_qb_id": ["live_a"],
    })
    result = apply_qb_input(df, qb_input)
    assert result.loc[0, "home_qb_id"] == "oracle_h"


def test_apply_qb_input_partial_match():
    """Partial overlap: matched games override, unmatched keep oracle."""
    df = pd.DataFrame({
        "game_id": ["G1", "G2", "G3"],
        "home_qb_id": ["h1", "h2", "h3"],
        "away_qb_id": ["a1", "a2", "a3"],
    })
    qb_input = pd.DataFrame({
        "game_id": ["G1", "G3"],
        "home_qb_id": ["LIVE_H1", "LIVE_H3"],
        "away_qb_id": ["LIVE_A1", "LIVE_A3"],
    })
    result = apply_qb_input(df, qb_input)
    assert result.loc[0, "home_qb_id"] == "LIVE_H1"
    assert result.loc[1, "home_qb_id"] == "h2"
    assert result.loc[2, "home_qb_id"] == "LIVE_H3"


def test_parse_missing_game_ids_column():
    """Completely missing game_id column raises ValueError."""
    content = "home_qb_id,away_qb_id\nJ.McCarthy,C.Williams\n"
    path = _write_temp_csv(content)
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            parse_qb_input_csv(path)
    finally:
        os.unlink(path)


def test_parse_special_characters():
    """QB IDs with special characters (dots, hyphens) are preserved."""
    content = "game_id,home_qb_id,away_qb_id\nG1,P. Mahomes II,Lamar 'MVP' Jackson\n"
    path = _write_temp_csv(content)
    try:
        result = parse_qb_input_csv(path)
        assert result.iloc[0]["home_qb_id"] == "P. Mahomes II"
        assert result.iloc[0]["away_qb_id"] == "Lamar 'MVP' Jackson"
    finally:
        os.unlink(path)
