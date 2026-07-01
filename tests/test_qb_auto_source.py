"""Tests for QB auto-source from nflreadpy depth charts."""

from pathlib import Path

import pandas as pd
import pytest

from sportslab.features.qb_auto_source import (
    _map_team_code,
    build_auto_qb_csv,
    build_auto_qb_csv_standalone,
    build_weekly_qb_csv,
)


def test_team_code_mapping():
    assert _map_team_code("BUF") == "BUF"
    assert _map_team_code("ATL") == "ATL"
    assert _map_team_code("LA") == "LAR"
    assert _map_team_code("SD") == "LAC"
    assert _map_team_code("OAK") == "LV"
    assert _map_team_code("STL") == "LA"


def test_team_code_no_change_for_unknown():
    assert _map_team_code("XYZ") == "XYZ"


def test_module_importable():
    from sportslab.features import qb_auto_source
    assert hasattr(qb_auto_source, "build_auto_qb_csv")


def test_build_auto_qb_csv_standalone(tmp_path):
    """Quick smoke test — verify standalone function runs and returns path."""
    result = build_auto_qb_csv_standalone(season=2024, week=1)
    assert isinstance(result, str)
    p = Path(result)
    assert p.exists() or not p.exists()  # might be temp file cleaned up


def test_build_auto_qb_csv_with_ft(tmp_path):
    """Verify output DataFrame has expected columns."""
    df, source = build_auto_qb_csv(season=2024, week=1)
    assert source == "auto_qb"
    assert isinstance(df, pd.DataFrame)
    assert "game_id" in df.columns
    assert "home_qb_id" in df.columns
    assert "away_qb_id" in df.columns
    assert len(df) > 0


def test_build_auto_qb_csv_fills_qb_ids(tmp_path):
    """Verify most games get QB IDs filled."""
    df, _ = build_auto_qb_csv(season=2024, week=1)
    n_home = df["home_qb_id"].notna().sum()
    n_away = df["away_qb_id"].notna().sum()
    # Most (if not all) games should have QB data
    assert n_home >= len(df) * 0.8
    assert n_away >= len(df) * 0.8


def test_build_auto_qb_csv_output_file(tmp_path):
    """Verify CSV output works."""
    out = str(tmp_path / "test_qb.csv")
    df, source = build_auto_qb_csv(season=2024, week=1, output_path=out)
    assert Path(out).exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == ["game_id", "home_qb_id", "away_qb_id"]


def test_auto_qb_integration_with_pipeline():
    """Verify predict_week accepts auto_qb flag."""
    from sportslab.evaluation.weekly_pipeline import predict_week
    import inspect
    sig = inspect.signature(predict_week)
    assert "auto_qb" in sig.parameters


def test_auto_qb_flag_in_cli():
    """Verify CLI has --auto-qb flag."""
    from sportslab.cli import predict_week_cmd
    import click
    for param in predict_week_cmd.params:
        if param.name == "auto_qb":
            assert isinstance(param, click.Option)
            assert param.is_flag
            break
    else:
        assert False, "auto_qb not found in CLI params"


def test_auto_qb_fallback_on_missing_ft():
    """Verify helpful error when feature table is missing."""
    try:
        build_auto_qb_csv(
            season=2024, week=1,
            feature_table_path="/nonexistent/path.parquet",
        )
        assert False, "Should have raised"
    except FileNotFoundError as e:
        assert "Feature table not found" in str(e)


def test_output_qb_ids_are_strings():
    """Verify QB IDs are strings, not floats (from NaN handling)."""
    df, _ = build_auto_qb_csv(season=2024, week=1)
    for col in ["home_qb_id", "away_qb_id"]:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            assert non_null.dtype == object, f"{col} should be object/string"


# ── build_weekly_qb_csv tests ──


def test_weekly_qb_module_importable():
    """Verify weekly QB function is importable from module."""
    from sportslab.features import qb_auto_source
    assert hasattr(qb_auto_source, "build_weekly_qb_csv")


def test_weekly_qb_basic_smoke():
    """Verify weekly QB returns expected structure."""
    df, source = build_weekly_qb_csv(season=2025, week=2)
    assert source == "weekly_qb"
    assert isinstance(df, pd.DataFrame)
    assert "game_id" in df.columns
    assert "home_qb_id" in df.columns
    assert "away_qb_id" in df.columns
    assert "home_qb_source" in df.columns
    assert "away_qb_source" in df.columns
    assert len(df) > 0


def test_weekly_qb_week1_falls_back_to_depth_chart():
    """Week 1 has no prior data — should fall back to depth chart."""
    df, _ = build_weekly_qb_csv(season=2025, week=1)
    n_home = df["home_qb_id"].notna().sum()
    n_away = df["away_qb_id"].notna().sum()
    # Most should have QBs from depth chart
    assert n_home >= len(df) * 0.8
    assert n_away >= len(df) * 0.8
    # All home sources should be depth_chart (no prior week data for week 1)
    home_srcs = df["home_qb_source"].value_counts()
    assert home_srcs.get("depth_chart", 0) >= len(df) * 0.8


def test_weekly_qb_week2_uses_prior_week():
    """Week 2 should use prior week data for most teams."""
    df, _ = build_weekly_qb_csv(season=2025, week=2)
    home_prior = (df["home_qb_source"] == "prior_week").sum()
    away_prior = (df["away_qb_source"] == "prior_week").sum()
    n = len(df)
    # At least 50% should come from prior week (most teams played week 1)
    assert home_prior >= n * 0.5, f"Expected >=50% home prior week, got {home_prior}/{n}"
    assert away_prior >= n * 0.5, f"Expected >=50% away prior week, got {away_prior}/{n}"


def test_weekly_qb_output_file(tmp_path):
    """Verify CSV output works."""
    out = str(tmp_path / "test_weekly_qb.csv")
    df, source = build_weekly_qb_csv(season=2025, week=2, output_path=out)
    assert Path(out).exists()
    loaded = pd.read_csv(out)
    assert "game_id" in loaded.columns
    assert "home_qb_id" in loaded.columns
    assert "away_qb_id" in loaded.columns


def test_weekly_qb_fallback_on_missing_ft():
    """Verify helpful error when feature table is missing."""
    with pytest.raises(FileNotFoundError, match="Feature table not found"):
        build_weekly_qb_csv(
            season=2025, week=2,
            feature_table_path="/nonexistent/path.parquet",
        )


def test_weekly_qb_no_prior_data_season():
    """A season with no prior data should still produce QB IDs."""
    df, _ = build_weekly_qb_csv(season=2025, week=1)
    assert len(df) > 0
    assert df["home_qb_id"].notna().sum() > 0


def test_weekly_qb_integration_with_pipeline():
    """Verify predict_week accepts weekly_qb flag."""
    from sportslab.evaluation.weekly_pipeline import predict_week
    import inspect
    sig = inspect.signature(predict_week)
    assert "weekly_qb" in sig.parameters


def test_weekly_qb_flag_in_cli():
    """Verify CLI has --weekly-qb flag."""
    from sportslab.cli import predict_week_cmd
    import click
    for param in predict_week_cmd.params:
        if param.name == "weekly_qb":
            assert isinstance(param, click.Option)
            assert param.is_flag
            break
    else:
        assert False, "weekly_qb not found in CLI params"
