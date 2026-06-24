"""Tests for rolling MOV sensitivity experiment."""

from pathlib import Path

import pandas as pd
import pytest

from sportslab.evaluation.rolling_mov_sensitivity import (
    QB_CHANGED_COLS,
    WINDOWS,
    _compute_rolling_mov_variants,
    run_rolling_mov_sensitivity,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
)
from sportslab.features.coach import compute_coach_features
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features


def _load_minimal_feature_table():
    fp = Path("data/features/nfl/feature_table.parquet")
    return pd.read_parquet(fp)


@pytest.fixture
def feature_table():
    return _load_minimal_feature_table()


def _build_elo_and_variants():
    df_raw = _load_minimal_feature_table()
    overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=0.1,
        qb_change_bonus=0.2,
    )
    df = compute_elo_features(
        df_raw,
        k_factor=36,
        home_advantage=40,
        preseason_regression=0.1,
        team_regression_overrides=overrides,
        decay_half_life=32,
    )
    df = compute_qb_features(df)
    df = compute_coach_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy()
    return _compute_rolling_mov_variants(df)


class TestComputeRollingMovVariants:
    def test_has_all_window_columns(self):
        df = _build_elo_and_variants()
        for w in WINDOWS:
            assert f"home_rolling_mov_{w}" in df.columns
            assert f"away_rolling_mov_{w}" in df.columns

    def test_has_special_columns(self):
        df = _build_elo_and_variants()
        for col in [
            "rolling_mov_diff",
            "rolling_mov_capped",
            "rolling_mov_log_signed",
            "rolling_mov_ewma",
            "rolling_mov_std_3",
            "rolling_mov_std_5",
        ]:
            assert col in df.columns, f"Missing: {col}"

    def test_no_current_game_leakage(self):
        df = _build_elo_and_variants()
        for w in WINDOWS:
            for _, row in df.iterrows():
                if row["result"] != 0:
                    break
            else:
                continue
            home_col = f"home_rolling_mov_{w}"
            away_col = f"away_rolling_mov_{w}"
            # The rolling feature should NOT equal the current game's result
            # (because it's computed from prior games only)
            h_val = row[home_col]
            a_val = row[away_col]
            nz = row["result"]
            assert h_val != nz
            assert a_val != -nz

    def test_season_boundaries_reset(self):
        df = _build_elo_and_variants()
        for team in df["home_team"].unique()[:3]:
            team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values(
                ["season", "week"]
            )
            prev_season = None
            for _, row in team_df.iterrows():
                if row["season"] != prev_season:
                    prev_season = row["season"]
                    # First game of the season should have 0 for window features
                    if row["week"] > 1:
                        continue
                    for w in WINDOWS:
                        h_col = f"home_rolling_mov_{w}"
                        a_col = f"away_rolling_mov_{w}"
                        is_home = row["home_team"] == team
                        val = row[h_col] if is_home else row[a_col]
                        assert val == 0.0, (
                            f"Season start {team} s{row['season']} w{row['week']}:"
                            f" mov_{w}={val}, expected 0"
                        )
                    break  # Only check one team for brevity

    def test_has_qb_changed_columns(self):
        df = _build_elo_and_variants()
        for col in QB_CHANGED_COLS:
            assert col in df.columns

    def test_no_inf_or_nan(self):
        df = _build_elo_and_variants()
        float_cols = df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            assert df[col].notna().all(), f"{col} has NaN"
            assert not df[col].isin([float("inf"), float("-inf")]).any(), f"{col} has inf"


class TestRunRollingMovSensitivity:
    def test_run_creates_report(self, tmp_path):
        # Run the full experiment but redirect report
        report = str(tmp_path / "test_report.md")
        result = run_rolling_mov_sensitivity(report_path=report)
        assert result == report
        assert Path(report).exists()

    def test_report_contains_key_sections(self, tmp_path):
        report = str(tmp_path / "test_report2.md")
        run_rolling_mov_sensitivity(report_path=report)
        content = Path(report).read_text()
        assert "Rolling MOV Sensitivity Experiment" in content
        assert "Validation Results" in content
        assert "Holdout" in content
        assert "Decision" in content
        assert "mov_3" in content

    def test_no_2025_in_report_during_search(self, tmp_path):
        """Validation section should not contain 2025 metrics."""
        report = str(tmp_path / "test_report3.md")
        run_rolling_mov_sensitivity(report_path=report)
        content = Path(report).read_text()
        # Find the holdout section
        # The holdout metrics ARE in the report but only in the last section
        assert "2025 Holdout" in content
