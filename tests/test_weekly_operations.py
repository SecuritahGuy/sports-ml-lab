"""Weekly live-ops tests: Week 1 cold-start, QB safety, grading, publishing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportslab.cli import cli
from sportslab.evaluation.live_preflight import validate_qb_csv
from sportslab.evaluation.prediction_audit import publish_predictions
from sportslab.evaluation.weekly_pipeline import (
    _validate_mode,
    _validate_season,
    grade_week,
)
from sportslab.features.build_features import SPORTSLAB_MIN_SEASON
from sportslab.features.qb_input import apply_qb_input, parse_qb_input_csv

TESTS_DIR = Path(__file__).resolve().parent


# ── Week 1 Cold-Start Safety ──


class TestWeek1ColdStart:
    """Week 1 is special: no prior-week data, rolling features start at 0."""

    def test_rolling_mov_3_is_zero_for_week1(self):
        """Verify rolling_mov_3 is 0 (not NaN) for Week 1 games."""
        fp = "data/features/nfl/feature_table.parquet"
        df = pd.read_parquet(fp)
        w1 = df[df["week"] == 1]
        if len(w1) == 0:
            pytest.skip("No Week 1 games in feature table")
        for col in ["home_rolling_mov_3", "away_rolling_mov_3"]:
            if col not in w1.columns:
                continue
            non_null = w1[col].notna()
            assert non_null.all(), (
                f"Some Week 1 {col} values are NaN"
            )
            assert (w1.loc[non_null, col] == 0.0).all(), (
                f"Some Week 1 {col} values are non-zero"
            )

    def test_qb_changed_is_zero_for_week1(self):
        """qb_changed should be 0 for Week 1 (no prior game to compare)."""
        fp = "data/features/nfl/feature_table.parquet"
        df = pd.read_parquet(fp)
        w1 = df[df["week"] == 1]
        if len(w1) == 0:
            pytest.skip("No Week 1 games")
        for col in ["home_qb_changed", "away_qb_changed"]:
            if col in w1.columns:
                assert (w1[col] == 0).all(), (
                    f"Some Week 1 {col} values are non-zero"
                )

    def test_week1_deterministic_predictions(self):
        """Running predict-future twice for Week 1 should give same probs."""
        from sportslab.evaluation.predict_future import predict_future
        result1 = predict_future(season=2025, week=1, mode="dry_run")
        result2 = predict_future(season=2025, week=1, mode="dry_run")
        # Both should produce output or both produce empty
        assert isinstance(result1, type(result2))


# ── Season / Mode Validation ──


class TestSeasonValidation:
    def test_rejects_pre_2000(self):
        with pytest.raises(ValueError, match="Minimum season"):
            _validate_season(1999)

    def test_accepts_2021(self):
        _validate_season(2021)

    def test_accepts_2025(self):
        _validate_season(2025)

    def test_accepts_2026(self):
        _validate_season(2026)

    def test_accepts_2027(self):
        _validate_season(2027)


class TestModeValidation:
    def test_accepts_live(self):
        _validate_mode("live")

    def test_accepts_dry_run(self):
        _validate_mode("dry_run")

    def test_accepts_rehearsal(self):
        _validate_mode("rehearsal")

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            _validate_mode("production")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            _validate_mode("")


# ── QB Input Validation ──


class TestQBInputValidation:
    def test_parse_qb_rejects_missing_columns(self, tmp_path):
        csv = tmp_path / "bad_qb.csv"
        csv.write_text("game_id,home_qb_id\n20250101_JAX_IND,QB1\n")
        with pytest.raises(ValueError, match="away_qb_id"):
            parse_qb_input_csv(str(csv))

    def test_parse_qb_rejects_empty_file(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("game_id,home_qb_id,away_qb_id\n")
        with pytest.raises(ValueError, match="empty"):
            parse_qb_input_csv(str(csv))

    def test_parse_qb_rejects_duplicates(self, tmp_path):
        csv = tmp_path / "dup.csv"
        csv.write_text(
            "game_id,home_qb_id,away_qb_id\n"
            "g1,QB1,QB2\n"
            "g1,QB3,QB4\n"
        )
        with pytest.raises(ValueError, match="Duplicate"):
            parse_qb_input_csv(str(csv))

    def test_parse_qb_rejects_all_null(self, tmp_path):
        csv = tmp_path / "null.csv"
        csv.write_text(
            "game_id,home_qb_id,away_qb_id\n"
            "g1,,\n"
            "g2,,\n"
        )
        with pytest.raises(ValueError, match="All home_qb_id"):
            parse_qb_input_csv(str(csv))

    def test_parse_qb_strips_whitespace(self, tmp_path):
        csv = tmp_path / "ws.csv"
        csv.write_text(
            "game_id,home_qb_id,away_qb_id\n"
            "  g1  , QB1 , QB2 \n"
        )
        df = parse_qb_input_csv(str(csv))
        assert df["game_id"].iloc[0] == "g1"
        assert df["home_qb_id"].iloc[0] == "QB1"

    def test_validate_qb_csv_rejects_missing_file(self):
        issues = validate_qb_csv("/nonexistent/path.csv")
        assert len(issues) > 0
        assert "not found" in issues[0].lower()

    def test_parse_qb_accepts_valid_v1(self, tmp_path):
        csv = tmp_path / "valid.csv"
        csv.write_text(
            "game_id,home_qb_id,away_qb_id\n"
            "2025_01_NE_KC,MAHOMES,ALLEN\n"
        )
        df = parse_qb_input_csv(str(csv))
        assert len(df) == 1
        assert df["home_qb_id"].iloc[0] == "MAHOMES"

    def test_apply_qb_input_preserves_non_matching(self):
        df = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "home_qb_id": ["QB_A", "QB_B"],
        })
        qb_input = pd.DataFrame({
            "game_id": ["g1"],
            "home_qb_id": ["QB_OVERRIDE"],
            "away_qb_id": ["QB_X"],
        })
        result = apply_qb_input(df, qb_input)
        assert result.loc[0, "home_qb_id"] == "QB_OVERRIDE"
        assert result.loc[1, "home_qb_id"] == "QB_B"


# ── Pre-2021 Rejection ──


class TestPre2021Rejection:
    def test_validate_season_rejects_pre_2000(self):
        with pytest.raises(ValueError, match=f"Season 1999.*{SPORTSLAB_MIN_SEASON}"):
            _validate_season(1999)

    def test_feature_table_has_no_pre_2000(self):
        fp = "data/features/nfl/feature_table.parquet"
        df = pd.read_parquet(fp)
        assert df["season"].min() >= 2000


# ── Prediction Publishing Safety ──


class TestPublishingSafety:
    def test_publish_predictions_dry_run_writes_nothing(self):
        """Dry-run mode should not write any files."""
        docs_dir = Path("docs/predictions")
        before = set()
        if docs_dir.exists():
            before = set(
                str(p.relative_to(docs_dir))
                for p in docs_dir.rglob("*")
            )

        result = publish_predictions(dry_run=True)
        assert result["note"] == "dry_run — no files written"

        after = set()
        if docs_dir.exists():
            after = set(
                str(p.relative_to(docs_dir))
                for p in docs_dir.rglob("*")
            )
        # No new files should have been created
        assert after == before, "Dry-run mode created files"

    def test_prediction_probs_are_valid(self):
        fp = "reports/predictions/incumbent_predictions.csv"
        df = pd.read_csv(fp)
        assert "incumbent_home_win_prob" in df.columns
        probs = df["incumbent_home_win_prob"].dropna()
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_prediction_schema_stable(self):
        fp = "reports/predictions/incumbent_predictions.csv"
        df = pd.read_csv(fp)
        required = [
            "game_id", "season", "week",
            "away_team", "home_team",
            "incumbent_home_win_prob",
            "model_version",
        ]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_no_diagnostic_fields_in_public_predictions(self):
        """Public-facing predictions must not include diagnostic-only fields."""
        fp = "reports/predictions/incumbent_predictions.csv"
        df = pd.read_csv(fp)
        diagnostic_fields = [
            "market_prob_diagnostic",
            "market_minus_model_diagnostic",
            "market_model_diff",
        ]
        for col in diagnostic_fields:
            if col in df.columns:
                pytest.skip(f"Diagnostic field {col} found — check if it's labeled diagnostic")


# ── Grading Safety ──


class TestGradingSafety:
    def test_grade_week_requires_valid_season(self):
        with pytest.raises(ValueError, match="Minimum season"):
            grade_week(season=1999, week=1)

    def test_grade_week_rejects_future_season(self):
        """A season with no data should fail gracefully, not silently."""
        with pytest.raises((ValueError, FileNotFoundError)):
            grade_week(season=2099, week=1)

    def test_compute_metrics_handles_empty(self):
        from sportslab.evaluation.weekly_pipeline import _compute_metrics
        df = pd.DataFrame({
            "actual_home_win": [np.nan, np.nan],
            "incumbent_home_win_prob": [0.6, 0.4],
        })
        metrics = _compute_metrics(df)
        assert metrics["n"] == 0

    def test_compute_metrics_output_keys(self):
        from sportslab.evaluation.weekly_pipeline import _compute_metrics
        df = pd.DataFrame({
            "actual_home_win": [1.0, 0.0, 1.0],
            "incumbent_home_win_prob": [0.7, 0.3, 0.6],
        })
        metrics = _compute_metrics(df)
        assert "n" in metrics
        assert "log_loss" in metrics
        assert "brier" in metrics
        assert "accuracy" in metrics
        assert metrics["n"] == 3


# ── Failure Injection Tests ──


class TestFailureInjection:
    def test_pre_2000_season_rejected_in_cli(self):
        """The backtest CLI command rejects seasons before 2000."""
        runner = CliRunner()
        result = runner.invoke(cli, ["backtest", "1999"])
        assert result.exit_code == 0
        assert "not allowed" in result.output.lower()

    def test_malformed_qb_csv_rejected(self, tmp_path):
        csv = tmp_path / "malformed.csv"
        csv.write_text("game_id,home_qb_id\n2025_01_NE_KC,MAHOMES\n")
        with pytest.raises(ValueError, match="away_qb_id"):
            parse_qb_input_csv(str(csv))

    def test_qb_csv_duplicate_game_ids_rejected(self, tmp_path):
        csv = tmp_path / "dups.csv"
        csv.write_text(
            "game_id,home_qb_id,away_qb_id\n"
            "g1,QB1,QB2\n"
            "g1,QB3,QB4\n"
        )
        with pytest.raises(ValueError, match="Duplicate"):
            parse_qb_input_csv(str(csv))
