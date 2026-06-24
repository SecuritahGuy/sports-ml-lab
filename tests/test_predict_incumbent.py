"""Tests for incumbent prediction artifacts and benchmark registry validation."""

from pathlib import Path

import pandas as pd
import pytest

from sportslab.evaluation.predict_incumbent import (
    CONFIDENCE_BINS,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
    TRAIN_SEASONS,
    _assign_confidence_bucket,
    generate_incumbent_predictions,
)

HOLDOUT_PATH = Path("reports/predictions/incumbent_predictions_2025_holdout.csv")
FULL_PATH = Path("reports/predictions/incumbent_predictions.csv")
LEADERBOARD_PATH = Path("reports/benchmarks/leaderboard.csv")
INCUMBENT_PATH = Path("reports/benchmarks/nfl_research_incumbent.md")
HISTORY_PATH = Path("reports/benchmarks/benchmark_history.md")
MODEL_CARD_PATH = Path("reports/benchmarks/incumbent_model_card.md")

ALL_REQUIRED_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "home_win_actual",
    "incumbent_home_win_prob",
    "predicted_winner",
    "confidence_bucket",
    "model_version",
    "feature_set",
    "calibration_method",
    "caution_qb_change",
    "caution_neutral",
    "caution_early_season",
    "caution_missing_features",
    "caution_model_market_disagreement",
    "market_model_diff",
    "market_prob_diagnostic",
    "market_minus_model_diagnostic",
    "qb_change_flag",
]


class TestConfidenceBuckets:
    def test_buckets_cover_range(self):
        assert CONFIDENCE_BINS[0][0] == 0.50
        assert CONFIDENCE_BINS[-1][1] == 1.01

    def test_bucket_low(self):
        assert _assign_confidence_bucket(0.50) == "50-55"
        assert _assign_confidence_bucket(0.54) == "50-55"

    def test_bucket_mid(self):
        assert _assign_confidence_bucket(0.62) == "60-65"
        assert _assign_confidence_bucket(0.68) == "65-70"

    def test_bucket_high(self):
        assert _assign_confidence_bucket(0.75) == "70-80"
        assert _assign_confidence_bucket(0.95) == "80+"

    def test_bucket_boundary(self):
        assert _assign_confidence_bucket(0.55) == "55-60"
        assert _assign_confidence_bucket(0.60) == "60-65"
        assert _assign_confidence_bucket(0.65) == "65-70"
        assert _assign_confidence_bucket(0.70) == "70-80"
        assert _assign_confidence_bucket(0.80) == "80+"

    def test_bucket_all_unique(self):
        labels = [lb for _, _, lb in CONFIDENCE_BINS]
        assert len(labels) == len(set(labels))


class TestPredictionSchema:
    @pytest.fixture(scope="class")
    def full_df(self):
        return pd.read_csv(FULL_PATH)

    @pytest.fixture(scope="class")
    def hold_df(self):
        return pd.read_csv(HOLDOUT_PATH)

    def test_files_exist(self):
        assert FULL_PATH.exists(), f"Missing: {FULL_PATH}"
        assert HOLDOUT_PATH.exists(), f"Missing: {HOLDOUT_PATH}"

    def test_full_has_rows(self, full_df):
        assert len(full_df) >= 1000

    def test_holdout_has_rows(self, hold_df):
        assert len(hold_df) >= 200

    def test_holdout_season(self, hold_df):
        assert (hold_df["season"] == 2025).all()

    def test_required_columns(self, full_df):
        required = [
            "game_id",
            "season",
            "week",
            "gameday",
            "away_team",
            "home_team",
            "incumbent_home_win_prob",
            "predicted_winner",
            "confidence_bucket",
            "model_version",
            "feature_set",
            "calibration_method",
        ]
        for col in required:
            assert col in full_df.columns, f"Missing column: {col}"

    def test_probabilities_in_01(self, full_df):
        p = full_df["incumbent_home_win_prob"]
        assert p.between(0, 1).all()

    def test_probabilities_not_constant(self, full_df):
        assert full_df["incumbent_home_win_prob"].nunique() > 10

    def test_predicted_winner_valid(self, full_df):
        valid = set(full_df["home_team"].unique()) | set(full_df["away_team"].unique())
        for w in full_df["predicted_winner"].unique():
            assert w in valid, f"Invalid predicted winner: {w}"

    def test_confidence_bucket_valid(self, full_df):
        valid_labels = {lb for _, _, lb in CONFIDENCE_BINS}
        for b in full_df["confidence_bucket"].unique():
            assert b in valid_labels, f"Invalid bucket: {b}"


class TestHoldoutMetrics:
    @pytest.fixture(scope="class")
    def hold_df(self):
        return pd.read_csv(HOLDOUT_PATH)

    def test_holdout_log_loss_matches(self, hold_df):
        from sklearn.metrics import log_loss

        valid = hold_df["home_win_actual"].notna()
        ll = log_loss(
            hold_df.loc[valid, "home_win_actual"],
            hold_df.loc[valid, "incumbent_home_win_prob"],
        )
        assert ll == pytest.approx(INCUMBENT_HOLDOUT_LL, abs=0.001)


class TestCautionFlags:
    @pytest.fixture(scope="class")
    def full_df(self):
        return pd.read_csv(FULL_PATH)

    def test_caution_qb_change_present(self, full_df):
        assert "caution_qb_change" in full_df.columns

    def test_caution_qb_change_binary(self, full_df):
        assert full_df["caution_qb_change"].isin([0, 1]).all()

    def test_caution_early_season_present(self, full_df):
        assert "caution_early_season" in full_df.columns

    def test_caution_early_season_binary(self, full_df):
        assert full_df["caution_early_season"].isin([0, 1]).all()

    def test_caution_market_disagreement_present(self, full_df):
        assert "caution_model_market_disagreement" in full_df.columns

    def test_caution_market_disagreement_binary(self, full_df):
        assert full_df["caution_model_market_disagreement"].isin([0, 1]).all()


class TestMarketFields:
    @pytest.fixture(scope="class")
    def full_df(self):
        return pd.read_csv(FULL_PATH)

    def test_market_prob_field_name_diagnostic(self, full_df):
        assert "market_prob_diagnostic" in full_df.columns

    def test_market_minus_model_field_name_diagnostic(self, full_df):
        assert "market_minus_model_diagnostic" in full_df.columns

    def test_market_prob_in_01(self, full_df):
        p = full_df["market_prob_diagnostic"].dropna()
        assert p.between(0, 1).all()

    def test_market_minus_model_not_constant(self, full_df):
        vals = full_df["market_minus_model_diagnostic"].dropna()
        assert vals.nunique() > 10


class TestQbChangeFlag:
    @pytest.fixture(scope="class")
    def full_df(self):
        return pd.read_csv(FULL_PATH)

    def test_qb_change_flag_present(self, full_df):
        assert "qb_change_flag" in full_df.columns

    def test_qb_change_flag_binary(self, full_df):
        assert full_df["qb_change_flag"].isin([0, 1]).all()

    def test_qb_change_flag_some_true(self, full_df):
        assert full_df["qb_change_flag"].sum() >= 200


class TestPredictionQA:
    """Comprehensive prediction artifact quality assurance."""

    @pytest.fixture(scope="class")
    def full_df(self):
        return pd.read_csv(FULL_PATH)

    @pytest.fixture(scope="class")
    def hold_df(self):
        return pd.read_csv(HOLDOUT_PATH)

    def test_all_required_columns_present(self, full_df):
        for col in ALL_REQUIRED_COLUMNS:
            assert col in full_df.columns, f"Missing required column: {col}"

    def test_no_nan_in_prob(self, full_df):
        assert full_df["incumbent_home_win_prob"].notna().all()

    def test_no_nan_in_predicted_winner(self, full_df):
        assert full_df["predicted_winner"].notna().all()

    def test_predicted_winner_consistent_with_prob(self, full_df):
        home_team = full_df["home_team"].values
        away_team = full_df["away_team"].values
        prob = full_df["incumbent_home_win_prob"].values
        expected = pd.Series([h if p >= 0.5 else a for h, a, p in zip(home_team, away_team, prob)])
        assert (full_df["predicted_winner"] == expected).all(), (
            "Some predicted_winners disagree with probability threshold"
        )

    def test_full_file_seasons(self, full_df):
        seasons = sorted(full_df["season"].unique())
        expected = sorted(TRAIN_SEASONS + [2025])
        assert seasons == expected, f"Unexpected seasons: {seasons}"

    def test_holdout_file_only_2025(self, hold_df):
        assert (hold_df["season"] == 2025).all()
        assert hold_df["week"].max() >= 17

    def test_model_version_matches(self, full_df):
        assert (full_df["model_version"] == INCUMBENT_VERSION).all()

    def test_feature_set_matches(self, full_df):
        expected = "qb_changed + rolling_mov_3"
        assert (full_df["feature_set"] == expected).all()

    def test_holdout_log_loss_matches_benchmark(self, hold_df):
        from sklearn.metrics import log_loss

        valid = hold_df["home_win_actual"].notna()
        ll = log_loss(
            hold_df.loc[valid, "home_win_actual"],
            hold_df.loc[valid, "incumbent_home_win_prob"],
        )
        assert ll == pytest.approx(INCUMBENT_HOLDOUT_LL, abs=0.001)

    def test_market_fields_labeled_diagnostic(self, full_df):
        assert "market_prob_diagnostic" in full_df.columns
        assert "market_minus_model_diagnostic" in full_df.columns
        # Should NOT have undecorated market fields used in training
        assert "market_prob" not in full_df.columns

    def test_market_model_diff_not_constant(self, full_df):
        assert full_df["market_model_diff"].dropna().nunique() > 10

    def test_confidence_buckets_all_covered(self, full_df):
        observed = set(full_df["confidence_bucket"].unique())
        expected = {lb for _, _, lb in CONFIDENCE_BINS}
        uncovered = expected - observed
        assert len(uncovered) == 0, f"Unused buckets: {uncovered}"


class TestModelCard:
    def test_model_card_exists(self):
        assert MODEL_CARD_PATH.exists(), f"Missing model card: {MODEL_CARD_PATH}"

    def test_model_card_contains_incumbent_version(self):
        text = MODEL_CARD_PATH.read_text()
        assert INCUMBENT_VERSION in text

    def test_model_card_contains_holdout_ll(self):
        text = MODEL_CARD_PATH.read_text()
        assert str(INCUMBENT_HOLDOUT_LL) in text

    def test_model_card_contains_feature_set(self):
        text = MODEL_CARD_PATH.read_text()
        assert "qb_changed" in text or "home_qb_changed" in text
        assert "rolling_mov_3" in text or "home_rolling_mov_3" in text

    def test_model_card_contains_promotion_criteria(self):
        text = MODEL_CARD_PATH.read_text()
        assert "Promotion Criteria" in text or "promotion" in text

    def test_model_card_contains_leakage_controls(self):
        text = MODEL_CARD_PATH.read_text()
        assert "Leakage" in text or "leakage" in text

    def test_model_card_contains_reproducibility_commands(self):
        text = MODEL_CARD_PATH.read_text()
        assert "predict-incumbent" in text or "weekly-report" in text


class TestCLI:
    def test_cli_importable(self):
        from sportslab.cli import cli

        assert cli is not None

    def test_predict_incumbent_importable(self):
        assert generate_incumbent_predictions is not None


class TestBenchmarkRegistry:
    def test_incumbent_md_exists(self):
        assert INCUMBENT_PATH.exists()

    def test_leaderboard_csv_exists(self):
        assert LEADERBOARD_PATH.exists()

    def test_leaderboard_parses(self):
        df = pd.read_csv(LEADERBOARD_PATH)
        assert len(df) > 0

    def test_leaderboard_has_expected_columns(self):
        df = pd.read_csv(LEADERBOARD_PATH)
        expected = ["experiment", "model_features", "decision", "holdout_ll"]
        for col in expected:
            assert col in df.columns, f"Missing leaderboard column: {col}"

    def test_incumbent_holdout_in_incumbent_md(self):
        text = INCUMBENT_PATH.read_text()
        assert str(INCUMBENT_HOLDOUT_LL) in text, (
            f"Incumbent holdout LL {INCUMBENT_HOLDOUT_LL} not in incumbent md"
        )

    def test_incumbent_holdout_in_leaderboard(self):
        df = pd.read_csv(LEADERBOARD_PATH)
        promoted = df[df["decision"] == "promoted"]
        assert len(promoted) >= 1
        vals = promoted["holdout_ll"].dropna().astype(float).values
        assert any(abs(v - float(INCUMBENT_HOLDOUT_LL)) < 0.001 for v in vals)

    def test_optuna_feature_selection_in_leaderboard(self):
        text = LEADERBOARD_PATH.read_text()
        assert "optuna_feature_selection" in text or "no-improvement" in text

    def test_no_diagnostic_labeled_promoted(self):
        df = pd.read_csv(LEADERBOARD_PATH)
        diag_promoted = df[
            (df["decision"] == "promoted")
            & (df["experiment"].str.contains("diagnostic|holdout", case=False))
        ]
        assert len(diag_promoted) == 0, (
            f"Diagnostic labeled as promoted: {diag_promoted['experiment'].tolist()}"
        )

    def test_benchmark_history_exists(self):
        assert HISTORY_PATH.exists()

    def test_combined_features_in_history(self):
        text = HISTORY_PATH.read_text()
        assert "Combined" in text or "combined" in text

    def test_optuna_feature_selection_in_history(self):
        text = HISTORY_PATH.read_text()
        assert "Optuna" in text or "optuna" in text

    def test_comprehensive_efficiency_in_leaderboard(self):
        text = LEADERBOARD_PATH.read_text()
        assert "comprehensive_efficiency" in text

    def test_comprehensive_efficiency_in_history(self):
        text = HISTORY_PATH.read_text()
        assert "Comprehensive Efficiency" in text

    def test_history_summary_matches_entries(self):
        """Validate that the benchmark_history.md summary table counts match
        the actual number of ###-prefixed entry headers."""
        text = HISTORY_PATH.read_text()
        lines = text.splitlines()
        entry_count = sum(
            1
            for line in lines
            if line.strip().startswith("### ") and not line.strip().startswith("### Summary")
        )
        # entries 31+ use "## <number>. Name" format
        entry_count += sum(
            1
            for line in lines
            if (
                line.strip().startswith("## ")
                and len(line.strip()) > 3
                and line.strip()[3].isdigit()
                and ". " in line.strip()
            )
        )
        total_line = [line for line in lines if "Total experiments" in line]
        assert total_line, "No summary table found"
        import re

        m = re.search(r"\b(\d+)\b", total_line[0])
        assert m, "Could not parse entry count from summary"
        parsed = int(m.group(1))
        assert parsed == entry_count, (
            f"Summary says {parsed} entries, but found {entry_count} headers"
        )

    def test_summary_promoted_rejected_diagnostic_counts(self):
        df = pd.read_csv(LEADERBOARD_PATH)
        promoted = df[
            df["decision"].str.contains("promote|superseded", case=False, na=False)
        ].drop_duplicates(subset="experiment")
        rejected = df[df["decision"] == "rejected"].drop_duplicates(subset="experiment")
        diagnostic = df[
            df["decision"].str.contains("diagnostic", case=False, na=False)
        ].drop_duplicates(subset="experiment")
        at_least_5_promoted = len(promoted) >= 5
        at_least_18_rejected = len(rejected) >= 18
        at_least_6_diagnostic = len(diagnostic) >= 6
        assert at_least_5_promoted, f"Expected >=5 promoted, got {len(promoted)}"
        assert at_least_18_rejected, f"Expected >=18 rejected, got {len(rejected)}"
        assert at_least_6_diagnostic, f"Expected >=6 diagnostic, got {len(diagnostic)}"
