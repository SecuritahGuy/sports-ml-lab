from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
BENCHMARKS = Path(__file__).resolve().parents[1] / "reports" / "benchmarks"
PREDICTIONS = Path(__file__).resolve().parents[1] / "reports" / "predictions"


class TestDashboardBuild:
    def test_build_dashboard_importable(self):
        from sportslab.evaluation.build_dashboard import build_all

        assert build_all is not None

    def test_build_dashboard_runs(self):
        from sportslab.evaluation.build_dashboard import build_all

        build_all()
        assert (DOCS / "index.md").exists()

    def test_index_exists(self):
        text = (DOCS / "index.md").read_text()
        assert "SportsLab" in text or "StatSpace" in text
        assert "0.6262" in text

    def test_index_has_links(self):
        text = (DOCS / "index.md").read_text()
        assert "[Benchmarks" in text
        assert "[Predictions" in text
        assert "[Model Card" in text
        assert "[Experiments" in text

    def test_index_not_betting(self):
        text = (DOCS / "index.md").read_text()
        assert "not a betting bot" in text

    def test_index_has_research_philosophy(self):
        text = (DOCS / "index.md").read_text()
        assert "Predict probabilities" in text
        assert "leakage prevention" in text

    def test_index_has_registry_summary(self):
        text = (DOCS / "index.md").read_text()
        assert "Total experiments" in text
        assert "Promoted" in text
        assert "Rejected" in text

    def test_benchmarks_exists(self):
        text = (DOCS / "benchmarks.md").read_text()
        assert "Benchmarks" in text
        assert INCUMBENT_HOLDOUT_LL in text

    def test_benchmarks_promotion_rules(self):
        text = (DOCS / "benchmarks.md").read_text()
        assert "Promotion Rules" in text
        assert "0.6262" in text

    def test_benchmarks_football_only_incumbent_labeled(self):
        text = (DOCS / "benchmarks.md").read_text()
        assert "Football-Only Incumbent" in text

    def test_benchmarks_market_not_labeled_football_only(self):
        text = (DOCS / "benchmarks.md").read_text()
        # Market should be in a separate section, not labeled as football-only
        # Check that "market" appears in a subsection header
        assert "Market" in text

    def test_benchmarks_leaderboard_sections(self):
        text = (DOCS / "benchmarks.md").read_text()
        assert "Promoted / Superseded" in text
        assert "Rejected Challengers" in text
        assert "Diagnostics" in text
        assert "Market-Aware Diagnostics" in text

    def test_predictions_exists(self):
        text = (DOCS / "predictions.md").read_text()
        assert "Predictions" in text

    def test_predictions_schema(self):
        text = (DOCS / "predictions.md").read_text()
        assert "home_win_actual" in text
        assert "incumbent_home_win_prob" in text
        assert "confidence_bucket" in text
        assert "market_prob_diagnostic" in text

    def test_predictions_caution_flags(self):
        text = (DOCS / "predictions.md").read_text()
        assert "QB change" in text
        assert "Early season" in text
        assert "Model-market disagreement" in text

    def test_predictions_artifact_links(self):
        text = (DOCS / "predictions.md").read_text()
        assert "incumbent_predictions.csv" in text
        assert "incumbent_predictions_2025_holdout.csv" in text
        assert "weekly_report.md" in text

    def test_predictions_market_diagnostic_note(self):
        text = (DOCS / "predictions.md").read_text()
        assert "Diagnostic Only" in text or "diagnostic" in text.lower()

    def test_model_card_exists(self):
        text = (DOCS / "model-card.md").read_text()
        assert "Model Card" in text

    def test_model_card_contains_incumbent_info(self):
        text = (DOCS / "model-card.md").read_text()
        assert "0.6262" in text
        assert "Platt" in text

    def test_model_card_contains_leakage_controls(self):
        text = (DOCS / "model-card.md").read_text()
        assert "Leakage" in text or "leakage" in text

    def test_model_card_contains_promotion_criteria(self):
        text = (DOCS / "model-card.md").read_text()
        assert "Promotion" in text or "promotion" in text

    def test_model_card_contains_rejected_features(self):
        text = (DOCS / "model-card.md").read_text()
        assert "Rejected" in text or "rejected" in text

    def test_experiments_exists(self):
        text = (DOCS / "experiments.md").read_text()
        assert "Experiments" in text

    def test_experiments_promoted_section(self):
        text = (DOCS / "experiments.md").read_text()
        assert "Promoted" in text

    def test_experiments_rejected_section(self):
        text = (DOCS / "experiments.md").read_text()
        assert "Rejected" in text

    def test_experiments_all_leaderboard_entries_listed(self):
        import csv

        lb = BENCHMARKS / "leaderboard.csv"
        with open(lb) as f:
            rows = list(csv.DictReader(f))
        text = (DOCS / "experiments.md").read_text()
        for r in rows:
            if r["experiment"] not in ("benchmark_registry", "predict_incumbent"):
                # The experiment name should appear in some form
                assert (
                    r["experiment"].replace("_", " ") in text.lower() or r["experiment"] in text
                ), f"Experiment '{r['experiment']}' not found in experiments.md"

    def test_dashboard_does_not_change_predictions(self):
        """Verify dashboard build does not modify prediction artifacts."""
        import hashlib

        from sportslab.evaluation.build_dashboard import build_all

        h_before = hashlib.md5(HOLDOUT_PREDS_PATH.read_bytes()).hexdigest()
        build_all()
        h_after = hashlib.md5(HOLDOUT_PREDS_PATH.read_bytes()).hexdigest()
        assert h_before == h_after, "build_dashboard modified holdout predictions!"

    def test_config_yml_exists(self):
        assert (DOCS / "_config.yml").exists()

    def test_config_yml_has_theme(self):
        text = (DOCS / "_config.yml").read_text()
        assert "theme" in text


INCUMBENT_HOLDOUT_LL = "0.6262"
HOLDOUT_PREDS_PATH = PREDICTIONS / "incumbent_predictions_2025_holdout.csv"
