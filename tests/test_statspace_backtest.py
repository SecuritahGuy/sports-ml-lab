"""Tests for StatSpace backtest module."""


from sportslab.evaluation.statspace_backtest import (
    _load_feature_table,
    run_statspace_backtest,
)


class TestStatSpaceBacktest:
    def test_importable(self):
        from sportslab.evaluation import statspace_backtest
        assert hasattr(statspace_backtest, "run_statspace_backtest")

    def test_load_feature_table(self):
        ft = _load_feature_table("data/features/nfl/feature_table.parquet")
        assert len(ft) > 0
        assert "elo_prob" in ft.columns
        assert "home_elo_pre" in ft.columns
        assert "away_elo_pre" in ft.columns

    def test_backtest_produces_report(self, tmp_path):
        report = tmp_path / "test_backtest.md"
        r = run_statspace_backtest(
            ft_path="data/features/nfl/feature_table.parquet",
            report_path=str(report),
        )
        assert r == str(report)
        assert report.exists()
        content = report.read_text()
        assert "Rankings" in content
        assert "FDR" in content
