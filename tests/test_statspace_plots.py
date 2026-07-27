"""Tests for StatSpace visualization module."""

from pathlib import Path

from sportslab.evaluation.statspace_plots import (
    plot_all_pairs,
)
from sportslab.evaluation.team_profiles import build_team_profiles


class TestStatSpacePlots:
    def test_importable(self):
        from sportslab.evaluation import statspace_plots
        assert hasattr(statspace_plots, "build_statspace_plots")

    def test_plot_all_pairs_produces_pngs(self, tmp_path):
        profiles = build_team_profiles()
        plot_all_pairs(profiles, out_dir=str(tmp_path))
        pngs = list(Path(tmp_path).glob("*.png"))
        assert len(pngs) >= 10

    def test_heatmap_produced(self, tmp_path):
        profiles = build_team_profiles()
        plot_all_pairs(profiles, out_dir=str(tmp_path))
        assert (Path(tmp_path) / "statspace_correlation_heatmap.png").exists()

    def test_cli_runs(self):
        from sportslab.cli import statspace_plots_cmd
        assert callable(statspace_plots_cmd)
