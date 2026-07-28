"""Tests for prediction_vintages module — list, load, compare, report."""

import json
from pathlib import Path

import pandas as pd
import pytest

from sportslab.evaluation.prediction_vintages import (
    compare_vintages,
    list_vintages,
    load_vintages,
    vintage_diff_report,
)
from sportslab.evaluation.weekly_pipeline import (
    VALID_VINTAGES,
    _read_manifest,
    _register_snapshot,
    _write_manifest,
)

# ── Helpers ──

def _fake_snapshot_df(n_games: int = 4, seed: float = 0.5) -> pd.DataFrame:
    """Create a fake prediction snapshot DataFrame."""
    import numpy as np
    rng = np.random.default_rng(42)
    teams = ["ARI", "ATL", "CHI", "DAL", "PHI", "GB", "KC", "SF"]
    games = []
    for i in range(n_games):
        away = teams[i % len(teams)]
        home = teams[(i + 1) % len(teams)]
        games.append({
            "game_id": f"2026_01_{away}_{home}",
            "season": 2026,
            "week": 1,
            "away_team": away,
            "home_team": home,
            "incumbent_home_win_prob": seed + rng.uniform(-0.1, 0.1),
            "qb_source": "weekly_qb",
            "confidence_bucket": "60-70",
            "qb_gate_fired": False,
        })
    return pd.DataFrame(games)


def _make_snapshot_file(manifest_entry: dict) -> Path:
    """Create a fake snapshot CSV on disk and return its path."""
    path = Path(manifest_entry["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _fake_snapshot_df(manifest_entry.get("n_games", 4))
    # Slightly different prob for each vintage
    v = manifest_entry.get("vintage", "locked")
    offsets = {"early": 0.05, "final-injury": 0.02, "locked": 0.0}
    offset = offsets.get(v, 0.0)
    df["incumbent_home_win_prob"] = df["incumbent_home_win_prob"].clip(0, 1) - offset
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def clean_manifest(tmp_path):
    """Replace MANIFEST_PATH with a clean manifest in tmp_path."""
    from sportslab.evaluation import weekly_pipeline as wp
    orig = wp.MANIFEST_PATH
    clean = tmp_path / "snapshot_manifest.json"
    clean.write_text(json.dumps({"manifest_version": "1.0", "snapshots": []}))
    wp.MANIFEST_PATH = clean
    yield clean
    wp.MANIFEST_PATH = orig


@pytest.fixture
def three_vintage_manifest(clean_manifest, tmp_path):
    """Register 3 vintages (early, final-injury, locked) for 2026 W1."""
    snapshots_dir = tmp_path / "snapshots"
    entries = []
    for vintage in ["early", "final-injury", "locked"]:
        sid = f"week_2026_01_{vintage}_live_20260901_000000"
        path = snapshots_dir / f"{sid}.csv"
        entry = {
            "snapshot_id": sid,
            "path": str(path),
            "season": 2026,
            "week": 1,
            "vintage": vintage,
            "mode": "live",
            "status": "initial",
            "created_at": f"2026-09-01T00:00:0{vintage[0]}",
            "qb_source": "weekly_qb",
            "n_games": 4,
            "checksum": "sha256:fakesum",
        }
        entries.append(entry)
    manifest = _read_manifest()
    manifest["snapshots"] = entries
    _write_manifest(manifest)
    # Create the snapshot files on disk
    for e in entries:
        _make_snapshot_file(e)
    return entries


# ── Tests ──

class TestListVintages:
    def test_no_vintages_returns_empty(self, clean_manifest):
        result = list_vintages(2026, 1)
        assert result == []

    def test_lists_three_vintages(self, three_vintage_manifest):
        result = list_vintages(2026, 1)
        assert len(result) == 3
        vintages = {e.get("vintage", "locked") for e in result}
        assert vintages == {"early", "final-injury", "locked"}

    def test_superseded_not_returned(self, clean_manifest):
        path = Path(str(clean_manifest).replace(".json", "_early.csv"))
        _make_snapshot_file({"path": str(path), "vintage": "early", "n_games": 4})
        _register_snapshot(path, 2026, 1, "weekly_qb", 4, vintage="early")
        _make_snapshot_file({"path": str(path), "vintage": "early", "n_games": 4})
        _register_snapshot(path, 2026, 1, "weekly_qb", 4, vintage="early")
        result = list_vintages(2026, 1)
        assert len(result) == 1  # second supersedes first

    def test_mode_filter(self, three_vintage_manifest):
        result = list_vintages(2026, 1, mode="dry_run")
        assert result == []


class TestLoadVintages:
    def test_returns_dict_by_vintage(self, three_vintage_manifest):
        result = load_vintages(2026, 1)
        assert set(result.keys()) == {"early", "final-injury", "locked"}
        for v, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 4
            assert "incumbent_home_win_prob" in df.columns

    def test_missing_file_prints_warning(self, three_vintage_manifest, capsys):
        manifest = _read_manifest()
        manifest["snapshots"][0]["path"] = "/nonexistent/snapshot.csv"
        _write_manifest(manifest)
        load_vintages(2026, 1)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "early" in captured.out


class TestCompareVintages:
    def test_requires_two_vintages(self, clean_manifest):
        path = Path(str(clean_manifest).replace(".json", "_locked.csv"))
        _make_snapshot_file({"path": str(path), "vintage": "locked", "n_games": 4})
        _register_snapshot(path, 2026, 1, "weekly_qb", 4, vintage="locked")
        result = compare_vintages(2026, 1)
        assert result is None

    def test_returns_comparison_table(self, three_vintage_manifest):
        result = compare_vintages(2026, 1)
        assert result is not None
        assert "game_id" in result.columns
        assert "prob_early" in result.columns
        assert "prob_final-injury" in result.columns
        assert "prob_locked" in result.columns
        assert "drift_early_to_final-injury" in result.columns
        assert "drift_final-injury_to_locked" in result.columns

    def test_drift_values_are_bounded(self, three_vintage_manifest):
        result = compare_vintages(2026, 1)
        for col in result.columns:
            if col.startswith("drift_"):
                vals = result[col].dropna()
                assert vals.between(-1.0, 1.0).all(), f"{col} out of bounds"

    def test_qb_source_per_vintage(self, three_vintage_manifest):
        result = compare_vintages(2026, 1)
        for v in ["early", "final-injury", "locked"]:
            assert f"qb_{v}" in result.columns


class TestVintageDiffReport:
    def test_report_generation(self, three_vintage_manifest):
        report = vintage_diff_report(2026, 1)
        assert "Vintage Comparison: 2026 Week 1 (live)" in report
        assert "early" in report
        assert "final-injury" in report
        assert "locked" in report
        assert "Per-Game Comparison" in report
        assert "Biggest Probability Drifts" in report

    def test_report_with_output_path(self, three_vintage_manifest, tmp_path):
        out = str(tmp_path / "vintage_diff.md")
        report = vintage_diff_report(2026, 1, output_path=out)
        assert Path(out).exists()
        content = Path(out).read_text()
        assert content == report

    def test_report_requires_two_vintages(self, clean_manifest):
        report = vintage_diff_report(2026, 1)
        assert "Need at least 2 vintages" in report


class TestValidVintages:
    def test_valid_vintages_are_three(self):
        assert VALID_VINTAGES == ["early", "final-injury", "locked"]


class TestCLIImportability:
    def test_list_vintages_importable(self):
        pass  # noqa: F811

    def test_compare_vintages_importable(self):
        pass  # noqa: F811

    def test_predict_week_has_vintage(self):
        import inspect

        from sportslab.evaluation.weekly_pipeline import predict_week
        sig = inspect.signature(predict_week)
        assert "vintage" in sig.parameters
        assert sig.parameters["vintage"].default == "locked"

    def test_grade_week_has_vintage(self):
        import inspect

        from sportslab.evaluation.weekly_pipeline import grade_week
        sig = inspect.signature(grade_week)
        assert "vintage" in sig.parameters
        assert sig.parameters["vintage"].default == "locked"
