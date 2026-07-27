"""Tests for retest_rejected_features experiment."""

import numpy as np
import pandas as pd

from sportslab.evaluation.retest_rejected_features import (
    _build_gate_mask,
    _logit,
    _sigmoid,
    run_retest_experiment,
)


def test_logit_sigmoid_inverse():
    """logit and sigmoid are inverses."""
    for p in [0.001, 0.1, 0.5, 0.9, 0.999]:
        assert abs(_sigmoid(_logit(p)) - p) < 1e-10


def test_logit_clips():
    """logit clips to avoid inf."""
    assert np.isfinite(_logit(0.0))
    assert np.isfinite(_logit(1.0))
    assert _logit(1e-20) < _logit(0.5)


def test_sigmoid_clips():
    """sigmoid clips extreme inputs."""
    assert 0 < _sigmoid(1e6) <= 1
    assert 0 <= _sigmoid(-1e6) < 1
    assert abs(_sigmoid(0) - 0.5) < 1e-10


def test_gate_mask():
    """Gate mask triggers on qb_changed or low team starts."""
    df = pd.DataFrame({
        "home_qb_changed": [1, 0, 0],
        "away_qb_changed": [0, 0, 0],
        "home_qb_team_starts_pre": [16.0, 2.0, 20.0],
        "away_qb_team_starts_pre": [20.0, 20.0, 20.0],
    })
    mask = _build_gate_mask(df)
    assert mask[0]  # home_qb_changed
    assert mask[1]  # home_starts < 17
    assert not mask[2]  # both experienced


def test_gate_mask_no_columns():
    """Gate mask handles missing columns (missing starts -> gate active)."""
    df = pd.DataFrame({
        "home_qb_changed": [pd.NA],
    })
    mask = _build_gate_mask(df)
    # Missing start columns default to 0 which is < 17 -> gate active
    assert bool(mask[0]) is True


def test_run_retest_experiment_report():
    """Report is generated and contains expected sections."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        report_path = run_retest_experiment(
            ft_path="data/features/nfl/feature_table.parquet",
            report_path=f.name,
        )
        content = Path(report_path).read_text()
        assert "Incumbent (Platt)" in content
        assert "Incumbent + Weather" in content
        assert "Incumbent + Scheduling" in content
        assert "Incumbent + Injury" in content
        assert "No model beats incumbent" in content
        assert "0.6305" in content or "0.6200" in content


def test_run_retest_experiment_importable():
    """Module is importable from CLI."""
    from sportslab import cli
    assert hasattr(cli, "retest_rejected_features_cmd")
