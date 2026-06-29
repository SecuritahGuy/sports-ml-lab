"""Tests for turnover features module and experiment."""

import numpy as np
import pandas as pd

from sportslab.features.turnovers import (
    TURNOVER_COLUMNS,
    _compute_rolling,
    compute_turnover_features,
)


def test_rolling_shift_excludes_current():
    """Rolling mean should exclude the current value via shift(1)."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _compute_rolling(s, window=3)
    # idx 3: mean of [1, 2, 3] = 2.0 (excludes current 4.0)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 1.0  # mean of [1]
    assert result.iloc[2] == 1.5  # mean of [1, 2]
    assert result.iloc[3] == 2.0  # mean of [1, 2, 3]


def test_column_completeness():
    """All TURNOVER_COLUMNS should be present after compute."""
    rows = [
        {"season": 2021, "week": 1, "home_team": "ARI", "away_team": "TEN",
         "game_id": "2021_01_ARI_TEN", "home_win": 1},
        {"season": 2021, "week": 2, "home_team": "ARI", "away_team": "MIN",
         "game_id": "2021_02_ARI_MIN", "home_win": 1},
    ]
    df = pd.DataFrame(rows)
    result = compute_turnover_features(df)
    for col in TURNOVER_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"
    assert len(TURNOVER_COLUMNS) == 6


def test_feature_values_type():
    """Features should be float and not NaN (filled with 0)."""
    rows = [
        {"season": 2021, "week": 1, "home_team": "ARI", "away_team": "TEN",
         "game_id": "2021_01_ARI_TEN", "home_win": 1},
    ]
    df = pd.DataFrame(rows)
    result = compute_turnover_features(df)
    for col in TURNOVER_COLUMNS:
        assert result[col].dtype in (np.float64, np.float32, float), f"Wrong dtype for {col}"
        assert not result[col].isna().any(), f"NaN found in {col}"


def test_diff_equals_home_minus_away():
    """to_net_diff should equal home minus away."""
    rows = [
        {"season": 2021, "week": 1, "home_team": "ARI", "away_team": "TEN",
         "game_id": "2021_01_ARI_TEN", "home_win": 1},
    ]
    df = pd.DataFrame(rows)
    result = compute_turnover_features(df)
    for w in (3, 5):
        expected = result[f"home_to_net_{w}"] - result[f"away_to_net_{w}"]
        assert np.allclose(result[f"to_net_diff_{w}"], expected)


# ── Experiment tests ──


def test_experiment_import():
    from sportslab.evaluation.turnover_experiment import run_turnover_experiment
    assert callable(run_turnover_experiment)


def test_experiment_constants():
    from sportslab.evaluation.turnover_experiment import MODEL_VARIANTS
    assert len(MODEL_VARIANTS) == 5
    names = [n for n, _, _ in MODEL_VARIANTS]
    assert "incumbent" in names
    assert "to_net_3" in names
    assert "to_net_5" in names
    assert "to_net_both" in names


def test_experiment_folds():
    from sportslab.evaluation.turnover_experiment import ROLLING_FOLDS
    assert len(ROLLING_FOLDS) == 3


def test_cli_importable():
    from sportslab.cli import turnover_cmd
    assert turnover_cmd is not None


# ── Integration: runs on real feature table (2 rows) ──


def test_turnover_feature_ranges():
    """Features should be in reasonable ranges for real data."""
    df = pd.read_parquet("data/features/nfl/feature_table.parquet")
    small = df.head(20).copy()
    result = compute_turnover_features(small)
    for col in TURNOVER_COLUMNS:
        vals = result[col].values
        assert np.all(np.isfinite(vals)), f"Non-finite in {col}"
        assert np.all(np.abs(vals) < 100), f"Outlier in {col}: {vals}"
