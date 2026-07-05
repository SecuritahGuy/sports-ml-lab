"""Tests for QB Lift rolling efficiency features."""

import numpy as np
import pandas as pd
import pytest

from sportslab.features.qb_lift import (
    QB_LIFT_COLUMNS,
    _short_to_long,
    compute_qb_game_stats,
    compute_qb_lift_features,
    compute_rolling_qb_features,
)


def test_short_to_long_exists():
    """Name conversion function exists."""
    assert callable(_short_to_long)


def test_qb_lift_columns_defined():
    """QB_LIFT_COLUMNS is defined with expected columns."""
    assert len(QB_LIFT_COLUMNS) >= 4
    assert "home_qb_epa_3" in QB_LIFT_COLUMNS
    assert "away_qb_epa_3" in QB_LIFT_COLUMNS
    assert "net_qb_epa_3" in QB_LIFT_COLUMNS
    assert "net_qb_epa_5" in QB_LIFT_COLUMNS


def test_compute_qb_game_stats_no_pbp():
    """compute_qb_game_stats returns empty DataFrame with empty PBP."""
    empty = pd.DataFrame()
    result = compute_qb_game_stats(empty)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_compute_qb_game_stats_no_qb_plays():
    """compute_qb_game_stats returns empty with no passer data."""
    pbp = pd.DataFrame({
        "game_id": ["g1"],
        "passer_player_id": [None],
        "passer_player_name": [None],
        "posteam": ["KC"],
        "epa": [0.5],
        "cpoe": [0.0],
        "success": [1],
    })
    result = compute_qb_game_stats(pbp)
    assert len(result) == 0


def test_compute_qb_game_stats_filters_below_min():
    """QB games with < 10 dropbacks are filtered."""
    rows = []
    for i in range(5):
        rows.append({
            "game_id": "g1",
            "passer_player_id": "P01",
            "passer_player_name": "P.Mahomes",
            "posteam": "KC",
            "epa": 0.3,
            "cpoe": 0.02,
            "success": 1,
        })
    pbp = pd.DataFrame(rows)
    result = compute_qb_game_stats(pbp)
    assert len(result) == 0  # only 5 dropbacks, < 10


def test_compute_qb_game_stats_basic():
    """Basic per-game QB stat computation."""
    np.random.seed(42)
    rows = []
    for i in range(10):
        rows.append({
            "game_id": "g1",
            "passer_player_id": "P01",
            "passer_player_name": "P.Mahomes",
            "posteam": "KC",
            "epa": float(np.random.uniform(-0.5, 1.0)),
            "cpoe": float(np.random.uniform(-0.1, 0.1)),
            "success": 1 if np.random.random() > 0.4 else 0,
        })
    pbp = pd.DataFrame(rows)
    result = compute_qb_game_stats(pbp)
    assert len(result) == 1
    assert result.iloc[0]["dropbacks"] == 10
    assert "epa_per_db" in result.columns
    assert result.iloc[0]["epa_per_db"] == pytest.approx(pbp["epa"].mean())
    assert result.iloc[0]["avg_cpoe"] == pytest.approx(pbp["cpoe"].mean())


def test_compute_qb_game_stats_multiple_games():
    """Multiple games produce multiple rows."""
    np.random.seed(99)
    rows = []
    for g in ["g1", "g2"]:
        for i in range(10):
            rows.append({
                "game_id": g,
                "passer_player_id": "P01",
                "passer_player_name": "P.Mahomes",
                "posteam": "KC",
                "epa": 0.3,
                "cpoe": 0.01,
                "success": 1,
            })
    pbp = pd.DataFrame(rows)
    result = compute_qb_game_stats(pbp)
    assert len(result) == 2


def test_rolling_qb_features_single_game():
    """Single game => rolling values are NaN (no prior)."""
    stats = pd.DataFrame({
        "game_id": ["g1"],
        "passer_player_id": ["P01"],
        "epa_per_db": [0.5],
        "avg_cpoe": [0.02],
        "avg_success": [0.6],
        "dropbacks": [15],
        "total_epa": [7.5],
        "passer_player_name": ["P.Mahomes"],
        "posteam": ["KC"],
        "season": [2021],
        "week": [1],
    })
    rolling = compute_rolling_qb_features(stats, windows=[3, 5])
    assert len(rolling) == 1
    assert np.isnan(rolling.iloc[0]["rolling_epa_3"])
    assert np.isnan(rolling.iloc[0]["rolling_epa_5"])
    assert np.isnan(rolling.iloc[0]["rolling_cpoe_3"])


def test_rolling_qb_features_two_games():
    """Two games => rolling_3 averages game 1, rolling_5 is NaN."""
    stats = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "passer_player_id": ["P01", "P01"],
        "epa_per_db": [0.5, -0.2],
        "avg_cpoe": [0.02, -0.01],
        "avg_success": [0.6, 0.4],
        "dropbacks": [15, 12],
        "total_epa": [7.5, -2.4],
        "passer_player_name": ["P.Mahomes", "P.Mahomes"],
        "posteam": ["KC", "KC"],
        "season": [2021, 2021],
        "week": [1, 2],
    })
    rolling = compute_rolling_qb_features(stats, windows=[3, 5])
    assert len(rolling) == 2
    # First game: no prior -> NaN
    assert np.isnan(rolling.iloc[0]["rolling_epa_3"])
    # Second game: prior is game 1
    assert rolling.iloc[1]["rolling_epa_3"] == pytest.approx(0.5)
    # rolling_5 for game 2: only 1 prior out of 5 -> mean of 1
    assert rolling.iloc[1]["rolling_epa_5"] == pytest.approx(0.5)


def test_rolling_qb_features_multiple_qbs():
    """Multiple QBs across games compute separately."""
    stats = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "passer_player_id": ["P01", "P01", "P02"],
        "epa_per_db": [0.5, -0.2, 0.8],
        "avg_cpoe": [0.02, -0.01, 0.03],
        "avg_success": [0.6, 0.4, 0.7],
        "dropbacks": [15, 12, 20],
        "total_epa": [7.5, -2.4, 16.0],
        "passer_player_name": ["P.Mahomes", "P.Mahomes", "J.Burrow"],
        "posteam": ["KC", "KC", "CIN"],
        "season": [2021, 2021, 2021],
        "week": [1, 2, 3],
    })
    rolling = compute_rolling_qb_features(stats, windows=[3])
    # P01 has 2 rows, P02 has 1 row
    assert len(rolling) == 3
    p01_rows = rolling[rolling["passer_player_id"] == "P01"]
    p02_rows = rolling[rolling["passer_player_id"] == "P02"]
    assert len(p01_rows) == 2
    assert len(p02_rows) == 1
    # P02 first game: NaN
    assert np.isnan(p02_rows.iloc[0]["rolling_epa_3"])


@pytest.mark.slow
def test_compute_qb_lift_features_from_full_pbp():
    """Integration: compute QB lift on full PBP and feature table.

    This test requires the actual PBP parquet files to be present.
    """
    from pathlib import Path
    pbp_dir = "data/interim/nfl"
    ft_path = "data/features/nfl/feature_table.parquet"
    if not Path(pbp_dir).exists() or not Path(ft_path).exists():
        pytest.skip("PBP or feature table not available in CI")

    ft = pd.read_parquet(ft_path)
    result = compute_qb_lift_features(ft)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(ft)
    for col in QB_LIFT_COLUMNS:
        assert col in result.columns


def test_importable():
    """Experiment module imports without error."""
    from sportslab.evaluation import qb_lift_experiment
    assert hasattr(qb_lift_experiment, "run_qb_lift_experiment")
    assert hasattr(qb_lift_experiment, "QB_LIFT_VARIANTS")
    assert len(qb_lift_experiment.QB_LIFT_VARIANTS) >= 1
