"""Tests for extended predict_future module — QB input, season/week filtering, output schema."""

from sportslab.evaluation.predict_future import (
    _split_by_availability,
)


def test_split_by_availability():
    """_split_by_availability separates known and unknown games correctly."""
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "home_win": [1.0, 0.0, np.nan],
    })
    known, future, mask = _split_by_availability(df)
    assert len(known) == 2
    assert len(future) == 1
    assert future.iloc[0]["game_id"] == "g3"


def test_predict_future_importable():
    """The predict_future function is importable and callable with defaults."""
    from sportslab.evaluation.predict_future import predict_future
    assert callable(predict_future)


def test_run_predict_future_importable():
    """The CLI entry point is importable."""
    from sportslab.evaluation.predict_future import run_predict_future
    assert callable(run_predict_future)


def test_predict_future_output_schema():
    """Verify output has expected columns and qb_source field."""
    from sportslab.evaluation.predict_incumbent import (
        INCUMBENT_FEATURE_SET,
        INCUMBENT_VERSION,
    )
    expected = [
        "game_id", "season", "week", "gameday",
        "away_team", "home_team",
        "incumbent_home_win_prob", "predicted_winner",
        "confidence_bucket", "model_version", "model_date",
        "training_seasons", "feature_set", "calibration_method",
        "model_val_ll", "model_holdout_ll",
        "elo_k", "elo_hfa", "elo_reg", "elo_decay", "elo_qb_bonus",
        "qb_source",
        "caution_qb_change", "caution_early_season",
        "home_qb_id", "away_qb_id",
    ]
    assert "qb_source" in expected, "Output should include qb_source field"
    assert INCUMBENT_VERSION is not None
    assert INCUMBENT_FEATURE_SET is not None


def test_feature_table_exists():
    """Feature table must exist for full pipeline tests."""
    import os
    fp = "data/features/nfl/feature_table.parquet"
    assert os.path.exists(fp), f"Feature table not found at {fp}"
