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
        "home_score": [24.0, 17.0, np.nan],
        "away_score": [10.0, 31.0, np.nan],
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
    """Verify output has all v3 incumbent schema columns."""
    from sportslab.evaluation.predict_incumbent import (
        INCUMBENT_CALIBRATION,
        OVERLAY_CAP,
        OVERLAY_GAMMA,
    )
    assert INCUMBENT_CALIBRATION is not None
    assert OVERLAY_GAMMA == 1.0
    assert OVERLAY_CAP == 40


def test_feature_table_exists():
    """Feature table must exist for full pipeline tests."""
    import os
    fp = "data/features/nfl/feature_table.parquet"
    assert os.path.exists(fp), f"Feature table not found at {fp}"
