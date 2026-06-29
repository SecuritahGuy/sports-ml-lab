"""Tests for no-QB live-safe baseline comparison experiment."""

from sportslab.evaluation.no_qb_baseline import (
    NO_QB_FEATURE_COLS,
    _build_incumbent_feature_set,
    _build_no_qb_feature_set,
    _compute_metrics,
    _fit_platt,
    run_no_qb_baseline,
)


def test_no_qb_feature_cols_defined():
    """NO_QB_FEATURE_COLS should not contain QB features."""
    assert len(NO_QB_FEATURE_COLS) > 0
    for col in NO_QB_FEATURE_COLS:
        assert "qb" not in col.lower()


def test_build_no_qb_feature_set():
    """Feature set built without QB features."""
    import pandas as pd
    df = pd.DataFrame({
        "elo_prob": [0.5, 0.6],
        "home_rolling_mov_3": [2.0, 3.0],
        "away_rolling_mov_3": [-1.0, 0.0],
        "home_qb_changed": [0, 1],
        "away_qb_changed": [0, 0],
    })
    x = _build_no_qb_feature_set(df)
    assert x.shape[1] == 1 + len(NO_QB_FEATURE_COLS)


def test_build_incumbent_feature_set():
    """Incumbent feature set includes all FEATURE_COLS."""
    import pandas as pd
    df = pd.DataFrame({
        "elo_prob": [0.5],
        "home_qb_changed": [0],
        "away_qb_changed": [0],
        "home_rolling_mov_3": [2.0],
        "away_rolling_mov_3": [-1.0],
    })
    x = _build_incumbent_feature_set(df)
    assert x.shape[1] == 5  # elo_prob + 4 features


def test_fit_platt():
    """Platt fitting produces callable pipeline."""
    import numpy as np
    x = np.array([[0.5, 2.0], [0.6, -1.0], [0.3, 0.0]])
    y = np.array([1, 0, 1])
    pipe = _fit_platt(x, y)
    prob = pipe.predict_proba(x)[:, 1]
    assert len(prob) == 3
    assert all(0 <= p <= 1 for p in prob)


def test_compute_metrics():
    """Metrics computation returns expected structure."""
    import numpy as np
    y_true = np.array([1, 0, 1])
    y_prob = np.array([0.99, 0.01, 0.99])
    m = _compute_metrics(y_true, y_prob)
    assert "log_loss" in m
    assert "brier" in m
    assert "accuracy" in m
    assert m["accuracy"] == 1.0


def test_compute_metrics_empty():
    """Empty input returns empty dict."""
    import numpy as np
    assert _compute_metrics(np.array([]), np.array([])) == {}


def test_run_no_qb_baseline_importable():
    """run_no_qb_baseline is callable."""
    assert callable(run_no_qb_baseline)
