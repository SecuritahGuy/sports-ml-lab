"""Tests for 2025 week-by-week simulation module."""

from sportslab.evaluation.simulate_2025 import (
    QB_MODE_LIVE,
    QB_MODE_ORACLE,
    _compute_metrics,
    _get_2025_weeks,
    _load_feature_table,
)


def test_qb_mode_constants():
    """QB mode constants are defined."""
    assert QB_MODE_ORACLE == "oracle"
    assert QB_MODE_LIVE == "live_pregame"


def test_compute_metrics_all_correct():
    """Perfect predictions give log loss ~0, brier ~0, acc=1."""
    import numpy as np
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.99, 0.01, 0.99, 0.01, 0.99])
    m = _compute_metrics(y_true, y_prob)
    assert m["log_loss"] < 0.1
    assert m["brier"] < 0.01
    assert m["accuracy"] == 1.0


def test_compute_metrics_all_wrong():
    """All-wrong predictions give high log loss."""
    import numpy as np
    y_true = np.array([1, 0, 1])
    y_prob = np.array([0.01, 0.99, 0.01])
    m = _compute_metrics(y_true, y_prob)
    assert m["log_loss"] > 2.0
    assert m["accuracy"] == 0.0


def test_compute_metrics_random():
    """Random (0.5) predictions give log loss ~0.693."""
    import numpy as np
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    m = _compute_metrics(y_true, y_prob)
    assert abs(m["log_loss"] - 0.6931) < 0.01
    assert abs(m["brier"] - 0.25) < 0.01


def test_compute_metrics_empty():
    """Empty input returns empty dict."""
    import numpy as np
    m = _compute_metrics(np.array([]), np.array([]))
    assert m == {}


def test_compute_metrics_clipped():
    """Extreme probabilities are clipped to avoid infinity."""
    import numpy as np
    y_true = np.array([1, 0])
    y_prob = np.array([0.0, 1.0])
    m = _compute_metrics(y_true, y_prob)
    assert np.isfinite(m["log_loss"]), "Log loss should be finite after clipping"
    assert m["log_loss"] > 0


def test_load_feature_table():
    """Feature table loads and has expected columns."""
    df = _load_feature_table()
    assert "game_id" in df.columns
    assert "season" in df.columns
    assert "home_team" in df.columns
    assert "away_team" in df.columns
    assert "home_win" in df.columns
    assert df["season"].min() >= 2021


def test_get_2025_weeks():
    """2025 weeks are properly extracted."""
    df = _load_feature_table()
    weeks = _get_2025_weeks(df)
    assert len(weeks) > 0
    assert min(weeks) >= 1
    assert max(weeks) <= 22


def test_simulate_2025_importable():
    """simulate_2025 function is callable."""
    from sportslab.evaluation.simulate_2025 import simulate_2025
    assert callable(simulate_2025)


def test_simulate_2025_output_schema():
    """Simulation output includes QB id columns."""
    from sportslab.evaluation.predict_incumbent import (
        BEST_DECAY,
        BEST_HFA,
        BEST_K,
        BEST_QB_BONUS,
        BEST_REG,
        INCUMBENT_DATE,
        INCUMBENT_FEATURE_SET,
        INCUMBENT_VERSION,
    )
    from sportslab.evaluation.simulate_2025 import (
        _load_feature_table,
    )
    df = _load_feature_table()
    assert "home_qb_id" in df.columns
    assert "away_qb_id" in df.columns
    # All metadata constants should be defined
    assert INCUMBENT_VERSION == "v2.0.0"
    assert INCUMBENT_DATE is not None
    assert INCUMBENT_FEATURE_SET is not None
    assert isinstance(BEST_K, int)
    assert isinstance(BEST_HFA, int)
    assert isinstance(BEST_REG, float)
    assert isinstance(BEST_DECAY, int)
    assert isinstance(BEST_QB_BONUS, float)
