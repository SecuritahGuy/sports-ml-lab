"""Tests for calibration_audit module."""

import numpy as np

from sportslab.evaluation.calibration_audit import (
    brier_decomposition,
    ece_mce,
    home_favorite_directional_error,
    reliability_diagram_text,
    sharpness_buckets,
)

# ── ECE / MCE ──


def test_ece_perfect():
    """Perfect calibration → ECE = MCE = 0."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    result = ece_mce(y_true, y_prob, n_bins=10)
    assert result["ece"] >= 0
    assert result["n"] == 6
    assert len(result["buckets"]) > 0


def test_ece_range():
    """ECE is bounded [0, 1]."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    result = ece_mce(y_true, y_prob, n_bins=10)
    assert 0.0 <= result["ece"] <= 1.0
    assert 0.0 <= result["mce"] <= 1.0


def test_ece_nan_input():
    """NaN in y_true is filtered."""
    y_true = np.array([0, 1, np.nan, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    result = ece_mce(y_true, y_prob)
    assert result["n"] == 4


def test_mce_geq_ece():
    """MCE >= ECE always."""
    rng = np.random.default_rng(99)
    y_true = rng.integers(0, 2, size=500)
    y_prob = rng.uniform(0, 1, size=500)
    result = ece_mce(y_true, y_prob)
    assert result["mce"] >= result["ece"]


def test_ece_bucket_structure():
    """Each bucket has required fields."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85])
    result = ece_mce(y_true, y_prob, n_bins=10)
    for b in result["buckets"]:
        assert "bin" in b
        assert "n" in b
        assert "mean_pred" in b
        assert "mean_actual" in b
        assert "cal_error" in b
        assert b["n"] > 0


# ── Brier decomposition ──


def test_brier_decomposition_perfect():
    """Perfect predictions → reliability = 0."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    result = brier_decomposition(y_true, y_prob)
    assert result["reliability"] <= 0.01  # close to zero
    assert result["brier_score"] >= 0


def test_brier_decomposition_sum():
    """Decomposed Brier ≈ raw Brier."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    result = brier_decomposition(y_true, y_prob)
    diff = abs(result["brier_score"] - result["brier_decomposed"])
    assert diff < 0.01, f"Brier decomposition mismatch: {diff}"


def test_brier_decomposition_components():
    """All components present and non-negative."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.3, 0.7])
    result = brier_decomposition(y_true, y_prob)
    for key in ["brier_score", "uncertainty", "resolution", "reliability"]:
        assert key in result
        assert result[key] >= 0


# ── Sharpness ──


def test_sharpness_buckets_total():
    """Sharpness buckets sum to total N."""
    y_prob = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    result = sharpness_buckets(y_prob)
    total = sum(b["n"] for b in result["bins"])
    assert total == result["n"] == 5


def test_sharpness_has_all_bins():
    """All 10 bins are present."""
    y_prob = np.linspace(0.05, 0.95, 100)
    result = sharpness_buckets(y_prob)
    assert len(result["bins"]) == 10


# ── Home-favorite directional error ──


def test_home_favorite_directional_error_all_wrong():
    """All predictions >0.5 and wrong → all overconfident."""
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7])
    result = home_favorite_directional_error(y_true, y_prob)
    assert result["overconfident"] == 3
    assert result["underconfident"] == 0


def test_home_favorite_directional_error_nan():
    """NaN in y_true is filtered."""
    y_true = np.array([0, np.nan, 1, 1])
    y_prob = np.array([0.9, 0.5, 0.9, 0.8])
    result = home_favorite_directional_error(y_true, y_prob)
    assert result["n"] == 3


# ── Reliability diagram text ──


def test_reliability_diagram_no_crash():
    """Reliability diagram produces string output for any bucket list."""
    buckets = [
        {"bin": "[0.0, 0.1)", "n": 10, "mean_pred": 0.05, "mean_actual": 0.1, "cal_error": 0.05},
        {"bin": "[0.9, 1.0)", "n": 20, "mean_pred": 0.95, "mean_actual": 0.85, "cal_error": 0.1},
    ]
    text = reliability_diagram_text(buckets)
    assert isinstance(text, str)
    assert len(text) > 50
    assert "Reliability Diagram" in text


def test_reliability_diagram_empty():
    """Empty bucket list produces minimal output."""
    text = reliability_diagram_text([])
    assert isinstance(text, str)
    assert "Reliability Diagram" in text


# ── Module smoke ──


def test_importable():
    """Module is importable with expected public API."""
    from sportslab.evaluation import calibration_audit
    assert hasattr(calibration_audit, "run_calibration_audit")


def test_run_is_callable():
    """run_calibration_audit is callable."""
    from sportslab.evaluation.calibration_audit import run_calibration_audit
    assert callable(run_calibration_audit)
