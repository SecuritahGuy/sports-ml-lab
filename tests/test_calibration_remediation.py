"""Tests for calibration_remediation module."""

import numpy as np

from sportslab.evaluation.calibration_remediation import (
    _build_variants,
    _shrink,
    _temperature_scale,
    run_calibration_remediation,
)

# ── Temperature scaling ──


def test_temperature_identity():
    """T=1.0 returns probs unchanged."""
    probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    result = _temperature_scale(probs, 1.0)
    np.testing.assert_array_almost_equal(result, probs)


def test_temperature_range():
    """Temperature scaling preserves [0, 1]."""
    probs = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    for t in [0.8, 1.2, 2.0, 5.0]:
        result = _temperature_scale(probs, t)
        assert np.all(result >= 0.0), f"T={t}: below 0"
        assert np.all(result <= 1.0), f"T={t}: above 1"


def test_temperature_negative_raises():
    """Non-positive temperature raises ValueError."""
    probs = np.array([0.5, 0.5])
    np.testing.assert_raises(ValueError, _temperature_scale, probs, -1.0)
    np.testing.assert_raises(ValueError, _temperature_scale, probs, 0.0)


def test_temperature_monotonic():
    """Temperature scaling preserves order."""
    rng = np.random.default_rng(42)
    probs = np.sort(rng.uniform(0, 1, size=50))
    for t in [0.8, 1.5, 3.0]:
        scaled = _temperature_scale(probs, t)
        assert np.all(np.diff(scaled) >= -1e-10), f"T={t}: not monotonic"


# ── Shrinkage ──


def test_shrink_identity():
    """Alpha=0 returns probs unchanged."""
    probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    result = _shrink(probs, 0.0)
    np.testing.assert_array_almost_equal(result, probs)


def test_shrink_preserves_range():
    """Shrinkage preserves [0, 1]."""
    probs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    for alpha in [0.05, 0.1, 0.2, 0.5, 1.0]:
        result = _shrink(probs, alpha)
        assert np.all(result >= 0.0), f"alpha={alpha}: below 0"
        assert np.all(result <= 1.0), f"alpha={alpha}: above 1"


def test_shrink_alpha_raises():
    """Alpha outside [0,1] raises ValueError."""
    probs = np.array([0.5])
    np.testing.assert_raises(ValueError, _shrink, probs, -0.1)
    np.testing.assert_raises(ValueError, _shrink, probs, 1.5)


def test_shrink_target_effect():
    """Shrink toward target pulls probs toward it."""
    probs = np.array([0.9, 0.8])
    result = _shrink(probs, 0.5, target=0.5)
    expected = np.array([0.7, 0.65])
    np.testing.assert_array_almost_equal(result, expected)


# ── Variants ──


def test_variants_include_baseline():
    """_build_variants includes baseline."""
    variants = _build_variants()
    baselines = [v for v in variants if v["method"] == "baseline"]
    assert len(baselines) == 1
    assert baselines[0]["label"] == "Baseline (v3.0.0)"


def test_variants_count():
    """Variants count is reasonable."""
    variants = _build_variants()
    baseline = 1
    global_t = len([v for v in variants if v["method"] == "global_temperature"])
    gate_t = len([v for v in variants if v["method"] == "gate_temperature"])
    qb_t = len([v for v in variants if v["method"] == "qb_temperature"])
    qb_shrink = len([v for v in variants if v["method"] == "qb_shrink"])
    qb_shrink_br = len([v for v in variants if v["method"] == "qb_shrink_baserate"])
    assert global_t == 8, f"Expected 8 global T, got {global_t}"
    assert gate_t == 24, f"Expected 24 gate T, got {gate_t}"
    assert qb_t == 24, f"Expected 24 QB T, got {qb_t}"
    assert qb_shrink == 4, f"Expected 4 QB shrink, got {qb_shrink}"
    assert qb_shrink_br == 4, f"Expected 4 QB shrink base, got {qb_shrink_br}"
    total = baseline + global_t + gate_t + qb_t + qb_shrink + qb_shrink_br
    assert len(variants) == total, f"Expected {total}, got {len(variants)}"


# ── No market columns ──


def test_no_market_columns():
    """Variant labels do not contain market."""
    variants = _build_variants()
    for v in variants:
        assert "market" not in v["label"].lower()


def test_no_market_params():
    """Variant params do not contain market."""
    variants = _build_variants()
    for v in variants:
        for k in v["params"]:
            assert "market" not in k.lower()


# ── Experiment callable ──


def test_importable():
    """Module is importable with expected public API."""
    from sportslab.evaluation import calibration_remediation
    assert hasattr(calibration_remediation, "run_calibration_remediation")


def test_run_is_callable():
    """run_calibration_remediation is callable."""
    assert callable(run_calibration_remediation)
