"""Tests for regularized_logistic_meta experiment module."""

import numpy as np
import pandas as pd

from sportslab.evaluation.regularized_logistic_meta import (  # noqa: F401
    FEATURE_SETS,
    META_FEATURE_FRIENDLY,
    MIN_PROMOTION_DELTA,
    PROHIBITED_COLS,
    V3_HOLDOUT_LL,
    V3_VAL_LL,
    _build_gate_mask,
    _build_meta_features,
    _early_season_flag,
    _logit,
    _sigmoid,
    _week_sin_cos,
    run_regularized_logistic_meta,
)

# ── Sigmoid / Logit ──


def test_sigmoid_bounds():
    """Sigmoid of ±inf is 0/1."""
    x = np.array([-1e10, 0, 1e10])
    s = _sigmoid(x)
    assert np.allclose(s[0], 0.0, atol=1e-10)
    assert np.allclose(s[1], 0.5, atol=1e-6)
    assert np.allclose(s[2], 1.0, atol=1e-10)


def test_sigmoid_no_nan():
    """Sigmoid should never produce NaN."""
    x = np.array([-500, 500, 0.0])
    s = _sigmoid(x)
    assert not np.any(np.isnan(s))


def test_logit_bounds():
    """Logit of 0.5 is 0."""
    p = np.array([0.5])
    assert np.allclose(_logit(p), [0.0], atol=1e-6)


def test_logit_clip():
    """Logit clips to avoid ±inf."""
    p = np.array([1e-20, 1 - 1e-20])
    res = _logit(p)
    assert np.all(np.isfinite(res))


def test_sigmoid_logit_inverse():
    """Sigmoid(logit(p)) ≈ p."""
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    assert np.allclose(p, _sigmoid(_logit(p)), atol=1e-10)


# ── Week sin/cos ──


def test_week_sin_cos_shape():
    """Returns 2-column array."""
    weeks = np.array([1, 9, 18])
    result = _week_sin_cos(weeks)
    assert result.shape == (3, 2)


def test_week_sin_cos_period():
    """sin(1)^2 + cos(1)^2 ≈ 1 for all weeks."""
    weeks = np.arange(1, 19)
    result = _week_sin_cos(weeks)
    norms = np.sum(result**2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


# ── Early season flag ──


def test_early_season_flag():
    """Weeks 1-4 are early season (1), 5+ are not (0)."""
    df = pd.DataFrame({"week": [1, 4, 5, 18]})
    flag = _early_season_flag(df)
    np.testing.assert_array_equal(flag, [1, 1, 0, 0])


# ── Build gate mask ──


def test_gate_mask_no_changed():
    """No qb_changed and many starts → gate closed."""
    df = pd.DataFrame(
        {
            "home_qb_changed": [0],
            "away_qb_changed": [0],
            "home_qb_team_starts_pre": [50],
            "away_qb_team_starts_pre": [60],
        }
    )
    mask = _build_gate_mask(df)
    assert not mask[0]


def test_gate_mask_changed():
    """Home qb_changed → gate open."""
    df = pd.DataFrame(
        {
            "home_qb_changed": [1],
            "away_qb_changed": [0],
            "home_qb_team_starts_pre": [50],
            "away_qb_team_starts_pre": [60],
        }
    )
    mask = _build_gate_mask(df)
    assert mask[0]


def test_gate_mask_few_starts():
    """Few career starts → gate open."""
    df = pd.DataFrame(
        {
            "home_qb_changed": [0],
            "away_qb_changed": [0],
            "home_qb_team_starts_pre": [5],
            "away_qb_team_starts_pre": [60],
        }
    )
    mask = _build_gate_mask(df)
    assert mask[0]


def test_gate_mask_missing_starts():
    """Missing starts treated as 0 → gate open."""
    df = pd.DataFrame(
        {
            "home_qb_changed": [0],
            "away_qb_changed": [0],
            "home_qb_team_starts_pre": [np.nan],
            "away_qb_team_starts_pre": [np.nan],
        }
    )
    mask = _build_gate_mask(df)
    assert mask[0]


# ── Build meta features ──


def test_build_meta_features_logit_only():
    """No extra columns → just logit."""
    df = pd.DataFrame({"dummy": [1, 2]})
    v3 = np.array([0.3, 0.7])
    result = _build_meta_features(df, v3, [])
    assert result.shape == (2, 1)
    assert np.allclose(result[:, 0], _logit(v3))


def test_build_meta_features_with_extra():
    """Extra columns are stacked."""
    df = pd.DataFrame({"div_game": [0, 1], "is_dome": [1, 0]})
    v3 = np.array([0.5, 0.5])
    result = _build_meta_features(df, v3, ["div_game", "is_dome"])
    assert result.shape == (2, 3)
    assert np.allclose(result[:, 0], _logit(v3))
    np.testing.assert_array_equal(result[:, 1], [0, 1])
    np.testing.assert_array_equal(result[:, 2], [1, 0])


def test_build_meta_features_missing_cols():
    """Missing extra columns → silently skipped."""
    df = pd.DataFrame({"div_game": [0]})
    v3 = np.array([0.5])
    result = _build_meta_features(df, v3, ["div_game", "missing_col"])
    assert result.shape == (1, 2)  # logit + div_game only


# ── Constants ──


def test_min_promotion_delta_positive():
    assert MIN_PROMOTION_DELTA > 0


def test_v3_ref_values():
    assert isinstance(V3_VAL_LL, float)
    assert isinstance(V3_HOLDOUT_LL, float)
    assert 0.5 < V3_VAL_LL < 0.8
    assert 0.5 < V3_HOLDOUT_LL < 0.8


def test_feature_sets_nonempty():
    assert len(FEATURE_SETS) > 0


def test_prohibited_cols_nonempty():
    assert len(PROHIBITED_COLS) > 0


def test_meta_feature_friendly_matches():
    """Every feature set has a friendly description."""
    for k in FEATURE_SETS:
        assert k in META_FEATURE_FRIENDLY


def test_no_market_or_score_in_features():
    """Verify feature set columns don't contain prohibited cols."""
    for name, cols in FEATURE_SETS.items():
        for c in cols:
            assert c not in PROHIBITED_COLS, (
                f"Feature set '{name}' contains prohibited column '{c}'"
            )


# ── Module smoke test ──


def test_importable():
    """Module is importable with expected public API."""
    from sportslab.evaluation import regularized_logistic_meta

    assert hasattr(regularized_logistic_meta, "run_regularized_logistic_meta")


def test_function_is_callable():
    """run_regularized_logistic_meta is a callable."""
    assert callable(run_regularized_logistic_meta)
