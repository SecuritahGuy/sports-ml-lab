"""Tests for market baseline experiment."""

import numpy as np
import pandas as pd
import pytest

from sportslab.evaluation.market_baseline import (
    HOLDOUT_SEASON,
    ROLLING_FOLDS,
    _moneyline_to_prob,
    compute_market_features,
    run_market_baseline,
)


def test_moneyline_to_prob_favorite():
    """Negative moneyline (favorite) should give prob > 0.5."""
    prob = _moneyline_to_prob(-200)
    assert prob == pytest.approx(2 / 3, abs=1e-4)


def test_moneyline_to_prob_underdog():
    """Positive moneyline (underdog) should give prob < 0.5."""
    prob = _moneyline_to_prob(200)
    assert prob == pytest.approx(1 / 3, abs=1e-4)


def test_moneyline_to_prob_even():
    """Even moneyline (-100 or +100) should give prob ~0.5."""
    fav = _moneyline_to_prob(-100)
    dog = _moneyline_to_prob(100)
    assert fav == pytest.approx(0.5, abs=1e-4)
    assert dog == pytest.approx(0.5, abs=1e-4)


def test_moneyline_to_prob_extreme():
    """Extreme moneylines should map to near 0 or 1."""
    assert _moneyline_to_prob(-10000) > 0.99
    assert _moneyline_to_prob(10000) < 0.01


def test_compute_market_features_adds_column():
    """compute_market_features should add market_home_prob column."""
    df = pd.DataFrame(
        {
            "home_moneyline": [-150, 200, -110],
            "away_moneyline": [130, -240, -110],
        }
    )
    out = compute_market_features(df)
    assert "market_home_prob" in out.columns
    assert len(out) == 3


def test_market_prob_between_0_and_1():
    """Market probabilities should be in [0, 1] range."""
    df = pd.DataFrame(
        {
            "home_moneyline": [-500, -200, -110, 150, 300, 1000],
            "away_moneyline": [400, 170, -110, -180, -400, -2000],
        }
    )
    out = compute_market_features(df)
    assert out["market_home_prob"].between(0, 1).all()


def test_market_de_vig_sums_properly():
    """After de-vig, home and away probs should sum to 1."""
    df = pd.DataFrame(
        {
            "home_moneyline": [-150, 200, -110],
            "away_moneyline": [130, -240, -110],
        }
    )
    out = compute_market_features(df)
    away_implied = df["away_moneyline"].apply(_moneyline_to_prob)
    home_implied = df["home_moneyline"].apply(_moneyline_to_prob)
    away_fair = away_implied / (home_implied + away_implied)
    assert np.allclose(out["market_home_prob"] + away_fair, 1.0, atol=1e-10)


def test_rolling_folds_exclude_holdout():
    """No fold should use HOLDOUT_SEASON."""
    for train_s, val_s in ROLLING_FOLDS:
        assert HOLDOUT_SEASON not in train_s
        assert val_s != HOLDOUT_SEASON


def test_market_baseline_importable():
    """Experiment function should be callable."""
    assert callable(run_market_baseline)
    assert run_market_baseline.__doc__ is not None


def test_missing_moneyline_columns_raises():
    """Missing moneyline columns should raise KeyError."""
    df = pd.DataFrame({"home_win": [1, 0]})
    with pytest.raises(KeyError):
        compute_market_features(df)
