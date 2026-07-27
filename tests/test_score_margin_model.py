"""Tests for score-margin distribution model."""

import numpy as np
import pytest

from sportslab.evaluation.score_margin_model import (
    ScoreMarginModel,
    estimate_sigma_from_residuals,
    margin_to_win_prob,
    predict_full_distribution,
)


class TestEstimateSigma:
    def test_perfect_prediction(self):
        y_true = np.array([3.0, -7.0, 10.0])
        y_pred = np.array([3.0, -7.0, 10.0])
        sigma = estimate_sigma_from_residuals(y_true, y_pred)
        assert sigma == 0.0

    def test_imperfect_prediction(self):
        y_true = np.array([3.0, -7.0, 10.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        sigma = estimate_sigma_from_residuals(y_true, y_pred)
        expected = np.sqrt(np.mean(np.array([3.0, -7.0, 10.0]) ** 2))
        assert sigma == pytest.approx(expected)


class TestMarginToWinProb:
    def test_large_positive_margin(self):
        assert margin_to_win_prob(28.0, 14.0) == pytest.approx(0.9772, abs=1e-4)

    def test_zero_margin(self):
        assert margin_to_win_prob(0.0, 14.0) == 0.5

    def test_large_negative_margin(self):
        assert margin_to_win_prob(-28.0, 14.0) == pytest.approx(0.0228, abs=1e-4)

    def test_small_sigma(self):
        assert margin_to_win_prob(3.0, 1.0) == pytest.approx(0.9987, abs=1e-4)


class TestPredictFullDistribution:
    def test_quantiles_symmetric(self):
        mu = np.array([0.0])
        sigma = 14.0
        quantiles = predict_full_distribution(mu, sigma)
        assert quantiles["margin_q50"] == pytest.approx(0.0, abs=0.1)
        assert quantiles["margin_q05"] == pytest.approx(-23.0, abs=1.0)
        assert quantiles["margin_q95"] == pytest.approx(23.0, abs=1.0)

    def test_quantiles_positive(self):
        mu = np.array([7.0])
        sigma = 14.0
        quantiles = predict_full_distribution(mu, sigma)
        assert quantiles["margin_q50"] == pytest.approx(7.0, abs=0.1)
        assert quantiles["margin_q05"] < quantiles["margin_q25"] < quantiles["margin_q50"]


class TestScoreMarginModel:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        true_coef = np.array([0.5, -0.3, 0.8])
        y = X @ true_coef + rng.normal(0, 1.0, n)

        model = ScoreMarginModel(fit_intercept=False)
        model.fit(X, y)

        assert model.coef_ is not None
        assert np.allclose(model.coef_, true_coef, atol=0.3)
        assert model.sigma_ is not None
        assert model.sigma_ < 1.5

    def test_predict_win_prob(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        true_coef = np.array([0.5, -0.3])
        y = X @ true_coef + rng.normal(0, 2.0, n)

        model = ScoreMarginModel(fit_intercept=True)
        model.fit(X, y)

        probs = model.predict_win_prob(X)
        assert len(probs) == n
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_win_prob_with_margin(self):
        X = np.array([[1.0, 0.5], [-0.5, 1.0], [0.0, 0.0]])
        y = np.array([10.0, -10.0, 0.0])
        model = ScoreMarginModel(fit_intercept=True)
        model.fit(X, y)

        result = model.predict_win_prob_with_margin(X)
        assert "home_win_prob" in result
        assert "predicted_margin" in result
        assert "margin_q50" in result
        assert len(result["home_win_prob"]) == 3
        assert result["predicted_margin"][2] < result["predicted_margin"][0]

    def test_intercept_default(self):
        model = ScoreMarginModel()
        assert model.fit_intercept is True
        assert model.intercept_ is None

    def test_no_coef_before_fit(self):
        model = ScoreMarginModel()
        assert model.coef_ is None
        assert model.sigma_ is None
