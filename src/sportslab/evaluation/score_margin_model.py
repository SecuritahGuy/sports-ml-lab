"""Score-margin distribution model — research / shadow experiment.

Instead of predicting win probability directly (Elo → logistic), models the
full distribution of home_score - away_score as Normal(μ, σ²).

    μ = β₀ + β₁ * elo_diff + β₂ * qb_changed + β₃ * rolling_mov_3
    σ = estimated from training residuals

Home win probability = P(margin > 0) = Φ(μ / σ)

Advantages over Elo+Platt:
  - Captures uncertainty (σ varies per game if heteroscedastic)
  - Naturally calibrated at extremes (no Platt distortion)
  - Produces full margin distribution (blowout probability, credible intervals)

This is a SHADOW experiment — no deployment pressure.
Not compared against incumbent for promotion.
"""

from typing import Dict, Optional

import numpy as np
from scipy.stats import norm


def estimate_sigma_from_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "constant",
) -> float:
    """Estimate σ of margin residuals.

    Args:
        y_true: Actual margins (home_score - away_score).
        y_pred: Predicted margins.
        method: 'constant' — single σ from RMSE.
                'grouped' — σ varies by bucket (future).

    Returns:
        σ estimate.
    """
    residuals = y_true - y_pred
    return float(np.sqrt(np.mean(residuals ** 2)))


def margin_to_win_prob(mu: float, sigma: float) -> float:
    """Convert predicted margin to win probability."""
    return float(norm.cdf(mu / sigma))


def predict_full_distribution(
    mu: np.ndarray,
    sigma: float,
    quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> Dict[str, np.ndarray]:
    """Return quantiles of the margin distribution."""
    results = {}
    for q in quantiles:
        pct = int(round(q * 100))
        results[f"margin_q{pct:02d}"] = norm.ppf(q, loc=mu, scale=sigma)
    return results


class ScoreMarginModel:
    """Linear model for score margin ~ Elo features.

    Fits on historical (season, week, team, margin) data and predicts
    home win probability from the full margin distribution.

    Shadow experiment — not compared against incumbent.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.sigma_: Optional[float] = None
        self.feature_names_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit OLS regression for margin ~ features.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target — home_score - away_score margin.
        """
        n = X.shape[0]
        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(n), X])
        else:
            X_aug = X

        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        y_pred = X_aug @ beta
        self.sigma_ = estimate_sigma_from_residuals(y, y_pred)

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Predict expected margin."""
        n = X.shape[0]
        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(n), X])
        else:
            X_aug = X
        beta = np.concatenate([[self.intercept_], self.coef_])
        return X_aug @ beta

    def predict_win_prob(self, X: np.ndarray) -> np.ndarray:
        """Predict home win probability from margin distribution."""
        mu = self.predict_margin(X)
        sigma = self.sigma_ if self.sigma_ is not None else 14.0
        return np.array([margin_to_win_prob(m, sigma) for m in mu])

    def predict_win_prob_with_margin(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return win probabilities and margin quantiles."""
        mu = self.predict_margin(X)
        sigma = self.sigma_ if self.sigma_ is not None else 14.0
        probs = np.array([margin_to_win_prob(m, sigma) for m in mu])
        quantiles = predict_full_distribution(mu, sigma)
        return {
            "home_win_prob": probs,
            "predicted_margin": mu,
            **quantiles,
        }
