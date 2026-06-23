"""Market feature utilities for NFL prediction.

Converts American moneyline odds to implied probabilities, computes
no-vig (de-vigged) probabilities, and maps spread lines to win
probabilities via logistic regression.

All features are pregame-safe. Spread→prob mapping must be fit
only on training data per fold.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MARKET_FEATURE_COLUMNS = [
    "market_home_prob_novig",
    "market_away_prob_novig",
    "market_overround",
    "spread_home_prob",
    "elo_vs_market_edge",
    "market_favorite_flag",
    "market_underdog_flag",
    "spread_bucket",
    "home_moneyline_prob_raw",
    "away_moneyline_prob_raw",
]


def moneyline_to_prob(odds: float) -> float:
    """Convert American moneyline odds to implied probability.

    Args:
        odds: American odds (e.g., -150, +200). Must be non-zero.

    Returns:
        Implied probability in [0, 1].
    """
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    odds = float(odds)
    if abs(odds) < 1:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def moneyline_to_prob_array(odds: np.ndarray) -> np.ndarray:
    """Vectorized American moneyline to implied probability."""
    result = np.full_like(odds, np.nan, dtype=float)
    pos = odds > 0
    neg = odds < 0
    result[pos] = 100.0 / (odds[pos].astype(float) + 100.0)
    result[neg] = -odds[neg].astype(float) / (-odds[neg].astype(float) + 100.0)
    return result


def compute_novig_prob(
    home_moneyline: np.ndarray,
    away_moneyline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute no-vig home and away probabilities from moneylines.

    Args:
        home_moneyline: Home team closing moneyline odds.
        away_moneyline: Away team closing moneyline odds.

    Returns:
        Tuple of (home_novig_prob, away_novig_prob, overround).
    """
    home_implied = moneyline_to_prob_array(home_moneyline)
    away_implied = moneyline_to_prob_array(away_moneyline)
    overround = home_implied + away_implied
    home_novig = home_implied / overround
    away_novig = away_implied / overround
    return home_novig, away_novig, overround


def fit_spread_model(
    spread: np.ndarray,
    y: np.ndarray,
) -> Pipeline:
    """Fit a logistic model mapping spread line to home win probability.

    Args:
        spread: Spread line (positive = home favored).
        y: Binary home win labels (0/1).

    Returns:
        Fitted sklearn Pipeline with scaler + logistic regression.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(spread.reshape(-1, 1), y)
    return pipe


def compute_spread_probs(
    spread: np.ndarray,
    model: Pipeline,
) -> np.ndarray:
    """Apply a fitted spread→prob model to spread data.

    Args:
        spread: Spread line values.
        model: Fitted Pipeline from fit_spread_model().

    Returns:
        Home win probability array.
    """
    return model.predict_proba(spread.reshape(-1, 1))[:, 1]


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market-derived features for each game.

    Adds de-vigged moneyline probabilities, spread→prob placeholder,
    and diagnostic flags.

    Note: Spread→prob requires per-fold fitting via
    `fit_spread_model` on training data. The `spread_home_prob`
    column is initialized as NaN and should be filled per fold.

    Args:
        df: DataFrame with home_moneyline, away_moneyline, spread_line.

    Returns:
        DataFrame with added MARKET_FEATURE_COLUMNS.
    """
    out = df.copy()

    # De-vigged moneyline probabilities
    home_novig, away_novig, overround = compute_novig_prob(
        out["home_moneyline"].values,
        out["away_moneyline"].values,
    )
    out["market_home_prob_novig"] = home_novig
    out["market_away_prob_novig"] = away_novig
    out["market_overround"] = overround

    # Raw moneyline implied probabilities (with vig)
    out["home_moneyline_prob_raw"] = moneyline_to_prob_array(out["home_moneyline"].values)
    out["away_moneyline_prob_raw"] = moneyline_to_prob_array(out["away_moneyline"].values)

    # Spread→prob placeholder (filled per-fold)
    out["spread_home_prob"] = np.nan

    # Favorite/underdog flags (based on moneyline)
    home_is_fav = out["home_moneyline"] < 0
    away_is_fav = out["away_moneyline"] < 0
    out["market_favorite_flag"] = home_is_fav.astype(int)
    out["market_underdog_flag"] = away_is_fav.astype(int)

    # Spread buckets
    spread = out["spread_line"].values
    spread_bucket = np.full(len(spread), "", dtype=object)
    spread_bucket[spread < -7] = "big_away_fav"
    spread_bucket[(spread >= -7) & (spread < -3)] = "mod_away_fav"
    spread_bucket[(spread >= -3) & (spread < 0)] = "slight_away_fav"
    spread_bucket[spread == 0] = "pickem"
    spread_bucket[(spread > 0) & (spread <= 3)] = "slight_home_fav"
    spread_bucket[(spread > 3) & (spread <= 7)] = "mod_home_fav"
    spread_bucket[spread > 7] = "big_home_fav"
    out["spread_bucket"] = spread_bucket

    # Edge placeholder (filled later)
    out["elo_vs_market_edge"] = np.nan

    return out
