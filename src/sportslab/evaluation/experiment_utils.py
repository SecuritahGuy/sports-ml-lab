"""Shared experiment utilities — metric helpers, bootstrap, analysis.

These are extracted from qb_ablation.py and qb_continuity.py
to avoid duplication in new experiment modules. Existing experiments
are not modified to keep this refactor minimal.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
N_WORST = 20


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Log loss, Brier, accuracy. Filters NaN from y_true."""
    valid = ~np.isnan(y_true)
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {}
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    labels = np.array([0, 1])
    ll = float(sk_log_loss(y_true, y_prob, labels=labels))
    brier = float(np.mean((y_true - y_prob) ** 2))
    acc = float(np.mean((y_prob >= 0.5) == y_true))
    return {"log_loss": round(ll, 4), "brier": round(brier, 4),
            "accuracy": round(acc, 4)}


def fit_platt(x: np.ndarray, y: np.ndarray, seed: int = RANDOM_SEED) -> Pipeline:
    """StandardScaler + LogisticRegression (Platt-style)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)),
    ])
    pipe.fit(x, y)
    return pipe


def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Stack elo_prob with feature columns. Missing features silently dropped."""
    elo = df["elo_prob"].values
    avail = [c for c in feature_cols if c in df.columns]
    if avail:
        feat = df[avail].values
        return np.column_stack([elo, feat])
    return elo.reshape(-1, 1)


def bootstrap_delta(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Bootstrap Δ log loss = prob_b - prob_a.
    Negative Δ means challenger (b) is better.
    Returns (mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    p_a = prob_a[valid]
    p_b = prob_b[valid]
    eps = 1e-15
    p_a = np.clip(p_a, eps, 1 - eps)
    p_b = np.clip(p_b, eps, 1 - eps)
    n = len(y_t)
    deltas = np.zeros(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        ll_a = sk_log_loss(y_t[idx], p_a[idx])
        ll_b = sk_log_loss(y_t[idx], p_b[idx])
        deltas[i] = ll_b - ll_a
    mean_d = float(np.mean(deltas))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return round(mean_d, 4), round(ci_low, 4), round(ci_high, 4)


def calibration_buckets(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> List[Dict]:
    """Split predictions into 10 equal-width probability buckets."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    labels = [f"{int(i*10)}-{int((i+1)*10)}%" for i in range(10)]
    indices = np.clip(np.floor(y_p * 10).astype(int), 0, 9)
    results = []
    for i in range(10):
        mask = indices == i
        if mask.sum() == 0:
            continue
        results.append({
            "bucket": labels[i],
            "n": int(mask.sum()),
            "mean_pred": round(float(y_p[mask].mean()), 4),
            "mean_actual": round(float(y_t[mask].mean()), 4),
            "cal_error": round(float(abs(y_p[mask].mean() - y_t[mask].mean())), 4),
        })
    return results


def confidence_buckets(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> List[Dict]:
    """Split by confidence: 2 * |prob - 0.5|, 5 bins 0-20 to 80-100."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    confidence = np.abs(y_p - 0.5) * 2
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    indices = np.clip(np.floor(confidence / 0.2).astype(int), 0, 4)
    results = []
    for i in range(5):
        mask = indices == i
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_t[mask], y_p[mask])
        results.append({"bucket": labels[i], "n": int(mask.sum()), **m})
    return results


def worst_predictions(
    y_true: np.ndarray, y_prob: np.ndarray,
    game_ids: np.ndarray, teams: np.ndarray,
    n: int = N_WORST,
) -> List[Dict]:
    """Find the n worst predictions by log loss contribution."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    gids = game_ids[valid]
    teams_arr = teams[valid] if len(teams) == len(y_true) else np.array(["?"] * len(y_true))
    eps = 1e-15
    y_p = np.clip(y_p, eps, 1 - eps)
    contrib = -(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p))
    worst_idx = np.argsort(-contrib)[:n]
    results = []
    for i in worst_idx:
        results.append({
            "game_id": str(gids[i]),
            "team": str(teams_arr[i]),
            "actual": int(y_t[i]),
            "pred": round(float(y_p[i]), 4),
            "log_loss_contrib": round(float(contrib[i]), 4),
        })
    return results
