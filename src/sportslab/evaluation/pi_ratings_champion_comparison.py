"""Standalone champion comparison: v3.0.0 vs Pi-Ratings best config.

Goal:
  Run both models through the exact same pipeline (fold-safe Platt + QB overlay)
  and compare head-to-head. Resolves the recomputation issue from the 144-combo
  Pi-Ratings grid, where the v3 baseline was re-computed internally rather than
  compared against the frozen v3.0.0 champion directly.

Champion configs:
  - v3.0.0: Elo(K=36, HFA=40, reg=0.1, decay=32) + Platt(qb_changed, mov_3) + QB overlay
  - Pi-Ratings best (from 144-combo grid): alpha=0.5, base_k=28, hk_ratio=1.25,
    HFA=30, reg=0.0 + Platt + QB overlay

Promotion requires Δ >= 0.001 on BOTH rolling val and 2025 holdout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import compute_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features, compute_pi_ratings_features
from sportslab.features.situational import compute_situational_features

FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
QB_GATE_GAMMA = 1.0
QB_GATE_CAP = 40
FEATURE_COLS = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]
MIN_PROMOTION_DELTA = 0.001
SEED = 42
ELO_TO_LOGIT = np.log(10) / 400.0

# Champion configs
V3_PARAMS = dict(k_factor=36, home_advantage=40,
                 preseason_regression=0.1, decay_half_life=32)
PI_PARAMS = dict(base_k=28, home_advantage=30,
                 preseason_regression=0.0, alpha=0.5, hk_ratio=1.25)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p + 1e-15))


def _build_qb_gate_mask(df: pd.DataFrame) -> np.ndarray:
    h_changed = df["home_qb_changed"].values.astype(float)
    a_changed = df["away_qb_changed"].values.astype(float)
    qb_changed_either = (h_changed == 1) | (a_changed == 1)
    h_starts_raw = df.get("home_qb_team_starts_pre", None)
    a_starts_raw = df.get("away_qb_team_starts_pre", None)
    low_starts = np.zeros(len(df), dtype=bool)
    if h_starts_raw is not None and a_starts_raw is not None:
        h_s = h_starts_raw.fillna(-1).values.astype(float)
        a_s = a_starts_raw.fillna(-1).values.astype(float)
        low_starts = ((h_s >= 0) & (h_s < 17)) | ((a_s >= 0) & (a_s < 17))
    return qb_changed_either | low_starts


def _apply_qb_overlay(
    base_logit: np.ndarray, qb_gate_mask: np.ndarray,
    home_qb_adj: np.ndarray, away_qb_adj: np.ndarray,
) -> np.ndarray:
    capped_h = np.clip(home_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    capped_a = np.clip(away_qb_adj, -QB_GATE_CAP, QB_GATE_CAP)
    net_adj = capped_h - capped_a
    overlay = QB_GATE_GAMMA * net_adj * ELO_TO_LOGIT
    return base_logit + overlay * qb_gate_mask.astype(float)


def _fold_frames(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for train_seasons, val_season in ROLLING_FOLDS:
        train_mask = df["season"].isin(train_seasons).values
        val_mask = (df["season"] == val_season).values
        frames.append((train_mask, val_mask))
    return frames


def _score_model(
    prob: np.ndarray, y: np.ndarray,
    feat: np.ndarray,
    qb_gate_mask: np.ndarray,
    home_qb_adj: np.ndarray, away_qb_adj: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    compute_holdout: bool = False,
) -> dict:
    """Return val LL and holdout LL for a model with fold-safe Platt + QB overlay."""

    # Val: per-fold Platt, evaluate on each fold's val set
    fold_lls = []
    hold_logits: list[np.ndarray] = []

    for train_mask, val_mask in folds:
        x_train = np.column_stack([prob[train_mask], feat[train_mask]])
        y_train = y[train_mask].astype(int)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, y_train)

        x_all = np.column_stack([prob, feat])
        platt_prob = pipe.predict_proba(x_all)[:, 1]
        base_logit = _logit(platt_prob)
        final_logit = _apply_qb_overlay(base_logit, qb_gate_mask, home_qb_adj, away_qb_adj)
        final_prob = _sigmoid(final_logit)

        val_prob = final_prob[val_mask]
        val_y = y[val_mask]
        valid = ~np.isnan(val_y)
        fold_lls.append(compute_metrics(val_y[valid], val_prob[valid])["log_loss"])

        hold_logits.append(final_logit)

    val_ll = float(np.mean(fold_lls))

    # Holdout: average logits across folds, evaluate on last fold's val
    holdout_mask = folds[-1][1]
    avg_logit = np.mean(hold_logits, axis=0)
    hold_prob = _sigmoid(avg_logit)[holdout_mask]
    hold_y = y[holdout_mask]
    valid = ~np.isnan(hold_y)
    hold_ll = compute_metrics(hold_y[valid], hold_prob[valid])["log_loss"]

    return {"val_ll": val_ll, "hold_ll": hold_ll}


def run_champion_comparison(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = "reports/experiments/pi_ratings_champion_comparison.md",
) -> str:
    """Compare v3.0.0 vs Pi-Ratings best through the same pipeline."""
    df_raw = pd.read_parquet(ft_path)

    # Build shared features
    df = compute_qb_features(df_raw)
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    y = df[TARGET_COLUMN].astype(float).values
    feat = df[FEATURE_COLS].values.astype(float)
    qb_gate_mask = _build_qb_gate_mask(df)
    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)
    folds = _fold_frames(df)

    # ── Model 1: v3.0.0 champion ──
    print("  Computing v3.0.0 champion...")
    v3 = compute_elo_features(df, **V3_PARAMS)
    v3_metrics = _score_model(
        v3["elo_prob"].values.astype(float), y, feat,
        qb_gate_mask, home_qb_adj, away_qb_adj, folds,
    )
    print(f"    val={v3_metrics['val_ll']:.4f}  hold={v3_metrics['hold_ll']:.4f}")

    # ── Model 2: Pi-Ratings best ──
    print("  Computing Pi-Ratings best...")
    pi = compute_pi_ratings_features(df, **PI_PARAMS)
    pi_metrics = _score_model(
        pi["pi_prob"].values.astype(float), y, feat,
        qb_gate_mask, home_qb_adj, away_qb_adj, folds,
    )
    print(f"    val={pi_metrics['val_ll']:.4f}  hold={pi_metrics['hold_ll']:.4f}")

    # ── Comparison ──
    delta_val = pi_metrics["val_ll"] - v3_metrics["val_ll"]
    delta_hold = pi_metrics["hold_ll"] - v3_metrics["hold_ll"]
    promoted = (delta_val <= -MIN_PROMOTION_DELTA) and (delta_hold <= -MIN_PROMOTION_DELTA)

    print(f"\n  Δval={delta_val:+.4f}  Δhold={delta_hold:+.4f}")
    print(f"  {'✅ PROMOTED' if promoted else '❌ NOT PROMOTED'}")

    # ── Write report ──
    lines = [
        "# Standalone Champion Comparison: v3.0.0 vs Pi-Ratings Best",
        "",
        "Both models run through the same pipeline: fold-safe Platt + QB overlay.",
        "",
        "## Configs",
        "",
        "### v3.0.0 Champion",
        "| Param | Value |",
        "|-------|-------|",
        "| Model | Standard Elo |",
        "| K | 36 |",
        "| HFA | 40 |",
        "| Regression | 0.1 |",
        "| Decay | 32 |",
        "| MOV type | capped_linear (scale=0.05, cap=2.0) |",
        "| Features | qb_changed + rolling_mov_3 |",
        "| Calibration | Platt per fold |",
        "| QB overlay | gamma=1.0, cap=40 |",
        "",
        "### Pi-Ratings Best (alpha=0.5, base_k=28, hk_ratio=1.25, HFA=30, reg=0.0)",
        "| Param | Value |",
        "|-------|-------|",
        "| Model | Pi-Ratings (power-law MOV, asymmetric K) |",
        "| α (power) | 0.5 |",
        "| base_k | 28 |",
        "| hk_ratio | 1.25 |",
        "| HFA | 30 |",
        "| Regression | 0.0 |",
        "| Features | qb_changed + rolling_mov_3 |",
        "| Calibration | Platt per fold |",
        "| QB overlay | gamma=1.0, cap=40 |",
        "",
        "## Results",
        "",
        "| Model | Avg Val LL | Hold LL |",
        "|-------|-----------|---------|",
        f"| v3.0.0 champion | {v3_metrics['val_ll']:.4f} | {v3_metrics['hold_ll']:.4f} |",
        f"| Pi-Ratings best | {pi_metrics['val_ll']:.4f} | {pi_metrics['hold_ll']:.4f} |",
        "",
        f"**Δ (Pi − v3):** val={delta_val:+.4f}, hold={delta_hold:+.4f}",
        "",
        f"**Promotion threshold:** Δ ≤ −{MIN_PROMOTION_DELTA} on BOTH val AND holdout",
        "",
f"**Verdict: {'✅ PROMOTED' if promoted else '❌ NOT PROMOTED'}**",
        "",
        f"Δval={delta_val:+.4f}, Δhold={delta_hold:+.4f}",
        "",
        "---",
        "Report generated by pi_ratings_champion_comparison.py",
        "",
    ]

    report = "\n".join(lines)
    report_p = Path(report_path)
    report_p.parent.mkdir(parents=True, exist_ok=True)
    report_p.write_text(report)
    print(f"\n  Report: {report_path}")
    return report_path
