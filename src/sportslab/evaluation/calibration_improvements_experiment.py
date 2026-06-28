"""Calibration improvements experiment — era-split Platt + high-confidence shrinkage."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2
INCUMBENT_HOLDOUT_LL = 0.6262
INCUMBENT_FEATURES = ["home_qb_changed", "away_qb_changed", "home_rolling_mov_3", "away_rolling_mov_3"]


def _shrink(p: np.ndarray, temp: float = 0.8, threshold: float = 0.80) -> np.ndarray:
    """Apply temperature scaling only to predictions above threshold."""
    out = p.copy()
    mask = p > threshold
    if not mask.any():
        return out
    logit = np.log(p[mask] / (1 - p[mask] + 1e-15))
    out[mask] = 1.0 / (1.0 + np.exp(-logit / temp))
    return out


def _era_fold_ll(
    df_all: pd.DataFrame, feat_cols: list[str], elo_prob: np.ndarray, y: np.ndarray,
    shrink: bool = False,
) -> list[float]:
    fold_lls = []
    for train_s, val_s in ROLLING_FOLDS:
        tr = df_all["season"].isin(train_s).values
        va = (df_all["season"] == val_s).values
        x_tr = np.column_stack([elo_prob[tr]] + [df_all.loc[tr, c].values for c in feat_cols])
        x_va = np.column_stack([elo_prob[va]] + [df_all.loc[va, c].values for c in feat_cols])

        # Standard Platt
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(x_tr, y[tr].astype(int))
        proba = pipe.predict_proba(x_va)[:, 1]

        if shrink:
            proba = _shrink(proba)

        fold_lls.append(compute_classification_metrics(y[va], proba)["log_loss"])
    return fold_lls


def _era_holdout_ll(
    df_all: pd.DataFrame, feat_cols: list[str], elo_prob: np.ndarray, y: np.ndarray,
    shrink: bool = False,
) -> float:
    tr = (df_all["season"] < HOLDOUT_SEASON).values
    va = (df_all["season"] == HOLDOUT_SEASON).values
    x_tr = np.column_stack([elo_prob[tr]] + [df_all.loc[tr, c].values for c in feat_cols])
    x_va = np.column_stack([elo_prob[va]] + [df_all.loc[va, c].values for c in feat_cols])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_tr, y[tr].astype(int))
    proba = pipe.predict_proba(x_va)[:, 1]

    if shrink:
        proba = _shrink(proba)
    return float(compute_classification_metrics(y[va], proba)["log_loss"])


def _split_era_fold_ll(
    df_all: pd.DataFrame, feat_cols: list[str], elo_prob: np.ndarray, y: np.ndarray,
    shrink: bool = False,
) -> list[float]:
    """Era-split calibration: separate Platt for W1-4 and W5+."""
    weeks = df_all["week"].values
    fold_lls = []
    for train_s, val_s in ROLLING_FOLDS:
        tr = df_all["season"].isin(train_s).values
        va = (df_all["season"] == val_s).values

        # Fit two Platt models
        tr_early = tr & (weeks <= 4)
        tr_late = tr & (weeks > 4)
        x_tr_early = np.column_stack([elo_prob[tr_early]] + [df_all.loc[tr_early, c].values for c in feat_cols])
        x_tr_late = np.column_stack([elo_prob[tr_late]] + [df_all.loc[tr_late, c].values for c in feat_cols])

        pipe_early = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
        pipe_late = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])

        if tr_early.sum() > 0:
            pipe_early.fit(x_tr_early, y[tr_early].astype(int))
        if tr_late.sum() > 0:
            pipe_late.fit(x_tr_late, y[tr_late].astype(int))

        # Predict using matching model
        va_early = va & (weeks <= 4)
        va_late = va & (weeks > 4)
        proba = np.zeros(va.sum())
        idx_early = np.where(va_early[va])[0]
        idx_late = np.where(va_late[va])[0]

        if len(idx_early) > 0 and tr_early.sum() > 0:
            x_va_early = np.column_stack([elo_prob[va_early]] + [df_all.loc[va_early, c].values for c in feat_cols])
            proba[idx_early] = pipe_early.predict_proba(x_va_early)[:, 1]
        if len(idx_late) > 0 and tr_late.sum() > 0:
            x_va_late = np.column_stack([elo_prob[va_late]] + [df_all.loc[va_late, c].values for c in feat_cols])
            proba[idx_late] = pipe_late.predict_proba(x_va_late)[:, 1]

        if shrink:
            proba = _shrink(proba)

        fold_lls.append(compute_classification_metrics(y[va], proba)["log_loss"])
    return fold_lls


def _split_era_holdout_ll(
    df_all: pd.DataFrame, feat_cols: list[str], elo_prob: np.ndarray, y: np.ndarray,
    shrink: bool = False,
) -> float:
    tr = (df_all["season"] < HOLDOUT_SEASON).values
    va = (df_all["season"] == HOLDOUT_SEASON).values
    weeks = df_all["week"].values
    tr_early = tr & (weeks <= 4)
    tr_late = tr & (weeks > 4)
    x_tr_early = np.column_stack([elo_prob[tr_early]] + [df_all.loc[tr_early, c].values for c in feat_cols])
    x_tr_late = np.column_stack([elo_prob[tr_late]] + [df_all.loc[tr_late, c].values for c in feat_cols])
    pipe_early = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
    pipe_late = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
    if tr_early.sum() > 0:
        pipe_early.fit(x_tr_early, y[tr_early].astype(int))
    if tr_late.sum() > 0:
        pipe_late.fit(x_tr_late, y[tr_late].astype(int))

    va_early = va & (weeks <= 4)
    va_late = va & (weeks > 4)
    proba = np.zeros(va.sum())
    idx_early = np.where(va_early[va])[0]
    idx_late = np.where(va_late[va])[0]
    if len(idx_early) > 0 and tr_early.sum() > 0:
        x_va_early = np.column_stack([elo_prob[va_early]] + [df_all.loc[va_early, c].values for c in feat_cols])
        proba[idx_early] = pipe_early.predict_proba(x_va_early)[:, 1]
    if len(idx_late) > 0 and tr_late.sum() > 0:
        x_va_late = np.column_stack([elo_prob[va_late]] + [df_all.loc[va_late, c].values for c in feat_cols])
        proba[idx_late] = pipe_late.predict_proba(x_va_late)[:, 1]
    if shrink:
        proba = _shrink(proba)
    return float(compute_classification_metrics(y[va], proba)["log_loss"])


def run_calibration_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/calibration_improvements.md",
) -> str:
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS
    )
    df = compute_elo_features(
        df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG, team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy()
    print(f"Rows: {len(df)}")

    elo = df["elo_prob"].values
    y = df[TARGET_COLUMN].astype(float).values
    feats = INCUMBENT_FEATURES

    models = {
        "Standard Platt": ("std", False),
        "Standard + Shrinkage": ("std", True),
        "Era-split Platt": ("era", False),
        "Era-split + Shrinkage": ("era", True),
    }

    results = {}
    for name, (mode, shrink) in models.items():
        if mode == "std":
            folds = _era_fold_ll(df, feats, elo, y, shrink=shrink)
            hold = _era_holdout_ll(df, feats, elo, y, shrink=shrink)
        else:
            folds = _split_era_fold_ll(df, feats, elo, y, shrink=shrink)
            hold = _split_era_holdout_ll(df, feats, elo, y, shrink=shrink)
        avg = float(np.mean(folds))
        results[name] = {"folds": [round(v, 4) for v in folds], "avg_val_ll": round(avg, 4), "holdout_ll": round(hold, 4)}
        print(f"  {name:30s}  val={avg:.4f}  hold={hold:.4f}")

    lines = []
    _w = lines.append
    _w("# Calibration Improvements Experiment")
    _w("")
    _w("*Era-split Platt scaling + high-confidence temperature shrinkage.*")
    _w("")
    _w(f"Test date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    _w("")
    _w("## Setup")
    _w("")
    _w("- **Incumbent features**: qb_changed + rolling_mov_3")
    _w("- **Standard Platt**: single logistic on elo_prob + features")
    _w("- **Era-split Platt**: separate calibrators for W1-4 and W5+")
    _w("- **Shrinkage**: temp scaling (T=0.8) on predictions >0.80")
    _w("- **Rolling folds**: 2021→2022, 2021-2022→2023, 2021-2023→2024")
    _w(f"- **Holdout**: {HOLDOUT_SEASON}")
    _w("")
    _w("## Results")
    _w("")
    _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 | Holdout LL |")
    _w("|-------|-----------|-------|-------|-------|-----------|")
    best_val = min(results.items(), key=lambda x: x[1]["avg_val_ll"])
    for name, r in sorted(results.items(), key=lambda x: x[1]["avg_val_ll"]):
        folds_str = " | ".join(f"{v:.4f}" for v in r["folds"])
        _w(f"| {name} | {r['avg_val_ll']:.4f} | {folds_str} | {r['holdout_ll']:.4f} |")
    _w("")
    _w(f"**Best on validation**: {best_val[0]} ({best_val[1]['avg_val_ll']:.4f})")

    incumbent_name = "Standard Platt"
    inc_val = results.get(incumbent_name, {}).get("avg_val_ll", 1.0)
    inc_hold = results.get(incumbent_name, {}).get("holdout_ll", 1.0)
    _w("")
    _w("## Promotion Check")
    _w("")
    _w(f"To beat: val < {inc_val:.4f} AND hold < {inc_hold:.4f}")
    _w("")
    promoted = []
    for name, r in sorted(results.items(), key=lambda x: x[1]["avg_val_ll"]):
        if name == incumbent_name:
            continue
        bv = r["avg_val_ll"] < inc_val
        bh = r["holdout_ll"] < inc_hold
        tag = "**PROMOTED**" if (bv and bh) else "Rejected"
        _w(f"| {name} | val {r['avg_val_ll']:.4f} ({'✓' if bv else '✗'}) | hold {r['holdout_ll']:.4f} ({'✓' if bh else '✗'}) | {tag} |")
        if bv and bh:
            promoted.append(name)
    _w("")
    _w("---")
    _w(f"*Report generated by `sportslab calibration-improvements`. Incumbent holdout: {INCUMBENT_HOLDOUT_LL}.*")
    _w("")

    rp = Path(report_path)
    rp.write_text("\n".join(lines))
    print(f"\nReport: {rp}")
    return str(rp)
