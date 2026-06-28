"""Adaptive Elo K by week — higher K early season, decays to base K."""

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
BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 40, 0.1, 32, 0.2
INCUMBENT_HOLDOUT_LL = 0.6262
INCUMBENT_FEATURES = ["home_qb_changed", "away_qb_changed", "home_rolling_mov_3", "away_rolling_mov_3"]

# Grid search over base_K and boost
K_GRID = [20, 28, 36, 44, 52]
BOOST_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _run_model(
    df_all: pd.DataFrame, feat_cols: list[str], elo_prob: np.ndarray, y: np.ndarray,
) -> tuple[list[float], float]:
    fold_lls = []
    for train_s, val_s in ROLLING_FOLDS:
        tr = df_all["season"].isin(train_s).values
        va = (df_all["season"] == val_s).values
        x_tr = np.column_stack([elo_prob[tr]] + [df_all.loc[tr, c].values for c in feat_cols])
        x_va = np.column_stack([elo_prob[va]] + [df_all.loc[va, c].values for c in feat_cols])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(x_tr, y[tr].astype(int))
        proba = pipe.predict_proba(x_va)[:, 1]
        fold_lls.append(compute_classification_metrics(y[va], proba)["log_loss"])

    tr = (df_all["season"] < HOLDOUT_SEASON).values
    va = (df_all["season"] == HOLDOUT_SEASON).values
    x_tr = np.column_stack([elo_prob[tr]] + [df_all.loc[tr, c].values for c in feat_cols])
    x_va = np.column_stack([elo_prob[va]] + [df_all.loc[va, c].values for c in feat_cols])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_tr, y[tr].astype(int))
    hold = float(compute_classification_metrics(y[va], pipe.predict_proba(x_va)[:, 1])["log_loss"])
    return fold_lls, hold


def run_adaptive_k_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/adaptive_k.md",
) -> str:
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)
    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS
    )
    feats = INCUMBENT_FEATURES

    # Baseline: compute standard Elo (boost=0) once
    df_base = compute_elo_features(
        df_raw, k_factor=36, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG, team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df_base = compute_qb_features(df_base)
    df_base = compute_situational_features(df_base)
    df_base = df_base[df_base[MODEL_ELIGIBLE_COLUMN] & ~df_base[NEUTRAL_COLUMN]].copy()
    elo_base = df_base["elo_prob"].values
    y = df_base[TARGET_COLUMN].astype(float).values

    # Compute baseline
    print("=== Adaptive K by Week: Grid Search ===")
    base_folds, base_hold = _run_model(df_base, feats, elo_base, y)
    base_avg = float(np.mean(base_folds))
    print(f"  Baseline (K=36, boost=0):  val={base_avg:.4f}  hold={base_hold:.4f}")

    results = {"Baseline K=36": {"folds": [round(v, 4) for v in base_folds], "avg_val_ll": round(base_avg, 4), "holdout_ll": round(base_hold, 4)}}

    for k in K_GRID:
        for boost in BOOST_GRID:
            if boost == 0.0 and k == 36:
                continue  # already computed
            df = compute_elo_features(
                df_raw, k_factor=k, home_advantage=BEST_HFA,
                preseason_regression=BEST_REG, team_regression_overrides=overrides,
                decay_half_life=BEST_DECAY, adaptive_k_boost=boost,
            )
            df = compute_qb_features(df)
            df = compute_situational_features(df)
            df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy()
            elo = df["elo_prob"].values
            folds, hold = _run_model(df, feats, elo, y)
            avg = float(np.mean(folds))
            name = f"K={k}, boost={boost}"
            results[name] = {"folds": [round(v, 4) for v in folds], "avg_val_ll": round(avg, 4), "holdout_ll": round(hold, 4)}
            print(f"  {name:25s}  val={avg:.4f}  hold={hold:.4f}")

    lines = []
    _w = lines.append
    _w("# Adaptive Elo K by Week Experiment")
    _w("")
    _w("*K(week) = base_K * (1 + boost * max(0, 1 - (week-1)/18))*")
    _w("")
    _w(f"Grid: K∈{K_GRID}, boost∈{BOOST_GRID} ({len(K_GRID)*(len(BOOST_GRID))} combos)")
    _w("")
    _w("## Results (top 10 by avg val LL)")
    _w("")
    sorted_r = sorted(results.items(), key=lambda x: x[1]["avg_val_ll"])
    _w("| Params | Avg Val LL | Fold1 | Fold2 | Fold3 | Holdout LL |")
    _w("|--------|-----------|-------|-------|-------|-----------|")
    for name, r in sorted_r[:10]:
        folds_str = " | ".join(f"{v:.4f}" for v in r["folds"])
        _w(f"| {name} | {r['avg_val_ll']:.4f} | {folds_str} | {r['holdout_ll']:.4f} |")
    _w("")

    best = sorted_r[0]
    _w(f"**Best**: {best[0]} (val={best[1]['avg_val_ll']:.4f}, hold={best[1]['holdout_ll']:.4f})")

    inc_name = "Baseline K=36"
    inc_val = results[inc_name]["avg_val_ll"]
    inc_hold = results[inc_name]["holdout_ll"]
    _w("")
    _w("## Promotion Check")
    _w("")
    _w(f"To beat Baseline K=36: val < {inc_val:.4f} AND hold < {inc_hold:.4f}")
    _w("")
    promoted = False
    for name, r in sorted_r:
        if name == inc_name:
            continue
        bv = r["avg_val_ll"] < inc_val
        bh = r["holdout_ll"] < inc_hold
        if bv and bh:
            _w(f"**{name}**: val {r['avg_val_ll']:.4f} ({'✓'}) hold {r['holdout_ll']:.4f} ({'✓'}) → **PROMOTED**")
            promoted = True
        elif bv or bh:
            _w(f"{name}: val {r['avg_val_ll']:.4f} ({'✓' if bv else '✗'}) hold {r['holdout_ll']:.4f} ({'✓' if bh else '✗'}) → Partial")
    if not promoted:
        _w("No adaptive K variant beats baseline on both val and holdout.")
    _w("")

    _w("---")
    _w(f"*Report generated by `sportslab adaptive-k`. Incumbent: {INCUMBENT_HOLDOUT_LL}.*")

    rp = Path(report_path)
    rp.write_text("\n".join(lines))
    print(f"\nReport: {rp}")
    return str(rp)
