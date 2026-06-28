"""QB change magnitude experiment — continuous QB quality dropoff features.

Tests whether rolling EPA-based QB magnitude features improve on the
binary qb_changed flag by capturing how much QB quality drops when
a QB change occurs.
"""

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
from sportslab.features.qb_magnitude import (
    QB_MAGNITUDE_COLUMNS,
    compute_qb_magnitude_features,
)
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2

# Incumbent holdout LL (from combined_features_experiment)
INCUMBENT_HOLDOUT_LL = 0.6262

MAGNITUDE_FEATURES = QB_MAGNITUDE_COLUMNS
INCIDENT_FEATURES = ["home_qb_changed", "away_qb_changed"]
REST_FEATURES = ["home_rolling_mov_3", "away_rolling_mov_3"]


def _run_rolling_ll(df_all: pd.DataFrame, feat_cols: list[str], elo_prob: np.ndarray, y: np.ndarray) -> list[float]:
    fold_lls = []
    for train_s, val_s in ROLLING_FOLDS:
        tr = df_all["season"].isin(train_s).values
        va = (df_all["season"] == val_s).values
        x_tr = (
            np.column_stack([elo_prob[tr]] + [df_all.loc[tr, c].values for c in feat_cols])
            if feat_cols
            else elo_prob[tr].reshape(-1, 1)
        )
        x_va = (
            np.column_stack([elo_prob[va]] + [df_all.loc[va, c].values for c in feat_cols])
            if feat_cols
            else elo_prob[va].reshape(-1, 1)
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(x_tr, y[tr].astype(int))
        proba = pipe.predict_proba(x_va)[:, 1]
        fold_lls.append(compute_classification_metrics(y[va], proba)["log_loss"])
    return fold_lls


def _holdout_ll(df_all: pd.DataFrame, feat_cols: list[str], elo_prob: np.ndarray, y: np.ndarray) -> float:
    tr = (df_all["season"] < HOLDOUT_SEASON).values
    va = (df_all["season"] == HOLDOUT_SEASON).values
    x_tr = (
        np.column_stack([elo_prob[tr]] + [df_all.loc[tr, c].values for c in feat_cols])
        if feat_cols
        else elo_prob[tr].reshape(-1, 1)
    )
    x_va = (
        np.column_stack([elo_prob[va]] + [df_all.loc[va, c].values for c in feat_cols])
        if feat_cols
        else elo_prob[va].reshape(-1, 1)
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(x_tr, y[tr].astype(int))
    proba = pipe.predict_proba(x_va)[:, 1]
    return float(compute_classification_metrics(y[va], proba)["log_loss"])


def run_qb_magnitude_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/qb_magnitude.md",
) -> str:
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS
    )
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)
    df = compute_qb_magnitude_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy()
    print(f"Rows: {len(df)}")

    elo = df["elo_prob"].values
    y = df[TARGET_COLUMN].astype(float).values

    models = {
        "Platt (elo only)": [],
        "Incumbent (qb+mov3)": INCIDENT_FEATURES + REST_FEATURES,
        "Platt + QB magnitude": MAGNITUDE_FEATURES,
        "Incumbent + magnitude": INCIDENT_FEATURES + REST_FEATURES + MAGNITUDE_FEATURES,
        "QB magnitude only": MAGNITUDE_FEATURES,
    }

    results = {}
    for name, feats in models.items():
        folds = _run_rolling_ll(df, feats, elo, y)
        avg = float(np.mean(folds))
        hold = _holdout_ll(df, feats, elo, y)
        results[name] = {"folds": [round(v, 4) for v in folds], "avg_val_ll": round(avg, 4), "holdout_ll": round(hold, 4)}
        print(f"  {name:30s}  val={avg:.4f}  hold={hold:.4f}  folds={[round(v,4) for v in folds]}")

    # Generate report
    lines: list[str] = []
    _w = lines.append

    _w("# QB Change Magnitude Experiment")
    _w("")
    _w("*Continuous QB change magnitude features using rolling passing_epa.*")
    _w("")
    _w(f"Test date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    _w("")

    _w("## Setup")
    _w("")
    _w(f"- **Elo params**: K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}, decay={BEST_DECAY}, qb_bonus={BEST_QB_BONUS}")
    _w(f"- **Incumbent**: Elo prob + qb_changed + rolling_mov_3 + Platt (reported holdout LL {INCUMBENT_HOLDOUT_LL})")
    _w(f"- **Magnitude features ({len(MAGNITUDE_FEATURES)})**: rolling_epa, change_magnitude (abs+signed), epa_diff, missing flags")
    _w("- **Magnitude computation**: 5-game rolling avg of passing_epa per QB, leakage-safe (shift(1))")
    _w("- **Rolling folds**: 2021→2022, 2021-2022→2023, 2021-2023→2024")
    _w(f"- **Holdout**: {HOLDOUT_SEASON} season")
    _w("")

    _w("## Model Comparison")
    _w("")
    _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 | Holdout LL |")
    _w("|-------|-----------|-------|-------|-------|-----------|")
    best_val = min(results.items(), key=lambda x: x[1]["avg_val_ll"])
    best_hold = min(results.items(), key=lambda x: x[1]["holdout_ll"])
    for name, r in sorted(results.items(), key=lambda x: x[1]["avg_val_ll"]):
        folds_str = " | ".join(f"{v:.4f}" for v in r["folds"])
        _w(f"| {name} | {r['avg_val_ll']:.4f} | {folds_str} | {r['holdout_ll']:.4f} |")
    _w("")

    _w(f"**Best on validation**: {best_val[0]} ({best_val[1]['avg_val_ll']:.4f})")
    _w(f"**Best on holdout**: {best_hold[0]} ({best_hold[1]['holdout_ll']:.4f})")
    _w("")

    # Promotion check
    incumbent_name = "Incumbent (qb+mov3)"
    incumbent_val = results.get(incumbent_name, {}).get("avg_val_ll", 1.0)
    incumbent_hold = results.get(incumbent_name, {}).get("holdout_ll", 1.0)

    _w("## Promotion Check")
    _w("")
    _w(f"To be promoted, a challenger must beat the full incumbent (qb_changed + rolling_mov_3) on BOTH validation ({incumbent_val:.4f}) AND holdout ({incumbent_hold:.4f}).")
    _w("")
    promoted = []
    for name, r in sorted(results.items(), key=lambda x: x[1]["avg_val_ll"]):
        if name == incumbent_name:
            continue
        beats_val = r["avg_val_ll"] < incumbent_val
        beats_hold = r["holdout_ll"] < incumbent_hold
        tag = "**PROMOTED**" if (beats_val and beats_hold) else "Rejected"
        _w(f"| {name} | {r['avg_val_ll']:.4f} vs {incumbent_val:.4f} ({'✓' if beats_val else '✗'}) | {r['holdout_ll']:.4f} vs {incumbent_hold:.4f} ({'✓' if beats_hold else '✗'}) | {tag} |")
        if beats_val and beats_hold:
            promoted.append(name)
    _w("")

    if promoted:
        _w(f"### ✅ {', '.join(promoted)} promoted as new football-only incumbent!")
        _w("")
        _w(f"New holdout log loss: {best_hold[1]['holdout_ll']:.4f}")
        _w("")
    else:
        _w("### ❌ No model beats the incumbent on both metrics.")
        _w("")
        best_challenger = sorted(
            [(n, r) for n, r in results.items() if n != incumbent_name],
            key=lambda x: x[1]["avg_val_ll"],
        )
        if best_challenger:
            _w(f"Closest challenger: **{best_challenger[0][0]}** (val {best_challenger[0][1]['avg_val_ll']:.4f}, hold {best_challenger[0][1]['holdout_ll']:.4f})")
        _w("")

    _w("## Analysis")
    _w("")
    _w("- **Magnitude on its own** — how much QB quality-drop signal exists outside Elo + binary qb_changed")
    _w("- **Platt + magnitude** — magnitude replaces binary qb_changed (tests if continuous > binary)")
    _w("- **Incumbent + magnitude** — magnitude added on top of existing feature set")
    _w("")

    # Feature importance
    _w("## Feature Importance (Incumbent + magnitude fit on all 2021-2024)")
    _w("")
    tr = (df["season"] < HOLDOUT_SEASON).values
    all_feats = INCIDENT_FEATURES + REST_FEATURES + MAGNITUDE_FEATURES
    x_all = np.column_stack([elo[tr]] + [df.loc[tr, c].values for c in all_feats])
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(x_all, y[tr].astype(int))
    coefs = pipe.named_steps["lr"].coef_[0]
    feat_names = ["elo_prob"] + all_feats
    importance = sorted(zip(feat_names, coefs), key=lambda x: -abs(x[1]))
    _w("| Feature | Coefficient |")
    _w("|---------|------------|")
    for fname, coef in importance:
        _w(f"| {fname} | {coef:+.4f} |")
    _w("")

    _w("---")
    _w(f"*Report generated by `sportslab qb-magnitude`. Incumbent holdout LL: {INCUMBENT_HOLDOUT_LL}.*")
    _w("")

    rp = Path(report_path)
    rp.write_text("\n".join(lines))
    print(f"\nReport: {rp}")
    return str(rp)
