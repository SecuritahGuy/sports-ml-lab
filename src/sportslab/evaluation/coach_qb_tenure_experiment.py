"""Coach-QB tenure experiment — tests whether QB-coach familiarity adds signal."""

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
from sportslab.features.coach_qb_tenure import (
    COACH_QB_TENURE_COLUMNS,
    compute_coach_qb_tenure_features,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2
INCUMBENT_HOLDOUT_LL = 0.6262
INCUMBENT_FEATURES = ["home_qb_changed", "away_qb_changed", "home_rolling_mov_3", "away_rolling_mov_3"]


def _run_rolling_ll(
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
        fold_lls.append(compute_classification_metrics(y[va], pipe.predict_proba(x_va)[:, 1])["log_loss"])

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


def run_coach_qb_tenure_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/coach_qb_tenure.md",
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
    df = compute_coach_qb_tenure_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy()
    print(f"Rows: {len(df)}")

    elo = df["elo_prob"].values
    y = df[TARGET_COLUMN].astype(float).values

    models = {
        "Platt (incumbent)": INCUMBENT_FEATURES,
        "Incumbent + tenure": INCUMBENT_FEATURES + COACH_QB_TENURE_COLUMNS,
        "Tenure only": COACH_QB_TENURE_COLUMNS,
    }

    results = {}
    for name, feats in models.items():
        folds, hold = _run_rolling_ll(df, feats, elo, y)
        avg = float(np.mean(folds))
        results[name] = {"folds": [round(v, 4) for v in folds], "avg_val_ll": round(avg, 4), "holdout_ll": round(hold, 4)}
        print(f"  {name:30s}  val={avg:.4f}  hold={hold:.4f}")

    lines = []
    _w = lines.append
    _w("# Coach-QB Tenure Experiment")
    _w("")
    _w("*Tests whether games-together between QB and coach adds predictive signal.*")
    _w("")
    _w(f"Test date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    _w("")
    _w("## Results")
    _w("")
    _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 | Holdout LL |")
    _w("|-------|-----------|-------|-------|-------|-----------|")
    for name, r in sorted(results.items(), key=lambda x: x[1]["avg_val_ll"]):
        folds_str = " | ".join(f"{v:.4f}" for v in r["folds"])
        _w(f"| {name} | {r['avg_val_ll']:.4f} | {folds_str} | {r['holdout_ll']:.4f} |")
    _w("")

    inc_name = "Platt (incumbent)"
    inc_val = results[inc_name]["avg_val_ll"]
    inc_hold = results[inc_name]["holdout_ll"]
    _w("## Promotion Check")
    _w("")
    bv = results["Incumbent + tenure"]["avg_val_ll"] < inc_val
    bh = results["Incumbent + tenure"]["holdout_ll"] < inc_hold
    tag = "**PROMOTED**" if (bv and bh) else "Rejected"
    _w(f"| Incumbent + tenure | val {results['Incumbent + tenure']['avg_val_ll']:.4f} ({'✓' if bv else '✗'}) | hold {results['Incumbent + tenure']['holdout_ll']:.4f} ({'✓' if bh else '✗'}) | {tag} |")
    _w("")

    _w("---")
    _w(f"*Report generated by `sportslab coach-qb-tenure`. Incumbent holdout: {INCUMBENT_HOLDOUT_LL}.*")

    rp = Path(report_path)
    rp.write_text("\n".join(lines))
    print(f"\nReport: {rp}")
    return str(rp)
