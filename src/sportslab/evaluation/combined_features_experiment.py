"""Combined feature experiment — tests qb_changed, rolling_mov_3, and coach tenure."""

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
from sportslab.features.coach import COACH_FEATURE_COLUMNS, compute_coach_features
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2


def _run_rolling_ll(df_all, feat_cols, elo_prob, y):
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


def run_combined_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/combined_features.md",
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
    df = compute_coach_features(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN] & ~df[NEUTRAL_COLUMN]].copy()
    print(f"Rows: {len(df)}")

    elo = df["elo_prob"].values
    y = df[TARGET_COLUMN].astype(float).values

    models = {
        "Platt": [],
        "Platt + qb_changed": ["home_qb_changed", "away_qb_changed"],
        "Platt + rolling_mov_3": ["home_rolling_mov_3", "away_rolling_mov_3"],
        "Platt + qb_changed + mov3": [
            "home_qb_changed",
            "away_qb_changed",
            "home_rolling_mov_3",
            "away_rolling_mov_3",
        ],
        "Platt + coach_win_pct": ["home_coach_win_pct", "away_coach_win_pct"],
        "Platt + coach_tenure": ["home_coach_tenure", "away_coach_tenure"],
        "Platt + all_coach": COACH_FEATURE_COLUMNS,
        "Platt + coach + qb": COACH_FEATURE_COLUMNS + ["home_qb_changed", "away_qb_changed"],
    }

    results = {}
    for name, cols in models.items():
        cols_avail = [c for c in cols if c in df.columns]
        fold_lls = _run_rolling_ll(df, cols_avail, elo, y)
        results[name] = {"fold_lls": fold_lls, "val_ll": float(np.mean(fold_lls))}
        print(f"  {name}: {results[name]['val_ll']:.4f}")

    # Holdout
    is_hold = df["season"] == HOLDOUT_SEASON
    hold_y = y[is_hold]
    hold_elo = elo[is_hold]
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    train_y = y[is_train].astype(int)
    train_elo = elo[is_train]

    hold_results = {}
    for name, cols in models.items():
        cols_avail = [c for c in cols if c in df.columns]
        x_tr = (
            np.column_stack([train_elo] + [df.loc[is_train, c].values for c in cols_avail])
            if cols_avail
            else train_elo.reshape(-1, 1)
        )
        x_ho = (
            np.column_stack([hold_elo] + [df.loc[is_hold, c].values for c in cols_avail])
            if cols_avail
            else hold_elo.reshape(-1, 1)
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(x_tr, train_y)
        proba = pipe.predict_proba(x_ho)[:, 1]
        m = compute_classification_metrics(hold_y, proba)
        hold_results[name] = m

    print("\n=== 2025 Holdout ===")
    for name in models:
        print(f"  {name}: {hold_results[name]['log_loss']:.4f}")

    # Report
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        f.write("# Combined Feature Experiment\n\n")
        f.write(
            "## Validation\n\n"
            "| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n"
            "|-------|-----------|-------|-------|-------|\n"
        )
        for name in models:
            r = results[name]
            f.write(
                f"| {name} | {r['val_ll']:.4f}"
                f" | {r['fold_lls'][0]:.4f}"
                f" | {r['fold_lls'][1]:.4f}"
                f" | {r['fold_lls'][2]:.4f} |\n"
            )
        f.write(
            "\n## Holdout\n\n"
            "| Model | Hold LL | Brier | AUC | Acc |\n"
            "|-------|---------|-------|------|\n"
        )
        for name in models:
            h = hold_results[name]
            f.write(
                f"| {name} | {h['log_loss']:.4f}"
                f" | {h['brier_score']:.4f}"
                f" | {h['roc_auc']:.4f}"
                f" | {h['accuracy']:.4f} |\n"
            )
        f.write("\n## Decision\n\n")
        best_val_name = min(models, key=lambda n: results[n]["val_ll"])
        best_val_ll = results[best_val_name]["val_ll"]
        platt_ll = results["Platt"]["val_ll"]
        inc_hold = hold_results["Platt"]["log_loss"]
        f.write(f"Best on validation: {best_val_name} ({best_val_ll:.4f}).\n")
        f.write(f"Incumbent holdout: {inc_hold:.4f}\n")
        beat_both = [
            n
            for n in models
            if n != "Platt"
            and results[n]["val_ll"] < platt_ll
            and hold_results[n]["log_loss"] < inc_hold
        ]
        if beat_both:
            f.write(f"**Promoted: {', '.join(beat_both)}**\n")
        else:
            f.write("**No model beats incumbent on both val and holdout.**\n")

    print(f"\nReport: {rp}")
    return str(rp)
