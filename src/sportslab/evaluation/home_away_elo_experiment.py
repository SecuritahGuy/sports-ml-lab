"""Test separate home/away Elo ratings — each team has independent home/away rating."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, NEUTRAL_COLUMN, TARGET_COLUMN
from sportslab.features.home_away_elo import compute_home_away_elo
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]


def run_home_away_elo_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/home_away_elo.md",
) -> str:
    df_raw = pd.read_parquet(ft_path)

    std = compute_elo_features(df_raw, k_factor=36, home_advantage=40, preseason_regression=0.1)
    ha = compute_home_away_elo(df_raw, k_factor=36, home_advantage=40, preseason_regression=0.1)
    std_qb = compute_qb_features(std)
    ha_qb = compute_qb_features(ha)

    for df_ in [std_qb, ha_qb]:
        df_.drop(
            columns=[
                c
                for c in df_.columns
                if c.startswith("qb_") and c not in ["home_qb_changed", "away_qb_changed"]
            ],
            inplace=True,
            errors="ignore",
        )

    mask = std_qb[MODEL_ELIGIBLE_COLUMN] & ~std_qb[NEUTRAL_COLUMN]
    std_f = std_qb[mask].copy()
    ha_f = ha_qb[mask].copy()

    print(f"Rows: {len(std_f)}")

    y = std_f[TARGET_COLUMN].astype(float).values

    configs = [
        ("Standard Elo + Platt", std_f["elo_prob"].values, []),
        ("HA Elo + Platt", ha_f["elo_prob"].values, []),
        ("Standard + qb_changed", std_f["elo_prob"].values, ["home_qb_changed", "away_qb_changed"]),
        ("HA + qb_changed", ha_f["elo_prob"].values, ["home_qb_changed", "away_qb_changed"]),
    ]

    results = {}
    for name, elo, feats in configs:
        fold_lls = []
        for tr_s, va_s in ROLLING_FOLDS:
            tr = std_f["season"].isin(tr_s).values
            va = (std_f["season"] == va_s).values
            cols = [std_f.loc[tr, c].values for c in feats] if feats else []
            x_tr = np.column_stack([elo[tr]] + cols)
            x_va = np.column_stack(
                [elo[va]] + [std_f.loc[va, c].values for c in feats] if feats else [elo[va]]
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
        results[name] = {"fold_lls": fold_lls, "val": float(np.mean(fold_lls))}
        print(f"  {name}: {results[name]['val']:.4f}")

    # Holdout
    print("\nHoldout:")
    for name, elo, feats in configs:
        tr = std_f["season"].isin([2021, 2022, 2023, 2024]).values
        va = (std_f["season"] == HOLDOUT_SEASON).values
        cols = [std_f.loc[tr, c].values for c in feats] if feats else []
        x_tr = np.column_stack([elo[tr]] + cols)
        x_va = np.column_stack(
            [elo[va]] + [std_f.loc[va, c].values for c in feats] if feats else [elo[va]]
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(x_tr, y[tr].astype(int))
        proba = pipe.predict_proba(x_va)[:, 1]
        m = compute_classification_metrics(y[va], proba)
        results[name]["hold"] = m["log_loss"]
        print(f"  {name}: {m['log_loss']:.4f}")

    # Report
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        f.write("# Home/Away Elo Experiment\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 | Hold LL |\n")
        f.write("|-------|-----------|-------|-------|-------|--------|\n")
        for name in [c[0] for c in configs]:
            r = results[name]
            f.write(
                f"| {name} | {r['val']:.4f}"
                f" | {r['fold_lls'][0]:.4f}"
                f" | {r['fold_lls'][1]:.4f}"
                f" | {r['fold_lls'][2]:.4f}"
                f" | {r['hold']:.4f} |\n"
            )
        f.write("\n## Decision\n\n")
        best_val = min(results.keys(), key=lambda n: results[n]["val"])
        inc = "Standard Elo + Platt"
        if (
            best_val != inc
            and results[best_val]["val"] < results[inc]["val"]
            and results[best_val].get("hold", 1) < results[inc].get("hold", 1)
        ):
            f.write(f"**{best_val} promoted.**\n")
        else:
            f.write("**No improvement. Standard Elo remains incumbent.**\n")

    print(f"\nReport: {rp}")
    return str(rp)
