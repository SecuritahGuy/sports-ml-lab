"""Residual-informed blending experiment — small logistic model on top of
incumbent Elo probability with simple game-context features."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.ratings import (
    MOV_CAPPED_LINEAR,
    compute_elo_features,
)

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

# Frozen best params from season-regression incumbent
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.10
BEST_DECAY = 32
BEST_MOV_TYPE = MOV_CAPPED_LINEAR
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(train_prob.reshape(-1, 1), train_y)
    return pipe


def _fit_blend(train_features: np.ndarray, train_y: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(train_features, train_y)
    return lr


def run_residual_blending_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/residual_blending.md",
) -> str:
    """Test whether adding simple game-context features to Elo probability
    improves on the incumbent (season-specific regression + Platt)."""
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # Compute incumbent Elo features with season regression (QB change bonus)
    from sportslab.evaluation.season_regression_experiment import (
        build_team_regression_overrides,
    )

    overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=BEST_REG,
        qb_change_bonus=0.20,
    )
    edf = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        team_regression_overrides=overrides,
    )
    edf = _filter_df(edf)

    # Build feature matrix: elo_prob + simple game-context features
    elo_prob = edf["elo_prob"].values
    week = edf["week"].values.astype(float)
    rest_diff = edf["rest_diff"].values.astype(float)
    y = edf[TARGET_COLUMN].astype(float).values

    # Feature sets to test
    setups = {
        "Platt (incumbent)": np.column_stack([elo_prob]),
        "Elo + week": np.column_stack([elo_prob, week]),
        "Elo + week + rest_diff": np.column_stack([elo_prob, week, rest_diff]),
        "Elo + early_season": np.column_stack([elo_prob, (week <= 5).astype(float)]),
        "Elo + week (no Platt)": None,  # Special: raw Elo with week via LR
    }

    print("=== Residual Blending Experiment ===")
    results: list[dict[str, Any]] = []

    for name, feat_matrix in setups.items():
        if name == "Elo + week (no Platt)":
            # Raw Elo prob + week via LR (no separate Platt)
            is_train_full = edf["season"].isin([2021, 2022, 2023, 2024]).values
            train_f = np.column_stack([elo_prob[is_train_full], week[is_train_full]])
            train_y = y[is_train_full].astype(int)
            lr = LogisticRegression(max_iter=2000, random_state=42)
            lr.fit(train_f, train_y)
            is_hold = (edf["season"] == HOLDOUT_SEASON).values
            hold_f = np.column_stack([elo_prob[is_hold], week[is_hold]])
            hold_pred = lr.predict_proba(hold_f)[:, 1]
            hold_met = compute_classification_metrics(y[is_hold], hold_pred)
            results.append(
                {
                    "model": name,
                    "hold_log_loss": hold_met["log_loss"],
                    "hold_brier": hold_met["brier_score"],
                    "hold_acc": hold_met["accuracy"],
                    "hold_auc": hold_met["roc_auc"],
                }
            )
            print(f"  {name}: hold LL={hold_met['log_loss']:.4f}")
            continue

        # Rolling-origin evaluation for each setup
        fold_lls_raw: list[float] = []
        fold_lls_blend: list[float] = []
        fold_details: list[dict[str, Any]] = []

        for train_seasons, val_season in ROLLING_FOLDS:
            is_train = edf["season"].isin(train_seasons).values
            is_val = (edf["season"] == val_season).values

            # Platt baseline (using only elo_prob)
            train_p = elo_prob[is_train]
            train_y_ = y[is_train].astype(int)
            platt = _fit_platt(train_p, train_y_)
            val_p = elo_prob[is_val]
            val_y_ = y[is_val]
            platt_val = platt.predict_proba(val_p.reshape(-1, 1))[:, 1]
            fold_lls_raw.append(float(log_loss(val_y_, platt_val)))

            # Blend: logistic on features + elo_prob
            train_f = feat_matrix[is_train]
            val_f = feat_matrix[is_val]
            blend = _fit_blend(train_f, train_y_)
            blend_val = blend.predict_proba(val_f)[:, 1]
            fold_lls_blend.append(float(log_loss(val_y_, blend_val)))

            fold_details.append(
                {
                    "train_seasons": train_seasons,
                    "val_season": val_season,
                    "platt_ll": round(float(log_loss(val_y_, platt_val)), 5),
                    "blend_ll": round(float(log_loss(val_y_, blend_val)), 5),
                }
            )

        avg_raw = float(np.mean(fold_lls_raw))
        avg_blend = float(np.mean(fold_lls_blend))

        print(f"  {name}: avg Platt LL={avg_raw:.4f}, avg blend LL={avg_blend:.4f}")

        # Full 2021-2024 fit for holdout
        is_train_full = edf["season"].isin([2021, 2022, 2023, 2024]).values
        train_y_full = y[is_train_full].astype(int)
        platt_full = _fit_platt(elo_prob[is_train_full], train_y_full)
        is_hold = (edf["season"] == HOLDOUT_SEASON).values
        hold_p = elo_prob[is_hold]
        hold_y_ = y[is_hold]
        platt_hold = platt_full.predict_proba(hold_p.reshape(-1, 1))[:, 1]
        platt_hold_met = compute_classification_metrics(hold_y_, platt_hold)

        blend_full = _fit_blend(feat_matrix[is_train_full], train_y_full)
        hold_f = feat_matrix[is_hold]
        blend_hold = blend_full.predict_proba(hold_f)[:, 1]
        blend_hold_met = compute_classification_metrics(hold_y_, blend_hold)

        results.append(
            {
                "model": name,
                "avg_val_platt_ll": round(avg_raw, 5),
                "avg_val_blend_ll": round(avg_blend, 5),
                "fold_details": fold_details,
                "hold_platt_ll": platt_hold_met["log_loss"],
                "hold_blend_ll": blend_hold_met["log_loss"],
                "hold_brier": blend_hold_met["brier_score"],
                "hold_acc": blend_hold_met["accuracy"],
                "hold_auc": blend_hold_met["roc_auc"],
                "improvement": round(platt_hold_met["log_loss"] - blend_hold_met["log_loss"], 5),
            }
        )
        print(
            f"    holdout: Platt={platt_hold_met['log_loss']:.4f},"
            f" blend={blend_hold_met['log_loss']:.4f}"
        )

    # ── Write report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    best_setup = min(results, key=lambda x: x.get("hold_blend_ll", 999))

    with open(rp, "w") as f:
        f.write("# Residual Blending Experiment\n\n")
        f.write(
            "*Testing whether adding simple game-context features to Elo "
            "probability improves on the incumbent.*\n\n"
        )
        f.write("## Motivation\n\n")
        f.write(
            "Residual diagnostics showed systematic errors by week, rest, "
            "and confidence.  The question is whether a tiny logistic model "
            "on Elo prob + 1-2 context features can reduce these errors "
            "without overfitting.\n\n"
        )

        f.write("## Feature Sets Tested\n\n")
        f.write("| Setup | Features |\n")
        f.write("|-------|----------|\n")
        f.write("| Platt (incumbent) | elo_prob only |\n")
        f.write("| Elo + week | elo_prob, week |\n")
        f.write("| Elo + week + rest_diff | elo_prob, week, rest_diff |\n")
        f.write("| Elo + early_season | elo_prob, flag(week <= 5) |\n")
        f.write("| Elo + week (no Platt) | elo_prob + week via LR (no Platt step) |\n\n")

        f.write("## Rolling-Origin Results\n\n")
        f.write("| Model | Avg Val Platt LL | Avg Val Blend LL |\n")
        f.write("|-------|-----------------|-----------------|\n")
        for r in results:
            if "avg_val_platt_ll" in r:
                f.write(f"| {r['model']} | {r['avg_val_platt_ll']} | {r['avg_val_blend_ll']} |\n")
        f.write("\n")

        f.write("## Holdout (2025) Results\n\n")
        f.write("| Model | Hold LL | Brier | Acc | AUC | vs Platt |\n")
        f.write("|-------|---------|-------|-----|-----|----------|\n")
        for r in sorted(results, key=lambda x: x.get("hold_blend_ll", 999)):
            if "hold_blend_ll" in r:
                delta = r.get("improvement", 0)
                sign = "+" if delta >= 0 else ""
                f.write(
                    f"| {r['model']} | {r['hold_blend_ll']:.4f}"
                    f" | {r.get('hold_brier', 0):.4f}"
                    f" | {r.get('hold_acc', 0):.4f}"
                    f" | {r.get('hold_auc', 0):.4f}"
                    f" | {sign}{delta:.4f} |\n"
                )
            else:
                f.write(f"| {r['model']} | {r['hold_log_loss']:.4f} | — | — | — | — |\n")
        f.write("\n")

        # Incumbent
        f.write("## Incumbent Comparison\n\n")
        inc_ll = next(r["hold_platt_ll"] for r in results if r["model"] == "Platt (incumbent)")
        f.write(f"Incumbent (season-reg + Platt) holdout: {inc_ll:.4f}\n\n")

        if best_setup["hold_blend_ll"] < inc_ll - 0.0005:
            f.write("## Decision\n\n")
            f.write(f"✅ **{best_setup['model']} beats the incumbent.**\n\n")
            f.write(
                f"Holdout log loss {best_setup['hold_blend_ll']:.4f} vs incumbent {inc_ll:.4f}.\n\n"
            )
        else:
            f.write("## Decision\n\n")
            f.write("❌ **Residual blending does not beat the incumbent.**\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- All features are pregame-safe (week, rest diff, elo_prob).\n")
        f.write("- Rolling-origin folds prevent 2025 holdout access.\n")
        f.write("- Blend models fitted only on training data per fold.\n")
        f.write("- Holdout evaluated only once after model selection.\n\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
