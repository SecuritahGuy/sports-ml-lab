"""Injury report feature experiment — rolling-origin vs O/D Elo+Platt incumbent.

Tests whether pregame injury report features (players ruled OUT, by position
group, by designation severity) improve prediction quality, especially around
the known QB-change failure mode.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.injuries import INJURY_FEATURE_COLUMNS, compute_injury_features
from sportslab.features.ratings import compute_od_elo_features

# Frozen incumbent O/D Elo params
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0
BEST_K_OFF = 52
BEST_K_DEF = 20


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    platt = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    platt.fit(train_prob.reshape(-1, 1), train_y)
    return platt


def _logistic_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def run_injury_features_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/injury_features.md",
    cache_dir: str = "data/interim/nfl",
) -> str:
    """Run injury feature experiment with rolling-origin validation.

    1. Compute O/D Elo with incumbent params.
    2. Compute injury features from nflreadpy.
    3. Rolling-origin evaluation for each challenger.
    4. One-time 2025 holdout evaluation.
    5. Report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)
    print(f"Loaded {len(df_raw)} rows from {fp}")

    # Build team regression overrides (no holdout leak)
    team_overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
    )

    # Compute O/D Elo on all data
    print("\n=== Computing O/D Elo ===")
    df_elo = compute_od_elo_features(
        df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE, mov_scale=BEST_MOV_SCALE, mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        k_off=BEST_K_OFF, k_def=BEST_K_DEF,
        team_regression_overrides=team_overrides,
    )

    # Compute injury features
    print("\n=== Computing injury features ===")
    df_inj = compute_injury_features(df_elo, cache_dir=cache_dir)

    # Merge and filter
    df_all = _filter_df(df_inj)
    print(f"  Total model-eligible rows: {len(df_all)}")

    # Available injury features
    avail_inj = [c for c in INJURY_FEATURE_COLUMNS if c in df_all.columns]
    print(f"  Available injury features ({len(avail_inj)}): {avail_inj}")

    # Separate holdout
    hold_mask = df_all["season"] == HOLDOUT_SEASON
    df_hold = df_all[hold_mask].copy().reset_index(drop=True)
    df_train_val = df_all[~hold_mask].copy().reset_index(drop=True)
    print(f"\n  Train/val rows: {len(df_train_val)}")
    print(f"  Holdout rows:   {len(df_hold)}")

    y_hold = df_hold[TARGET_COLUMN].astype(float).values
    elo_prob_hold = df_hold["elo_prob"].values.astype(float)

    # === Rolling-origin validation ===
    fold_results: dict[str, list[float]] = {
        "platt": [],
        "elo_inj": [],
        "inj_only": [],
    }
    fold_models: dict[str, list] = {
        "platt": [],
        "elo_inj": [],
        "inj_only": [],
    }

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}: train {train_seasons} → val {val_season}")
        print(f"{'='*60}")

        train_mask = df_train_val["season"].isin(train_seasons)
        val_mask = df_train_val["season"] == val_season

        df_train = df_train_val[train_mask].reset_index(drop=True)
        df_val = df_train_val[val_mask].reset_index(drop=True)

        y_train = df_train[TARGET_COLUMN].astype(float).values
        y_val = df_val[TARGET_COLUMN].astype(float).values
        elo_train = df_train["elo_prob"].values.astype(float)
        elo_val = df_val["elo_prob"].values.astype(float)

        print(f"  Train: {len(df_train)} rows, Val: {len(df_val)} rows")

        # === Platt (incumbent) ===
        platt = _fit_platt(elo_train, y_train)
        platt_val_prob = platt.predict_proba(elo_val.reshape(-1, 1))[:, 1]
        platt_val_ll = compute_classification_metrics(y_val, platt_val_prob)["log_loss"]
        fold_results["platt"].append(platt_val_ll)
        fold_models["platt"].append(platt)
        print(f"  Platt (incumbent) val LL: {platt_val_ll:.4f}")

        # === Elo + Injury features (logistic) ===
        x_elo_inj_train = np.column_stack([
            elo_train,
            df_train[avail_inj].values,
        ])
        x_elo_inj_val = np.column_stack([
            elo_val,
            df_val[avail_inj].values,
        ])
        elo_inj_model = _logistic_model()
        elo_inj_model.fit(x_elo_inj_train, y_train)
        elo_inj_val_prob = elo_inj_model.predict_proba(x_elo_inj_val)[:, 1]
        elo_inj_val_ll = compute_classification_metrics(y_val, elo_inj_val_prob)["log_loss"]
        fold_results["elo_inj"].append(elo_inj_val_ll)
        fold_models["elo_inj"].append(elo_inj_model)
        print(f"  Elo + Injury val LL: {elo_inj_val_ll:.4f}")

        # === Injury only (logistic) ===
        x_inj_train = df_train[avail_inj].values
        x_inj_val = df_val[avail_inj].values
        inj_model = _logistic_model()
        inj_model.fit(x_inj_train, y_train)
        inj_val_prob = inj_model.predict_proba(x_inj_val)[:, 1]
        inj_val_ll = compute_classification_metrics(y_val, inj_val_prob)["log_loss"]
        fold_results["inj_only"].append(inj_val_ll)
        fold_models["inj_only"].append(inj_model)
        print(f"  Injury only val LL: {inj_val_ll:.4f}")

    # === Average validation results ===
    print(f"\n{'='*60}")
    print("Rolling-Origin Validation Summary")
    print(f"{'='*60}")
    for label in ["platt", "elo_inj", "inj_only"]:
        avg = np.mean(fold_results[label])
        print(f"  {label}: avg val LL = {avg:.4f}")

    # === One-shot 2025 holdout ===
    print(f"\n{'='*60}")
    print("2025 Holdout Evaluation")
    print(f"{'='*60}")

    hold_metrics: dict[str, dict] = {}

    # Platt incumbent
    platt_hold_prob = fold_models["platt"][-1].predict_proba(
        elo_prob_hold.reshape(-1, 1)
    )[:, 1]
    hold_metrics["platt"] = compute_classification_metrics(y_hold, platt_hold_prob)
    print(f"  Platt (incumbent): holdout LL = {hold_metrics['platt']['log_loss']:.4f}")

    # Elo + Injury (retrain on all 2021-2024)
    x_all_elo_inj = np.column_stack([
        df_train_val["elo_prob"].values.astype(float),
        df_train_val[avail_inj].values,
    ])
    y_all = df_train_val[TARGET_COLUMN].astype(float).values
    elo_inj_final = _logistic_model()
    elo_inj_final.fit(x_all_elo_inj, y_all)

    x_hold_elo_inj = np.column_stack([
        elo_prob_hold,
        df_hold[avail_inj].values,
    ])
    elo_inj_hold_prob = elo_inj_final.predict_proba(x_hold_elo_inj)[:, 1]
    hold_metrics["elo_inj"] = compute_classification_metrics(y_hold, elo_inj_hold_prob)
    print(f"  Elo + Injury: holdout LL = {hold_metrics['elo_inj']['log_loss']:.4f}")

    # Injury only (retrain)
    x_all_inj = df_train_val[avail_inj].values
    inj_final = _logistic_model()
    inj_final.fit(x_all_inj, y_all)
    inj_hold_prob = inj_final.predict_proba(df_hold[avail_inj].values)[:, 1]
    hold_metrics["inj_only"] = compute_classification_metrics(y_hold, inj_hold_prob)
    print(f"  Injury only: holdout LL = {hold_metrics['inj_only']['log_loss']:.4f}")

    # QB-change subset analysis on holdout (injured QB)
    qb_out_col = "home_injuries_qb_out"
    if qb_out_col in df_hold.columns:
        hold_qb_out = (df_hold[qb_out_col] > 0).values
        if hold_qb_out.sum() >= 2:
            qb_out_ll = compute_classification_metrics(
                y_hold[hold_qb_out], elo_inj_hold_prob[hold_qb_out]
            )["log_loss"]
            qb_healthy_ll = compute_classification_metrics(
                y_hold[~hold_qb_out], elo_inj_hold_prob[~hold_qb_out]
            )["log_loss"]
            print(f"\n  QB Out subset (n={int(hold_qb_out.sum())}): LL = {qb_out_ll:.4f}")
            print(f"  QB healthy subset (n={int((~hold_qb_out).sum())}): LL = {qb_healthy_ll:.4f}")

    # === Write report ===
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Injury Features Experiment\n\n")
        f.write("*Testing whether pregame injury report features"
                " improve on O/D Elo+Platt incumbent.*\n\n")
        f.write("## Method\n\n")
        f.write("Rolling-origin 3-fold validation, one-shot 2025 holdout.\n\n")

        f.write("### Competing Models\n\n")
        f.write("| Model | Description |\n")
        f.write("|------|------------|\n")
        f.write("| **Platt (incumbent)** | O/D Elo (ko52_kd20) + logistic calibration |\n")
        f.write("| **Elo + Injury** | O/D Elo features + injury features + logistic regression |\n")
        f.write("| **Injury only** | Injury features only + logistic regression |\n\n")

        f.write("### Injury Features\n\n")
        for c in avail_inj:
            f.write(f"- {c}\n")
        f.write("\n")

        # Validation table
        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|-----------|-------|-------|-------|\n")
        for label, name in [("platt", "Platt (incumbent)"),
                            ("elo_inj", "Elo + Injury"),
                            ("inj_only", "Injury only")]:
            avg = np.mean(fold_results[label])
            f.write(f"| {name} | {avg:.4f}")
            for v in fold_results[label]:
                f.write(f" | {v:.4f}")
            f.write(" |\n")
        f.write("\n")

        # Holdout table
        f.write("## 2025 Holdout\n\n")
        f.write("| Model | Hold LL | Brier | AUC | Acc |\n")
        f.write("|-------|---------|-------|-----|-----|\n")
        for label, name in [
            ("platt", "Platt (incumbent)"),
            ("elo_inj", "Elo + Injury"),
            ("inj_only", "Injury only"),
        ]:
            if label in hold_metrics:
                hm = hold_metrics[label]
                f.write(f"| {name} | {hm['log_loss']:.4f}")
                f.write(f" | {hm['brier_score']:.4f}")
                f.write(f" | {hm['roc_auc']:.4f}")
                f.write(f" | {hm['accuracy']:.4f}")
                f.write(" |\n")
        f.write("\n")

        # QB-change subset
        if qb_out_col in df_hold.columns:
            hold_qb_out = (df_hold[qb_out_col] > 0).values
            if hold_qb_out.sum() >= 2:
                qb_out_ll = compute_classification_metrics(
                    y_hold[hold_qb_out], elo_inj_hold_prob[hold_qb_out]
                )["log_loss"]
                qb_healthy_ll = compute_classification_metrics(
                    y_hold[~hold_qb_out], elo_inj_hold_prob[~hold_qb_out]
                )["log_loss"]
                f.write("### QB Injury Subset (Elo + Injury)\n\n")
                f.write("| Subset | N | Log Loss |\n")
                f.write("|--------|---|---------|\n")
                f.write(f"| QB Out | {int(hold_qb_out.sum())} | {qb_out_ll:.4f} |\n")
                f.write(f"| QB healthy | {int((~hold_qb_out).sum())} | {qb_healthy_ll:.4f} |\n\n")

        # Decision
        challenger_keys = [k for k in hold_metrics if k != "platt"]
        if challenger_keys:
            best_hold_label = min(challenger_keys, key=lambda k: hold_metrics[k]["log_loss"])
        else:
            best_hold_label = "platt"
        best_hold_ll = hold_metrics[best_hold_label]["log_loss"]
        incumbent_ll = hold_metrics["platt"]["log_loss"]

        if best_hold_ll < incumbent_ll:
            f.write(f"**Winner: {best_hold_label}** (holdout LL {best_hold_ll:.4f}")
            f.write(f" vs incumbent {incumbent_ll:.4f})\n\n")
        else:
            f.write(f"**Incumbent retains champion.** Best challenger {best_hold_label}")
            f.write(f" holdout LL {best_hold_ll:.4f}")
            f.write(f" vs incumbent {incumbent_ll:.4f}\n\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
