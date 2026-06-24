"""QB injury flag experiment — single feature test.

Tests whether a single binary flag for 'home starting QB ruled OUT'
improves on the O/D Elo+Platt incumbent (holdout LL 0.6258).

Motivation: residual diagnostics showed QB change is the #1 failure
mode. The injury features experiment showed QB-out subset (n=28) had
holdout LL=0.5506 with Elo+Injury. But the full 19-feature set added
noise. A single binary flag captures this signal with minimal overfit.
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
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, NEUTRAL_COLUMN, TARGET_COLUMN
from sportslab.features.injuries import compute_injury_features
from sportslab.features.ratings import compute_od_elo_features

# Frozen incumbent params
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

# Single binary feature we're testing
QB_OUT_FEATURE = "home_injuries_qb_out"


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    platt = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    platt.fit(train_prob.reshape(-1, 1), train_y)
    return platt


def _logistic_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=42)


def run_qb_injury_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/qb_injury_flag.md",
    cache_dir: str = "data/interim/nfl",
) -> str:
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)
    print(f"Loaded {len(df_raw)} rows from {fp}")

    # Build team regression overrides
    team_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )

    # Compute O/D Elo
    print("\n=== Computing O/D Elo ===")
    df_elo = compute_od_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        k_off=BEST_K_OFF,
        k_def=BEST_K_DEF,
        team_regression_overrides=team_overrides,
    )

    # Compute injury features (we only use the QB out columns)
    print("\n=== Computing injury features ===")
    df_inj = compute_injury_features(df_elo, cache_dir=cache_dir)

    df_all = _filter_df(df_inj)
    print(f"  Total model-eligible rows: {len(df_all)}")

    y = df_all[TARGET_COLUMN].astype(float).values
    qb_out = df_all[QB_OUT_FEATURE].values.astype(float)

    n_qb_out = int(qb_out.sum())
    print(f"  Home QB OUT games: {n_qb_out}/{len(df_all)} ({n_qb_out / len(df_all) * 100:.1f}%)")

    # Separate holdout
    hold_mask = df_all["season"] == HOLDOUT_SEASON
    df_hold = df_all[hold_mask].copy().reset_index(drop=True)
    df_train_val = df_all[~hold_mask].copy().reset_index(drop=True)
    print(f"\n  Train/val rows: {len(df_train_val)}")
    print(f"  Holdout rows:   {len(df_hold)}")

    y_hold = df_hold[TARGET_COLUMN].astype(float).values
    elo_prob_hold = df_hold["elo_prob"].values.astype(float)
    qb_out_hold = df_hold[QB_OUT_FEATURE].values.astype(float)
    n_qb_out_hold = int(qb_out_hold.sum())
    print(f"  Holdout home QB OUT games: {n_qb_out_hold}/{len(df_hold)}")

    # === Rolling-origin validation ===
    fold_results: dict[str, list[float]] = {
        "platt": [],
        "platt_qb_out": [],
        "qb_out_only": [],
    }

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}: train {train_seasons} → val {val_season}")
        print(f"{'=' * 60}")

        train_mask = df_train_val["season"].isin(train_seasons).values
        val_mask = (df_train_val["season"] == val_season).values

        df_train = df_train_val[train_mask].reset_index(drop=True)
        df_val = df_train_val[val_mask].reset_index(drop=True)

        y_train = df_train[TARGET_COLUMN].astype(float).values
        y_val = df_val[TARGET_COLUMN].astype(float).values
        elo_train = df_train["elo_prob"].values.astype(float)
        elo_val = df_val["elo_prob"].values.astype(float)
        qb_out_train = df_train[QB_OUT_FEATURE].values.astype(float)
        qb_out_val = df_val[QB_OUT_FEATURE].values.astype(float)

        print(f"  Train: {len(df_train)} rows, Val: {len(df_val)} rows")
        print(f"  Home QB OUT in val: {int(qb_out_val.sum())}")

        # === Platt (incumbent) ===
        platt = _fit_platt(elo_train, y_train)
        platt_val_prob = platt.predict_proba(elo_val.reshape(-1, 1))[:, 1]
        platt_ll = compute_classification_metrics(y_val, platt_val_prob)["log_loss"]
        fold_results["platt"].append(platt_ll)
        print(f"  Platt (incumbent) val LL: {platt_ll:.4f}")

        # === Platt + QB OUT flag (logistic regression on 2 features) ===
        x_train = np.column_stack([elo_train, qb_out_train])
        x_val = np.column_stack([elo_val, qb_out_val])
        model = _logistic_model()
        model.fit(x_train, y_train)
        prob_val = model.predict_proba(x_val)[:, 1]
        platt_qb_ll = compute_classification_metrics(y_val, prob_val)["log_loss"]
        fold_results["platt_qb_out"].append(platt_qb_ll)
        print(f"  Platt + QB OUT val LL: {platt_qb_ll:.4f}")

        # === QB OUT only ===
        qb_only_model = _logistic_model()
        qb_only_model.fit(qb_out_train.reshape(-1, 1), y_train)
        qb_only_prob = qb_only_model.predict_proba(qb_out_val.reshape(-1, 1))[:, 1]
        qb_only_ll = compute_classification_metrics(y_val, qb_only_prob)["log_loss"]
        fold_results["qb_out_only"].append(qb_only_ll)
        print(f"  QB OUT only val LL: {qb_only_ll:.4f}")

    # === Average validation ===
    print(f"\n{'=' * 60}")
    print("Rolling-Origin Validation Summary")
    print(f"{'=' * 60}")
    avgs = {}
    for label in ["platt", "platt_qb_out", "qb_out_only"]:
        avg = np.mean(fold_results[label])
        avgs[label] = avg
        print(f"  {label}: avg val LL = {avg:.4f}")

    # === One-shot 2025 holdout ===
    print(f"\n{'=' * 60}")
    print("2025 Holdout Evaluation")
    print(f"{'=' * 60}")

    hold_metrics: dict[str, dict] = {}

    # Retrain Platt + QB OUT on full 2021-2024
    x_all = np.column_stack(
        [
            df_train_val["elo_prob"].values.astype(float),
            df_train_val[QB_OUT_FEATURE].values.astype(float),
        ]
    )
    y_all = df_train_val[TARGET_COLUMN].astype(float).values
    final_model = _logistic_model()
    final_model.fit(x_all, y_all)
    x_hold = np.column_stack([elo_prob_hold, qb_out_hold])
    final_hold_prob = final_model.predict_proba(x_hold)[:, 1]
    hold_metrics["platt_qb_out"] = compute_classification_metrics(y_hold, final_hold_prob)
    print(f"  Platt + QB OUT: holdout LL = {hold_metrics['platt_qb_out']['log_loss']:.4f}")

    # Platt incumbent (re-fit on train only — all non-holdout)
    train_elo_all = df_all["elo_prob"].values[~hold_mask]
    train_y_all = y[~hold_mask]
    platt_final = _fit_platt(train_elo_all, train_y_all)
    platt_hold_final = platt_final.predict_proba(elo_prob_hold.reshape(-1, 1))[:, 1]
    hold_metrics["platt"] = compute_classification_metrics(y_hold, platt_hold_final)
    print(f"  Platt (incumbent): holdout LL = {hold_metrics['platt']['log_loss']:.4f}")

    # QB OUT only
    qb_final = _logistic_model()
    qb_final.fit(qb_out[~hold_mask].reshape(-1, 1), y[~hold_mask])
    qb_hold_prob = qb_final.predict_proba(qb_out_hold.reshape(-1, 1))[:, 1]
    hold_metrics["qb_out_only"] = compute_classification_metrics(y_hold, qb_hold_prob)
    print(f"  QB OUT only: holdout LL = {hold_metrics['qb_out_only']['log_loss']:.4f}")

    # QB-out subset analysis
    if n_qb_out_hold >= 5:
        qb_out_mask = qb_out_hold > 0
        qb_out_sub_ll = compute_classification_metrics(
            y_hold[qb_out_mask], final_hold_prob[qb_out_mask]
        )["log_loss"]
        qb_healthy_sub_ll = compute_classification_metrics(
            y_hold[~qb_out_mask], final_hold_prob[~qb_out_mask]
        )["log_loss"]
        print(f"\n  QB OUT subset (n={n_qb_out_hold}): LL = {qb_out_sub_ll:.4f}")
        print(
            f"  QB healthy subset (n={len(df_hold) - n_qb_out_hold}): LL = {qb_healthy_sub_ll:.4f}"
        )

    # === Write report ===
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    inc_ll = hold_metrics["platt"]["log_loss"]
    challenger_ll = hold_metrics["platt_qb_out"]["log_loss"]
    winner = "platt" if inc_ll <= challenger_ll else "platt_qb_out"

    with open(rp, "w") as f:
        f.write("# QB Injury Flag Experiment\n\n")
        f.write("*Testing whether a single binary 'home starting QB OUT' flag")
        f.write(" improves on O/D Elo+Platt incumbent.*\n\n")

        f.write("## Motivation\n\n")
        f.write("Residual diagnostics identified QB change as the #1 failure mode ")
        f.write("(LL gap 0.042 vs stable QB). The full 19-feature injury set added ")
        f.write("too much noise, but QB-out subset showed strong signal ")
        f.write("(holdout LL=0.5506, n=28).\n\n")

        f.write("## Method\n\n")
        f.write(f"Single binary feature `{QB_OUT_FEATURE}` = 1 if home team's ")
        f.write("starting QB is ruled OUT on the final injury report.\n\n")
        f.write("Logistic regression on [elo_prob, qb_out_flag].\n\n")
        f.write("Rolling-origin 3-fold validation, one-shot 2025 holdout.\n\n")

        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|-----------|-------|-------|-------|\n")
        for label, name in [
            ("platt", "Platt (incumbent)"),
            ("platt_qb_out", "Platt + QB OUT"),
            ("qb_out_only", "QB OUT only"),
        ]:
            f.write(f"| {name} | {avgs[label]:.4f}")
            for v in fold_results[label]:
                f.write(f" | {v:.4f}")
            f.write(" |\n")
        f.write("\n")

        f.write("## 2025 Holdout\n\n")
        f.write("| Model | Hold LL | Brier | AUC | Acc |\n")
        f.write("|-------|---------|-------|-----|-----|\n")
        for label, name in [
            ("platt", "Platt (incumbent)"),
            ("platt_qb_out", "Platt + QB OUT"),
            ("qb_out_only", "QB OUT only"),
        ]:
            hm = hold_metrics[label]
            f.write(f"| {name} | {hm['log_loss']:.4f}")
            f.write(f" | {hm['brier_score']:.4f}")
            f.write(f" | {hm['roc_auc']:.4f}")
            f.write(f" | {hm['accuracy']:.4f} |\n")
        f.write("\n")

        if n_qb_out_hold >= 5:
            f.write("### QB-Out Subset (2025 Holdout)\n\n")
            f.write("| Subset | N | LL |\n")
            f.write("|--------|---|----|\n")
            f.write(f"| QB OUT (home) | {n_qb_out_hold} | {qb_out_sub_ll:.4f} |\n")
            n_healthy = len(df_hold) - n_qb_out_hold
            f.write(f"| QB healthy | {n_healthy} | {qb_healthy_sub_ll:.4f} |\n\n")

        if winner == "platt_qb_out":
            f.write("**Platt + QB OUT beats the incumbent.**")
            f.write(" A single binary QB injury flag improves prediction.\n")
        else:
            f.write("**QB OUT flag rejected.** The single binary flag does not ")
            f.write("improve on the incumbent.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
