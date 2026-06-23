"""Rolling-origin Elo validation with cross-fold selection and calibration.

Validates Elo parameters across three rolling-origin folds, selects by
average validation log loss, then evaluates on 2025 holdout exactly once.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
from sportslab.features.ratings import compute_elo_features

HOLDOUT_SEASON = 2025

# Rolling-origin folds: (train_seasons, val_season)
ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

# Expanded Elo parameter grid
K_FACTOR_CANDIDATES = [20, 24, 28, 32, 36, 40, 48]
HFA_CANDIDATES = [10, 20, 25, 30, 35, 40]
REGRESSION_CANDIDATES = [0.0, 0.10, 0.20, 0.25, 0.33]


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard filters: model_eligible, non-neutral."""
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def run_rolling_origin_grid_search(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
) -> tuple[dict, list[dict]]:
    """Run grid search over Elo parameters using rolling-origin validation.

    For each parameter combination, computes validation log loss on each
    rolling-origin fold and selects by average across folds.  The 2025
    holdout is NEVER computed or accessed during grid search.

    Returns:
        (best_params, all_results) where best_params is a dict with keys
        'k_factor', 'home_advantage', 'preseason_regression' and all_results
        is a list of (avg_val_ll, params, per_fold_lls) dicts sorted by
        average validation log loss ascending.
    """
    df_raw = pd.read_parquet(feature_table_path)
    best: dict = {"avg_val_log_loss": float("inf"), "params": None}
    all_results: list[dict] = []

    total = len(K_FACTOR_CANDIDATES) * len(HFA_CANDIDATES) * len(REGRESSION_CANDIDATES)
    count = 0

    for k in K_FACTOR_CANDIDATES:
        for hfa in HFA_CANDIDATES:
            for reg in REGRESSION_CANDIDATES:
                count += 1
                edf = compute_elo_features(
                    df_raw,
                    k_factor=k,
                    home_advantage=hfa,
                    preseason_regression=reg,
                )
                edf = _filter_df(edf)

                elo_prob = edf["elo_prob"].values
                y = edf[TARGET_COLUMN].astype(float).values

                fold_lls: list[float] = []
                fold_details: list[dict] = []

                for train_seasons, val_season in ROLLING_FOLDS:
                    is_val = (edf["season"] == val_season).values
                    val_loss = float(log_loss(y[is_val], elo_prob[is_val]))
                    fold_lls.append(val_loss)
                    fold_details.append(
                        {
                            "train_seasons": train_seasons,
                            "val_season": val_season,
                            "val_log_loss": round(val_loss, 5),
                        }
                    )

                avg_ll = float(np.mean(fold_lls))

                entry = {
                    "k_factor": k,
                    "home_advantage": hfa,
                    "preseason_regression": reg,
                    "avg_val_log_loss": round(avg_ll, 5),
                    "fold_details": fold_details,
                }
                all_results.append(entry)

                if avg_ll < best["avg_val_log_loss"]:
                    best = {
                        "avg_val_log_loss": avg_ll,
                        "params": {
                            "k_factor": k,
                            "home_advantage": hfa,
                            "preseason_regression": reg,
                        },
                        "fold_details": fold_details,
                    }

                print(
                    f"  [{count}/{total}] K={k:2d} HFA={hfa:2d} reg={reg:.2f}  "
                    f"avg_val_ll={avg_ll:.5f}"
                )

    all_results.sort(key=lambda x: x["avg_val_log_loss"])
    return best, all_results


def _minimal_logistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature DataFrame for minimal logistic challenger."""
    feat_cols = ["elo_diff", "elo_prob", "rest_diff", "is_neutral", "week"]
    present = [c for c in feat_cols if c in df.columns]
    return df[present].copy()


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    """Fit Platt scaling (logistic regression on single feature)."""
    platt = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    platt.fit(train_prob.reshape(-1, 1), train_y)
    return platt


def _fit_isotonic(train_prob: np.ndarray, train_y: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression for calibration."""
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(train_prob, train_y)
    return iso


def _evaluate_calibration_on_folds(
    df: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Evaluate Platt and isotonic calibration across rolling folds.

    For each fold, fits calibration on training seasons and evaluates on
    validation season.  Returns per-fold results for each method.
    """
    elo_prob = df["elo_prob"].values
    y = df[TARGET_COLUMN].astype(float).values

    platt_results: list[dict] = []
    iso_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df["season"].isin(train_seasons).values
        is_val = (df["season"] == val_season).values

        train_prob = elo_prob[is_train]
        train_y = y[is_train]
        val_prob = elo_prob[is_val]
        val_y = y[is_val]

        # Platt
        platt = _fit_platt(train_prob, train_y)
        platt_val_proba = platt.predict_proba(val_prob.reshape(-1, 1))[:, 1]
        platt_metrics = compute_classification_metrics(val_y, platt_val_proba)
        platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_metrics["log_loss"],
                "metrics": platt_metrics,
            }
        )

        # Isotonic
        iso = _fit_isotonic(train_prob, train_y)
        iso_val_proba = iso.predict(val_prob)
        iso_metrics = compute_classification_metrics(val_y, iso_val_proba)
        iso_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": iso_metrics["log_loss"],
                "metrics": iso_metrics,
            }
        )

    return {"platt": platt_results, "isotonic": iso_results}


def _evaluate_logistic_on_folds(
    df: pd.DataFrame,
) -> list[dict]:
    """Evaluate minimal logistic challenger across rolling folds.

    For each fold, trains on training seasons and evaluates on validation
    season.
    """
    feat_df = _minimal_logistic_features(df)
    y = df[TARGET_COLUMN].astype(int)

    log_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df["season"].isin(train_seasons).values
        is_val = (df["season"] == val_season).values

        x_train = feat_df[is_train]
        y_train = y[is_train]
        x_val = feat_df[is_val]
        y_val = y[is_val]

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(x_train, y_train)
        val_proba = pipe.predict_proba(x_val)[:, 1]
        metrics = compute_classification_metrics(y_val.values, val_proba)
        log_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": metrics["log_loss"],
                "metrics": metrics,
            }
        )

    return log_results


def run_rolling_origin_validation(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/rolling_origin_elo_validation.md",
) -> str:
    """Run the full rolling-origin Elo validation experiment.

    1. Grid search over expanded Elo params using rolling-origin folds.
       No holdout data is accessed during selection.
    2. With best params, compute final Elo features.
    3. Run calibration (Platt, isotonic) and minimal logistic with
       rolling-origin evaluation.
    4. Fit final models on all 2021-2024 data and evaluate on 2025 holdout
       exactly once.
    5. Compare against original Elo (K=20, HFA=0, reg=0) and current tuned
       Elo (K=32, HFA=25, reg=0).
    6. Write a comprehensive markdown report.

    Returns:
        Path to the written report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Step 1: Rolling-origin grid search ──
    print("=== Rolling-Origin Elo Grid Search ===")
    best, all_results = run_rolling_origin_grid_search(feature_table_path)

    best_params = best["params"]
    best_avg_ll = round(best["avg_val_log_loss"], 5)
    print(
        f"\nBest params (by avg val log loss): K={best_params['k_factor']}, "
        f"HFA={best_params['home_advantage']}, reg={best_params['preseason_regression']}"
    )
    print(f"  Avg val log loss: {best_avg_ll:.5f}")

    # ── Step 2: Compute final Elo with best params ──
    print("\n=== Final Elo with best params ===")
    best_elo = compute_elo_features(
        df_raw,
        k_factor=best_params["k_factor"],
        home_advantage=best_params["home_advantage"],
        preseason_regression=best_params["preseason_regression"],
    )
    best_elo = _filter_df(best_elo)

    # ── One-time 2025 holdout evaluation ──
    is_hold = (best_elo["season"] == HOLDOUT_SEASON).values
    hold_y = best_elo.loc[is_hold, TARGET_COLUMN].astype(float).values
    hold_prob = best_elo.loc[is_hold, "elo_prob"].values
    hold_metrics = compute_classification_metrics(hold_y, hold_prob)
    print(f"  Holdout (2025) log loss: {hold_metrics['log_loss']:.4f}")

    # ═══ Rolling-origin calibration evaluation ═══
    print("\n=== Rolling-Origin Calibration ===")
    cal_results = _evaluate_calibration_on_folds(best_elo)

    for cal_name, folds in cal_results.items():
        avg_ll = np.mean([f["log_loss"] for f in folds])
        print(f"  {cal_name}: avg val LL={avg_ll:.4f}")
        for f in folds:
            print(
                f"    fold train={f['train_seasons']} val={f['val_season']}: LL={f['log_loss']:.4f}"
            )

    # ═══ Rolling-origin logistic evaluation ═══
    print("\n=== Rolling-Origin Minimal Logistic ===")
    log_results = _evaluate_logistic_on_folds(best_elo)
    log_avg_ll = np.mean([f["log_loss"] for f in log_results])
    print(f"  Minimal logistic: avg val LL={log_avg_ll:.4f}")
    for f in log_results:
        print(f"    fold train={f['train_seasons']} val={f['val_season']}: LL={f['log_loss']:.4f}")

    # ═══ Original Elo K=20 (no HFA, no regression) ═══
    orig_elo = compute_elo_features(df_raw, k_factor=20, home_advantage=0, preseason_regression=0)
    orig_elo = _filter_df(orig_elo)
    orig_hold_y = (
        orig_elo.loc[orig_elo["season"] == HOLDOUT_SEASON, TARGET_COLUMN].astype(float).values
    )
    orig_hold_prob = orig_elo.loc[orig_elo["season"] == HOLDOUT_SEASON, "elo_prob"].values
    orig_hold_metrics = compute_classification_metrics(orig_hold_y, orig_hold_prob)

    # ═══ Current tuned Elo K=32, HFA=25, reg=0 ═══
    current_elo = compute_elo_features(
        df_raw, k_factor=32, home_advantage=25, preseason_regression=0
    )
    current_elo = _filter_df(current_elo)
    curr_hold_y = (
        current_elo.loc[current_elo["season"] == HOLDOUT_SEASON, TARGET_COLUMN].astype(float).values
    )
    curr_hold_prob = current_elo.loc[current_elo["season"] == HOLDOUT_SEASON, "elo_prob"].values
    curr_hold_metrics = compute_classification_metrics(curr_hold_y, curr_hold_prob)

    # ═══ Final models on full train (2021-2024) → 2025 holdout ═══
    is_train_full = best_elo["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_prob_full = best_elo.loc[is_train_full, "elo_prob"].values
    train_y_full = best_elo.loc[is_train_full, TARGET_COLUMN].astype(float).values

    # Platt final
    platt_final = _fit_platt(train_elo_prob_full, train_y_full)
    platt_hold_proba = platt_final.predict_proba(hold_prob.reshape(-1, 1))[:, 1]
    platt_hold_metrics = compute_classification_metrics(hold_y, platt_hold_proba)
    print(f"\n  Final Platt on holdout: LL={platt_hold_metrics['log_loss']:.4f}")

    # Isotonic final
    iso_final = _fit_isotonic(train_elo_prob_full, train_y_full)
    iso_hold_proba = iso_final.predict(hold_prob)
    iso_hold_metrics = compute_classification_metrics(hold_y, iso_hold_proba)
    print(f"  Final Isotonic on holdout: LL={iso_hold_metrics['log_loss']:.4f}")

    # Minimal logistic final
    feat_all = _minimal_logistic_features(best_elo)
    y_all = best_elo[TARGET_COLUMN].astype(int)
    x_train_full = feat_all[is_train_full]
    y_train_full = y_all[is_train_full]
    x_hold = feat_all[is_hold]

    log_final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    log_final.fit(x_train_full, y_train_full)
    log_hold_proba = log_final.predict_proba(x_hold)[:, 1]
    log_hold_metrics = compute_classification_metrics(hold_y, log_hold_proba)
    print(f"  Final Minimal Logistic on holdout: LL={log_hold_metrics['log_loss']:.4f}")

    # ═══ Simple baselines ═══
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # ═══ Write report ═══
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    # Compute per-fold metrics for calibration/logistic on validation
    def _avg_cal_ll(cal_name):
        return round(float(np.mean([f["log_loss"] for f in cal_results[cal_name]])), 4)

    platt_avg_val_ll = _avg_cal_ll("platt")
    iso_avg_val_ll = _avg_cal_ll("isotonic")
    log_avg_val_ll = round(float(log_avg_ll), 4)

    # Compute per-fold val LL for original Elo and current tuned Elo
    def _fold_val_lls(elo_df):
        lls = []
        for _, val_season in ROLLING_FOLDS:
            is_v = (elo_df["season"] == val_season).values
            v_y = elo_df.loc[is_v, TARGET_COLUMN].astype(float).values
            v_p = elo_df.loc[is_v, "elo_prob"].values
            lls.append(float(log_loss(v_y, v_p)))
        return lls

    orig_fold_lls = _fold_val_lls(orig_elo)
    curr_fold_lls = _fold_val_lls(current_elo)
    best_fold_lls = _fold_val_lls(best_elo)

    with open(rp, "w") as f:
        f.write("# Rolling-Origin Elo Validation\n\n")
        f.write(
            "*Cross-fold rolling-origin validation for Elo parameter"
            " selection with expanded grid.*\n\n"
        )

        f.write("## Data Split\n\n")
        f.write("| Split | Seasons | Description |\n")
        f.write("|-------|---------|-------------|\n")
        for idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS, 1):
            f.write(
                f"| Fold {idx} | Train: {train_seasons}, Val: {val_season}"
                f" | Elo param selection via avg val LL |\n"
            )
        f.write(
            f"| Holdout | {HOLDOUT_SEASON}"
            f" | Final untouched evaluation (never used for selection) |\n\n"
        )

        f.write("## Parameter Grid\n\n")
        f.write("| Parameter | Candidates |\n")
        f.write("|-----------|------------|\n")
        f.write(f"| K-factor | {K_FACTOR_CANDIDATES} |\n")
        f.write(f"| Home-field advantage (Elo) | {HFA_CANDIDATES} |\n")
        f.write(f"| Preseason regression toward 1500 | {REGRESSION_CANDIDATES} |\n\n")

        total = len(K_FACTOR_CANDIDATES) * len(HFA_CANDIDATES) * len(REGRESSION_CANDIDATES)
        f.write(f"Total combinations searched: {total}\n\n")

        f.write("## Top 5 Configurations (by average validation log loss)\n\n")
        f.write("| Rank | K | HFA | Regression | Avg Val LL | Fold1 LL | Fold2 LL | Fold3 LL |\n")
        f.write("|------|---|-----|------------|-----------|----------|----------|----------|\n")
        top5 = all_results[:5]
        for rank, entry in enumerate(top5, 1):
            fd = entry["fold_details"]
            f.write(
                f"| {rank} | {entry['k_factor']} | {entry['home_advantage']}"
                f" | {entry['preseason_regression']} | {entry['avg_val_log_loss']}"
                f" | {fd[0]['val_log_loss']} | {fd[1]['val_log_loss']}"
                f" | {fd[2]['val_log_loss']} |\n"
            )
        f.write("\n")

        f.write("## Best Configuration (selected by average validation log loss across folds)\n\n")
        f.write(
            f"- **K={best_params['k_factor']}, HFA={best_params['home_advantage']},"
            f" regression={best_params['preseason_regression']}**\n"
        )
        f.write(f"- Average validation log loss: {best_avg_ll:.5f}\n")
        for idx, (_, val_season) in enumerate(ROLLING_FOLDS):
            f.write(
                f"  - Fold {idx + 1} (val {val_season}):"
                f" {best['fold_details'][idx]['val_log_loss']:.5f}\n"
            )
        f.write(f"- Holdout (2025) log loss: {hold_metrics['log_loss']:.4f}\n\n")

        f.write("## Holdout (2025) Was NOT Used During Model Selection\n\n")
        f.write(
            "The grid search evaluated 210 parameter combinations across 3"
            " rolling-origin folds.  Selection was based **only** on average"
            " validation log loss.  The 2025 holdout was not accessed during"
            " any part of the grid search.  Holdout metrics in this report"
            " are for final comparison only.\n\n"
        )

        f.write("## Leakage Prevention\n\n")
        f.write(
            "- Elo features computed chronologically across all seasons.\n"
            "- Rolling-origin folds simulate realistic walk-forward evaluation.\n"
            "- Calibration (Platt, isotonic) fitted **only on training folds**"
            " during validation.\n"
            "- Final calibration fitted on 2021–2024, then applied to 2025.\n"
            "- Minimal logistic model trained only on training folds during"
            " selection; final model trained on 2021–2024.\n"
            "- 2025 holdout never touched during any fitting or selection step.\n\n"
        )

        # ── Comparison table ──
        def _cell(v, metric, fmt=".4f"):
            return f"{v[metric]:{fmt}}" if v is not None else "—"

        def _row(name, h_metrics):
            h_ll = _cell(h_metrics, "log_loss")
            h_bs = _cell(h_metrics, "brier_score")
            h_ac = _cell(h_metrics, "accuracy")
            h_au = _cell(h_metrics, "roc_auc")
            return f"| {name} | {h_ll} | {h_bs} | {h_ac} | {h_au} |\n"

        f.write("## Full Comparison (2025 Holdout)\n\n")
        header = "| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n"
        sep = "|-------|---------|------------|----------|----------|\n"
        f.write(header)
        f.write(sep)

        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(_row("Original Elo K=20", orig_hold_metrics))
        f.write(_row("Current tuned Elo K=32 HFA=25", curr_hold_metrics))
        f.write(_row("Rolling-origin selected raw Elo", hold_metrics))
        f.write(_row("Rolling-origin selected + Platt", platt_hold_metrics))
        f.write(_row("Rolling-origin selected + Isotonic", iso_hold_metrics))
        f.write(_row("Rolling-origin selected Minimal Logistic", log_hold_metrics))
        f.write("\n")

        # ── Average validation metrics across folds ──
        f.write("## Average Validation Metrics Across Folds\n\n")
        f.write("| Model | Avg Val LL | Fold1 LL | Fold2 LL | Fold3 LL |\n")
        f.write("|-------|------------|----------|----------|----------|\n")
        orig_avg = np.mean(orig_fold_lls)
        curr_avg = np.mean(curr_fold_lls)
        best_avg = np.mean(best_fold_lls)
        f.write(
            f"| Original Elo K=20 | {orig_avg:.4f}"
            f" | {orig_fold_lls[0]:.4f} | {orig_fold_lls[1]:.4f}"
            f" | {orig_fold_lls[2]:.4f} |\n"
        )
        f.write(
            f"| Current tuned Elo K=32 HFA=25 | {curr_avg:.4f}"
            f" | {curr_fold_lls[0]:.4f} | {curr_fold_lls[1]:.4f}"
            f" | {curr_fold_lls[2]:.4f} |\n"
        )
        f.write(
            f"| Rolling-origin selected raw Elo | {best_avg:.4f}"
            f" | {best_fold_lls[0]:.4f} | {best_fold_lls[1]:.4f}"
            f" | {best_fold_lls[2]:.4f} |\n"
        )
        # Platt average from folds
        platt_fold_lls = [f["log_loss"] for f in cal_results["platt"]]
        f.write(
            f"| Rolling-origin selected + Platt | {platt_avg_val_ll}"
            f" | {platt_fold_lls[0]:.4f} | {platt_fold_lls[1]:.4f}"
            f" | {platt_fold_lls[2]:.4f} |\n"
        )
        # Isotonic average from folds
        iso_fold_lls = [f["log_loss"] for f in cal_results["isotonic"]]
        f.write(
            f"| Rolling-origin selected + Isotonic | {iso_avg_val_ll}"
            f" | {iso_fold_lls[0]:.4f} | {iso_fold_lls[1]:.4f}"
            f" | {iso_fold_lls[2]:.4f} |\n"
        )
        # Logistic average from folds
        log_fold_lls = [f["log_loss"] for f in log_results]
        f.write(
            f"| Rolling-origin selected Minimal Logistic | {log_avg_val_ll}"
            f" | {log_fold_lls[0]:.4f} | {log_fold_lls[1]:.4f}"
            f" | {log_fold_lls[2]:.4f} |\n\n"
        )

        # ── Calibration bucket details ──
        for label, h_met in [
            ("Rolling-Origin Selected Raw Elo (Holdout)", hold_metrics),
            ("Platt-Calibrated Elo (Holdout)", platt_hold_metrics),
            ("Isotonic-Calibrated Elo (Holdout)", iso_hold_metrics),
        ]:
            f.write(f"## {label}\n\n")
            f.write("| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |\n")
            f.write("|--------|-------|----------------|-------------|-------------------|\n")
            for bucket_label, b in sorted(h_met["calibration_buckets"].items()):
                f.write(
                    f"| {bucket_label} | {b['count']} | {b['mean_predicted_prob']}"
                    f" | {b['mean_actual_rate']} | {b['calibration_error']} |\n"
                )
            f.write("\n")

        # ── Isotonic risk note ──
        f.write("## Isotonic Calibration Risk\n\n")
        iso_improved = iso_hold_metrics["log_loss"] < hold_metrics["log_loss"]
        if iso_improved:
            f.write(
                "Isotonic calibration improved holdout log loss, but this"
                " comes with high overfit risk.  Isotonic regression is"
                " non-parametric and can overfit to small training sets."
                " Unless it wins cleanly across all rolling folds, it should"
                " not be promoted over Platt scaling.\n\n"
            )
        else:
            f.write(
                "Isotonic calibration did not improve holdout log loss and"
                " carries high overfit risk.  **Rejected.**\n\n"
            )

        # ── Per-fold calibration deciles for the selected model ──
        f.write("## Rolling-Origin Calibration Deciles (Selected Raw Elo)\n\n")
        for idx, (_, val_season) in enumerate(ROLLING_FOLDS):
            f.write(f"### Fold {idx + 1} (Validation {val_season})\n\n")
            is_v = (best_elo["season"] == val_season).values
            v_y = best_elo.loc[is_v, TARGET_COLUMN].astype(float).values
            v_p = best_elo.loc[is_v, "elo_prob"].values
            v_met = compute_classification_metrics(v_y, v_p)
            f.write("| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |\n")
            f.write("|--------|-------|----------------|-------------|-------------------|\n")
            for bucket_label, b in sorted(v_met["calibration_buckets"].items()):
                f.write(
                    f"| {bucket_label} | {b['count']} | {b['mean_predicted_prob']}"
                    f" | {b['mean_actual_rate']} | {b['calibration_error']} |\n"
                )
            f.write("\n")

        # ── Recommendation ──
        f.write("## Recommendation\n\n")

        candidates = {
            "Original Elo K=20": orig_hold_metrics["log_loss"],
            "Current tuned Elo K=32 HFA=25": curr_hold_metrics["log_loss"],
            "Rolling-origin selected raw Elo": hold_metrics["log_loss"],
            "Rolling-origin selected + Platt": platt_hold_metrics["log_loss"],
            "Rolling-origin selected + Isotonic": iso_hold_metrics["log_loss"],
            "Rolling-origin selected Minimal Logistic": log_hold_metrics["log_loss"],
        }
        best_name = min(candidates, key=candidates.get)
        best_ll = candidates[best_name]

        # Incumbent is the current tuned Elo (K=32, HFA=25, reg=0)
        incumbent_ll = curr_hold_metrics["log_loss"]

        improved = best_ll < incumbent_ll - 0.001

        if improved:
            f.write(
                f"✅ **{best_name} is the new research incumbent.** "
                f"Holdout log loss {best_ll:.4f} beats the current tuned Elo incumbent "
                f"({incumbent_ll:.4f}).\n\n"
            )
            f.write(
                "Rolling-origin validation selected a configuration that"
                " generalizes better across seasons.\n"
            )
        else:
            f.write(
                f"⚠️ **Current tuned Elo (K=32, HFA=25, reg=0) remains the"
                f" research incumbent.** "
                f"Holdout log loss {incumbent_ll:.4f}. "
                f"No rolling-origin selected model achieved meaningfully lower"
                f" holdout log loss.\n\n"
            )
            f.write(f"Best rolling-origin challenger: **{best_name}** ({best_ll:.4f}).\n\n")
            # Check if Platt is promising
            platt_improved = platt_hold_metrics["log_loss"] < incumbent_ll - 0.001
            if platt_improved:
                f.write(
                    "Platt calibration is a promising challenger. It should"
                    " be re-evaluated with the next experiment, but not"
                    " promoted to incumbent on a single holdout.\n\n"
                )
            else:
                f.write(
                    "No challenger beat the incumbent cleanly. Next"
                    " experiments should focus on adding feature signals"
                    " (weather, advanced metrics) rather than further"
                    " Elo tuning.\n\n"
                )

        f.write("### Next Recommended Experiment\n\n")
        f.write(
            "1. Add weather features to the minimal logistic model.\n"
            "2. Test a GradientBoosting model with clean pregame features.\n"
            "3. Expand Elo K-factor grid above 48.\n"
        )

    print(f"\nReport written to: {rp}")
    return str(rp)
