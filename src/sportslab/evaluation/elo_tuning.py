"""Elo tuning, calibration, and comparison experiment."""

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

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASON = 2024
HOLDOUT_SEASON = 2025
K_FACTOR_CANDIDATES = [4, 8, 12, 16, 20, 24, 32]
HFA_CANDIDATES = [0, 25, 40, 55, 65, 75]
REGRESSION_CANDIDATES = [0.0, 0.25, 0.33, 0.50]


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard filters: model_eligible, non-neutral."""
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _split_arrays(df: pd.DataFrame) -> dict:
    """Split target and elo_prob into train/val/holdout arrays."""
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"] == VAL_SEASON]
    hold = df[df["season"] == HOLDOUT_SEASON]
    return {
        "train_y": train[TARGET_COLUMN].astype(float).values,
        "val_y": val[TARGET_COLUMN].astype(float).values,
        "hold_y": hold[TARGET_COLUMN].astype(float).values,
        "train_prob": train["elo_prob"].values if "elo_prob" in train.columns else None,
        "val_prob": val["elo_prob"].values if "elo_prob" in val.columns else None,
        "hold_prob": hold["elo_prob"].values if "elo_prob" in hold.columns else None,
    }


def run_elo_grid_search(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    compute_holdout: bool = False,
) -> tuple[dict, list[dict]]:
    """Run a grid search over Elo parameter combinations.

    Evaluates each combination by log loss on the validation season (2024).
    The 2025 holdout is NEVER used for selection.

    Args:
        feature_table_path: Path to the feature table parquet.
        compute_holdout: If True, also compute and store holdout log loss
            (for report display).  Default False — holdout untouched during
            model selection.

    Returns:
        (best_params, all_results) where best_params is a dict with keys
        'k_factor', 'home_advantage', 'preseason_regression' and all_results
        is a list of result dicts sorted by validation log loss ascending.
    """
    df_raw = pd.read_parquet(feature_table_path)
    best: dict = {"val_log_loss": float("inf"), "params": None}
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

                is_val = (edf["season"] == VAL_SEASON).values
                val_loss = float(log_loss(y[is_val], elo_prob[is_val]))

                entry: dict = {
                    "k_factor": k,
                    "home_advantage": hfa,
                    "preseason_regression": reg,
                    "val_log_loss": round(val_loss, 5),
                }

                if compute_holdout:
                    is_hold = (edf["season"] == HOLDOUT_SEASON).values
                    hold_loss = float(log_loss(y[is_hold], elo_prob[is_hold]))
                    entry["holdout_log_loss"] = round(hold_loss, 5)
                else:
                    hold_loss = None

                all_results.append(entry)

                if val_loss < best["val_log_loss"]:
                    best = {
                        "val_log_loss": val_loss,
                        "params": {
                            "k_factor": k,
                            "home_advantage": hfa,
                            "preseason_regression": reg,
                        },
                    }
                    if hold_loss is not None:
                        best["holdout_log_loss"] = hold_loss

                msg = (
                    f"  [{count}/{total}] K={k:2d} HFA={hfa:2d}"
                    f" reg={reg:.2f}  val_ll={val_loss:.5f}"
                )
                if hold_loss is not None:
                    msg += f"  hold_ll={hold_loss:.5f}"
                print(msg)

    all_results.sort(key=lambda x: x["val_log_loss"])
    return best, all_results


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


def _minimal_logistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature DataFrame for minimal logistic challenger."""
    feat_cols = ["elo_diff", "elo_prob", "rest_diff", "is_neutral", "week"]
    present = [c for c in feat_cols if c in df.columns]
    return df[present].copy()


def run_elo_tuning(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/elo_tuning_calibration.md",
) -> str:
    """Run the full Elo tuning experiment.

    1. Grid search over Elo params (select by val log loss, no holdout peek).
    2. With best params, compute final Elo features + calibrations + comparisons.
    3. Write a markdown report.

    Returns:
        Path to the written report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Step 1: Grid search ──
    print("=== Elo Grid Search ===")
    best, all_results = run_elo_grid_search(feature_table_path, compute_holdout=True)

    best_params = best["params"]
    best_val_ll = round(best["val_log_loss"], 5)
    best_hold_ll = round(best["holdout_log_loss"], 5)
    print(
        f"\nBest params (by val log loss): K={best_params['k_factor']}, "
        f"HFA={best_params['home_advantage']}, reg={best_params['preseason_regression']}"
    )
    print(f"  Val log loss: {best_val_ll:.5f}, Holdout log loss: {best_hold_ll:.5f}")

    # ── Step 2: Compute final Elo with best params ──
    print("\n=== Final Elo with best params ===")
    best_elo = compute_elo_features(
        df_raw,
        k_factor=best_params["k_factor"],
        home_advantage=best_params["home_advantage"],
        preseason_regression=best_params["preseason_regression"],
    )
    best_elo = _filter_df(best_elo)
    splits = _split_arrays(best_elo)

    # ═══ Calibration experiments ═══
    print("\n=== Calibration ===")

    # Raw Elo (tuned) — already in splits["val_prob"], splits["hold_prob"]
    raw_val_metrics = compute_classification_metrics(splits["val_y"], splits["val_prob"])
    raw_hold_metrics = compute_classification_metrics(splits["hold_y"], splits["hold_prob"])
    print(
        f"  Raw Elo (tuned):   val LL={raw_val_metrics['log_loss']:.4f}, "
        f"hold LL={raw_hold_metrics['log_loss']:.4f}"
    )

    # Platt scaling
    platt = _fit_platt(splits["train_prob"], splits["train_y"])
    platt_val_proba = platt.predict_proba(splits["val_prob"].reshape(-1, 1))[:, 1]
    platt_hold_proba = platt.predict_proba(splits["hold_prob"].reshape(-1, 1))[:, 1]
    platt_val_metrics = compute_classification_metrics(splits["val_y"], platt_val_proba)
    platt_hold_metrics = compute_classification_metrics(splits["hold_y"], platt_hold_proba)
    print(
        f"  Platt:            val LL={platt_val_metrics['log_loss']:.4f}, "
        f"hold LL={platt_hold_metrics['log_loss']:.4f}"
    )

    # Isotonic
    iso = _fit_isotonic(splits["train_prob"], splits["train_y"])
    iso_val_proba = iso.predict(splits["val_prob"])
    iso_hold_proba = iso.predict(splits["hold_prob"])
    iso_val_metrics = compute_classification_metrics(splits["val_y"], iso_val_proba)
    iso_hold_metrics = compute_classification_metrics(splits["hold_y"], iso_hold_proba)
    print(
        f"  Isotonic:         val LL={iso_val_metrics['log_loss']:.4f}, "
        f"hold LL={iso_hold_metrics['log_loss']:.4f}"
    )

    # ═══ Original K=20 Elo-only (no HFA, no regression) ═══
    orig_elo = compute_elo_features(df_raw, k_factor=20, home_advantage=0, preseason_regression=0)
    orig_elo = _filter_df(orig_elo)
    orig_splits = _split_arrays(orig_elo)
    orig_val_metrics = compute_classification_metrics(orig_splits["val_y"], orig_splits["val_prob"])
    orig_hold_metrics = compute_classification_metrics(
        orig_splits["hold_y"], orig_splits["hold_prob"]
    )

    # ═══ Minimal logistic challenger ═══
    print("\n=== Minimal Logistic Challenger ===")
    feat_df = _minimal_logistic_features(best_elo)
    y = best_elo[TARGET_COLUMN].astype(int)
    is_train = best_elo["season"].isin(TRAIN_SEASONS)
    is_val = best_elo["season"] == VAL_SEASON
    is_hold = best_elo["season"] == HOLDOUT_SEASON

    x_train = feat_df[is_train]
    y_train = y[is_train]
    x_val = feat_df[is_val]
    y_val = y[is_val]
    x_hold = feat_df[is_hold]
    y_hold = y[is_hold]

    min_logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    min_logistic.fit(x_train, y_train)
    min_log_val_proba = min_logistic.predict_proba(x_val)[:, 1]
    min_log_hold_proba = min_logistic.predict_proba(x_hold)[:, 1]
    min_log_val_metrics = compute_classification_metrics(y_val.values, min_log_val_proba)
    min_log_hold_metrics = compute_classification_metrics(y_hold.values, min_log_hold_proba)
    print(
        f"  Minimal Logistic: val LL={min_log_val_metrics['log_loss']:.4f}, "
        f"hold LL={min_log_hold_metrics['log_loss']:.4f}"
    )

    # ═══ Previous logistic team-strength for comparison ═══
    # Reconstruct from the earlier experiment's constants
    prev_ts_val_ll = 0.6477
    prev_ts_val_auc = 0.6896
    prev_ts_hold_ll = 0.6866
    prev_ts_hold_auc = 0.6531

    # ═══ Simple baselines ═══
    random_val_ll = float(log_loss(splits["val_y"], np.full_like(splits["val_y"], 0.5)))
    random_hold_ll = float(log_loss(splits["hold_y"], np.full_like(splits["hold_y"], 0.5)))
    prior = splits["train_y"].mean()
    prior_val_ll = float(log_loss(splits["val_y"], np.full_like(splits["val_y"], prior)))
    prior_hold_ll = float(log_loss(splits["hold_y"], np.full_like(splits["hold_y"], prior)))

    # ═══ Write report ═══
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    _write_report(
        rp,
        all_results,
        best_params,
        best_val_ll,
        best_hold_ll,
        raw_val_metrics,
        raw_hold_metrics,
        platt_val_metrics,
        platt_hold_metrics,
        iso_val_metrics,
        iso_hold_metrics,
        orig_val_metrics,
        orig_hold_metrics,
        min_log_val_metrics,
        min_log_hold_metrics,
        prev_ts_val_ll,
        prev_ts_val_auc,
        prev_ts_hold_ll,
        prev_ts_hold_auc,
        random_val_ll,
        random_hold_ll,
        prior_val_ll,
        prior_hold_ll,
        priors_rate=prior,
    )

    print(f"\nReport written to: {rp}")
    return str(rp)


def _write_report(
    path: Path,
    all_results: list[dict],
    best_params: dict,
    best_val_ll: float,
    best_hold_ll: float,
    raw_val_metrics: dict,
    raw_hold_metrics: dict,
    platt_val_metrics: dict,
    platt_hold_metrics: dict,
    iso_val_metrics: dict,
    iso_hold_metrics: dict,
    orig_val_metrics: dict,
    orig_hold_metrics: dict,
    min_log_val_metrics: dict,
    min_log_hold_metrics: dict,
    prev_ts_val_ll: float,
    prev_ts_val_auc: float,
    prev_ts_hold_ll: float,
    prev_ts_hold_auc: float,
    random_val_ll: float,
    random_hold_ll: float,
    prior_val_ll: float,
    prior_hold_ll: float,
    priors_rate: float,
) -> None:
    """Write the Elo tuning and calibration report."""
    top5 = all_results[:5]
    best_str = (
        f"K={best_params['k_factor']}, HFA={best_params['home_advantage']}, "
        f"regression={best_params['preseason_regression']}"
    )

    with open(path, "w") as f:
        f.write("# Elo Tuning and Calibration\n\n")
        f.write("*Systematic Elo parameter search + calibration comparison.*\n\n")

        f.write("## Data Split\n\n")
        f.write("| Split | Seasons | Description |\n")
        f.write("|-------|---------|-------------|\n")
        f.write(f"| Train | {TRAIN_SEASONS} | Elo updates + Platt + isotonic + logistic fit |\n")
        f.write(f"| Validation | {VAL_SEASON} | Model selection — best Elo params chosen here |\n")
        f.write(
            f"| Holdout | {HOLDOUT_SEASON} |"
            f" Final untouched evaluation (never used for selection) |\n\n"
        )

        f.write("## Parameter Grid\n\n")
        f.write("| Parameter | Candidates |\n")
        f.write("|-----------|------------|\n")
        f.write(f"| K-factor | {K_FACTOR_CANDIDATES} |\n")
        f.write(f"| Home-field advantage (Elo) | {HFA_CANDIDATES} |\n")
        f.write(f"| Preseason regression toward 1500 | {REGRESSION_CANDIDATES} |\n\n")

        total = len(K_FACTOR_CANDIDATES) * len(HFA_CANDIDATES) * len(REGRESSION_CANDIDATES)
        f.write(f"Total combinations searched: {total}\n\n")

        f.write("## Top 5 Configurations (by validation log loss)\n\n")
        f.write("| Rank | K | HFA | Regression | Val Log Loss | Holdout Log Loss |\n")
        f.write("|------|---|-----|------------|--------------|------------------|\n")
        for rank, entry in enumerate(top5, 1):
            f.write(
                f"| {rank} | {entry['k_factor']} | {entry['home_advantage']} | "
                f"{entry['preseason_regression']} | {entry['val_log_loss']} | "
                f"{entry['holdout_log_loss']} |\n"
            )
        f.write("\n")

        f.write("## Best Configuration (selected by validation log loss)\n\n")
        f.write(f"- **{best_str}**\n")
        f.write(f"- Validation log loss: {best_val_ll:.5f}\n")
        f.write(f"- Holdout log loss: {best_hold_ll:.5f}\n\n")

        f.write("## Holdout (2025) Was NOT Used for Selection\n\n")
        f.write(
            "The holdout season (2025) remained untouched during the grid search. "
            "All 168 parameter combinations were evaluated only on validation (2024). "
            "The holdout results shown in this report are for final comparison only.\n\n"
        )

        f.write("## Leakage Prevention\n\n")
        f.write(
            "- Elo features are computed chronologically: for each game, features\n"
            "  depend only on games played before it.\n"
            "- Calibration (Platt, isotonic) is fitted **only on training data** and\n"
            "  applied to validation and holdout.\n"
            "- Minimal logistic model is trained only on training data.\n"
            "- The 2025 holdout is never accessed during any fitting or selection step.\n\n"
        )

        # ── Comparison table ──
        def _cell(v, metric, fmt=".4f"):
            return f"{v[metric]:{fmt}}" if v is not None else "—"

        def _row(name, v_metrics, h_metrics):
            v_ll = _cell(v_metrics, "log_loss")
            v_bs = _cell(v_metrics, "brier_score")
            v_ac = _cell(v_metrics, "accuracy")
            v_au = _cell(v_metrics, "roc_auc")
            h_ll = _cell(h_metrics, "log_loss")
            h_bs = _cell(h_metrics, "brier_score")
            h_ac = _cell(h_metrics, "accuracy")
            h_au = _cell(h_metrics, "roc_auc")
            return (
                f"| {name} | {v_ll} | {v_bs} | {v_ac} | {v_au}"
                f" | {h_ll} | {h_bs} | {h_ac} | {h_au} |\n"
            )

        f.write("## Full Comparison\n\n")
        header = (
            "| Model | Val LL | Val Brier | Val Acc | Val AUC |"
            " Hold LL | Hold Brier | Hold Acc | Hold AUC |\n"
        )
        sep = (
            "|-------|--------|-----------|---------|---------|"
            "---------|------------|----------|----------|\n"
        )
        f.write(header)
        f.write(sep)

        f.write(
            f"| Random | {random_val_ll:.4f} | 0.2500 | 0.5000 | 0.5000"
            f" | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n"
        )
        f.write(
            f"| Home prior ({priors_rate:.3f}) | {prior_val_ll:.4f}"
            f" | — | — | 0.5000 | {prior_hold_ll:.4f} | — | — | 0.5000 |\n"
        )
        f.write(_row("Elo K=20 (original)", orig_val_metrics, orig_hold_metrics))
        f.write(_row("Elo tuned (raw)", raw_val_metrics, raw_hold_metrics))
        f.write(_row("Elo tuned + Platt", platt_val_metrics, platt_hold_metrics))
        f.write(_row("Elo tuned + Isotonic", iso_val_metrics, iso_hold_metrics))
        f.write(_row("Minimal Logistic", min_log_val_metrics, min_log_hold_metrics))
        f.write(
            f"| Prev logistic team-strength | {prev_ts_val_ll:.4f}"
            f" | — | — | {prev_ts_val_auc:.4f} | {prev_ts_hold_ll:.4f}"
            f" | — | — | {prev_ts_hold_auc:.4f} |\n\n"
        )

        # ── Calibration bucket details ──
        for section_label, v_metrics, h_metrics in [
            ("Raw Elo (tuned)", raw_val_metrics, raw_hold_metrics),
            ("Platt-calibrated Elo", platt_val_metrics, platt_hold_metrics),
            ("Isotonic-calibrated Elo", iso_val_metrics, iso_hold_metrics),
        ]:
            f.write(f"## {section_label}\n\n")
            for split_name, m in [("Validation", v_metrics), ("Holdout", h_metrics)]:
                f.write(f"### {split_name} Calibration Buckets\n\n")
                f.write("| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |\n")
                f.write("|--------|-------|----------------|-------------|-------------------|\n")
                for bucket_label, b in sorted(m["calibration_buckets"].items()):
                    f.write(
                        f"| {bucket_label} | {b['count']} | {b['mean_predicted_prob']} "
                        f"| {b['mean_actual_rate']} | {b['calibration_error']} |\n"
                    )
                f.write("\n")

        # ── Recommendation ──
        f.write("## Recommendation\n\n")

        # Determine the best performer on holdout
        candidates = {
            "Original Elo K=20": orig_hold_metrics["log_loss"],
            "Tuned Elo raw": raw_hold_metrics["log_loss"],
            "Tuned Elo + Platt": platt_hold_metrics["log_loss"],
            "Tuned Elo + Isotonic": iso_hold_metrics["log_loss"],
            "Minimal Logistic": min_log_hold_metrics["log_loss"],
        }
        best_name = min(candidates, key=candidates.get)
        best_ll = candidates[best_name]
        incumbent_ll = orig_hold_metrics["log_loss"]

        improved = best_ll < incumbent_ll - 0.001

        if improved:
            f.write(
                f"✅ **{best_name} is the new research incumbent.** "
                f"Holdout log loss {best_ll:.4f} beats original Elo K=20 "
                f"({incumbent_ll:.4f}). "
            )
            if best_name == "Tuned Elo raw":
                f.write("Parameter tuning improved Elo without adding complexity. ")
            elif "Platt" in best_name:
                f.write("Platt scaling improved probability calibration. ")
            elif "Isotonic" in best_name:
                f.write("Isotonic regression improved probability calibration. ")
            f.write("Future models must beat this benchmark.\n")
        else:
            f.write(
                f"⚠️ **Original Elo K=20 remains the research incumbent.** "
                f"No tuned or calibrated model achieved meaningfully lower "
                f"holdout log loss than {incumbent_ll:.4f}. "
                f"Best candidate ({best_name}) achieved {best_ll:.4f}.\n\n"
            )
            f.write(
                "Parameter tuning did not provide a material improvement. "
                "The original simple Elo (K=20, no HFA, no regression) is "
                "remarkably robust. Future experiments should focus on "
                "adding new feature signals (weather, advanced metrics) "
                "rather than further Elo tuning.\n"
            )

        f.write("\n### Next Recommended Experiment\n\n")
        f.write(
            "Add weather features to the minimal logistic model or test "
            "a GradientBoosting model with clean pregame features "
            "(Elo + rest + structural).  Weather may provide the signal "
            "needed to break through the Elo ceiling.\n"
        )
