"""Scheduling/rest feature experiment on top of Elo+Platt incumbent.

Rolling-origin validation across 3 folds, one-shot 2025 holdout.
"""

from pathlib import Path

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
    SCHEDULING_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from sportslab.features.ratings import compute_elo_features
from sportslab.features.scheduling import compute_scheduling_features

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

# Frozen incumbent Elo params
BEST_K = 40
BEST_HFA = 40
BEST_REG = 0.25


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


def _logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def run_schedule_rest_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/schedule_rest_features.md",
) -> str:
    """Run scheduling/rest feature experiment with rolling-origin validation.

    1. Compute Elo with frozen incumbent params (K=40, HFA=40, reg=0.25).
    2. Compute scheduling features chronologically.
    3. Rolling-origin evaluation for each challenger.
    4. One-time 2025 holdout evaluation.
    5. Report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Compute Elo with frozen incumbent params ──
    print("=== Computing Elo features (incumbent params) ===")
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
    )
    print(f"  K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}")

    # ── Compute scheduling features ──
    print("=== Computing scheduling features ===")
    df_all = compute_scheduling_features(df_elo)
    added = [c for c in df_all.columns if c not in df_elo.columns]
    print(f"  Added: {added}")

    # ── Filter ──
    df_all = _filter_df(df_all)

    # ═══ Rolling-origin evaluation ═══
    print("\n=== Rolling-Origin Evaluation ===")

    scheduling_available = [c for c in SCHEDULING_FEATURE_COLUMNS if c in df_all.columns]
    print(f"  Scheduling features used: {scheduling_available}")

    elo_prob = df_all["elo_prob"].values
    y = df_all[TARGET_COLUMN].astype(float).values

    elo_only_results: list[dict] = []
    platt_results: list[dict] = []
    incumbent_plus_sched_results: list[dict] = []
    raw_elo_sched_results: list[dict] = []
    sched_only_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values

        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]

        sched_train = df_all.loc[is_train, scheduling_available]
        sched_val = df_all.loc[is_val, scheduling_available]

        # 1. Raw Elo only
        raw_metrics = compute_classification_metrics(val_y_, val_elo)
        elo_only_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": raw_metrics["log_loss"],
                "metrics": raw_metrics,
            }
        )

        # 2. Platt-scaled Elo (incumbent)
        platt = _fit_platt(train_elo, train_y_)
        platt_val_proba = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        platt_metrics = compute_classification_metrics(val_y_, platt_val_proba)
        platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_metrics["log_loss"],
                "metrics": platt_metrics,
                "model": platt,
            }
        )

        # 3. Incumbent (Platt) + Scheduling logistic
        platt_train_proba = platt.predict_proba(train_elo.reshape(-1, 1))[:, 1]
        ip_sched_train = np.column_stack([platt_train_proba, sched_train.values])
        ip_sched_val = np.column_stack([platt_val_proba, sched_val.values])

        ip_sched_pipe = _logistic_model()
        ip_sched_pipe.fit(ip_sched_train, train_y_)
        ip_sched_val_proba = ip_sched_pipe.predict_proba(ip_sched_val)[:, 1]
        ip_sched_metrics = compute_classification_metrics(val_y_, ip_sched_val_proba)
        incumbent_plus_sched_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": ip_sched_metrics["log_loss"],
                "metrics": ip_sched_metrics,
                "model": ip_sched_pipe,
            }
        )

        # 4. Raw Elo + Scheduling logistic
        re_sched_train = np.column_stack([train_elo, sched_train.values])
        re_sched_val = np.column_stack([val_elo, sched_val.values])

        re_sched_pipe = _logistic_model()
        re_sched_pipe.fit(re_sched_train, train_y_)
        re_sched_val_proba = re_sched_pipe.predict_proba(re_sched_val)[:, 1]
        re_sched_metrics = compute_classification_metrics(val_y_, re_sched_val_proba)
        raw_elo_sched_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": re_sched_metrics["log_loss"],
                "metrics": re_sched_metrics,
                "model": re_sched_pipe,
            }
        )

        # 5. Scheduling-only logistic
        sched_only_pipe = _logistic_model()
        sched_only_pipe.fit(sched_train.values, train_y_)
        sched_only_val_proba = sched_only_pipe.predict_proba(sched_val.values)[:, 1]
        sched_only_metrics = compute_classification_metrics(val_y_, sched_only_val_proba)
        sched_only_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": sched_only_metrics["log_loss"],
                "metrics": sched_only_metrics,
                "model": sched_only_pipe,
            }
        )

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" raw_elo={raw_metrics['log_loss']:.4f}"
            f" platt={platt_metrics['log_loss']:.4f}"
            f" inc+sched={ip_sched_metrics['log_loss']:.4f}"
            f" raw+sched={re_sched_metrics['log_loss']:.4f}"
            f" sched={sched_only_metrics['log_loss']:.4f}"
        )

    # ── Compute average validation metrics ──
    def _avg_ll(results):
        return float(np.mean([r["log_loss"] for r in results]))

    avg_elo = _avg_ll(elo_only_results)
    avg_platt = _avg_ll(platt_results)
    avg_ip_sched = _avg_ll(incumbent_plus_sched_results)
    avg_re_sched = _avg_ll(raw_elo_sched_results)
    avg_sched_only = _avg_ll(sched_only_results)

    print("\n=== Average Validation Log Loss ===")
    print(f"  Raw Elo:        {avg_elo:.4f}")
    print(f"  Platt (incumb): {avg_platt:.4f}")
    print(f"  Inc+Sched:      {avg_ip_sched:.4f}")
    print(f"  Raw+Sched:      {avg_re_sched:.4f}")
    print(f"  Sched only:     {avg_sched_only:.4f}")

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    # Train final models on ALL 2021-2024
    is_train_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    sched_full = df_all.loc[is_train_full, scheduling_available]
    sched_hold = df_all.loc[is_hold, scheduling_available]

    # Raw Elo holdout
    hold_raw_metrics = compute_classification_metrics(hold_y, hold_elo)
    print(f"  Raw Elo:            {hold_raw_metrics['log_loss']:.4f}")

    # Platt incumbent holdout
    platt_full = _fit_platt(train_elo_full, train_y_full)
    platt_hold_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_metrics = compute_classification_metrics(hold_y, platt_hold_proba)
    print(f"  Platt (incumbent):  {hold_platt_metrics['log_loss']:.4f}")

    # Incumbent + Scheduling holdout
    platt_train_full = platt_full.predict_proba(train_elo_full.reshape(-1, 1))[:, 1]
    ip_sched_train_full = np.column_stack([platt_train_full, sched_full.values])
    platt_hold_full = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    ip_sched_hold = np.column_stack([platt_hold_full, sched_hold.values])

    ip_sched_final = _logistic_model()
    ip_sched_final.fit(ip_sched_train_full, train_y_full)
    ip_sched_hold_proba = ip_sched_final.predict_proba(ip_sched_hold)[:, 1]
    hold_ip_sched_metrics = compute_classification_metrics(hold_y, ip_sched_hold_proba)
    print(f"  Inc+Sched:          {hold_ip_sched_metrics['log_loss']:.4f}")

    # Raw Elo + Scheduling holdout
    re_sched_train_full = np.column_stack([train_elo_full, sched_full.values])
    re_sched_hold = np.column_stack([hold_elo, sched_hold.values])

    re_sched_final = _logistic_model()
    re_sched_final.fit(re_sched_train_full, train_y_full)
    re_sched_hold_proba = re_sched_final.predict_proba(re_sched_hold)[:, 1]
    hold_re_sched_metrics = compute_classification_metrics(hold_y, re_sched_hold_proba)
    print(f"  Raw+Sched:          {hold_re_sched_metrics['log_loss']:.4f}")

    # Scheduling-only holdout
    sched_only_final = _logistic_model()
    sched_only_final.fit(sched_full.values, train_y_full)
    sched_only_hold_proba = sched_only_final.predict_proba(sched_hold.values)[:, 1]
    hold_sched_only_metrics = compute_classification_metrics(hold_y, sched_only_hold_proba)
    print(f"  Sched only:         {hold_sched_only_metrics['log_loss']:.4f}")

    # ═══ Comparison baselines ═══
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # ═══ Report ═══
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    def _cell(v, metric, fmt=".4f"):
        return f"{v[metric]:{fmt}}" if v is not None else "—"

    def _row(name, h_metrics):
        return (
            f"| {name} | {_cell(h_metrics, 'log_loss')}"
            f" | {_cell(h_metrics, 'brier_score')}"
            f" | {_cell(h_metrics, 'accuracy')}"
            f" | {_cell(h_metrics, 'roc_auc')} |\n"
        )

    with open(rp, "w") as f:
        f.write("# Scheduling/Rest Feature Experiment\n\n")
        f.write(
            "*Adding pregame scheduling and rest features on top of the Elo+Platt incumbent.*\n\n"
        )

        f.write("## Feature Definitions\n\n")
        f.write("| Feature | Source | Description |\n")
        f.write("|---------|--------|-------------|\n")
        f.write("| `home_short_week` | `home_rest` | Home on short rest (≤6d) |\n")
        f.write("| `away_short_week` | `away_rest` | Away on short rest (≤6d) |\n")
        f.write("| `home_off_bye` | `home_rest` | Home off bye (≥13d rest) |\n")
        f.write("| `away_off_bye` | `away_rest` | Away off bye (≥13d rest) |\n")
        f.write("| `thursday_flag` | `weekday` | Thursday game |\n")
        f.write("| `monday_flag` | `weekday` | Monday game |\n")
        f.write("| `is_neutral` | `location` | Neutral site |\n")
        f.write("| `is_international` | `stadium_id` | Outside US |\n")
        f.write(
            "| `home_consecutive_road` | chronological | Home consecutive road/neutral games |\n"
        )
        f.write("| `away_consecutive_road` | chronological | Away consecutive road games |\n\n")

        f.write("## Incumbent Elo Params\n\n")
        f.write(f"- K={BEST_K}, HFA={BEST_HFA}, preseason regression={BEST_REG}\n\n")

        f.write("## Data Split\n\n")
        f.write("| Split | Seasons | Description |\n")
        f.write("|-------|---------|-------------|\n")
        for idx, (train_s, val_s) in enumerate(ROLLING_FOLDS, 1):
            f.write(f"| Fold {idx} | Train: {train_s}, Val: {val_s} | Rolling-origin selection |\n")
        f.write(f"| Holdout | {HOLDOUT_SEASON} | One-shot final evaluation |\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write(
            "- Elo features computed chronologically across all seasons.\n"
            "- Scheduling features computed chronologically in a single pass.\n"
            "- Consecutive road game counts reflect the streak **before** each game.\n"
            "- Rest days (`home_rest`, `away_rest`) are provided by nflreadpy as pregame\n"
            "  data (days since each team's prior game).\n"
            "- Day-of-week flags are determined solely from the `weekday` column\n"
            "  (known at schedule release).\n"
            "- International flag is based on `stadium_id` (known at schedule release).\n"
            "- Rolling-origin folds ensure no 2025 data touches model selection.\n\n"
        )

        f.write("## Average Validation Metrics Across Folds\n\n")
        f.write("| Model | Avg Val LL | Fold1 LL | Fold2 LL | Fold3 LL |\n")
        f.write("|-------|------------|----------|----------|----------|\n")

        def _fold_ll_row(name, results):
            lls = [r["log_loss"] for r in results]
            avg = np.mean(lls)
            return f"| {name} | {avg:.4f} | {lls[0]:.4f} | {lls[1]:.4f} | {lls[2]:.4f} |\n"

        f.write(_fold_ll_row("Raw Elo", elo_only_results))
        f.write(_fold_ll_row("Platt (incumbent)", platt_results))
        f.write(_fold_ll_row("Incumbent + Scheduling", incumbent_plus_sched_results))
        f.write(_fold_ll_row("Raw Elo + Scheduling", raw_elo_sched_results))
        f.write(_fold_ll_row("Scheduling only", sched_only_results))
        f.write("\n")

        f.write("## Full Comparison (2025 Holdout)\n\n")
        header = "| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n"
        sep = "|-------|---------|------------|----------|----------|\n"
        f.write(header)
        f.write(sep)
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(_row("Raw Elo", hold_raw_metrics))
        f.write(_row("Platt (incumbent)", hold_platt_metrics))
        f.write(_row("Incumbent + Scheduling", hold_ip_sched_metrics))
        f.write(_row("Raw Elo + Scheduling", hold_re_sched_metrics))
        f.write(_row("Scheduling only", hold_sched_only_metrics))
        f.write("\n")

        # ── Calibration buckets ──
        for label, h_met in [
            ("Incumbent (Platt, Holdout)", hold_platt_metrics),
            ("Incumbent + Scheduling (Holdout)", hold_ip_sched_metrics),
            ("Raw Elo + Scheduling (Holdout)", hold_re_sched_metrics),
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

        # ── Recommendation ──
        f.write("## Recommendation\n\n")
        hold_metrics = {
            "Raw Elo": hold_raw_metrics["log_loss"],
            "Platt (incumbent)": hold_platt_metrics["log_loss"],
            "Incumbent + Scheduling": hold_ip_sched_metrics["log_loss"],
            "Raw Elo + Scheduling": hold_re_sched_metrics["log_loss"],
            "Scheduling only": hold_sched_only_metrics["log_loss"],
        }

        # Challengers only (scheduling-augmented models)
        challenger_val = {
            "Incumbent + Scheduling": avg_ip_sched,
            "Raw Elo + Scheduling": avg_re_sched,
            "Scheduling only": avg_sched_only,
        }
        best_chal_name = min(challenger_val, key=challenger_val.get)
        best_chal_val_ll = challenger_val[best_chal_name]

        incumbent_val_ll = avg_platt
        improved = best_chal_val_ll < incumbent_val_ll - 0.001

        if improved:
            f.write(
                f"✅ **{best_chal_name} won across rolling validation and"
                f" is the new research incumbent.**\n\n"
            )
            hold_ll = hold_metrics[best_chal_name]
            f.write(
                f"Average validation log loss {best_chal_val_ll:.4f} beats"
                f" the Platt incumbent ({incumbent_val_ll:.4f})."
                f" Holdout log loss: {hold_ll:.4f} (Platt: {hold_platt_metrics['log_loss']:.4f}).\n"
            )
            if best_chal_name == "Incumbent + Scheduling":
                f.write(
                    "Adding scheduling to Platt-calibrated Elo improved predictive performance.\n"
                )
            else:
                f.write("Scheduling features provided an improvement over the incumbent.\n")
        else:
            f.write("⚠️ **Platt incumbent remains the research incumbent.**\n\n")
            f.write(
                f"Average validation log loss: Platt={incumbent_val_ll:.4f},"
                f" best challenger={best_chal_name} ({best_chal_val_ll:.4f}).  "
            )
            f.write("No challenger achieved meaningfully lower validation log loss.\n\n")
            f.write(
                "Scheduling features did not provide a material improvement"
                " over the Elo+Platt baseline on this dataset.\n"
            )

        f.write("\n### Next Recommended Experiment\n\n")
        f.write("1. Add weather features (temp, wind, precipitation).\n")
        f.write("2. Test a GradientBoosting model with Elo + scheduling + weather.\n")
        f.write("3. Explore advanced team metrics (DVOA, EPA).\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
