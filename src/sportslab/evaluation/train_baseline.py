"""Train the first pure non-market NFL home-win baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    BASELINE_FEATURE_COLUMNS,
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.models.logistic import build_baseline_pipeline

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASON = 2024
HOLDOUT_SEASON = 2025

EXCLUDED_REASONS = {
    "leakage (score/result/overtime/target flag)": [
        "away_score",
        "home_score",
        "result",
        "total",
        "overtime",
    ],
    "market / odds": [
        "away_moneyline",
        "home_moneyline",
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "total_line",
        "under_odds",
        "over_odds",
    ],
    "weather (deferred to iteration 2)": [
        "weather_temp",
        "weather_tmin",
        "weather_tmax",
        "weather_humidity",
        "weather_precip",
        "weather_wind_speed",
        "weather_pressure",
        "weather_cloud_cover",
    ],
    "row identifiers": [
        "game_id",
        "gameday",
        "gametime",
        "stadium",
        "old_game_id",
        "gsis",
        "nfl_detail_id",
        "pfr",
        "pff",
        "espn",
        "ftn",
    ],
    "raw string (encoded version used)": [
        "away_team",
        "home_team",
        "away_qb_id",
        "home_qb_id",
        "away_qb_name",
        "home_qb_name",
        "away_coach",
        "home_coach",
        "referee",
        "stadium_id",
        "game_type",
        "weekday",
        "roof",
        "surface",
        "location",
    ],
    "target / flag columns": [TARGET_COLUMN, MODEL_ELIGIBLE_COLUMN, "is_tie", NEUTRAL_COLUMN],
}


def train_baseline(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/first_logistic_baseline.md",
) -> str:
    """Train the first pure non-market logistic regression baseline.

    Splits by season, excludes neutral-site games and ties, fits a pipeline,
    reports metrics on validation (2024) and holdout (2025), and writes a
    markdown report.

    Args:
        feature_table_path: Path to the feature table parquet.
        report_path: Path to write the experiment report.

    Returns:
        The report path written.

    Raises:
        FileNotFoundError: If the feature table is missing.
        ValueError: If required columns are missing.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df = pd.read_parquet(fp)

    # --- Filtering ---
    before = len(df)
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    print(f"  model_eligible filter: {before} -> {len(df)} (ties removed)")

    initial_neutral = df[NEUTRAL_COLUMN].sum()
    if initial_neutral > 0:
        before_neutral = len(df)
        df = df[~df[NEUTRAL_COLUMN]].copy()
        print(
            f"  neutral-site filter: {before_neutral} -> {len(df)}"
            f" ({int(initial_neutral)} neutral games removed)"
        )

    # --- Split by season ---
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"] == VAL_SEASON]
    holdout_df = df[df["season"] == HOLDOUT_SEASON]

    for name, subset in [("Train", train_df), ("Validation", val_df), ("Holdout", holdout_df)]:
        if subset.empty:
            raise ValueError(f"No rows for season(s) in {name} set")

    print(f"  Train:     {len(train_df)} rows (seasons {TRAIN_SEASONS})")
    print(f"  Validation: {len(val_df)} rows (season {VAL_SEASON})")
    print(f"  Holdout:    {len(holdout_df)} rows (season {HOLDOUT_SEASON})")

    # --- Feature / target split ---
    feature_cols = [c for c in BASELINE_FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in BASELINE_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing baseline feature columns in table: {missing}")

    x_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLUMN].astype(int)
    x_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df[TARGET_COLUMN].astype(int)

    # --- Train ---
    print("\nTraining baseline logistic regression pipeline...")
    pipeline = build_baseline_pipeline()
    pipeline.fit(x_train, y_train)

    # --- Predict ---
    train_proba = pipeline.predict_proba(x_train)[:, 1]
    val_proba = pipeline.predict_proba(x_val)[:, 1]
    holdout_proba = pipeline.predict_proba(x_holdout)[:, 1]

    # --- Evaluate ---
    train_metrics = compute_classification_metrics(y_train.values, train_proba)
    val_metrics = compute_classification_metrics(y_val.values, val_proba)
    holdout_metrics = compute_classification_metrics(y_holdout.values, holdout_proba)

    print(f"\n  Train log loss:     {train_metrics['log_loss']:.4f}")
    print(f"  Validation log loss: {val_metrics['log_loss']:.4f}")
    print(f"  Holdout log loss:    {holdout_metrics['log_loss']:.4f}")
    print(f"  Validation Brier:    {val_metrics['brier_score']:.4f}")
    print(f"  Holdout Brier:       {holdout_metrics['brier_score']:.4f}")
    print(f"  Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Holdout accuracy:    {holdout_metrics['accuracy']:.4f}")
    if val_metrics["roc_auc"] is not None:
        print(f"  Validation ROC AUC:  {val_metrics['roc_auc']:.4f}")
    if holdout_metrics["roc_auc"] is not None:
        print(f"  Holdout ROC AUC:     {holdout_metrics['roc_auc']:.4f}")

    # --- Calibration notes ---
    cal_notes = _calibration_notes(val_metrics, holdout_metrics)

    # --- Leakage check (spot-check coefficient) ---
    leakage_pass = _leakage_check(pipeline, feature_cols)

    # --- Write report ---
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    coeffs = pipeline.named_steps["classifier"].coef_[0]
    coef_summary = dict(zip(feature_cols, [round(c, 4) for c in coeffs]))

    _write_report(
        rp,
        TRAIN_SEASONS,
        VAL_SEASON,
        HOLDOUT_SEASON,
        len(train_df),
        len(val_df),
        len(holdout_df),
        feature_cols,
        coef_summary,
        train_metrics,
        val_metrics,
        holdout_metrics,
        cal_notes,
        leakage_pass,
    )

    print(f"\nReport written to: {rp}")
    return str(rp)


def _calibration_notes(val_metrics: dict, holdout_metrics: dict) -> str:
    """Generate calibration notes from metrics."""
    notes = []
    for label, m in [("Validation", val_metrics), ("Holdout", holdout_metrics)]:
        buckets = m["calibration_buckets"]
        if not buckets:
            notes.append(f"{label}: No calibration buckets (too few predictions in range?).")
            continue
        max_err = max(b["calibration_error"] for b in buckets.values())
        avg_err = np.mean([b["calibration_error"] for b in buckets.values()])
        notes.append(
            f"{label}: max decile calibration error {max_err:.4f}, mean decile error {avg_err:.4f}"
        )
    return "\n".join(notes)


def _leakage_check(pipeline, feature_cols) -> str:
    """Check for potential leakage signals in feature importance."""
    coeffs = pipeline.named_steps["classifier"].coef_[0]
    top = sorted(zip(feature_cols, coeffs), key=lambda x: abs(x[1]), reverse=True)
    # If any feature has coefficient magnitude > 5, flag it
    flagged = [f"{name}: {c:.2f}" for name, c in top if abs(c) > 5]
    if flagged:
        return f"⚠️ CHECK LEAKAGE: features with unusually large coefficients (>5): {flagged}"
    return "✅ No suspiciously large coefficients detected."


def _write_report(
    path: Path,
    train_seasons: list,
    val_season: int,
    holdout_season: int,
    n_train: int,
    n_val: int,
    n_holdout: int,
    features: list,
    coef_summary: dict,
    train_metrics: dict,
    val_metrics: dict,
    holdout_metrics: dict,
    calibration_notes: str,
    leakage_check: str,
) -> None:
    """Write the experiment report markdown file."""
    with open(path, "w") as f:
        f.write("# First Logistic Regression Baseline\n\n")
        f.write("*Pure non-market NFL home-win baseline.*\n\n")
        f.write("## Data Split\n\n")
        f.write("| Split | Seasons | Rows |\n")
        f.write("|-------|---------|------|\n")
        f.write(f"| Train | {train_seasons} | {n_train} |\n")
        f.write(f"| Validation | {val_season} | {n_val} |\n")
        f.write(f"| Holdout | {holdout_season} | {n_holdout} |\n\n")

        f.write("## Filtering Applied\n\n")
        f.write("- `model_eligible == True` (ties excluded)\n")
        f.write("- `is_neutral == False` (neutral-site games excluded)\n\n")

        f.write("## Excluded Column Groups\n\n")
        f.write("| Reason | Columns |\n")
        f.write("|--------|---------|\n")
        for reason, cols in EXCLUDED_REASONS.items():
            f.write(f"| {reason} | {', '.join(cols)} |\n")
        f.write("\n")

        f.write("## Included Features ({})\n\n".format(len(features)))
        f.write("| Feature | Coefficient |\n")
        f.write("|---------|-------------|\n")
        for col in features:
            coef = coef_summary.get(col, 0)
            f.write(f"| {col} | {coef} |\n")
        f.write("\n")

        for label, metrics in [
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Holdout", holdout_metrics),
        ]:
            f.write(f"## {label} Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Log loss | {metrics['log_loss']:.4f} |\n")
            f.write(f"| Brier score | {metrics['brier_score']:.4f} |\n")
            f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
            roc = metrics["roc_auc"]
            f.write(
                f"| ROC AUC | {roc:.4f}"
                if roc is not None
                else "| ROC AUC | N/A (single class in split)"
            )
            f.write(" |\n\n")

            f.write("### Calibration Buckets\n\n")
            f.write("| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |\n")
            f.write("|--------|-------|----------------|-------------|-------------------|\n")
            for bucket_label, b in sorted(metrics["calibration_buckets"].items()):
                f.write(
                    f"| {bucket_label} | {b['count']} | {b['mean_predicted_prob']} "
                    f"| {b['mean_actual_rate']} | {b['calibration_error']} |\n"
                )
            f.write("\n")

        f.write("## Calibration Notes\n\n")
        f.write(f"{calibration_notes}\n\n")

        f.write("## Leakage Check\n\n")
        f.write(f"{leakage_check}\n\n")

        # Recommendation
        if holdout_metrics["log_loss"] < 0.65:
            recommendation = (
                "✅ **Accept as baseline champion.** Log loss is reasonable "
                "for a first pure non-market baseline. Proceed to iterate "
                "with weather, interaction features, or more advanced models."
            )
        else:
            recommendation = (
                "⚠️ **Improve before championing.** Log loss is high. "
                "Consider adding weather features, team-strength ratings, "
                "or switching to a more expressive model "
                "(GradientBoosting, RandomForest, AutoGluon)."
            )
        f.write("## Recommendation\n\n")
        f.write(f"{recommendation}\n")
