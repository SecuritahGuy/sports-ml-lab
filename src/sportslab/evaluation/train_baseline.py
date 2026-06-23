"""Train NFL home-win baseline with configurable feature set."""

from pathlib import Path

import numpy as np
import pandas as pd

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    BASELINE_FEATURE_COLUMNS,
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
    TEAM_STRENGTH_FEATURE_COLUMNS,
)
from sportslab.features.ratings import compute_elo_features, elo_diff_to_win_prob
from sportslab.features.rolling import compute_rolling_features
from sportslab.models.logistic import build_baseline_pipeline

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASON = 2024
HOLDOUT_SEASON = 2025

FEATURE_SETS = {
    "baseline": {
        "columns": BASELINE_FEATURE_COLUMNS,
        "needs_ratings": False,
        "label": "Baseline Logistic (identity features)",
        "file_suffix": "first_logistic_baseline",
    },
    "team_strength": {
        "columns": TEAM_STRENGTH_FEATURE_COLUMNS,
        "needs_ratings": True,
        "label": "Team Strength Logistic (Elo + rolling)",
        "file_suffix": "team_strength_logistic_baseline",
    },
}

EXCLUDED_REASONS: dict[str, list[str]] = {
    "leakage (score/result/overtime)": [
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
    "weather (deferred)": [
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
    "raw string columns": [
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
    "target / flags": [TARGET_COLUMN, MODEL_ELIGIBLE_COLUMN, "is_tie", NEUTRAL_COLUMN],
}


def _compute_team_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo and rolling features on the full dataframe (leakage-safe).

    Processes games in chronological order so features for game N only
    depend on games 0..N-1.

    Args:
        df: Full feature table (including ties and neutral games).

    Returns:
        DataFrame with added Elo and rolling feature columns.
    """
    df = compute_elo_features(df)
    df = compute_rolling_features(df)
    return df


def _simple_baselines(
    df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    holdout_mask: pd.Series,
) -> dict:
    """Compute comparison baseline metrics.

    Returns dict with keys "random", "home_prior", "elo_only", each containing
    a dict of metric_name -> val/holdout metrics.
    """
    y_val = df.loc[val_mask, TARGET_COLUMN].astype(float).values
    y_holdout = df.loc[holdout_mask, TARGET_COLUMN].astype(float).values

    results = {}

    # 1) Random baseline: always predict 0.5
    results["random"] = {
        "val": compute_classification_metrics(y_val, np.full(len(y_val), 0.5)),
        "holdout": compute_classification_metrics(y_holdout, np.full(len(y_holdout), 0.5)),
    }

    # 2) Home prior: historical home win rate on training data
    train_home_rate = df.loc[train_mask, TARGET_COLUMN].astype(float).mean()
    results["home_prior"] = {
        "val": compute_classification_metrics(y_val, np.full(len(y_val), train_home_rate)),
        "holdout": compute_classification_metrics(
            y_holdout, np.full(len(y_holdout), train_home_rate)
        ),
    }

    # 3) Elo-only: Elo-diff → win probability (no logistic regression)
    if "elo_diff" in df.columns:
        elo_val = df.loc[val_mask, "elo_diff"].values
        elo_holdout = df.loc[holdout_mask, "elo_diff"].values
        results["elo_only"] = {
            "val": compute_classification_metrics(
                y_val, np.array([elo_diff_to_win_prob(max(e, 0), max(-e, 0)) for e in elo_val])
            ),
            "holdout": compute_classification_metrics(
                y_holdout,
                np.array([elo_diff_to_win_prob(max(e, 0), max(-e, 0)) for e in elo_holdout]),
            ),
        }

    return results


def _format_metrics_table(metrics_dict: dict) -> str:
    """Format a metrics dict as a markdown table row."""
    m = metrics_dict
    parts = [
        f"{m['log_loss']:.4f}",
        f"{m['brier_score']:.4f}",
        f"{m['accuracy']:.4f}",
    ]
    if m["roc_auc"] is not None:
        parts.append(f"{m['roc_auc']:.4f}")
    else:
        parts.append("N/A")
    return " | ".join(parts)


def train_baseline(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    feature_set: str = "baseline",
) -> str:
    """Train a logistic regression baseline with the specified feature set.

    Splits by season, computes Elo/rolling features if needed, fits a
    pipeline, evaluates on validation (2024) and holdout (2025), compares
    against simple baselines (random, home-prior, Elo-only), and writes a
    markdown report.

    Args:
        feature_table_path: Path to the feature table parquet.
        feature_set: One of "baseline" or "team_strength".

    Returns:
        The report path written.

    Raises:
        FileNotFoundError: If the feature table is missing.
        ValueError: If the feature set is unknown or required columns are
            missing.
    """
    if feature_set not in FEATURE_SETS:
        known = list(FEATURE_SETS)
        raise ValueError(f"Unknown feature_set '{feature_set}'. Known: {known}")

    cfg = FEATURE_SETS[feature_set]
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df = pd.read_parquet(fp)

    # --- Compute Elo / rolling features on full dataset (leakage-safe) ---
    if cfg["needs_ratings"]:
        print("Computing Elo and rolling features (chronological, no leakage)...")
        before_cols = set(df.columns)
        df = _compute_team_strength_features(df)
        new_cols = set(df.columns) - before_cols
        print(f"  Added {len(new_cols)} feature columns: {sorted(new_cols)}")

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
    train_mask = df["season"].isin(TRAIN_SEASONS)
    val_mask = df["season"] == VAL_SEASON
    holdout_mask = df["season"] == HOLDOUT_SEASON

    train_df = df[train_mask]
    val_df = df[val_mask]
    holdout_df = df[holdout_mask]

    for name, subset in [("Train", train_df), ("Validation", val_df), ("Holdout", holdout_df)]:
        if subset.empty:
            raise ValueError(f"No rows for season(s) in {name} set")

    print(f"  Train:     {len(train_df)} rows (seasons {TRAIN_SEASONS})")
    print(f"  Validation: {len(val_df)} rows (season {VAL_SEASON})")
    print(f"  Holdout:    {len(holdout_df)} rows (season {HOLDOUT_SEASON})")

    # --- Feature / target split ---
    feature_cols = [c for c in cfg["columns"] if c in df.columns]
    missing = [c for c in cfg["columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {cfg['label']} feature columns in table: {missing}")

    x_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLUMN].astype(int)
    x_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df[TARGET_COLUMN].astype(int)

    # --- Train ---
    print(f"\nTraining {cfg['label']} pipeline...")
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

    # --- Leakage check ---
    leakage_pass = _leakage_check(pipeline, feature_cols)

    # --- Comparison baselines ---
    baselines = _simple_baselines(df, train_mask, val_mask, holdout_mask)
    for bl_name, bl_data in baselines.items():
        print(
            f"  {bl_name}: val log loss {bl_data['val']['log_loss']:.4f},"
            f" holdout {bl_data['holdout']['log_loss']:.4f}"
        )

    # --- Write report ---
    report_path = f"reports/experiments/{cfg['file_suffix']}.md"
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    coeffs = pipeline.named_steps["classifier"].coef_[0]
    coef_summary = dict(zip(feature_cols, [round(c, 4) for c in coeffs]))

    _write_report(
        rp,
        cfg["label"],
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
        baselines,
    )

    print(f"\nReport written to: {rp}")
    return str(rp)


def _calibration_notes(val_metrics: dict, holdout_metrics: dict) -> str:
    """Generate calibration notes from metrics."""
    notes = []
    for label, m in [("Validation", val_metrics), ("Holdout", holdout_metrics)]:
        buckets = m["calibration_buckets"]
        if not buckets:
            notes.append(f"{label}: no calibration buckets.")
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
    flagged = [f"{name}: {c:.2f}" for name, c in top if abs(c) > 5]
    if flagged:
        return f"⚠️ CHECK LEAKAGE: features with unusually large coefficients (>5): {flagged}"
    return "✅ No suspiciously large coefficients detected."


def _write_report(
    path: Path,
    label: str,
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
    baselines: dict,
) -> None:
    """Write the experiment report markdown file."""
    with open(path, "w") as f:
        f.write(f"# {label}\n\n")
        f.write("*Pregame-only NFL home-win baseline.*\n\n")

        f.write("## Data Split\n\n")
        f.write("| Split | Seasons | Rows |\n")
        f.write("|-------|---------|------|\n")
        f.write(f"| Train | {train_seasons} | {n_train} |\n")
        f.write(f"| Validation | {val_season} | {n_val} |\n")
        f.write(f"| Holdout | {holdout_season} | {n_holdout} |\n\n")

        f.write("## Filtering Applied\n\n")
        f.write("- `model_eligible == True` (ties excluded)\n")
        f.write("- `is_neutral == False` (neutral-site games excluded)\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write(
            "- Elo features: ratings updated **after** computing pregame "
            "features for each game. Games processed chronologically.\n"
        )
        f.write(
            "- Rolling features: computed from games **before** the current "
            "game only. The current game's result is never included.\n"
        )
        f.write("- All features are pregame-safe: no future information leaks into any row.\n\n")

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

        for label_name, metrics in [
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Holdout", holdout_metrics),
        ]:
            f.write(f"## {label_name} Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Log loss | {metrics['log_loss']:.4f} |\n")
            f.write(f"| Brier score | {metrics['brier_score']:.4f} |\n")
            f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
            roc = metrics["roc_auc"]
            if roc is not None:
                f.write(f"| ROC AUC | {roc:.4f} |\n\n")
            else:
                f.write("| ROC AUC | N/A (single class in split) |\n\n")

            f.write("### Calibration Buckets\n\n")
            f.write("| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |\n")
            f.write("|--------|-------|----------------|-------------|-------------------|\n")
            for bucket_label, b in sorted(metrics["calibration_buckets"].items()):
                f.write(
                    f"| {bucket_label} | {b['count']} |"
                    f" {b['mean_predicted_prob']} | {b['mean_actual_rate']}"
                    f" | {b['calibration_error']} |\n"
                )
            f.write("\n")

        # --- Comparison baselines ---
        f.write("## Comparison Baselines\n\n")
        f.write(
            "| Baseline | Val LL | Val Brier | Val Acc | Val AUC |"
            " Hold LL | Hold Brier | Hold Acc | Hold AUC |\n"
        )
        f.write(
            "|----------|-------------|----------|---------|---------|---------|------------|----------|----------|\n"
        )

        for bl_name, bl_data in baselines.items():
            v = _format_metrics_table(bl_data["val"])
            h = _format_metrics_table(bl_data["holdout"])
            f.write(f"| {bl_name} | {v} | {h} |\n")

        # Logistic model row
        v = _format_metrics_table(val_metrics)
        h = _format_metrics_table(holdout_metrics)
        f.write(f"| **Logistic ({label})** | {v} | {h} |\n\n")

        f.write("## Calibration Notes\n\n")
        f.write(f"{calibration_notes}\n\n")

        f.write("## Leakage Check\n\n")
        f.write(f"{leakage_check}\n\n")

        # Recommendation
        holdout_ll = holdout_metrics["log_loss"]
        best_bl_holdout = min(bl_data["holdout"]["log_loss"] for bl_data in baselines.values())
        improvement = best_bl_holdout - holdout_ll

        if holdout_ll < 0.65 and improvement > 0.01:
            recommendation = (
                "✅ **Accept as baseline champion.** "
                f"Holdout log loss {holdout_ll:.4f} beats the best simple "
                f"baseline ({best_bl_holdout:.4f}) by {improvement:.4f}. "
                "Proceed to iterate with weather, interaction features, "
                "or more expressive models."
            )
        elif holdout_ll < best_bl_holdout:
            recommendation = (
                "⚠️ **Marginal improvement over simple baselines.** "
                f"Holdout log loss {holdout_ll:.4f} vs best simple "
                f"baseline {best_bl_holdout:.4f} (Δ={improvement:.4f}). "
                "Consider adding weather features, team-strength ratings "
                "with different K-factors, or rolling windows."
            )
        else:
            recommendation = (
                "❌ **Do not champion.** "
                f"Holdout log loss {holdout_ll:.4f} does not beat the best "
                f"simple baseline ({best_bl_holdout:.4f}). "
                "Consider adding weather features, team-strength ratings "
                "with different K-factors, or switching to a more expressive "
                "model (GradientBoosting, RandomForest, AutoGluon)."
            )
        f.write("## Recommendation\n\n")
        f.write(f"{recommendation}\n")
