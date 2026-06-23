"""AutoGluon AutoML experiment — rolling-origin vs O/D Elo+Platt incumbent.

Compares AutoGluon (with all available pregame features) against the
research incumbent.  Features include O/D Elo ratings, scheduling flags,
QB continuity features, weather flags, and basic game context.

AutoGluon variant is also run on Elo-only features for an apples-to-apples
comparison against Platt calibration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
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
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_od_elo_features
from sportslab.features.scheduling import compute_scheduling_features
from sportslab.features.weather import compute_weather_features

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

AG_TIME_LIMIT_PER_FOLD = 1800  # 30 minutes per fold

ELO_ONLY_FEATURES = [
    "home_off_elo", "away_off_elo", "home_def_elo", "away_def_elo",
    "elo_diff", "elo_prob",
]

SCHEDULING_FEATURES = [
    "home_short_week", "away_short_week", "home_off_bye", "away_off_bye",
    "thursday_flag", "monday_flag", "is_international",
    "home_consecutive_road", "away_consecutive_road",
]

QB_FLAG_FEATURES = [
    "home_qb_changed", "away_qb_changed", "qb_change_diff",
    "home_qb_starts_this_season_pre", "away_qb_starts_this_season_pre",
    "qb_starts_diff", "home_qb_team_starts_pre", "away_qb_team_starts_pre",
    "home_qb_win_pct_pre", "away_qb_win_pct_pre", "qb_win_pct_diff",
    "home_games_since_qb_change", "away_games_since_qb_change",
    "games_since_qb_change_diff", "home_new_qb_flag", "away_new_qb_flag",
    "new_qb_diff", "home_qb_missing_flag", "away_qb_missing_flag",
]

WEATHER_FEATURES = [
    "temperature_f", "wind_mph", "precipitation_flag",
    "cold_flag", "very_cold_flag", "hot_flag", "windy_flag",
    "very_windy_flag", "bad_weather_flag", "outdoor_game_flag",
    "weather_missing_flag", "temp_missing_flag", "wind_missing_flag",
]

BASIC_FEATURES = [
    "week", "rest_diff", "div_game", "is_dome", "is_neutral",
    "game_type_enc", "roof_enc", "surface_enc", "weekday_enc",
    "home_team_enc", "away_team_enc", "home_coach_enc", "away_coach_enc",
]


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


def _compute_all_features(
    df: pd.DataFrame,
    team_overrides: dict[str, float] | None,
) -> pd.DataFrame:
    """Compute all pregame-safe features into a single dataframe.

    Returns feature columns only (no leakage or raw identifiers).
    """
    out = df.copy()

    # 1. O/D Elo features
    elo_df = compute_od_elo_features(
        out, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE, mov_scale=BEST_MOV_SCALE, mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        k_off=BEST_K_OFF, k_def=BEST_K_DEF,
        team_regression_overrides=team_overrides,
    )
    for c in ELO_ONLY_FEATURES:
        out[c] = elo_df[c]

    # 2. Scheduling features
    sched_df = compute_scheduling_features(out)
    for c in SCHEDULING_FEATURES:
        if c in sched_df.columns:
            out[c] = sched_df[c]

    # 3. QB features (flags only, no identity OHE)
    qb_df = compute_qb_features(out)
    for c in QB_FLAG_FEATURES:
        if c in qb_df.columns:
            out[c] = qb_df[c]

    # 4. Weather features (if weather columns exist in the feature table)
    weather_cols = ["weather_tmin", "weather_tmax", "weather_wind_speed", "weather_precip"]
    if all(c in out.columns for c in weather_cols):
        weather_df = compute_weather_features(out)
        for c in WEATHER_FEATURES:
            if c in weather_df.columns:
                out[c] = weather_df[c]

    return out


def _available_features(df: pd.DataFrame) -> list[str]:
    """Return list of actually-available feature columns in df."""
    all_feature_candidates = (
        ELO_ONLY_FEATURES
        + SCHEDULING_FEATURES
        + QB_FLAG_FEATURES
        + BASIC_FEATURES
    )

    # Check for weather columns
    weather_cols = ["weather_tmin", "weather_tmax", "weather_wind_speed", "weather_precip"]
    if all(c in df.columns for c in weather_cols):
        all_feature_candidates += WEATHER_FEATURES

    return [c for c in all_feature_candidates if c in df.columns]


def run_autogluon_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/autogluon.md",
) -> str:
    """Run AutoGluon experiment with rolling-origin validation.

    1. Compute all features globally on the full 2021-2024 dataset.
    2. Rolling-origin evaluation for each fold.
    3. One-time 2025 holdout evaluation.
    4. Compare AutoGluon (full features) vs AutoGluon (Elo only) vs Platt incumbent.
    5. Report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)
    print(f"Loaded {len(df_raw)} rows from {fp}")

    # Build team regression overrides from all data (no holdout leak)
    team_overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
    )

    # Compute all features on the full dataset
    print("\n=== Computing all features ===")
    df_all = _compute_all_features(df_raw, team_overrides)
    df_all = _filter_df(df_all)
    feature_cols = _available_features(df_all)
    print(f"  Available features: {len(feature_cols)}")
    print(f"  Feature columns: {feature_cols}")
    print(f"  Feature set shape: {df_all.shape}")

    # Separate holdout (2025)
    hold_mask = df_all["season"] == HOLDOUT_SEASON
    df_hold = df_all[hold_mask].copy().reset_index(drop=True)
    df_train_val = df_all[~hold_mask].copy().reset_index(drop=True)
    print(f"\n  Train/val rows: {len(df_train_val)}")
    print(f"  Holdout rows:    {len(df_hold)}")

    y_hold = df_hold[TARGET_COLUMN].astype(float).values

    # === Rolling-origin validation ===
    fold_results: dict[str, list[float]] = {
        "platt": [],
        "ag_full": [],
        "ag_elo": [],
    }
    fold_models: dict[str, list] = {
        "platt": [],
        "ag_full": [],
        "ag_elo": [],
    }

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}: train {train_seasons} → val {val_season}")
        print(f"{'='*60}")

        train_mask = df_train_val["season"].isin(train_seasons)
        val_mask = df_train_val["season"] == val_season

        df_train = df_train_val[train_mask].reset_index(drop=True)
        df_val = df_train_val[val_mask].reset_index(drop=True)

        x_train = df_train[feature_cols].copy()
        y_train = df_train[TARGET_COLUMN].astype(float).values
        x_val = df_val[feature_cols].copy()
        y_val = df_val[TARGET_COLUMN].astype(float).values

        x_train_elo = df_train[ELO_ONLY_FEATURES].copy()
        x_val_elo = df_val[ELO_ONLY_FEATURES].copy()

        print(f"  Train: {len(x_train)} rows, Val: {len(x_val)} rows")
        print(f"  Features: {len(feature_cols)}")

        # === Incumbent: Platt-calibrated O/D Elo ===
        elo_prob_train = df_train["elo_prob"].values.astype(float)
        elo_prob_val = df_val["elo_prob"].values.astype(float)
        platt = _fit_platt(elo_prob_train, y_train)
        platt_val_prob = platt.predict_proba(elo_prob_val.reshape(-1, 1))[:, 1]
        platt_val_ll = compute_classification_metrics(y_val, platt_val_prob)["log_loss"]
        fold_results["platt"].append(platt_val_ll)
        fold_models["platt"].append(platt)
        print(f"  Platt (incumbent) val LL: {platt_val_ll:.4f}")

        # === AutoGluon (full features) ===
        ag_full = TabularPredictor(
            label=TARGET_COLUMN,
            problem_type="binary",
            eval_metric="log_loss",
        )
        ag_full.fit(
            train_data=pd.concat([x_train, pd.Series(y_train, name=TARGET_COLUMN)], axis=1),
            presets="medium_quality",
            time_limit=AG_TIME_LIMIT_PER_FOLD,
            verbosity=1,
        )
        ag_full_val_prob = ag_full.predict_proba(x_val)[1]
        ag_full_val_ll = compute_classification_metrics(y_val, ag_full_val_prob)["log_loss"]
        fold_results["ag_full"].append(ag_full_val_ll)
        fold_models["ag_full"].append(ag_full)
        print(f"  AutoGluon (full) val LL: {ag_full_val_ll:.4f}")

        # === AutoGluon (Elo only) ===
        ag_elo = TabularPredictor(
            label=TARGET_COLUMN,
            problem_type="binary",
            eval_metric="log_loss",
        )
        ag_elo.fit(
            train_data=pd.concat([x_train_elo, pd.Series(y_train, name=TARGET_COLUMN)], axis=1),
            presets="medium_quality",
            time_limit=AG_TIME_LIMIT_PER_FOLD,
            verbosity=1,
        )
        ag_elo_val_prob = ag_elo.predict_proba(x_val_elo)[1]
        ag_elo_val_ll = compute_classification_metrics(y_val, ag_elo_val_prob)["log_loss"]
        fold_results["ag_elo"].append(ag_elo_val_ll)
        fold_models["ag_elo"].append(ag_elo)
        print(f"  AutoGluon (Elo only) val LL: {ag_elo_val_ll:.4f}")

    # === Average validation results ===
    print(f"\n{'='*60}")
    print("Rolling-Origin Validation Summary")
    print(f"{'='*60}")
    for label in ["platt", "ag_full", "ag_elo"]:
        avg = np.mean(fold_results[label])
        print(f"  {label}: avg val LL = {avg:.4f}"
                  f"  folds={fold_results[label]}")

    # === One-shot 2025 holdout ===
    print(f"\n{'='*60}")
    print("2025 Holdout Evaluation")
    print(f"{'='*60}")

    hold_features = df_hold[feature_cols].copy()
    hold_elo_features = df_hold[ELO_ONLY_FEATURES].copy()
    elo_prob_hold = df_hold["elo_prob"].values.astype(float)

    hold_metrics: dict[str, dict] = {}

    # Platt incumbent
    platt_hold_prob = fold_models["platt"][-1].predict_proba(elo_prob_hold.reshape(-1, 1))[:, 1]
    hold_metrics["platt"] = compute_classification_metrics(y_hold, platt_hold_prob)
    print(f"  Platt (incumbent): holdout LL = {hold_metrics['platt']['log_loss']:.4f}")

    # AutoGluon (full features) - retrain on all 2021-2024
    print("\n=== Retraining AutoGluon (full features) on all 2021-2024 ===")
    x_all = df_train_val[feature_cols].copy()
    y_all = df_train_val[TARGET_COLUMN].astype(float).values
    ag_full_final = TabularPredictor(
        label=TARGET_COLUMN,
        problem_type="binary",
        eval_metric="log_loss",
    )
    ag_full_final.fit(
        train_data=pd.concat([x_all, pd.Series(y_all, name=TARGET_COLUMN)], axis=1),
        presets="medium_quality",
        time_limit=AG_TIME_LIMIT_PER_FOLD,
        verbosity=1,
    )
    ag_full_hold_prob = ag_full_final.predict_proba(hold_features)[1]
    hold_metrics["ag_full"] = compute_classification_metrics(y_hold, ag_full_hold_prob)
    print(f"  AutoGluon (full): holdout LL = {hold_metrics['ag_full']['log_loss']:.4f}")

    # AutoGluon (Elo only) - retrain on all 2021-2024
    print("\n=== Retraining AutoGluon (Elo only) on all 2021-2024 ===")
    x_all_elo = df_train_val[ELO_ONLY_FEATURES].copy()
    ag_elo_final = TabularPredictor(
        label=TARGET_COLUMN,
        problem_type="binary",
        eval_metric="log_loss",
    )
    ag_elo_final.fit(
        train_data=pd.concat([x_all_elo, pd.Series(y_all, name=TARGET_COLUMN)], axis=1),
        presets="medium_quality",
        time_limit=AG_TIME_LIMIT_PER_FOLD,
        verbosity=1,
    )
    ag_elo_hold_prob = ag_elo_final.predict_proba(hold_elo_features)[1]
    hold_metrics["ag_elo"] = compute_classification_metrics(y_hold, ag_elo_hold_prob)
    print(f"  AutoGluon (Elo only): holdout LL = {hold_metrics['ag_elo']['log_loss']:.4f}")

    # Also try Platt calibration on top of AutoGluon outputs
    for variant in ["ag_full", "ag_elo"]:
        ag_train_prob = np.asarray(fold_models[variant][-1].predict_proba(
            x_all if variant == "ag_full" else x_all_elo
        )[1])
        ag_platt = _fit_platt(ag_train_prob, y_all)
        ag_platt_hold = ag_platt.predict_proba(
            np.asarray(
                ag_full_hold_prob if variant == "ag_full" else ag_elo_hold_prob
            ).reshape(-1, 1)
        )[:, 1]
        key = f"{variant}_platt"
        hold_metrics[key] = compute_classification_metrics(y_hold, ag_platt_hold)
        print(f"  {variant} + Platt: holdout LL = {hold_metrics[key]['log_loss']:.4f}")

    # === Write report ===
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    # Feature importance from final AutoGluon model
    try:
        feat_imp = ag_full_final.feature_importance(hold_features, silent=True)
    except Exception:
        feat_imp = None

    with open(rp, "w") as f:
        f.write("# AutoGluon AutoML Experiment\n\n")
        f.write("*Testing whether AutoGluon (with all pregame features)"
                " beats O/D Elo+Platt incumbent.*\n\n")
        f.write("## Method\n\n")
        f.write("Rolling-origin 3-fold validation, one-shot 2025 holdout.\n\n")

        f.write("### Competing Models\n\n")
        f.write("| Model | Description |\n")
        f.write("|------|------------|\n")
        f.write("| **Platt (incumbent)** | O/D Elo (ko52_kd20) + logistic calibration |\n")
        f.write("| **AutoGluon (full)** | All pregame features + AutoGluon medium_quality |\n")
        f.write("| **AutoGluon (Elo only)** | 6 O/D Elo features only + AutoGluon |\n")
        f.write("| **AutoGluon + Platt** | AutoGluon outputs recalibrated with Platt |\n\n")

        f.write("### AutoGluon Configuration\n\n")
        f.write("| Setting | Value |\n")
        f.write("|--------|-------|\n")
        f.write("| presets | medium_quality |\n")
        f.write("| eval_metric | log_loss |\n")
        f.write("| problem_type | binary |\n")
        f.write(f"| time_limit_per_fold | {AG_TIME_LIMIT_PER_FOLD}s |\n")
        f.write(f"| total_folds | {len(ROLLING_FOLDS)} |\n\n")

        f.write(f"### Feature Set ({len(feature_cols)} features)\n\n")
        for grp_name, grp_cols in [
            ("O/D Elo ratings", ELO_ONLY_FEATURES),
            ("Scheduling", SCHEDULING_FEATURES),
            ("QB flags", QB_FLAG_FEATURES),
            ("Basic context", BASIC_FEATURES),
        ]:
            available = [c for c in grp_cols if c in feature_cols]
            f.write(f"- **{grp_name}** ({len(available)}): {', '.join(available)}\n")
        # Weather if available
        weather_avail = [c for c in WEATHER_FEATURES if c in feature_cols]
        if weather_avail:
            f.write(f"- **Weather** ({len(weather_avail)}): {', '.join(weather_avail)}\n")
        f.write("\n")

        # Validation table
        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|-----------|-------|-------|-------|\n")
        for label, name in [("platt", "Platt (incumbent)"),
                            ("ag_full", "AutoGluon (full)"),
                            ("ag_elo", "AutoGluon (Elo only)")]:
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
            ("ag_full", "AutoGluon (full)"),
            ("ag_elo", "AutoGluon (Elo only)"),
            ("ag_full_platt", "AutoGluon (full) + Platt"),
            ("ag_elo_platt", "AutoGluon (Elo only) + Platt"),
        ]:
            if label in hold_metrics:
                hm = hold_metrics[label]
                f.write(f"| {name} | {hm['log_loss']:.4f}")
                f.write(f" | {hm['brier_score']:.4f}")
                f.write(f" | {hm['roc_auc']:.4f}")
                f.write(f" | {hm['accuracy']:.4f}")
                f.write(" |\n")
        f.write("\n")

        # Feature importance (top 20)
        if feat_imp is not None and not feat_imp.empty:
            f.write("## Feature Importance (AutoGluon full, top 20)\n\n")
            f.write("| Feature | Importance |\n")
            f.write("|---------|-----------|\n")
            top20 = feat_imp.head(20)
            for feat, row in top20.iterrows():
                imp = row.get("importance", 0)
                f.write(f"| {feat} | {imp:.4f} |\n")
            f.write("\n")

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
            f.write("AutoGluon did not improve on simple Platt calibration on this dataset.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)


if __name__ == "__main__":
    run_autogluon_experiment()
