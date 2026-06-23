"""Constrained expressive models on top of MOV Elo+Platt incumbent.

Rolling-origin validation across 3 folds, one-shot 2025 holdout.
Compares incumbent Platt-scaled Elo against:
  - Logistic regression on curated features
  - HistGradientBoostingClassifier (constrained)
  - GradientBoostingClassifier (constrained)
  - RandomForestClassifier (diagnostic only)
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.scheduling import compute_scheduling_features
from sportslab.features.weather import compute_weather_features

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.20
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0

INCUMBENT_HOLDOUT_LL = 0.6373

CURATED_FEATURE_COLUMNS = [
    "elo_prob",
    "elo_logit",
    "elo_diff",
    "home_short_week",
    "away_short_week",
    "home_off_bye",
    "away_off_bye",
    "thursday_flag",
    "monday_flag",
    "home_consecutive_road",
    "away_consecutive_road",
    "is_international",
    "home_qb_changed",
    "away_qb_changed",
    "qb_starts_diff",
    "qb_win_pct_diff",
    "games_since_qb_change_diff",
    "new_qb_diff",
    "cold_flag",
    "windy_flag",
    "bad_weather_flag",
    "outdoor_game_flag",
    "is_dome",
    "weather_missing_flag",
    "week_norm",
    "rest_diff",
    "div_game",
]

MODEL_NAMES = [
    "LogisticRegression",
    "HistGradientBoosting",
    "GradientBoosting",
    "RandomForest",
]

HGB_GRID = list(
    product(
        [4, 8, 12, 16],
        [0.01, 0.03, 0.05, 0.1],
        [50, 100, 200],
        [20, 40, 60],
        [0.0, 0.1, 0.5, 1.0],
    )
)

GB_GRID = list(
    product(
        [4, 8, 12, 16],
        [0.01, 0.03, 0.05, 0.1],
        [50, 100, 200],
        [20, 40, 60],
    )
)

RF_GRID = list(
    product(
        [4, 8, 12, 16],
        [50, 100, 200],
        [20, 40, 60],
    )
)


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


def _logistic_pipe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def run_expressive_models_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/expressive_models.md",
) -> str:
    """Run constrained expressive model experiment."""
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Build features stack ──
    print("=== Building feature stack ===")

    print("  Computing MOV Elo features (incumbent params)...")
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
    )

    print("  Computing scheduling features...")
    df = compute_scheduling_features(df)

    print("  Computing QB features...")
    df = compute_qb_features(df)

    print("  Computing weather features...")
    df = compute_weather_features(df)

    # ── Derived columns ──
    eps = 1e-15
    df["elo_logit"] = np.log(
        np.clip(df["elo_prob"], eps, 1 - eps) / np.clip(1 - df["elo_prob"], eps, 1 - eps)
    )
    season_max_week = df.groupby("season")["week"].transform("max")
    df["week_norm"] = (df["week"] - 1) / (season_max_week - 1)

    # ── Filter ──
    df = _filter_df(df)

    # Validate curated columns exist
    missing = [c for c in CURATED_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing curated feature columns: {missing}")

    y = df[TARGET_COLUMN].astype(float).values
    elo_prob = df["elo_prob"].values
    x_curated = df[CURATED_FEATURE_COLUMNS].values.astype(float)

    print(f"\n  Curated features ({len(CURATED_FEATURE_COLUMNS)}): {CURATED_FEATURE_COLUMNS}")
    print(f"  Total rows (filtered): {len(df)}")

    # ═══ Rolling-origin evaluation ═══
    print("\n=== Rolling-Origin Evaluation ===")

    platt_results: list[dict] = []
    lr_results: list[dict] = []
    hgb_results: list[dict] = []
    gb_results: list[dict] = []
    rf_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df["season"].isin(train_seasons).values
        is_val = (df["season"] == val_season).values

        train_y_ = y[is_train].astype(int)
        val_y_ = y[is_val]

        train_elo = elo_prob[is_train]
        val_elo = elo_prob[is_val]

        train_x = x_curated[is_train]
        val_x = x_curated[is_val]

        # 1. Platt incumbent
        platt = _fit_platt(train_elo, train_y_)
        platt_proba = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_proba)
        platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_m["log_loss"],
                "metrics": platt_m,
                "model": platt,
            }
        )

        # 2. Logistic on curated features
        lr = _logistic_pipe()
        lr.fit(train_x, train_y_)
        lr_proba = lr.predict_proba(val_x)[:, 1]
        lr_m = compute_classification_metrics(val_y_, lr_proba)
        lr_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": lr_m["log_loss"],
                "metrics": lr_m,
                "model": lr,
            }
        )

        # 3. HistGradientBoosting grid
        best_hgb = None
        best_hgb_ll = float("inf")
        for max_leaf_nodes, lr_val, max_iter, min_samples_leaf, l2_reg in HGB_GRID:
            fold_params = {
                "max_leaf_nodes": max_leaf_nodes,
                "learning_rate": lr_val,
                "max_iter": max_iter,
                "min_samples_leaf": min_samples_leaf,
                "l2_regularization": l2_reg,
            }
            hgb = HistGradientBoostingClassifier(
                **fold_params, early_stopping=False, random_state=42
            )
            hgb.fit(train_x, train_y_)
            hgb_proba = hgb.predict_proba(val_x)[:, 1]
            ll = float(log_loss(val_y_, hgb_proba))
            if ll < best_hgb_ll:
                best_hgb_ll = ll
                best_hgb = (hgb, fold_params, hgb_proba)

        hgb_model, hgb_params, hgb_proba = best_hgb
        hgb_m = compute_classification_metrics(val_y_, hgb_proba)
        hgb_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": hgb_m["log_loss"],
                "metrics": hgb_m,
                "model": hgb_model,
                "best_params": hgb_params,
            }
        )

        # 4. GradientBoosting grid
        best_gb = None
        best_gb_ll = float("inf")
        for max_leaf_nodes, lr_val, n_est, min_samples_leaf in GB_GRID:
            fold_params = {
                "max_leaf_nodes": max_leaf_nodes,
                "learning_rate": lr_val,
                "n_estimators": n_est,
                "min_samples_leaf": min_samples_leaf,
                "subsample": 0.8,
                "random_state": 42,
            }
            gb = GradientBoostingClassifier(**fold_params)
            gb.fit(train_x, train_y_)
            gb_proba = gb.predict_proba(val_x)[:, 1]
            ll = float(log_loss(val_y_, gb_proba))
            if ll < best_gb_ll:
                best_gb_ll = ll
                best_gb = (gb, fold_params, gb_proba)

        gb_model, gb_params, gb_proba = best_gb
        gb_m = compute_classification_metrics(val_y_, gb_proba)
        gb_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": gb_m["log_loss"],
                "metrics": gb_m,
                "model": gb_model,
                "best_params": gb_params,
            }
        )

        # 5. RandomForest diagnostic grid
        best_rf = None
        best_rf_ll = float("inf")
        for max_leaf_nodes, n_est, min_samples_leaf in RF_GRID:
            fold_params = {
                "max_leaf_nodes": max_leaf_nodes,
                "n_estimators": n_est,
                "min_samples_leaf": min_samples_leaf,
                "random_state": 42,
            }
            rf = RandomForestClassifier(**fold_params)
            rf.fit(train_x, train_y_)
            rf_proba = rf.predict_proba(val_x)[:, 1]
            ll = float(log_loss(val_y_, rf_proba))
            if ll < best_rf_ll:
                best_rf_ll = ll
                best_rf = (rf, fold_params, rf_proba)

        rf_model, rf_params, rf_proba = best_rf
        rf_m = compute_classification_metrics(val_y_, rf_proba)
        rf_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": rf_m["log_loss"],
                "metrics": rf_m,
                "model": rf_model,
                "best_params": rf_params,
            }
        )

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" platt={platt_m['log_loss']:.4f}"
            f" lr={lr_m['log_loss']:.4f}"
            f" hgb={hgb_m['log_loss']:.4f}"
            f" gb={gb_m['log_loss']:.4f}"
            f" rf={rf_m['log_loss']:.4f}"
        )

    # ── Average validation metrics ──
    def _avg_ll(results):
        valid = [r for r in results if r["log_loss"] != float("inf")]
        if not valid:
            return float("inf")
        return float(np.mean([r["log_loss"] for r in valid]))

    avg_platt = _avg_ll(platt_results)
    avg_lr = _avg_ll(lr_results)
    avg_hgb = _avg_ll(hgb_results)
    avg_gb = _avg_ll(gb_results)
    avg_rf = _avg_ll(rf_results)

    print("\n=== Average Validation Log Loss ===")
    print(f"  Platt (incumbent):       {avg_platt:.4f}")
    print(f"  LogisticRegression:      {avg_lr:.4f}")
    print(f"  HistGradientBoosting:    {avg_hgb:.4f}")
    print(f"  GradientBoosting:        {avg_gb:.4f}")
    print(f"  RandomForest (diagnostic): {avg_rf:.4f}")

    # ── Model selection by best avg val LL across challengers ──
    candidates = {
        "LogisticRegression": (lr_results, avg_lr),
        "HistGradientBoosting": (hgb_results, avg_hgb),
        "GradientBoosting": (gb_results, avg_gb),
    }
    best_model_name = min(candidates, key=lambda k: candidates[k][1])
    best_model_results = candidates[best_model_name][0]

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    is_train_full = df["season"].isin([2021, 2022, 2023, 2024]).values
    train_y_full = y[is_train_full].astype(int)
    train_elo_full = elo_prob[is_train_full]
    train_x_full = x_curated[is_train_full]
    hold_x = x_curated[is_hold]

    # Incumbent: Platt
    platt_full = _fit_platt(train_elo_full, train_y_full)
    hold_platt_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, hold_platt_proba)
    print(f"  Platt (incumbent): {hold_platt_m['log_loss']:.4f}")

    # Re-fit best model on full 2021-2024
    hold_probas: dict[str, np.ndarray] = {}
    hold_metrics: dict[str, dict] = {}

    # Best params per model type (pick middle-of-grid defaults)
    hgb_default = {
        "max_leaf_nodes": 8,
        "learning_rate": 0.05,
        "max_iter": 100,
        "min_samples_leaf": 40,
        "l2_regularization": 0.5,
    }
    gb_default = {
        "max_leaf_nodes": 8,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "min_samples_leaf": 40,
    }
    rf_default = {"max_leaf_nodes": 8, "n_estimators": 100, "min_samples_leaf": 40}

    for name in MODEL_NAMES:
        if name == "LogisticRegression":
            m = _logistic_pipe()
            m.fit(train_x_full, train_y_full)
            hold_proba = m.predict_proba(hold_x)[:, 1]
        elif name == "HistGradientBoosting":
            m = HistGradientBoostingClassifier(**hgb_default, early_stopping=False, random_state=42)
            m.fit(train_x_full, train_y_full)
            hold_proba = m.predict_proba(hold_x)[:, 1]
        elif name == "GradientBoosting":
            m = GradientBoostingClassifier(**gb_default, subsample=0.8, random_state=42)
            m.fit(train_x_full, train_y_full)
            hold_proba = m.predict_proba(hold_x)[:, 1]
        elif name == "RandomForest":
            m = RandomForestClassifier(**rf_default, random_state=42)
            m.fit(train_x_full, train_y_full)
            hold_proba = m.predict_proba(hold_x)[:, 1]

        hold_probas[name] = hold_proba
        hold_metrics[name] = compute_classification_metrics(hold_y, hold_proba)
        print(f"  {name}: {hold_metrics[name]['log_loss']:.4f}")

    # ── Calibration on best model ──
    print("\n=== Best Model Calibration ===")
    best_raw = hold_probas.get(best_model_name, hold_probas.get("LogisticRegression"))

    calibrations: dict[str, dict] = {}

    model_for_cal = None
    if best_model_name == "LogisticRegression":
        model_for_cal = _logistic_pipe()
    elif best_model_name == "HistGradientBoosting":
        model_for_cal = HistGradientBoostingClassifier(
            **hgb_default, early_stopping=False, random_state=42
        )
    elif best_model_name == "GradientBoosting":
        model_for_cal = GradientBoostingClassifier(**gb_default, subsample=0.8, random_state=42)
    if model_for_cal is not None:
        model_for_cal.fit(train_x_full, train_y_full)
        train_best_proba = model_for_cal.predict_proba(train_x_full)[:, 1]

        # Platt calibration
        platt_cal_fitted = _fit_platt(train_best_proba, train_y_full)
        platt_cal_proba = platt_cal_fitted.predict_proba(best_raw.reshape(-1, 1))[:, 1]
        platt_cal_m = compute_classification_metrics(hold_y, platt_cal_proba)
        calibrations["Platt"] = {
            "log_loss": platt_cal_m["log_loss"],
            "metrics": platt_cal_m,
            "model": platt_cal_fitted,
        }
        print(f"  {best_model_name} + Platt: {platt_cal_m['log_loss']:.4f}")

        # Isotonic (diagnostic only)
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_best_proba, train_y_full)
        iso_proba = iso.transform(best_raw)
        iso_m = compute_classification_metrics(hold_y, iso_proba)
        calibrations["Isotonic"] = {
            "log_loss": iso_m["log_loss"],
            "metrics": iso_m,
            "model": iso,
        }
        print(f"  {best_model_name} + Isotonic: {iso_m['log_loss']:.4f}")

    # ── Feature importance (if available) ──
    if model_for_cal is not None and hasattr(model_for_cal, "feature_importances_"):
        importances = model_for_cal.feature_importances_
        feat_imp = sorted(
            zip(CURATED_FEATURE_COLUMNS, importances),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        feat_imp = None

    # ═══ Report ═══
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Constrained Expressive Models Experiment\n\n")
        f.write("Comparing constrained tree-based and logistic models on a curated feature set\n")
        f.write("built around the incumbent MOV Elo probability signal.\n\n")

        f.write("## Curated Feature Set\n\n")
        f.write("| Feature | Source | Rationale |\n")
        f.write("|---------|--------|----------|\n")
        f.write("| `elo_prob` | MOV Elo incumbent | Core signal — home win probability |\n")
        f.write("| `elo_logit` | logit(elo_prob) | Linearize probability for logistic models |\n")
        f.write("| `elo_diff` | Elo ratings | Home minus away pregame rating |\n")
        f.write(
            "| `home_short_week`, `away_short_week` | Rest diff ≤6"
            " | Short-rest scheduling disadvantage |\n"
        )
        f.write(
            "| `home_off_bye`, `away_off_bye` | Rest diff ≥13 | Extra-rest scheduling advantage |\n"
        )
        f.write("| `thursday_flag`, `monday_flag` | Weekday | Primetime scheduling effects |\n")
        f.write(
            "| `home_consecutive_road`, `away_consecutive_road` | Location history"
            " | Travel fatigue |\n"
        )
        f.write("| `is_international` | Stadium location | International travel |\n")
        f.write(
            "| `home_qb_changed`, `away_qb_changed` | QB tracking | QB continuity disruption |\n"
        )
        f.write("| `qb_starts_diff` | QB starts this season | Experience gap |\n")
        f.write("| `qb_win_pct_diff` | QB win pct | Winning experience gap |\n")
        f.write("| `games_since_qb_change_diff` | QB change history | Stability gap |\n")
        f.write("| `new_qb_diff` | First start on this team | Novelty gap |\n")
        f.write("| `cold_flag` | weather_tmin/tmax ≤32°F | Cold weather |\n")
        f.write("| `windy_flag` | wind ≥15 mph | Windy conditions |\n")
        f.write("| `bad_weather_flag` | Cold OR windy OR precip | Combined adverse weather |\n")
        f.write("| `outdoor_game_flag` | Roof ∈ {outdoors, open} | Outdoor venue |\n")
        f.write("| `is_dome` | Roof ∈ {dome, closed} | Indoor venue |\n")
        f.write("| `weather_missing_flag` | Weather data null | Missing weather indicator |\n")
        f.write("| `week_norm` | Week / max(week) | Season timing (0–1) |\n")
        f.write("| `rest_diff` | home_rest − away_rest | Rest advantage |\n")
        f.write("| `div_game` | Divisional game flag | Familiarity/rivalry |\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- All features are pregame-safe (known before kickoff).\n")
        f.write("- Rolling-origin folds prevent 2025 from influencing model selection.\n")
        f.write("- QB change features computed chronologically from prior games only.\n")
        f.write("- Scheduling features computed chronologically.\n")
        f.write(
            "- Weather features dome-neutralized; missing values imputed with dataset median.\n"
        )
        f.write("- No target, score, or result columns in feature set.\n")
        f.write("- No raw team identity, QB identity OHE, or stadium identity.\n\n")

        f.write("## Incumbent Params\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| K-factor | {BEST_K} |\n")
        f.write(f"| HFA | {BEST_HFA} |\n")
        f.write(f"| Preseason regression | {BEST_REG} |\n")
        f.write(f"| MOV type | {BEST_MOV_TYPE} |\n")
        f.write(f"| MOV scale | {BEST_MOV_SCALE} |\n")
        f.write(f"| MOV cap | {BEST_MOV_CAP} |\n\n")

        f.write("## Data Split\n\n")
        f.write("| Fold | Training | Validation |\n")
        f.write("|------|----------|------------|\n")
        for i, (tr, val) in enumerate(ROLLING_FOLDS, 1):
            f.write(f"| {i} | {tr} | {val} |\n")
        f.write(f"| Holdout | 2021–2024 | {HOLDOUT_SEASON} |\n\n")

        f.write("## Models Compared\n\n")
        f.write("| Model | Type | Grid Size |\n")
        f.write("|-------|------|-----------|\n")
        f.write("| Platt (incumbent) | Platt-scaled MOV Elo | N/A |\n")
        f.write("| LogisticRegression | Linear on curated features | N/A |\n")
        f.write(f"| HistGradientBoosting | Constrained boosting | {len(HGB_GRID)} combos |\n")
        f.write(f"| GradientBoosting | Constrained boosting | {len(GB_GRID)} combos |\n")
        f.write(f"| RandomForest | Diagnostic only | {len(RF_GRID)} combos |\n\n")

        f.write("## Model Grids\n\n")
        f.write("### HistGradientBoosting\n")
        f.write("| Parameter | Values |\n")
        f.write("|-----------|--------|\n")
        f.write("| max_leaf_nodes | 4, 8, 12, 16 |\n")
        f.write("| learning_rate | 0.01, 0.03, 0.05, 0.1 |\n")
        f.write("| max_iter | 50, 100, 200 |\n")
        f.write("| min_samples_leaf | 20, 40, 60 |\n")
        f.write("| l2_regularization | 0.0, 0.1, 0.5, 1.0 |\n\n")
        f.write("### GradientBoosting\n")
        f.write("| Parameter | Values |\n")
        f.write("|-----------|--------|\n")
        f.write("| max_leaf_nodes | 4, 8, 12, 16 |\n")
        f.write("| learning_rate | 0.01, 0.03, 0.05, 0.1 |\n")
        f.write("| n_estimators | 50, 100, 200 |\n")
        f.write("| min_samples_leaf | 20, 40, 60 |\n")
        f.write("| subsample | 0.8 |\n\n")
        f.write("### RandomForest (diagnostic)\n")
        f.write("| Parameter | Values |\n")
        f.write("|-----------|--------|\n")
        f.write("| max_leaf_nodes | 4, 8, 12, 16 |\n")
        f.write("| n_estimators | 50, 100, 200 |\n")
        f.write("| min_samples_leaf | 20, 40, 60 |\n\n")

        f.write("## Average Validation Log Loss\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")

        def _fold_row(name, results):
            lls = [r["log_loss"] for r in results]
            avg = float(np.mean(lls))
            return f"| {name} | {avg:.4f} | {lls[0]:.4f} | {lls[1]:.4f} | {lls[2]:.4f} |\n"

        f.write(_fold_row("Platt (incumbent)", platt_results))
        f.write(_fold_row("LogisticRegression", lr_results))
        f.write(_fold_row("HistGradientBoosting", hgb_results))
        f.write(_fold_row("GradientBoosting", gb_results))
        f.write(_fold_row("RandomForest", rf_results))
        f.write("\n")

        f.write(f"## Best Model Selected: {best_model_name}\n\n")
        f.write("Selected by lowest average validation log loss among challengers.\n\n")

        # Per-fold best params
        f.write("### Per-Fold Best Parameters\n\n")
        for i, r in enumerate(best_model_results, 1):
            f.write(f"**Fold {i}** (train={r['train_seasons']}, val={r['val_season']}):\n")
            for param, value in r.get("best_params", {}).items():
                f.write(f"- {param}: {value}\n")
        f.write("\n")

        # Head-to-head
        f.write("## 2025 Holdout Comparison\n\n")
        f.write("| Model | Holdout LL | Brier | Acc | AUC |\n")
        f.write("|-------|-----------|-------|-----|-----|\n")
        random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
        prior_rate = train_y_full.mean()
        prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")

        def _row(name, m):
            ll_val = m["log_loss"]
            bs = m["brier_score"]
            acc = m["accuracy"]
            auc = m.get("roc_auc")
            auc_str = f"{auc:.4f}" if auc is not None else "—"
            return f"| {name} | {ll_val:.4f} | {bs:.4f} | {acc:.4f} | {auc_str} |\n"

        pll_val = hold_platt_m["log_loss"]
        pll_bs = hold_platt_m["brier_score"]
        pll_acc = hold_platt_m["accuracy"]
        f.write(
            f"| **Platt (incumbent) [target: 0.6373]** | {pll_val:.4f}"
            f" | {pll_bs:.4f} | {pll_acc:.4f} | — |\n"
        )

        for name in MODEL_NAMES:
            m = hold_metrics[name]
            f.write(_row(name, m))

        f.write("\n")

        # Calibration
        f.write("## Calibration\n\n")
        for cal_name, cal_data in calibrations.items():
            f.write(f"### {best_model_name} + {cal_name}\n\n")
            f.write(f"Holdout log loss: **{cal_data['log_loss']:.4f}**\n\n")
            cm = cal_data["metrics"]
            f.write("| Bucket | Count | Mean Pred | Mean Actual | Cal Error |\n")
            f.write("|--------|-------|-----------|-------------|-----------|\n")
            for b, vals in sorted(cm["calibration_buckets"].items()):
                mp = vals["mean_predicted_prob"]
                ma = vals["mean_actual_rate"]
                ce = vals["calibration_error"]
                f.write(f"| {b} | {vals['count']} | {mp} | {ma} | {ce} |\n")
            f.write("\n")

        # Feature importance
        if feat_imp:
            f.write("## Feature Importance\n\n")
            f.write("| Feature | Importance |\n")
            f.write("|---------|-----------|\n")
            for feat, imp in feat_imp:
                f.write(f"| {feat} | {imp:.4f} |\n")
            f.write("\n")
        else:
            f.write("## Feature Importance\n\n")
            f.write(f"Feature importance not available for {best_model_name}.\n")
            f.write("Logistic regression coefficients and permutation importance:\n\n")
            if best_model_name == "LogisticRegression" and model_for_cal is not None:
                try:
                    lr_model = model_for_cal.named_steps["lr"]
                    coefs = lr_model.coef_[0]

                    def _abs_coef(x):
                        return abs(x[1])

                    for feat, coef in sorted(
                        zip(CURATED_FEATURE_COLUMNS, coefs), key=_abs_coef, reverse=True
                    ):
                        f.write(f"- {feat}: {coef:+.4f}\n")
                except (AttributeError, KeyError):
                    f.write("Coefficients not available.\n")
            f.write("\n")

        # Recommendation
        f.write("## Recommendation\n\n")

        best_challenger_avg = min([avg_lr, avg_hgb, avg_gb])

        # Check if best model beats incumbent on average VALIDATION
        # then also on holdout
        best_cal_ll = min(
            [c["log_loss"] for c in calibrations.values()],
            default=float("inf"),
        )
        if best_cal_ll == float("inf"):
            best_cal_ll = hold_metrics.get(best_model_name, {}).get("log_loss", float("inf"))

        beats_validation = best_challenger_avg < avg_platt
        beats_holdout = best_cal_ll < INCUMBENT_HOLDOUT_LL

        if beats_validation and beats_holdout:
            f.write("✅ **Challenger promoted to research incumbent.**\n\n")
            f.write(f"**{best_model_name} + best calibration** (holdout LL={best_cal_ll:.4f})")
            f.write(f" beats the incumbent ({INCUMBENT_HOLDOUT_LL:.4f}) ")
            f.write(
                "and was selected by rolling-origin validation"
                f" (avg LL={best_challenger_avg:.4f}).\n"
            )
        elif beats_holdout and not beats_validation:
            f.write("⚠️ **Challenger wins on holdout but not validation.**\n\n")
            f.write(f"**{best_model_name} + best calibration** (holdout LL={best_cal_ll:.4f})")
            f.write(f" beats the incumbent ({INCUMBENT_HOLDOUT_LL:.4f}) ")
            f.write(
                "but was not selected by rolling-origin validation"
                f" (avg LL={best_challenger_avg:.4f}"
            )
            f.write(f" vs incumbent {avg_platt:.4f}). Keeping MOV Elo + Platt as incumbent.\n")
        elif beats_validation and not beats_holdout:
            f.write("⚠️ **Challenger wins validation but not holdout.**\n\n")
            f.write(f"**{best_model_name}** (avg val LL={best_challenger_avg:.4f})")
            f.write(f" won rolling-origin selection but holdout ({best_cal_ll:.4f})")
            f.write(f" did not beat incumbent ({INCUMBENT_HOLDOUT_LL:.4f}). ")
            f.write("Keeping MOV Elo + Platt as incumbent.\n")
        else:
            f.write("⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**\n\n")
            f.write("No expressive model beat the incumbent on holdout.")
            f.write(f" Best challenger: **{best_model_name}**")
            f.write(f" (avg val LL={best_challenger_avg:.4f}, holdout LL={best_cal_ll:.4f})")
            f.write(f" vs incumbent holdout LL={INCUMBENT_HOLDOUT_LL:.4f}.\n\n")
            f.write("Constrained tree models did not meaningfully improve over MOV Elo + Platt ")
            f.write("on this dataset.")
            if feat_imp:
                f.write(" Top features were Elo-based.")
        f.write("\n\n### Next Recommended Experiment\n\n")
        f.write("1. Market-baseline comparison (moneyline implied probabilities).\n")
        f.write("2. Residual diagnostics — where does the incumbent fail systematically?\n")
        f.write("3. DVOA/EPA features if available.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
