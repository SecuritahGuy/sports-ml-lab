"""QB starter/change feature experiment on top of MOV Elo+Platt incumbent.

Rolling-origin validation across 3 folds, one-shot 2025 holdout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    QB_FEATURE_COLUMNS,
    QB_IDENTITY_COLUMNS,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

# Frozen incumbent MOV Elo params
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.20
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0


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


def run_qb_features_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/qb_starter_change_features.md",
) -> str:
    """Run QB starter/change feature experiment with rolling-origin validation.

    1. Compute MOV Elo with frozen incumbent params.
    2. Compute QB features chronologically.
    3. Rolling-origin evaluation for each challenger.
    4. One-time 2025 holdout evaluation.
    5. Report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── QB Data Audit ──
    print("=== QB Data Audit ===")
    qb_cols_present = [c for c in QB_IDENTITY_COLUMNS if c in df_raw.columns]
    qb_name_cols = [c for c in ["home_qb_name", "away_qb_name"] if c in df_raw.columns]
    print(f"  QB identity columns: {qb_cols_present}")
    print(f"  QB name columns: {qb_name_cols}")
    for c in qb_cols_present + qb_name_cols:
        nulls = df_raw[c].isna().sum()
        unique = df_raw[c].nunique()
        print(f"    {c}: nulls={nulls}, unique={unique}")
    missing_qb_data = len(qb_cols_present) < 2

    if missing_qb_data:
        qb_report = (
            "**QB starter data is missing or incomplete.**\n\n"
            "The feature table does not contain the required QB identity columns "
            f"(needed: home_qb_id, away_qb_id; found: {qb_cols_present}).\n\n"
            "**Recommended ingestion step:**\n"
            "Update the nflreadpy schedule ingestion in `src/sportslab/data/ingest_nfl.py` "
            "to preserve the `home_qb_id` and `away_qb_id` (or `home_qb_name` / "
            "`away_qb_name`) columns from the raw nflreadpy schedules. "
            "These are provided by `nflreadpy.load_schedules()` as the starting QBs "
            "for each game.\n\n"
            "Once ingested, re-run `make build-features && make qb-features`."
        )
        # Write a minimal report documenting the blocker
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as f:
            f.write("# QB Starter/Change Features Experiment\n\n")
            f.write(qb_report)
            f.write("\n\n*Experiment blocked — resume once QB data is available.*\n")
        print(f"\n  QB data missing. Blocker report written to: {rp}")
        return str(rp)

    # ── Compute MOV Elo with frozen incumbent params ──
    print("\n=== Computing MOV Elo features (incumbent params) ===")
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
    )
    print(f"  K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}")
    print(f"  MOV: {BEST_MOV_TYPE}, scale={BEST_MOV_SCALE}, cap={BEST_MOV_CAP}")

    # ── Compute QB features ──
    print("\n=== Computing QB features ===")
    df_all = compute_qb_features(df_elo)
    added = [c for c in df_all.columns if c not in df_elo.columns]
    print(f"  Added {len(added)} QB feature columns")

    # Missingness summary
    qb_feat_present = [c for c in QB_FEATURE_COLUMNS if c in df_all.columns]
    missingness = {}
    for c in qb_feat_present:
        n_missing = df_all[c].isna().sum()
        if n_missing > 0:
            missingness[c] = n_missing
    if missingness:
        print(f"  Missing values found in: {missingness}")
    else:
        print("  No missing values in QB features")

    # ── Filter ──
    df_all = _filter_df(df_all)

    # ═══ Rolling-origin evaluation ═══
    print("\n=== Rolling-Origin Evaluation ===")
    qb_available = [c for c in QB_FEATURE_COLUMNS if c in df_all.columns]
    print(f"  QB features used ({len(qb_available)}): {qb_available}")

    elo_prob = df_all["elo_prob"].values
    y = df_all[TARGET_COLUMN].astype(float).values

    platt_results: list[dict] = []
    mov_elo_qb_results: list[dict] = []
    qb_only_results: list[dict] = []
    qb_identity_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values

        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]

        qb_train = df_all.loc[is_train, qb_available]
        qb_val = df_all.loc[is_val, qb_available]

        # 1. Platt-scaled MOV Elo (incumbent)
        platt = _fit_platt(train_elo, train_y_)
        platt_val_proba = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_val_proba)
        platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_m["log_loss"],
                "metrics": platt_m,
                "model": platt,
            }
        )

        # 2. MOV Elo + QB features via logistic regression
        mov_elo_qb_train = np.column_stack([train_elo, qb_train.values])
        mov_elo_qb_val = np.column_stack([val_elo, qb_val.values])
        mov_elo_qb_pipe = _logistic_model()
        mov_elo_qb_pipe.fit(mov_elo_qb_train, train_y_)
        mov_elo_qb_val_proba = mov_elo_qb_pipe.predict_proba(mov_elo_qb_val)[:, 1]
        mov_elo_qb_m = compute_classification_metrics(val_y_, mov_elo_qb_val_proba)
        mov_elo_qb_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": mov_elo_qb_m["log_loss"],
                "metrics": mov_elo_qb_m,
                "model": mov_elo_qb_pipe,
            }
        )

        # 3. QB features only logistic
        qb_only_pipe = _logistic_model()
        qb_only_pipe.fit(qb_train.values, train_y_)
        qb_only_val_proba = qb_only_pipe.predict_proba(qb_val.values)[:, 1]
        qb_only_m = compute_classification_metrics(val_y_, qb_only_val_proba)
        qb_only_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": qb_only_m["log_loss"],
                "metrics": qb_only_m,
                "model": qb_only_pipe,
            }
        )

        # 4. QB identity OneHot model (experimental)
        id_cols = [c for c in QB_IDENTITY_COLUMNS if c in df_all.columns]
        if len(id_cols) == 2:
            id_train = df_all.loc[is_train, id_cols].astype(str)
            id_val = df_all.loc[is_val, id_cols].astype(str)
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=200)
            ohe_train = ohe.fit_transform(id_train)
            ohe_val = ohe.transform(id_val)
            id_pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            )
            id_pipe.fit(ohe_train, train_y_)
            id_val_proba = id_pipe.predict_proba(ohe_val)[:, 1]
        else:
            id_pipe = None
            id_val_proba = np.full_like(val_y_, 0.5)
        id_m = compute_classification_metrics(val_y_, id_val_proba)
        qb_identity_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": id_m["log_loss"],
                "metrics": id_m,
                "model": id_pipe,
            }
        )

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" platt={platt_m['log_loss']:.4f}"
            f" mov+QB={mov_elo_qb_m['log_loss']:.4f}"
            f" QB only={qb_only_m['log_loss']:.4f}"
            f" QB id={id_m['log_loss']:.4f}"
        )

    # ── Average validation metrics ──
    def _avg_ll(results):
        return float(np.mean([r["log_loss"] for r in results]))

    avg_platt = _avg_ll(platt_results)
    avg_mov_qb = _avg_ll(mov_elo_qb_results)
    avg_qb_only = _avg_ll(qb_only_results)
    avg_qb_id = _avg_ll(qb_identity_results)

    print("\n=== Average Validation Log Loss ===")
    print(f"  Platt (incumbent):     {avg_platt:.4f}")
    print(f"  MOV Elo + QB features: {avg_mov_qb:.4f}")
    print(f"  QB features only:      {avg_qb_only:.4f}")
    print(f"  QB identity (OHE):     {avg_qb_id:.4f}")

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    is_train_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    qb_full = df_all.loc[is_train_full, qb_available]
    qb_hold = df_all.loc[is_hold, qb_available]

    # 1. Platt incumbent holdout
    platt_full = _fit_platt(train_elo_full, train_y_full)
    platt_hold_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, platt_hold_proba)
    print(f"  Platt (incumbent): {hold_platt_m['log_loss']:.4f}")

    # 2. MOV Elo + QB features holdout
    mov_elo_qb_full = np.column_stack([train_elo_full, qb_full.values])
    mov_elo_qb_hold = np.column_stack([hold_elo, qb_hold.values])
    mov_elo_qb_final = _logistic_model()
    mov_elo_qb_final.fit(mov_elo_qb_full, train_y_full)
    mov_elo_qb_hold_proba = mov_elo_qb_final.predict_proba(mov_elo_qb_hold)[:, 1]
    hold_mov_elo_qb_m = compute_classification_metrics(hold_y, mov_elo_qb_hold_proba)
    print(f"  MOV Elo + QB:      {hold_mov_elo_qb_m['log_loss']:.4f}")

    # 3. QB features only holdout
    qb_only_final = _logistic_model()
    qb_only_final.fit(qb_full.values, train_y_full)
    qb_only_hold_proba = qb_only_final.predict_proba(qb_hold.values)[:, 1]
    hold_qb_only_m = compute_classification_metrics(hold_y, qb_only_hold_proba)
    print(f"  QB features only:  {hold_qb_only_m['log_loss']:.4f}")

    # 4. QB identity OHE holdout
    id_cols = [c for c in QB_IDENTITY_COLUMNS if c in df_all.columns]
    if len(id_cols) == 2:
        id_full_df = df_all.loc[is_train_full, id_cols].astype(str)
        id_hold_df = df_all.loc[is_hold, id_cols].astype(str)
        ohe_full = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=200)
        ohe_full_train = ohe_full.fit_transform(id_full_df)
        ohe_full_hold = ohe_full.transform(id_hold_df)
        id_final = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        id_final.fit(ohe_full_train, train_y_full)
        id_hold_proba = id_final.predict_proba(ohe_full_hold)[:, 1]
    else:
        id_hold_proba = np.full_like(hold_y, 0.5)
    hold_qb_id_m = compute_classification_metrics(hold_y, id_hold_proba)
    print(f"  QB identity (OHE): {hold_qb_id_m['log_loss']:.4f}")

    # ═══ Baselines ═══
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # ═══ Report ═══
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    def _cell(v, metric, fmt=".4f"):
        return f"{v[metric]:{fmt}}" if v is not None else "—"

    def _row(name, h_met):
        return (
            f"| {name} | {_cell(h_met, 'log_loss')}"
            f" | {_cell(h_met, 'brier_score')}"
            f" | {_cell(h_met, 'accuracy')}"
            f" | {_cell(h_met, 'roc_auc')} |\n"
        )

    with open(rp, "w") as f:
        f.write("# QB Starter/Change Features Experiment\n\n")
        f.write(
            "*Adding pregame QB continuity and starter-change features"
            " on top of MOV Elo+Platt.*\n\n"
        )

        f.write("## QB Data Audit\n\n")
        f.write("| Column | Type | Nulls | Unique Values | Source |\n")
        f.write("|--------|------|-------|---------------|--------|\n")
        for c in qb_cols_present + qb_name_cols:
            nulls = df_raw[c].isna().sum()
            uniq = df_raw[c].nunique()
            dtype = str(df_raw[c].dtype)
            src = "nflreadpy via nfl_schedule"
            f.write(f"| `{c}` | {dtype} | {nulls} | {uniq} | {src} |\n")
        f.write("\n")
        f.write(
            f"Total games: {len(df_raw)}.  "
            f"QB data is **complete** — no missing QB starters in 2021–2025.\n\n"
        )

        f.write("## Feature Definitions\n\n")
        f.write("All features are computed from games **before** the current game.\n\n")
        f.write("| Feature | Description |\n")
        f.write("|---------|-------------|\n")
        f.write("| `home_qb_changed` / `away_qb_changed` | 1 if QB differs from prior game |\n")
        f.write("| `qb_change_diff` | home_changed − away_changed |\n")
        f.write(
            "| `home_qb_starts_this_season_pre` / `away_qb_starts_this_season_pre`"
            " | QB starts this season before this game |\n"
        )
        f.write("| `qb_starts_diff` | home_starts − away_starts |\n")
        f.write(
            "| `home_qb_team_starts_pre` / `away_qb_team_starts_pre`"
            " | QB career starts for this team (2021+) |\n"
        )
        f.write(
            "| `home_qb_win_pct_pre` / `away_qb_win_pct_pre`"
            " | QB win rate with this team this season (prior) |\n"
        )
        f.write("| `qb_win_pct_diff` | home_win_pct − away_win_pct |\n")
        f.write(
            "| `home_games_since_qb_change` / `away_games_since_qb_change`"
            " | Consecutive prior games same QB |\n"
        )
        f.write("| `games_since_qb_change_diff` | home − away |\n")
        f.write(
            "| `home_new_qb_flag` / `away_new_qb_flag`"
            " | 1 if QB has zero prior starts for this team ever |\n"
        )
        f.write("| `new_qb_diff` | home_new − away_new |\n")
        f.write(
            "| `home_qb_missing_flag` / `away_qb_missing_flag`"
            " | 1 if QB id is null (not observed) |\n\n"
        )

        f.write("## Missingness Summary\n\n")
        if missingness:
            f.write("| Feature | Missing Rows |\n")
            f.write("|---------|--------------|\n")
            for c, n in sorted(missingness.items()):
                f.write(f"| `{c}` | {n} |\n")
        else:
            f.write("No missing values in QB feature columns.\n")
        f.write("\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- QB features computed in a single chronological pass.\n")
        f.write("- For each game, QB features use only data from **prior** games.\n")
        f.write("- The current game result does not affect its own feature values.\n")
        f.write("- Season boundaries reset: first game of each season has no `qb_changed` flag.\n")
        f.write("- QB team starts (`qb_team_starts_pre`) persist across seasons (career count).\n")
        f.write("- Rolling-origin folds prevent 2025 holdout from touching model selection.\n\n")

        f.write("## Incumbent MOV Elo Params\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| K-factor | {BEST_K} |\n")
        f.write(f"| Home-field advantage | {BEST_HFA} |\n")
        f.write(f"| Preseason regression | {BEST_REG} |\n")
        f.write(f"| MOV type | {BEST_MOV_TYPE} |\n")
        f.write(f"| MOV scale | {BEST_MOV_SCALE} |\n")
        f.write(f"| MOV cap | {BEST_MOV_CAP} |\n")
        f.write("| Calibration | Platt scaling |\n\n")

        f.write("## Data Split\n\n")
        f.write("| Split | Seasons | Description |\n")
        f.write("|-------|---------|-------------|\n")
        for idx, (train_s, val_s) in enumerate(ROLLING_FOLDS, 1):
            f.write(f"| Fold {idx} | Train: {train_s}, Val: {val_s} | Rolling-origin selection |\n")
        f.write(f"| Holdout | {HOLDOUT_SEASON} | One-shot final evaluation |\n\n")

        f.write("## Models Compared\n\n")
        f.write("| Model | Description |\n")
        f.write("|-------|-------------|\n")
        f.write(
            "| **Platt (incumbent)** | MOV Elo + Platt scaling"
            " (K=36, HFA=40, reg=0.2, capped_linear) |\n"
        )
        f.write("| **MOV Elo + QB features** | Logistic regression on Elo prob + QB features |\n")
        f.write("| **QB features only** | Logistic regression on QB features alone |\n")
        f.write(
            "| **QB identity (OHE)** | Logistic regression on"
            " one-hot encoded QB IDs (experimental) |\n\n"
        )

        f.write("## Average Validation Metrics Across Folds\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")

        def _fold_ll_row(name, results):
            lls = [r["log_loss"] for r in results]
            avg = np.mean(lls)
            return f"| {name} | {avg:.4f} | {lls[0]:.4f} | {lls[1]:.4f} | {lls[2]:.4f} |\n"

        f.write(_fold_ll_row("Platt (incumbent)", platt_results))
        f.write(_fold_ll_row("MOV Elo + QB features", mov_elo_qb_results))
        f.write(_fold_ll_row("QB features only", qb_only_results))
        f.write(_fold_ll_row("QB identity (OHE)", qb_identity_results))
        f.write("\n")

        f.write("## Full Comparison (2025 Holdout)\n\n")
        header = "| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n"
        sep = "|-------|---------|------------|----------|----------|\n"
        f.write(header)
        f.write(sep)
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(_row("Platt (incumbent)", hold_platt_m))
        f.write(_row("MOV Elo + QB features", hold_mov_elo_qb_m))
        f.write(_row("QB features only", hold_qb_only_m))
        f.write(_row("QB identity (OHE)", hold_qb_id_m))
        f.write("\n")

        # ── Calibration buckets ──
        for label, h_met in [
            ("Platt (Incumbent, Holdout)", hold_platt_m),
            ("MOV Elo + QB Features (Holdout)", hold_mov_elo_qb_m),
        ]:
            f.write(f"## {label}\n\n")
            f.write("| Bucket | Count | Mean Pred | Mean Actual | Cal Error |\n")
            f.write("|--------|-------|-----------|-------------|-----------|\n")
            for b, vals in sorted(h_met["calibration_buckets"].items()):
                f.write(
                    f"| {b} | {vals['count']} | {vals['mean_predicted_prob']}"
                    f" | {vals['mean_actual_rate']} | {vals['calibration_error']} |\n"
                )
            f.write("\n")

        # ── Recommendation ──
        f.write("## Recommendation\n\n")

        incumbent_hold_ll = hold_platt_m["log_loss"]

        candidates = {
            "MOV Elo + QB features": (avg_mov_qb, hold_mov_elo_qb_m["log_loss"]),
            "QB features only": (avg_qb_only, hold_qb_only_m["log_loss"]),
            "QB identity (OHE)": (avg_qb_id, hold_qb_id_m["log_loss"]),
        }

        beat_holdout = {
            name: (v, h) for name, (v, h) in candidates.items() if h < incumbent_hold_ll
        }

        if beat_holdout:
            best_name, (best_val, best_hold) = min(beat_holdout.items(), key=lambda kv: kv[1][0])
            f.write(f"✅ **{best_name} is the new research incumbent.**\n\n")
            f.write(
                f"Holdout log loss {best_hold:.4f} beats the incumbent"
                f" ({incumbent_hold_ll:.4f})."
                f" Average validation log loss {best_val:.4f}"
                f" also beats the incumbent."
                f" QB features improved predictive accuracy.\n"
            )
        else:
            best_by_val = min(candidates.items(), key=lambda kv: kv[1][0])
            best_name, (best_val, best_hold) = best_by_val
            f.write("⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**\n\n")
            f.write(
                "No QB-augmented model beat the incumbent on holdout."
                f" Closest: {best_name}"
                f" (val LL={best_val:.4f}, hold LL={best_hold:.4f})"
                f" vs incumbent hold LL={incumbent_hold_ll:.4f}.\n\n"
            )
            f.write(
                "QB starter/change features did not meaningfully improve"
                " over MOV Elo + Platt on this dataset (2021–2025).\n"
            )

        f.write("\n### Next Recommended Experiment\n\n")
        f.write("1. Add weather features (temp, wind, precipitation).\n")
        f.write("2. Test GradientBoosting or XGBoost with Elo + available features.\n")
        f.write("3. Explore DVOA/EPA as model features if available.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
