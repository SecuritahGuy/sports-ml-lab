"""Rolling-origin experiment: DOBA vs FDR incumbent."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.epa import load_pbp_data
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features
from sportslab.features.statspace import (
    compute_statspace_doba,
    compute_statspace_fdr,
    merge_team_season_metrics,
    schedule_to_nfl_historical_games,
)

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2
SEED = 42

DOBA_VALUE_COLS = ["doba_score"]


def compute_doba_features(df, pbp_df):
    all_seasons = sorted(df[df[MODEL_ELIGIBLE_COLUMN]]["season"].unique())
    all_seasons = [s for s in all_seasons if s != 2026]
    all_doba = []
    for season in all_seasons:
        s_pbp = pbp_df[pbp_df["season"] == season].copy()
        if s_pbp.empty:
            continue
        doba = compute_statspace_doba(s_pbp, season=season)
        if not doba.empty:
            doba["season"] = season
            all_doba.append(doba)
    if not all_doba:
        return df
    doba_all = pd.concat(all_doba, ignore_index=True)
    result = merge_team_season_metrics(
        df, doba_all, prefix="doba", value_columns=DOBA_VALUE_COLS,
    )
    return result


def compute_fdr_features(df, pbp_df):
    all_seasons = sorted(df[df[MODEL_ELIGIBLE_COLUMN]]["season"].unique())
    all_seasons = [s for s in all_seasons if s != 2026]
    all_fdr = []
    for season in all_seasons:
        season_schedule = df[(df["season"] == season) & df[MODEL_ELIGIBLE_COLUMN]].copy()
        if season_schedule.empty:
            continue
        games = schedule_to_nfl_historical_games(season_schedule)
        if not games:
            continue
        fdr = compute_statspace_fdr(games, pbp_df=pbp_df, season=season)
        if not fdr.empty:
            fdr["season"] = season
            all_fdr.append(fdr)
    if not all_fdr:
        return df
    fdr_all = pd.concat(all_fdr, ignore_index=True)
    result = merge_team_season_metrics(
        df, fdr_all, prefix="fdr", value_columns=["fraud_detector_rating"],
    )
    return result


def _fit_platt(x_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(x_train, y_train.astype(int))
    return pipe


def run_statspace_doba_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/statspace_doba.md",
) -> str:
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG, team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)

    all_seasons_numeric = sorted(int(s) for s in df["season"].unique() if s != 2026)
    print(f"Loading PBP for {all_seasons_numeric}...")
    pbp = load_pbp_data(seasons=all_seasons_numeric)
    print(f"PBP loaded: {pbp.shape}")

    # Compute FDR (incumbent) and DOBA
    df = compute_fdr_features(df, pbp)
    df = compute_doba_features(df, pbp)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"Eligible games: {len(df)}")

    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values

    feat_cols = ["home_qb_changed", "away_qb_changed",
                 "home_rolling_mov_3", "away_rolling_mov_3"]
    all_feat = np.column_stack([df[c].values for c in feat_cols])

    # FDR columns
    has_fdr = "home_fdr_fraud_detector_rating" in df.columns
    if has_fdr:
        fdr_feat = np.column_stack([
            df["home_fdr_fraud_detector_rating"].values,
            df["away_fdr_fraud_detector_rating"].values,
        ])
    else:
        fdr_feat = np.ones((len(df), 2)) * -999

    # DOBA columns
    has_doba = "home_doba_doba_score" in df.columns
    if has_doba:
        doba_feat = np.column_stack([
            df["home_doba_doba_score"].values,
            df["away_doba_doba_score"].values,
        ])
    else:
        doba_feat = np.ones((len(df), 2)) * -999

    model_configs = {
        "A. Platt (no features)": {"use_fdr": False, "use_doba": False},
        "B. Platt + FDR (incumbent)": {"use_fdr": True, "use_doba": False},
        "C. Platt + DOBA": {"use_fdr": False, "use_doba": True},
        "D. Platt + FDR + DOBA": {"use_fdr": True, "use_doba": True},
        "E. DOBA only": {"use_fdr": False, "use_doba": True},
    }

    results = {name: {"fold_lls": []} for name in model_configs}
    hold_results = {name: {} for name in model_configs}

    for fold_idx, (train_s, val_s) in enumerate(ROLLING_FOLDS):
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values
        train_y_int = y[tr].astype(int)

        for name, cfg in model_configs.items():
            x_cols = []
            if name != "E. DOBA only":
                x_cols.append(elo_prob)
            if cfg["use_fdr"]:
                x_cols.append(fdr_feat)
            if cfg["use_doba"]:
                x_cols.append(doba_feat)
            x_all = np.column_stack(x_cols + [all_feat])
            x_train = x_all[tr]
            pipe = _fit_platt(x_train, train_y_int)
            proba = pipe.predict_proba(x_all[va])[:, 1]
            val_ll = compute_classification_metrics(y[va], proba)["log_loss"]
            results[name]["fold_lls"].append(val_ll)

    # Holdout
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    train_y_all = y[is_train].astype(int)

    for name, cfg in model_configs.items():
        x_cols = []
        if name != "E. DOBA only":
            x_cols.append(elo_prob)
        if cfg["use_fdr"]:
            x_cols.append(fdr_feat)
        if cfg["use_doba"]:
            x_cols.append(doba_feat)
        x_all = np.column_stack(x_cols + [all_feat])
        pipe = _fit_platt(x_all[is_train], train_y_all)
        proba = pipe.predict_proba(x_all[is_hold])[:, 1]
        m = compute_classification_metrics(hold_y, proba)
        hold_results[name] = m

    for name in model_configs:
        avg = float(np.mean(results[name]["fold_lls"]))
        results[name]["val_ll"] = avg
        print(f"  {name}: val={avg:.4f}  hold={hold_results[name]['log_loss']:.4f}")

    # Report
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        f.write("# StatSpace DOBA Experiment\n\n")
        f.write(
            "## Methods\n\n"
            "DOBA (sustainable Offensive efficiency composite) is computed per "
            "team-season from nflverse PBP. It blends offensive EPA/play, "
            "success rate, early-down efficiency, third/fourth-down efficiency, "
            "explosive rate, red zone efficiency, negative play rate, turnover "
            "rate, and a dependency penalty into a z-scored composite where "
            "higher = more sustainable offensive quality.\n\n"
        )

        f.write("## Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|-----------|-------|-------|-------|\n")
        for name in model_configs:
            r = results[name]
            f.write(f"| {name} | {r['val_ll']:.4f}"
                    f" | {r['fold_lls'][0]:.4f}"
                    f" | {r['fold_lls'][1]:.4f}"
                    f" | {r['fold_lls'][2]:.4f} |\n")

        f.write("\n## Holdout\n\n")
        f.write("| Model | Hold LL | Brier | AUC | Acc |\n")
        f.write("|-------|---------|-------|-----|------|\n")
        for name in model_configs:
            h = hold_results[name]
            f.write(f"| {name} | {h['log_loss']:.4f}"
                    f" | {h['brier_score']:.4f}"
                    f" | {h['roc_auc']:.4f}"
                    f" | {h['accuracy']:.4f} |\n")

        f.write("\n## Decision\n\n")
        inc_val = results["B. Platt + FDR (incumbent)"]["val_ll"]
        inc_hold = hold_results["B. Platt + FDR (incumbent)"]["log_loss"]
        f.write(f"Incumbent (FDR): val={inc_val:.4f}, hold={inc_hold:.4f}\n\n")

        prom = []
        for name in model_configs:
            if name == "B. Platt + FDR (incumbent)":
                continue
            dv = results[name]["val_ll"] - inc_val
            dh = hold_results[name]["log_loss"] - inc_hold
            f.write(f"  {name}: Δval={dv:+.4f}, Δhold={dh:+.4f}\n")
            if dv <= -0.001 and dh <= -0.001:
                prom.append(name)

        if prom:
            f.write(f"\n**Promoted: {', '.join(prom)}**\n")
        else:
            f.write("\n**No model beats incumbent on both val and holdout by ≥ 0.001.**\n")

        f.write("\n---\nReport: statspace_doba_experiment.py\n")
    print(f"\nReport: {rp}")
    return str(rp)


if __name__ == "__main__":
    run_statspace_doba_experiment()
