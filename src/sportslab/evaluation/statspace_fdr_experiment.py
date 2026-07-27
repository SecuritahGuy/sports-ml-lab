"""Rolling-origin experiment: FDR vs full champion (incl. QB overlay)."""

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
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features
from sportslab.features.statspace import (
    compute_statspace_fdr,
    merge_team_season_metrics,
    schedule_to_nfl_historical_games,
)

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
BEST_K, BEST_HFA, BEST_REG, BEST_DECAY, BEST_QB_BONUS = 36, 40, 0.1, 32, 0.2
SEED = 42

ELO_TO_LOGIT = np.log(10) / 400.0

FDR_VALUE_COLS = ["fraud_detector_rating"]


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


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
        df, fdr_all, prefix="fdr", value_columns=FDR_VALUE_COLS,
    )
    return result


def _fit_platt(x_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(x_train, y_train.astype(int))
    return pipe


def run_statspace_fdr_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/statspace_fdr.md",
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
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)

    all_seasons_numeric = sorted(int(s) for s in df["season"].unique() if s != 2026)
    print(f"Loading PBP for {all_seasons_numeric}...")
    pbp = load_pbp_data(seasons=all_seasons_numeric)
    print(f"PBP loaded: {pbp.shape}")

    df = compute_fdr_features(df, pbp)
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"Eligible games: {len(df)}")

    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values

    # Feature cols for Platt
    feat_cols = ["home_qb_changed", "away_qb_changed",
                 "home_rolling_mov_3", "away_rolling_mov_3"]
    all_feat = np.column_stack([df[c].values for c in feat_cols])

    home_fdr_col = "home_fdr_fraud_detector_rating"
    away_fdr_col = "away_fdr_fraud_detector_rating"
    has_fdr = home_fdr_col in df.columns and away_fdr_col in df.columns
    if has_fdr:
        fdr_feat = np.column_stack([df[home_fdr_col].values, df[away_fdr_col].values])
    else:
        fdr_feat = np.column_stack([elo_prob * 0, elo_prob * 0])  # dummy fallback

    # QB overlay arrays
    home_qb_adj = df["home_qb_adj"].values.astype(float)
    away_qb_adj = df["away_qb_adj"].values.astype(float)
    net_adj_elo = home_qb_adj - away_qb_adj
    qb_changed_either = (df["home_qb_changed"].values == 1) | (df["away_qb_changed"].values == 1)

    h_s = df.get("home_qb_team_starts_pre", pd.Series(0.0, index=df.index))
    a_s = df.get("away_qb_team_starts_pre", pd.Series(0.0, index=df.index))
    h_starts = h_s.fillna(0).values.astype(float)
    a_starts = a_s.fillna(0).values.astype(float)
    gate_active = qb_changed_either | (h_starts < 17) | (a_starts < 17)

    # ── Rolling-origin ──
    model_configs = {
        "A. Platt (no overlay)": {"use_fdr": False, "use_overlay": False},
        "B. Platt + overlay (champion)": {"use_fdr": False, "use_overlay": True},
        "C. Platt + FDR": {"use_fdr": True, "use_overlay": False},
        "D. Platt + FDR + overlay": {"use_fdr": True, "use_overlay": True},
    }

    results = {name: {"fold_lls": []} for name in model_configs}
    hold_results = {name: {} for name in model_configs}

    for fold_idx, (train_s, val_s) in enumerate(ROLLING_FOLDS):
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values

        train_elo = elo_prob[tr]
        train_feat = all_feat[tr]
        train_fdr = fdr_feat[tr]
        train_y_int = y[tr].astype(int)

        for name, cfg in model_configs.items():
            x_cols = [train_elo]
            if cfg["use_fdr"]:
                x_cols.append(train_fdr)
            if cfg["use_fdr"]:
                pass  # FDR included above
            x_train = np.column_stack(x_cols + [train_feat])
            pipe = _fit_platt(x_train, train_y_int)

            # Predict on ALL data for this fold (fold-safe: incumb prob only uses train)
            x_all_cols = [elo_prob]
            if cfg["use_fdr"]:
                x_all_cols.append(fdr_feat)
            x_all = np.column_stack(x_all_cols + [all_feat])
            base_prob = pipe.predict_proba(x_all)[:, 1]

            if cfg["use_overlay"]:
                overlay_adj = 1.0 * net_adj_elo * ELO_TO_LOGIT * gate_active.astype(float)
                final_logit = _logit(base_prob) + overlay_adj
                final_prob = _sigmoid(final_logit)
            else:
                final_prob = base_prob

            va_prob = final_prob[va]
            val_ll = compute_classification_metrics(y[va], va_prob)["log_loss"]
            results[name]["fold_lls"].append(val_ll)

    # Holdout (fit once on 2021-2024, eval on 2025)
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]

    train_elo_all = elo_prob[is_train]
    train_feat_all = all_feat[is_train]
    train_fdr_all = fdr_feat[is_train]
    train_y_all = y[is_train].astype(int)

    for name, cfg in model_configs.items():
        x_cols = [train_elo_all]
        if cfg["use_fdr"]:
            x_cols.append(train_fdr_all)
        x_train = np.column_stack(x_cols + [train_feat_all])
        pipe = _fit_platt(x_train, train_y_all)

        x_hold_cols = [elo_prob[is_hold]]
        if cfg["use_fdr"]:
            x_hold_cols.append(fdr_feat[is_hold])
        x_hold = np.column_stack(x_hold_cols + [all_feat[is_hold]])
        base_prob = pipe.predict_proba(x_hold)[:, 1]

        if cfg["use_overlay"]:
            h_gate = gate_active[is_hold]
            h_net = net_adj_elo[is_hold]
            final_logit = _logit(base_prob) + 1.0 * h_net * ELO_TO_LOGIT * h_gate.astype(float)
            final_prob = _sigmoid(final_logit)
        else:
            final_prob = base_prob

        m = compute_classification_metrics(hold_y, final_prob)
        hold_results[name] = m

    for name in model_configs:
        avg = float(np.mean(results[name]["fold_lls"]))
        results[name]["val_ll"] = avg
        print(f"  {name}: val={avg:.4f}  hold={hold_results[name]['log_loss']:.4f}")

    # ── Report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        f.write("# StatSpace FDR Experiment (vs Full Champion)\n\n")
        f.write(
            "## Methods\n\n"
            "FDR (Fraud Detector Rating) is computed per team-season from "
            "nflverse PBP, schedule results, and Elo ratings. It blends "
            "record strength, underlying quality, luck gap, close-game luck, "
            "turnover luck, and schedule suspicion into a z-scored composite "
            "where positive = overachieving (regression risk) and negative = "
            "underachieving (upside).\n\n"
            "The champion model is:\n"
            "  `Platt(elo_prob + qb_changed + rolling_mov_3) + QB overlay`\n"
            "  (gate = qb_changed OR starts<17, gamma=1.0, cap=40)\n\n"
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
        champ_val = results["B. Platt + overlay (champion)"]["val_ll"]
        champ_hold = hold_results["B. Platt + overlay (champion)"]["log_loss"]
        f.write(f"Champion: val={champ_val:.4f}, hold={champ_hold:.4f}\n\n")

        prom = []
        for name in model_configs:
            if name == "B. Platt + overlay (champion)":
                continue
            dv = results[name]["val_ll"] - champ_val
            dh = hold_results[name]["log_loss"] - champ_hold
            f.write(f"  {name}: Δval={dv:+.4f}, Δhold={dh:+.4f}\n")
            if dv <= -0.001 and dh <= -0.001:
                prom.append(name)

        if prom:
            f.write(f"\n**Promoted: {', '.join(prom)}**\n")
        else:
            f.write("\n**No model beats champion on both val and holdout by ≥ 0.001.**\n")

        f.write("\n---\nReport: statspace_fdr_experiment.py\n")
    print(f"\nReport: {rp}")
    return str(rp)


if __name__ == "__main__":
    run_statspace_fdr_experiment()
