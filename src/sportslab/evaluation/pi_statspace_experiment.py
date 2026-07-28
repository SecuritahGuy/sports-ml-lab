"""Test StatSpace features (FDR, DOBA, Chaos) on Pi-Ratings base.

Compares 5 model configs:
  1. Standard Elo + FDR + DOBA + Chaos (current overall champion)
  2. Pi-Ratings only (football-only champion)
  3. Pi-Ratings + FDR
  4. Pi-Ratings + FDR + DOBA
  5. Pi-Ratings + FDR + DOBA + Chaos

Follows same pipeline as statspace_chaos_experiment but swaps elo_prob for pi_prob.
"""

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
from sportslab.features.ratings import compute_elo_features, compute_pi_ratings_features
from sportslab.features.situational import compute_situational_features
from sportslab.features.statspace import (
    compute_statspace_chaos_rate,
    compute_statspace_doba,
    compute_statspace_fdr,
    merge_team_season_metrics,
    schedule_to_nfl_historical_games,
)

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
SEED = 42

PI_ALPHA = 0.5
PI_BASE_K = 28
PI_HK_RATIO = 1.25
PI_HFA = 30
PI_REG = 0.0

ELO_K = 36
ELO_HFA = 40
ELO_REG = 0.1
ELO_DECAY = 32
ELO_QB_BONUS = 0.2


def compute_metric_features(df, pbp_df, metric_name):
    metric_fn = {
        "chaos": compute_statspace_chaos_rate,
        "doba": compute_statspace_doba,
    }[metric_name]
    value_cols = {
        "chaos": ["chaos_rate"],
        "doba": ["doba_score"],
    }[metric_name]
    prefix = metric_name

    all_seasons = sorted(df[df[MODEL_ELIGIBLE_COLUMN]]["season"].unique())
    all_seasons = [s for s in all_seasons if s != 2026]
    all_metric = []
    for season in all_seasons:
        s_pbp = pbp_df[pbp_df["season"] == season].copy()
        if s_pbp.empty:
            continue
        result = metric_fn(s_pbp, season=season)
        if not result.empty:
            result["season"] = season
            all_metric.append(result)
    if not all_metric:
        return df
    metric_all = pd.concat(all_metric, ignore_index=True)
    result = merge_team_season_metrics(
        df, metric_all, prefix=prefix, value_columns=value_cols,
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


def run_pi_statspace_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/pi_statspace.md",
) -> str:
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    # Build standard Elo base for champion comparison
    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=ELO_REG, qb_change_bonus=ELO_QB_BONUS,
    )
    df_elo = compute_elo_features(
        df_raw, k_factor=ELO_K, home_advantage=ELO_HFA,
        preseason_regression=ELO_REG, team_regression_overrides=overrides,
        decay_half_life=ELO_DECAY,
    )
    df_elo = compute_qb_features(df_elo)
    df_elo = compute_situational_features(df_elo)

    # Build Pi-Ratings base
    df_pi = compute_pi_ratings_features(
        df_raw, base_k=PI_BASE_K, home_advantage=PI_HFA,
        preseason_regression=PI_REG, alpha=PI_ALPHA, hk_ratio=PI_HK_RATIO,
    )
    df_pi = compute_qb_features(df_pi)
    df_pi = compute_situational_features(df_pi)

    # Load PBP once, share between both bases
    all_seasons_numeric = sorted(int(s) for s in df_elo["season"].unique() if s != 2026)
    print(f"Loading PBP for {all_seasons_numeric}...")
    pbp = load_pbp_data(seasons=all_seasons_numeric)
    print(f"PBP loaded: {pbp.shape}")

    # Compute StatSpace features on both bases separately
    print("Computing StatSpace features on elo base...")
    df_elo = compute_fdr_features(df_elo, pbp)
    df_elo = compute_metric_features(df_elo, pbp, "doba")
    df_elo = compute_metric_features(df_elo, pbp, "chaos")

    print("Computing StatSpace features on pi base...")
    df_pi = compute_fdr_features(df_pi, pbp)
    df_pi = compute_metric_features(df_pi, pbp, "doba")
    df_pi = compute_metric_features(df_pi, pbp, "chaos")

    # Filter and align
    mask = df_elo[MODEL_ELIGIBLE_COLUMN].values & ~df_elo[NEUTRAL_COLUMN].values
    df_elo = df_elo[mask].copy().reset_index(drop=True)
    df_pi = df_pi[mask].copy().reset_index(drop=True)
    print(f"Eligible games: {len(df_elo)}")

    elo_prob = df_elo["elo_prob"].values.astype(float)
    pi_prob = df_pi["pi_prob"].values.astype(float)
    y = df_elo[TARGET_COLUMN].astype(float).values

    feat_cols = ["home_qb_changed", "away_qb_changed",
                 "home_rolling_mov_3", "away_rolling_mov_3"]
    feat_elo = np.column_stack([df_elo[c].values for c in feat_cols])
    feat_pi = np.column_stack([df_pi[c].values for c in feat_cols])

    def _get2(df, col_base):
        h = f"home_{col_base}"
        a = f"away_{col_base}"
        if h in df.columns and a in df.columns:
            return np.column_stack([df[h].values, df[a].values])
        return np.ones((len(df), 2)) * -999

    # StatSpace features from elo base
    ss_fdr_e = _get2(df_elo, "fdr_fraud_detector_rating")
    ss_doba_e = _get2(df_elo, "doba_doba_score")
    ss_chaos_e = _get2(df_elo, "chaos_chaos_rate")

    # StatSpace features from pi base (should be near-identical since metrics
    # are team-season composites, independent of Elo)
    ss_fdr_p = _get2(df_pi, "fdr_fraud_detector_rating")
    ss_doba_p = _get2(df_pi, "doba_doba_score")
    ss_chaos_p = _get2(df_pi, "chaos_chaos_rate")

    # 5 model configs
    model_configs = {
        "A. Elo + FDR + DOBA + Chaos (champion)": {"prob": elo_prob, "feat": feat_elo,
                                                     "fdr": ss_fdr_e, "doba": ss_doba_e,
                                                     "chaos": ss_chaos_e},
        "B. Pi-Ratings only": {"prob": pi_prob, "feat": feat_pi,
                                "fdr": None, "doba": None, "chaos": None},
        "C. Pi + FDR": {"prob": pi_prob, "feat": feat_pi,
                         "fdr": ss_fdr_p, "doba": None, "chaos": None},
        "D. Pi + FDR + DOBA": {"prob": pi_prob, "feat": feat_pi,
                                "fdr": ss_fdr_p, "doba": ss_doba_p, "chaos": None},
        "E. Pi + FDR + DOBA + Chaos": {"prob": pi_prob, "feat": feat_pi,
                                        "fdr": ss_fdr_p, "doba": ss_doba_p,
                                        "chaos": ss_chaos_p},
    }

    def _stack(cfg):
        cols = [cfg["prob"]]
        if cfg["fdr"] is not None:
            cols.append(cfg["fdr"])
        if cfg["doba"] is not None:
            cols.append(cfg["doba"])
        if cfg["chaos"] is not None:
            cols.append(cfg["chaos"])
        return np.column_stack(cols + [cfg["feat"]])

    results = {name: {"fold_lls": []} for name in model_configs}
    hold_results = {name: {} for name in model_configs}

    for fold_idx, (train_s, val_s) in enumerate(ROLLING_FOLDS):
        tr = df_elo["season"].isin(train_s).values
        va = (df_elo["season"] == val_s).values
        train_y_int = y[tr].astype(int)

        for name, cfg in model_configs.items():
            x_all = _stack(cfg)
            pipe = _fit_platt(x_all[tr], train_y_int)
            proba = pipe.predict_proba(x_all[va])[:, 1]
            val_ll = compute_classification_metrics(y[va], proba)["log_loss"]
            results[name]["fold_lls"].append(val_ll)

    # Holdout
    is_train = df_elo["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = (df_elo["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    train_y_all = y[is_train].astype(int)

    for name, cfg in model_configs.items():
        x_all = _stack(cfg)
        pipe = _fit_platt(x_all[is_train], train_y_all)
        proba = pipe.predict_proba(x_all[is_hold])[:, 1]
        m = compute_classification_metrics(hold_y, proba)
        hold_results[name] = m

    for name in model_configs:
        avg = float(np.mean(results[name]["fold_lls"]))
        results[name]["val_ll"] = avg
        print(f"  {name}: val={avg:.4f}  hold={hold_results[name]['log_loss']:.4f}")

    # Report
    inc_name = "A. Elo + FDR + DOBA + Chaos (champion)"
    inc_val = results[inc_name]["val_ll"]
    inc_hold = hold_results[inc_name]["log_loss"]

    pi_only = "B. Pi-Ratings only"
    pi_only_val = results[pi_only]["val_ll"]
    pi_only_hold = hold_results[pi_only]["log_loss"]

    report_lines = [
        "# StatSpace Features on Pi-Ratings Base",
        "",
        "Testing whether StatSpace PBP composites (FDR, DOBA, Chaos) improve",
        "on the Pi-Ratings football-only champion.",
        "",
        "## Configs",
        "",
        "| ID | Model | Base Rating | StatSpace Features |",
        "|---|-------|-------------|-------------------|",
        "| A | Current overall champion | Standard Elo (K=36) | FDR + DOBA + Chaos |",
        f"| B | Football-only champion | Pi-Ratings (α={PI_ALPHA}) | None |",
        "| C | Pi + FDR | Pi-Ratings | FDR |",
        "| D | Pi + FDR + DOBA | Pi-Ratings | FDR + DOBA |",
        "| E | Pi + FDR + DOBA + Chaos | Pi-Ratings | FDR + DOBA + Chaos |",
        "",
        "## Validation (Rolling-Origin 3-Fold)",
        "",
        "| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |",
        "|-------|-----------|-------|-------|-------|",
    ]
    for name in model_configs:
        r = results[name]
        report_lines.append(
            f"| {name} | {r['val_ll']:.4f}"
            f" | {r['fold_lls'][0]:.4f}"
            f" | {r['fold_lls'][1]:.4f}"
            f" | {r['fold_lls'][2]:.4f} |"
        )

    report_lines.extend([
        "",
        "## Holdout (2025)",
        "",
        "| Model | Hold LL | Brier | AUC | Acc |",
        "|-------|---------|-------|-----|------|",
    ])
    for name in model_configs:
        h = hold_results[name]
        report_lines.append(
            f"| {name} | {h['log_loss']:.4f}"
            f" | {h['brier_score']:.4f}"
            f" | {h['roc_auc']:.4f}"
            f" | {h['accuracy']:.4f} |"
        )

    report_lines.extend([
        "",
        "## Comparison vs Current Champion (Elo + FDR + DOBA + Chaos)",
        "",
        f"Incumbent (A): val={inc_val:.4f}, hold={inc_hold:.4f}",
        "",
        "| Model | Δval | Δhold | Decision |",
        "|-------|------|-------|----------|",
    ])

    prom = []
    for name in model_configs:
        if name == inc_name:
            continue
        dv = results[name]["val_ll"] - inc_val
        dh = hold_results[name]["log_loss"] - inc_hold
        verdict = ""
        if dv <= -0.001 and dh <= -0.001:
            verdict = "✅ PROMOTED"
            prom.append(name)
        elif dv <= -0.001 and dh > -0.001:
            verdict = "Wins val, loses hold"
        elif dv > -0.001 and dh <= -0.001:
            verdict = "Loses val, wins hold"
        elif dv <= 0 and dh <= 0:
            verdict = "Better but below threshold"
        else:
            verdict = "Worse on both"
        report_lines.append(f"| {name} | {dv:+.4f} | {dh:+.4f} | {verdict} |")

    report_lines.extend([
        "",
        "## Comparison vs Pi-Ratings Only (football-only champion)",
        "",
        f"Pi-only (B): val={pi_only_val:.4f}, hold={pi_only_hold:.4f}",
        "",
        "| Model | Δval | Δhold | Decision |",
        "|-------|------|-------|----------|",
    ])

    for name in ["C. Pi + FDR", "D. Pi + FDR + DOBA", "E. Pi + FDR + DOBA + Chaos"]:
        dv = results[name]["val_ll"] - pi_only_val
        dh = hold_results[name]["log_loss"] - pi_only_hold
        verdict = ""
        if dv <= -0.001 and dh <= -0.001:
            verdict = "✅ PROMOTED over Pi-only"
        elif dv <= -0.001 and dh > -0.001:
            verdict = "Wins val, loses hold"
        elif dv > -0.001 and dh <= -0.001:
            verdict = "Loses val, wins hold"
        else:
            verdict = "Worse than Pi-only"
        report_lines.append(f"| {name} | {dv:+.4f} | {dh:+.4f} | {verdict} |")

    report_lines.extend([
        "",
        "## Decision",
        "",
    ])

    if prom:
        report_lines.append(f"**Promoted: {', '.join(prom)}**")
        report_lines.append("")
        report_lines.append("StatSpace features improve on Pi-Ratings base.")
    else:
        msg = "**No model beats the current champion on both val and holdout by ≥ 0.001.**"
        report_lines.append(msg)

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("Report: pi_statspace_experiment.py")

    report = "\n".join(report_lines)
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report)
    print(f"\nReport: {rp}")
    return str(rp)
