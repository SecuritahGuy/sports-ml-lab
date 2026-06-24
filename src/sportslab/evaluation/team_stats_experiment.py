"""Team stat features experiment — rolling origin, vs incumbent."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import build_team_regression_overrides
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, NEUTRAL_COLUMN, TARGET_COLUMN
from sportslab.features.ratings import compute_elo_features
from sportslab.features.team_stats import TEAM_STATS_COLUMNS, compute_team_stats_features

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0


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


def _logistic_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def run_team_stats_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/team_stats.md",
) -> str:
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    team_overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS
    )

    print("\n=== Computing Elo features ===")
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE, mov_scale=BEST_MOV_SCALE, mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        team_regression_overrides=team_overrides,
    )

    print("\n=== Computing team stat features ===")
    df_all = compute_team_stats_features(df_elo)
    added = [c for c in df_all.columns if c not in df_elo.columns]
    print(f"  Added {len(added)} team stat columns")

    df_all = _filter_df(df_all)
    ts_available = [c for c in TEAM_STATS_COLUMNS if c in df_all.columns]
    print(f"  Team stat features ({len(ts_available)}): {ts_available[:6]}...")

    elo_prob = df_all["elo_prob"].values
    y = df_all[TARGET_COLUMN].astype(float).values

    print("\n=== Rolling-Origin Evaluation ===")
    platt_results: list[dict] = []
    ts_only_results: list[dict] = []
    elo_ts_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values

        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]

        ts_train = df_all.loc[is_train, ts_available].values
        ts_val = df_all.loc[is_val, ts_available].values

        # 1. Platt incumbent
        platt = _fit_platt(train_elo, train_y_)
        platt_val_proba = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_val_proba)
        platt_results.append({"train_seasons": train_seasons, "val_season": val_season, "log_loss": platt_m["log_loss"], "metrics": platt_m, "model": platt})

        # 2. Team stats only
        ts_only_pipe = _logistic_model()
        ts_only_pipe.fit(ts_train, train_y_)
        ts_only_val_proba = ts_only_pipe.predict_proba(ts_val)[:, 1]
        ts_only_m = compute_classification_metrics(val_y_, ts_only_val_proba)
        ts_only_results.append({"train_seasons": train_seasons, "val_season": val_season, "log_loss": ts_only_m["log_loss"], "metrics": ts_only_m, "model": ts_only_pipe})

        # 3. Elo + team stats
        elo_ts_train = np.column_stack([train_elo, ts_train])
        elo_ts_val = np.column_stack([val_elo, ts_val])
        elo_ts_pipe = _logistic_model()
        elo_ts_pipe.fit(elo_ts_train, train_y_)
        elo_ts_val_proba = elo_ts_pipe.predict_proba(elo_ts_val)[:, 1]
        elo_ts_m = compute_classification_metrics(val_y_, elo_ts_val_proba)
        elo_ts_results.append({"train_seasons": train_seasons, "val_season": val_season, "log_loss": elo_ts_m["log_loss"], "metrics": elo_ts_m, "model": elo_ts_pipe})

        print(f"  Fold train={train_seasons} val={val_season}: platt={platt_m['log_loss']:.4f} ts_only={ts_only_m['log_loss']:.4f} elo+ts={elo_ts_m['log_loss']:.4f}")

    avg_platt = float(np.mean([r["log_loss"] for r in platt_results]))
    avg_ts_only = float(np.mean([r["log_loss"] for r in ts_only_results]))
    avg_elo_ts = float(np.mean([r["log_loss"] for r in elo_ts_results]))

    print(f"\n  Platt (incumbent):   {avg_platt:.4f}")
    print(f"  Team stats only:     {avg_ts_only:.4f}")
    print(f"  Elo + team stats:    {avg_elo_ts:.4f}")

    # ── Holdout ──
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    is_train_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    ts_full = df_all.loc[is_train_full, ts_available].values
    ts_hold = df_all.loc[is_hold, ts_available].values

    platt_full = _fit_platt(train_elo_full, train_y_full)
    platt_hold_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, platt_hold_proba)

    ts_only_final = _logistic_model()
    ts_only_final.fit(ts_full, train_y_full)
    ts_only_hold_proba = ts_only_final.predict_proba(ts_hold)[:, 1]
    hold_ts_only_m = compute_classification_metrics(hold_y, ts_only_hold_proba)

    elo_ts_full = np.column_stack([train_elo_full, ts_full])
    elo_ts_hold = np.column_stack([hold_elo, ts_hold])
    elo_ts_final = _logistic_model()
    elo_ts_final.fit(elo_ts_full, train_y_full)
    elo_ts_hold_proba = elo_ts_final.predict_proba(elo_ts_hold)[:, 1]
    hold_elo_ts_m = compute_classification_metrics(hold_y, elo_ts_hold_proba)

    print(f"  Platt (incumbent):   {hold_platt_m['log_loss']:.4f}")
    print(f"  Team stats only:     {hold_ts_only_m['log_loss']:.4f}")
    print(f"  Elo + team stats:    {hold_elo_ts_m['log_loss']:.4f}")

    # ── Baselines ──
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # ── Subset: QB change ──
    from sportslab.features.qb import compute_qb_features
    df_with_qb = compute_qb_features(df_all)
    hold_qb_changed = df_with_qb.loc[is_hold, "home_qb_changed"].fillna(0).astype(bool).values
    hold_qb_stable = ~hold_qb_changed

    # ── Report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    def _maybe_winner():
        if hold_elo_ts_m["log_loss"] < hold_platt_m["log_loss"] and avg_elo_ts < avg_platt:
            return "Elo + Team Stats"
        return None

    winner = _maybe_winner()

    with open(rp, "w") as f:
        f.write("# Team Stats Features Experiment\n\n")
        f.write("*Testing whether rolling team stat aggregates (yards, fantasy pts, sacks) improve on the incumbent.*\n\n")
        f.write("## Data Source\n\n")
        f.write("| Source | Description | Coverage |\n")
        f.write("|--------|-------------|----------|\n")
        f.write("| nflreadpy.load_player_stats | Weekly player stats aggregated to team level | 2021–2025 |\n\n")
        f.write("## Features\n\n")
        f.write("| Feature | Windows | Description |\n")
        f.write("|---------|---------|-------------|\n")
        f.write("| off_yds | 3, 5 | Team total offensive yards (pass+rush) |\n")
        f.write("| def_yds_allowed | 3, 5 | Opponent offensive yards |\n")
        f.write("| fantasy_pts | 3, 5 | Total fantasy points (all players) |\n")
        f.write("| def_sacks | 3, 5 | Defensive sacks |\n")
        f.write("| off_yds_net | 3, 5 | Home offense − away defense |\n\n")
        f.write(f"Total feature columns: {len(ts_available)}\n\n")
        f.write("## Incumbent Params\n\n")
        f.write(f"K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}, decay={BEST_DECAY}, qb_bonus={BEST_QB_BONUS}\n\n")
        f.write("## Results\n\n")
        f.write("### Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")
        f.write(f"| Platt (incumbent) | {avg_platt:.4f} | {platt_results[0]['log_loss']:.4f} | {platt_results[1]['log_loss']:.4f} | {platt_results[2]['log_loss']:.4f} |\n")
        f.write(f"| Team stats only | {avg_ts_only:.4f} | {ts_only_results[0]['log_loss']:.4f} | {ts_only_results[1]['log_loss']:.4f} | {ts_only_results[2]['log_loss']:.4f} |\n")
        f.write(f"| Elo + Team Stats | {avg_elo_ts:.4f} | {elo_ts_results[0]['log_loss']:.4f} | {elo_ts_results[1]['log_loss']:.4f} | {elo_ts_results[2]['log_loss']:.4f} |\n\n")
        f.write("### 2025 Holdout\n\n")
        f.write("| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n")
        f.write("|-------|---------|------------|----------|----------|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        for name, hm in [("Platt (incumbent)", hold_platt_m), ("Team stats only", hold_ts_only_m), ("Elo + Team Stats", hold_elo_ts_m)]:
            f.write(f"| {name} | {hm['log_loss']:.4f} | {hm['brier_score']:.4f} | {hm['accuracy']:.4f} | {hm['roc_auc']:.4f} |\n")
        f.write("\n")

        if hold_qb_changed.sum() >= 5:
            qb_changed_ll = compute_classification_metrics(hold_y[hold_qb_changed], platt_hold_proba[hold_qb_changed])["log_loss"]
            qb_stable_ll = compute_classification_metrics(hold_y[hold_qb_stable], platt_hold_proba[hold_qb_stable])["log_loss"]
            f.write("### QB-Change Subset (Platt)\n\n")
            f.write(f"QB changed (n={int(hold_qb_changed.sum())}): {qb_changed_ll:.4f} | QB stable (n={int(hold_qb_stable.sum())}): {qb_stable_ll:.4f}\n\n")

        if winner:
            f.write(f"**{winner} beats incumbent!** New research champion.\n")
        else:
            f.write("**Incumbent remains champion.** No team-stat model beat it.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
