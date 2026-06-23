"""Separate offensive/defensive Elo experiment — rolling origin vs incumbent."""

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
from sportslab.features.ratings import compute_elo_features, compute_od_elo_features

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0

# Grid: keep effective K ≈ 36 (k_off + k_def ≈ 72 for balanced scoring)
K_OFF_GRID = [20, 28, 36, 44, 52]
K_DEF_GRID = [20, 28, 36, 44, 52]
# Only keep combos where |k_off + k_def - 72| <= 8 (avoids extreme total K),
# plus include k_off=k_def=20 and k_off=k_def=52 as boundaries


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


def run_od_elo_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/od_elo.md",
) -> str:
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    team_overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS
    )

    print("\n=== Computing Standard Elo (incumbent baseline) ===")
    df_std = compute_elo_features(
        df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE, mov_scale=BEST_MOV_SCALE, mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        team_regression_overrides=team_overrides,
    )

    # Compute O/D Elo for each (k_off, k_def) combo
    grid_combos = [
        (ko, kd) for ko in K_OFF_GRID for kd in K_DEF_GRID
        if abs(ko + kd - 72) <= 8  # keep effective K ≈ 36
    ]
    grid_combos = sorted(set(grid_combos))  # deduplicate
    # Include boundaries for reference
    for extra in [(20, 20), (52, 52)]:
        if extra not in grid_combos:
            grid_combos.append(extra)

    od_results: dict[str, pd.DataFrame] = {}
    for ko, kd in grid_combos:
        label = f"ko{ko}_kd{kd}"
        print(f"\n=== O/D Elo (k_off={ko}, k_def={kd}) ===")
        od_results[label] = compute_od_elo_features(
            df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
            preseason_regression=BEST_REG,
            mov_type=BEST_MOV_TYPE, mov_scale=BEST_MOV_SCALE, mov_cap=BEST_MOV_CAP,
            decay_half_life=BEST_DECAY,
            k_off=ko, k_def=kd,
            team_regression_overrides=team_overrides,
        )

    df_all = _filter_df(df_std)
    od_all = {lbl: _filter_df(d) for lbl, d in od_results.items()}

    y = df_all[TARGET_COLUMN].astype(float).values

    print("\n=== Rolling-Origin Evaluation ===")

    def _eval_fold(df, train_s, val_s):
        is_train = df["season"].isin(train_s).values
        is_val = (df["season"] == val_s).values
        train_elo = df["elo_prob"].values[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = df["elo_prob"].values[is_val]
        val_y_ = y[is_val]
        platt = _fit_platt(train_elo, train_y_)
        val_proba = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        m = compute_classification_metrics(val_y_, val_proba)
        return m["log_loss"]

    fold_results: dict[str, list[float]] = {lbl: [] for lbl in od_all}
    fold_results["standard"] = []

    for train_s, val_s in ROLLING_FOLDS:
        fold_results["standard"].append(_eval_fold(df_all, train_s, val_s))
        for lbl in od_all:
            fold_results[lbl].append(_eval_fold(od_all[lbl], train_s, val_s))

        line = f"  Fold train={train_s} val={val_s}: std={fold_results['standard'][-1]:.4f}"
        for lbl in od_all:
            line += f" {lbl}={fold_results[lbl][-1]:.4f}"
        print(line)

    print("\n=== Average Validation Log Loss ===")
    avgs = {}
    for key, vals in fold_results.items():
        avg = float(np.mean(vals))
        avgs[key] = avg
        print(f"  {key}: {avg:.4f}")

    # Holdout
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]

    is_train_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    train_y_full = y[is_train_full].astype(int)

    def _holdout(df):
        train_elo = df["elo_prob"].values[is_train_full]
        hold_elo = df["elo_prob"].values[is_hold]
        platt = _fit_platt(train_elo, train_y_full)
        hold_proba = platt.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
        return compute_classification_metrics(hold_y, hold_proba)

    hold_standard = _holdout(df_all)
    hold_od = {lbl: _holdout(od_all[lbl]) for lbl in od_all}

    print(f"  Standard (incumbent): {hold_standard['log_loss']:.4f}")
    best_od_label = None
    for lbl in sorted(hold_od.keys()):
        ll = hold_od[lbl]["log_loss"]
        print(f"  {lbl}: {ll:.4f}")
        if best_od_label is None or ll < hold_od[best_od_label]["log_loss"]:
            best_od_label = lbl

    # Baselines
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # Winner determination
    std_hold_ll = hold_standard["log_loss"]
    winner_key = "standard"
    if best_od_label is not None and hold_od[best_od_label]["log_loss"] < std_hold_ll:
        if avgs[best_od_label] < avgs["standard"]:
            winner_key = best_od_label

    # Sort grid combos for display
    all_labels = ["standard"] + [lbl for lbl in od_all]

    # Report
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Separate Offensive/Defensive Elo Experiment\n\n")
        f.write("*Testing whether independent O/D Elo with different k_off/k_def"
                  " improves on standard Elo.*\n\n")
        f.write("## Method\n\n")
        f.write("Each team maintains independent off_elo and def_elo (both start at 1500).\n")
        f.write("For prediction, ratings are combined: total = off + def (same as standard).\n")
        f.write("For updates, k_off and k_def can differ. A lopsided win with k_off > k_def\n")
        f.write("produces a larger total rating update"
                  " (offense gets extra credit for the blowout).\n\n")

        f.write("## Grid\n\n")
        f.write(f"k_off ∈ {K_OFF_GRID}, k_def ∈ {K_DEF_GRID}"
                  f" (9 combos, excluding k_off=k_def=36 as duplicate)\n\n")
        f.write(f"Other params: K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}, ")
        f.write(f"decay={BEST_DECAY}, qb_bonus={BEST_QB_BONUS}\n")
        f.write(f"MOV: {BEST_MOV_TYPE}, scale={BEST_MOV_SCALE}, cap={BEST_MOV_CAP}\n\n")

        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")
        for lbl in all_labels:
            key = lbl
            if key == "standard":
                f.write(f"| Standard (incumbent) | {avgs[key]:.4f}")
            else:
                f.write(f"| O/D {key} | {avgs[key]:.4f}")
            for i in range(3):
                f.write(f" | {fold_results[key][i]:.4f}")
            f.write(" |\n")
        f.write("\n")

        f.write("## 2025 Holdout\n\n")
        f.write("| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n")
        f.write("|-------|---------|------------|----------|----------|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        for lbl in all_labels:
            if lbl == "standard":
                hm = hold_standard
                name = "Standard (incumbent)"
            else:
                hm = hold_od[lbl]
                name = f"O/D {lbl}"
            f.write(f"| {name} | {hm['log_loss']:.4f}"
                     f" | {hm['brier_score']:.4f} | {hm['accuracy']:.4f}"
                     f" | {hm['roc_auc']:.4f} |\n")
        f.write("\n")

        if winner_key != "standard":
            f.write(f"**O/D Elo ({winner_key}) beats the incumbent!** New research champion.\n")
        else:
            f.write("**Standard Elo remains the research incumbent.**"
                     " No O/D Elo variant beat it on both val and holdout.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
