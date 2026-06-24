"""Glicko rating system experiment — rolling origin vs incumbent.

Glicko-1 extends Elo with a rating deviation (RD) that tracks per-team
uncertainty.  High RD → conservative predictions — naturally handles
early-season, QB-change, and new-team uncertainty.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
    qb_change_across_seasons,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, NEUTRAL_COLUMN, TARGET_COLUMN
from sportslab.features.glicko import compute_glicko_features
from sportslab.features.ratings import compute_od_elo_features

# Incumbent parameters
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

# Glicko grid
HFA_GRID = [20, 30, 40, 50]
INITIAL_RD_GRID = [200, 250, 300, 350, 400, 500]
SYSTEM_C_GRID = [125, 150, 175, 200, 225, 250]
QB_RD_BONUS_GRID = [0.0, 50.0, 100.0]


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


def _invert_change_map(
    change_map: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Invert season->teams map to team->seasons map."""
    inverted: dict[str, list[int]] = {}
    for season, teams in change_map.items():
        for team in teams:
            inverted.setdefault(team, []).append(int(season))
    return inverted


def run_glicko_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/glicko_rating.md",
) -> str:
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # --- Compute incumbent baseline ---
    team_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )
    print("\n=== Computing Incumbent (O/D Elo) ===")
    df_inc = compute_od_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
        k_off=BEST_K_OFF,
        k_def=BEST_K_DEF,
        team_regression_overrides=team_overrides,
    )

    # --- QB change map for Glicko ---
    change_map_season_teams = qb_change_across_seasons(df_raw)
    qb_change_map = _invert_change_map(change_map_season_teams)

    # --- Build Glicko grid ---
    glicko_results = {}
    for hfa in HFA_GRID:
        for init_rd in INITIAL_RD_GRID:
            for sys_c in SYSTEM_C_GRID:
                for qb_bonus in QB_RD_BONUS_GRID:
                    label = f"hfa{hfa}_rd{init_rd}_c{sys_c}_qb{qb_bonus:.0f}"
                    print(f"\n=== Glicko {label} ===")
                    glicko_results[label] = compute_glicko_features(
                        df_raw,
                        home_advantage=hfa,
                        initial_rd=float(init_rd),
                        system_constant_c=float(sys_c),
                        mov_type=BEST_MOV_TYPE,
                        mov_scale=BEST_MOV_SCALE,
                        mov_cap=BEST_MOV_CAP,
                        qb_rd_bonus=float(qb_bonus),
                        qb_change_map=qb_change_map,
                    )

    df_inc_f = _filter_df(df_inc)
    glicko_f = {lbl: _filter_df(d) for lbl, d in glicko_results.items()}

    y = df_inc_f[TARGET_COLUMN].astype(float).values

    # --- Rolling-Origin Validation ---
    print("\n=== Rolling-Origin Evaluation ===")
    print(f"Total configurations: {len(glicko_f)}")

    def _eval_fold_glicko(df, train_s, val_s, prob_col="glicko_prob"):
        is_train = df["season"].isin(train_s).values
        is_val = (df["season"] == val_s).values
        train_probs = df[prob_col].values[is_train]
        train_y_ = y[is_train].astype(int)
        val_probs = df[prob_col].values[is_val]
        val_y_ = y[is_val]
        platt = _fit_platt(train_probs, train_y_)
        val_proba = platt.predict_proba(val_probs.reshape(-1, 1))[:, 1]
        m = compute_classification_metrics(val_y_, val_proba)
        return m["log_loss"]

    def _eval_fold_inc(df, train_s, val_s):
        return _eval_fold_glicko(df, train_s, val_s, prob_col="elo_prob")

    fold_results: dict[str, list[float]] = {lbl: [] for lbl in glicko_f}
    fold_results["incumbent"] = []

    for train_s, val_s in ROLLING_FOLDS:
        fold_results["incumbent"].append(_eval_fold_inc(df_inc_f, train_s, val_s))
        for lbl in glicko_f:
            fold_results[lbl].append(_eval_fold_glicko(glicko_f[lbl], train_s, val_s))

        line = f"  Fold train={train_s} val={val_s}: inc={fold_results['incumbent'][-1]:.4f}"
        print(line)

    print("\n=== Average Validation Log Loss ===")
    avgs = {}
    for key, vals in fold_results.items():
        avg = float(np.mean(vals))
        avgs[key] = avg

    sorted_keys = sorted(avgs, key=avgs.get)
    print(f"  {'Model':<35s} {'Avg Val LL':>10s}")
    print(f"  {'-' * 35} {'-' * 10}")
    for key in sorted_keys:
        marker = " <<<" if key == "incumbent" else ""
        if key == "incumbent":
            print(f"  {key:<35s} {avgs[key]:>10.4f}{marker}")
    print()
    for key in sorted_keys:
        if key != "incumbent":
            print(f"  {key:<35s} {avgs[key]:>10.4f}")
    print()

    # Top 10 Glicko configs by validation
    glicko_keys = [k for k in sorted_keys if k != "incumbent"]
    print("=== Top 10 Glicko Configurations (avg val LL) ===")
    print(f"  {'Config':<35s} {'Avg Val LL':>10s}")
    print(f"  {'-' * 35} {'-' * 10}")
    for key in glicko_keys[:10]:
        print(f"  {key:<35s} {avgs[key]:>10.4f}")

    best_glicko_key = glicko_keys[0] if glicko_keys else None

    # --- 2025 Holdout ---
    print("\n=== 2025 Holdout ===")
    is_hold = (df_inc_f["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]

    is_train_full = df_inc_f["season"].isin([2021, 2022, 2023, 2024]).values
    train_y_full = y[is_train_full].astype(int)

    def _holdout_glicko(df, prob_col="glicko_prob"):
        train_probs = df[prob_col].values[is_train_full]
        hold_probs = df[prob_col].values[is_hold]
        platt = _fit_platt(train_probs, train_y_full)
        hold_proba = platt.predict_proba(hold_probs.reshape(-1, 1))[:, 1]
        return compute_classification_metrics(hold_y, hold_proba)

    def _holdout_inc(df):
        return _holdout_glicko(df, prob_col="elo_prob")

    hold_inc = _holdout_inc(df_inc_f)
    hold_glicko = {lbl: _holdout_glicko(glicko_f[lbl]) for lbl in glicko_f}

    print(f"  Incumbent: {hold_inc['log_loss']:.4f}")
    print(f"  Glicko best: {min(hm['log_loss'] for hm in hold_glicko.values()):.4f}")

    best_hold_key = min(hold_glicko, key=lambda k: hold_glicko[k]["log_loss"])
    print(f"    = {best_hold_key}: {hold_glicko[best_hold_key]['log_loss']:.4f}")

    # Baselines
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # --- Report ---
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Glicko Rating System Experiment\n\n")
        f.write(
            "*Testing whether the Glicko-1 rating system (with uncertainty"
            " tracking via RD) improves on O/D Elo + Platt.*\n\n"
        )

        f.write("## Method\n\n")
        f.write("Glicko-1 extends Elo with a Rating Deviation (RD) per team.\n")
        f.write(
            "- **g(RD)** = 1 / sqrt(1 + 3q²RD²/π²) — scales prediction"
            " toward 0.5 when opponent RD is high\n"
        )
        f.write("- High RD → conservative predictions (early season, QB change, new team)\n")
        f.write("- Between seasons: RD ← sqrt(RD² + c²) — uncertainty grows during offseason\n")
        f.write(
            f"- MOV: {BEST_MOV_TYPE}, scale={BEST_MOV_SCALE}, cap={BEST_MOV_CAP}"
            f" (same as incumbent)\n\n"
        )

        f.write("## Grid\n\n")
        f.write("| Parameter | Values |\n")
        f.write("|-----------|--------|\n")
        f.write(f"| HFA | {HFA_GRID} |\n")
        f.write(f"| Initial RD | {INITIAL_RD_GRID} |\n")
        f.write(f"| System constant c | {SYSTEM_C_GRID} |\n")
        f.write(f"| QB RD bonus | {QB_RD_BONUS_GRID} |\n")
        total_configs = (
            len(HFA_GRID) * len(INITIAL_RD_GRID) * len(SYSTEM_C_GRID) * len(QB_RD_BONUS_GRID)
        )
        f.write(f"\nTotal: {total_configs} configurations\n\n")

        f.write("## Rolling-Origin Validation\n\n")
        f.write(
            f"| {'Config':<35s} | {'Avg Val LL':>10s}"
            f" | {'Fold1':>7s} | {'Fold2':>7s} | {'Fold3':>7s} |\n"
        )
        f.write(f"| {'-' * 35} | {'-' * 10} | {'-' * 7} | {'-' * 7} | {'-' * 7} |\n")
        for key in sorted_keys:
            if key == "incumbent":
                name = "Incumbent (O/D Elo + Platt)"
            else:
                name = key
            f.write(f"| {name:<35s} | {avgs[key]:>10.4f}")
            for i in range(3):
                f.write(f" | {fold_results[key][i]:>7.4f}")
            f.write(" |\n")
        f.write("\n")

        f.write("## 2025 Holdout\n\n")
        f.write("| Config | Hold LL | Brier | AUC | Acc |\n")
        f.write("|--------|---------|-------|-----|-----|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(
            f"| Incumbent (O/D Elo + Platt) | {hold_inc['log_loss']:.4f}"
            f" | {hold_inc['brier_score']:.4f}"
            f" | {hold_inc['roc_auc']:.4f}"
            f" | {hold_inc['accuracy']:.4f} |\n"
        )

        # Top 30 Glicko configs on holdout
        hold_sorted = sorted(hold_glicko, key=lambda k: hold_glicko[k]["log_loss"])
        for key in hold_sorted[:30]:
            hm = hold_glicko[key]
            f.write(
                f"| Glicko {key} | {hm['log_loss']:.4f}"
                f" | {hm['brier_score']:.4f}"
                f" | {hm['roc_auc']:.4f}"
                f" | {hm['accuracy']:.4f} |\n"
            )
        f.write("\n")

        # Decision
        inc_hold_ll = hold_inc["log_loss"]
        best_g_hold = min(hm["log_loss"] for hm in hold_glicko.values())
        best_val_key = min(hold_glicko, key=lambda k: avgs[k]) if best_glicko_key else None

        if best_g_hold < inc_hold_ll and best_val_key and avgs[best_val_key] < avgs["incumbent"]:
            f.write(
                f"**Glicko ({best_hold_key}) beats the incumbent on both"
                f" validation and holdout!**\n"
            )
        elif best_g_hold < inc_hold_ll:
            f.write(
                f"**Glicko ({best_hold_key}) beats the incumbent on holdout"
                f" but not on validation.** (*Diagnostic only — not promoted.*)\n"
            )
            f.write(
                f"Best Glicko val: {avgs[best_val_key]:.4f} vs incumbent {avgs['incumbent']:.4f}\n"
            )
        else:
            f.write("**Glicko does not beat the incumbent.**\n")

        f.write(f"\nIncumbent holdout LL: {inc_hold_ll:.4f}\n")
        f.write(f"Best Glicko holdout LL: {best_g_hold:.4f}\n")
        f.write(f"Gap: {best_g_hold - inc_hold_ll:+.4f}\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
