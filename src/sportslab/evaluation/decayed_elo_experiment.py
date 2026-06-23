"""Exponential decay Elo experiment — testing whether weighting recent games more
heavily improves predictive performance over standard MOV Elo."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from sportslab.features.ratings import (
    MOV_CAPPED_LINEAR,
    compute_elo_features,
)

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

# Frozen incumbent MOV Elo params (best known)
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.20
BEST_MOV_TYPE = MOV_CAPPED_LINEAR
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0

# Grid candidates
K_CANDIDATES = [20, 24, 28, 32, 36, 40, 44, 48]
HFA_CANDIDATES = [30, 40]
REG_CANDIDATES = [0.0, 0.20]
DECAY_CANDIDATES = [None, 32, 16, 8, 4]


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(train_prob.reshape(-1, 1), train_y)
    return pipe


def run_decayed_elo_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/decayed_elo.md",
) -> str:
    """Rolling-origin grid over K, HFA, reg, decay_half_life.

    No 2025 holdout access during search.  Final 2025 evaluation
    once after best config is selected.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Grid search ──
    print("=== Decayed Elo Grid Search ===")
    total = len(K_CANDIDATES) * len(HFA_CANDIDATES) * len(REG_CANDIDATES) * len(DECAY_CANDIDATES)
    count = 0
    best: dict[str, Any] = {"avg_val_log_loss": float("inf"), "params": None}
    all_results: list[dict[str, Any]] = []

    for k in K_CANDIDATES:
        for hfa in HFA_CANDIDATES:
            for reg in REG_CANDIDATES:
                for decay in DECAY_CANDIDATES:
                    count += 1
                    edf = compute_elo_features(
                        df_raw,
                        k_factor=k,
                        home_advantage=hfa,
                        preseason_regression=reg,
                        mov_type=BEST_MOV_TYPE,
                        mov_scale=BEST_MOV_SCALE,
                        mov_cap=BEST_MOV_CAP,
                        decay_half_life=decay,
                    )
                    edf = _filter_df(edf)

                    elo_prob = edf["elo_prob"].values
                    y = edf[TARGET_COLUMN].astype(float).values

                    fold_lls: list[float] = []
                    fold_details: list[dict[str, Any]] = []

                    for train_seasons, val_season in ROLLING_FOLDS:
                        is_val = (edf["season"] == val_season).values
                        val_loss = float(log_loss(y[is_val], elo_prob[is_val]))
                        fold_lls.append(val_loss)
                        fold_details.append(
                            {
                                "train_seasons": train_seasons,
                                "val_season": val_season,
                                "val_log_loss": round(val_loss, 5),
                            }
                        )

                    avg_ll = float(np.mean(fold_lls))

                    entry = {
                        "k_factor": k,
                        "home_advantage": hfa,
                        "preseason_regression": reg,
                        "decay_half_life": decay,
                        "avg_val_log_loss": round(avg_ll, 5),
                        "fold_details": fold_details,
                    }
                    all_results.append(entry)

                    if avg_ll < best["avg_val_log_loss"]:
                        best = {
                            "avg_val_log_loss": avg_ll,
                            "params": {
                                "k_factor": k,
                                "home_advantage": hfa,
                                "preseason_regression": reg,
                                "decay_half_life": decay,
                            },
                            "fold_details": fold_details,
                        }

                    decay_str = f"decay={decay}" if decay is not None else "decay=None"
                    print(
                        f"  [{count}/{total}] K={k} HFA={hfa} reg={reg} {decay_str}"
                        f"  avg_ll={avg_ll:.5f}"
                    )

    all_results.sort(key=lambda x: x["avg_val_log_loss"])
    best_params = best["params"]
    best_avg_ll = round(best["avg_val_log_loss"], 5)
    print(
        f"\nBest params (by avg val log loss):"
        f" K={best_params['k_factor']}, HFA={best_params['home_advantage']},"
        f" reg={best_params['preseason_regression']},"
        f" decay_half_life={best_params['decay_half_life']}"
    )
    print(f"  Avg val log loss: {best_avg_ll:.5f}")

    # ── Final Elo with best params ──
    print("\n=== Final Elo with best params ===")
    best_elo = compute_elo_features(
        df_raw,
        k_factor=best_params["k_factor"],
        home_advantage=best_params["home_advantage"],
        preseason_regression=best_params["preseason_regression"],
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=best_params["decay_half_life"],
    )
    best_elo = _filter_df(best_elo)

    is_hold = (best_elo["season"] == HOLDOUT_SEASON).values
    hold_y = best_elo.loc[is_hold, TARGET_COLUMN].astype(float).values
    hold_prob = best_elo.loc[is_hold, "elo_prob"].values
    hold_metrics = compute_classification_metrics(hold_y, hold_prob)
    print(f"  Holdout (2025) log loss: {hold_metrics['log_loss']:.4f}")

    # ── Rolling-origin calibration ──
    print("\n=== Rolling-Origin Calibration ===")
    elo_prob = best_elo["elo_prob"].values
    y = best_elo[TARGET_COLUMN].astype(float).values
    platt_folds: list[dict[str, Any]] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = best_elo["season"].isin(train_seasons).values
        is_val = (best_elo["season"] == val_season).values
        train_p = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_p = elo_prob[is_val]
        val_y_ = y[is_val]
        platt = _fit_platt(train_p, train_y_)
        platt_val = platt.predict_proba(val_p.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_val)
        platt_folds.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_m["log_loss"],
            }
        )

    platt_avg = float(np.mean([f["log_loss"] for f in platt_folds]))
    print(f"  Platt: avg val LL={platt_avg:.4f}")

    # Incumbent (current best MOV Elo + Platt)
    print("\n=== Incumbent Comparison ===")
    inc_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
    )
    inc_elo = _filter_df(inc_elo)
    is_train_full_inc = inc_elo["season"].isin([2021, 2022, 2023, 2024]).values
    hold_y_inc = inc_elo.loc[is_hold, TARGET_COLUMN].astype(float).values
    hold_p_inc = inc_elo.loc[is_hold, "elo_prob"].values
    train_p_inc = inc_elo.loc[is_train_full_inc, "elo_prob"].values
    train_y_inc = inc_elo.loc[is_train_full_inc, TARGET_COLUMN].astype(float).values
    platt_inc = _fit_platt(train_p_inc, train_y_inc)
    platt_hold_inc = platt_inc.predict_proba(hold_p_inc.reshape(-1, 1))[:, 1]
    inc_hold_metrics = compute_classification_metrics(hold_y_inc, platt_hold_inc)

    # Full 2021-2024 Platt for best decayed Elo
    is_train_full = best_elo["season"].isin([2021, 2022, 2023, 2024]).values
    train_p_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    platt_full = _fit_platt(train_p_full, train_y_full)
    platt_hold = platt_full.predict_proba(hold_prob.reshape(-1, 1))[:, 1]
    platt_hold_met = compute_classification_metrics(hold_y, platt_hold)

    # Baselines
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = float(y[is_train_full].mean())
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    print(f"  Best decayed raw Elo holdout: {hold_metrics['log_loss']:.4f}")
    print(f"  Best decayed Platt holdout:   {platt_hold_met['log_loss']:.4f}")
    print(f"  Incumbent (MOV+Platt) holdout: {inc_hold_metrics['log_loss']:.4f}")

    # ── Write report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Decayed Elo Experiment\n\n")
        f.write("*Testing exponential decay toward mean for Elo ratings.*\n\n")

        f.write("## Motivation\n\n")
        f.write(
            "Standard Elo weights all past games equally across a team's history.\n"
            "Exponential decay decreases the weight of older games, so recent form\n"
            "matters more.  After each game, both teams' ratings decay toward the\n"
            f"mean ({BEST_MOV_TYPE} = {BEST_K}).\n\n"
        )

        f.write("## Method\n\n")
        f.write("After each game's rating update, apply:\n\n")
        f.write("```\n")
        f.write("decay = 2^(-1 / half_life)\n")
        f.write("rating = 1500 + (rating - 1500) * decay\n")
        f.write("```\n\n")
        f.write("`half_life` = number of games to halve deviation from 1500.\n")
        f.write("Lower half_life = faster mean reversion.\n\n")

        f.write("## Parameter Grid\n\n")
        f.write("| Parameter | Candidates |\n")
        f.write("|-----------|------------|\n")
        f.write(f"| K-factor | {K_CANDIDATES} |\n")
        f.write(f"| HFA | {HFA_CANDIDATES} |\n")
        f.write(f"| Regression | {REG_CANDIDATES} |\n")
        f.write(f"| Decay half-life | {DECAY_CANDIDATES} (games) |\n")
        f.write(
            f"| MOV (frozen) | {BEST_MOV_TYPE}, scale={BEST_MOV_SCALE}, cap={BEST_MOV_CAP} |\n\n"
        )
        f.write(f"Total combinations searched: {len(all_results)}\n\n")

        best_platt_str = f"{platt_hold_met['log_loss']:.4f}"
        inc_platt_str = f"{inc_hold_metrics['log_loss']:.4f}"

        f.write("## Best Configuration (by avg val LL across folds)\n\n")
        f.write(
            f"- **K={best_params['k_factor']}, HFA={best_params['home_advantage']}, "
            f"reg={best_params['preseason_regression']}**\n"
        )
        d_str = (
            str(best_params["decay_half_life"])
            if best_params["decay_half_life"] is not None
            else "None"
        )
        f.write(f"- **Decay half-life: {d_str} games**\n")
        f.write(f"- Average validation log loss: {best_avg_ll:.5f}\n")
        f.write(f"- Holdout (2025) raw Elo: {hold_metrics['log_loss']:.4f}\n")
        f.write(f"- Holdout (2025) + Platt: {platt_hold_met['log_loss']:.4f}\n")
        f.write(f"- Incumbent (MOV+Platt) holdout: {inc_hold_metrics['log_loss']:.4f}\n\n")

        f.write("## Top 8 Configurations\n\n")
        f.write("| Rank | K | HFA | Reg | Decay | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|------|---|-----|-----|-------|-----------|-------|-------|-------|\n")
        for rank, e in enumerate(all_results[:8], 1):
            fd = e["fold_details"]
            d_str = str(e["decay_half_life"]) if e["decay_half_life"] is not None else "None"
            f.write(
                f"| {rank} | {e['k_factor']} | {e['home_advantage']}"
                f" | {e['preseason_regression']} | {d_str}"
                f" | {e['avg_val_log_loss']} | {fd[0]['val_log_loss']}"
                f" | {fd[1]['val_log_loss']} | {fd[2]['val_log_loss']} |\n"
            )
        f.write("\n")

        f.write("## Validation Comparison\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")
        f.write(
            f"| Decayed Elo raw | {best_avg_ll:.4f}"
            f" | {best['fold_details'][0]['val_log_loss']:.4f}"
            f" | {best['fold_details'][1]['val_log_loss']:.4f}"
            f" | {best['fold_details'][2]['val_log_loss']:.4f} |\n"
        )
        f.write(
            f"| Decayed + Platt | {platt_avg:.4f}"
            f" | {platt_folds[0]['log_loss']:.4f}"
            f" | {platt_folds[1]['log_loss']:.4f}"
            f" | {platt_folds[2]['log_loss']:.4f} |\n\n"
        )

        f.write("## Holdout (2025) Comparison\n\n")
        f.write("| Model | Hold LL | Brier | Acc | AUC |\n")
        f.write("|-------|---------|-------|-----|-----|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | — | — | — |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | — |\n")
        f.write(f"| Incumbent (MOV+Platt) | {inc_platt_str} | — | — | — |\n")
        f.write(
            f"| Best decayed raw Elo | {hold_metrics['log_loss']:.4f}"
            f" | {hold_metrics['brier_score']:.4f}"
            f" | {hold_metrics['accuracy']:.4f}"
            f" | {hold_metrics['roc_auc']:.4f} |\n"
        )
        f.write(
            f"| Best decayed + Platt | {best_platt_str}"
            f" | {platt_hold_met['brier_score']:.4f}"
            f" | {platt_hold_met['accuracy']:.4f}"
            f" | {platt_hold_met['roc_auc']:.4f} |\n\n"
        )

        dec_str = (
            str(best_params["decay_half_life"])
            if best_params["decay_half_life"] is not None
            else "None"
        )
        if platt_hold_met["log_loss"] < inc_hold_metrics["log_loss"]:
            f.write("## Decision\n\n")
            f.write(
                f"✅ **Decayed Elo (K={best_params['k_factor']}, decay={dec_str})"
                f" + Platt beats the incumbent.**\n\n"
            )
            f.write(
                f"Holdout log loss {platt_hold_met['log_loss']:.4f}"
                f" vs incumbent {inc_hold_metrics['log_loss']:.4f}.\n\n"
            )
        else:
            f.write("## Decision\n\n")
            f.write("❌ **Decayed Elo does not beat the incumbent.**\n\n")
            f.write(f"Best decayed + Platt holdout: {platt_hold_met['log_loss']:.4f}\n")
            f.write(f"Incumbent (MOV+Platt) holdout: {inc_hold_metrics['log_loss']:.4f}\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- Decay is applied post-game, never pregame.\n")
        f.write("- Pregame Elo features are recorded before the update.\n")
        f.write("- Rolling-origin folds prevent 2025 holdout access.\n")
        f.write("- Platt calibration fitted only on training data.\n\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
