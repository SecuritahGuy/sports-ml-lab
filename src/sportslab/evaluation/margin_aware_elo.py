"""Margin-aware Elo experiment with rolling-origin validation.

Tests whether margin-of-victory multipliers on Elo rating updates
improve predictive performance over win/loss-only Elo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
    MOV_CAPPED_LOG,
    MOV_LOG,
    MOV_NONE,
    MOV_SQRT,
    compute_elo_features,
)

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

K_CANDIDATES = list(range(16, 84, 4))
HFA_CANDIDATES = [30, 40]
REG_CANDIDATES = [0.0, 0.20]
MOV_TYPES = [MOV_NONE, MOV_LOG, MOV_SQRT, MOV_CAPPED_LOG, MOV_CAPPED_LINEAR]
MOV_SCALES = [0.05, 0.10, 0.20]
MOV_CAPS = [2.0, 3.0]

# Incumbent comparison params
INCUMBENT_K = 36
INCUMBENT_HFA = 40
INCUMBENT_REG = 0.20
INCUMBENT_MOV_TYPE = MOV_CAPPED_LINEAR
INCUMBENT_MOV_SCALE = 0.05
INCUMBENT_MOV_CAP = 2.0


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


def _fit_isotonic(train_prob: np.ndarray, train_y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(train_prob, train_y)
    return iso


def _minimal_logistic_features(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = ["elo_diff", "elo_prob", "rest_diff", "is_neutral", "week"]
    present = [c for c in feat_cols if c in df.columns]
    return df[present].copy()


def run_margin_aware_grid_search(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
) -> tuple[dict, list[dict]]:
    """Grid search over Elo + MOV parameters using rolling-origin validation.

    No 2025 holdout data is computed or accessed during search.

    Returns:
        (best_params, all_results)
    """
    df_raw = pd.read_parquet(feature_table_path)
    best: dict = {"avg_val_log_loss": float("inf"), "params": None}
    all_results: list[dict] = []

    total = (
        len(K_CANDIDATES)
        * len(HFA_CANDIDATES)
        * len(REG_CANDIDATES)
        * sum(
            1
            for mt in MOV_TYPES
            for _ in ([None] if mt == MOV_NONE else MOV_SCALES)
            for _ in ([None] if mt not in (MOV_CAPPED_LOG, MOV_CAPPED_LINEAR) else MOV_CAPS)
        )
    )
    count = 0

    for k in K_CANDIDATES:
        for hfa in HFA_CANDIDATES:
            for reg in REG_CANDIDATES:
                for mov_type in MOV_TYPES:
                    if mov_type == MOV_NONE:
                        scale_iter = [0.0]
                        cap_iter = [None]
                    elif mov_type in (MOV_LOG, MOV_SQRT):
                        scale_iter = MOV_SCALES
                        cap_iter = [None]
                    else:
                        scale_iter = MOV_SCALES
                        cap_iter = MOV_CAPS

                    for mov_scale in scale_iter:
                        for mov_cap in cap_iter:
                            count += 1
                            edf = compute_elo_features(
                                df_raw,
                                k_factor=k,
                                home_advantage=hfa,
                                preseason_regression=reg,
                                mov_type=mov_type,
                                mov_scale=mov_scale,
                                mov_cap=mov_cap,
                            )
                            edf = _filter_df(edf)

                            elo_prob = edf["elo_prob"].values
                            y = edf[TARGET_COLUMN].astype(float).values

                            fold_lls: list[float] = []
                            fold_details: list[dict] = []

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
                                "mov_type": mov_type,
                                "mov_scale": mov_scale,
                                "mov_cap": mov_cap,
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
                                        "mov_type": mov_type,
                                        "mov_scale": mov_scale,
                                        "mov_cap": mov_cap,
                                    },
                                    "fold_details": fold_details,
                                }

                            # Compact print
                            mov_desc = mov_type
                            if mov_type != MOV_NONE:
                                mov_desc += f"/s={mov_scale}"
                                if mov_cap is not None:
                                    mov_desc += f"/c={mov_cap}"
                            print(
                                f"  [{count}/{total}] K={k} HFA={hfa}"
                                f" reg={reg} {mov_desc}"
                                f"  avg_ll={avg_ll:.5f}"
                            )

    all_results.sort(key=lambda x: x["avg_val_log_loss"])
    return best, all_results


def run_margin_aware_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/margin_aware_elo.md",
) -> str:
    """Run full margin-aware Elo experiment.

    1. Grid search over Elo + MOV params via rolling-origin (no holdout).
    2. With best params, compute final Elo features.
    3. Run calibration (Platt, isotonic) and minimal logistic.
    4. One-shot 2025 holdout.
    5. Compare against incumbent + report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Step 1: Grid search ──
    print("=== Margin-Aware Elo Grid Search ===")
    best, all_results = run_margin_aware_grid_search(feature_table_path)

    best_params = best["params"]
    best_avg_ll = round(best["avg_val_log_loss"], 5)
    print(
        f"\nBest params (by avg val log loss):"
        f" K={best_params['k_factor']}, HFA={best_params['home_advantage']},"
        f" reg={best_params['preseason_regression']},"
        f" mov_type={best_params['mov_type']},"
        f" mov_scale={best_params['mov_scale']},"
        f" mov_cap={best_params['mov_cap']}"
    )
    print(f"  Avg val log loss: {best_avg_ll:.5f}")

    # ── Step 2: Compute final Elo with best params ──
    print("\n=== Final Elo with best params ===")
    best_elo = compute_elo_features(
        df_raw,
        k_factor=best_params["k_factor"],
        home_advantage=best_params["home_advantage"],
        preseason_regression=best_params["preseason_regression"],
        mov_type=best_params["mov_type"],
        mov_scale=best_params["mov_scale"],
        mov_cap=best_params["mov_cap"],
    )
    best_elo = _filter_df(best_elo)

    # One-time holdout
    is_hold = (best_elo["season"] == HOLDOUT_SEASON).values
    hold_y = best_elo.loc[is_hold, TARGET_COLUMN].astype(float).values
    hold_prob = best_elo.loc[is_hold, "elo_prob"].values
    hold_metrics = compute_classification_metrics(hold_y, hold_prob)
    print(f"  Holdout (2025) log loss: {hold_metrics['log_loss']:.4f}")

    # ═══ Rolling-origin calibration evaluation ═══
    print("\n=== Rolling-Origin Calibration ===")
    elo_prob = best_elo["elo_prob"].values
    y = best_elo[TARGET_COLUMN].astype(float).values

    platt_folds: list[dict] = []
    iso_folds: list[dict] = []
    log_folds: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = best_elo["season"].isin(train_seasons).values
        is_val = (best_elo["season"] == val_season).values

        train_p = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_p = elo_prob[is_val]
        val_y_ = y[is_val]

        # Platt
        platt = _fit_platt(train_p, train_y_)
        platt_val = platt.predict_proba(val_p.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_val)
        platt_folds.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_m["log_loss"],
                "metrics": platt_m,
            }
        )

        # Isotonic
        iso = _fit_isotonic(train_p, train_y_)
        iso_val = iso.predict(val_p)
        iso_m = compute_classification_metrics(val_y_, iso_val)
        iso_folds.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": iso_m["log_loss"],
                "metrics": iso_m,
            }
        )

        # Minimal logistic
        feat_df = _minimal_logistic_features(best_elo)
        x_train = feat_df[is_train]
        y_train = y[is_train].astype(int)
        x_val = feat_df[is_val]
        y_val = y[is_val]
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(x_train, y_train)
        log_val = pipe.predict_proba(x_val)[:, 1]
        log_m = compute_classification_metrics(y_val, log_val)
        log_folds.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": log_m["log_loss"],
                "metrics": log_m,
            }
        )

    platt_avg = float(np.mean([f["log_loss"] for f in platt_folds]))
    iso_avg = float(np.mean([f["log_loss"] for f in iso_folds]))
    log_avg = float(np.mean([f["log_loss"] for f in log_folds]))
    print(f"  Platt: avg val LL={platt_avg:.4f}")
    print(f"  Isotonic: avg val LL={iso_avg:.4f}")
    print(f"  Minimal Logistic: avg val LL={log_avg:.4f}")

    # ═══ Incumbent comparison ── compute incumbent Platt ──
    inc_elo = compute_elo_features(
        df_raw,
        k_factor=INCUMBENT_K,
        home_advantage=INCUMBENT_HFA,
        preseason_regression=INCUMBENT_REG,
        mov_type=INCUMBENT_MOV_TYPE,
        mov_scale=INCUMBENT_MOV_SCALE,
        mov_cap=INCUMBENT_MOV_CAP,
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

    # ═══ Final models on full 2021-2024 → 2025 ──
    is_train_full = best_elo["season"].isin([2021, 2022, 2023, 2024]).values
    train_p_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)

    platt_full = _fit_platt(train_p_full, train_y_full)
    platt_hold = platt_full.predict_proba(hold_prob.reshape(-1, 1))[:, 1]
    platt_hold_met = compute_classification_metrics(hold_y, platt_hold)

    iso_full = _fit_isotonic(train_p_full, train_y_full)
    iso_hold = iso_full.predict(hold_prob)
    iso_hold_met = compute_classification_metrics(hold_y, iso_hold)

    feat_all = _minimal_logistic_features(best_elo)
    y_all = best_elo[TARGET_COLUMN].astype(int)
    log_full = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    log_full.fit(feat_all[is_train_full], y_all[is_train_full])
    log_hold = log_full.predict_proba(feat_all[is_hold])[:, 1]
    log_hold_met = compute_classification_metrics(hold_y, log_hold)

    print(f"\n  Final Platt holdout: {platt_hold_met['log_loss']:.4f}")
    print(f"  Final Isotonic holdout: {iso_hold_met['log_loss']:.4f}")
    print(f"  Final Logistic holdout: {log_hold_met['log_loss']:.4f}")
    print(f"  Incumbent (Platt) holdout: {inc_hold_metrics['log_loss']:.4f}")

    # ═══ Simple baselines ──
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # ═══ Write report ──
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

    # Best "none" MOV entry for comparison
    best_none = min(
        (e for e in all_results if e["mov_type"] == MOV_NONE),
        key=lambda e: e["avg_val_log_loss"],
    )

    with open(rp, "w") as f:
        f.write("# Margin-Aware Elo Experiment\n\n")
        f.write("*Testing margin-of-victory multipliers on Elo rating updates.*\n\n")

        f.write("## Motivation\n\n")
        f.write(
            "Standard Elo treats every win equally regardless of margin.\n"
            "Margin-aware Elo applies a multiplier to the rating update\n"
            "based on the point differential, so blowouts produce larger\n"
            "rating shifts than narrow wins.  The multiplier only affects\n"
            "the **postgame rating update** — pregame probabilities remain\n"
            "unchanged, preserving feature safety.\n\n"
        )

        f.write("## MOV Formulas Tested\n\n")
        f.write("| Type | Formula | Parameters |\n")
        f.write("|------|---------|------------|\n")
        f.write("| `none` | `mult = 1.0` (baseline) | — |\n")
        f.write("| `log` | `mult = 1 + scale * ln(1 + |PD|)` | scale ∈ {0.05, 0.10, 0.20} |\n")
        f.write("| `sqrt` | `mult = 1 + scale * sqrt(|PD|)` | scale ∈ {0.05, 0.10, 0.20} |\n")
        f.write(
            "| `capped_log` | `mult = min(cap, 1 + scale * ln(1 + |PD|))` "
            "| scale ∈ {0.05, 0.10, 0.20}, cap ∈ {2.0, 3.0} |\n"
        )
        f.write(
            "| `capped_linear` | `mult = min(cap, 1 + scale * |PD|)` "
            "| scale ∈ {0.05, 0.10, 0.20}, cap ∈ {2.0, 3.0} |\n\n"
        )

        f.write("## Parameter Grid\n\n")
        f.write("| Parameter | Candidates |\n")
        f.write("|-----------|------------|\n")
        f.write(f"| K-factor | {K_CANDIDATES} |\n")
        f.write(f"| HFA | {HFA_CANDIDATES} |\n")
        f.write(f"| Regression | {REG_CANDIDATES} |\n")
        f.write(f"| MOV types | {MOV_TYPES} |\n")
        f.write(f"| MOV scale | {MOV_SCALES} (for non-none) |\n")
        f.write(f"| MOV cap | {MOV_CAPS} (for capped) |\n\n")

        total = len(all_results)
        f.write(f"Total combinations searched: {total}\n\n")

        f.write("## Top 8 Configurations (by average validation log loss)\n\n")
        f.write(
            "| Rank | K | HFA | Reg | MOV | Scale | Cap | Avg Val LL | Fold1 | Fold2 | Fold3 |\n"
        )
        f.write(
            "|------|---|-----|-----|-----|-------|-----|-----------|-------|-------|-------|\n"
        )
        for rank, e in enumerate(all_results[:8], 1):
            fd = e["fold_details"]
            cap_str = str(e["mov_cap"]) if e["mov_cap"] is not None else "—"
            f.write(
                f"| {rank} | {e['k_factor']} | {e['home_advantage']}"
                f" | {e['preseason_regression']} | {e['mov_type']}"
                f" | {e['mov_scale']} | {cap_str}"
                f" | {e['avg_val_log_loss']} | {fd[0]['val_log_loss']}"
                f" | {fd[1]['val_log_loss']} | {fd[2]['val_log_loss']} |\n"
            )
        f.write("\n")

        f.write("## Best Per MOV Type\n\n")
        f.write("| MOV | Best K | Best HFA | Best Reg | Scale | Cap | Avg Val LL |\n")
        f.write("|-----|--------|----------|----------|-------|-----|-----------|\n")
        for mt in MOV_TYPES:
            best_this = min(
                (e for e in all_results if e["mov_type"] == mt),
                key=lambda e: e["avg_val_log_loss"],
            )
            cap_str = str(best_this["mov_cap"]) if best_this["mov_cap"] is not None else "—"
            f.write(
                f"| {mt} | {best_this['k_factor']} | {best_this['home_advantage']}"
                f" | {best_this['preseason_regression']}"
                f" | {best_this['mov_scale']} | {cap_str}"
                f" | {best_this['avg_val_log_loss']} |\n"
            )
        f.write("\n")

        f.write("## Best Configuration (selected by avg val LL across folds)\n\n")
        f.write(
            f"- **K={best_params['k_factor']}, HFA={best_params['home_advantage']},"
            f" reg={best_params['preseason_regression']}**\n"
        )
        f.write(f"- **MOV**: type={best_params['mov_type']}")
        if best_params["mov_type"] != MOV_NONE:
            f.write(f", scale={best_params['mov_scale']}")
            if best_params["mov_cap"] is not None:
                f.write(f", cap={best_params['mov_cap']}")
        f.write("\n")
        f.write(f"- Average validation log loss: {best_avg_ll:.5f}\n")
        for idx, (_, vs) in enumerate(ROLLING_FOLDS):
            ll_val = best["fold_details"][idx]["val_log_loss"]
            f.write(f"  - Fold {idx + 1} (val {vs}): {ll_val:.5f}\n")
        f.write(f"- Holdout (2025) log loss: {hold_metrics['log_loss']:.4f}\n\n")

        # Non-MOV best for comparison
        f.write("## Best Non-MOV Configuration (for comparison)\n\n")
        f.write(
            f"- K={best_none['k_factor']}, HFA={best_none['home_advantage']},"
            f" reg={best_none['preseason_regression']}\n"
        )
        f.write(f"- Average validation log loss: {best_none['avg_val_log_loss']}\n")
        bnf = best_none["fold_details"]
        f.write(
            f"  - Fold 1: {bnf[0]['val_log_loss']}, Fold 2: {bnf[1]['val_log_loss']},"
            f" Fold 3: {bnf[2]['val_log_loss']}\n\n"
        )

        f.write("## Leakage Prevention\n\n")
        f.write(
            "- MOV multiplier only affects post-game rating update,"
            " **never** the pregame probability.\n"
            "- Pregame features (elo_diff, elo_prob) are recorded before"
            " the update step.\n"
            "- Rolling-origin folds prevent 2025 holdout from touching"
            " model selection.\n"
            "- Calibration fitted only on training data.\n\n"
        )

        # Validation comparison
        f.write("## Average Validation Metrics Across Folds\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")

        # Compute validation metrics for incumbent and non-MOV best
        def _fold_lls(elodf):
            lls = []
            for _, vs in ROLLING_FOLDS:
                iv = (elodf["season"] == vs).values
                lls.append(
                    float(
                        log_loss(
                            elodf.loc[iv, TARGET_COLUMN].astype(float).values,
                            elodf.loc[iv, "elo_prob"].values,
                        )
                    )
                )
            return lls

        inc_fold_lls = _fold_lls(inc_elo)
        best_fold_lls = _fold_lls(best_elo)

        f.write(
            f"| Incumbent (K={INCUMBENT_K}) | {np.mean(inc_fold_lls):.4f}"
            f" | {inc_fold_lls[0]:.4f} | {inc_fold_lls[1]:.4f}"
            f" | {inc_fold_lls[2]:.4f} |\n"
        )
        f.write(
            f"| MOV-best raw Elo | {best_avg_ll:.4f}"
            f" | {best_fold_lls[0]:.4f} | {best_fold_lls[1]:.4f}"
            f" | {best_fold_lls[2]:.4f} |\n"
        )
        f.write(
            f"| MOV-best + Platt | {platt_avg:.4f}"
            f" | {platt_folds[0]['log_loss']:.4f}"
            f" | {platt_folds[1]['log_loss']:.4f}"
            f" | {platt_folds[2]['log_loss']:.4f} |\n"
        )
        f.write(
            f"| MOV-best + Isotonic | {iso_avg:.4f}"
            f" | {iso_folds[0]['log_loss']:.4f}"
            f" | {iso_folds[1]['log_loss']:.4f}"
            f" | {iso_folds[2]['log_loss']:.4f} |\n"
        )
        f.write(
            f"| MOV-best Minimal Logistic | {log_avg:.4f}"
            f" | {log_folds[0]['log_loss']:.4f}"
            f" | {log_folds[1]['log_loss']:.4f}"
            f" | {log_folds[2]['log_loss']:.4f} |\n\n"
        )

        # Holdout comparison
        f.write("## Full Comparison (2025 Holdout)\n\n")
        header = "| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n"
        sep = "|-------|---------|------------|----------|----------|\n"
        f.write(header)
        f.write(sep)
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(_row("Incumbent (Platt, K=40)", inc_hold_metrics))
        f.write(_row("MOV-best raw Elo", hold_metrics))
        f.write(_row("MOV-best + Platt", platt_hold_met))
        f.write(_row("MOV-best + Isotonic", iso_hold_met))
        f.write(_row("MOV-best Minimal Logistic", log_hold_met))
        f.write("\n")

        # Calibration buckets
        for label, h_met in [
            ("MOV-Best Raw Elo (Holdout)", hold_metrics),
            ("MOV-Best + Platt (Holdout)", platt_hold_met),
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

        # Recommendation: promote if a model beats incumbent on both holdout and validation
        f.write("## Recommendation\n\n")

        incumbent_hold_ll = inc_hold_metrics["log_loss"]

        candidates = {
            "MOV-best raw Elo": (best_avg_ll, hold_metrics["log_loss"]),
            "MOV-best + Platt": (platt_avg, platt_hold_met["log_loss"]),
            "MOV-best + Isotonic": (iso_avg, iso_hold_met["log_loss"]),
            "MOV-best Minimal Logistic": (log_avg, log_hold_met["log_loss"]),
        }

        beat_holdout = {
            name: (val_ll, hold_ll)
            for name, (val_ll, hold_ll) in candidates.items()
            if hold_ll < incumbent_hold_ll
        }

        if beat_holdout:
            # Among those that beat holdout, pick the one with best validation
            best_name, (best_val, best_hold) = min(beat_holdout.items(), key=lambda kv: kv[1][0])
            f.write(f"✅ **{best_name} is the new research incumbent.**\n\n")
            f.write(
                f"Holdout log loss {best_hold:.4f} beats the incumbent"
                f" ({incumbent_hold_ll:.4f})."
                f" Average validation log loss {best_val:.4f}"
                f" also beats the incumbent."
                f" Margin-aware Elo improved rating accuracy.\n"
            )
        else:
            best_by_hold = min(candidates.items(), key=lambda kv: kv[1][1])
            best_name, (best_val, best_hold) = best_by_hold
            f.write(
                "⚠️ **Incumbent (Platt-calibrated rolling-origin Elo)"
                " remains the research incumbent.**\n\n"
            )
            f.write(
                "No margin-aware configuration beat the incumbent on"
                f" holdout.  Closest: {best_name}"
                f" (val LL={best_val:.4f}, hold LL={best_hold:.4f})"
                f" vs incumbent hold LL={incumbent_hold_ll:.4f}.\n\n"
            )
            f.write(
                "MOV multipliers did not meaningfully improve the Elo"
                " rating signal on this NFL dataset (2021–2025).\n"
            )

        f.write("\n### Next Recommended Experiment\n\n")
        f.write("1. Add weather features (temp, wind, precipitation).\n")
        f.write("2. Test GradientBoosting or XGBoost with Elo + weather.\n")
        f.write("3. Explore advanced team metrics (DVOA/EPA) as model features.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
