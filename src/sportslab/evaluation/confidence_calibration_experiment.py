"""Confidence calibration and probability shrinkage experiment.

Tests whether post-processing methods (clipping, temperature scaling,
shrinkage) improve on the MOV Elo+Platt incumbent by reducing
overconfidence, especially at extreme probability ranges.

Targets the QB-change failure mode and high-confidence calibration
identified by residual diagnostics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
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

CLIP_BOUNDS = [
    (0.01, 0.99),
    (0.03, 0.97),
    (0.05, 0.95),
    (0.08, 0.92),
    (0.10, 0.90),
]

TEMPERATURES = [1.05, 1.10, 1.15, 1.20, 1.30, 1.50]

SHRINK_STRENGTHS = [0.02, 0.05, 0.08, 0.10, 0.15]

HIGH_CONF_THRESHOLDS = [
    (0.10, 0.90),
    (0.08, 0.92),
    (0.05, 0.95),
]

EARLY_WEEKS = {1, 2, 3, 4}


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


# ── Post-processing functions ──


def clip_probabilities(p: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    return np.clip(p, lo, hi)


def temperature_scale(p: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    if temperature == 1.0:
        return p.copy()
    logit = np.log(np.clip(p, 1e-15, 1 - 1e-15) / (1 - np.clip(p, 1e-15, 1 - 1e-15)))
    logit = logit / temperature
    return 1.0 / (1.0 + np.exp(-logit))


def shrink_to_prior(p: np.ndarray, alpha: float = 0.05, prior: float = 0.5) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return (1.0 - alpha) * p + alpha * prior


def high_confidence_shrink(
    p: np.ndarray,
    alpha: float = 0.05,
    threshold_lo: float = 0.10,
    threshold_hi: float = 0.90,
    prior: float = 0.5,
) -> np.ndarray:
    result = p.copy().astype(float)
    mask = (p <= threshold_lo) | (p >= threshold_hi)
    result[mask] = (1.0 - alpha) * result[mask] + alpha * prior
    # Not-clipped extreme: p <= lo maps to lower, p >= hi maps to mid
    return result


def early_season_shrink(
    p: np.ndarray,
    weeks: np.ndarray,
    alpha_early: float = 0.10,
    alpha_late: float = 0.0,
    prior: float = 0.5,
) -> np.ndarray:
    result = p.copy().astype(float)
    early = np.isin(weeks, list(EARLY_WEEKS))
    result[early] = (1.0 - alpha_early) * result[early] + alpha_early * prior
    result[~early] = (1.0 - alpha_late) * result[~early] + alpha_late * prior
    return result


# ── Rolling-origin grid search ──


def _run_method_on_fold(
    train_p: np.ndarray,
    train_y: np.ndarray,
    val_p: np.ndarray,
    val_y: np.ndarray,
    method_fn,
    **kwargs,
) -> float:
    """Apply a post-processing method to train and val probabilities."""
    # Handle early-season: needs weeks array
    if "weeks_train" in kwargs:
        w_train = kwargs.pop("weeks_train")
        w_val = kwargs.pop("weeks_val")
        method_fn(train_p, w_train, **kwargs)
        applied_val = method_fn(val_p, w_val, **kwargs)
    else:
        method_fn(train_p, **kwargs)
        applied_val = method_fn(val_p, **kwargs)
    return float(log_loss(val_y, applied_val))


def run_grid_search(
    elo_prob: np.ndarray,
    y: np.ndarray,
    weeks: np.ndarray | None = None,
    seasons: np.ndarray | None = None,
) -> dict:
    """Run rolling-origin grid search over all calibration methods.

    Returns dict with best method info and all results.
    """
    all_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = np.isin(seasons, train_seasons) if seasons is not None else None
        is_val = (seasons == val_season) if seasons is not None else None

        if is_train is None:
            is_train = np.ones(len(elo_prob), dtype=bool)
            is_val = np.zeros(len(elo_prob), dtype=bool)

        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]

        train_weeks = weeks[is_train] if weeks is not None else None
        val_weeks = weeks[is_val] if weeks is not None else None

        # -- Baseline: raw Elo probability (no Platt) --
        baseline_ll = float(log_loss(val_y_, val_elo))
        all_results.append(
            {
                "fold": val_season,
                "method": "baseline_raw_elo",
                "params": {},
                "val_log_loss": baseline_ll,
            }
        )

        # -- A. Clip probabilities --
        for lo, hi in CLIP_BOUNDS:
            ll = _run_method_on_fold(
                train_elo, train_y_, val_elo, val_y_, clip_probabilities, lo=lo, hi=hi
            )
            all_results.append(
                {
                    "fold": val_season,
                    "method": "clip",
                    "params": {"lo": lo, "hi": hi},
                    "val_log_loss": ll,
                }
            )

        # -- B. Temperature scaling --
        for temp in TEMPERATURES:
            ll = _run_method_on_fold(
                train_elo, train_y_, val_elo, val_y_, temperature_scale, temperature=temp
            )
            all_results.append(
                {
                    "fold": val_season,
                    "method": "temperature",
                    "params": {"temperature": temp},
                    "val_log_loss": ll,
                }
            )

        # -- C. Global shrinkage toward prior (0.5) --
        for alpha in SHRINK_STRENGTHS:
            ll = _run_method_on_fold(
                train_elo, train_y_, val_elo, val_y_, shrink_to_prior, alpha=alpha, prior=0.5
            )
            all_results.append(
                {
                    "fold": val_season,
                    "method": "shrink_50",
                    "params": {"alpha": alpha, "prior": 0.5},
                    "val_log_loss": ll,
                }
            )

        # -- Global shrinkage toward home prior --
        home_prior = float(train_y_.mean())
        for alpha in SHRINK_STRENGTHS:
            ll = _run_method_on_fold(
                train_elo, train_y_, val_elo, val_y_, shrink_to_prior, alpha=alpha, prior=home_prior
            )
            all_results.append(
                {
                    "fold": val_season,
                    "method": "shrink_home_prior",
                    "params": {"alpha": alpha, "prior": home_prior},
                    "val_log_loss": ll,
                }
            )

        # -- D. High-confidence-only shrinkage --
        for lo, hi in HIGH_CONF_THRESHOLDS:
            for alpha in SHRINK_STRENGTHS:
                ll = _run_method_on_fold(
                    train_elo,
                    train_y_,
                    val_elo,
                    val_y_,
                    high_confidence_shrink,
                    alpha=alpha,
                    threshold_lo=lo,
                    threshold_hi=hi,
                    prior=0.5,
                )
                all_results.append(
                    {
                        "fold": val_season,
                        "method": "high_conf_shrink",
                        "params": {
                            "alpha": alpha,
                            "threshold_lo": lo,
                            "threshold_hi": hi,
                            "prior": 0.5,
                        },
                        "val_log_loss": ll,
                    }
                )

        # -- E. Early-season shrinkage --
        if train_weeks is not None and val_weeks is not None:
            for alpha_early in SHRINK_STRENGTHS:
                ll = _run_method_on_fold(
                    train_elo,
                    train_y_,
                    val_elo,
                    val_y_,
                    early_season_shrink,
                    weeks_train=train_weeks,
                    weeks_val=val_weeks,
                    alpha_early=alpha_early,
                    alpha_late=0.0,
                    prior=0.5,
                )
                all_results.append(
                    {
                        "fold": val_season,
                        "method": "early_season_shrink",
                        "params": {"alpha_early": alpha_early, "alpha_late": 0.0, "prior": 0.5},
                        "val_log_loss": ll,
                    }
                )

    # Aggregate across folds by method+params
    df_results = pd.DataFrame(all_results)

    def _agg_key(row):
        m = row["method"]
        if m == "baseline_raw_elo":
            return m
        params = row["params"]
        if m == "clip":
            return f"{m}_lo={params['lo']:.2f}_hi={params['hi']:.2f}"
        if m == "temperature":
            return f"{m}_t={params['temperature']:.2f}"
        if m == "shrink_50":
            return f"{m}_a={params['alpha']:.3f}"
        if m == "shrink_home_prior":
            return f"{m}_a={params['alpha']:.3f}_p={params['prior']:.3f}"
        if m == "high_conf_shrink":
            p = params
            return f"{m}_a={p['alpha']:.3f}_lo={p['threshold_lo']:.2f}_hi={p['threshold_hi']:.2f}"
        if m == "early_season_shrink":
            return f"{m}_a={params['alpha_early']:.3f}"
        return str(row)

    df_results["key"] = df_results.apply(_agg_key, axis=1)
    avg_ll = df_results.groupby("key")["val_log_loss"].mean().reset_index()
    avg_ll.columns = ["key", "avg_val_log_loss"]

    best = avg_ll.loc[avg_ll["avg_val_log_loss"].idxmin()]
    best_details = df_results[df_results["key"] == best["key"]].iloc[0].to_dict()

    return {
        "best_key": best["key"],
        "best_avg_val_ll": best["avg_val_log_loss"],
        "best_details": best_details,
        "all_avg": avg_ll.sort_values("avg_val_log_loss"),
        "all_results": df_results,
    }


def run_confidence_calibration_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/confidence_calibration.md",
) -> str:
    """Run confidence calibration experiment.

    1. Compute MOV Elo with frozen incumbent params.
    2. Compute QB features for subset analysis.
    3. Run rolling-origin grid search over all calibration methods.
    4. Select best method by average validation log loss.
    5. One-time 2025 holdout evaluation.
    6. Report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Compute MOV Elo ──
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

    # ── QB features for subset analysis ──
    print("\n=== Computing QB features ===")
    df_elo = compute_qb_features(df_elo)

    # ── Filter ──
    df = _filter_df(df_elo)

    elo_prob = df["elo_prob"].values
    y = df[TARGET_COLUMN].astype(float).values
    weeks = df["week"].values
    seasons = df["season"].values

    # ═══ Rolling-origin grid search ═══
    print("\n=== Rolling-Origin Grid Search ===")
    grid_result = run_grid_search(elo_prob, y, weeks=weeks, seasons=seasons)

    best_key = grid_result["best_key"]
    best_avg_ll = grid_result["best_avg_val_ll"]
    best_details = grid_result["best_details"]

    print(f"\n  Best method: {best_key}")
    print(f"  Avg val log loss: {best_avg_ll:.4f}")
    print("\nTop 10 methods by avg val log loss:")
    for _, row in grid_result["all_avg"].head(10).iterrows():
        print(f"  {row['key']}: {row['avg_val_log_loss']:.4f}")

    # ═══ Build the best post-processor function ═══
    best_method = best_details["method"]
    best_params = best_details.get("params", {})

    def build_post_processor(method: str, params: dict):
        if method == "clip":
            return lambda p: clip_probabilities(p, lo=params["lo"], hi=params["hi"])
        if method == "temperature":
            return lambda p: temperature_scale(p, temperature=params["temperature"])
        if method == "shrink_50":
            return lambda p: shrink_to_prior(p, alpha=params["alpha"], prior=0.5)
        if method == "shrink_home_prior":
            return lambda p: shrink_to_prior(p, alpha=params["alpha"], prior=params["prior"])
        if method == "high_conf_shrink":
            return lambda p: high_confidence_shrink(
                p,
                alpha=params["alpha"],
                threshold_lo=params["threshold_lo"],
                threshold_hi=params["threshold_hi"],
                prior=0.5,
            )
        if method == "early_season_shrink":
            return lambda p, w=weeks: early_season_shrink(
                p,
                w,
                alpha_early=params["alpha_early"],
                alpha_late=0.0,
                prior=0.5,
            )
        return lambda p: p

    best_fn = build_post_processor(best_method, best_params)

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (seasons == HOLDOUT_SEASON)
    is_train_full = np.isin(seasons, [2021, 2022, 2023, 2024])

    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]
    hold_weeks = weeks[is_hold]

    train_y_full = y[is_train_full].astype(int)
    home_prior_full = float(train_y_full.mean())

    # Apply methods on holdout

    # 1. Raw Elo (no Platt)
    raw_hold_m = compute_classification_metrics(hold_y, hold_elo)

    # 2. Best selected method
    best_hold_proba = best_fn(hold_elo)
    best_hold_m = compute_classification_metrics(hold_y, best_hold_proba)
    print(f"  Best ({best_key}): {best_hold_m['log_loss']:.4f}")

    # 3. All clipping variants on holdout
    clip_hold_metrics = {}
    for lo, hi in CLIP_BOUNDS:
        clip_p = clip_probabilities(hold_elo, lo=lo, hi=hi)
        clip_hold_metrics[f"clip_{lo}_{hi}"] = compute_classification_metrics(hold_y, clip_p)

    # 4. All temperature variants on holdout
    temp_hold_metrics = {}
    for t in TEMPERATURES:
        temp_p = temperature_scale(hold_elo, temperature=t)
        temp_hold_metrics[f"temp_{t}"] = compute_classification_metrics(hold_y, temp_p)

    # 5. All global shrinkage variants on holdout
    shrink50_hold_metrics = {}
    for a in SHRINK_STRENGTHS:
        shrink_p = shrink_to_prior(hold_elo, alpha=a, prior=0.5)
        shrink50_hold_metrics[f"shrink50_{a}"] = compute_classification_metrics(hold_y, shrink_p)

    shrink_hp_hold_metrics = {}
    for a in SHRINK_STRENGTHS:
        shrink_p = shrink_to_prior(hold_elo, alpha=a, prior=home_prior_full)
        shrink_hp_hold_metrics[f"shrink_hp_{a}"] = compute_classification_metrics(hold_y, shrink_p)

    # 6. High-confidence-only shrinkage on holdout
    hc_hold_metrics = {}
    for lo, hi in HIGH_CONF_THRESHOLDS:
        for a in SHRINK_STRENGTHS:
            hc_p = high_confidence_shrink(
                hold_elo, alpha=a, threshold_lo=lo, threshold_hi=hi, prior=0.5
            )
            hc_hold_metrics[f"hc_{lo}_{hi}_a={a}"] = compute_classification_metrics(hold_y, hc_p)

    # 7. Early-season shrinkage on holdout
    early_hold_metrics = {}
    for a in SHRINK_STRENGTHS:
        early_p = early_season_shrink(
            hold_elo, hold_weeks, alpha_early=a, alpha_late=0.0, prior=0.5
        )
        early_hold_metrics[f"early_a={a}"] = compute_classification_metrics(hold_y, early_p)

    # ── Subset analyses ──
    print("\n=== Subset Analysis ===")

    def _subset_report(mask, label, proba, hold_metrics, n_min=5):
        n = int(mask.sum())
        if n < n_min:
            return {"label": label, "n": n, "log_loss": None}
        sub_y = hold_y[mask]
        sub_p = proba[mask]
        m = compute_classification_metrics(sub_y, sub_p)
        return {"label": label, "n": n, "log_loss": m["log_loss"]}

    # QB-change subset
    hold_qb_changed = df.loc[is_hold, "home_qb_changed"].fillna(0).astype(bool).values
    hold_qb_stable = ~hold_qb_changed

    # High-confidence subset
    hold_high_conf = best_hold_proba > 0.9
    hold_low_conf = best_hold_proba <= 0.6

    # Early-season subset
    hold_early = np.isin(hold_weeks, list(EARLY_WEEKS))
    hold_late = ~hold_early

    subsets = {
        "QB changed (home)": hold_qb_changed,
        "QB stable (home)": hold_qb_stable,
        "High confidence (>0.9)": hold_high_conf,
        "Low confidence (<=0.6)": hold_low_conf,
        "Early season (W1-4)": hold_early,
        "Late season (W5+)": hold_late,
    }

    # ── Baselines ──
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, home_prior_full)))

    # ── Report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Confidence Calibration and Probability Shrinkage\n\n")
        f.write(
            "*Testing post-processing methods to reduce overconfidence"
            " in MOV Elo+Platt probabilities.*\n\n"
        )

        f.write("## Methods Tested\n\n")
        f.write("| Method | Variants | Description |\n")
        f.write("|--------|----------|-------------|\n")
        f.write("| Baseline (raw Elo) | — | Uncalibrated MOV Elo probability |\n")
        f.write("| Probability clipping | 5 thresholds (0.01–0.10) | Clip to [lo, hi] range |\n")
        f.write(
            "| Temperature scaling | 6 temperatures (1.05–1.50)"
            " | Soften logit by dividing by T > 1 |\n"
        )
        f.write("| Global shrinkage (p=0.5) | 5 strengths (0.02–0.15) | Shrink toward 0.5 |\n")
        f.write("| Global shrinkage (home prior) | 5 strengths | Shrink toward home win rate |\n")
        f.write(
            "| High-confidence-only shrinkage | 3 thresholds × 5 strengths"
            " | Shrink only p ≤ lo or p ≥ hi |\n"
        )
        f.write("| Early-season shrinkage | 5 strengths | Shrink weeks 1–4 only |\n\n")

        f.write("## Incumbent MOV Elo Params\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| K-factor | {BEST_K} |\n")
        f.write(f"| Home-field advantage | {BEST_HFA} |\n")
        f.write(f"| Preseason regression | {BEST_REG} |\n")
        f.write(f"| MOV type | {BEST_MOV_TYPE} |\n")
        f.write(f"| MOV scale | {BEST_MOV_SCALE} |\n")
        f.write(f"| MOV cap | {BEST_MOV_CAP} |\n\n")

        f.write("## Rolling-Origin Selection\n\n")
        f.write("| Split | Seasons | Role |\n")
        f.write("|-------|---------|------|\n")
        for i, (tr, va) in enumerate(ROLLING_FOLDS, 1):
            f.write(f"| Fold {i} | Train: {tr}, Val: {va} | Selection |\n")
        f.write(f"| Holdout | {HOLDOUT_SEASON} | Final eval |\n\n")

        f.write("## Top 10 Methods (Avg Validation Log Loss)\n\n")
        f.write("| Method | Avg Val LL |\n")
        f.write("|--------|------------|\n")
        for _, row in grid_result["all_avg"].head(10).iterrows():
            f.write(f"| {row['key']} | {row['avg_val_log_loss']:.4f} |\n")
        f.write("\n")

        f.write("## 2025 Holdout Comparison\n\n")
        f.write("| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n")
        f.write("|-------|---------|------------|----------|----------|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({home_prior_full:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(
            f"| Raw Elo (no Platt) | {raw_hold_m['log_loss']:.4f}"
            f" | {raw_hold_m['brier_score']:.4f}"
            f" | {raw_hold_m['accuracy']:.4f}"
            f" | {raw_hold_m['roc_auc']:.4f} |\n"
        )
        f.write(
            f"| **Best: {best_key}** | {best_hold_m['log_loss']:.4f}"
            f" | {best_hold_m['brier_score']:.4f}"
            f" | {best_hold_m['accuracy']:.4f}"
            f" | {best_hold_m['roc_auc']:.4f} |\n"
        )
        f.write("| MOV Elo + Platt (incumbent) | 0.6373 | — | — | — |\n\n")

        # Full table
        f.write("### All Clip Variants\n\n")
        f.write("| Clip | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n")
        f.write("|------|---------|------------|----------|----------|\n")
        for key, m in sorted(clip_hold_metrics.items()):
            f.write(
                f"| [{key.replace('_', ', ')}] | {m['log_loss']:.4f}"
                f" | {m['brier_score']:.4f}"
                f" | {m['accuracy']:.4f}"
                f" | {m['roc_auc']:.4f} |\n"
            )
        f.write("\n")

        f.write("### All Temperature Variants\n\n")
        f.write("| Temp | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n")
        f.write("|------|---------|------------|----------|----------|\n")
        for key, m in sorted(temp_hold_metrics.items()):
            f.write(
                f"| T={key.split('_')[1]} | {m['log_loss']:.4f}"
                f" | {m['brier_score']:.4f}"
                f" | {m['accuracy']:.4f}"
                f" | {m['roc_auc']:.4f} |\n"
            )
        f.write("\n")

        f.write("### All Global Shrinkage Variants\n\n")
        f.write("| Shrinkage | Hold LL | Hold Brier | Hold Acc | Hold AUC |\n")
        f.write("|-----------|---------|------------|----------|----------|\n")
        for key, m in sorted({**shrink50_hold_metrics, **shrink_hp_hold_metrics}.items()):
            f.write(
                f"| {key} | {m['log_loss']:.4f}"
                f" | {m['brier_score']:.4f}"
                f" | {m['accuracy']:.4f}"
                f" | {m['roc_auc']:.4f} |\n"
            )
        f.write("\n")

        # Subset analysis
        f.write("## Subset Analysis (2025 Holdout)\n\n")
        f.write("| Subset | N | Raw Elo | Best Method |\n")
        f.write("|--------|---|---------|-------------|\n")
        for label, mask in sorted(subsets.items()):
            n = int(mask.sum())
            if n < 5:
                f.write(f"| {label} | {n} | insufficient | insufficient |\n")
                continue
            raw_sub = compute_classification_metrics(hold_y[mask], hold_elo[mask])
            best_sub = compute_classification_metrics(hold_y[mask], best_hold_proba[mask])
            f.write(f"| {label} | {n} | {raw_sub['log_loss']:.4f} | {best_sub['log_loss']:.4f} |\n")
        f.write("\n")

        # Calibration
        f.write("## Calibration Deciles\n\n")
        for label, metrics in [
            ("Raw Elo (Holdout)", raw_hold_m),
            (f"Best: {best_key} (Holdout)", best_hold_m),
        ]:
            f.write(f"### {label}\n\n")
            f.write("| Bucket | Count | Mean Pred | Mean Actual | Cal Error |\n")
            f.write("|--------|-------|-----------|-------------|-----------|\n")
            for b, vals in sorted(metrics["calibration_buckets"].items()):
                f.write(
                    f"| {b} | {vals['count']} | {vals['mean_predicted_prob']:.4f}"
                    f" | {vals['mean_actual_rate']:.4f}"
                    f" | {vals['calibration_error']:.4f} |\n"
                )
            f.write("\n")

        # Recommendation
        f.write("## Recommendation\n\n")

        incumbent_hold_ll = 0.6373
        best_hold_ll = best_hold_m["log_loss"]

        if best_hold_ll < incumbent_hold_ll:
            f.write(f"✅ **{best_key} is the new research incumbent.**\n\n")
            f.write(
                f"Holdout log loss {best_hold_ll:.4f} beats the incumbent"
                f" ({incumbent_hold_ll:.4f})."
                f" Average validation log loss {best_avg_ll:.4f}."
                f" Selected by rolling-origin validation.\n"
            )
        else:
            f.write("⚠️ **MOV Elo + Platt remains the research incumbent.**\n\n")
            f.write(
                f"No calibration method beat the incumbent on holdout."
                f" Best: {best_key}"
                f" (val LL={best_avg_ll:.4f}, hold LL={best_hold_ll:.4f})"
                f" vs incumbent (0.6373).\n\n"
            )

        # High-confidence assessment
        f.write("### High-Confidence Assessment\n\n")
        hc = subsets["High confidence (>0.9)"]
        if hc.sum() >= 5:
            raw_hc = compute_classification_metrics(hold_y[hc], hold_elo[hc])
            best_hc = compute_classification_metrics(hold_y[hc], best_hold_proba[hc])
            f.write(f"Raw Elo high-confidence ({hc.sum()} games): LL={raw_hc['log_loss']:.4f}\n")
            f.write(f"Best method high-confidence: LL={best_hc['log_loss']:.4f}\n")
        else:
            f.write(f"High-confidence subset too small ({hc.sum()} games).\n")
        f.write("\n")

        # QB-change assessment
        f.write("### QB-Change Assessment\n\n")
        qb_c = subsets["QB changed (home)"]
        qb_s = subsets["QB stable (home)"]
        if qb_c.sum() >= 5 and qb_s.sum() >= 5:
            raw_qb_c = compute_classification_metrics(hold_y[qb_c], hold_elo[qb_c])
            best_qb_c = compute_classification_metrics(hold_y[qb_c], best_hold_proba[qb_c])
            raw_qb_s = compute_classification_metrics(hold_y[qb_s], hold_elo[qb_s])
            best_qb_s = compute_classification_metrics(hold_y[qb_s], best_hold_proba[qb_s])
            f.write(
                f"Raw Elo: QB-changed LL={raw_qb_c['log_loss']:.4f}"
                f" | QB-stable LL={raw_qb_s['log_loss']:.4f}"
                f" | gap={raw_qb_c['log_loss'] - raw_qb_s['log_loss']:.4f}\n"
            )
            f.write(
                f"Best: QB-changed LL={best_qb_c['log_loss']:.4f}"
                f" | QB-stable LL={best_qb_s['log_loss']:.4f}"
                f" | gap={best_qb_c['log_loss'] - best_qb_s['log_loss']:.4f}\n"
            )
        else:
            f.write(f"QB-change subset too small ({qb_c.sum()} games).\n")
        f.write("\n")

        # Early vs late
        f.write("### Early vs Late Season\n\n")
        early = subsets["Early season (W1-4)"]
        late = subsets["Late season (W5+)"]
        if early.sum() >= 5 and late.sum() >= 5:
            raw_early = compute_classification_metrics(hold_y[early], hold_elo[early])
            best_early = compute_classification_metrics(hold_y[early], best_hold_proba[early])
            raw_late = compute_classification_metrics(hold_y[late], hold_elo[late])
            best_late = compute_classification_metrics(hold_y[late], best_hold_proba[late])
            f.write(
                f"Raw Elo: Early LL={raw_early['log_loss']:.4f}"
                f" | Late LL={raw_late['log_loss']:.4f}\n"
            )
            f.write(
                f"Best: Early LL={best_early['log_loss']:.4f}"
                f" | Late LL={best_late['log_loss']:.4f}\n"
            )
        else:
            f.write(f"Early-season subset too small ({early.sum()} games).\n")

        f.write("\n### Next Recommended Experiment\n\n")
        f.write("1. Test isotonic regression or Platt-only recalibration.\n")
        f.write("2. Ensemble methods combining multiple shrinkage variants.\n")
        f.write("3. Early-season feature enrichment to reduce initial uncertainty.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
