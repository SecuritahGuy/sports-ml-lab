"""QB-continuity refinement experiment — tests narrow QB-continuity
feature sets against the incumbent.

Compares 6 model variants:
  A. Incumbent (qb_changed + rolling_mov_3)
  B. No-QB fallback (rolling_mov_3 only)
  C. QB continuity minimal (+ new_qb_flag)
  D. QB continuity experience (+ starts, diff)
  E. QB continuity recovery (+ games_since_change, diff)
  F. QB continuity full-small (all QB-continuity features)

All features are derived from `compute_qb_features()` which reads
home_qb_id/away_qb_id from the feature table. These are oracle-only
(research) unless supplied via pregame QB input CSV.

Usage:
    sportslab qb-continuity
    make qb-continuity
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.predict_incumbent import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    FEATURE_COLS,
    INCUMBENT_DATE,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
)
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

# --- Constants ---

ALL_SEASONS = [2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
DEFAULT_REPORT = "reports/experiments/qb_continuity.md"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
N_WORST = 20

QB_CHANGE_COLS = ["home_qb_changed", "away_qb_changed"]
MOV_COLS = ["home_rolling_mov_3", "away_rolling_mov_3"]
NEW_QB_COLS = ["home_new_qb_flag", "away_new_qb_flag"]
STARTS_COLS = [
    "home_qb_starts_this_season_pre", "away_qb_starts_this_season_pre",
    "qb_starts_diff",
]
RECOVERY_COLS = [
    "home_games_since_qb_change", "away_games_since_qb_change",
    "games_since_qb_change_diff",
]
FULL_CONTINUITY_COLS = QB_CHANGE_COLS + NEW_QB_COLS + STARTS_COLS + RECOVERY_COLS + [
    "home_qb_team_starts_pre", "away_qb_team_starts_pre",
    "home_qb_missing_flag", "away_qb_missing_flag",
]

# --- Model variant definitions ---
# (feature_set_name, feature_cols, live_safety_label, description)

MODEL_VARIANTS = [
    ("incumbent", FEATURE_COLS,
     "research_oracle",
     "Incumbent: qb_changed + rolling_mov_3"),
    ("no_qb", [c for c in FEATURE_COLS if "qb" not in c],
     "live_safe_no_qb",
     "No-QB fallback: rolling_mov_3 only"),
    ("qb_minimal", QB_CHANGE_COLS + NEW_QB_COLS + MOV_COLS,
     "research_oracle",
     "QB continuity minimal: qb_changed + new_qb_flag + rolling_mov_3"),
    ("qb_experience", QB_CHANGE_COLS + STARTS_COLS + MOV_COLS,
     "research_oracle",
     "QB continuity experience: qb_changed + starts + rolling_mov_3"),
    ("qb_recovery", QB_CHANGE_COLS + RECOVERY_COLS + MOV_COLS,
     "research_oracle",
     "QB continuity recovery: qb_changed + games_since + rolling_mov_3"),
    ("qb_full_small", FULL_CONTINUITY_COLS + MOV_COLS,
     "research_oracle",
     "QB continuity full-small: all QB features + rolling_mov_3"),
]

LIVE_SAFE_LABELS = {
    "incumbent": "research_oracle",
    "no_qb": "live_safe_no_qb",
    "qb_minimal": "research_oracle",
    "qb_experience": "research_oracle",
    "qb_recovery": "research_oracle",
    "qb_full_small": "research_oracle",
}

# --- Shared helpers ---


def _compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    valid = ~np.isnan(y_true)
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {}
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    labels = np.array([0, 1])
    ll = float(sk_log_loss(y_true, y_prob, labels=labels))
    brier = float(np.mean((y_true - y_prob) ** 2))
    acc = float(np.mean((y_prob >= 0.5) == y_true))
    return {"log_loss": round(ll, 4), "brier": round(brier, 4),
            "accuracy": round(acc, 4)}


def _fit_platt(x: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, solver="lbfgs",
                                  random_state=RANDOM_SEED)),
    ])
    pipe.fit(x, y)
    return pipe


def _build_feature_matrix(
    df: pd.DataFrame, feature_cols: List[str]
) -> np.ndarray:
    elo = df["elo_prob"].values
    avail = [c for c in feature_cols if c in df.columns]
    if avail:
        feat = df[avail].values
        return np.column_stack([elo, feat])
    return elo.reshape(-1, 1)


def _bootstrap_delta(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Bootstrap Δ log loss (prob_b - prob_a).

    Δ = challenger − incumbent.  Negative Δ means challenger is better.
    Returns mean, ci_low, ci_high.
    """
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    p_a = prob_a[valid]
    p_b = prob_b[valid]
    eps = 1e-15
    p_a = np.clip(p_a, eps, 1 - eps)
    p_b = np.clip(p_b, eps, 1 - eps)
    n = len(y_t)
    deltas = np.zeros(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        ll_a = sk_log_loss(y_t[idx], p_a[idx])
        ll_b = sk_log_loss(y_t[idx], p_b[idx])
        deltas[i] = ll_b - ll_a
    mean_delta = float(np.mean(deltas))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return round(mean_delta, 4), round(ci_low, 4), round(ci_high, 4)


# --- Calibration / confidence / worst-pred analysis ---


def _calibration_buckets(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> List[Dict]:
    """Split predictions into 10 equal-width probability buckets."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    labels_list = [f"{int(i*10)}-{int((i+1)*10)}%" for i in range(10)]
    bucket_indices = np.clip(np.floor(y_p * 10).astype(int), 0, 9)
    results = []
    for i in range(10):
        mask = bucket_indices == i
        if mask.sum() == 0:
            continue
        bin_actual = y_t[mask].mean()
        bin_prob = y_p[mask].mean()
        n = int(mask.sum())
        results.append({
            "bucket": labels_list[i],
            "n": n,
            "mean_pred": round(float(bin_prob), 4),
            "mean_actual": round(float(bin_actual), 4),
            "cal_error": round(float(abs(bin_prob - bin_actual)), 4),
        })
    return results


def _confidence_buckets(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> List[Dict]:
    """Split by confidence (closer to 0 or 1 = higher confidence)."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    confidence = np.abs(y_p - 0.5) * 2  # 0=random, 1=certain
    labels_list = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    indices = np.clip(np.floor(confidence / 0.2).astype(int), 0, 4)
    results = []
    for i in range(5):
        mask = indices == i
        if mask.sum() == 0:
            continue
        m = _compute_metrics(y_t[mask], y_p[mask])
        results.append({
            "bucket": labels_list[i],
            "n": int(mask.sum()),
            **m,
        })
    return results


def _worst_predictions(
    y_true: np.ndarray, y_prob: np.ndarray,
    game_ids: np.ndarray, teams: np.ndarray,
    n: int = N_WORST,
) -> List[Dict]:
    """Find the n worst predictions by log loss contribution."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    gids = game_ids[valid]
    teams_arr = teams[valid] if len(teams) == len(y_true) else np.array(["?"] * len(y_true))
    eps = 1e-15
    y_p = np.clip(y_p, eps, 1 - eps)
    contrib = -(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p))
    worst_idx = np.argsort(-contrib)[:n]
    results = []
    for i in worst_idx:
        results.append({
            "game_id": str(gids[i]),
            "team": str(teams_arr[i]),
            "actual": int(y_t[i]),
            "pred": round(float(y_p[i]), 4),
            "log_loss_contrib": round(float(contrib[i]), 4),
        })
    return results


# --- Rolling simulation ---


def _run_rolling_simulation(
    df_raw: pd.DataFrame,
    feat_cols: List[str],
    label: str = "model",
) -> Dict:
    """Week-by-week rolling simulation across all eligible seasons."""
    eligible = (
        df_raw.get(MODEL_ELIGIBLE_COLUMN, pd.Series(True))
        & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
    )

    weeks_data = df_raw[eligible][["season", "week"]].drop_duplicates()
    weeks_data = weeks_data.sort_values(["season", "week"])
    week_tuples = list(zip(weeks_data["season"], weeks_data["week"]))

    all_preds = []
    week_metrics = []

    for season, week in week_tuples:
        train_mask = (
            (df_raw["season"] < season)
            | ((df_raw["season"] == season) & (df_raw["week"] < week))
        ) & eligible

        pred_mask = (
            (df_raw["season"] == season)
            & (df_raw["week"] == week)
            & eligible
        )

        if pred_mask.sum() == 0 or train_mask.sum() < 10:
            continue

        df_train = df_raw[train_mask].copy()
        df_pred = df_raw[pred_mask].copy()
        pred_game_ids = set(df_pred["game_id"].values)
        actuals = df_pred.set_index("game_id")["home_win"].copy()
        df_pred["home_win"] = np.nan
        df_pred["home_win"] = df_pred["home_win"].astype(float)
        df_combined = pd.concat([df_train, df_pred], ignore_index=True)
        df_combined = df_combined.sort_values(
            ["season", "week", "gameday"]
        ).reset_index(drop=True)
        df_combined["_is_pred"] = df_combined["game_id"].isin(pred_game_ids)

        overrides = build_team_regression_overrides(
            df_combined, preseason_regression=BEST_REG,
            qb_change_bonus=BEST_QB_BONUS,
        )
        df_feat = compute_elo_features(
            df_combined,
            k_factor=BEST_K, home_advantage=BEST_HFA,
            preseason_regression=BEST_REG,
            team_regression_overrides=overrides,
            decay_half_life=BEST_DECAY,
        )
        df_feat = compute_qb_features(df_feat)
        df_feat = compute_situational_features(df_feat)

        is_pred = df_feat["_is_pred"].values
        is_train = ~is_pred

        train_y = df_feat.loc[is_train, "home_win"].astype(int).values
        x_train = _build_feature_matrix(df_feat.loc[is_train], feat_cols)
        x_pred = _build_feature_matrix(df_feat.loc[is_pred], feat_cols)

        pipe = _fit_platt(x_train, train_y)
        prob = pipe.predict_proba(x_pred)[:, 1]

        pred_ids = df_feat.loc[is_pred, "game_id"].values
        actual = np.array([
            actuals.get(gid, np.nan)
            if pd.notna(actuals.get(gid, np.nan)) else np.nan
            for gid in pred_ids
        ], dtype=float)

        df_week = pd.DataFrame({
            "game_id": pred_ids,
            "season": season,
            "week": week,
            "home_win_actual": actual,
            f"{label}_prob": prob.round(4),
        })
        for col in QB_CHANGE_COLS:
            if col in df_feat.columns:
                df_week[col] = df_feat.loc[is_pred, col].values

        metrics = _compute_metrics(actual, prob)
        week_metrics.append({"season": season, "week": week, **metrics})
        all_preds.append(df_week)

    if not all_preds:
        return {"overall": {}, "weekly": [], "num_games": 0}

    df_all = pd.concat(all_preds, ignore_index=True)
    overall_prob = df_all[f"{label}_prob"].values
    actual = df_all["home_win_actual"].astype(int).values
    overall = _compute_metrics(actual, overall_prob)

    return {
        "overall": overall,
        "weekly": week_metrics,
        "num_games": len(df_all),
        "df": df_all,
    }


# --- Fitted-once ---


def _run_fitted_once(
    df_raw: pd.DataFrame,
    feat_cols: List[str],
    label: str = "model",
) -> Dict:
    """Fitted-once: train on 2021–2024, predict 2025.
    Matches the incumbent pipeline: compute on full dataset,
    mask by season for train/pred split.
    """
    eligible = (
        df_raw.get(MODEL_ELIGIBLE_COLUMN, pd.Series(True))
        & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
    )

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)
    df = df[eligible].copy().reset_index(drop=True)

    is_train = df["season"].isin(TRAIN_SEASONS).values
    is_pred = (df["season"] == HOLDOUT_SEASON).values

    if is_pred.sum() == 0:
        return {"overall": {}, "num_games": 0}

    x_train = _build_feature_matrix(df.loc[is_train], feat_cols)
    x_pred = _build_feature_matrix(df.loc[is_pred], feat_cols)
    train_y = df.loc[is_train, TARGET_COLUMN].astype(int).values

    pipe = _fit_platt(x_train, train_y)
    prob = pipe.predict_proba(x_pred)[:, 1]

    actual = df.loc[is_pred, TARGET_COLUMN].astype(float).values

    overall = _compute_metrics(actual, prob)

    df_out = pd.DataFrame({
        "game_id": df.loc[is_pred, "game_id"].values,
        "season": HOLDOUT_SEASON,
        "home_win_actual": actual,
        f"{label}_prob": prob.round(4),
    })
    for col in QB_CHANGE_COLS:
        if col in df.columns:
            df_out[col] = df.loc[is_pred, col].values

    return {"overall": overall, "num_games": len(df_out), "df": df_out}


# --- Subset analysis ---


def _qb_subset_analysis(
    results: Dict[str, Dict],
    qb_changed: bool,
) -> Dict[str, Dict]:
    """Evaluate models on QB-change or non-QB-change games."""
    subset = {}
    for label, res in results.items():
        df = res.get("df")
        if df is None or len(df) == 0:
            continue
        if qb_changed:
            mask = (
                df.get("home_qb_changed", pd.Series(0)).fillna(0).astype(bool)
                | df.get("away_qb_changed", pd.Series(0)).fillna(0).astype(bool)
            )
        else:
            has_home = df.get("home_qb_changed", pd.Series(0)).fillna(0).astype(bool)
            has_away = df.get("away_qb_changed", pd.Series(0)).fillna(0).astype(bool)
            mask = (~has_home) & (~has_away)
        if mask.sum() == 0:
            subset[label] = {"num_games": 0, "overall": {}}
            continue
        sub = df[mask]
        prob_col = f"{label}_prob"
        if prob_col not in sub.columns:
            subset[label] = {"num_games": 0, "overall": {}}
            continue
        m = _compute_metrics(
            sub["home_win_actual"].values,
            sub[prob_col].values,
        )
        subset[label] = {
            "num_games": int(mask.sum()),
            "overall": m,
        }
    return subset


# --- Run / report ---


def run_qb_continuity(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = DEFAULT_REPORT,
) -> str:
    """Run QB-continuity experiment with all model variants."""
    print("=== QB Continuity Experiment ===")

    fp = Path(ft_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    df_raw = pd.read_parquet(fp)

    eligible = (
        df_raw.get(MODEL_ELIGIBLE_COLUMN, pd.Series(True))
        & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
    )
    total_eligible = int(eligible.sum())
    print(f"  Total eligible games: {total_eligible}")

    # --- Rolling simulation ---
    print("\n--- Rolling Simulation (All Seasons) ---")
    rolling_results = {}
    for name, cols, _, desc in MODEL_VARIANTS:
        print(f"  Running {name}...")
        rolling_results[name] = _run_rolling_simulation(df_raw, cols, label=name)
        m = rolling_results[name]["overall"]
        print(f"    {name}: {m.get('log_loss', 'N/A')} "
              f"({rolling_results[name]['num_games']} games)")

    # --- Fitted-once ---
    print("\n--- Fitted-Once (2021-2024 train, 2025 predict) ---")
    fitted_results = {}
    for name, cols, _, desc in MODEL_VARIANTS:
        print(f"  Running {name}...")
        fitted_results[name] = _run_fitted_once(df_raw, cols, label=name)
        m = fitted_results[name]["overall"]
        print(f"    {name}: {m.get('log_loss', 'N/A')} "
              f"({fitted_results[name]['num_games']} games)")

    # --- Bootstrap CI vs incumbent (rolling) ---
    print("\n--- Bootstrap CI (Rolling, Δ = challenger − incumbent) ---")
    bootstrap_ci = {}
    inc_roll_df = rolling_results["incumbent"].get("df")
    if inc_roll_df is not None and len(inc_roll_df) > 0:
        y_t = inc_roll_df["home_win_actual"].values
        p_inc = inc_roll_df["incumbent_prob"].values
        for name, _, _, desc in MODEL_VARIANTS:
            if name == "incumbent":
                continue
            df_challenger = rolling_results[name].get("df")
            if df_challenger is None or len(df_challenger) == 0:
                continue
            prob_col = f"{name}_prob"
            if prob_col not in df_challenger.columns:
                continue
            p_chal = df_challenger[prob_col].values
            mean_d, ci_l, ci_h = _bootstrap_delta(y_t, p_inc, p_chal)
            bootstrap_ci[name] = {
                "mean_delta": mean_d, "ci_low": ci_l, "ci_high": ci_h,
            }
            print(f"  {name} vs incumbent: {mean_d:.4f} "
                  f"[{ci_l:.4f}, {ci_h:.4f}]")

    # --- QB-change subset ---
    print("\n--- Subset Analysis ---")
    qb_changed_subset = _qb_subset_analysis(rolling_results, qb_changed=True)
    no_qb_change_subset = _qb_subset_analysis(rolling_results, qb_changed=False)

    # --- Incumbent holdout reference ---
    inc_holdout = fitted_results["incumbent"]["overall"].get("log_loss", 1)

    # --- Calibration / confidence / worst pred for incumbent + best challenger ---
    # Find best challenger on rolling
    inc_roll_ll = rolling_results["incumbent"]["overall"].get("log_loss", 1)
    best_challenger = "incumbent"
    best_chal_diff = 0.0
    for name, _, _, _ in MODEL_VARIANTS:
        if name == "incumbent":
            continue
        ll = rolling_results[name]["overall"].get("log_loss", 1)
        diff = ll - inc_roll_ll  # positive = incumbent better
        if diff < best_chal_diff:
            best_chal_diff = diff
            best_challenger = name

    # --- Generate report ---
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# QB Continuity Experiment\n\n")
        f.write(
            f"*Generated by `sportslab qb-continuity`"
            f" ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*\n\n"
        )

        f.write("## Executive Summary\n\n")
        f.write(f"Incumbent holdout (2025) LL: **{inc_holdout:.4f}**\n\n")
        f.write(
            "Tests 5 QB-continuity feature variants against the incumbent "
            "on rolling all-season simulation and fitted-once holdout. "
            "No model is promoted in this research pass.\n\n"
        )

        f.write("## Model Variants\n\n")
        f.write("| Label | Features | Live Safety | Description |\n")
        f.write("|-------|----------|-------------|-------------|\n")
        for name, cols, live_safety, desc in MODEL_VARIANTS:
            feat_str = ", ".join(cols) if len(cols) <= 6 else f"{len(cols)} features"
            f.write(f"| {name} | {feat_str} | {live_safety} | {desc} |\n")
        f.write("\n")

        f.write("## Rolling Simulation (All Seasons)\n\n")
        f.write("| Model | Log Loss | Brier | Accuracy | Games | Δ vs Incumbent |\n")
        f.write("|-------|----------|-------|----------|-------|----------------|\n")
        for name, _, _, _ in MODEL_VARIANTS:
            r = rolling_results[name]
            m = r["overall"]
            delta = m.get("log_loss", 0) - inc_roll_ll
            delta_str = f"{delta:+.4f}" if name != "incumbent" else "—"
            f.write(
                f"| {name} | {m.get('log_loss', 'N/A')}"
                f" | {m.get('brier', 'N/A')}"
                f" | {m.get('accuracy', 'N/A')}"
                f" | {r['num_games']}"
                f" | {delta_str} |\n"
            )
        f.write("\n")

        f.write("### Bootstrap CI vs Incumbent (Rolling, 1000 iterations)\n\n")
        f.write("Δ = challenger LL − incumbent LL. Negative Δ means challenger better.\n\n")
        f.write("| Challenger | Mean Δ | 95% CI Lower | 95% CI Upper |\n")
        f.write("|------------|--------|--------------|--------------|\n")
        for name, _, _, _ in MODEL_VARIANTS:
            if name == "incumbent":
                continue
            ci = bootstrap_ci.get(name, {})
            mean_d = ci.get("mean_delta", "—")
            ci_l = ci.get("ci_low", "—")
            ci_h = ci.get("ci_high", "—")
            f.write(f"| {name} | {mean_d} | {ci_l} | {ci_h} |\n")
        f.write("\n")

        f.write("## Fitted-Once (Holdout: 2025)\n\n")
        f.write("| Model | Log Loss | Brier | Accuracy | Games | Δ vs Incumbent |\n")
        f.write("|-------|----------|-------|----------|-------|----------------|\n")
        for name, _, _, _ in MODEL_VARIANTS:
            r = fitted_results[name]
            m = r["overall"]
            delta = m.get("log_loss", 0) - inc_holdout
            delta_str = f"{delta:+.4f}" if name != "incumbent" else "—"
            f.write(
                f"| {name} | {m.get('log_loss', 'N/A')}"
                f" | {m.get('brier', 'N/A')}"
                f" | {m.get('accuracy', 'N/A')}"
                f" | {r['num_games']}"
                f" | {delta_str} |\n"
            )
        f.write("\n")

        f.write("## QB-Change Subset (Rolling)\n\n")
        f.write("| Model | Log Loss | Games |\n")
        f.write("|-------|----------|-------|\n")
        for name, _, _, _ in MODEL_VARIANTS:
            sub = qb_changed_subset.get(name, {})
            m = sub.get("overall", {})
            f.write(f"| {name} | {m.get('log_loss', 'N/A')} | {sub.get('num_games', 0)} |\n")
        f.write("\n")

        f.write("## Non-QB-Change Subset (Rolling)\n\n")
        f.write("| Model | Log Loss | Games |\n")
        f.write("|-------|----------|-------|\n")
        for name, _, _, _ in MODEL_VARIANTS:
            sub = no_qb_change_subset.get(name, {})
            m = sub.get("overall", {})
            f.write(f"| {name} | {m.get('log_loss', 'N/A')} | {sub.get('num_games', 0)} |\n")
        f.write("\n")

        # Calibration buckets for incumbent + best challenger
        f.write("## Calibration Buckets\n\n")
        f.write(f"Incumbent vs best challenger (**{best_challenger}**) on rolling all-season.\n\n")

        for display_name, key in [("Incumbent", "incumbent"),
                                   ("Best challenger", best_challenger)]:
            if key not in rolling_results:
                continue
            df = rolling_results[key].get("df")
            if df is None or len(df) == 0:
                continue
            prob_col = f"{key}_prob"
            if prob_col not in df.columns:
                continue
            buckets = _calibration_buckets(
                df["home_win_actual"].values,
                df[prob_col].values,
            )
            f.write(f"### {display_name} ({key})\n\n")
            f.write("| Bucket | N | Mean Pred | Mean Actual | Cal Error |\n")
            f.write("|--------|---|-----------|-------------|-----------|\n")
            for b in buckets:
                f.write(
                    f"| {b['bucket']} | {b['n']} | {b['mean_pred']:.4f}"
                    f" | {b['mean_actual']:.4f} | {b['cal_error']:.4f} |\n"
                )
            f.write("\n")

        f.write("## Confidence Buckets\n\n")
        f.write("Confidence = 2 × |prob − 0.5|. Higher = more confident.\n\n")

        for display_name, key in [("Incumbent", "incumbent"),
                                   ("Best challenger", best_challenger)]:
            if key not in rolling_results:
                continue
            df = rolling_results[key].get("df")
            if df is None or len(df) == 0:
                continue
            prob_col = f"{key}_prob"
            if prob_col not in df.columns:
                continue
            buckets = _confidence_buckets(
                df["home_win_actual"].values,
                df[prob_col].values,
            )
            f.write(f"### {display_name} ({key})\n\n")
            f.write("| Confidence | N | Log Loss | Brier | Accuracy |\n")
            f.write("|------------|---|----------|-------|----------|\n")
            for b in buckets:
                f.write(
                    f"| {b['bucket']} | {b['n']} | {b['log_loss']:.4f}"
                    f" | {b['brier']:.4f} | {b['accuracy']:.4f} |\n"
                )
            f.write("\n")

        # Worst 20 predictions
        f.write("## Worst 20 Predictions (Incumbent, Rolling)\n\n")
        df_inc = rolling_results["incumbent"].get("df")
        if df_inc is not None and len(df_inc) > 0:
            worst = _worst_predictions(
                df_inc["home_win_actual"].values,
                df_inc["incumbent_prob"].values,
                df_inc.get("game_id", pd.Series([""] * len(df_inc))).values,
                df_inc.get("home_team", pd.Series([""] * len(df_inc))).values,
                n=N_WORST,
            )
            f.write("| # | Game ID | Home | Actual | Pred | Log Loss |\n")
            f.write("|---|---------|------|--------|------|----------|\n")
            for i, w in enumerate(worst, 1):
                f.write(
                    f"| {i} | {w['game_id']} | {w['team']}"
                    f" | {w['actual']} | {w['pred']:.4f} | {w['log_loss_contrib']:.4f} |\n"
                )
            f.write("\n")

        # Live-safety summary
        f.write("## Live-Safety Labels\n\n")
        f.write(
            "All QB features in this experiment are computed from "
            "oracle (final actual starter) data. They are "
            "`research_oracle` unless the user supplies a pregame "
            "QB input CSV through the auditable QB input workflow "
            "(`sportslab predict-future --qb-input`).\n\n"
        )
        f.write("| Label | Meaning |\n")
        f.write("|-------|--------|\n")
        f.write(
            "| research_oracle | Uses final actual starter from "
            "nflreadpy schedule. Not live-safe unless overridden.\n"
        )
        f.write(
            "| live_safe_no_qb | Uses only rolling_mov_3. "
            "No QB dependency. Always live-safe.\n"
        )
        f.write(
            "| live_safe_if_qb_input_available | Uses QB starter "
            "identity. Live-safe only if pregame QB input CSV "
            "is provided.\n\n"
        )

        # Recommendation
        f.write("## Recommendation\n\n")
        f.write("**No model is promoted in this research pass.**\n\n")

        best_roll_name = min(
            [n for n, _, _, _ in MODEL_VARIANTS if n != "incumbent"],
            key=lambda n: rolling_results[n]["overall"].get("log_loss", 1),
        )
        best_roll_ll = rolling_results[best_roll_name]["overall"].get("log_loss", "N/A")
        best_ci = bootstrap_ci.get(best_roll_name, {})
        best_mean_d = best_ci.get("mean_delta", "N/A")
        best_ci_l = best_ci.get("ci_low", "N/A")
        best_ci_h = best_ci.get("ci_high", "N/A")

        f.write(f"Best challenger on rolling: **{best_roll_name}** "
                f"({best_roll_ll}, Δ={best_mean_d} "
                f"[{best_ci_l}, {best_ci_h}]).\n\n")

        # Answers to key questions
        beats_roll = any(
            rolling_results[n]["overall"].get("log_loss", 1) < inc_roll_ll
            for n, _, _, _ in MODEL_VARIANTS if n != "incumbent"
        )
        beats_hold = any(
            fitted_results[n]["overall"].get("log_loss", 1) < inc_holdout
            for n, _, _, _ in MODEL_VARIANTS if n != "incumbent"
        )

        f.write("1. **Does any QB-continuity variant beat the incumbent "
                f"in rolling all-season simulation?** "
                f"{'Yes' if beats_roll else 'No'}.\n")
        f.write("2. **Does any QB-continuity variant beat the incumbent "
                f"in fitted-once 2025?** "
                f"{'Yes' if beats_hold else 'No'}.\n")
        f.write("3. **Does any QB-continuity variant materially improve "
                "QB-change games?** See subset table above.\n")
        f.write("4. **Does the improvement survive bootstrap uncertainty?** "
                "Check CI intervals above.\n")
        f.write("5. **Should the live default change?** No. Continue "
                "with incumbent (oracle) or no-QB fallback (live).\n\n")

        f.write("---\n")
        f.write(
            f"*Incumbent: {INCUMBENT_VERSION}, "
            f"{INCUMBENT_HOLDOUT_LL} holdout LL. "
            f"No model promoted in this pass.*\n"
        )

    # --- Console summary ---
    print(f"\nQB Continuity report: {rp}")
    print(f"  Incumbent rolling:  {inc_roll_ll:.4f} "
          f"({rolling_results['incumbent']['num_games']} games)")
    print(f"  Incumbent holdout:  {inc_holdout:.4f} "
          f"({fitted_results['incumbent']['num_games']} games)")
    print(f"  Best challenger rolling: {best_roll_name} "
          f"({best_roll_ll})")
    print(f"  Bootstrap Δ vs incumbent: {best_mean_d} [{best_ci_l}, {best_ci_h}]")

    return str(rp)
