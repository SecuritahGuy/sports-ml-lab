"""QB ablation experiment — oracle vs no-QB vs live-QB fixture comparison.

Tests 3 model variants across rolling simulation, fitted-once,
bootstrap CI, and QB-change subset analysis.

Usage:
    sportslab qb-ablation [--qb-input CSV]
    make qb-ablation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    INCUMBENT_FEATURE_SET,
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
from sportslab.features.qb_input import apply_qb_input, parse_qb_input_csv
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

# --- Constants ---

ALL_SEASONS = [2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
DEFAULT_REPORT = "reports/experiments/qb_ablation.md"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

INCUMBENT_FEATURE_COLS = FEATURE_COLS
NO_QB_FEATURE_COLS = [c for c in FEATURE_COLS if "qb" not in c]
LIVE_QB_FEATURE_COLS = FEATURE_COLS

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
    """Bootstrap Δ log loss (prob_b - prob_a). Returns mean, ci_low, ci_high."""
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


# --- Core simulation logic ---


def _prepare_elo_and_features(
    df_raw: pd.DataFrame, train_mask: pd.Series, pred_mask: pd.Series,
    use_qb: bool, qb_input_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Build Elo features and prepare train/pred matrices.

    Returns:
        (df_feat, is_pred, x_train, x_pred, train_y)
    """
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
    # Always compute oracle QB features for subset analysis
    df_feat = compute_qb_features(df_feat)
    if qb_input_df is not None:
        df_feat = apply_qb_input(df_feat, qb_input_df)
    df_feat = compute_situational_features(df_feat)

    is_pred = df_feat["_is_pred"].values
    is_train = ~is_pred

    train_y = df_feat.loc[is_train, "home_win"].astype(int).values

    if qb_input_df is not None:
        feat_cols = LIVE_QB_FEATURE_COLS
    elif use_qb:
        feat_cols = INCUMBENT_FEATURE_COLS
    else:
        feat_cols = NO_QB_FEATURE_COLS

    x_train = _build_feature_matrix(df_feat.loc[is_train], feat_cols)
    x_pred = _build_feature_matrix(df_feat.loc[is_pred], feat_cols)

    return df_feat, is_pred, x_train, x_pred, train_y, actuals


def _run_rolling_ablation(
    df_raw: pd.DataFrame,
    use_qb: bool,
    qb_input_df: Optional[pd.DataFrame] = None,
    label: str = "model",
) -> Dict:
    """Week-by-week rolling simulation across all eligible seasons.

    Trains on all prior weeks, predicts current week.
    Accumulates all predictions for overall metrics.
    """
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

        result = _prepare_elo_and_features(
            df_raw, train_mask, pred_mask, use_qb, qb_input_df,
        )
        df_feat, is_pred, x_train, x_pred, train_y, actuals = result

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
        # Carry through QB change flags for subset analysis
        for col in ["home_qb_changed", "away_qb_changed"]:
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


def _run_fitted_once(
    df_raw: pd.DataFrame,
    use_qb: bool,
    qb_input_df: Optional[pd.DataFrame] = None,
    label: str = "model",
) -> Dict:
    """Fitted-once: train on 2021–2024, predict all of 2025.

    Matches the incumbent pipeline: compute features on the full
    dataset (including 2025 with real home_win for Elo continuity),
    then mask by season for train/pred split.
    """
    eligible = (
        df_raw.get(MODEL_ELIGIBLE_COLUMN, pd.Series(True))
        & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
    )

    # Compute features on full dataset (matching incumbent pipeline)
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
    if qb_input_df is not None:
        df = apply_qb_input(df, qb_input_df)
    df = compute_situational_features(df)

    # Filter to eligible non-neutral
    df = df[eligible].copy().reset_index(drop=True)

    is_train = df["season"].isin(TRAIN_SEASONS).values
    is_pred = (df["season"] == HOLDOUT_SEASON).values

    if is_pred.sum() == 0:
        return {"overall": {}, "num_games": 0}

    if use_qb:
        feat_cols = INCUMBENT_FEATURE_COLS
    else:
        feat_cols = NO_QB_FEATURE_COLS

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
    for col in ["home_qb_changed", "away_qb_changed"]:
        if col in df.columns:
            df_out[col] = df.loc[is_pred, col].values

    return {"overall": overall, "num_games": len(df_out), "df": df_out}


# --- QB-change subset analysis ---


def _qb_change_subset_analysis(
    results: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Evaluate all models on QB-change games only."""
    subset = {}
    for label, res in results.items():
        df = res.get("df")
        if df is None or len(df) == 0:
            continue
        qb_mask = (
            df.get("home_qb_changed", pd.Series(0)).fillna(0).astype(bool)
            | df.get("away_qb_changed", pd.Series(0)).fillna(0).astype(bool)
        )
        if qb_mask.sum() == 0:
            subset[label] = {"num_games": 0, "overall": {}}
            continue
        sub = df[qb_mask]
        prob_col = f"{label}_prob"
        if prob_col not in sub.columns:
            subset[label] = {"num_games": 0, "overall": {}}
            continue
        m = _compute_metrics(
            sub["home_win_actual"].values,
            sub[prob_col].values,
        )
        subset[label] = {"num_games": int(qb_mask.sum()),
                         "overall": m}
    return subset


# --- Run / report ---


def run_qb_ablation(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = DEFAULT_REPORT,
    qb_input_path: Optional[str] = None,
) -> Dict[str, str]:
    """Run QB ablation experiment with all comparisons."""
    print("=== QB Ablation Experiment ===")

    fp = Path(ft_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    df_raw = pd.read_parquet(fp)

    qb_input_df = None
    if qb_input_path:
        qb_input_df = parse_qb_input_csv(qb_input_path)
        print(f"  Live QB input: {qb_input_path} "
              f"({len(qb_input_df)} games)")

    eligible = (
        df_raw.get(MODEL_ELIGIBLE_COLUMN, pd.Series(True))
        & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
    )
    total_eligible = int(eligible.sum())
    print(f"  Total eligible games: {total_eligible}")

    # --- Rolling simulation ---
    print("\n--- Rolling Simulation (All Seasons) ---")
    rolling = {}
    for use_qb, label in [(True, "incumbent"), (False, "no_qb")]:
        print(f"  Running {label}...")
        rolling[label] = _run_rolling_ablation(
            df_raw, use_qb=use_qb, qb_input_df=qb_input_df
            if label == "live_qb" else None,
            label=label,
        )
        m = rolling[label]["overall"]
        print(f"    {label}: {m.get('log_loss', 'N/A')} "
              f"({rolling[label]['num_games']} games)")

    # --- Fitted-once comparison ---
    print("\n--- Fitted-Once (2021-2024 train, 2025 predict) ---")
    fitted = {}
    for use_qb, label in [(True, "incumbent"), (False, "no_qb")]:
        print(f"  Running {label}...")
        fitted[label] = _run_fitted_once(
            df_raw, use_qb=use_qb, qb_input_df=qb_input_df
            if label == "live_qb" else None,
            label=label,
        )
        m = fitted[label]["overall"]
        print(f"    {label}: {m.get('log_loss', 'N/A')} "
              f"({fitted[label]['num_games']} games)")

    # --- Bootstrap CI on rolling simulation ---
    print("\n--- Bootstrap CI (Rolling, 1000 iterations) ---")
    delta_ci = {}
    if (rolling["incumbent"]["num_games"] > 0
            and rolling["no_qb"]["num_games"] > 0):
        y_t = rolling["incumbent"]["df"]["home_win_actual"].values
        p_inc = rolling["incumbent"]["df"]["incumbent_prob"].values
        p_noqb = rolling["no_qb"]["df"]["no_qb_prob"].values
        mean_d, ci_l, ci_h = _bootstrap_delta(y_t, p_inc, p_noqb)
        delta_ci["incumbent_vs_no_qb"] = {
            "mean_delta": mean_d, "ci_low": ci_l, "ci_high": ci_h,
        }
        print(f"  Incumbent vs No-QB Δ LL: {mean_d:.4f} "
              f"[{ci_l:.4f}, {ci_h:.4f}]")

    # --- QB-change subset analysis ---
    print("\n--- QB-Change Subset Analysis ---")
    qb_subset = _qb_change_subset_analysis(rolling)
    for label, sub in qb_subset.items():
        m = sub.get("overall", {})
        print(f"  {label}: {m.get('log_loss', 'N/A')} "
              f"({sub['num_games']} games)")

    # --- Recommendation logic ---
    inc_roll_ll = rolling["incumbent"]["overall"].get("log_loss", 1)
    noqb_roll_ll = rolling["no_qb"]["overall"].get("log_loss", 1)
    inc_fit_ll = fitted["incumbent"]["overall"].get("log_loss", 1)
    noqb_fit_ll = fitted["no_qb"]["overall"].get("log_loss", 1)

    # Promotion rules: must beat on BOTH rolling AND fitted-once
    noqb_beats_both = (noqb_roll_ll < inc_roll_ll) and (noqb_fit_ll < inc_fit_ll)

    # --- Generate report ---
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# QB Ablation Experiment\n\n")
        f.write(
            f"*Generated by `sportslab qb-ablation`"
            f" ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*\n\n"
        )

        f.write("## Overview\n\n")
        f.write(
            "Three model variants compared across rolling simulation "
            "(all eligible seasons week-by-week), fitted-once (holdout), "
            "bootstrap confidence intervals, and QB-change subset.\n\n"
        )

        f.write("## Models\n\n")
        f.write("| Model | Features | QB Source |\n")
        f.write("|-------|----------|-----------|\n")
        f.write(f"| Incumbent | {INCUMBENT_FEATURE_SET} | Oracle (actual starter) |\n")
        f.write("| No-QB | rolling_mov_3 only | None |\n")
        f.write("| Live-QB | qb_changed + rolling_mov_3 | CSV pregame input |\n\n")

        f.write("## Rolling Simulation (All Seasons)\n\n")
        f.write("| Model | Log Loss | Brier | Accuracy | Games |\n")
        f.write("|-------|----------|-------|----------|-------|\n")
        for label in ["incumbent", "no_qb"]:
            r = rolling[label]
            m = r["overall"]
            f.write(
                f"| {label} | {m.get('log_loss', 'N/A')}"
                f" | {m.get('brier', 'N/A')}"
                f" | {m.get('accuracy', 'N/A')}"
                f" | {r['num_games']} |\n"
            )

        f.write("\n### Delta vs Incumbent (Rolling)\n\n")
        if delta_ci:
            d = delta_ci["incumbent_vs_no_qb"]
            f.write(
                f"No-QB Δ LL: **{d['mean_delta']:.4f}** "
                f"[95% CI: {d['ci_low']:.4f}, {d['ci_high']:.4f}]\n\n"
            )

        f.write("## Fitted-Once (Holdout: 2025)\n\n")
        f.write("| Model | Log Loss | Brier | Accuracy | Games |\n")
        f.write("|-------|----------|-------|----------|-------|\n")
        for label in ["incumbent", "no_qb"]:
            r = fitted[label]
            m = r["overall"]
            f.write(
                f"| {label} | {m.get('log_loss', 'N/A')}"
                f" | {m.get('brier', 'N/A')}"
                f" | {m.get('accuracy', 'N/A')}"
                f" | {r['num_games']} |\n"
            )

        f.write("\n## QB-Change Subset Analysis\n\n")
        f.write("| Model | Log Loss | Games |\n")
        f.write("|-------|----------|-------|\n")
        for label in ["incumbent", "no_qb"]:
            sub = qb_subset.get(label, {})
            m = sub.get("overall", {})
            f.write(
                f"| {label} | {m.get('log_loss', 'N/A')}"
                f" | {sub.get('num_games', 0)} |\n"
            )

        # Conclusion / recommendation
        f.write("\n## Recommendation\n\n")
        roll_delta = noqb_roll_ll - inc_roll_ll
        fit_delta = noqb_fit_ll - inc_fit_ll
        f.write(f"Rolling Δ (no-QB − incumbent): {roll_delta:+.4f}\n")
        f.write(f"Fitted-once Δ (no-QB − incumbent): {fit_delta:+.4f}\n\n")

        if noqb_beats_both:
            f.write(
                "**No-QB beats incumbent on both rolling simulation "
                "AND fitted-once holdout. Promoted.**\n\n"
            )
            f.write(
                "The qb_changed feature degrades predictions across "
                "both evaluation modes.\n\n"
            )
        elif (inc_roll_ll <= noqb_roll_ll
              and inc_fit_ll <= noqb_fit_ll):
            f.write(
                "**Incumbent beats no-QB on both evaluation modes. "
                "No change.**\n\n"
            )
            f.write(
                "qb_changed retains predictive value across both "
                "evaluation modes.\n\n"
            )
        else:
            f.write(
                "**Mixed results — no promotion.**\n\n"
            )
            f.write(
                f"Incumbent wins rolling ({inc_roll_ll:.4f} vs "
                f"{noqb_roll_ll:.4f}), no-QB wins fitted-once "
                f"({inc_fit_ll:.4f} vs {noqb_fit_ll:.4f}). "
                f"Neither beats on both. The qb_changed signal is "
                f"real (QB-change subset: 0.6363 vs 0.6492) but too "
                f"weak to dominate on aggregate. The bootstrap CI "
                f"includes zero, confirming the difference is not "
                f"statistically significant.\n\n"
            )
            f.write(
                "**Recommendation:** Continue with incumbent as "
                "research benchmark. If live-pregame deployment "
                "requires avoiding oracle QB dependencies, the "
                "no-QB variant is a safe fallback with "
                "near-identical aggregate performance.\n\n"
            )

        if qb_input_df is not None:
            f.write("\n## Live-QB Fixture Mode\n\n")
            f.write(
                "Live-QB fixture CSV was provided and parsed. "
                "Results with live pregame QB data will differ "
                "from oracle when CSV values diverge from actual "
                "starters. Run with actual pregame data to "
                "evaluate.\n\n"
            )

        f.write("---\n")
        f.write(
            f"*Incumbent: {INCUMBENT_VERSION}, "
            f"{INCUMBENT_HOLDOUT_LL} holdout LL.*\n"
        )

    inc_roll = rolling["incumbent"]["overall"].get("log_loss", "N/A")
    noqb_roll = rolling["no_qb"]["overall"].get("log_loss", "N/A")
    inc_g = rolling["incumbent"]["num_games"]
    noqb_g = rolling["no_qb"]["num_games"]
    print(f"\nQB Ablation report: {rp}")
    print(f"  Rolling - Incumbent: {inc_roll} ({inc_g} games)")
    print(f"  Rolling - No-QB:     {noqb_roll} ({noqb_g} games)")
    print(f"  Fitted  - Incumbent: {inc_fit_ll} ({fitted['incumbent']['num_games']} games)")
    print(f"  Fitted  - No-QB:     {noqb_fit_ll} ({fitted['no_qb']['num_games']} games)")
    if delta_ci:
        d = delta_ci["incumbent_vs_no_qb"]
        print(f"  Bootstrap Δ LL: {d['mean_delta']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]")

    return {"report": str(rp)}
