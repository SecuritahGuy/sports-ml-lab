"""Gated QB-experience experiment — tests whether QB experience features
help only when gated behind a QB-change flag.

The hypothesis:
  QB experience features (starts, diff) help when a QB change occurs,
  but add noise when applied to every game. Gating them by the
  QB-change flag may preserve the lift on QB-change games without
  damaging non-QB-change games.

Compares 5 model variants:
  A. Incumbent (qb_changed + rolling_mov_3)
  B. qb_experience_global (qb_changed + starts + diff + rolling_mov_3)
  C. qb_experience_gated_binary (global gate: all experience gated)
  D. qb_experience_gated_team_specific (per-team gates)
  E. qb_experience_gated_simple_diff (only qb_starts_diff gated)

Usage:
    sportslab qb-gated-experience
    make qb-gated-experience
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sportslab.evaluation.experiment_utils import (
    bootstrap_delta,
    build_feature_matrix,
    calibration_buckets,
    compute_metrics,
    confidence_buckets,
    fit_platt,
    worst_predictions,
)
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
DEFAULT_REPORT = "reports/experiments/qb_gated_experience.md"
N_BOOTSTRAP = 1000

QB_CHANGE_COLS = ["home_qb_changed", "away_qb_changed"]
MOV_COLS = ["home_rolling_mov_3", "away_rolling_mov_3"]
STARTS_COLS = [
    "home_qb_starts_this_season_pre", "away_qb_starts_this_season_pre",
    "qb_starts_diff",
]

# Gated column names
GATED_HOME_STARTS = "gated_home_qb_starts"
GATED_AWAY_STARTS = "gated_away_qb_starts"
GATED_STARTS_DIFF = "gated_qb_starts_diff"
GATED_STARTS_COLS = [GATED_HOME_STARTS, GATED_AWAY_STARTS, GATED_STARTS_DIFF]

# --- Model variant definitions ---
# (name, feature_cols, live_safety, description)

MODEL_VARIANTS = [
    ("incumbent", FEATURE_COLS,
     "research_oracle",
     "Incumbent: qb_changed + rolling_mov_3"),
    ("qb_experience_global", QB_CHANGE_COLS + STARTS_COLS + MOV_COLS,
     "research_oracle",
     "QB experience global: qb_changed + starts + diff + rolling_mov_3"),
    ("qb_experience_gated_binary", QB_CHANGE_COLS + GATED_STARTS_COLS + MOV_COLS,
     "research_oracle",
     "QB experience gated (binary): all starts gated by global QB-change flag"),
    ("qb_experience_gated_team_specific", QB_CHANGE_COLS + GATED_STARTS_COLS + MOV_COLS,
     "research_oracle",
     "QB experience gated (team): home starts * home_changed, away * away_changed"),
    ("qb_experience_gated_simple_diff", QB_CHANGE_COLS + [GATED_STARTS_DIFF] + MOV_COLS,
     "research_oracle",
     "QB experience gated (diff only): only qb_starts_diff gated"),
]

LIVE_SAFE_LABELS = {
    "incumbent": "research_oracle",
    "qb_experience_global": "research_oracle",
    "qb_experience_gated_binary": "research_oracle",
    "qb_experience_gated_team_specific": "research_oracle",
    "qb_experience_gated_simple_diff": "research_oracle",
}

# --- Gated feature computation ---


def compute_gated_columns(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Add gated QB-experience columns to DataFrame in-place.

    Variants:
      gated_binary: gate = 1 if home_qb_changed OR away_qb_changed else 0
      gated_team_specific: home gate = home_qb_changed, away gate = away_qb_changed
      gated_simple_diff: only diff gated by global gate

    Returns the DataFrame (modified in-place) with added gated columns.
    """
    has_home = df.get("home_qb_changed", pd.Series(0)).fillna(0).astype(int)
    has_away = df.get("away_qb_changed", pd.Series(0)).fillna(0).astype(int)
    global_gate = (has_home | has_away).values.astype(float)

    if variant in ("gated_binary", "gated_simple_diff"):
        home_starts = df.get("home_qb_starts_this_season_pre", pd.Series(0)).values
        away_starts = df.get("away_qb_starts_this_season_pre", pd.Series(0)).values
        starts_diff = df.get("qb_starts_diff", pd.Series(0)).values

        df[GATED_HOME_STARTS] = (home_starts * global_gate).round(4)
        df[GATED_AWAY_STARTS] = (away_starts * global_gate).round(4)
        df[GATED_STARTS_DIFF] = (starts_diff * global_gate).round(4)

    elif variant == "gated_team_specific":
        home_gate = has_home.values.astype(float)
        away_gate = has_away.values.astype(float)
        home_starts = df.get("home_qb_starts_this_season_pre", pd.Series(0)).values
        away_starts = df.get("away_qb_starts_this_season_pre", pd.Series(0)).values
        starts_diff = df.get("qb_starts_diff", pd.Series(0)).values

        df[GATED_HOME_STARTS] = (home_starts * home_gate).round(4)
        df[GATED_AWAY_STARTS] = (away_starts * away_gate).round(4)
        df[GATED_STARTS_DIFF] = (starts_diff * global_gate).round(4)

    return df


# --- Rolling simulation ---


def _run_rolling_simulation(
    df_raw: pd.DataFrame,
    feat_cols: List[str],
    gated_variant: str = "",
    label: str = "model",
) -> Dict:
    """Week-by-week rolling simulation across all eligible seasons.

    If gated_variant is non-empty, computes gated columns before
    building the feature matrix.
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
    all_coefs = []  # (season, week, coef_array) for coefficient diagnostics

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

        if gated_variant:
            df_feat = compute_gated_columns(df_feat, gated_variant)

        is_pred = df_feat["_is_pred"].values
        is_train = ~is_pred

        train_y = df_feat.loc[is_train, "home_win"].astype(int).values
        x_train = build_feature_matrix(df_feat.loc[is_train], feat_cols)
        x_pred = build_feature_matrix(df_feat.loc[is_pred], feat_cols)

        pipe = fit_platt(x_train, train_y)
        prob = pipe.predict_proba(x_pred)[:, 1]

        # Capture coefficients for diagnostic
        lr = pipe.named_steps["lr"]
        all_coefs.append({
            "season": season, "week": week,
            "coef": lr.coef_.flatten().copy(),
        })

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

        metrics = compute_metrics(actual, prob)
        week_metrics.append({"season": season, "week": week, **metrics})
        all_preds.append(df_week)

    if not all_preds:
        return {"overall": {}, "weekly": [], "num_games": 0,
                "coefs": [], "feature_names": []}

    df_all = pd.concat(all_preds, ignore_index=True)
    overall_prob = df_all[f"{label}_prob"].values
    actual = df_all["home_win_actual"].astype(int).values
    overall = compute_metrics(actual, overall_prob)

    # Feature names for coefficient diagnostics
    lr_first = all_coefs[0]["coef"]
    feat_names = ["elo_prob"] + feat_cols if len(lr_first) == 1 + len(feat_cols) else feat_cols

    return {
        "overall": overall,
        "weekly": week_metrics,
        "num_games": len(df_all),
        "df": df_all,
        "coefs": all_coefs,
        "feature_names": feat_names,
    }


# --- Fitted-once ---


def _run_fitted_once(
    df_raw: pd.DataFrame,
    feat_cols: List[str],
    gated_variant: str = "",
    label: str = "model",
) -> Dict:
    """Fitted-once: train on 2021-2024, predict 2025."""
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

    if gated_variant:
        df = compute_gated_columns(df, gated_variant)

    df = df[eligible].copy().reset_index(drop=True)

    is_train = df["season"].isin(TRAIN_SEASONS).values
    is_pred = (df["season"] == HOLDOUT_SEASON).values

    if is_pred.sum() == 0:
        return {"overall": {}, "num_games": 0, "coef": None}

    x_train = build_feature_matrix(df.loc[is_train], feat_cols)
    x_pred = build_feature_matrix(df.loc[is_pred], feat_cols)
    train_y = df.loc[is_train, TARGET_COLUMN].astype(int).values

    pipe = fit_platt(x_train, train_y)
    prob = pipe.predict_proba(x_pred)[:, 1]

    actual = df.loc[is_pred, TARGET_COLUMN].astype(float).values

    overall = compute_metrics(actual, prob)

    lr = pipe.named_steps["lr"]
    feat_names = ["elo_prob"] + feat_cols

    df_out = pd.DataFrame({
        "game_id": df.loc[is_pred, "game_id"].values,
        "season": HOLDOUT_SEASON,
        "home_win_actual": actual,
        f"{label}_prob": prob.round(4),
    })
    for col in QB_CHANGE_COLS:
        if col in df.columns:
            df_out[col] = df.loc[is_pred, col].values

    return {
        "overall": overall,
        "num_games": len(df_out),
        "df": df_out,
        "coef": lr.coef_.flatten().copy(),
        "feature_names": feat_names,
    }


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
        m = compute_metrics(
            sub["home_win_actual"].values,
            sub[prob_col].values,
        )
        subset[label] = {"num_games": int(mask.sum()), "overall": m}
    return subset


# --- Coefficient diagnostics ---


def _coefficient_diagnostics(
    results: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Compute coefficient diagnostics for each variant.

    Uses fitted-once coefficients (one per variant). Reports
    feature names, coefficients, and sign stability from
    per-week rolling coefficients.
    """
    diag = {}
    for label, res in results.items():
        coef = res.get("coef")
        feat_names = res.get("feature_names", [])
        if coef is None or len(feat_names) == 0:
            continue

        # Rolling coefficient sign stability
        rolling_coefs = res.get("coefs", [])
        if rolling_coefs:
            coef_matrix = np.array([c["coef"] for c in rolling_coefs])
            mean_coefs = np.mean(coef_matrix, axis=0)
            std_coefs = np.std(coef_matrix, axis=0)
            pos_frac = np.mean(coef_matrix > 0, axis=0)
        else:
            mean_coefs = coef
            std_coefs = np.zeros_like(coef)
            pos_frac = np.where(coef > 0, 1.0, 0.0)

        n_feat = len(feat_names)
        if len(mean_coefs) != n_feat:
            mean_coefs = np.full(n_feat, np.nan)
            std_coefs = np.full(n_feat, np.nan)
            pos_frac = np.full(n_feat, np.nan)

        features = []
        for i, name in enumerate(feat_names):
            features.append({
                "feature": name,
                "coef": round(float(mean_coefs[i]), 6),
                "std": round(float(std_coefs[i]), 6) if std_coefs[i] is not None else 0,
                "pos_frac": round(float(pos_frac[i]), 4),
                "sign_stable": bool(pos_frac[i] >= 0.95 or pos_frac[i] <= 0.05),
            })

        # Sort by absolute coefficient
        features.sort(key=lambda x: abs(x["coef"]), reverse=True)

        diag[label] = {
            "features": features,
            "n_rolling_models": len(rolling_coefs),
        }
    return diag


# --- Season/week breakdown ---


def _season_week_breakdown(
    df: pd.DataFrame,
    prob_col: str,
) -> List[Dict]:
    """Metrics by season and week bucket."""
    results = []
    if df is None or len(df) == 0:
        return results

    seasons = sorted(df["season"].unique())
    week_buckets = [
        (1, 4, "Weeks 1-4"),
        (5, 9, "Weeks 5-9"),
        (10, 14, "Weeks 10-14"),
        (15, 99, "Weeks 15+"),
    ]

    for season in seasons:
        mask = df["season"] == season
        sub = df[mask]
        y_t = sub["home_win_actual"].values
        y_p = sub[prob_col].values
        m = compute_metrics(y_t, y_p)
        results.append({
            "segment": f"Season {season}",
            "n": len(sub),
            "log_loss": m.get("log_loss", "N/A"),
        })

    for lo, hi, label in week_buckets:
        mask = df["week"].between(lo, hi)
        if mask.sum() == 0:
            continue
        sub = df[mask]
        y_t = sub["home_win_actual"].values
        y_p = sub[prob_col].values
        m = compute_metrics(y_t, y_p)
        results.append({
            "segment": label,
            "n": len(sub),
            "log_loss": m.get("log_loss", "N/A"),
        })

    return results


# --- QB-change side breakdown ---


def _qb_side_breakdown(
    df: pd.DataFrame,
    prob_col: str,
) -> List[Dict]:
    """Metrics by which side(s) had a QB change."""
    results = []
    if df is None or len(df) == 0:
        return results

    home_changed = df.get("home_qb_changed", pd.Series(0)).fillna(0).astype(bool)
    away_changed = df.get("away_qb_changed", pd.Series(0)).fillna(0).astype(bool)

    breakdowns = [
        ("Home QB changed", home_changed & ~away_changed),
        ("Away QB changed", ~home_changed & away_changed),
        ("Both QBs changed", home_changed & away_changed),
        ("Neither QB changed", ~home_changed & ~away_changed),
    ]

    for label, mask in breakdowns:
        if mask.sum() == 0:
            continue
        sub = df[mask]
        y_t = sub["home_win_actual"].values
        y_p = sub[prob_col].values
        m = compute_metrics(y_t, y_p)
        results.append({
            "segment": label,
            "n": int(mask.sum()),
            "log_loss": m.get("log_loss", "N/A"),
        })

    return results


# --- Main experiment ---


def run_qb_gated_experience(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = DEFAULT_REPORT,
) -> str:
    """Run gated QB-experience experiment."""
    print("=== Gated QB Experience Experiment ===")

    fp = Path(ft_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")
    df_raw = pd.read_parquet(fp)

    eligible_mask = (
        df_raw.get(MODEL_ELIGIBLE_COLUMN, pd.Series(True))
        & (~df_raw.get("is_neutral", pd.Series(False)).fillna(False))
    )
    total_eligible = int(eligible_mask.sum())
    print(f"  Total eligible games: {total_eligible}")

    # Build gated variant names
    gated_map = {
        "incumbent": "",
        "qb_experience_global": "",
        "qb_experience_gated_binary": "gated_binary",
        "qb_experience_gated_team_specific": "gated_team_specific",
        "qb_experience_gated_simple_diff": "gated_simple_diff",
    }

    # --- Rolling simulation ---
    print("\n--- Rolling Simulation (All Seasons) ---")
    rolling_results = {}
    for name, cols, _, _ in MODEL_VARIANTS:
        gv = gated_map.get(name, "")
        print(f"  Running {name}...")
        rolling_results[name] = _run_rolling_simulation(
            df_raw, cols, gated_variant=gv, label=name,
        )
        m = rolling_results[name]["overall"]
        print(f"    {name}: {m.get('log_loss', 'N/A')} "
              f"({rolling_results[name]['num_games']} games)")

    # --- Fitted-once ---
    print("\n--- Fitted-Once (2021-2024 train, 2025 predict) ---")
    fitted_results = {}
    for name, cols, _, _ in MODEL_VARIANTS:
        gv = gated_map.get(name, "")
        print(f"  Running {name}...")
        fitted_results[name] = _run_fitted_once(
            df_raw, cols, gated_variant=gv, label=name,
        )
        m = fitted_results[name]["overall"]
        print(f"    {name}: {m.get('log_loss', 'N/A')} "
              f"({fitted_results[name]['num_games']} games)")

    # --- Bootstrap CI vs incumbent (rolling) ---
    print("\n--- Bootstrap CI (Rolling, Δ = challenger - incumbent) ---")
    bootstrap_ci = {}
    inc_roll_df = rolling_results["incumbent"].get("df")
    if inc_roll_df is not None and len(inc_roll_df) > 0:
        y_t = inc_roll_df["home_win_actual"].values
        p_inc = inc_roll_df["incumbent_prob"].values
        for name, _, _, _ in MODEL_VARIANTS:
            if name == "incumbent":
                continue
            df_challenger = rolling_results[name].get("df")
            if df_challenger is None or len(df_challenger) == 0:
                continue
            prob_col = f"{name}_prob"
            if prob_col not in df_challenger.columns:
                continue
            p_chal = df_challenger[prob_col].values
            mean_d, ci_l, ci_h = bootstrap_delta(y_t, p_inc, p_chal)
            bootstrap_ci[name] = {
                "mean_delta": mean_d, "ci_low": ci_l, "ci_high": ci_h,
            }
            print(f"  {name} vs incumbent: {mean_d:.4f} "
                  f"[{ci_l:.4f}, {ci_h:.4f}]")

    # --- Subset analysis ---
    print("\n--- Subset Analysis ---")
    qb_changed_subset = _qb_subset_analysis(rolling_results, qb_changed=True)
    no_qb_change_subset = _qb_subset_analysis(rolling_results, qb_changed=False)

    # --- Coefficient diagnostics ---
    print("\n--- Coefficient Diagnostics ---")
    coef_diag = _coefficient_diagnostics(fitted_results)

    # --- Incumbent holdout reference ---
    inc_holdout = fitted_results["incumbent"]["overall"].get("log_loss", 1)

    # --- Best challenger on rolling ---
    inc_roll_ll = rolling_results["incumbent"]["overall"].get("log_loss", 1)
    best_challenger = "incumbent"
    best_chal_diff = 0.0
    for name, _, _, _ in MODEL_VARIANTS:
        if name == "incumbent":
            continue
        ll = rolling_results[name]["overall"].get("log_loss", 1)
        diff = ll - inc_roll_ll
        if diff < best_chal_diff:
            best_chal_diff = diff
            best_challenger = name

    print(f"  Best challenger on rolling: {best_challenger} "
          f"({best_chal_diff:+.4f})")

    # --- Generate report ---
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    def _w(text: str = "") -> None:
        f.write(text + "\n")

    with open(rp, "w") as f:
        _w("# Gated QB Experience Experiment")
        _w()
        _w(f"*Generated by `sportslab qb-gated-experience`"
           f" ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*")
        _w()

        # Hypothesis
        _w("## Hypothesis")
        _w()
        _w("QB experience features (starts this season, starts diff) "
           "help when a QB change occurs, but add noise when applied "
           "to every game. Gating them by the QB-change binary flag "
           "may preserve the lift on QB-change games without damaging "
           "non-QB-change games.")
        _w()

        # Model variants
        _w("## Model Variants")
        _w()
        _w("| Label | Features | Description |")
        _w("|-------|----------|-------------|")
        for name, cols, _, desc in MODEL_VARIANTS:
            feat_str = ", ".join(cols) if len(cols) <= 8 else f"{len(cols)} features"
            _w(f"| {name} | {feat_str} | {desc} |")
        _w()

        # Live-safety
        _w("### Live-Safety Labels")
        _w()
        _w("All variants use `research_oracle` QB data "
           "(final actual starter from nflreadpy).")
        _w("No variant is live-safe unless pregame QB input CSV is provided.")
        _w()

        # Rolling
        _w("## Rolling Simulation (All Seasons)")
        _w()
        _w("| Model | Log Loss | Brier | Accuracy | Games | Δ vs Incumbent |")
        _w("|-------|----------|-------|----------|-------|----------------|")
        for name, _, _, _ in MODEL_VARIANTS:
            r = rolling_results[name]
            m = r["overall"]
            delta = m.get("log_loss", 0) - inc_roll_ll
            delta_str = f"{delta:+.4f}" if name != "incumbent" else "—"
            _w(f"| {name} | {m.get('log_loss', 'N/A')}"
               f" | {m.get('brier', 'N/A')}"
               f" | {m.get('accuracy', 'N/A')}"
               f" | {r['num_games']}"
               f" | {delta_str} |")
        _w()

        # Bootstrap CI
        _w("### Bootstrap CI vs Incumbent (Rolling, 1000 iterations)")
        _w()
        _w("Δ = challenger LL - incumbent LL. Negative Δ means challenger better.")
        _w()
        _w("| Challenger | Mean Δ | 95% CI Lower | 95% CI Upper |")
        _w("|------------|--------|--------------|--------------|")
        for name, _, _, _ in MODEL_VARIANTS:
            if name == "incumbent":
                continue
            ci = bootstrap_ci.get(name, {})
            _w(f"| {name} | {ci.get('mean_delta', '—')} "
               f"| {ci.get('ci_low', '—')} "
               f"| {ci.get('ci_high', '—')} |")
        _w()

        # Fitted-once
        _w("## Fitted-Once (Holdout: 2025)")
        _w()
        _w("| Model | Log Loss | Brier | Accuracy | Games | Δ vs Incumbent |")
        _w("|-------|----------|-------|----------|-------|----------------|")
        for name, _, _, _ in MODEL_VARIANTS:
            r = fitted_results[name]
            m = r["overall"]
            delta = m.get("log_loss", 0) - inc_holdout
            delta_str = f"{delta:+.4f}" if name != "incumbent" else "—"
            _w(f"| {name} | {m.get('log_loss', 'N/A')}"
               f" | {m.get('brier', 'N/A')}"
               f" | {m.get('accuracy', 'N/A')}"
               f" | {r['num_games']}"
               f" | {delta_str} |")
        _w()

        # QB-change subset
        _w("## QB-Change Subset (Rolling)")
        _w()
        _w("| Model | Log Loss | Games |")
        _w("|-------|----------|-------|")
        for name, _, _, _ in MODEL_VARIANTS:
            sub = qb_changed_subset.get(name, {})
            m = sub.get("overall", {})
            _w(f"| {name} | {m.get('log_loss', 'N/A')} "
               f"| {sub.get('num_games', 0)} |")
        _w()

        # Non-QB-change subset
        _w("## Non-QB-Change Subset (Rolling)")
        _w()
        _w("| Model | Log Loss | Games |")
        _w("|-------|----------|-------|")
        for name, _, _, _ in MODEL_VARIANTS:
            sub = no_qb_change_subset.get(name, {})
            m = sub.get("overall", {})
            _w(f"| {name} | {m.get('log_loss', 'N/A')} "
               f"| {sub.get('num_games', 0)} |")
        _w()

        # QB-side breakdown
        df_best = rolling_results[best_challenger].get("df")
        df_inc = rolling_results["incumbent"].get("df")
        if df_best is not None and len(df_best) > 0:
            _w("## QB-Change Side Breakdown (Best Challenger)")
            _w()
            bprob = f"{best_challenger}_prob"
            side_b = _qb_side_breakdown(df_best, bprob)
            _w("| Segment | N | Log Loss |")
            _w("|---------|---|----------|")
            for s in side_b:
                _w(f"| {s['segment']} | {s['n']} | {s['log_loss']} |")
            _w()

        # Coefficient diagnostics
        _w("## Coefficient Diagnostics")
        _w()
        for label in [n for n, _, _, _ in MODEL_VARIANTS if n in coef_diag]:
            cd = coef_diag[label]
            _w(f"### {label}")
            _w()
            if cd["n_rolling_models"] > 0:
                _w(f"Based on {cd['n_rolling_models']} weekly rolling models.")
            else:
                _w("Based on fitted-once model.")
            _w()
            _w("| Feature | Mean Coef | Std | Pos Fraction | Sign Stable |")
            _w("|---------|-----------|-----|--------------|-------------|")
            for feat in cd["features"]:
                stable = "✓" if feat["sign_stable"] else "✗"
                _w(f"| {feat['feature']} | {feat['coef']:+.6f} "
                   f"| {feat['std']:.6f} | {feat['pos_frac']:.2f} "
                   f"| {stable} |")
            _w()

        # Season/week breakdown
        if df_best is not None and len(df_best) > 0:
            _w("## Season/Week Breakdown (Best Challenger)")
            _w()
            bprob = f"{best_challenger}_prob"
            swb = _season_week_breakdown(df_best, bprob)
            _w("| Segment | N | Log Loss |")
            _w("|---------|---|----------|")
            for s in swb:
                _w(f"| {s['segment']} | {s['n']} | {s['log_loss']} |")
            _w()

        # Calibration buckets
        _w("## Calibration Buckets")
        _w()
        _w(f"Incumbent vs best challenger (**{best_challenger}**) on rolling all-season.")
        _w()
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
            buckets = calibration_buckets(
                df["home_win_actual"].values,
                df[prob_col].values,
            )
            _w(f"### {display_name} ({key})")
            _w()
            _w("| Bucket | N | Mean Pred | Mean Actual | Cal Error |")
            _w("|--------|---|-----------|-------------|-----------|")
            for b in buckets:
                _w(f"| {b['bucket']} | {b['n']} | {b['mean_pred']:.4f}"
                   f" | {b['mean_actual']:.4f} | {b['cal_error']:.4f} |")
            _w()

        # Confidence buckets
        _w("## Confidence Buckets")
        _w()
        _w("Confidence = 2 * |prob - 0.5|. Higher = more confident.")
        _w()
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
            buckets = confidence_buckets(
                df["home_win_actual"].values,
                df[prob_col].values,
            )
            _w(f"### {display_name} ({key})")
            _w()
            _w("| Confidence | N | Log Loss | Brier | Accuracy |")
            _w("|------------|---|----------|-------|----------|")
            for b in buckets:
                _w(f"| {b['bucket']} | {b['n']} | {b['log_loss']:.4f}"
                   f" | {b['brier']:.4f} | {b['accuracy']:.4f} |")
            _w()

        # Worst 20 predictions
        _w("## Worst 20 Predictions (Incumbent, Rolling)")
        _w()
        if df_inc is not None and len(df_inc) > 0:
            worst = worst_predictions(
                df_inc["home_win_actual"].values,
                df_inc["incumbent_prob"].values,
                df_inc.get("game_id", pd.Series([""] * len(df_inc))).values,
                df_inc.get("home_team", pd.Series([""] * len(df_inc))).values,
            )
            _w("| # | Game ID | Home | Actual | Pred | Log Loss |")
            _w("|---|---------|------|--------|------|----------|")
            for i, w in enumerate(worst, 1):
                _w(f"| {i} | {w['game_id']} | {w['team']}"
                   f" | {w['actual']} | {w['pred']:.4f} | {w['log_loss_contrib']:.4f} |")
            _w()

        # Recommendation
        _w("## Recommendation")
        _w()
        _w("**No model is promoted in this research pass.**")
        _w()

        beats_roll = any(
            rolling_results[n]["overall"].get("log_loss", 1) < inc_roll_ll
            for n, _, _, _ in MODEL_VARIANTS if n != "incumbent"
        )
        beats_hold = any(
            fitted_results[n]["overall"].get("log_loss", 1) < inc_holdout
            for n, _, _, _ in MODEL_VARIANTS if n != "incumbent"
        )
        beats_both = any(
            rolling_results[n]["overall"].get("log_loss", 1) < inc_roll_ll
            and fitted_results[n]["overall"].get("log_loss", 1) < inc_holdout
            for n, _, _, _ in MODEL_VARIANTS if n != "incumbent"
        )

        best_roll_name = min(
            [n for n, _, _, _ in MODEL_VARIANTS if n != "incumbent"],
            key=lambda n: rolling_results[n]["overall"].get("log_loss", 1),
        )
        best_hold_name = min(
            [n for n, _, _, _ in MODEL_VARIANTS if n != "incumbent"],
            key=lambda n: fitted_results[n]["overall"].get("log_loss", 1),
        )
        best_roll_ll = rolling_results[best_roll_name]["overall"].get("log_loss", "N/A")
        best_hold_ll = fitted_results[best_hold_name]["overall"].get("log_loss", "N/A")
        best_ci = bootstrap_ci.get(best_roll_name, {})
        best_mean_d = best_ci.get("mean_delta", "N/A")
        best_ci_l = best_ci.get("ci_low", "N/A")
        best_ci_h = best_ci.get("ci_high", "N/A")

        _w(f"Best challenger on rolling: **{best_roll_name}** "
           f"({best_roll_ll}, Δ={best_mean_d} "
           f"[{best_ci_l}, {best_ci_h}]).")
        _w(f"Best challenger on holdout: **{best_hold_name}** ({best_hold_ll}).")
        _w()

        _w("1. **Does gating QB experience improve rolling all-season "
           f"performance?** {'Yes' if beats_roll else 'No'}.")
        _w("2. **Does gating QB experience improve fitted-once 2025 "
           f"performance?** {'Yes' if beats_hold else 'No'}.")
        _w("3. **Does any variant beat incumbent on BOTH rolling and "
           f"holdout?** {'Yes' if beats_both else 'No'}.")
        _w("4. **Does gating preserve the QB-change subset lift?** "
           "See QB-change subset table above.")
        _w("5. **Does gating avoid damaging non-QB-change games?** "
           "See non-QB-change subset table above.")
        _w("6. **Are coefficients stable enough to trust?** "
           "See coefficient diagnostics above.")
        _w("7. **Should live predictions continue using no-QB fallback "
           "unless audited QB inputs are available?** Yes. "
           "All variants except incumbent use research_oracle QB data.")
        _w()
        _w("---")
        _w(f"*Incumbent: {INCUMBENT_VERSION}, "
           f"{INCUMBENT_HOLDOUT_LL} holdout LL. "
           f"No model promoted in this pass.*")

    # Console summary
    print(f"\nGated QB Experience report: {rp}")
    print(f"  Incumbent rolling:  {inc_roll_ll:.4f} "
          f"({rolling_results['incumbent']['num_games']} games)")
    print(f"  Incumbent holdout:  {inc_holdout:.4f} "
          f"({fitted_results['incumbent']['num_games']} games)")
    print(f"  Best challenger rolling: {best_roll_name} ({best_roll_ll})")
    print(f"  Bootstrap Δ vs incumbent: {best_mean_d} [{best_ci_l}, {best_ci_h}]")

    return str(rp)
