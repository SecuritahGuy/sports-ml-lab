"""Backtest each StatSpace metric individually as a standalone game predictor.

Approach: for each season S, use each team's metric value from season S-1
to predict games in season S. This tests year-over-year predictive signal.
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
from sportslab.evaluation.team_profiles import build_team_profiles
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

_SEED = 42

# Metrics to backtest in order
_METRICS = [
    ("elo_prob", "Raw Elo", "Pre-game Elo probability (full-history)"),
    ("fraud_detector_rating", "FDR", "StatSpace FDR"),
    ("doba_score", "DOBA", "StatSpace DOBA"),
    ("chaos_rate", "Chaos Rate", "StatSpace Chaos Rate"),
    ("aggression_score", "Coward Tax", "StatSpace Coward Tax"),
    ("qb_lift_index", "QB Lift", "StatSpace QB Lift"),
]

_BACKTEST_SEASONS = [2022, 2023, 2024]
_HOLDOUT_SEASON = 2025


def _fit_platt(x_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=_SEED)),
    ])
    pipe.fit(x_train, y_train.astype(int))
    return pipe


def _load_feature_table(ft_path):
    df_raw = pd.read_parquet(ft_path)
    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=0.1, qb_change_bonus=0.2,
    )
    df = compute_elo_features(
        df_raw, k_factor=36, home_advantage=40,
        preseason_regression=0.1, team_regression_overrides=overrides,
        decay_half_life=32,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    return df[mask].copy().reset_index(drop=True)


def _elo_from_prior_season(ft):
    """Get end-of-prior-season Elo ratings for each team."""
    elo_map = {}
    for _, g in ft.iterrows():
        season = int(g["season"])
        for team, pref in [(g["home_team"], "home"), (g["away_team"], "away")]:
            elo_col = f"{pref}_elo" if f"{pref}_elo" in ft.columns else None
            if elo_col is None:
                continue
            elo_val = g[elo_col]
            key = (team, season)
            elo_map[key] = elo_val
    return elo_map


def _avg_value(lagged_map, team, *seasons):
    """Average metric value across multiple seasons, ignoring NaN."""
    vals = [lagged_map.get((team, s)) for s in seasons]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def build_lagged_predictions(profiles: pd.DataFrame, ft: pd.DataFrame) -> dict:
    """For each metric, build season-lagged predictions for backtest seasons."""

    def _single_pass(lag_window=1, metric_subset=None):
        """Run pass with a given lag window. window=1: use S-1. window=2: avg S-1,S-2. etc."""
        # Build lagged metric map: (team, season) -> value
        lagged = {}
        metric_list = [m for m in _METRICS if metric_subset is None or m[0] in metric_subset]
        for metric_col, _, _ in metric_list:
            lagged[metric_col] = {}
        # Ensure elo_prob is always allocated (needed for prior-elo below)
        if "elo_prob" not in lagged:
            lagged["elo_prob"] = {}

        for _, r in profiles.iterrows():
            team = r["team"]
            season = int(r["season"])
            for metric_col, _, _ in metric_list:
                if metric_col not in r or pd.isna(r[metric_col]):
                    continue
                lagged[metric_col][(team, season)] = r[metric_col]

        # Add Elo from prior season
        prior_elo = {}
        for _, g in ft.sort_values(["season", "week"]).iterrows():
            season = int(g["season"])
            for team, pref in [(g["home_team"], "home"), (g["away_team"], "away")]:
                key = (team, season)
                prior_elo[key] = g[f"{pref}_elo_pre"]
        for (team, season), elo_val in prior_elo.items():
            lagged["elo_prob"][(team, season)] = elo_val

        seasons_lag = list(range(1, lag_window + 1))

        # Build game-level predictions for each metric
        local_results = {}
        for metric_col, name, desc in metric_list:
            diffs, hold_diffs = [], []
            y_train, y_hold = [], []
            hold_games = []

            for _, g in ft.iterrows():
                season = int(g["season"])
                home = g["home_team"]
                away = g["away_team"]
                y_val = g[TARGET_COLUMN]

                home_val = _avg_value(lagged[metric_col], home, *(season - s for s in seasons_lag))
                away_val = _avg_value(lagged[metric_col], away, *(season - s for s in seasons_lag))
                if pd.isna(home_val) or pd.isna(away_val):
                    continue

                diff = float(home_val - away_val)

                if season == _HOLDOUT_SEASON:
                    hold_diffs.append(diff)
                    y_hold.append(y_val)
                    hold_games.append(g["game_id"])
                elif season in _BACKTEST_SEASONS:
                    diffs.append(diff)
                    y_train.append(y_val)

            if len(diffs) < 10:
                continue

            diffs_arr = np.array(diffs).reshape(-1, 1)
            hold_diffs_arr = np.array(hold_diffs).reshape(-1, 1)
            y_train_arr = np.array(y_train)
            y_hold_arr = np.array(y_hold)

            # Raw metric as probability
            if metric_col == "elo_prob":
                raw_probs = 1.0 / (1.0 + 10.0 ** (-hold_diffs_arr.flatten() / 400.0))
            else:
                raw_probs = 1.0 / (1.0 + np.exp(-np.clip(hold_diffs_arr.flatten(), -10, 10)))
            raw_metrics = compute_classification_metrics(y_hold_arr, raw_probs)

            # Platt-calibrated version
            platt = _fit_platt(diffs_arr, y_train_arr)
            platt_probs = platt.predict_proba(hold_diffs_arr)[:, 1]
            platt_metrics = compute_classification_metrics(y_hold_arr, platt_probs)

            local_results[metric_col] = {
                "name": name + (f" ({lag_window}y avg)" if lag_window > 1 else ""),
                "description": desc,
                "n_train": len(diffs),
                "n_hold": len(hold_diffs),
                "raw_ll": raw_metrics["log_loss"],
                "raw_brier": raw_metrics["brier_score"],
                "raw_auc": raw_metrics["roc_auc"],
                "raw_acc": raw_metrics["accuracy"],
                "platt_ll": platt_metrics["log_loss"],
                "platt_brier": platt_metrics["brier_score"],
                "platt_auc": platt_metrics["roc_auc"],
                "platt_acc": platt_metrics["accuracy"],
            }
        return local_results

    # Run single-season (baseline) for all metrics
    results = _single_pass(lag_window=1)

    # Run multi-season averages for Chaos Rate specifically
    for window in [2, 3]:
        chaos_results = _single_pass(lag_window=window, metric_subset={"chaos_rate"})
        if "chaos_rate" in chaos_results:
            cr = chaos_results["chaos_rate"]
            cr_key = f"chaos_rate_{window}y"
            cr["name"] = f"Chaos Rate ({window}y avg)"
            results[cr_key] = cr

    return results


def run_statspace_backtest(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    profile_path: str = "reports/team_profiles.csv",
    report_path: str = "reports/experiments/statspace_backtest.md",
) -> str:
    print("Loading feature table...")
    ft = _load_feature_table(ft_path)
    print(f"  {len(ft)} eligible games")

    print("Loading team profiles...")
    pp = Path(profile_path)
    if not pp.exists():
        print("  Profiles not found, building...")
        build_team_profiles(ft_path=ft_path)
    profiles = pd.read_csv(profile_path)
    print(f"  {len(profiles)} team-seasons")

    print("\nRunning backtests...")
    header = f"{'Metric':<20s} {'Raw LL':>8s} {'Platt LL':>9s} {'Raw Brier':>9s} "
    header += f"{'Platt Brier':>10s} {'Raw AUC':>7s} {'Platt AUC':>9s}"
    print(header)
    print("-" * 80)

    results = build_lagged_predictions(profiles, ft)

    display_order = [m[0] for m in _METRICS] + ["chaos_rate_2y", "chaos_rate_3y"]
    for metric_key in display_order:
        if metric_key not in results:
            if metric_key in [m[0] for m in _METRICS]:
                sep = f"{'---':20s} {'---':>8s} {'---':>9s} {'---':>9s} "
                sep += f"{'---':>10s} {'---':>7s} {'---':>9s}"
                print(sep)
            continue
        r = results[metric_key]
        print(f"{r['name']:<20s} {r['raw_ll']:>8.4f} {r['platt_ll']:>9.4f} "
              f"{r['raw_brier']:>9.4f} {r['platt_brier']:>10.4f} "
              f"{r['raw_auc']:>7.3f} {r['platt_auc']:>9.3f}")

    # Combined: logistic regression on all metrics together
    print("\n  Building combined model...")
    combined_results = _run_combined(profiles, ft)
    if combined_results:
        r = combined_results
        print(f"{'Combined (all)':<20s} {r['raw_ll']:>8.4f} {r['platt_ll']:>9.4f} "
              f"{r['raw_brier']:>9.4f} {r['platt_brier']:>10.4f} "
              f"{r['raw_auc']:>7.3f} {r['platt_auc']:>9.3f}")
        results["combined"] = combined_results

    # Report
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        f.write("# StatSpace Metric Backtest\n\n")
        f.write(
            "Each metric is tested as a standalone predictor using prior-season "
            "team values (lagged by 1 year). Raw = sigmoid(home - away diff) as "
            "probability. Platt = logistic regression fit on 2022-2024 diffs."
            "\n\n"
        )
        f.write("| Metric | Raw LL | Platt LL | Raw Brier | Platt Brier | Raw AUC | Platt AUC |\n")
        f.write("|--------|--------|----------|-----------|-------------|---------|-----------|\n")
        for metric_key in display_order:
            if metric_key not in results:
                continue
            r = results[metric_key]
            f.write(f"| {r['name']} | {r['raw_ll']:.4f} | {r['platt_ll']:.4f} "
                    f"| {r['raw_brier']:.4f} | {r['platt_brier']:.4f} "
                    f"| {r['raw_auc']:.3f} | {r['platt_auc']:.3f} |\n")
        if "combined" in results:
            r = results["combined"]
            f.write(f"| Combined (all) | {r['raw_ll']:.4f} | {r['platt_ll']:.4f} "
                    f"| {r['raw_brier']:.4f} | {r['platt_brier']:.4f} "
                    f"| {r['raw_auc']:.3f} | {r['platt_auc']:.3f} |\n")

        f.write("\n## Rankings (by Platt-calibrated log loss)\n\n")
        ranked = sorted(
            [(k, v) for k, v in results.items()],
            key=lambda x: x[1]["platt_ll"],
        )
        for i, (key, r) in enumerate(ranked, 1):
            f.write(f"{i}. {r['name']}: {r['platt_ll']:.4f} (AUC={r['platt_auc']:.3f})\n")

        f.write("\n---\nReport: statspace_backtest.py\n")

    print(f"\nReport: {rp}")
    return str(rp)


def _run_combined(profiles, ft):
    """Train on all metrics together."""
    metric_cols = [m[0] for m in _METRICS if m[0] != "elo_prob"]
    lagged = {}
    for mc in metric_cols:
        lagged[mc] = {}
    for _, r in profiles.iterrows():
        for mc in metric_cols:
            if mc in r and not pd.isna(r[mc]):
                lagged[mc][(r["team"], int(r["season"]))] = r[mc]

    x_train, x_hold = [], []
    y_train, y_hold = [], []

    for _, g in ft.iterrows():
        season = int(g["season"])
        home, away = g["home_team"], g["away_team"]
        row = []
        missing = False
        for mc in metric_cols:
            hv = lagged[mc].get((home, season - 1), np.nan)
            av = lagged[mc].get((away, season - 1), np.nan)
            if pd.isna(hv) or pd.isna(av):
                missing = True
                break
            row.append(hv - av)
        if missing:
            continue
        if season == _HOLDOUT_SEASON:
            x_hold.append(row)
            y_hold.append(g[TARGET_COLUMN])
        elif season in _BACKTEST_SEASONS:
            x_train.append(row)
            y_train.append(g[TARGET_COLUMN])

    if len(x_train) < 10:
        return None

    x_train = np.array(x_train)
    x_hold = np.array(x_hold)
    y_train = np.array(y_train)
    y_hold = np.array(y_hold)

    raw_probs = 1.0 / (1.0 + np.exp(-np.clip(x_hold.mean(axis=1), -10, 10)))
    raw_metrics = compute_classification_metrics(y_hold, raw_probs)

    platt = _fit_platt(x_train, y_train)
    platt_probs = platt.predict_proba(x_hold)[:, 1]
    platt_metrics = compute_classification_metrics(y_hold, platt_probs)

    return {
        "name": "Combined (all)",
        "n_train": len(x_train),
        "n_hold": len(x_hold),
        "raw_ll": raw_metrics["log_loss"],
        "raw_brier": raw_metrics["brier_score"],
        "raw_auc": raw_metrics["roc_auc"],
        "raw_acc": raw_metrics["accuracy"],
        "platt_ll": platt_metrics["log_loss"],
        "platt_brier": platt_metrics["brier_score"],
        "platt_auc": platt_metrics["roc_auc"],
        "platt_acc": platt_metrics["accuracy"],
    }


if __name__ == "__main__":
    run_statspace_backtest()
