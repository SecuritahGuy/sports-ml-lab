"""Forward feature selection experiment.

Tests each candidate feature individually (on top of Elo+Platt) via
rolling-origin, then tests the best combo if any feature improves.
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
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import (
    compute_situational_features,
)

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2

# Candidate feature groups (each is a list of columns to add together)
CANDIDATE_FEATURES = {
    "rolling_mov_3": ["home_rolling_mov_3", "away_rolling_mov_3"],
    "rolling_mov_5": ["home_rolling_mov_5", "away_rolling_mov_5"],
    "rolling_pts_for": ["home_rolling_pts_for", "away_rolling_pts_for"],
    "rolling_pts_against": ["home_rolling_pts_against", "away_rolling_pts_against"],
    "win_streak": ["home_win_streak", "away_win_streak"],
    "ytd_win_pct": ["home_ytd_win_pct", "away_ytd_win_pct"],
    "turf_flag": ["turf_flag"],
    "high_altitude": ["high_altitude_flag"],
    "prime_time": ["prime_time_flag"],
    "rest_diff_squared": ["rest_diff_squared"],
}


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


def _logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def run_feature_selection_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/feature_selection.md",
) -> str:
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # Compute Elo
    print("\n=== Computing Elo features ===")
    team_overrides = build_team_regression_overrides(
        df_raw,
        preseason_regression=BEST_REG,
        qb_change_bonus=BEST_QB_BONUS,
    )
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=team_overrides,
        decay_half_life=BEST_DECAY,
    )

    # Compute QB features
    print("=== Computing QB features ===")
    df_qb = compute_qb_features(df_elo)

    # Compute situational features
    print("=== Computing situational features ===")
    df_all = compute_situational_features(df_qb)

    # Filter
    df_all = _filter_df(df_all)
    print(f"  After filter: {len(df_all)} rows\n")

    elo_prob = df_all["elo_prob"].values
    y = df_all[TARGET_COLUMN].astype(float).values

    # ── Rolling-origin evaluation ──
    print("=== Individual Feature Evaluation ===")

    results = {}

    # 1. Platt baseline
    platt_fold_lls = []
    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values
        platt = _fit_platt(elo_prob[is_train], y[is_train].astype(int))
        proba = platt.predict_proba(elo_prob[is_val].reshape(-1, 1))[:, 1]
        platt_fold_lls.append(compute_classification_metrics(y[is_val], proba)["log_loss"])
    avg_platt = float(np.mean(platt_fold_lls))
    results["platt"] = {"val_ll": avg_platt, "fold_lls": platt_fold_lls}
    print(f"  Platt (incumbent): avg val LL = {avg_platt:.4f}")

    # 2. Each candidate feature individually
    candidate_results = {}
    for name, cols in CANDIDATE_FEATURES.items():
        available = [c for c in cols if c in df_all.columns]
        if not available:
            continue
        fold_lls = []
        for train_seasons, val_season in ROLLING_FOLDS:
            is_train = df_all["season"].isin(train_seasons).values
            is_val = (df_all["season"] == val_season).values
            train_elo = elo_prob[is_train]
            train_y_ = y[is_train].astype(int)
            val_elo = elo_prob[is_val]
            val_y_ = y[is_val]
            train_feat = df_all.loc[is_train, available].values
            val_feat = df_all.loc[is_val, available].values
            x_train = np.column_stack([train_elo, train_feat])
            x_val = np.column_stack([val_elo, val_feat])
            pipe = _logistic_model()
            pipe.fit(x_train, train_y_)
            proba = pipe.predict_proba(x_val)[:, 1]
            fold_lls.append(compute_classification_metrics(val_y_, proba)["log_loss"])
        avg = float(np.mean(fold_lls))
        candidate_results[name] = {"val_ll": avg, "fold_lls": fold_lls}
        diff = avg - avg_platt
        marker = "✓" if diff < 0 else "✗"
        print(f"  {marker} {name}: avg val LL = {avg:.4f} (Δ={diff:+.4f})")

    results["candidates"] = candidate_results

    # 3. Best individual feature combo
    best_name = min(candidate_results, key=lambda k: candidate_results[k]["val_ll"])
    best_ll = candidate_results[best_name]["val_ll"]
    print(f"\n  Best individual: {best_name} ({best_ll:.4f})")

    # 4. All situational features together
    all_sit_cols = [c for cols in CANDIDATE_FEATURES.values() for c in cols if c in df_all.columns]
    all_fold_lls = []
    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values
        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]
        train_feat = df_all.loc[is_train, all_sit_cols].values
        val_feat = df_all.loc[is_val, all_sit_cols].values
        x_train = np.column_stack([train_elo, train_feat])
        x_val = np.column_stack([val_elo, val_feat])
        pipe = _logistic_model()
        pipe.fit(x_train, train_y_)
        proba = pipe.predict_proba(x_val)[:, 1]
        all_fold_lls.append(compute_classification_metrics(val_y_, proba)["log_loss"])
    avg_all = float(np.mean(all_fold_lls))
    results["all_situational"] = {"val_ll": avg_all, "fold_lls": all_fold_lls}
    print(f"  All situational features: avg val LL = {avg_all:.4f} (Δ={avg_all - avg_platt:+.4f})")

    # 5. QB features (previously rejected, but testing individually)
    print("\n=== QB Feature Subset Evaluation ===")
    qb_candidates = {
        "qb_changed": ["home_qb_changed", "away_qb_changed"],
        "qb_win_pct": ["home_qb_win_pct_pre", "away_qb_win_pct_pre"],
        "games_since_change": ["home_games_since_qb_change", "away_games_since_qb_change"],
        "new_qb": ["home_new_qb_flag", "away_new_qb_flag"],
    }
    qb_results = {}
    for name, cols in qb_candidates.items():
        available = [c for c in cols if c in df_all.columns]
        fold_lls = []
        for train_seasons, val_season in ROLLING_FOLDS:
            is_train = df_all["season"].isin(train_seasons).values
            is_val = (df_all["season"] == val_season).values
            train_elo = elo_prob[is_train]
            train_y_ = y[is_train].astype(int)
            val_elo = elo_prob[is_val]
            val_y_ = y[is_val]
            train_feat = df_all.loc[is_train, available].values
            val_feat = df_all.loc[is_val, available].values
            x_train = np.column_stack([train_elo, train_feat])
            x_val = np.column_stack([val_elo, val_feat])
            pipe = _logistic_model()
            pipe.fit(x_train, train_y_)
            proba = pipe.predict_proba(x_val)[:, 1]
            fold_lls.append(compute_classification_metrics(val_y_, proba)["log_loss"])
        avg = float(np.mean(fold_lls))
        qb_results[name] = {"val_ll": avg, "fold_lls": fold_lls}
        diff = avg - avg_platt
        marker = "✓" if diff < 0 else "✗"
        print(f"  {marker} {name}: avg val LL = {avg:.4f} (Δ={diff:+.4f})")
    results["qb_candidates"] = qb_results

    # ── Best overall combo (forward selection) ──
    print("\n=== Forward Selection ===")
    selected = []
    remaining = dict(CANDIDATE_FEATURES)
    # Also add QB features
    remaining.update(qb_candidates)
    current_ll = avg_platt
    improvement = True
    round_num = 1
    while improvement and remaining:
        improvement = False
        best_round_name = None
        best_round_ll = current_ll
        for name, cols in remaining.items():
            available = [c for c in cols if c in df_all.columns]
            all_selected_cols = [
                c for n in selected for c in CANDIDATE_FEATURES.get(n, []) if c in df_all.columns
            ]
            if name in qb_candidates:
                all_selected_cols += [
                    c
                    for n in selected
                    if n in qb_candidates
                    for c in qb_candidates[n]
                    if c in df_all.columns
                ]
            all_cols = list(set(all_selected_cols + available))
            fold_lls = []
            for train_seasons, val_season in ROLLING_FOLDS:
                is_train = df_all["season"].isin(train_seasons).values
                is_val = (df_all["season"] == val_season).values
                train_elo = elo_prob[is_train]
                train_y_ = y[is_train].astype(int)
                val_elo = elo_prob[is_val]
                val_y_ = y[is_val]
                train_feat = df_all.loc[is_train, all_cols].values
                val_feat = df_all.loc[is_val, all_cols].values
                x_train = np.column_stack([train_elo, train_feat])
                x_val = np.column_stack([val_elo, val_feat])
                pipe = _logistic_model()
                pipe.fit(x_train, train_y_)
                proba = pipe.predict_proba(x_val)[:, 1]
                fold_lls.append(compute_classification_metrics(val_y_, proba)["log_loss"])
            avg = float(np.mean(fold_lls))
            if avg < best_round_ll:
                best_round_ll = avg
                best_round_name = name
                improvement = True
        if improvement and best_round_name:
            selected.append(best_round_name)
            current_ll = best_round_ll
            del remaining[best_round_name]
            print(
                f"  Round {round_num}: +{best_round_name}"
                f" (LL={current_ll:.4f}, Δ={current_ll - avg_platt:+.4f})"
            )
            round_num += 1

    results["forward_selection"] = {"selected": selected, "final_ll": current_ll}
    print(f"  Forward selection complete. Selected: {selected} (LL={current_ll:.4f})")

    # ── One-shot 2025 holdout for best models ──
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    is_train_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    train_elo_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)

    # Platt incumbent
    platt_full = _fit_platt(train_elo_full, train_y_full)
    platt_hold_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, platt_hold_proba)
    print(f"  Platt (incumbent): holdout LL = {hold_platt_m['log_loss']:.4f}")

    # Best individual
    if best_name:
        best_cols = [c for c in CANDIDATE_FEATURES[best_name] if c in df_all.columns]
        train_best = np.column_stack([train_elo_full, df_all.loc[is_train_full, best_cols].values])
        hold_best = np.column_stack([hold_elo, df_all.loc[is_hold, best_cols].values])
        best_pipe = _logistic_model()
        best_pipe.fit(train_best, train_y_full)
        best_hold_proba = best_pipe.predict_proba(hold_best)[:, 1]
        hold_best_m = compute_classification_metrics(hold_y, best_hold_proba)
        print(f"  Best ({best_name}): holdout LL = {hold_best_m['log_loss']:.4f}")

    # Forward selection final combo
    if selected:
        fwd_cols = []
        for n in selected:
            if n in CANDIDATE_FEATURES:
                fwd_cols += [c for c in CANDIDATE_FEATURES[n] if c in df_all.columns]
            elif n in qb_candidates:
                fwd_cols += [c for c in qb_candidates[n] if c in df_all.columns]
        fwd_cols = list(set(fwd_cols))
        train_fwd = np.column_stack([train_elo_full, df_all.loc[is_train_full, fwd_cols].values])
        hold_fwd = np.column_stack([hold_elo, df_all.loc[is_hold, fwd_cols].values])
        fwd_pipe = _logistic_model()
        fwd_pipe.fit(train_fwd, train_y_full)
        fwd_hold_proba = fwd_pipe.predict_proba(hold_fwd)[:, 1]
        hold_fwd_m = compute_classification_metrics(hold_y, fwd_hold_proba)
        print(f"  Forward selection ({selected}): holdout LL = {hold_fwd_m['log_loss']:.4f}")
    else:
        hold_fwd_m = hold_platt_m

    # ── Report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Feature Selection Experiment\n\n")
        f.write("*Systematic forward feature selection on top of Elo+Platt.*\n\n")

        f.write("## Method\n\n")
        f.write(
            "Rolling-origin 3-fold validation. Each feature group tested"
            " individually on top of Elo probability + logistic regression."
            " Forward selection builds up the best combination.\n\n"
        )
        f.write("### Elo Params\n\n")
        f.write(
            f"K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG},"
            f" decay={BEST_DECAY}, qb_bonus={BEST_QB_BONUS}\n\n"
        )

        f.write("### Candidate Features\n\n")
        f.write("| Group | Columns | Description |\n")
        f.write("|-------|---------|-------------|\n")
        f.write("| `rolling_mov_3` | home/away | Avg MOV last 3 games |\n")
        f.write("| `rolling_mov_5` | home/away | Avg MOV last 5 games |\n")
        f.write("| `rolling_pts_for` | home/away | Season avg points scored |\n")
        f.write("| `rolling_pts_against` | home/away | Season avg points allowed |\n")
        f.write("| `win_streak` | home/away | Current win/loss streak |\n")
        f.write("| `ytd_win_pct` | home/away | Season win % YTD |\n")
        f.write("| `turf_flag` | binary | Artificial turf surface |\n")
        f.write("| `high_altitude` | binary | DEN/MEX altitude stadiums |\n")
        f.write("| `prime_time` | binary | Mon/Thu/Sun night or 8PM+ |\n")
        f.write("| `rest_diff_squared` | numeric | Nonlinear rest edge |\n\n")

        def _ll_line(name, fold_lls, avg_ll, delta=None):
            d = f" {delta:+.4f}" if delta is not None else " —"
            return (
                f"| {name} | {avg_ll:.4f} |{d}"
                f" | {fold_lls[0]:.4f} | {fold_lls[1]:.4f} | {fold_lls[2]:.4f} |\n"
            )

        f.write("## Individual Feature Results\n\n")
        f.write("| Feature | Avg Val LL | Δ vs Platt | Fold1 | Fold2 | Fold3 |\n")
        f.write("|--------|-----------|-----------|-------|-------|-------|\n")
        f.write(_ll_line("Platt (incumbent)", platt_fold_lls, avg_platt))
        for name, res in sorted(candidate_results.items(), key=lambda kv: kv[1]["val_ll"]):
            d = res["val_ll"] - avg_platt
            f.write(_ll_line(name, res["fold_lls"], res["val_ll"], delta=d))
        d = avg_all - avg_platt
        f.write(_ll_line("All situational", all_fold_lls, avg_all, delta=d))
        f.write("\n")

        f.write("## QB Subset Results\n\n")
        f.write("| Feature | Avg Val LL | Δ vs Platt |\n")
        f.write("|--------|-----------|-----------|\n")
        for name, res in sorted(qb_results.items(), key=lambda kv: kv[1]["val_ll"]):
            d = res["val_ll"] - avg_platt
            f.write(f"| {name} | {res['val_ll']:.4f} | {d:+.4f} |\n")
        f.write("\n")

        f.write("## Forward Selection\n\n")
        f.write(f"Starting from Platt baseline ({avg_platt:.4f}).\n\n")
        f.write("| Round | Added | Val LL | Δ |\n")
        f.write("|-------|-------|--------|---|\n")
        for i, sel in enumerate(selected):
            f.write(f"| {i + 1} | {sel} | — | — |\n")
        sel_display = ", ".join(selected) if selected else "(none)"
        f.write(f"| Final | {sel_display} | {current_ll:.4f} | {current_ll - avg_platt:+.4f} |\n\n")

        f.write("## 2025 Holdout\n\n")
        f.write("| Model | Holdout LL |\n")
        f.write("|-------|-----------|\n")
        f.write(f"| Platt (incumbent) | {hold_platt_m['log_loss']:.4f} |\n")
        if best_name:
            f.write(f"| Best ({best_name}) | {hold_best_m['log_loss']:.4f} |\n")
        if selected:
            f.write(f"| Forward selection | {hold_fwd_m['log_loss']:.4f} |\n")
        f.write("\n")

        f.write("## Decision\n\n")
        beat_val = any(candidate_results[n]["val_ll"] < avg_platt for n in candidate_results)
        if beat_val:
            best_val_name = min(candidate_results, key=lambda k: candidate_results[k]["val_ll"])
            best_val_ll = candidate_results[best_val_name]["val_ll"]
            f.write(
                f"**{best_val_name} beats incumbent on validation**"
                f" ({best_val_ll:.4f} vs {avg_platt:.4f}).\n\n"
            )
        else:
            f.write("**No feature beats the incumbent on validation.**\n\n")

        if selected:
            f.write(f"Forward selection converged on: {', '.join(selected)}.\n")
        else:
            f.write("Forward selection found no improving features.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
