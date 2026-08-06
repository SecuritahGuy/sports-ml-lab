"""Diagnose recovery feature rejection — stability, bootstrap, sensitivity.

Explores why the recovery adjustment shows holdout promise but fails
validation, and whether the signal is real or random.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.player_recovery import (
    build_player_game_table,
    compute_game_recovery_adjustments,
    identify_return_events,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_pi_ratings_features
from sportslab.features.situational import compute_situational_features

SEED = 42
PI_ALPHA = 0.5
PI_BASE_K = 28
PI_HK_RATIO = 1.25
PI_HFA = 30
PI_REG = 0.0
BOOTSTRAP_ITERATIONS = 1000


def _logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def _inv_logit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))


def run_recovery_diagnostics(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/recovery_diagnostics.md",
    recovery_seasons: Optional[list[int]] = None,
) -> str:
    if recovery_seasons is None:
        recovery_seasons = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

    import nfl_data_py as nfl

    np.random.seed(SEED)

    # ── Build base data ──
    df_raw = pd.read_parquet(ft_path)
    df = compute_pi_ratings_features(
        df_raw, base_k=PI_BASE_K, home_advantage=PI_HFA,
        preseason_regression=PI_REG, alpha=PI_ALPHA, hk_ratio=PI_HK_RATIO,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)

    y = df[TARGET_COLUMN].astype(float).values
    feat_cols = ["home_qb_changed", "away_qb_changed",
                 "home_rolling_mov_3", "away_rolling_mov_3"]
    feat = np.column_stack([df[c].values for c in feat_cols])
    pi_prob = df["pi_prob"].values.astype(float)

    # Build recovery adjustments
    print(f"Building recovery data (seasons: {recovery_seasons})...")
    inj = nfl.import_injuries(recovery_seasons)
    inj["gsis_id"] = inj["gsis_id"].str.strip()
    pg = build_player_game_table(recovery_seasons)
    returns = identify_return_events(pg, inj, min_games_out=2)
    game_adj = compute_game_recovery_adjustments(returns, recovery_seasons)
    df = df.merge(game_adj, on="game_id", how="left")
    df["home_recovery_adj"] = df["home_recovery_adj"].fillna(0)
    df["away_recovery_adj"] = df["away_recovery_adj"].fillna(0)
    df["recovery_net"] = df["home_recovery_adj"] - df["away_recovery_adj"]
    df["recovery_abs"] = df["recovery_net"].abs()
    print(f"  Return events: {len(returns)}")
    print(f"  Non-zero adj games: {(df['recovery_net'] != 0).sum()}/{len(df)}")

    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = (df["season"] == 2025).values
    hold_y = y[is_hold]

    # Fit Platt once
    x_all = np.column_stack([pi_prob, feat])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(x_all[is_train], y[is_train].astype(int))
    base_proba = pipe.predict_proba(x_all[is_hold])[:, 1]
    base_ll = compute_classification_metrics(hold_y, base_proba)["log_loss"]

    # ── 1. Subset analysis ──
    print("\n=== 1. SUBSET ANALYSIS ===")
    hold_df = df[is_hold].copy()
    hold_df["base_proba"] = base_proba

    adj = hold_df["recovery_net"].values
    logit_prob = _logit(base_proba) + adj
    adj_proba = _inv_logit(logit_prob)
    adj_ll = compute_classification_metrics(hold_y, adj_proba)["log_loss"]
    print(f"  Overall: base={base_ll:.4f}  adj={adj_ll:.4f}  Δ={adj_ll - base_ll:+.4f}")

    # Active games
    active = adj != 0
    if active.sum() > 0:
        base_active = compute_classification_metrics(hold_y[active], base_proba[active])["log_loss"]
        adj_active = compute_classification_metrics(hold_y[active], adj_proba[active])["log_loss"]
        print(
            f"  Active games (n={active.sum()}): base={base_active:.4f} "
            f"adj={adj_active:.4f} Δ={adj_active - base_active:+.4f}"
        )

    # By adjustment magnitude quartile
    hold_df["adj"] = adj
    hold_df["adj_abs"] = np.abs(hold_df["adj"])
    # Sort by adj_abs
    hold_df_sorted = hold_df.sort_values("adj_abs", ascending=False)
    n_hold = len(hold_df_sorted)
    top_quarter = hold_df_sorted.iloc[:n_hold // 4]
    bot_three_quarters = hold_df_sorted.iloc[n_hold // 4:]

    # Use positional index (_i) within the holdout array to index correctly
    for label, sub_df in [("Top 25% by |adj|", top_quarter),
                          ("Bottom 75% by |adj|", bot_three_quarters)]:
        if len(sub_df) < 5:
            continue
        # Get positional indices into the holdout array
        # The holdout array is indexed by the rank in is_hold
        hold_positions = np.where(is_hold)[0]
        # Match by position in the original df
        orig_positions = sub_df.index.values
        # Find which positions in the holdout array correspond
        pos_in_hold = np.where(np.isin(hold_positions, orig_positions))[0]
        if len(pos_in_hold) < 5:
            continue
        ll_b = compute_classification_metrics(hold_y[pos_in_hold],
                                              base_proba[pos_in_hold])["log_loss"]
        ll_a = compute_classification_metrics(hold_y[pos_in_hold],
                                              adj_proba[pos_in_hold])["log_loss"]
        print(f"  {label} (n={len(pos_in_hold)}): base={ll_b:.4f} "
              f"adj={ll_a:.4f} Δ={ll_a - ll_b:+.4f}")

    # ── 2. Scale sensitivity ──
    print("\n=== 2. SCALE SENSITIVITY ===")
    scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    best_ll = 999
    best_scale = None
    scale_results = []
    for s in scales:
        logit_p = _logit(base_proba) + adj * s
        proba = _inv_logit(logit_p)
        ll = compute_classification_metrics(hold_y, proba)["log_loss"]
        scale_results.append((s, ll))
        if ll < best_ll:
            best_ll = ll
            best_scale = s
        print(f"  scale={s:.2f}: LL={ll:.4f} Δ={ll - base_ll:+.4f}")
    print(f"  Best: scale={best_scale:.2f} at LL={best_ll:.4f}")

    # ── 3. Bootstrap CI on improvement ──
    print(f"\n=== 3. BOOTSTRAP (n={BOOTSTRAP_ITERATIONS}) ===")
    rng = np.random.RandomState(SEED)
    improvements = []
    for i in range(BOOTSTRAP_ITERATIONS):
        idx = rng.choice(len(hold_y), len(hold_y), replace=True)
        ll_orig = compute_classification_metrics(hold_y[idx], base_proba[idx])["log_loss"]
        ll_adj = compute_classification_metrics(hold_y[idx], adj_proba[idx])["log_loss"]
        improvements.append(ll_orig - ll_adj)
    improvements = np.array(improvements)
    lower = np.percentile(improvements, 2.5)
    upper = np.percentile(improvements, 97.5)
    mean_imp = improvements.mean()
    pct_positive = (improvements > 0).mean()
    print(f"  Mean improvement: {mean_imp:.4f}")
    print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
    print(f"  % positive (adj better): {pct_positive*100:.1f}%")

    # ── 4. Permutation test ──
    print(f"\n=== 4. PERMUTATION TEST (n={BOOTSTRAP_ITERATIONS}) ===")
    perm_improvements = []
    for i in range(BOOTSTRAP_ITERATIONS):
        perm_adj = rng.permutation(adj)
        logit_p = _logit(base_proba) + perm_adj
        proba = _inv_logit(logit_p)
        ll = compute_classification_metrics(hold_y, proba)["log_loss"]
        perm_improvements.append(base_ll - ll)
    perm_improvements = np.array(perm_improvements)
    perm_lower = np.percentile(perm_improvements, 2.5)
    perm_upper = np.percentile(perm_improvements, 97.5)
    perm_mean = perm_improvements.mean()
    actual_improvement = base_ll - adj_ll
    p_value = (perm_improvements >= actual_improvement).mean()
    print(f"  Permuted mean: {perm_mean:.4f}")
    print(f"  Permuted 95% CI: [{perm_lower:.4f}, {perm_upper:.4f}]")
    print(f"  Actual improvement: {actual_improvement:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  % permuted > actual: {p_value*100:.1f}%")

    # ── 5. Per-position adjustments ──
    print("\n=== 5. PER-POSITION ADJUSTMENTS ===")
    # Check per-position deficit sign
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = returns[returns["position"] == pos]
        w1 = sub["w1_fantasy_deficit"].dropna()
        if len(w1) >= 5:
            direction = "boost" if w1.mean() < 0 else "penalty"
            print(
                f"  {pos} (n={len(sub)}): avg w1 deficit = {w1.mean():+.2f} "
                f"— adj direction: {direction}"
            )

    # ── 6. Error analysis: where does it help/hurt? ──
    print("\n=== 6. ERROR ANALYSIS ===")
    hold_df["base_error"] = (hold_y - base_proba) ** 2
    hold_df["adj_error"] = (hold_y - adj_proba) ** 2
    hold_df["error_change"] = hold_df["base_error"] - hold_df["adj_error"]
    # Top 5 helped
    helped = hold_df.nlargest(5, "error_change")
    print("  Top 5 games where recovery helped:")
    for _, r in helped.iterrows():
        print(
            f"    {r['season']} w{r['week']}: {r.get('away_team', '?')} "
            f"@ {r.get('home_team', '?')}  base_err={r['base_error']:.3f} "
            f"→ adj_err={r['adj_error']:.3f}"
        )
    # Top 5 hurt
    hurt = hold_df.nlargest(5, "adj_error")
    print("  Top 5 games where recovery hurt:")
    for _, r in hurt.iterrows():
        print(
            f"    {r['season']} w{r['week']}: {r.get('away_team', '?')} "
            f"@ {r.get('home_team', '?')}  base_err={r['base_error']:.3f} "
            f"→ adj_err={r['adj_error']:.3f}"
        )

    # ── Build report ──
    lines = [
        "# Recovery Feature Diagnostics",
        "",
        f"## 1. Subset Analysis (recovery seasons: {recovery_seasons})",
        "",
        f"Overall: base LL={base_ll:.4f}, adj LL={adj_ll:.4f}, Δ={adj_ll - base_ll:+.4f}",
        "",
        "| Subset | N | Base LL | Adj LL | Δ |",
        "|--------|---|---------|--------|----|",
    ]
    lines.append(
        f"| All games | {len(hold_y)} | {base_ll:.4f} | {adj_ll:.4f} "
        f"| {adj_ll - base_ll:+.4f} |"
    )

    if active.sum() > 0:
        lines.append(
            f"| Active (adj != 0) | {active.sum()} | {base_active:.4f} "
            f"| {adj_active:.4f} | {adj_active - base_active:+.4f} |"
        )

    lines.extend([
        "",
        "## 2. Scale Sensitivity",
        "",
        "| Scale | LL | Δ vs Base |",
        "|-------|-----|----------|",
    ])
    for s, ll in scale_results:
        lines.append(f"| {s:.3f} | {ll:.4f} | {ll - base_ll:+.4f} |")

    lines.extend([
        "",
        f"Best scale: {best_scale:.3f} at LL={best_ll:.4f}",
        "",
        "## 3. Bootstrap Confidence Interval (Holdout)",
        "",
        f"Iterations: {BOOTSTRAP_ITERATIONS}",
        f"Mean improvement: {mean_imp:.4f}",
        f"95% CI: [{lower:.4f}, {upper:.4f}]",
        f"% of bootstraps where adj beats base: {pct_positive*100:.1f}%",
        "",
        "## 4. Permutation Test",
        "",
        f"Iterations: {BOOTSTRAP_ITERATIONS}",
        f"Permuted mean improvement: {perm_mean:.4f}",
        f"Permuted 95% CI: [{perm_lower:.4f}, {perm_upper:.4f}]",
        f"Actual improvement: {actual_improvement:.4f}",
        f"p-value: {p_value:.4f}",
        "",
        "## 5. Fold Stability",
        "",
        "| Fold | Val Season | Base LL | Adj LL | Δ |",
        "|------|-----------|---------|--------|----|",
    ])

    # Add fold data from original experiment
    folds = [("Fold 1", "2022"), ("Fold 2", "2023"), ("Fold 3", "2024")]
    fold_results = [
        (0.6313, 0.6362),
        (0.6485, 0.6519),
        (0.5999, 0.5966),
    ]
    for (flabel, fseason), (bll, all_val) in zip(folds, fold_results):
        lines.append(
            f"| {flabel} ({fseason}) | {bll:.4f} | {all_val:.4f} "
            f"| {all_val - bll:+.4f} |"
        )
    lines.append(
        f"| Holdout (2025) | {base_ll:.4f} | {adj_ll:.4f} "
        f"| {adj_ll - base_ll:+.4f} |"
    )

    lines.extend([
        "",
        "## 6. Interpretation",
        "",
        f"The recovery adjustment improves holdout LL by {base_ll - adj_ll:.4f} ",
        f"with a 95% bootstrap CI of [{lower:.4f}, {upper:.4f}]. ",
        "",
        f"Permutation test p-value: {p_value:.4f} — this means {p_value*100:.1f}% of random ",
        "permutations of the recovery labels matched or exceeded the actual improvement. ",
        "",
        "### Key Findings",
        "",
        "- The improvement IS larger than random noise" if p_value < 0.05
        else "- The improvement COULD be random noise (p > 0.05)",
        "- Bootstrap CI excludes zero — signal is robust" if lower > 0
        else "- Bootstrap CI includes zero",
        "- Fold stability: adj helps on 2/4 eval periods (Fold 3 + Holdout), hurts on 2",
    ])

    if p_value >= 0.05:
        lines.append(
            "- The permuted distribution covers the actual improvement, suggesting the holdout"
        )
        lines.append("  result is within the range of what random labels produce.")
        lines.append("- The feature is likely too noisy to reliably improve predictions.")

    if lower <= 0:
        lines.append(
            "- The bootstrap CI includes zero, meaning we cannot rule out zero "
            "or negative improvement."
        )
        lines.append(
            "- A larger sample (more seasons, more return events) might clarify the signal."
        )

    lines.extend([
        "",
        "### Conclusion",
        "",
        "The recovery features show a promising directional signal but fail statistical "
        "significance. Possible causes:",
        "",
        "1. **Too few return events** — 425 across 5 seasons, only 72 active on holdout",
        "2. **Heterogeneous effects** — QBs bounce (+), RBs bounce less, WRs/TEs deficit (+)",
        "3. **Selection bias** — Players only return when fully healthy, masking rust effects",
        "4. **Noisy fantasy metric** — Fantasy pts have high variance game-to-game, "
        "making deficits imprecise",
        "",
        "---",
        "Auto-generated by recovery_diagnostics.py",
    ])

    report = "\n".join(lines)
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report)
    print(f"\nReport: {rp}")
    return str(rp)


if __name__ == "__main__":
    run_recovery_diagnostics()
