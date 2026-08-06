"""Test player recovery features against the incumbent.

Uses the player recovery analysis pipeline to identify returning players
and compute team-level expected deficits, then applies them as logit-space
adjustments on top of the Pi-Ratings + qb_changed + mov_3 base.
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

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
# Expanded folds include pre-2021 training data
EXPANDED_FOLDS = [
    ([2016, 2017, 2018, 2019, 2021], 2022),
    ([2016, 2017, 2018, 2019, 2021, 2022], 2023),
    ([2016, 2017, 2018, 2019, 2021, 2022, 2023], 2024),
]
SEED = 42

PI_ALPHA = 0.5
PI_BASE_K = 28
PI_HK_RATIO = 1.25
PI_HFA = 30
PI_REG = 0.0


def run_recovery_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/player_recovery_experiment.md",
    recovery_seasons: Optional[list[int]] = None,
    use_expanded_folds: bool = False,
) -> str:
    if recovery_seasons is None:
        recovery_seasons = [2021, 2022, 2023, 2024, 2025]
    folds = EXPANDED_FOLDS if use_expanded_folds else ROLLING_FOLDS

    import nfl_data_py as nfl

    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    # Build Pi-Ratings base
    df = compute_pi_ratings_features(
        df_raw, base_k=PI_BASE_K, home_advantage=PI_HFA,
        preseason_regression=PI_REG, alpha=PI_ALPHA, hk_ratio=PI_HK_RATIO,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)

    # Filter eligible
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"Eligible games: {len(df)}")

    y = df[TARGET_COLUMN].astype(float).values
    feat_cols = ["home_qb_changed", "away_qb_changed",
                 "home_rolling_mov_3", "away_rolling_mov_3"]
    feat = np.column_stack([df[c].values for c in feat_cols])

    # Build recovery adjustments using specified seasons
    print(f"\nBuilding player recovery data (seasons: {recovery_seasons})...")
    inj = nfl.import_injuries(recovery_seasons)
    inj["gsis_id"] = inj["gsis_id"].str.strip()
    pg = build_player_game_table(recovery_seasons)
    returns = identify_return_events(pg, inj, min_games_out=2)
    print(f"  Return events: {len(returns)}")

    game_adj = compute_game_recovery_adjustments(returns, recovery_seasons)
    df = df.merge(game_adj, on="game_id", how="left")
    df["home_recovery_adj"] = df["home_recovery_adj"].fillna(0)
    df["away_recovery_adj"] = df["away_recovery_adj"].fillna(0)
    df["recovery_net"] = df["home_recovery_adj"] - df["away_recovery_adj"]
    print(f"  Games with non-zero recovery adj: {(df['recovery_net']!=0).sum()}/{len(df)}")

    pil_prob = df["pi_prob"].values.astype(float)

    def _stack(prob, extra=None):
        cols = [prob]
        if extra is not None:
            cols.append(extra)
        return np.column_stack(cols + [feat])

    def _logit(p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))

    def _inv_logit(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

    variants = {
        "Incumbent (Pi only)": {"prob": pil_prob, "extra": None, "adj": None},
        "Incumbent + Recovery (logit adj)": {
            "prob": pil_prob, "extra": None, "adj": "recovery_net",
        },
        "Recovery only": {"prob": None, "extra": None, "adj": "recovery_net"},
    }

    results = {name: {"fold_lls": []} for name in variants}
    hold_results = {name: {} for name in variants}

    # Rolling-origin 3-fold
    for fold_idx, (train_s, val_s) in enumerate(folds):
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values

        for name, cfg in variants.items():
            if cfg["prob"] is None:
                # No prob model, just adjustment
                y_adj = y[va].copy()
                logit_prob = _logit(0.5 * np.ones_like(y_adj)) + df[va][cfg["adj"]].values
                proba = _inv_logit(logit_prob)
                val_ll = compute_classification_metrics(y_adj, proba)["log_loss"]
                results[name]["fold_lls"].append(val_ll)
            else:
                x_all = _stack(cfg["prob"], cfg["extra"])
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
                ])
                pipe.fit(x_all[tr], y[tr].astype(int))

                if cfg["adj"] is not None:
                    logit_prob = _logit(pipe.predict_proba(x_all[va])[:, 1])
                    logit_prob += df[va][cfg["adj"]].values
                    proba = _inv_logit(logit_prob)
                else:
                    proba = pipe.predict_proba(x_all[va])[:, 1]

                val_ll = compute_classification_metrics(y[va], proba)["log_loss"]
                results[name]["fold_lls"].append(val_ll)

    for name in variants:
        avg = float(np.mean(results[name]["fold_lls"]))
        results[name]["val_ll"] = avg

    # Holdout
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]

    for name, cfg in variants.items():
        if cfg["prob"] is None:
            logit_prob = _logit(0.5 * np.ones_like(hold_y)) + df[is_hold][cfg["adj"]].values
            proba = _inv_logit(logit_prob)
        else:
            x_all = _stack(cfg["prob"], cfg["extra"])
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
            ])
            pipe.fit(x_all[is_train], y[is_train].astype(int))
            if cfg["adj"] is not None:
                logit_prob = _logit(pipe.predict_proba(x_all[is_hold])[:, 1])
                logit_prob += df[is_hold][cfg["adj"]].values
                proba = _inv_logit(logit_prob)
            else:
                proba = pipe.predict_proba(x_all[is_hold])[:, 1]

        m = compute_classification_metrics(hold_y, proba)
        hold_results[name] = m

    for name in variants:
        print(
            f"  {name}: val={results[name]['val_ll']:.4f}  "
            f"hold={hold_results[name]['log_loss']:.4f}"
        )

    # Subgroup analysis: games where recovery adjustment is non-zero
    print("\n--- Subgroup: Games with non-zero recovery adj ---")
    adj_col = df["recovery_net"].values
    active_games = adj_col != 0
    print(f"  Non-zero adj games: {active_games.sum()}")

    for name in ["Incumbent (Pi only)", "Incumbent + Recovery (logit adj)"]:
        cfg = variants[name]
        x_all = _stack(cfg["prob"], cfg["extra"])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_all[is_train], y[is_train].astype(int))

        if cfg["adj"] is not None:
            logit_prob = _logit(pipe.predict_proba(x_all[is_hold])[:, 1])
            logit_prob += df[is_hold][cfg["adj"]].values
            proba = _inv_logit(logit_prob)
        else:
            proba = pipe.predict_proba(x_all[is_hold])[:, 1]

        # Only active games in holdout
        hold_active = active_games & is_hold
        if hold_active.sum() > 0:
            ll_active = compute_classification_metrics(hold_y[hold_active[is_hold]],
                                                        proba[hold_active[is_hold]])["log_loss"]
            print(f"  {name} (active games only, n={hold_active.sum()}): LL={ll_active:.4f}")

    # Generate report
    inc_name = "Incumbent (Pi only)"
    inc_val = results[inc_name]["val_ll"]
    inc_hold = hold_results[inc_name]["log_loss"]

    lines = [
        "# Player Recovery Experiment",
        "",
        "## Data",
        "",
        f"- Recovery data seasons: {recovery_seasons}",
        f"- Return events identified: {len(returns)}",
        f"- Games with non-zero recovery adjustment: {(df['recovery_net']!=0).sum()}/{len(df)}",
        "- Rolling-origin 3-fold: 2022/2023/2024 val, 2025 holdout",
        "",
        "## Variants",
        "",
        "| ID | Model | Description |",
        "|---|-------|-------------|",
        "| A | Incumbent (Pi-Ratings + qb_changed + mov_3 + Platt) | Base model |",
        "| B | Incumbent + Recovery | Logit-space adjustment from player recovery curves |",
        "| C | Recovery only | Recovery adjustment on 0.5 baseline (diagnostic) |",
        "",
        "## Validation (Rolling-Origin 3-Fold)",
        "",
        "| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |",
        "|-------|-----------|-------|-------|-------|",
    ]

    for name in variants:
        r = results[name]
        lines.append(
            f"| {name} | {r['val_ll']:.4f}"
            f" | {r['fold_lls'][0]:.4f}"
            f" | {r['fold_lls'][1]:.4f}"
            f" | {r['fold_lls'][2]:.4f} |"
        )

    lines.extend([
        "",
        "## Holdout (2025)",
        "",
        "| Model | Hold LL | Brier | AUC | Acc |",
        "|-------|---------|-------|-----|------|",
    ])
    for name in variants:
        h = hold_results[name]
        lines.append(
            f"| {name} | {h['log_loss']:.4f}"
            f" | {h['brier_score']:.4f}"
            f" | {h['roc_auc']:.4f}"
            f" | {h['accuracy']:.4f} |"
        )

    lines.extend([
        "",
        "## Comparison vs Incumbent",
        "",
        f"Incumbent (A): val={inc_val:.4f}, hold={inc_hold:.4f}",
        "",
        "| Model | Δval | Δhold | Decision |",
        "|-------|------|-------|----------|",
    ])

    for name in variants:
        if name == inc_name:
            continue
        dv = results[name]["val_ll"] - inc_val
        dh = hold_results[name]["log_loss"] - inc_hold
        if dv <= -0.001 and dh <= -0.001:
            verdict = "✅ PROMOTED"
        elif dv <= 0 and dh <= 0:
            verdict = "Better but below threshold"
        elif dv <= -0.001 and dh > -0.001:
            verdict = "Wins val, loses hold"
        elif dv > -0.001 and dh <= -0.001:
            verdict = "Loses val, wins hold"
        else:
            verdict = "Worse on both"
        lines.append(f"| {name} | {dv:+.4f} | {dh:+.4f} | {verdict} |")

    lines.extend([
        "",
        "## Recovery Curve Summary",
        "",
        "Key findings from the recovery analysis:",
        "",
        "- **QB**: Week 1 bounce (−4.7 fantasy deficit, i.e. *better* than baseline). "
        "Week 2 regression (+2.8).",
        "- **RB**: Week 1 slight bounce (−1.0). Week 2 regression (+2.0).",
        "- **WR**: Small persistent deficit (+0.5 week 1, +0.5 week 2).",
        "- **TE**: Small persistent deficit (+0.6 week 1, +0.5 week 2).",
        "- **Compounding**: WR repeat injuries (+1.86) worse than single (+0.16).",
        "",
        "## Decision",
        "",
    ])

    prom = [n for n in variants if n != inc_name and
            results[n]["val_ll"] <= inc_val - 0.001 and
            hold_results[n]["log_loss"] <= inc_hold - 0.001]
    if prom:
        lines.append(f"**Promoted: {', '.join(prom)}**")
    else:
        lines.append("**No recovery variant beats incumbent on both val and holdout by ≥ 0.001.**")

    lines.append("")
    lines.append("---")
    lines.append("Report: player_recovery_experiment.py")

    report = "\n".join(lines)
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report)
    print(f"\nReport: {rp}")
    return str(rp)


if __name__ == "__main__":
    run_recovery_experiment(recovery_seasons=[2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
                            use_expanded_folds=True)
