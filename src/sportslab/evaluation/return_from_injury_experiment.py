"""Test return-from-injury rust features against the incumbent.

Hypothesis: Players returning from multi-game absences need 1-2 games
to ramp up. Teams with multiple returning players (especially QBs) will
underperform their Elo expectation in that first game back.

Uses nflreadpy injury report data to track multi-week absences and
computes team-level rust scores. Tests as logit-space features on top
of the Pi-Ratings + qb_changed + mov_3 incumbent.
"""

from pathlib import Path

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
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_pi_ratings_features
from sportslab.features.return_from_injury import compute_rust_features
from sportslab.features.situational import compute_situational_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
SEED = 42

PI_ALPHA = 0.5
PI_BASE_K = 28
PI_HK_RATIO = 1.25
PI_HFA = 30
PI_REG = 0.0

RUST_COLUMNS = [
    "home_rust_score", "away_rust_score",
    "home_rust_qb", "away_rust_qb",
    "home_rust_skill", "away_rust_skill",
    "home_rust_games_missed", "away_rust_games_missed",
]

INJURY_SEASONS = [2021, 2022, 2023, 2024, 2025]


def run_return_from_injury_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/return_from_injury.md",
) -> str:
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    # Build Pi-Ratings base features
    df = compute_pi_ratings_features(
        df_raw, base_k=PI_BASE_K, home_advantage=PI_HFA,
        preseason_regression=PI_REG, alpha=PI_ALPHA, hk_ratio=PI_HK_RATIO,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)

    # Compute return-from-injury rust features
    df = compute_rust_features(df, seasons=INJURY_SEASONS)

    # Filter eligible non-neutral games
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"Eligible games: {len(df)}")

    # Feature columns
    feat_cols = ["home_qb_changed", "away_qb_changed",
                 "home_rolling_mov_3", "away_rolling_mov_3"]

    pi_prob = df["pi_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values
    feat = np.column_stack([df[c].values for c in feat_cols])

    # Rust features
    rust_arr = np.column_stack([df[c].values.astype(float) for c in RUST_COLUMNS])
    rust_qb = np.column_stack([df["home_rust_qb"].values, df["away_rust_qb"].values])
    rust_skill = np.column_stack([df["home_rust_skill"].values, df["away_rust_skill"].values])

    # Variants
    def _stack(prob, extra=None):
        cols = [prob]
        if extra is not None:
            cols.append(extra)
        return np.column_stack(cols + [feat])

    variants = {
        "Incumbent (Pi only)": {"prob": pi_prob, "extra": None},
        "Incumbent + Rust (all 8)": {"prob": pi_prob, "extra": rust_arr},
        "Incumbent + Rust (QB only)": {"prob": pi_prob, "extra": rust_qb},
        "Incumbent + Rust (Skill only)": {"prob": pi_prob, "extra": rust_skill},
    }

    # Check rust column coverage
    rust_active = (rust_arr.sum(axis=1) != 0).sum()
    print(f"  Games with any rust feature active: {rust_active}/{len(df)}")

    def _fit_platt(x_train, y_train):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_train, y_train.astype(int))
        return pipe

    results = {name: {"fold_lls": []} for name in variants}
    hold_results = {name: {} for name in variants}

    # Rolling-origin 3-fold
    for fold_idx, (train_s, val_s) in enumerate(ROLLING_FOLDS):
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values

        for name, cfg in variants.items():
            x_all = _stack(cfg["prob"], cfg["extra"])
            pipe = _fit_platt(x_all[tr], y[tr].astype(int))
            proba = pipe.predict_proba(x_all[va])[:, 1]
            val_ll = compute_classification_metrics(y[va], proba)["log_loss"]
            results[name]["fold_lls"].append(val_ll)

    # Holdout
    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]

    for name, cfg in variants.items():
        x_all = _stack(cfg["prob"], cfg["extra"])
        pipe = _fit_platt(x_all[is_train], y[is_train].astype(int))
        proba = pipe.predict_proba(x_all[is_hold])[:, 1]
        m = compute_classification_metrics(hold_y, proba)
        hold_results[name] = m

    for name in variants:
        avg = float(np.mean(results[name]["fold_lls"]))
        results[name]["val_ll"] = avg
        print(f"  {name}: val={avg:.4f}  hold={hold_results[name]['log_loss']:.4f}")

    # Report
    inc_name = "Incumbent (Pi only)"
    inc_val = results[inc_name]["val_ll"]
    inc_hold = hold_results[inc_name]["log_loss"]

    report_lines = [
        "# Return-from-Injury Rust Features",
        "",
        "Testing whether players returning from multi-game absences cause teams",
        "to underperform their Elo expectation.",
        "",
        "## Data",
        "",
        f"- Injury sources: {len(INJURY_SEASONS)} seasons (2021-2025), nflreadpy import_injuries()",
        "- Return events: players with 2+ consecutive \"Out\" weeks, then not Out",
        "- Rust weight: position_weight × sqrt(games_missed)",
        "- Weights: QB=5.0, RB=3.0, WR/TE=2.0, OL=1.5, DL/LB/DB=1.0",
        "",
        "## Variants",
        "",
        "| ID | Model | Rust Features |",
        "|---|-------|--------------|",
        "| A | Incumbent (Pi-Ratings + qb_changed + mov_3 + Platt) | None |",
        "| B | Incumbent + All Rust | 8 columns: score, QB, skill, games_missed (H/A) |",
        "| C | Incumbent + QB Rust | home_rust_qb, away_rust_qb |",
        "| D | Incumbent + Skill Rust | home_rust_skill, away_rust_skill |",
        "",
    ]

    # Rust coverage summary
    report_lines.append("## Rust Coverage\n")
    report_lines.append(f"Games with any rust feature active: {rust_active}/{len(df)}\n")
    report_lines.append(f"Injury data rows loaded: from {INJURY_SEASONS}\n")
    report_lines.append("")

    report_lines.extend([
        "## Validation (Rolling-Origin 3-Fold)",
        "",
        "| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |",
        "|-------|-----------|-------|-------|-------|",
    ])
    for name in variants:
        r = results[name]
        report_lines.append(
            f"| {name} | {r['val_ll']:.4f}"
            f" | {r['fold_lls'][0]:.4f}"
            f" | {r['fold_lls'][1]:.4f}"
            f" | {r['fold_lls'][2]:.4f} |"
        )

    report_lines.extend([
        "",
        "## Holdout (2025)",
        "",
        "| Model | Hold LL | Brier | AUC | Acc |",
        "|-------|---------|-------|-----|------|",
    ])
    for name in variants:
        h = hold_results[name]
        report_lines.append(
            f"| {name} | {h['log_loss']:.4f}"
            f" | {h['brier_score']:.4f}"
            f" | {h['roc_auc']:.4f}"
            f" | {h['accuracy']:.4f} |"
        )

    report_lines.extend([
        "",
        "## Comparison vs Incumbent",
        "",
        f"Incumbent (A): val={inc_val:.4f}, hold={inc_hold:.4f}",
        "",
        "| Model | Δval | Δhold | Decision |",
        "|-------|------|-------|----------|",
    ])

    prom = []
    for name in variants:
        if name == inc_name:
            continue
        dv = results[name]["val_ll"] - inc_val
        dh = hold_results[name]["log_loss"] - inc_hold
        verdict = ""
        if dv <= -0.001 and dh <= -0.001:
            verdict = "✅ PROMOTED"
            prom.append(name)
        elif dv <= -0.001 and dh > -0.001:
            verdict = "Wins val, loses hold"
        elif dv > -0.001 and dh <= -0.001:
            verdict = "Loses val, wins hold"
        elif dv <= 0 and dh <= 0:
            verdict = "Better but below threshold"
        else:
            verdict = "Worse on both"
        report_lines.append(f"| {name} | {dv:+.4f} | {dh:+.4f} | {verdict} |")

    report_lines.extend([
        "",
        "## Decision",
        "",
    ])
    if prom:
        report_lines.append(f"**Promoted: {', '.join(prom)}**")
        report_lines.append("")
        report_lines.append("Return-from-injury rust features improve prediction.")
    else:
        report_lines.append(
            "**No rust variant beats incumbent on both val and holdout by ≥ 0.001.**"
        )
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("Report: return_from_injury_experiment.py")

    report = "\n".join(report_lines)
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report)
    print(f"\nReport: {rp}")
    return str(rp)
