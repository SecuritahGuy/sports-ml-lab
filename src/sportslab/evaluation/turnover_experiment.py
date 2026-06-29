"""Focused turnover features — tests rolling TO differential in isolation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, TARGET_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features
from sportslab.features.turnovers import compute_turnover_features

HOLDOUT_SEASON = 2025
ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
DEFAULT_REPORT = "reports/experiments/turnover_features.md"

TO3 = ["home_to_net_3", "away_to_net_3", "to_net_diff_3"]
TO5 = ["home_to_net_5", "away_to_net_5", "to_net_diff_5"]

MODEL_VARIANTS = [
    ("incumbent", FEATURE_COLS, "Incumbent (qb_changed + rolling_mov_3)"),
    ("to_net_3", FEATURE_COLS + TO3, "+ rolling TO net (3-game)"),
    ("to_net_5", FEATURE_COLS + TO5, "+ rolling TO net (5-game)"),
    ("to_net_both", FEATURE_COLS + TO3 + TO5, "+ rolling TO net (3+5 game)"),
    ("to_all", FEATURE_COLS + TO3 + TO5, "+ all turnover features"),
]


def _build_feature_matrix(df: pd.DataFrame, feat_cols) -> np.ndarray:
    elo = df["elo_prob"].values
    avail = [c for c in feat_cols if c in df.columns]
    if avail:
        return np.column_stack([elo] + [df[c].values for c in avail])
    return elo.reshape(-1, 1)


def _run_rolling_ll(df_all: pd.DataFrame, feat_cols, y: np.ndarray) -> list[float]:
    fold_lls = []
    for train_s, val_s in ROLLING_FOLDS:
        tr = df_all["season"].isin(train_s).values
        va = (df_all["season"] == val_s).values
        x_tr = _build_feature_matrix(df_all.loc[tr], feat_cols)
        x_va = _build_feature_matrix(df_all.loc[va], feat_cols)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(x_tr, y[tr].astype(int))
        prob = pipe.predict_proba(x_va)[:, 1]
        valid = ~np.isnan(y[va])
        from sklearn.metrics import log_loss as sk_ll
        ll = float(sk_ll(y[va][valid].astype(int), prob[valid]))
        fold_lls.append(ll)
    return fold_lls


def _run_holdout(df_all: pd.DataFrame, feat_cols, y: np.ndarray) -> float:
    tr = (df_all["season"] < HOLDOUT_SEASON).values
    va = (df_all["season"] == HOLDOUT_SEASON).values
    x_tr = _build_feature_matrix(df_all.loc[tr], feat_cols)
    x_va = _build_feature_matrix(df_all.loc[va], feat_cols)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(x_tr, y[tr].astype(int))
    prob = pipe.predict_proba(x_va)[:, 1]
    valid = ~np.isnan(y[va])
    from sklearn.metrics import log_loss as sk_ll
    return float(sk_ll(y[va][valid].astype(int), prob[valid]))


def run_turnover_experiment(
    ft_path: str = FEATURE_TABLE_PATH,
    report_path: str = DEFAULT_REPORT,
) -> str:
    """Run focused turnover features experiment (5 variants)."""
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    print("Building features...")
    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG, team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_situational_features(df)

    print("Computing turnover features...")
    df = compute_turnover_features(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df["is_neutral"].fillna(False).values
    df = df[mask].copy().reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(float).values
    print(f"  Eligible games: {len(df)}")

    print("\n--- Rolling-Origin Validation ---")
    rolling_results = {}
    for name, feats, desc in MODEL_VARIANTS:
        folds = _run_rolling_ll(df, feats, y)
        avg = float(np.mean(folds))
        rolling_results[name] = {"folds": [round(v, 4) for v in folds],
                                  "avg_val_ll": round(avg, 4)}
        print(f"  {name:20s}  val={avg:.4f}  folds={[round(v,4) for v in folds]}")

    print("\n--- Fitted-Once (2021-2024 train, 2025 predict) ---")
    holdout_results = {}
    for name, feats, desc in MODEL_VARIANTS:
        hold = _run_holdout(df, feats, y)
        holdout_results[name] = round(hold, 4)
        print(f"  {name:20s}  hold={hold:.4f}")

    inc_roll = rolling_results["incumbent"]["avg_val_ll"]
    inc_hold = holdout_results["incumbent"]

    # Report
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    _w = lines.append
    _w("# Turnover Features Experiment")
    _w("")
    _w(f"*Generated by `sportslab turnover-experiment` ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*")
    _w("")
    _w("## Model Variants")
    _w("")
    _w("| Label | Features | Description |")
    _w("|-------|----------|-------------|")
    for name, feats, desc in MODEL_VARIANTS:
        _w(f"| {name} | {len(feats)} feats | {desc} |")
    _w("")

    _w("## Rolling-Origin Validation (3 folds: 2022, 2023, 2024)")
    _w("")
    _w("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |")
    _w("|-------|-----------|-------|-------|-------|")
    for name, _, _ in MODEL_VARIANTS:
        r = rolling_results[name]
        folds_str = " | ".join(f"{v:.4f}" for v in r["folds"])
        d = r["avg_val_ll"] - inc_roll
        d_str = f" ({d:+.4f})" if name != "incumbent" else ""
        _w(f"| {name} | {r['avg_val_ll']:.4f}{d_str} | {folds_str} |")
    _w("")

    _w("## Fitted-Once (Holdout: 2025)")
    _w("")
    _w("| Model | Holdout LL | Δ vs Incumbent |")
    _w("|-------|-----------|----------------|")
    for name, _, _ in MODEL_VARIANTS:
        h = holdout_results[name]
        d = h - inc_hold
        d_str = f"{d:+.4f}" if name != "incumbent" else "—"
        _w(f"| {name} | {h:.4f} | {d_str} |")
    _w("")

    _w("## Promotion Check")
    _w("")
    for name, _, _ in MODEL_VARIANTS:
        if name == "incumbent":
            continue
        bv = rolling_results[name]["avg_val_ll"] < inc_roll
        bh = holdout_results[name] < inc_hold
        tag = "**PROMOTED**" if (bv and bh) else "Rejected"
        _w(f"| {name} | val {'✓' if bv else '✗'} | hold {'✓' if bh else '✗'} | {tag} |")
    _w("")

    beats_val = any(rolling_results[n]["avg_val_ll"] < inc_roll
                    for n, _, _ in MODEL_VARIANTS if n != "incumbent")
    beats_hold = any(holdout_results[n] < inc_hold
                     for n, _, _ in MODEL_VARIANTS if n != "incumbent")
    if beats_val and beats_hold:
        _w("**✅ New incumbent candidate found.**")
    elif beats_val or beats_hold:
        _w("**Mixed results — wins one metric but not both.**")
    else:
        _w("**No model promoted.** Turnover features rejected.")

    _w("")
    _w("---")
    _w(f"*Incumbent: {INCUMBENT_VERSION}, {INCUMBENT_HOLDOUT_LL} holdout LL.*")

    rp.write_text("\n".join(lines))

    print(f"\nTurnover report: {rp}")
    print(f"  Incumbent val:   {inc_roll:.4f}")
    print(f"  Incumbent hold:  {inc_hold:.4f}")
    print(f"  Best val:  {min(rolling_results[n]['avg_val_ll'] for n, _, _ in MODEL_VARIANTS):.4f}")
    print(f"  Best hold: {min(holdout_results.values()):.4f}")

    return str(rp)
