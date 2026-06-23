"""Market baseline comparison against MOV Elo+Platt incumbent.

Compares the incumbent, market-implied probabilities, and their combination
via rolling-origin validation with a one-shot 2025 holdout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.ratings import compute_elo_features

HOLDOUT_SEASON = 2025

ROLLING_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.20
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0

INCUMBENT_HOLDOUT_LL = 0.6373


def _moneyline_to_prob(moneyline: float) -> float:
    """Convert American moneyline odds to implied probability."""
    if moneyline > 0:
        return 100.0 / (moneyline + 100.0)
    return -moneyline / (-moneyline + 100.0)


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add de-vigged home win probability from moneyline odds.

    Args:
        df: Must contain 'home_moneyline' and 'away_moneyline' columns.

    Returns:
        DataFrame with added 'market_home_prob' column.
    """
    out = df.copy()
    home_implied = out["home_moneyline"].apply(_moneyline_to_prob)
    away_implied = out["away_moneyline"].apply(_moneyline_to_prob)
    overround = home_implied + away_implied
    out["market_home_prob"] = home_implied / overround
    return out


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


def _logistic_pipe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def run_market_baseline(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/market_baseline.md",
) -> str:
    """Run market baseline comparison experiment."""
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Build features ──
    print("=== Building features ===")

    print("  Computing MOV Elo features (incumbent params)...")
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
    )

    print("  Computing market features...")
    df = compute_market_features(df)

    # ── Filter ──
    df = _filter_df(df)

    y = df[TARGET_COLUMN].astype(float).values
    elo_prob = df["elo_prob"].values
    market_prob = df["market_home_prob"].values

    print(f"  Total rows (filtered): {len(df)}")
    print(f"  Market prob range: [{market_prob.min():.4f}, {market_prob.max():.4f}]")
    print(f"  Market prob mean: {market_prob.mean():.4f}")

    # ═══ Rolling-origin evaluation ═══
    print("\n=== Rolling-Origin Evaluation ===")

    platt_results: list[dict] = []
    market_results: list[dict] = []
    market_platt_results: list[dict] = []
    elo_market_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df["season"].isin(train_seasons).values
        is_val = (df["season"] == val_season).values

        train_y_ = y[is_train].astype(int)
        val_y_ = y[is_val]

        train_elo = elo_prob[is_train]
        val_elo = elo_prob[is_val]

        train_market = market_prob[is_train]
        val_market = market_prob[is_val]

        # 1. Platt-scaled MOV Elo (incumbent)
        platt = _fit_platt(train_elo, train_y_)
        platt_proba = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_proba)
        platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_m["log_loss"],
                "metrics": platt_m,
                "model": platt,
            }
        )

        # 2. Raw market (de-vigged)
        market_m = compute_classification_metrics(val_y_, val_market)
        market_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": market_m["log_loss"],
                "metrics": market_m,
                "model": None,
            }
        )

        # 3. Platt-calibrated market
        market_platt = _fit_platt(train_market, train_y_)
        market_platt_proba = market_platt.predict_proba(val_market.reshape(-1, 1))[:, 1]
        market_platt_m = compute_classification_metrics(val_y_, market_platt_proba)
        market_platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": market_platt_m["log_loss"],
                "metrics": market_platt_m,
                "model": market_platt,
            }
        )

        # 4. Elo + Market combined
        elo_market_train = np.column_stack([train_elo, train_market])
        elo_market_val = np.column_stack([val_elo, val_market])
        elo_market_pipe = _logistic_pipe()
        elo_market_pipe.fit(elo_market_train, train_y_)
        elo_market_proba = elo_market_pipe.predict_proba(elo_market_val)[:, 1]
        elo_market_m = compute_classification_metrics(val_y_, elo_market_proba)
        elo_market_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": elo_market_m["log_loss"],
                "metrics": elo_market_m,
                "model": elo_market_pipe,
            }
        )

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" platt={platt_m['log_loss']:.4f}"
            f" market={market_m['log_loss']:.4f}"
            f" mkt+platt={market_platt_m['log_loss']:.4f}"
            f" elo+mkt={elo_market_m['log_loss']:.4f}"
        )

    # ── Average validation metrics ──
    def _avg_ll(results):
        valid = [r for r in results if r["log_loss"] != float("inf")]
        if not valid:
            return float("inf")
        return float(np.mean([r["log_loss"] for r in valid]))

    avg_platt = _avg_ll(platt_results)
    avg_market = _avg_ll(market_results)
    avg_market_platt = _avg_ll(market_platt_results)
    avg_elo_market = _avg_ll(elo_market_results)

    print("\n=== Average Validation Log Loss ===")
    print(f"  Platt (incumbent):       {avg_platt:.4f}")
    print(f"  Market (raw):            {avg_market:.4f}")
    print(f"  Market + Platt:          {avg_market_platt:.4f}")
    print(f"  Elo + Market:            {avg_elo_market:.4f}")

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (df["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]
    hold_market = market_prob[is_hold]

    is_train_full = df["season"].isin([2021, 2022, 2023, 2024]).values
    train_y_full = y[is_train_full].astype(int)
    train_elo_full = elo_prob[is_train_full]
    train_market_full = market_prob[is_train_full]

    # 1. Platt incumbent
    platt_full = _fit_platt(train_elo_full, train_y_full)
    hold_platt_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, hold_platt_proba)
    print(f"  Platt (incumbent): {hold_platt_m['log_loss']:.4f}")

    # 2. Raw market
    hold_market_m = compute_classification_metrics(hold_y, hold_market)
    print(f"  Market (raw):      {hold_market_m['log_loss']:.4f}")

    # 3. Platt-calibrated market
    market_platt_full = _fit_platt(train_market_full, train_y_full)
    hold_market_platt_proba = market_platt_full.predict_proba(hold_market.reshape(-1, 1))[:, 1]
    hold_market_platt_m = compute_classification_metrics(hold_y, hold_market_platt_proba)
    print(f"  Market + Platt:    {hold_market_platt_m['log_loss']:.4f}")

    # 4. Elo + Market
    elo_market_full = np.column_stack([train_elo_full, train_market_full])
    elo_market_hold = np.column_stack([hold_elo, hold_market])
    elo_market_final = _logistic_pipe()
    elo_market_final.fit(elo_market_full, train_y_full)
    hold_elo_market_proba = elo_market_final.predict_proba(elo_market_hold)[:, 1]
    hold_elo_market_m = compute_classification_metrics(hold_y, hold_elo_market_proba)
    print(f"  Elo + Market:      {hold_elo_market_m['log_loss']:.4f}")

    # ═══ Report ═══
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# Market Baseline Comparison\n\n")
        f.write(
            "Comparing the MOV Elo+Platt incumbent against moneyline-implied probabilities.\n\n"
        )

        f.write("## Methodology\n\n")
        f.write("- Moneyline odds converted to implied probabilities:\n")
        f.write("  - Negative odds (favorite): `prob = -odds / (-odds + 100)`\n")
        f.write("  - Positive odds (underdog): `prob = 100 / (odds + 100)`\n")
        f.write("- Vig (overround) removed via multiplicative normalization:\n")
        f.write("  `fair_home_prob = home_implied / (home_implied + away_implied)`\n")
        f.write("- All comparisons use rolling-origin validation (2025 untouched until final).\n")
        f.write("- Elo + Market combines both signals via logistic regression.\n\n")

        f.write("## Data\n\n")
        f.write("| Stat | Value |\n")
        f.write("|------|-------|\n")
        f.write(f"| Games (filtered) | {len(df)} |\n")
        f.write(f"| Market prob range | [{market_prob.min():.4f}, {market_prob.max():.4f}] |\n")
        f.write(f"| Market prob mean | {market_prob.mean():.4f} |\n")
        hi = df["home_moneyline"].apply(_moneyline_to_prob)
        ai = df["away_moneyline"].apply(_moneyline_to_prob)
        f.write(f"| Avg overround | {(hi + ai - 1).mean():.4f}' |\n")

        f.write("## Incumbent Params\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| K-factor | {BEST_K} |\n")
        f.write(f"| HFA | {BEST_HFA} |\n")
        f.write(f"| Preseason regression | {BEST_REG} |\n")
        f.write(f"| MOV type | {BEST_MOV_TYPE} |\n")
        f.write(f"| MOV scale | {BEST_MOV_SCALE} |\n")
        f.write(f"| MOV cap | {BEST_MOV_CAP} |\n\n")

        f.write("## Data Split\n\n")
        f.write("| Fold | Training | Validation |\n")
        f.write("|------|----------|------------|\n")
        for i, (tr, val) in enumerate(ROLLING_FOLDS, 1):
            f.write(f"| {i} | {tr} | {val} |\n")
        f.write(f"| Holdout | 2021–2024 | {HOLDOUT_SEASON} |\n\n")

        f.write("## Models Compared\n\n")
        f.write("| Model | Description |\n")
        f.write("|-------|-------------|\n")
        f.write("| **Platt (incumbent)** | MOV Elo + Platt scaling |\n")
        f.write("| **Market (raw)** | De-vigged moneyline implied probability |\n")
        f.write("| **Market + Platt** | Platt-calibrated market (tests favorite-longshot bias) |\n")
        f.write("| **Elo + Market** | Logistic regression on both signals |\n\n")

        f.write("## Average Validation Log Loss\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")

        def _fold_row(name, results):
            lls = [r["log_loss"] for r in results]
            avg = float(np.mean(lls))
            return f"| {name} | {avg:.4f} | {lls[0]:.4f} | {lls[1]:.4f} | {lls[2]:.4f} |\n"

        f.write(_fold_row("Platt (incumbent)", platt_results))
        f.write(_fold_row("Market (raw)", market_results))
        f.write(_fold_row("Market + Platt", market_platt_results))
        f.write(_fold_row("Elo + Market", elo_market_results))
        f.write("\n")

        f.write("## 2025 Holdout Comparison\n\n")
        f.write("| Model | Hold LL | Brier | Acc | AUC |\n")
        f.write("|-------|---------|-------|-----|-----|\n")

        random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
        prior_rate = train_y_full.mean()
        prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")

        def _row(name, m):
            ll_val = m["log_loss"]
            bs = m["brier_score"]
            acc = m["accuracy"]
            auc = m.get("roc_auc")
            auc_str = f"{auc:.4f}" if auc is not None else "—"
            return f"| {name} | {ll_val:.4f} | {bs:.4f} | {acc:.4f} | {auc_str} |\n"

        f.write(_row("Platt (incumbent)", hold_platt_m))
        f.write(_row("Market (raw)", hold_market_m))
        f.write(_row("Market + Platt", hold_market_platt_m))
        f.write(_row("Elo + Market", hold_elo_market_m))
        f.write("\n")

        # Calibration
        for label, h_met in [
            ("Market (Raw, Holdout)", hold_market_m),
            ("Market + Platt (Holdout)", hold_market_platt_m),
            ("Platt Incumbent (Holdout)", hold_platt_m),
            ("Elo + Market (Holdout)", hold_elo_market_m),
        ]:
            f.write(f"## {label}\n\n")
            f.write("| Bucket | Count | Mean Pred | Mean Actual | Cal Error |\n")
            f.write("|--------|-------|-----------|-------------|-----------|\n")
            for b, vals in sorted(h_met["calibration_buckets"].items()):
                mp = vals["mean_predicted_prob"]
                ma = vals["mean_actual_rate"]
                ce = vals["calibration_error"]
                f.write(f"| {b} | {vals['count']} | {mp} | {ma} | {ce} |\n")
            f.write("\n")

        # Recommendation
        f.write("## Recommendation\n\n")

        hold_best = min(
            [
                ("Market (raw)", hold_market_m["log_loss"]),
                ("Market + Platt", hold_market_platt_m["log_loss"]),
                ("Elo + Market", hold_elo_market_m["log_loss"]),
            ],
            key=lambda x: x[1],
        )

        beats_incumbent = hold_best[1] < INCUMBENT_HOLDOUT_LL
        elo_adds_value = hold_elo_market_m["log_loss"] < hold_market_m["log_loss"]

        if beats_incumbent:
            f.write(f"✅ **{hold_best[0]} beats the incumbent.**\n\n")
            f.write(
                f"Holdout log loss {hold_best[1]:.4f} vs incumbent {INCUMBENT_HOLDOUT_LL:.4f}.\n\n"
            )
        else:
            f.write("⚠️ **No market-based model beat the incumbent on holdout.**\n\n")
            f.write(f"Best market model: {hold_best[0]} (hold LL={hold_best[1]:.4f}) ")
            f.write(f"vs incumbent (hold LL={INCUMBENT_HOLDOUT_LL:.4f}).\n\n")

        if elo_adds_value:
            f.write("✅ **Elo adds independent information beyond market odds.**\n")
            f.write(f"Elo + Market (hold LL={hold_elo_market_m['log_loss']:.4f}) ")
            f.write(f"beats Market alone ({hold_market_m['log_loss']:.4f}).\n\n")
        else:
            f.write("⚠️ **Elo does not add information beyond market odds.**\n")
            f.write(f"Elo + Market (hold LL={hold_elo_market_m['log_loss']:.4f}) ")
            f.write(f"does not beat Market alone ({hold_market_m['log_loss']:.4f}).\n\n")

        # Favorite-longshot bias
        f.write("### Favorite-Longshot Bias\n\n")
        f.write("Market + Platt calibration result indicates whether the market ")
        f.write("has systematic favorite-longshot bias:\n")
        if hold_market_platt_m["log_loss"] < hold_market_m["log_loss"]:
            f.write("- ✅ Platt calibration improves market: favorite-longshot bias detected.\n")
        else:
            f.write(
                "- ⚠️ Platt calibration does not improve market: no strong favorite-longshot bias.\n"
            )

        f.write("\n### Next Recommended Experiment\n\n")
        f.write("1. Residual diagnostics — where does the incumbent fail systematically?\n")
        f.write("2. DVOA/EPA features if available.\n")
        f.write("3. Expand Elo K > 48 if needed.\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
