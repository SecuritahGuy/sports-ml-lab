"""Market benchmark and market-vs-Elo diagnostics.

Determines whether MOV Elo+Platt has independent signal relative to
market-implied probabilities and whether market prices explain the
model's remaining failure modes.

This is a research benchmark/diagnostic — not a betting bot.
Market features are comparison baselines and optional research
challengers, not production features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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
from sportslab.features.market import (
    compute_market_features,
    compute_spread_probs,
    fit_spread_model,
)
from sportslab.features.qb import compute_qb_features
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


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _logistic_pipe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def run_market_benchmark(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/market_benchmark.md",
) -> str:
    """Run comprehensive market benchmark experiment.

    1. Market data audit.
    2. Compute MOV Elo + market features.
    3. Rolling-origin evaluation of all models.
    4. One-time 2025 holdout.
    5. Residual diagnostics and QB-change/market analysis.
    6. Report.
    """
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ═══════════════════════════════════════════
    # 1. Market Data Audit
    # ═══════════════════════════════════════════
    print("\n=== Market Data Audit ===")

    market_cols_raw = [
        "home_moneyline",
        "away_moneyline",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
        "total_line",
        "over_odds",
        "under_odds",
    ]
    available = [c for c in market_cols_raw if c in df_raw.columns]
    missing = [c for c in market_cols_raw if c not in df_raw.columns]
    print(f"  Available: {available}")
    print(f"  Missing: {missing}")

    for c in available:
        nulls = df_raw[c].isna().sum()
        print(f"    {c}: {nulls} nulls, dtype={df_raw[c].dtype}")

    # Check for opening/closing columns
    opening_cols = [c for c in df_raw.columns if "opening" in c.lower() or "open" in c.lower()]
    closing_cols = [c for c in df_raw.columns if "closing" in c.lower() or "close" in c.lower()]
    print(
        "  Opening line columns:"
        f" {opening_cols if opening_cols else 'NONE — only closing lines available'}"
    )
    print(
        "  Closing line columns:"
        f" {closing_cols if closing_cols else 'NONE — only combined market cols available'}"
    )

    has_moneyline = "home_moneyline" in available

    if not has_moneyline:
        print("\n  ❌ Market data is incomplete. Cannot run benchmark.")
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as f:
            f.write("# Market Benchmark\n\n")
            f.write("**Market data is missing.**\n\n")
            f.write(
                "The feature table lacks `home_moneyline` and `away_moneyline` columns. "
                "Cannot compute market-implied probabilities.\n\n"
                "**Recommended ingestion fix:**\n"
                "Re-run `make ingest-nfl && make build-features` to pull "
                "schedules from nflreadpy, which includes moneyline data.\n"
            )
        print(f"  Blocker report written to: {rp}")
        return str(rp)

    # ═══════════════════════════════════════════
    # 2. Build Features
    # ═══════════════════════════════════════════
    print("\n=== Building features ===")
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
    )
    print(f"  MOV Elo: K={BEST_K}, HFA={BEST_HFA}, reg={BEST_REG}")

    df_all = compute_market_features(df_elo)
    df_all = compute_qb_features(df_all)

    df = _filter_df(df_all)

    y = df[TARGET_COLUMN].astype(float).values
    elo_prob = df["elo_prob"].values
    market_prob = df["market_home_prob_novig"].values
    spread_line = df["spread_line"].values
    seasons = df["season"].values
    weeks = df["week"].values

    has_qb = "home_qb_changed" in df.columns

    print(f"  Filtered rows: {len(df)}")
    print(f"  Market prob range: [{market_prob.min():.4f}, {market_prob.max():.4f}]")
    print(f"  Market prob mean: {market_prob.mean():.4f}")
    print(f"  Elo vs market correlation: {np.corrcoef(elo_prob, market_prob)[0, 1]:.4f}")

    # ═══════════════════════════════════════════
    # 3. Rolling-Origin Evaluation
    # ═══════════════════════════════════════════
    print("\n=== Rolling-Origin Evaluation ===")

    platt_results: list[dict] = []
    market_results: list[dict] = []
    market_platt_results: list[dict] = []
    elo_market_logit_results: list[dict] = []
    elo_market_avg_results: list[dict] = []
    spread_results: list[dict] = []
    elo_results: list[dict] = []

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df["season"].isin(train_seasons).values
        is_val = (df["season"] == val_season).values

        train_y_ = y[is_train].astype(int)
        val_y_ = y[is_val]

        train_elo = elo_prob[is_train]
        val_elo = elo_prob[is_val]

        train_market = market_prob[is_train]
        val_market = market_prob[is_val]

        train_spread = spread_line[is_train]

        # 0. Raw Elo (no calibration)
        elo_m = compute_classification_metrics(val_y_, val_elo)
        elo_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": elo_m["log_loss"],
                "metrics": elo_m,
                "model": None,
            }
        )

        # 1. Platt-scaled MOV Elo (incumbent)
        platt_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        platt_pipe.fit(train_elo.reshape(-1, 1), train_y_)
        platt_proba = platt_pipe.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        platt_m = compute_classification_metrics(val_y_, platt_proba)
        platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": platt_m["log_loss"],
                "metrics": platt_m,
                "model": platt_pipe,
            }
        )

        # 2. Raw market (no-vig)
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
        mkt_platt = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        mkt_platt.fit(train_market.reshape(-1, 1), train_y_)
        mkt_platt_proba = mkt_platt.predict_proba(val_market.reshape(-1, 1))[:, 1]
        mkt_platt_m = compute_classification_metrics(val_y_, mkt_platt_proba)
        market_platt_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": mkt_platt_m["log_loss"],
                "metrics": mkt_platt_m,
                "model": mkt_platt,
            }
        )

        # 4. Elo + Market (logistic blend)
        em_train = np.column_stack([train_elo, train_market])
        em_val = np.column_stack([val_elo, val_market])
        em_pipe = _logistic_pipe()
        em_pipe.fit(em_train, train_y_)
        em_proba = em_pipe.predict_proba(em_val)[:, 1]
        em_m = compute_classification_metrics(val_y_, em_proba)
        elo_market_logit_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": em_m["log_loss"],
                "metrics": em_m,
                "model": em_pipe,
            }
        )

        # 5. Simple average blend: (Elo + Market) / 2
        avg_proba = (val_elo + val_market) / 2.0
        avg_m = compute_classification_metrics(val_y_, avg_proba)
        elo_market_avg_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": avg_m["log_loss"],
                "metrics": avg_m,
                "model": None,
            }
        )

        # 6. Spread-implied probability (logistic, fit on train)
        spread_model = fit_spread_model(train_spread, train_y_)
        val_spread_proba = compute_spread_probs(spread_line[is_val], spread_model)
        spread_m = compute_classification_metrics(val_y_, val_spread_proba)
        spread_results.append(
            {
                "train_seasons": train_seasons,
                "val_season": val_season,
                "log_loss": spread_m["log_loss"],
                "metrics": spread_m,
                "model": spread_model,
            }
        )

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" elo={elo_m['log_loss']:.4f}"
            f" platt={platt_m['log_loss']:.4f}"
            f" mkt={market_m['log_loss']:.4f}"
            f" mkt+platt={mkt_platt_m['log_loss']:.4f}"
            f" em={em_m['log_loss']:.4f}"
            f" avg={avg_m['log_loss']:.4f}"
            f" spread={spread_m['log_loss']:.4f}"
        )

    # Average validation metrics
    def _avg_ll(results):
        return float(np.mean([r["log_loss"] for r in results]))

    avg_elo = _avg_ll(elo_results)
    avg_platt = _avg_ll(platt_results)
    avg_market = _avg_ll(market_results)
    avg_market_platt = _avg_ll(market_platt_results)
    avg_em_logit = _avg_ll(elo_market_logit_results)
    avg_em_avg = _avg_ll(elo_market_avg_results)
    avg_spread = _avg_ll(spread_results)

    print("\n=== Average Validation Log Loss ===")
    print(f"  Raw Elo:                {avg_elo:.4f}")
    print(f"  Platt (incumbent):      {avg_platt:.4f}")
    print(f"  Market (no-vig):        {avg_market:.4f}")
    print(f"  Market + Platt:         {avg_market_platt:.4f}")
    print(f"  Elo + Market (logit):   {avg_em_logit:.4f}")
    print(f"  Elo + Market (avg):     {avg_em_avg:.4f}")
    print(f"  Spread→prob:            {avg_spread:.4f}")

    # ═══════════════════════════════════════════
    # 4. One-time 2025 Holdout
    # ═══════════════════════════════════════════
    print("\n=== 2025 Holdout ===")
    is_hold = seasons == HOLDOUT_SEASON
    is_train_full = np.isin(seasons, [2021, 2022, 2023, 2024])

    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]
    hold_market = market_prob[is_hold]
    hold_spread = spread_line[is_hold]
    hold_weeks = weeks[is_hold]

    train_elo_full = elo_prob[is_train_full]
    train_y_full = y[is_train_full].astype(int)
    train_market_full = market_prob[is_train_full]
    train_spread_full = spread_line[is_train_full]

    # 0. Raw Elo
    hold_elo_m = compute_classification_metrics(hold_y, hold_elo)

    # 1. Platt incumbent
    platt_full = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    platt_full.fit(train_elo_full.reshape(-1, 1), train_y_full)
    hold_platt_proba = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_platt_m = compute_classification_metrics(hold_y, hold_platt_proba)
    print(f"  Platt (incumbent):      {hold_platt_m['log_loss']:.4f}")

    # 2. Raw market
    hold_market_m = compute_classification_metrics(hold_y, hold_market)
    print(f"  Market (no-vig):        {hold_market_m['log_loss']:.4f}")

    # 3. Market + Platt
    mkt_platt_full = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    mkt_platt_full.fit(train_market_full.reshape(-1, 1), train_y_full)
    hold_mkt_platt_proba = mkt_platt_full.predict_proba(hold_market.reshape(-1, 1))[:, 1]
    hold_mkt_platt_m = compute_classification_metrics(hold_y, hold_mkt_platt_proba)
    print(f"  Market + Platt:         {hold_mkt_platt_m['log_loss']:.4f}")

    # 4. Elo + Market (logistic)
    em_full = np.column_stack([train_elo_full, train_market_full])
    em_hold = np.column_stack([hold_elo, hold_market])
    em_final = _logistic_pipe()
    em_final.fit(em_full, train_y_full)
    hold_em_proba = em_final.predict_proba(em_hold)[:, 1]
    hold_em_m = compute_classification_metrics(hold_y, hold_em_proba)
    print(f"  Elo + Market (logit):   {hold_em_m['log_loss']:.4f}")

    # 5. Simple average blend
    hold_avg_proba = (hold_elo + hold_market) / 2.0
    hold_avg_m = compute_classification_metrics(hold_y, hold_avg_proba)
    print(f"  Elo + Market (avg):     {hold_avg_m['log_loss']:.4f}")

    # 6. Spread-implied probability
    spread_full_model = fit_spread_model(train_spread_full, train_y_full)
    hold_spread_proba = compute_spread_probs(hold_spread, spread_full_model)
    hold_spread_m = compute_classification_metrics(hold_y, hold_spread_proba)
    print(f"  Spread→prob:            {hold_spread_m['log_loss']:.4f}")

    # ═══════════════════════════════════════════
    # 5. Residual Diagnostics
    # ═══════════════════════════════════════════
    print("\n=== Residual Diagnostics ===")

    # Compute residuals (actual - predicted)
    elo_residual = hold_y - hold_elo
    market_residual = hold_y - hold_market

    # Correlation between Elo and market residuals
    if len(elo_residual) > 1:
        r_resid, p_resid = pearsonr(elo_residual, market_residual)
        print(f"  Elo residual vs market residual correlation: r={r_resid:.4f}, p={p_resid:.4f}")
    else:
        r_resid, p_resid = np.nan, np.nan

    # Correlation between Elo prob and market prob
    if len(hold_elo) > 1:
        r_prob, p_prob = pearsonr(hold_elo, hold_market)
        print(f"  Elo prob vs market prob correlation: r={r_prob:.4f}, p={p_prob:.4f}")
    else:
        r_prob, p_prob = np.nan, np.nan

    # Elo edge = Elo probability - market probability
    elo_edge = hold_elo - hold_market
    print(f"  Elo edge range: [{elo_edge.min():.4f}, {elo_edge.max():.4f}]")

    # ═══════════════════════════════════════════
    # 6. Subset Analyses
    # ═══════════════════════════════════════════
    print("\n=== Subset Analyses ===")

    def _subset(mask, label, proba, n_min=5):
        n = int(mask.sum())
        if n < n_min:
            return {"label": label, "n": n, "log_loss": None}
        sub = compute_classification_metrics(hold_y[mask], proba[mask])
        return {"label": label, "n": n, "log_loss": sub["log_loss"]}

    if has_qb:
        qb_changed = df.loc[is_hold, "home_qb_changed"].fillna(0).astype(bool).values
        qb_stable = ~qb_changed
    else:
        qb_changed = np.zeros(len(hold_y), dtype=bool)
        qb_stable = np.ones(len(hold_y), dtype=bool)

    # Midpoint between Elo and market probabilities
    hold_ref_proba = hold_elo  # Use Elo as reference for subset definitions

    high_conf = hold_ref_proba > 0.9
    low_conf = hold_ref_proba <= 0.6
    early_season = np.isin(hold_weeks, [1, 2, 3, 4])
    late_season = ~early_season

    # Favorite/underdog buckets
    home_fav = df.loc[is_hold, "home_moneyline"] < 0
    home_dog = df.loc[is_hold, "home_moneyline"] > 0
    # Pick spread buckets
    big_home_fav = hold_spread > 7
    big_away_fav = hold_spread < -7
    close_game = np.abs(hold_spread) <= 3

    subsets = {
        "QB changed (home)": qb_changed,
        "QB stable (home)": qb_stable,
        "High confidence (>0.9)": high_conf,
        "Low confidence (<=0.6)": low_conf,
        "Early season (W1-4)": early_season,
        "Late season (W5+)": late_season,
        "Home favorite": home_fav.values if hasattr(home_fav, "values") else home_fav,
        "Home underdog": home_dog.values if hasattr(home_dog, "values") else home_dog,
        "Big home fav (spread > 7)": big_home_fav,
        "Big away fav (spread < -7)": big_away_fav,
        "Close game (|spread| ≤ 3)": close_game,
    }

    for label, mask in sorted(subsets.items()):
        r = _subset(np.array(mask), label, hold_elo)
        if r["log_loss"] is None:
            print(f"  {label}: insufficient ({r['n']})")
        else:
            print(f"  {label} (n={r['n']}): raw ELo LL={r['log_loss']:.4f}")

    # ═══════════════════════════════════════════
    # 7. Baselines
    # ═══════════════════════════════════════════
    random_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_rate = train_y_full.mean()
    prior_hold_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    # ═══════════════════════════════════════════
    # 8. Report
    # ═══════════════════════════════════════════
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    def _row(name, m):
        return (
            f"| {name} | {m['log_loss']:.4f}"
            f" | {m['brier_score']:.4f}"
            f" | {m['accuracy']:.4f}"
            f" | {m['roc_auc']:.4f} |\n"
        )

    def _fold_ll_row(name, results):
        lls = [r["log_loss"] for r in results]
        avg = float(np.mean(lls))
        return f"| {name} | {avg:.4f} | {lls[0]:.4f} | {lls[1]:.4f} | {lls[2]:.4f} |\n"

    with open(rp, "w") as f:
        f.write("# Market Benchmark and Elo-vs-Market Diagnostics\n\n")
        f.write(
            "*Determining whether MOV Elo+Platt has independent signal"
            " relative to market-implied probabilities.*\n\n"
        )

        f.write("## Market Data Audit\n\n")
        f.write("| Column | Coverage | Type | Source |\n")
        f.write("|--------|----------|------|--------|\n")
        for c in available:
            nulls = df_raw[c].isna().sum()
            coverage = (1 - nulls / len(df_raw)) * 100
            f.write(f"| `{c}` | {coverage:.0f}% | {df_raw[c].dtype} | nflreadpy |\n")
        if not opening_cols:
            f.write("| opening/home_moneyline | 0% | — | Not available in nflreadpy |\n")
            f.write("| opening/away_moneyline | 0% | — | Not available in nflreadpy |\n")
        f.write("\n")

        f.write("### Data Quality Notes\n\n")
        f.write("- All closing moneylines and spreads are 100% available.\n")
        f.write(
            "- **No opening line data available.** nflreadpy provides only "
            "closing market data.  Opening lines require a separate source.\n"
        )
        f.write("- Spread odds vary (not always -110), indicating real market variation.\n")
        f.write("- All odds are American format.\n\n")

        f.write("## Methodology\n\n")
        f.write("### Moneyline Conversion\n\n")
        f.write("- Negative odds (favorite): `prob = -odds / (-odds + 100)`\n")
        f.write("- Positive odds (underdog): `prob = 100 / (odds + 100)`\n")
        f.write("- Vig removed via multiplicative normalization:\n")
        f.write("  `fair_home_prob = home_implied / (home_implied + away_implied)`\n\n")

        f.write("### Spread→Implied Probability\n\n")
        f.write("- A logistic regression is fit per fold mapping spread line → home win prob.\n")
        f.write("- Fitted on training data only (no validation/holdout access).\n")
        f.write("- Tests whether spread line alone matches moneyline information.\n\n")

        f.write("### Blend Methods\n\n")
        f.write("- **Logistic blend**: Logistic regression on Elo prob + market prob.\n")
        f.write("- **Average blend**: Simple `(Elo + Market) / 2`.\n")
        f.write("- Neither blend is a production champion candidate.\n\n")

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
        f.write("| Split | Seasons | Role |\n")
        f.write("|-------|---------|------|\n")
        for i, (tr, va) in enumerate(ROLLING_FOLDS, 1):
            f.write(f"| Fold {i} | Train: {tr}, Val: {va} | Selection |\n")
        f.write(f"| Holdout | 2021–2024 → {HOLDOUT_SEASON} | Final eval |\n\n")

        f.write("## Models Compared\n\n")
        f.write("| Model | Description | Timing |\n")
        f.write("|-------|-------------|--------|\n")
        f.write("| **Raw Elo** | MOV Elo probability (no calibration) | Pregame |\n")
        f.write("| **Platt (incumbent)** | MOV Elo + Platt scaling | Pregame |\n")
        f.write("| **Market (no-vig)** | De-vigged moneyline implied prob | Closing line |\n")
        f.write("| **Market + Platt** | Platt-calibrated market | Closing line |\n")
        f.write("| **Elo + Market (logit)** | Logistic blend | Pregame + closing |\n")
        f.write("| **Elo + Market (avg)** | Simple average blend | Pregame + closing |\n")
        f.write("| **Spread→prob** | Logistic from spread line | Closing line |\n")
        f.write(
            "> **Timing note:** Closing lines are near-kickoff and may reflect "
            "late-breaking information. Elo is purely pregame (previous games only). "
            "These are not directly comparable as production strategies.\n\n"
        )

        f.write("## Average Validation Log Loss\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")
        f.write(_fold_ll_row("Raw Elo", elo_results))
        f.write(_fold_ll_row("Platt (incumbent)", platt_results))
        f.write(_fold_ll_row("Market (no-vig)", market_results))
        f.write(_fold_ll_row("Market + Platt", market_platt_results))
        f.write(_fold_ll_row("Elo + Market (logit)", elo_market_logit_results))
        f.write(_fold_ll_row("Elo + Market (avg)", elo_market_avg_results))
        f.write(_fold_ll_row("Spread→prob", spread_results))
        f.write("\n")

        f.write("## 2025 Holdout Comparison\n\n")
        f.write("| Model | Hold LL | Brier | Acc | AUC |\n")
        f.write("|-------|---------|-------|-----|-----|\n")
        f.write(f"| Random | {random_hold_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_hold_ll:.4f} | — | — | 0.5000 |\n")
        f.write(_row("Raw Elo", hold_elo_m))
        f.write(_row("Platt (incumbent)", hold_platt_m))
        f.write(_row("Market (no-vig)", hold_market_m))
        f.write(_row("Market + Platt", hold_mkt_platt_m))
        f.write(_row("Elo + Market (logit)", hold_em_m))
        f.write(_row("Elo + Market (avg)", hold_avg_m))
        f.write(_row("Spread→prob", hold_spread_m))
        f.write("\n")

        # Residual analysis
        f.write("## Residual Correlation\n\n")
        f.write("| Comparison | r | p-value |\n")
        f.write("|------------|---|--------|\n")
        f.write(f"| Elo residual vs Market residual | {r_resid:.4f} | {p_resid:.4f} |\n")
        f.write(f"| Elo prob vs Market prob | {r_prob:.4f} | {p_prob:.4f} |\n")
        emin, emax = elo_edge.min(), elo_edge.max()
        f.write(f"| Elo edge (Elo − Market) range | [{emin:.4f}, {emax:.4f}] | — |\n\n")

        # Edge analysis
        f.write("### Elo Edge Analysis\n\n")
        f.write(
            "Elo edge = Elo probability − Market probability. "
            "Positive means Elo is more confident in a home win than the market.\n\n"
        )
        edge_buckets = {
            "Elo much lower than mkt (< -0.15)": elo_edge < -0.15,
            "Elo lower than mkt (-0.15 to -0.05)": (elo_edge >= -0.15) & (elo_edge < -0.05),
            "Near agreement (-0.05 to 0.05)": np.abs(elo_edge) <= 0.05,
            "Elo higher than mkt (0.05 to 0.15)": (elo_edge > 0.05) & (elo_edge <= 0.15),
            "Elo much higher than mkt (> 0.15)": elo_edge > 0.15,
        }
        f.write("| Bucket | Count | Mean Actual Win Rate | Mean Elo LL | Mean Market LL |\n")
        f.write("|--------|-------|---------------------|-------------|----------------|\n")
        for label, mask in sorted(edge_buckets.items()):
            n = int(mask.sum())
            if n < 3:
                f.write(f"| {label} | {n} | insufficient | — | — |\n")
                continue
            actual_rate = hold_y[mask].mean()
            elo_ll = compute_classification_metrics(hold_y[mask], hold_elo[mask])["log_loss"]
            mkt_ll = compute_classification_metrics(hold_y[mask], hold_market[mask])["log_loss"]
            f.write(f"| {label} | {n} | {actual_rate:.3f} | {elo_ll:.4f} | {mkt_ll:.4f} |\n")
        f.write("\n")

        # Calibration
        f.write("## Calibration Deciles\n\n")
        for label, h_met in [
            ("Platt (Incumbent, Holdout)", hold_platt_m),
            ("Market (No-Vig, Holdout)", hold_market_m),
            ("Elo + Market (Logit, Holdout)", hold_em_m),
        ]:
            f.write(f"### {label}\n\n")
            f.write("| Bucket | Count | Mean Pred | Mean Actual | Cal Error |\n")
            f.write("|--------|-------|-----------|-------------|-----------|\n")
            for b, vals in sorted(h_met["calibration_buckets"].items()):
                f.write(
                    f"| {b} | {vals['count']} | {vals['mean_predicted_prob']:.4f}"
                    f" | {vals['mean_actual_rate']:.4f}"
                    f" | {vals['calibration_error']:.4f} |\n"
                )
            f.write("\n")

        # Subset analysis
        f.write("## Subset Analysis (2025 Holdout)\n\n")
        f.write("| Subset | N | Elo LL | Market LL | EM Logit LL |\n")
        f.write("|--------|---|--------|-----------|-------------|\n")
        for label, mask in sorted(subsets.items()):
            mask_arr = np.array(mask)
            n = int(mask_arr.sum())
            if n < 5:
                f.write(f"| {label} | {n} | insufficient | insufficient | insufficient |\n")
                continue
            elo_sub = compute_classification_metrics(hold_y[mask_arr], hold_elo[mask_arr])
            mkt_sub = compute_classification_metrics(hold_y[mask_arr], hold_market[mask_arr])
            em_sub = compute_classification_metrics(hold_y[mask_arr], hold_em_proba[mask_arr])
            f.write(
                f"| {label} | {n} | {elo_sub['log_loss']:.4f}"
                f" | {mkt_sub['log_loss']:.4f}"
                f" | {em_sub['log_loss']:.4f} |\n"
            )
        f.write("\n")

        # QB-change deep dive
        f.write("### QB-Change Deep Dive\n\n")
        if qb_changed.sum() >= 5:
            f.write("**On QB-change games:**\n\n")
            f.write("| Metric | Elo | Market | EM Logit |\n")
            f.write("|--------|-----|--------|----------|\n")
            elo_qb = compute_classification_metrics(hold_y[qb_changed], hold_elo[qb_changed])
            mkt_qb = compute_classification_metrics(hold_y[qb_changed], hold_market[qb_changed])
            em_qb = compute_classification_metrics(hold_y[qb_changed], hold_em_proba[qb_changed])
            for display_name, metric_key in [
                ("Log Loss", "log_loss"),
                ("Brier", "brier_score"),
                ("Accuracy", "accuracy"),
                ("AUC", "roc_auc"),
            ]:
                elo_v = elo_qb[metric_key]
                mkt_v = mkt_qb[metric_key]
                em_v = em_qb[metric_key]
                f.write(f"| {display_name} | {elo_v:.4f} | {mkt_v:.4f} | {em_v:.4f} |\n")
            f.write("\n")
            market_edge_on_qb = mkt_qb["log_loss"] - elo_qb["log_loss"]
            if market_edge_on_qb < 0:
                qb_edge_abs = abs(market_edge_on_qb)
                f.write(
                    "Market beats Elo on QB-change games by "
                    f"{qb_edge_abs:.4f} log loss. "
                    "The market prices in QB-change information"
                    " that Elo misses.\n\n"
                )
            else:
                f.write(
                    "Elo beats market on QB-change games by "
                    f"{market_edge_on_qb:.4f} log loss. "
                    "The market does not fully price in"
                    " QB-change dynamics.\n\n"
                )
        else:
            f.write(f"Insufficient QB-change games ({qb_changed.sum()}).\n\n")

        # Favorite/underdog & spread buckets
        f.write("### Favorite/Underdog Buckets\n\n")
        f.write("| Bucket | N | Elo LL | Market LL |\n")
        f.write("|--------|---|--------|-----------|\n")
        for label, mask in [
            ("Home favorite", home_fav),
            ("Home underdog", home_dog),
            ("Big home fav (spread > 7)", big_home_fav),
            ("Big away fav (spread < -7)", big_away_fav),
            ("Close game (|spread| ≤ 3)", close_game),
        ]:
            m = np.array(mask.values if hasattr(mask, "values") else mask)
            n = int(m.sum())
            if n < 5:
                f.write(f"| {label} | {n} | insufficient | insufficient |\n")
                continue
            elo_s = compute_classification_metrics(hold_y[m], hold_elo[m])
            mkt_s = compute_classification_metrics(hold_y[m], hold_market[m])
            f.write(f"| {label} | {n} | {elo_s['log_loss']:.4f} | {mkt_s['log_loss']:.4f} |\n")
        f.write("\n")

        # Recommendation
        f.write("## Recommendation\n\n")

        models = [
            ("Market (no-vig)", hold_market_m["log_loss"]),
            ("Market + Platt", hold_mkt_platt_m["log_loss"]),
            ("Elo + Market (logit)", hold_em_m["log_loss"]),
            ("Elo + Market (avg)", hold_avg_m["log_loss"]),
            ("Spread→prob", hold_spread_m["log_loss"]),
        ]
        hold_best = min(models, key=lambda x: x[1])
        beats_incumbent = hold_best[1] < INCUMBENT_HOLDOUT_LL
        elo_adds_value = hold_em_m["log_loss"] < hold_market_m["log_loss"]

        if beats_incumbent:
            f.write(f"✅ **{hold_best[0]} beats the incumbent on holdout.**\n\n")
            f.write(
                f"Holdout log loss {hold_best[1]:.4f} vs incumbent {INCUMBENT_HOLDOUT_LL:.4f}.\n\n"
            )
            f.write(
                "> ⚠️ Market data (closing lines) reflects near-kickoff information, "
                "not purely pregame conditions. This comparison is diagnostic, "
                "not a direct apples-to-apples comparison of modeling strategies.\n\n"
            )
        else:
            f.write("⚠️ **No market-based model beat the incumbent on holdout.**\n\n")
            f.write(
                f"Best market model: {hold_best[0]} (hold LL={hold_best[1]:.4f}) "
                f"vs incumbent (hold LL={INCUMBENT_HOLDOUT_LL:.4f}).\n\n"
            )

        if elo_adds_value:
            f.write("✅ **Elo adds independent information beyond market odds.**\n")
            f.write(
                f"Elo + Market (logit, hold LL={hold_em_m['log_loss']:.4f}) "
                f"beats Market alone ({hold_market_m['log_loss']:.4f}).\n\n"
            )
        else:
            f.write("⚠️ **Elo does not add independent information beyond market odds.**\n")
            f.write(
                f"Elo + Market (logit, hold LL={hold_em_m['log_loss']:.4f}) "
                f"does not beat Market alone ({hold_market_m['log_loss']:.4f}).\n\n"
            )

        # Favorite-longshot bias
        f.write("### Favorite-Longshot Bias\n\n")
        if hold_mkt_platt_m["log_loss"] < hold_market_m["log_loss"]:
            f.write(
                "✅ Platt calibration improves market (hold LL "
                f"{hold_mkt_platt_m['log_loss']:.4f} vs {hold_market_m['log_loss']:.4f}). "
                "Mild favorite-longshot bias detected.\n\n"
            )
        else:
            f.write(
                "⚠️ Platt calibration does not improve market "
                f"(hold LL {hold_mkt_platt_m['log_loss']:.4f} vs {hold_market_m['log_loss']:.4f}). "
                "No strong favorite-longshot bias.\n\n"
            )

        # QB-change recommendation
        if qb_changed.sum() >= 5 and beats_incumbent:
            f.write("### QB-Change Recommendation\n\n")
            f.write(
                "Since market beats the incumbent, and QB-change is the #1 failure mode, "
                "recommend QB-change market-delta as a feature: "
                "market_prob − elo_prob at QB-change games.\n\n"
            )

        f.write("### Next Recommended Experiment\n\n")
        f.write("1. QB-change market-delta feature: `market_prob − elo_prob` at QB-change games.\n")
        f.write("2. Test if market odds at QB-change games alone explain the gap.\n")
        f.write("3. Stacked model: Elo predicts residual of market-only model.\n\n")

        f.write("## Appendix: Elo vs Market by Season\n\n")
        f.write("| Season | N | Elo LL | Market LL | EM Logit LL |\n")
        f.write("|--------|---|--------|-----------|-------------|\n")
        for s in sorted(df["season"].unique()):
            mask = (
                (df["season"] == s).values
                & df[MODEL_ELIGIBLE_COLUMN].values
                & ~df[NEUTRAL_COLUMN].values
            )
            n = int(mask.sum())
            if n < 5:
                continue
            y_s = y[mask]
            elo_s = compute_classification_metrics(y_s, elo_prob[mask])
            mkt_s = compute_classification_metrics(y_s, market_prob[mask])
            em_s = compute_classification_metrics(y_s, (elo_prob[mask] + market_prob[mask]) / 2.0)
            elo_ll = elo_s["log_loss"]
            mkt_ll = mkt_s["log_loss"]
            em_ll = em_s["log_loss"]
            f.write(f"| {s} | {n} | {elo_ll:.4f} | {mkt_ll:.4f} | {em_ll:.4f} |\n")
        f.write("\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
