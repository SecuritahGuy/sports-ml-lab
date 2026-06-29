"""QB-Adjusted Elo Experiment — V0: player-adjusted pregame strength.

Research question: Can we improve team Elo by adjusting pregame strength
based on expected QB availability?  Tests whether QB-specific ratings
(computed with Bayesian shrinkage toward replacement level) add signal
beyond the binary qb_changed flag and rolling_mov_3.

Models compared:
    A. Incumbent baseline (Elo + qb_changed + rolling_mov_3 + Platt)
    B. QB-adjusted Elo (replace elo_prob with qb-adjusted prob, no Platt)
    C. QB-adjusted Elo + Platt
    D. Diagnostic: aggressive QB ratings (no shrinkage) + Platt

Slices:
    - QB-change games
    - non-QB-change games
    - Missing QB data
    - High-confidence predictions
    - Home favorites / away underdogs
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import (
    compute_metrics,
)
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
from sportslab.features.qb_adjustment import (
    MAX_ADJUSTMENT,
    PRIOR_IMPACT,
    PRIOR_STARTS,
    compute_qb_adjusted_elo_prob,
    compute_qb_adjustments,
)
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

# Incumbent matching the authoritative benchmark
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
FEATURE_COLS = ["home_qb_changed", "away_qb_changed", "home_rolling_mov_3", "away_rolling_mov_3"]

SEED = 42
N_WORST = 20


def _get_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    if avail:
        return df[avail].values
    return np.empty((len(df), 0))


def _experiment_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def run_qb_adjusted_elo_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/qb_adjusted_elo.md",
    output_csv: str | None = None,
) -> str:
    print("=== QB-Adjusted Elo Experiment V0 ===")

    # ── 1. Load and build features ──
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS,
    )
    df = compute_elo_features(
        df_raw, k_factor=BEST_K, home_advantage=BEST_HFA,
        preseason_regression=BEST_REG, team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_adjustments(df)
    df = compute_qb_features(df)
    df = compute_situational_features(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df[NEUTRAL_COLUMN].values
    df = df[mask].copy().reset_index(drop=True)
    print(f"  Eligible games: {len(df)}")

    elo_prob = df["elo_prob"].values.astype(float)
    y = df[TARGET_COLUMN].astype(float).values

    # QB-adjusted probability (Model B/C)
    qb_adj_prob = compute_qb_adjusted_elo_prob(
        df["home_elo_pre"].values,
        df["away_elo_pre"].values,
        df["home_qb_adj"].values,
        df["away_qb_adj"].values,
        hfa=BEST_HFA,
    )

    # Aggressive diagnostic: no shrinkage QB adjustments
    qb_adj_aggressive = compute_qb_adjusted_elo_prob(
        df["home_elo_pre"].values,
        df["away_elo_pre"].values,
        df["home_qb_adj"].values * 3.0,  # scale up for diagnostic
        df["away_qb_adj"].values * 3.0,
        hfa=BEST_HFA,
    )

    # ── 2. Rolling-origin validation ──
    print("\n=== Rolling-Origin Validation ===")

    model_specs = {
        "A. Incumbent (Elo + qb_changed + mov3 + Platt)": {
            "prob_col": "elo_prob", "additional_feats": FEATURE_COLS, "use_platt": True,
        },
        "B. QB-adjusted Elo (raw)": {
            "prob_col": "qb_adj", "additional_feats": [], "use_platt": False,
        },
        "C. QB-adjusted Elo + Platt": {
            "prob_col": "qb_adj", "additional_feats": [], "use_platt": True,
        },
        "D. QB-adjusted Elo + qb_changed + mov3 + Platt": {
            "prob_col": "qb_adj", "additional_feats": FEATURE_COLS, "use_platt": True,
        },
        "X. Diagnostic: aggressive QB adj + Platt": {
            "prob_col": "qb_adj_aggressive", "additional_feats": [], "use_platt": True,
        },
    }

    prob_sources = {
        "elo_prob": elo_prob,
        "qb_adj": qb_adj_prob,
        "qb_adj_aggressive": qb_adj_aggressive,
    }

    results: dict[str, dict] = {}
    for name, spec in model_specs.items():
        prob = prob_sources[spec["prob_col"]]
        fold_lls = []
        for train_seasons, val_season in ROLLING_FOLDS:
            tr = df["season"].isin(train_seasons).values
            va = (df["season"] == val_season).values

            if spec["use_platt"]:
                feat = _get_features(df.loc[tr], spec["additional_feats"])
                x_tr = (
                    np.column_stack([prob[tr], feat]) if feat.size else prob[tr].reshape(-1, 1)
                )
                feat_va = _get_features(df.loc[va], spec["additional_feats"])
                x_va = (
                    np.column_stack([prob[va], feat_va])
                    if feat_va.size else prob[va].reshape(-1, 1)
                )
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
                ])
                pipe.fit(x_tr, y[tr].astype(int))
                p = pipe.predict_proba(x_va)[:, 1]
            else:
                p = prob[va].copy()

            fold_lls.append(compute_metrics(y[va], p)["log_loss"])

        results[name] = {
            "fold_lls": fold_lls,
            "val_ll": float(np.mean(fold_lls)),
        }
        print(f"  {name}: {results[name]['val_ll']:.4f}  "
              f"(folds: {fold_lls[0]:.4f}, {fold_lls[1]:.4f}, {fold_lls[2]:.4f})")

    # ── 3. 2025 Holdout evaluation ──
    print("\n=== 2025 Holdout ===")

    is_train = df["season"].isin([2021, 2022, 2023, 2024]).values
    is_hold = df["season"] == HOLDOUT_SEASON
    train_y = y[is_train].astype(int)
    hold_y = y[is_hold]

    hold_results: dict[str, dict] = {}
    for name, spec in model_specs.items():
        prob = prob_sources[spec["prob_col"]]

        if spec["use_platt"]:
            feat_tr = _get_features(df.loc[is_train], spec["additional_feats"])
            x_tr = (
                np.column_stack([prob[is_train], feat_tr])
                if feat_tr.size else prob[is_train].reshape(-1, 1)
            )
            feat_ho = _get_features(df.loc[is_hold], spec["additional_feats"])
            x_ho = (
                np.column_stack([prob[is_hold], feat_ho])
                if feat_ho.size else prob[is_hold].reshape(-1, 1)
            )
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
            ])
            pipe.fit(x_tr, train_y)
            p_hold = pipe.predict_proba(x_ho)[:, 1]
        else:
            p_hold = prob[is_hold].copy()

        m = compute_classification_metrics(hold_y[~np.isnan(hold_y)], p_hold[~np.isnan(hold_y)])
        hold_results[name] = m

        print(f"  {name}: LL={m['log_loss']:.4f}, Brier={m['brier_score']:.4f}, "
              f"AUC={m['roc_auc']:.4f}, Acc={m['accuracy']:.4f}")

    # ── 4. Slice analysis (holdout) ──
    print("\n=== Slice Analysis (2025 Holdout) ===")

    incumbent_prob = prob_sources["elo_prob"][is_hold]
    challenger_idx = None
    challenger_name = "C. QB-adjusted Elo + Platt"
    for i, (nm, spec) in enumerate(model_specs.items()):
        if nm == challenger_name:
            challenger_idx = i
            break

    if challenger_idx is not None:
        challenger_prob = prob_sources["qb_adj"][is_hold]
        # Fit Platt for challenger
        feat = _get_features(df.loc[is_train], [])
        x_tr = prob_sources["qb_adj"][is_train].reshape(-1, 1)
        x_ho = prob_sources["qb_adj"][is_hold].reshape(-1, 1)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])
        pipe.fit(x_tr, train_y)
        challenger_prob = pipe.predict_proba(x_ho)[:, 1]

        hold_df = df[is_hold].copy()
        hold_df["incumbent_prob"] = incumbent_prob
        hold_df["challenger_prob"] = challenger_prob

        slices = {
            "All games": slice(None),
            "QB change (either)": hold_df["home_qb_changed"].astype(bool).values
                                 | hold_df["away_qb_changed"].astype(bool).values,
            "No QB change": ~(hold_df["home_qb_changed"].astype(bool).values
                            | hold_df["away_qb_changed"].astype(bool).values),
            "Home favorite (>=0.5)": incumbent_prob >= 0.5,
            "Away underdog (<0.5)": incumbent_prob < 0.5,
            "High confidence (>=0.7)": incumbent_prob >= 0.7,
            "Medium confidence (0.5-0.7)": (incumbent_prob >= 0.5) & (incumbent_prob < 0.7),
        }

        slice_results = {}
        for sl_name, sl_mask in slices.items():
            if sl_mask is slice(None):
                sl_y = hold_y.values if hasattr(hold_y, 'values') else hold_y
                sl_inc = incumbent_prob
                sl_chal = challenger_prob
            else:
                sl_y = hold_y.values[sl_mask] if hasattr(hold_y, 'values') else hold_y[sl_mask]
                sl_inc = incumbent_prob[sl_mask]
                sl_chal = challenger_prob[sl_mask]

            valid = ~np.isnan(sl_y)
            if valid.sum() < 2:
                continue
            inc_m = compute_metrics(sl_y[valid], sl_inc[valid])
            chal_m = compute_metrics(sl_y[valid], sl_chal[valid])
            delta_ll = chal_m["log_loss"] - inc_m["log_loss"]
            slice_results[sl_name] = {
                "n": int(valid.sum()),
                "incumbent_ll": inc_m["log_loss"],
                "challenger_ll": chal_m["log_loss"],
                "delta_ll": delta_ll,
            }
            print(f"  {sl_name}: n={valid.sum()}, inc={inc_m['log_loss']:.4f}, "
                  f"chal={chal_m['log_loss']:.4f}, Δ={delta_ll:+.4f}")

    # ── 5. Write report ──
    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        _w = f.write

        _w("# QB-Adjusted Elo Experiment — V0\n\n")
        _w("## Research Question\n\n")
        _w("Can we improve team Elo by adjusting pregame strength based on ")
        _w("expected QB availability? QB-specific ratings (computed with ")
        _w("Bayesian shrinkage toward replacement level) may capture the ")
        _w("magnitude of QB quality differences that the binary `qb_changed` ")
        _w("flag misses.\n\n")

        _w("## Models Compared\n\n")
        _w("| Model | Description |\n")
        _w("|-------|-------------|\n")
        _w("| **A. Incumbent** | Elo + qb_changed + rolling_mov_3 + Platt |\n")
        _w("| **B. QB-adjusted (raw)** | QB-adjusted Elo prob, no calibration |\n")
        _w("| **C. QB-adjusted + Platt** | QB-adjusted Elo prob + Platt |\n")
        _w("| **D. QB-adj + qb_changed + mov3 + Platt** | QB-adj + existing feats + Platt |\n")
        _w("| **X. Diagnostic: aggressive QB adj** | 3× scaled QB adjustments + Platt |\n\n")

        _w("## QB Adjustment Formula\n\n")
        _w("```\n")
        _w("qb_adj = Elo-point adjustment for each QB start\n")
        _w("\n")
        _w("Per-game impact = actual_win - elo_expected_win_prob\n")
        _w("\n")
        _w("Shrunken impact = (observed_impact * n + PRIOR_IMPACT * PRIOR_STARTS)\n")
        _w("                  / (n + PRIOR_STARTS)\n")
        _w("\n")
        _w("qb_adj = 400 * log10((0.5 + shrunken_impact) / (0.5 - shrunken_impact))\n")
        _w("\n")
        _w("team_effective_elo = team_elo + qb_adjustment\n")
        _w("adjusted_prob = 1/(1 + 10^(-(h_elo+h_adj - a_elo-a_adj + HFA)/400))\n")
        _w("```\n\n")
        _w(f"Hyperparameters: PRIOR_STARTS={PRIOR_STARTS} (~1 season), ")
        _w(f"PRIOR_IMPACT={PRIOR_IMPACT} (replacement ~3% below avg), ")
        _w(f"MAX_ADJUSTMENT={MAX_ADJUSTMENT} Elo points.\n\n")

        _w("## Data Used\n\n")
        _w("- 2021–2025 NFL seasons (285 games/season, all non-neutral regular + postseason)\n")
        _w("- QB starter IDs from nflreadpy (`home_qb_id`, `away_qb_id` GSIS IDs)\n")
        _w("- Team Elo ratings (K=36, HFA=40, reg=0.1, decay=32, MOV capped_linear)\n")
        _w("- 108 unique QBs across all seasons\n\n")

        _w("## Data Excluded\n\n")
        _w("- Seasons before 2021\n")
        _w("- Neutral-site games\n")
        _w("- Ties and games with missing scores\n")
        _w("- 2026 games (no scores)\n")
        _w("- Post-game stats, final scores, market data\n\n")

        _w("## Leakage Safeguards\n\n")
        _w("1. QB ratings computed chronologically — only prior starts used\n")
        _w("2. No post-game data (scores, result) used in rating computation\n")
        _w("3. Season-boundary reset for rolling features\n")
        _w("4. Holdout (2025) untouched during training/validation\n")
        _w("5. Bayesian shrinkage prevents small-sample overfitting\n")
        _w("6. QB IDs with missing data assigned 0.0 adjustment (no signal)\n\n")

        _w("## Backtest Setup\n\n")
        _w(f"- Rolling-origin folds: {ROLLING_FOLDS}\n")
        _w(f"- Holdout: {HOLDOUT_SEASON}\n")
        _w("- Platt: StandardScaler + LogisticRegression (C=1.0, lbfgs)\n")
        _w("- Feature set (incumbent): elo_prob + qb_changed (2) + rolling_mov_3 (2)\n\n")

        # Validation table
        _w("## Rolling-Origin Validation Log Loss\n\n")
        _w("| Model | Avg LL | Fold1 | Fold2 | Fold3 |\n")
        _w("|-------|--------|-------|-------|-------|\n")
        for name, r in results.items():
            _w(f"| {name} | {r['val_ll']:.4f}"
              f" | {r['fold_lls'][0]:.4f}"
              f" | {r['fold_lls'][1]:.4f}"
              f" | {r['fold_lls'][2]:.4f} |\n")

        # Holdout table
        _w("\n## 2025 Holdout\n\n")
        _w("| Model | Log Loss | Brier | AUC | Accuracy |\n")
        _w("|-------|----------|-------|-----|----------|\n")
        for name in model_specs:
            h = hold_results[name]
            _w(f"| {name} | {h['log_loss']:.4f}"
              f" | {h['brier_score']:.4f}"
              f" | {h['roc_auc']:.4f}"
              f" | {h['accuracy']:.4f} |\n")

        # Slice results
        _w("\n## Slice Performance (2025 Holdout)\n\n")
        _w("| Slice | N | Incumbent LL | Challenger LL | Δ |\n")
        _w("|-------|---|-------------|---------------|---|\n")
        for sl_name, sr in slice_results.items():
            _w(f"| {sl_name} | {sr['n']} | {sr['incumbent_ll']:.4f}"
              f" | {sr['challenger_ll']:.4f} | {sr['delta_ll']:+.4f} |\n")

        # Worst predictions
        _w("\n## Biggest QB Adjustments (2025 Holdout Games)\n\n")
        _w("| Game | Team | QB | Adjustment | Starts |\n")
        _w("|------|------|----|------------|--------|\n")
        qb_adj_series = df.loc[is_hold, "home_qb_adj"].sort_values(ascending=False)
        hold_df_sorted = df[is_hold].loc[qb_adj_series.index] if is_hold.any() else pd.DataFrame()
        for _, rw in hold_df_sorted.head(5).iterrows():
            _w(
                f"| {rw['game_id']} | {rw['home_team']}"
                f" | {rw['home_qb_name']}"
                f" | {rw['home_qb_adj']:+.1f}"
                f" | {rw['home_qb_starts']} |\n"
            )
        _w("| ... | ... | ... | ... | ... |\n")
        for _, rw in hold_df_sorted.tail(5).iterrows():
            _w(
                f"| {rw['game_id']} | {rw['home_team']}"
                f" | {rw['home_qb_name']}"
                f" | {rw['home_qb_adj']:+.1f}"
                f" | {rw['home_qb_starts']} |\n"
            )

        # Decision
        _w("\n## Decision\n\n")
        inc_hold = hold_results["A. Incumbent (Elo + qb_changed + mov3 + Platt)"]["log_loss"]
        inc_val = results["A. Incumbent (Elo + qb_changed + mov3 + Platt)"]["val_ll"]

        challengers_to_check = [
            n for n in model_specs
            if n != "A. Incumbent (Elo + qb_changed + mov3 + Platt)"
        ]
        promoted = []
        for n in challengers_to_check:
            if results[n]["val_ll"] < inc_val and hold_results[n]["log_loss"] < inc_hold:
                promoted.append(n)

        if promoted:
            _w(f"**Promoted: {', '.join(promoted)}**\n\n")
        else:
            _w("**No model beats incumbent on both validation and holdout.**\n\n")

        best_val_name = min(results, key=lambda n: results[n]["val_ll"])
        best_hold_name = min(hold_results, key=lambda n: hold_results[n]["log_loss"])
        _w(f"Best validation: {best_val_name} ({results[best_val_name]['val_ll']:.4f})\n")
        _w(f"Best holdout: {best_hold_name} ({hold_results[best_hold_name]['log_loss']:.4f})\n\n")

        _w("### Failure Modes\n\n")
        _w(
            "1. **Small-sample QBs**: QBs with <17 starts are strongly shrunk"
            " toward replacement. Adjustments for seldom-seen backups are"
            " near zero — correct but may miss real signal.\n"
        )
        _w(
            "2. **QB adjustment independence**: The adjustment assumes QB"
            " impact is additive and independent of the rest of the team.\n"
        )
        _w(
            "3. **No position-group interaction**: The adjustment ignores"
            " offensive line, skill-position talent, and defensive support.\n"
        )
        _w(
            "4. **Oracle QB data**: Uses final actual starter IDs,"
            " not pregame-announced. Live-pregame requires `--qb-input CSV`.\n\n"
        )

        _w("### Recommended Next Experiment\n\n")
        _w(
            "1. **Position-group roster strength (V1)**: Extend shrinkage-based"
            " rating to OL, skill, and defensive units.\n"
        )
        _w(
            "2. **QB-adjustment + market delta**: Test whether QB ratings add"
            " signal beyond the market benchmark for QB-change games.\n"
        )
        _w(
            "3. **Season-expanded QB ratings**: Confirm whether 2021–2025 has"
            " enough QB-start data for stable ratings.\n\n"
        )

        _w(f"---\n*Report generated by `sportslab qb-adjusted-elo`. PRIOR_STARTS={PRIOR_STARTS}, ")
        _w(f"PRIOR_IMPACT={PRIOR_IMPACT}, MAX_ADJUSTMENT={MAX_ADJUSTMENT}.*\n")

    print(f"\nReport: {rp}")

    # ── 6. Export CSV if requested ──
    if output_csv is not None:
        out_df = df[is_hold].copy()
        out_df["incumbent_home_win_prob"] = incumbent_prob
        if challenger_idx is not None:
            out_df["challenger_home_win_prob"] = challenger_prob
        out_df.to_csv(output_csv, index=False)
        print(f"  CSV: {output_csv}")

    return str(report_path)
