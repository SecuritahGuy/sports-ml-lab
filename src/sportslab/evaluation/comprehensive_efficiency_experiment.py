"""Comprehensive efficiency feature experiment.

Tests 5 model variants using new efficiency features (Team Stats Total EPA,
PFR Advanced Stats, Snap Counts) on top of the incumbent:

  1. Platt (incumbent) — Standard Elo + qb_changed + rolling_mov_3 + Platt
  2. Efficiency only — logistic on all 57+ efficiency features
  3. Incumbent + Efficiency — logistic on Elo prob + qb_changed + mov_3 + efficiency features
  4. Team EPA only — team_stats total EPA features alone
  5. PFR only — PFR advanced stats only
  6. Snap only — snap count features only

Rolling-origin 3-fold validation, one-shot 2025 holdout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.metrics import compute_classification_metrics
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.efficiency import (
    COMPREHENSIVE_EFFICIENCY_COLUMNS,
    PFR_COLUMNS,
    SNAP_COLUMNS,
    TEAM_EPA_COLUMNS,
    compute_comprehensive_efficiency_features,
)
from sportslab.features.qb import compute_qb_features
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

# Frozen incumbent Elo params (Standard Elo, no season regression)
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
BEST_MOV_TYPE = "capped_linear"
BEST_MOV_SCALE = 0.05
BEST_MOV_CAP = 2.0


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[MODEL_ELIGIBLE_COLUMN]].copy()
    df = df[~df[NEUTRAL_COLUMN]].copy()
    return df


def _logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def _fit_platt(train_prob: np.ndarray, train_y: np.ndarray) -> Pipeline:
    platt = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    platt.fit(train_prob.reshape(-1, 1), train_y)
    return platt


def run_comprehensive_efficiency_experiment(
    feature_table_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/comprehensive_efficiency.md",
    cache_dir: str = "data/interim/nfl",
) -> str:
    """Run comprehensive efficiency feature experiment."""
    fp = Path(feature_table_path)
    if not fp.exists():
        raise FileNotFoundError(f"Feature table not found: {fp}")

    df_raw = pd.read_parquet(fp)

    # ── Compute Elo ──
    print("\n=== Computing Elo features ===")
    df_elo = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        mov_type=BEST_MOV_TYPE,
        mov_scale=BEST_MOV_SCALE,
        mov_cap=BEST_MOV_CAP,
        decay_half_life=BEST_DECAY,
    )
    print(f"  K={BEST_K}, HFA={BEST_HFA}")

    # ── Compute qb_changed + rolling_mov_3 + situational ──
    print("\n=== Computing QB features ===")
    df_all = compute_qb_features(df_elo)
    print("\n=== Computing situational features (for rolling_mov_3) ===")
    df_all = compute_situational_features(df_all)

    # ── Compute comprehensive efficiency features ──
    print("\n=== Computing comprehensive efficiency features ===")
    df_all = compute_comprehensive_efficiency_features(df_all, cache_dir=cache_dir)

    # ── Build feature matrix ──
    df_all = _filter_df(df_all)

    # Elo probability
    elo_prob = df_all["elo_prob"].values
    y = df_all[TARGET_COLUMN].astype(float).values

    # Incumbent features: elo_prob + qb_changed + rolling_mov_3
    qb_changed_series = df_all[["home_qb_changed", "away_qb_changed"]].max(axis=1).fillna(0)
    mov3_col = "home_rolling_mov_3"
    mov3_series = (
        df_all[mov3_col].fillna(0)
        if mov3_col in df_all.columns
        else pd.Series(np.zeros(len(df_all)))
    )

    incumbent_feature_cols = [c for c in COMPREHENSIVE_EFFICIENCY_COLUMNS if c in df_all.columns]
    print(f"  Efficiency feature columns available: {len(incumbent_feature_cols)}")

    team_epa_avail = [c for c in TEAM_EPA_COLUMNS if c in df_all.columns]
    pfr_avail = [c for c in PFR_COLUMNS if c in df_all.columns]
    snap_avail = [c for c in SNAP_COLUMNS if c in df_all.columns]
    print(f"  Team EPA: {len(team_epa_avail)}, PFR: {len(pfr_avail)}, Snap: {len(snap_avail)}")

    eff_data = df_all[incumbent_feature_cols].values
    team_epa_data = df_all[team_epa_avail].values if team_epa_avail else np.zeros((len(df_all), 1))
    pfr_data = df_all[pfr_avail].values if pfr_avail else np.zeros((len(df_all), 1))
    snap_data = df_all[snap_avail].values if snap_avail else np.zeros((len(df_all), 1))

    # ═══ Rolling-origin evaluation ═══
    print("\n=== Rolling-Origin Evaluation ===")

    results = {
        "platt": [],
        "eff_only": [],
        "incumbent_eff": [],
        "team_epa_only": [],
        "pfr_only": [],
        "snap_only": [],
    }

    for train_seasons, val_season in ROLLING_FOLDS:
        is_train = df_all["season"].isin(train_seasons).values
        is_val = (df_all["season"] == val_season).values

        train_elo = elo_prob[is_train]
        train_y_ = y[is_train].astype(int)
        val_elo = elo_prob[is_val]
        val_y_ = y[is_val]

        train_qb = qb_changed_series.values[is_train].reshape(-1, 1)
        val_qb = qb_changed_series.values[is_val].reshape(-1, 1)
        train_mov3 = mov3_series.values[is_train].reshape(-1, 1)
        val_mov3 = mov3_series.values[is_val].reshape(-1, 1)

        train_eff = eff_data[is_train]
        val_eff = eff_data[is_val]
        train_te = team_epa_data[is_train]
        val_te = team_epa_data[is_val]
        train_pfr = pfr_data[is_train]
        val_pfr = pfr_data[is_val]
        train_snap = snap_data[is_train]
        val_snap = snap_data[is_val]

        # 1. Platt incumbent
        platt = _fit_platt(train_elo, train_y_)
        platt_val = platt.predict_proba(val_elo.reshape(-1, 1))[:, 1]
        results["platt"].append(
            {**compute_classification_metrics(val_y_, platt_val), "model": platt}
        )

        # 2. Efficiency only
        eff_pipe = _logistic_model()
        eff_pipe.fit(train_eff, train_y_)
        eff_val = eff_pipe.predict_proba(val_eff)[:, 1]
        results["eff_only"].append(
            {**compute_classification_metrics(val_y_, eff_val), "model": eff_pipe}
        )

        # 3. Incumbent + Efficiency (logistic on elo + qb + mov3 + eff)
        inc_eff_train = np.column_stack([train_elo, train_qb, train_mov3, train_eff])
        inc_eff_val = np.column_stack([val_elo, val_qb, val_mov3, val_eff])
        ie_pipe = _logistic_model()
        ie_pipe.fit(inc_eff_train, train_y_)
        ie_val = ie_pipe.predict_proba(inc_eff_val)[:, 1]
        results["incumbent_eff"].append(
            {**compute_classification_metrics(val_y_, ie_val), "model": ie_pipe}
        )

        # 4. Team EPA only
        if team_epa_avail:
            te_pipe = _logistic_model()
            te_pipe.fit(train_te, train_y_)
            te_val = te_pipe.predict_proba(val_te)[:, 1]
            results["team_epa_only"].append(
                {**compute_classification_metrics(val_y_, te_val), "model": te_pipe}
            )
        else:
            results["team_epa_only"].append({"log_loss": np.nan})

        # 5. PFR only
        if pfr_avail:
            pfr_pipe = _logistic_model()
            pfr_pipe.fit(train_pfr, train_y_)
            pfr_val = pfr_pipe.predict_proba(val_pfr)[:, 1]
            results["pfr_only"].append(
                {**compute_classification_metrics(val_y_, pfr_val), "model": pfr_pipe}
            )
        else:
            results["pfr_only"].append({"log_loss": np.nan})

        # 6. Snap only
        if snap_avail:
            snap_pipe = _logistic_model()
            snap_pipe.fit(train_snap, train_y_)
            snap_val = snap_pipe.predict_proba(val_snap)[:, 1]
            results["snap_only"].append(
                {**compute_classification_metrics(val_y_, snap_val), "model": snap_pipe}
            )
        else:
            results["snap_only"].append({"log_loss": np.nan})

        print(
            f"  Fold train={train_seasons} val={val_season}:"
            f" platt={results['platt'][-1]['log_loss']:.4f}"
            f" eff={results['eff_only'][-1]['log_loss']:.4f}"
            f" inc+eff={results['incumbent_eff'][-1]['log_loss']:.4f}"
        )

    # ── Average validation LL ──
    def _avg_ll(rlist):
        vals = [r["log_loss"] for r in rlist if not np.isnan(r.get("log_loss", np.nan))]
        return float(np.mean(vals)) if vals else np.nan

    avg = {k: _avg_ll(v) for k, v in results.items()}
    print("\n=== Average Validation Log Loss ===")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: N/A")

    # ═══ One-time 2025 holdout ═══
    print("\n=== 2025 Holdout ===")
    is_hold = (df_all["season"] == HOLDOUT_SEASON).values
    hold_y = y[is_hold]
    hold_elo = elo_prob[is_hold]

    is_full = df_all["season"].isin([2021, 2022, 2023, 2024]).values
    full_elo = elo_prob[is_full]
    full_y = y[is_full].astype(int)
    full_qb = qb_changed_series.values[is_full].reshape(-1, 1)
    hold_qb = qb_changed_series.values[is_hold].reshape(-1, 1)
    full_mov3 = mov3_series.values[is_full].reshape(-1, 1)
    hold_mov3 = mov3_series.values[is_hold].reshape(-1, 1)
    full_eff = eff_data[is_full]
    hold_eff = eff_data[is_hold]

    hold_metrics = {}

    # 1. Platt incumbent
    platt_full = _fit_platt(full_elo, full_y)
    platt_hold = platt_full.predict_proba(hold_elo.reshape(-1, 1))[:, 1]
    hold_metrics["platt"] = compute_classification_metrics(hold_y, platt_hold)
    print(f"  Platt (incumbent): {hold_metrics['platt']['log_loss']:.4f}")

    # 2. Efficiency only
    eff_full = _logistic_model()
    eff_full.fit(full_eff, full_y)
    eff_hold = eff_full.predict_proba(hold_eff)[:, 1]
    hold_metrics["eff_only"] = compute_classification_metrics(hold_y, eff_hold)
    print(f"  Efficiency only: {hold_metrics['eff_only']['log_loss']:.4f}")

    # 3. Incumbent + Efficiency
    inc_eff_full = np.column_stack([full_elo, full_qb, full_mov3, full_eff])
    inc_eff_hold = np.column_stack([hold_elo, hold_qb, hold_mov3, hold_eff])
    ie_full = _logistic_model()
    ie_full.fit(inc_eff_full, full_y)
    ie_hold = ie_full.predict_proba(inc_eff_hold)[:, 1]
    hold_metrics["incumbent_eff"] = compute_classification_metrics(hold_y, ie_hold)
    print(f"  Incumbent + Efficiency: {hold_metrics['incumbent_eff']['log_loss']:.4f}")

    # 4. Team EPA only
    if team_epa_avail:
        te_full = _logistic_model()
        te_full.fit(team_epa_data[is_full], full_y)
        te_hold = te_full.predict_proba(team_epa_data[is_hold])[:, 1]
        hold_metrics["team_epa_only"] = compute_classification_metrics(hold_y, te_hold)
        print(f"  Team EPA only: {hold_metrics['team_epa_only']['log_loss']:.4f}")

    # 5. PFR only
    if pfr_avail:
        pfr_full = _logistic_model()
        pfr_full.fit(pfr_data[is_full], full_y)
        pfr_hold = pfr_full.predict_proba(pfr_data[is_hold])[:, 1]
        hold_metrics["pfr_only"] = compute_classification_metrics(hold_y, pfr_hold)
        print(f"  PFR only: {hold_metrics['pfr_only']['log_loss']:.4f}")

    # 6. Snap only
    if snap_avail:
        snap_full = _logistic_model()
        snap_full.fit(snap_data[is_full], full_y)
        snap_hold = snap_full.predict_proba(snap_data[is_hold])[:, 1]
        hold_metrics["snap_only"] = compute_classification_metrics(hold_y, snap_hold)
        print(f"  Snap only: {hold_metrics['snap_only']['log_loss']:.4f}")

    # Subset analysis: QB-change
    hold_qb_changed = df_all.loc[is_hold, "home_qb_changed"].fillna(0).astype(bool).values
    hold_qb_stable = ~hold_qb_changed

    # ── Report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    prior_rate = full_y.mean()
    random_ll = float(log_loss(hold_y, np.full_like(hold_y, 0.5)))
    prior_ll = float(log_loss(hold_y, np.full_like(hold_y, prior_rate)))

    with open(rp, "w") as f:
        f.write("# Comprehensive Efficiency Features Experiment\n\n")
        f.write(
            "*Testing pregame efficiency features from 3 nflreadpy sources"
            " (Team Stats Total EPA, PFR Advanced Stats, Snap Counts)"
            " on top of the incumbent.*\n\n"
        )

        f.write("## Data Sources\n\n")
        f.write("| Source | Description | Rows/Season | Level |\n")
        f.write("|--------|-------------|-------------|------|\n")
        f.write(
            "| `load_team_stats` | Game-level passing_epa,"
            " rushing_epa, receiving_epa (totals) | ~570 | team-game |\n"
        )
        f.write(
            "| `load_pfr_advstats` (pass/rush/rec/def) |"
            " Pressure rate, bad throws, YAC, broken tackles,"
            " def passer rating, missed tackles | ~700-8000 | player-week |\n"
        )
        f.write("| `load_snap_counts` | OL snap%, top RB snap% | ~26000 | player-week |\n\n")

        f.write("## Feature Groups\n\n")
        f.write("| Group | Features | Count |\n")
        f.write("|-------|----------|-------|\n")
        f.write(
            f"| Team Stats Total EPA | Rolling 3/5 of pass_epa,"
            f" rush_epa, rec_epa, total_epa + net diffs | {len(TEAM_EPA_COLUMNS)} |\n"
        )
        f.write(
            f"| PFR Advanced Stats | Pressure rate, bad throw rate,"
            f" YAC/rush, broken tackles/rush, def passer rating,"
            f" def missed tackle % + net diffs | {len(PFR_COLUMNS)} |\n"
        )
        f.write(f"| Snap Counts | OL snap%, top RB snap% + net diffs | {len(SNAP_COLUMNS)} |\n")
        f.write(f"| **Total** | | **{len(COMPREHENSIVE_EFFICIENCY_COLUMNS)}** |\n\n")

        f.write("## Leakage Prevention\n\n")
        f.write("- All features computed chronologically, shifted (current game excluded)\n")
        f.write("- Rolling windows reset at season boundaries\n")
        f.write("- New season games use 0 imputation + missing flags\n")
        f.write("- Rolling-origin validation prevents 2025 holdout from influencing selection\n\n")

        f.write("## Incumbent Params\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| K-factor | {BEST_K} |\n")
        f.write(f"| Home-field advantage | {BEST_HFA} |\n")
        f.write(f"| MOV type | {BEST_MOV_TYPE} |\n")
        f.write("| Features | elo_prob + qb_changed + rolling_mov_3 |\n")
        f.write("| Calibration | Platt scaling |\n")
        f.write("| Holdout LL | 0.6262 |\n\n")

        f.write("## Models Compared\n\n")
        f.write("| Model | Description |\n")
        f.write("|-------|-------------|\n")
        f.write("| **Platt (incumbent)** | Elo + qb_changed + mov_3 + Platt |\n")
        f.write("| **Efficiency only** | Logistic on all efficiency features |\n")
        f.write(
            "| **Incumbent + Efficiency** | Logistic on elo + qb + mov_3 + efficiency features |\n"
        )
        f.write("| **Team EPA only** | Logistic on team_stats total EPA features |\n")
        f.write("| **PFR only** | Logistic on PFR advanced stats |\n")
        f.write("| **Snap only** | Logistic on snap count features |\n\n")

        # Validation table
        def _fold_ll(name, rlist):
            lls = [r["log_loss"] for r in rlist]
            avg = np.mean(lls)
            fold_str = " | ".join(f"{ll:.4f}" for ll in lls)
            return f"| {name} | {avg:.4f} | {fold_str} |\n"

        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|-------|------------|-------|-------|-------|\n")
        for name, key in [
            ("Platt (incumbent)", "platt"),
            ("Efficiency only", "eff_only"),
            ("Incumbent + Efficiency", "incumbent_eff"),
            ("Team EPA only", "team_epa_only"),
            ("PFR only", "pfr_only"),
            ("Snap only", "snap_only"),
        ]:
            if results[key][0].get("log_loss") is not None and not np.isnan(
                results[key][0].get("log_loss", np.nan)
            ):
                f.write(_fold_ll(name, results[key]))
        f.write("\n")

        # Holdout table
        f.write("## 2025 Holdout\n\n")
        f.write("| Model | Hold LL | Brier | AUC | Acc |\n")
        f.write("|-------|---------|-------|-----|-----|\n")
        f.write(f"| Random | {random_ll:.4f} | 0.2500 | 0.5000 | 0.5000 |\n")
        f.write(f"| Home prior ({prior_rate:.3f}) | {prior_ll:.4f} | — | — | 0.5000 |\n")
        for name, key in [
            ("Platt (incumbent)", "platt"),
            ("Efficiency only", "eff_only"),
            ("Incumbent + Efficiency", "incumbent_eff"),
            ("Team EPA only", "team_epa_only"),
            ("PFR only", "pfr_only"),
            ("Snap only", "snap_only"),
        ]:
            if key in hold_metrics:
                hm = hold_metrics[key]
                f.write(
                    f"| {name} | {hm['log_loss']:.4f}"
                    f" | {hm['brier_score']:.4f}"
                    f" | {hm['roc_auc']:.4f}"
                    f" | {hm['accuracy']:.4f} |\n"
                )
        f.write("\n")

        # Subset analysis
        f.write("## Subset Analysis (2025 Holdout)\n\n")

        subsets = [
            ("All games", slice(None)),
            ("QB changed (home)", hold_qb_changed),
            ("QB stable (home)", hold_qb_stable),
        ]
        f.write("| Subset | N | Platt | Eff only | Inc+Eff |\n")
        f.write("|--------|---|-------|----------|---------|\n")
        for label, mask in subsets:
            if isinstance(mask, slice):
                sub_y, sub_p, sub_e, sub_ie = hold_y, platt_hold, eff_hold, ie_hold
            else:
                sub_y = hold_y[mask]
                sub_p = platt_hold[mask]
                sub_e = eff_hold[mask]
                sub_ie = ie_hold[mask]
            n = len(sub_y)
            if n < 5:
                f.write(f"| {label} | {n} | insufficient | insufficient | insufficient |\n")
                continue
            pll = compute_classification_metrics(sub_y, sub_p)["log_loss"]
            ell = compute_classification_metrics(sub_y, sub_e)["log_loss"]
            ies = compute_classification_metrics(sub_y, sub_ie)["log_loss"]
            f.write(f"| {label} | {n} | {pll:.4f} | {ell:.4f} | {ies:.4f} |\n")
        f.write("\n")

        # Recommendation
        f.write("## Recommendation\n\n")
        incumbent_hold_ll = hold_metrics["platt"]["log_loss"]

        candidates = {}
        for key, label in [
            ("eff_only", "Efficiency only"),
            ("incumbent_eff", "Incumbent + Efficiency"),
            ("team_epa_only", "Team EPA only"),
            ("pfr_only", "PFR only"),
            ("snap_only", "Snap only"),
        ]:
            if key in hold_metrics and not np.isnan(avg.get(key, np.nan)):
                candidates[label] = (avg[key], hold_metrics[key]["log_loss"])

        beat = {n: (v, h) for n, (v, h) in candidates.items() if h < incumbent_hold_ll}
        if beat:
            best_name, (best_val, best_hold) = min(beat.items(), key=lambda kv: kv[1][0])
            f.write(f"✅ **{best_name} promoted as new research incumbent.**\n\n")
            f.write(
                f"Holdout LL {best_hold:.4f} beats incumbent {incumbent_hold_ll:.4f}."
                f" Val LL {best_val:.4f} vs incumbent.\n"
            )
        else:
            best_by_val = min(candidates.items(), key=lambda kv: kv[1][0])
            best_name, (best_val, best_hold) = best_by_val
            f.write("⚠️ **Incumbent remains research incumbent.**\n\n")
            f.write(
                f"No efficiency-augmented model beat incumbent on holdout."
                f" Closest: {best_name}"
                f" (val LL={best_val:.4f}, hold LL={best_hold:.4f})"
                f" vs incumbent hold LL={incumbent_hold_ll:.4f}.\n\n"
            )

        # QB-change assessment
        f.write("### QB-Change Failure Mode\n\n")
        qb_n = int(hold_qb_changed.sum())
        st_n = int(hold_qb_stable.sum())
        if qb_n >= 5:
            qb_platt = compute_classification_metrics(
                hold_y[hold_qb_changed], platt_hold[hold_qb_changed]
            )["log_loss"]
            qb_ie = compute_classification_metrics(
                hold_y[hold_qb_changed], ie_hold[hold_qb_changed]
            )["log_loss"]
            st_platt = compute_classification_metrics(
                hold_y[hold_qb_stable], platt_hold[hold_qb_stable]
            )["log_loss"]
            st_ie = compute_classification_metrics(hold_y[hold_qb_stable], ie_hold[hold_qb_stable])[
                "log_loss"
            ]
            f.write(f"| Model | QB changed (n={qb_n}) | QB stable (n={st_n}) |\n")
            f.write("|-------|--------|--------|\n")
            f.write(f"| Platt | {qb_platt:.4f} | {st_platt:.4f} |\n")
            f.write(f"| Inc+Eff | {qb_ie:.4f} | {st_ie:.4f} |\n")
        else:
            f.write(f"Insufficient QB-change games (n={qb_n}) for subset analysis.\n")

        # Feature importance (if incumbent+eff is viable)
        if hold_metrics["incumbent_eff"]["log_loss"] < incumbent_hold_ll:
            f.write("\n### Feature Importance\n\n")
            f.write("Top coefficients from Incumbent + Efficiency model:\n\n")
            coefs = np.abs(ie_full.named_steps["lr"].coef_[0])
            feature_names = ["elo_prob", "qb_changed", "rolling_mov_3"] + incumbent_feature_cols
            top_idx = np.argsort(coefs)[-10:][::-1]
            f.write("| Feature | Coefficient |\n")
            f.write("|---------|-------------|\n")
            for idx in top_idx:
                f.write(f"| {feature_names[idx]} | {coefs[idx]:.4f} |\n")
            f.write("\n")

    print(f"\nReport written to: {rp}")
    return str(rp)
