"""RALPH Loop 8: Preseason Elo Prior experiment.

Tests whether adding prior-season Elo rating as an explicit feature
in the Platt model improves early-season prediction.

Hypothesis: The model underperforms in Weeks 1-4 because season-start
team strength is not initialized with enough useful prior-season signal.
A preseason Elo prior based on previous-season final Elo, regressed
toward league average, may improve early-season predictions without
leaking future information.
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.experiment_utils import (
    compute_metrics,
    fit_platt,
)
from sportslab.evaluation.fold_safe import (
    INCUMBENT_HOLDOUT_LL,
    MIN_PROMOTION_DELTA,
    build_base_features,
    check_promotion,
    load_feature_table,
)
from sportslab.features.build_features import (
    MODEL_ELIGIBLE_COLUMN,
    NEUTRAL_COLUMN,
    TARGET_COLUMN,
)
from sportslab.features.ratings import compute_elo_features

EXPERIMENT_REPORT = "reports/experiments/preseason_elo_prior.md"

DEFAULT_ELO = 1500.0


def compute_prior_season_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prior-season final Elo for each team at each game.

    Runs the standard Elo pipeline, then for each game extracts the
    *pre-regression* final Elo rating from the previous season for
    each team.  For 2021 (first season), prior Elo defaults to 1500.

    Returns:
        DataFrame with added columns:
            home_prior_elo_raw: final Elo from previous season (no regression)
            away_prior_elo_raw: final Elo from previous season (no regression)
    """
    elo_df = compute_elo_features(
        df,
        k_factor=36,
        home_advantage=40,
        preseason_regression=0.1,
        decay_half_life=32,
    )

    team_final_elo: Dict[str, Dict[int, float]] = {}
    sorted_df = elo_df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    for _, row in sorted_df.iterrows():
        season = int(row["season"])
        home = row["home_team"]
        away = row["away_team"]

        h_elo = row["home_elo_pre"]
        a_elo = row["away_elo_pre"]

        if season not in team_final_elo:
            team_final_elo[season] = {}
        team_final_elo[season][home] = h_elo
        team_final_elo[season][away] = a_elo

    home_prior: List[float] = []
    away_prior: List[float] = []

    for _, row in sorted_df.iterrows():
        season = int(row["season"])
        home = row["home_team"]
        away = row["away_team"]
        prior_season = season - 1

        if prior_season in team_final_elo and prior_season >= 2000:
            h_prior = team_final_elo[prior_season].get(home, DEFAULT_ELO)
            a_prior = team_final_elo[prior_season].get(away, DEFAULT_ELO)
        else:
            h_prior = DEFAULT_ELO
            a_prior = DEFAULT_ELO

        home_prior.append(h_prior)
        away_prior.append(a_prior)

    out = df.copy()
    out["home_prior_elo_raw"] = home_prior
    out["away_prior_elo_raw"] = away_prior
    return out


def _build_base_spine() -> pd.DataFrame:
    df = load_feature_table()
    df = build_base_features(df)
    df = compute_prior_season_elo(df)
    me = df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    neut = df[NEUTRAL_COLUMN].fillna(False).values
    mask = me & ~neut
    return df[mask].copy().reset_index(drop=True)


def _regress_prior(elo_raw: float, reg: float) -> float:
    return DEFAULT_ELO + (1.0 - reg) * (elo_raw - DEFAULT_ELO)


def _x_matrix(df, cols):
    elo = df["elo_prob"].values.reshape(-1, 1)
    f = df[cols].values if cols else None
    return np.column_stack([elo, f]) if f is not None else elo


# ── Variant builders ──

BASE_FEATURES = [
    "home_qb_changed", "away_qb_changed",
    "home_rolling_mov_3", "away_rolling_mov_3",
]


def build_incumbent_fn() -> Callable:
    def model_fn(df, train_mask, val_mask):
        cols = [c for c in BASE_FEATURES if c in df.columns]
        x = _x_matrix(df, cols)
        y = df[TARGET_COLUMN].fillna(0.5).astype(int).values
        pipe = fit_platt(x[train_mask], y[train_mask])
        return pipe.predict_proba(x[val_mask])[:, 1]
    return model_fn


def build_prior_elo_fn(prior_cols: List[str]) -> Callable:
    def model_fn(df, train_mask, val_mask):
        cols = BASE_FEATURES + prior_cols
        avail = [c for c in cols if c in df.columns]
        x = _x_matrix(df, avail)
        y = df[TARGET_COLUMN].fillna(0.5).astype(int).values
        pipe = fit_platt(x[train_mask], y[train_mask])
        return pipe.predict_proba(x[val_mask])[:, 1]
    return model_fn


def build_decay_prior_fn() -> Callable:
    def model_fn(df, train_mask, val_mask):
        cols = BASE_FEATURES + ["home_prior_elo_raw", "away_prior_elo_raw"]
        avail = [c for c in cols if c in df.columns]
        x = _x_matrix(df, avail)
        week = df["week"].fillna(1).values.astype(float).reshape(-1, 1)
        decay = np.exp(-0.15 * week)
        if avail and "home_prior_elo_raw" in df.columns:
            prior_idx_start = len([c for c in BASE_FEATURES if c in df.columns])
            for i in range(prior_idx_start, x.shape[1]):
                x[:, i] = x[:, i] * decay.flatten()
        y = df[TARGET_COLUMN].fillna(0.5).astype(int).values
        pipe = fit_platt(x[train_mask], y[train_mask])
        return pipe.predict_proba(x[val_mask])[:, 1]
    return model_fn


# ── Runner ──

def _fold_safe_cv(df, model_fn) -> Tuple[List[Dict], float]:
    fold_metrics = []
    for train_seasons, val_season in ROLLING_FOLDS:
        me = df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
        tr = df["season"].isin(train_seasons).values & me
        va = (df["season"] == val_season).values & me
        preds = model_fn(df, tr, va)
        y_val = df.loc[va, TARGET_COLUMN].astype(float).values
        m = compute_metrics(y_val, preds)
        fold_metrics.append(m)
    avg_ll = float(np.mean([m["log_loss"] for m in fold_metrics if "log_loss" in m]))
    return fold_metrics, round(avg_ll, 4)


def _score_holdout(df, model_fn) -> Dict:
    train_mask = (df["season"].isin([2021, 2022, 2023, 2024]).values
                  & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values)
    ho = (df["season"] == HOLDOUT_SEASON).values & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    preds = model_fn(df, train_mask, ho)
    y_ho = df.loc[ho, TARGET_COLUMN].astype(float).values
    m = compute_metrics(y_ho, preds)
    from sklearn.metrics import roc_auc_score
    valid = ~np.isnan(y_ho)
    if valid.sum() > 1 and len(np.unique(y_ho[valid])) > 1:
        m["roc_auc"] = round(float(roc_auc_score(y_ho[valid], preds[valid])), 4)
    else:
        m["roc_auc"] = None
    m["ece"] = _compute_ece(y_ho, preds)
    return m


def _compute_ece(y_true, y_prob, n_bins=10):
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_p, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        ece += abs(y_p[mask].mean() - y_t[mask].mean()) * mask.sum()
    return round(ece / len(y_t), 6)


def _run_variant(df, name, model_fn) -> Dict:
    print(f"\n=== {name} ===")
    fold_metrics, val_ll = _fold_safe_cv(df, model_fn)
    hold_metrics = _score_holdout(df, model_fn)
    print(f"  Val LL: {val_ll:.4f}")
    print(f"  Holdout LL: {hold_metrics.get('log_loss', 'N/A')}")
    return {
        "name": name,
        "val_ll": val_ll,
        "fold_metrics": fold_metrics,
        "hold": hold_metrics,
    }


def _subgroup_metrics(df, model_fn, condition, label) -> Dict:
    if condition.sum() == 0:
        return {"label": label, "n": 0, "log_loss": None}
    train_mask = (df["season"].isin([2021, 2022, 2023, 2024]).values
                  & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values)
    ho = (df["season"] == HOLDOUT_SEASON).values & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    target_mask = ho & condition
    if target_mask.sum() == 0:
        return {"label": label, "n": 0, "log_loss": None}
    preds = model_fn(df, train_mask, target_mask)
    y_sub = df.loc[target_mask, TARGET_COLUMN].astype(float).values
    m = compute_metrics(y_sub, preds)
    return {"label": label, "n": int(target_mask.sum()), "log_loss": m.get("log_loss"),
            "brier": m.get("brier"), "accuracy": m.get("accuracy")}


def _season_metrics(df, model_fn, season: int) -> Dict:
    train_mask = (df["season"].isin([2021, 2022, 2023, 2024]).values
                  & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values)
    ho = (df["season"] == season).values & df[MODEL_ELIGIBLE_COLUMN].fillna(False).values
    preds = model_fn(df, train_mask, ho)
    y_season = df.loc[ho, TARGET_COLUMN].astype(float).values
    m = compute_metrics(y_season, preds)
    return {"season": season, "n": int(ho.sum()), **m}


# ── CLI entry point ──

def run_preseason_elo_experiment(
    report_path: str = EXPERIMENT_REPORT,
) -> str:
    print("=== RALPH Loop 8: Preseason Elo Prior ===\n")
    df = _build_base_spine()
    print(f"  Eligible games: {len(df)}")

    # Verify prior_elo columns exist
    has_raw = "home_prior_elo_raw" in df.columns
    print(f"  prior_elo_raw columns: {'✅' if has_raw else '❌'}")

    if has_raw:
        raw_vals = df["home_prior_elo_raw"]
        print(f"  home_prior_elo_raw: mean={raw_vals.mean():.0f}, "
              f"std={raw_vals.std():.0f}, min={raw_vals.min():.0f}, "
              f"max={raw_vals.max():.0f}")

    # Build variants
    variants = {
        "incumbent": {
            "fn": build_incumbent_fn(),
            "desc": "Platt(qb_changed + rolling_mov_3) — no prior_elo",
        },
        "prior_elo_raw": {
            "fn": build_prior_elo_fn(["home_prior_elo_raw", "away_prior_elo_raw"]),
            "desc": "Incumbent + raw prior-season final Elo (no regression)",
        },
        "prior_elo_reg10": {
            "fn": build_prior_elo_fn(["home_prior_elo_raw", "away_prior_elo_raw"]),
            "desc": "Incumbent + regressed prior Elo (10%)",
        },
        "prior_elo_reg50": {
            "fn": build_prior_elo_fn(["home_prior_elo_raw", "away_prior_elo_raw"]),
            "desc": "Incumbent + regressed prior Elo (50%)",
        },
        "prior_elo_diff": {
            "fn": build_prior_elo_fn(["home_prior_elo_raw", "away_prior_elo_raw"]),
            "desc": "Incumbent + prior elo diff",
        },
        "prior_elo_raw_decay": {
            "fn": build_decay_prior_fn(),
            "desc": "Incumbent + decay-weighted prior elo",
        },
    }

    results = {}

    for name, cfg in variants.items():
        fn = cfg["fn"]

        # For regressed variants, inject regression into prior columns before running
        if "reg10" in name or "reg50" in name:
            reg = 0.1 if "reg10" in name else 0.5
            df_reg = df.copy()
            df_reg["home_prior_elo_raw"] = df["home_prior_elo_raw"].apply(
                lambda v: _regress_prior(v, reg))
            df_reg["away_prior_elo_raw"] = df["away_prior_elo_raw"].apply(
                lambda v: _regress_prior(v, reg))
        elif "diff" in name:
            df_reg = df.copy()
            df_reg["home_prior_elo_raw"] = df["home_prior_elo_raw"] - df["away_prior_elo_raw"]
            df_reg["away_prior_elo_raw"] = 0.0
        else:
            df_reg = df.copy()

        r = _run_variant(df_reg, name, fn)
        results[name] = r

    # ── Compare vs incumbent ──
    inc_val = results["incumbent"]["val_ll"]
    inc_hold = results["incumbent"]["hold"]["log_loss"]
    print("\n\n=== Incumbent Reference ===")
    print(f"  Val LL: {inc_val:.4f}")
    print(f"  Holdout LL: {inc_hold:.4f}")

    promoted = []
    rejected = []
    for name, r in results.items():
        if name == "incumbent":
            continue
        v = r["val_ll"]
        h = r["hold"]["log_loss"]
        verdict = check_promotion(
            v, h, incumbent_val=inc_val, incumbent_holdout=inc_hold,
            delta=MIN_PROMOTION_DELTA,
        )
        if verdict["promoted"]:
            promoted.append(name)
            print(f"\n  ✅ PROMOTED: {name}")
        else:
            rejected.append(name)
            print(f"\n  ❌ REJECTED: {name}")

    # ── Detailed subgroup analysis for best variant ──
    best_name = None
    best_delta = 999
    for name, r in results.items():
        if name == "incumbent":
            continue
        v = r["val_ll"]
        h = r["hold"]["log_loss"]
        total_delta = (v - inc_val) + (h - inc_hold)
        if total_delta < best_delta:
            best_delta = total_delta
            best_name = name

    subgroup_results = {}
    if best_name and best_name in results:
        print(f"\n\n=== Subgroup Analysis (best: {best_name}) ===")
        best_fn = variants[best_name]["fn"]
        inc_fn = variants["incumbent"]["fn"]

        # Apply any needed regression for best variant
        if best_name == "prior_elo_raw":
            df_best = df.copy()
        elif best_name == "prior_elo_reg10":
            df_best = df.copy()
            reg = 0.1
            df_best["home_prior_elo_raw"] = df["home_prior_elo_raw"].apply(
                lambda v: _regress_prior(v, reg))
            df_best["away_prior_elo_raw"] = df["away_prior_elo_raw"].apply(
                lambda v: _regress_prior(v, reg))
        elif best_name == "prior_elo_reg50":
            df_best = df.copy()
            reg = 0.5
            df_best["home_prior_elo_raw"] = df["home_prior_elo_raw"].apply(
                lambda v: _regress_prior(v, reg))
            df_best["away_prior_elo_raw"] = df["away_prior_elo_raw"].apply(
                lambda v: _regress_prior(v, reg))
        elif best_name == "prior_elo_diff":
            df_best = df.copy()
            df_best["home_prior_elo_raw"] = df["home_prior_elo_raw"] - df["away_prior_elo_raw"]
            df_best["away_prior_elo_raw"] = 0.0
        else:
            df_best = df.copy()

        for label, key in [
            ("Early season (Weeks 1-4)", "early"),
            ("Mid season (Weeks 5-12)", "mid"),
            ("Late season (Weeks 13+)", "late"),
            ("Playoffs", "playoffs"),
            ("QB changed", "qb_changed"),
            ("Missing weather", "weather_missing"),
        ]:
            if key == "early":
                cond = df_best["week"].values <= 4
            elif key == "mid":
                cond = (df_best["week"].values >= 5) & (df_best["week"].values <= 12)
            elif key == "late":
                cond = df_best["week"].values >= 13
            elif key == "playoffs":
                cond = df_best["game_type"].astype(str).str.contains(
                    "Wild|Divisional|Conference|SuperBowl", na=False
                ).values
            elif key == "qb_changed":
                hc = df_best.get("home_qb_changed", pd.Series(0)).values
                ac = df_best.get("away_qb_changed", pd.Series(0)).values
                cond = (hc == 1) | (ac == 1)
            elif key == "weather_missing":
                cond = df_best.get("weather_missing_flag", pd.Series(0)).values == 1
            else:
                cond = np.zeros(len(df_best), dtype=bool)

            inc_sr = _subgroup_metrics(df_best, inc_fn, cond, f"incumbent_{key}")
            best_sr = _subgroup_metrics(df_best, best_fn, cond, f"{best_name}_{key}")
            subgroup_results[key] = {"incumbent": inc_sr, "best": best_sr}
            if inc_sr["n"] > 0:
                bl = best_sr["log_loss"]
                il = inc_sr["log_loss"]
                delta_ll = (bl - il) if bl is not None and il is not None else None
                delta_str = f"{delta_ll:+.4f}" if delta_ll is not None else "—"
                print(f"  {label} (n={inc_sr['n']}): incumbent LL={inc_sr['log_loss']}, "
                      f"{best_name} LL={best_sr.get('log_loss', '—')}, Δ={delta_str}")

    # ── Write report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    with open(rp, "w") as f:
        f.write("# RALPH Loop 8: Preseason Elo Prior\n\n")
        f.write(f"*Incumbent: v3.0.0 Frozen QB Overlay (holdout LL {INCUMBENT_HOLDOUT_LL})*\n\n")

        f.write("## Hypothesis\n\n")
        f.write("The model underperforms in Weeks 1-4 (LL=0.6727 vs late season 0.6032) "
                "because season-start Elo ratings carry only a diluted signal "
                "from the prior season (10% regression toward 1500). "
                "Adding the pre-regression prior-season final Elo as an explicit "
                "Platt feature provides a stronger preseason reference, "
                "especially before current-season rolling averages accumulate.\n\n")

        f.write("## Variants\n\n")
        f.write("| ID | Description |\n")
        f.write("|----|-------------|\n")
        for name, cfg in variants.items():
            f.write(f"| {name} | {cfg['desc']} |\n")
        f.write("\n")

        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model | Avg Val LL | Fold1 | Fold2 | Fold3 | Δ vs Inc |\n")
        f.write("|-------|-----------|-------|-------|-------|----------|\n")
        for name, r in results.items():
            fm = r["fold_metrics"]
            delta = r["val_ll"] - inc_val
            f.write(
                f"| {name} | {r['val_ll']:.4f}"
                f" | {fm[0]['log_loss']:.4f}"
                f" | {fm[1]['log_loss']:.4f}"
                f" | {fm[2]['log_loss']:.4f}"
                f" | {delta:+.4f} |\n"
            )
        f.write("\n")

        f.write("## Holdout (2025)\n\n")
        f.write("| Model | Hold LL | Brier | Acc | AUC | ECE | Δ vs Inc |\n")
        f.write("|-------|---------|-------|-----|-----|-----|----------|\n")
        for name, r in results.items():
            h = r["hold"]
            delta = h["log_loss"] - inc_hold
            f.write(f"| {name} | {h.get('log_loss', 'N/A')}"
                    f" | {h.get('brier', 'N/A')} | {h.get('accuracy', 'N/A')}"
                    f" | {h.get('roc_auc', 'N/A')}"
                    f" | {h.get('ece', 'N/A')}"
                    f" | {delta:+.4f} |\n")
        f.write("\n")

        f.write("## Season-by-Season (2025 holdout)\n\n")
        f.write("| Model | Season | N | Log Loss |\n")
        f.write("|-------|--------|---|----------|\n")
        for season in [2025]:
            for name, r in results.items():
                h = r["hold"]
                f.write(f"| {name} | {season} | 276 | {h.get('log_loss', 'N/A')} |\n")
        f.write("\n")

        if best_name:
            f.write(f"## Subgroup Analysis (Best: {best_name})\n\n")
            f.write("| Subgroup | N | Incumbent LL | Best LL | Δ |\n")
            f.write("|----------|---|-------------|---------|---|\n")
            for key, sg in subgroup_results.items():
                inc_s = sg["incumbent"]
                best_s = sg["best"]
                inc_ok = inc_s["n"] > 0 and inc_s["log_loss"] is not None
                best_ok = best_s["log_loss"] is not None
                if inc_ok and best_ok:
                    delta = best_s["log_loss"] - inc_s["log_loss"]
                    f.write(f"| {key} | {inc_s['n']} | {inc_s['log_loss']:.4f}"
                            f" | {best_s['log_loss']:.4f} | {delta:+.4f} |\n")
                elif inc_s["n"] > 0:
                    ill = inc_s["log_loss"]
                    bll = best_s["log_loss"]
                    f.write(f"| {key} | {inc_s['n']} | {ill} | {bll} | — |\n")
            f.write("\n")

        f.write("## Decisions\n\n")
        if promoted:
            for name in promoted:
                r = results[name]
                v = r["val_ll"]
                h = r["hold"]["log_loss"]
                f.write(f"### ✅ {name}\n\n")
                f.write(f"Val LL: {v:.4f} (Δ={v-inc_val:.4f}), "
                        f"Holdout: {h:.4f} (Δ={h-inc_hold:.4f})\n\n")
                f.write("Promoted — beats incumbent on both val and holdout with Δ≥0.001.\n\n")
        else:
            f.write("**No challenger beats the incumbent.**\n\n")

        for name in rejected:
            if name == "incumbent":
                continue
            r = results[name]
            v = r["val_ll"]
            h = r["hold"]["log_loss"]
            reason = "val "
            if v >= inc_val + MIN_PROMOTION_DELTA:
                reason += f"worse ({v-inc_val:+.4f}) "
            else:
                reason += f"better ({v-inc_val:+.4f}) "
            reason += "holdout "
            if h >= inc_hold + MIN_PROMOTION_DELTA:
                reason += f"worse ({h-inc_hold:+.4f})"
            else:
                reason += f"better ({h-inc_hold:+.4f})"
            f.write(f"### ❌ {name}\n\n")
            f.write(f"Val LL: {v:.4f} (Δ={v-inc_val:.4f}), "
                    f"Holdout: {h:.4f} (Δ={h-inc_hold:.4f})\n\n")
            f.write(f"Rejected — {reason}.\n\n")

        f.write("## Leakage Assessment\n\n")
        f.write("| Feature | Source | Leakage Risk | Live-Safe |\n")
        f.write("|---------|--------|--------------|-----------|\n")
        f.write("| home_prior_elo_raw | Prior-season final Elo | None (pre-kickoff) | Yes |\n")
        f.write("| away_prior_elo_raw | Prior-season final Elo | None (pre-kickoff) | Yes |\n\n")

        f.write("## Operational Impact\n\n")
        f.write("* No change to Elo pipeline\n")
        f.write("* No new external data sources\n")
        f.write("* No change to live weekly prediction mode\n")
        f.write("* All prior-season Elo values computed from existing games\n\n")

        f.write("## Audit Answers\n\n")
        if best_name:
            best_r = results[best_name]
            f.write(f"Best variant: **{best_name}**\n\n")
            bv = best_r["val_ll"]
            bh = best_r["hold"]["log_loss"]
            ans1 = _audit_answer(best_name, subgroup_results, "early")
            f.write(f"1. **Improves Weeks 1-4?** {ans1}\n")
            wins = bv <= inc_val - MIN_PROMOTION_DELTA
            wins = wins and bh <= inc_hold - MIN_PROMOTION_DELTA
            f.write(f"2. **Improves both val and holdout by ≥0.001?** "
                    f"{'✅ Yes' if wins else '❌ No'}\n")
            f.write("3. **Stable across folds?** See fold table above.\n")
            f.write(f"4. **Increases overconfidence?** "
                    f"{_audit_calibration(best_name, results, 'ece')}\n")
            f.write(f"5. **Worsens QB-change games?** "
                    f"{_audit_answer(best_name, subgroup_results, 'qb_changed')}\n")
            f.write(f"6. **Worsens missing-weather games?** "
                    f"{_audit_answer(best_name, subgroup_results, 'weather_missing')}\n")
            f.write("7. **Data available before kickoff?** Yes — prior-season Elo is known\n")
            f.write("8. **Changes live weekly operation?** No — feature added to Platt model\n")
            f.write("9. **Adds operational fragility?** No — static feature\n")
            wins = bv <= inc_val - MIN_PROMOTION_DELTA
            wins = wins and bh <= inc_hold - MIN_PROMOTION_DELTA
            f.write(f"10. **Result large enough?** "
                    f"{'✅ Yes' if wins else '❌ No'}\n\n")
        else:
            f.write("No best variant identified.\n\n")

    print(f"\nReport: {rp}")
    return str(rp)


def _audit_answer(name, subgroup_results, key):
    if key not in subgroup_results:
        return "N/A"
    sg = subgroup_results[key]
    inc_s = sg.get("incumbent", {})
    best_s = sg.get("best", {})
    if inc_s.get("n", 0) == 0:
        return "N/A (0 games)"
    inc_ll = inc_s.get("log_loss")
    best_ll = best_s.get("log_loss")
    if inc_ll is None or best_ll is None:
        return "N/A"
    delta = best_ll - inc_ll
    if delta < -0.001:
        return f"✅ Improves by {delta:.4f}"
    elif delta > 0.001:
        return f"❌ Worsens by {delta:.4f}"
    return f"≈ Flat (Δ={delta:.4f})"


def _audit_calibration(name, results, key):
    if name not in results:
        return "N/A"
    r = results[name]
    ece = r.get("hold", {}).get(key)
    inc_ece = results["incumbent"]["hold"].get(key)
    if ece is None or inc_ece is None:
        return "N/A"
    delta = ece - inc_ece
    if delta < -0.01:
        return f"✅ Improves by {delta:.4f}"
    elif delta > 0.01:
        return f"❌ Worsens by {delta:.4f}"
    return f"≈ Flat (Δ={delta:.4f})"
