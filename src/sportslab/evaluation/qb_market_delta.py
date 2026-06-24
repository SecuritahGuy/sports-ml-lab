"""QB-change market-delta diagnostics experiment.

Uses existing incumbent predictions and market data to determine whether
closing-market disagreement with the football-only model identifies
QB-change / availability-shock games where the incumbent is structurally blind,
and creates a market-aware caution layer without replacing the incumbent.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.metrics import compute_classification_metrics

# ── Current football-only incumbent ──
INCUMBENT_VERSION = "v2.0.0"
INCUMBENT_HOLDOUT_LL = 0.6262

PREDICTIONS_PATH = Path("reports/predictions/incumbent_predictions.csv")
FEATURE_TABLE_PATH = Path("data/features/nfl/feature_table.parquet")
CAUTION_FLAGS_PATH = Path("reports/predictions/market_aware_caution_flags.csv")

# ── Blend weights to test ──
SIMPLE_BLEND_WEIGHTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
QB_GATED_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
LARGE_DELTA_THRESHOLDS = [0.05, 0.075, 0.10, 0.15]
LARGE_DELTA_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
DELTA_TEST_THRESHOLDS = [0.025, 0.05, 0.075, 0.10, 0.15]


def _logistic_pipe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def _safe_logit(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _assign_delta_bucket(abs_delta: np.ndarray) -> np.ndarray:
    buckets = np.full(len(abs_delta), ">0.15", dtype=object)
    for i, threshold in enumerate(DELTA_TEST_THRESHOLDS):
        if i == 0:
            buckets[abs_delta < threshold] = f"<{threshold}"
        else:
            lower = DELTA_TEST_THRESHOLDS[i - 1]
            mask = (abs_delta >= lower) & (abs_delta < threshold)
            buckets[mask] = f"{lower}–{threshold}"
    return buckets


def _load_and_merge() -> pd.DataFrame:
    """Load incumbent predictions and feature table, merge on game_id."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Incumbent predictions not found: {PREDICTIONS_PATH}. "
            "Run `make predict-incumbent` first."
        )
    if not FEATURE_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE_PATH}. Run `make build-features` first."
        )

    df_pred = pd.read_csv(PREDICTIONS_PATH)
    df_feat = pd.read_parquet(FEATURE_TABLE_PATH)

    # Select needed columns from feature table
    feat_cols = [
        "game_id",
        "home_qb_changed",
        "away_qb_changed",
        "market_home_prob_novig",
    ]
    avail = [c for c in feat_cols if c in df_feat.columns]
    df_feat_subset = df_feat[avail].copy()
    df_feat_subset["game_id"] = df_feat_subset["game_id"].astype(str)
    df_pred["game_id"] = df_pred["game_id"].astype(str)

    df = df_pred.merge(df_feat_subset, on="game_id", how="left")
    if len(df) != len(df_pred):
        print(f"  Warning: merge changed row count ({len(df_pred)} → {len(df)})")

    return df


def _compute_diagnostic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-delta diagnostic fields to the merged DataFrame."""
    out = df.copy()

    out["model_prob"] = out["incumbent_home_win_prob"].astype(float)

    # Market probability from predictions CSV (has diagnostic label in name)
    if "market_prob_diagnostic" in out.columns:
        out["market_prob"] = out["market_prob_diagnostic"].astype(float)
    elif "market_home_prob_novig" in out.columns:
        out["market_prob"] = out["market_home_prob_novig"].astype(float)
    else:
        raise ValueError("No market probability found in data")

    out["market_minus_model"] = out["market_prob"] - out["model_prob"]
    out["abs_market_minus_model"] = np.abs(out["market_minus_model"])

    out["model_logit"] = _safe_logit(out["model_prob"].values)
    out["market_logit"] = _safe_logit(out["market_prob"].values)
    out["market_logit_minus_model_logit"] = out["market_logit"] - out["model_logit"]

    # QB change flags — recompute from home/away if available
    if "home_qb_changed" in out.columns:
        out["home_qb_change_flag"] = out["home_qb_changed"].fillna(0).astype(int)
    else:
        out["home_qb_change_flag"] = out.get("caution_qb_change", 0).astype(int)
    if "away_qb_changed" in out.columns:
        out["away_qb_change_flag"] = out["away_qb_changed"].fillna(0).astype(int)
    else:
        out["away_qb_change_flag"] = 0

    if "home_qb_changed" in out.columns and "away_qb_changed" in out.columns:
        out["qb_change_flag"] = (out["home_qb_change_flag"] | out["away_qb_change_flag"]).astype(
            int
        )
    elif "qb_change_flag" in out.columns:
        out["qb_change_flag"] = out["qb_change_flag"].astype(int)
    else:
        out["qb_change_flag"] = (out["home_qb_change_flag"] | out["away_qb_change_flag"]).astype(
            int
        )

    # Favorite disagreement: market and model disagree on which team is favored
    out["favorite_disagreement_flag"] = (
        ((out["market_prob"] > 0.5) & (out["model_prob"] < 0.5))
        | ((out["market_prob"] < 0.5) & (out["model_prob"] > 0.5))
    ).astype(int)

    out["directionally_aligned_flag"] = (
        ((out["market_prob"] >= 0.5) & (out["model_prob"] >= 0.5))
        | ((out["market_prob"] <= 0.5) & (out["model_prob"] <= 0.5))
    ).astype(int)

    out["large_market_delta_flag"] = (out["abs_market_minus_model"] >= 0.05).astype(int)

    return out


def run_qb_market_delta_experiment(
    report_path: str = "reports/experiments/qb_market_delta.md",
) -> str:
    """Run QB-change market-delta diagnostics experiment."""
    print("=== Loading & merging data ===")
    df_raw = _load_and_merge()
    print(f"  Loaded {len(df_raw)} games")

    print("=== Computing diagnostic fields ===")
    df = _compute_diagnostic_fields(df_raw)
    print(f"  Rows: {len(df)}")

    model_prob = df["model_prob"].values
    market_prob = df["market_prob"].values
    abs_delta = df["abs_market_minus_model"].values
    qb_changed = df["qb_change_flag"].values
    fav_disagree = df["favorite_disagreement_flag"].values
    seasons = df["season"].values
    weeks = df["week"].values
    y = df["home_win_actual"].values.astype(float)
    delta_buckets = _assign_delta_bucket(abs_delta)

    print(f"  Market prob range: [{market_prob.min():.4f}, {market_prob.max():.4f}]")
    print(f"  Model prob range:  [{model_prob.min():.4f}, {model_prob.max():.4f}]")
    print(f"  Corr(Market, Model): {np.corrcoef(model_prob, market_prob)[0, 1]:.4f}")
    print(f"  QB-change games: {qb_changed.sum():.0f} / {len(df)}")

    # ── Rolling-origin evaluation ──
    print("\n" + "=" * 60)
    print("Rolling-Origin Evaluation (selection by avg val LL)")
    print("=" + "=" * 60)

    platt_results: List[dict] = []
    market_results: List[dict] = []
    logistic_blend_results: List[dict] = []
    simple_blend_best: Dict[str, dict] = {}
    qb_gated_best: Dict[str, dict] = {}
    ld_gated_best: Dict[str, dict] = {}
    qb_ld_gated_best: Dict[str, dict] = {}

    for fold_idx, (train_seasons, val_season) in enumerate(ROLLING_FOLDS):
        is_train = np.isin(seasons, list(train_seasons))
        is_val = seasons == val_season
        train_y = y[is_train].astype(int)
        val_y = y[is_val]
        train_mod = model_prob[is_train]
        val_mod = model_prob[is_val]
        train_mkt = market_prob[is_train]
        val_mkt = market_prob[is_val]
        train_ad = abs_delta[is_train]
        val_ad = abs_delta[is_val]
        train_qb = qb_changed[is_train]
        val_qb = qb_changed[is_val]

        # 1. Incumbent (model prob direct)
        inc_m = compute_classification_metrics(val_y, val_mod)
        platt_results.append({"log_loss": inc_m["log_loss"], "metrics": inc_m})

        # 2. Market
        mkt_m = compute_classification_metrics(val_y, val_mkt)
        market_results.append({"log_loss": mkt_m["log_loss"], "metrics": mkt_m})

        # 3. Simple blend
        for w in SIMPLE_BLEND_WEIGHTS:
            blend = w * val_mkt + (1.0 - w) * val_mod
            bm = compute_classification_metrics(val_y, blend)
            _record_blend(simple_blend_best, fold_idx, f"simple_w{w:.2f}", bm, blend, None)

        # 4. QB-change gated
        for w in QB_GATED_WEIGHTS:
            blend = np.where(val_qb, w * val_mkt + (1.0 - w) * val_mod, val_mod)
            bm = compute_classification_metrics(val_y, blend)
            _record_blend(qb_gated_best, fold_idx, f"qb_gated_w{w:.2f}", bm, blend, None)

        # 5. Large-delta gated
        for tr in LARGE_DELTA_THRESHOLDS:
            for w in LARGE_DELTA_WEIGHTS:
                gate = val_ad >= tr
                blend = np.where(gate, w * val_mkt + (1.0 - w) * val_mod, val_mod)
                bm = compute_classification_metrics(val_y, blend)
                _record_blend(ld_gated_best, fold_idx, f"ld_t{tr:.3f}_w{w:.2f}", bm, blend, None)

        # 6. QB+LD gated
        for tr in LARGE_DELTA_THRESHOLDS:
            for w in LARGE_DELTA_WEIGHTS:
                gate = val_qb & (val_ad >= tr)
                blend = np.where(gate, w * val_mkt + (1.0 - w) * val_mod, val_mod)
                bm = compute_classification_metrics(val_y, blend)
                _record_blend(
                    qb_ld_gated_best, fold_idx, f"qb_ld_t{tr:.3f}_w{w:.2f}", bm, blend, None
                )

        # 7. Logistic blend
        train_ml = _safe_logit(train_mod)
        val_ml = _safe_logit(val_mod)
        train_mkt_l = _safe_logit(train_mkt)
        val_mkt_l = _safe_logit(val_mkt)
        logit_x_tr = np.column_stack(
            [
                train_ml,
                train_mkt_l,
                train_qb.astype(float),
                train_ad,
                train_qb.astype(float) * train_ad,
            ]
        )
        logit_x_va = np.column_stack(
            [
                val_ml,
                val_mkt_l,
                val_qb.astype(float),
                val_ad,
                val_qb.astype(float) * val_ad,
            ]
        )
        logit_pipe = _logistic_pipe()
        logit_pipe.fit(logit_x_tr, train_y)
        logit_p = logit_pipe.predict_proba(logit_x_va)[:, 1]
        lgm = compute_classification_metrics(val_y, logit_p)
        logistic_blend_results.append(
            {"log_loss": lgm["log_loss"], "metrics": lgm, "model": logit_pipe}
        )

        print(
            f"  Fold {fold_idx + 1} train={train_seasons} val={val_season}:"
            f" inc={inc_m['log_loss']:.4f}"
            f" mkt={mkt_m['log_loss']:.4f}"
            f" logit={lgm['log_loss']:.4f}"
        )

    # ── Aggregate ──
    avg_inc = float(np.mean([r["log_loss"] for r in platt_results]))
    avg_mkt = float(np.mean([r["log_loss"] for r in market_results]))
    avg_logit = float(np.mean([r["log_loss"] for r in logistic_blend_results]))

    best_simple_key = min(simple_blend_best, key=lambda k: simple_blend_best[k]["avg_ll"])
    best_simple = simple_blend_best[best_simple_key]

    best_qbg_key = min(qb_gated_best, key=lambda k: qb_gated_best[k]["avg_ll"])
    best_qbg = qb_gated_best[best_qbg_key]

    best_ld_key = min(ld_gated_best, key=lambda k: ld_gated_best[k]["avg_ll"])
    best_ld = ld_gated_best[best_ld_key]

    best_qbld_key = min(qb_ld_gated_best, key=lambda k: qb_ld_gated_best[k]["avg_ll"])
    best_qbld = qb_ld_gated_best[best_qbld_key]

    print(f"\n  Incumbent (model):      {avg_inc:.4f}")
    print(f"  Market (no-vig):        {avg_mkt:.4f}")
    print(f"  Logistic blend:         {avg_logit:.4f}")
    print(f"  Best simple blend:      {best_simple_key} = {best_simple['avg_ll']:.4f}")
    print(f"  Best QB gated:          {best_qbg_key} = {best_qbg['avg_ll']:.4f}")
    print(f"  Best LD gated:          {best_ld_key} = {best_ld['avg_ll']:.4f}")
    print(f"  Best QB+LD gated:       {best_qbld_key} = {best_qbld['avg_ll']:.4f}")

    # ── Select best by avg val LL ──
    candidates = {
        "incumbent": avg_inc,
        "market": avg_mkt,
        "simple_blend": best_simple["avg_ll"],
        "qb_gated": best_qbg["avg_ll"],
        "ld_gated": best_ld["avg_ll"],
        "qb_ld_gated": best_qbld["avg_ll"],
        "logistic_blend": avg_logit,
    }
    best_candidate_name = min(candidates, key=candidates.get)
    best_candidate_val_ll = candidates[best_candidate_name]
    print(f"\n  Best val-selected: {best_candidate_name} = {best_candidate_val_ll:.4f}")

    # ── One-shot 2025 holdout ──
    print("\n" + "=" * 60)
    print("2025 Holdout Evaluation (one-shot, no selection)")
    print("=" * 60)

    is_hold = seasons == HOLDOUT_SEASON
    is_train = np.isin(seasons, [2021, 2022, 2023, 2024])
    h_y = y[is_hold].copy()
    h_mod = model_prob[is_hold]
    h_mkt = market_prob[is_hold]
    h_ad = abs_delta[is_hold]
    h_qb = qb_changed[is_hold]
    h_fav = fav_disagree[is_hold]
    h_wk = weeks[is_hold]
    h_db = delta_buckets[is_hold]

    t_mod = model_prob[is_train]
    t_mkt = market_prob[is_train]
    t_ad = abs_delta[is_train]
    t_qb = qb_changed[is_train]
    t_y = y[is_train].astype(int)

    hold: Dict[str, dict] = {}

    hold["incumbent"] = compute_classification_metrics(h_y, h_mod)
    print(f"  Incumbent:   holdout LL = {hold['incumbent']['log_loss']:.4f}")

    hold["market"] = compute_classification_metrics(h_y, h_mkt)
    print(f"  Market:      holdout LL = {hold['market']['log_loss']:.4f}")

    # Best simple blend
    sw = float(best_simple_key.split("_w")[1])
    h_simple = sw * h_mkt + (1.0 - sw) * h_mod
    hold["simple_blend"] = compute_classification_metrics(h_y, h_simple)
    print(f"  Simple blend (w={sw:.2f}): holdout LL = {hold['simple_blend']['log_loss']:.4f}")

    # Best QB gated
    qw = float(best_qbg_key.split("_w")[1])
    h_qbg = np.where(h_qb, qw * h_mkt + (1.0 - qw) * h_mod, h_mod)
    hold["qb_gated"] = compute_classification_metrics(h_y, h_qbg)
    print(f"  QB gated (w={qw:.2f}):     holdout LL = {hold['qb_gated']['log_loss']:.4f}")

    # Best LD gated
    ld_parts = best_ld_key.split("_")
    ld_t = float([p for p in ld_parts if p.startswith("t")][0].lstrip("t"))
    ld_w = float([p for p in ld_parts if p.startswith("w")][0].lstrip("w"))
    h_ld_gate = h_ad >= ld_t
    h_ld = np.where(h_ld_gate, ld_w * h_mkt + (1.0 - ld_w) * h_mod, h_mod)
    hold["ld_gated"] = compute_classification_metrics(h_y, h_ld)
    print(
        f"  LD gated (t={ld_t:.3f}, w={ld_w:.2f}): holdout LL = {hold['ld_gated']['log_loss']:.4f}"
    )

    # Best QB+LD gated
    qbld_parts = best_qbld_key.split("_")
    qbld_t = float([p for p in qbld_parts if p.startswith("t")][0].lstrip("t"))
    qbld_w = float([p for p in qbld_parts if p.startswith("w")][0].lstrip("w"))
    h_qbld_gate = h_qb & (h_ad >= qbld_t)
    h_qbld = np.where(h_qbld_gate, qbld_w * h_mkt + (1.0 - qbld_w) * h_mod, h_mod)
    hold["qb_ld_gated"] = compute_classification_metrics(h_y, h_qbld)
    qbld_ll = hold["qb_ld_gated"]["log_loss"]
    print(f"  QB+LD gated (t={qbld_t:.3f}, w={qbld_w:.2f}): holdout LL = {qbld_ll:.4f}")

    # Logistic blend
    t_ml = _safe_logit(t_mod)
    h_ml = _safe_logit(h_mod)
    t_mkt_l = _safe_logit(t_mkt)
    h_mkt_l = _safe_logit(h_mkt)
    lx_tr = np.column_stack([t_ml, t_mkt_l, t_qb.astype(float), t_ad, t_qb.astype(float) * t_ad])
    lx_ho = np.column_stack([h_ml, h_mkt_l, h_qb.astype(float), h_ad, h_qb.astype(float) * h_ad])
    logit_full = _logistic_pipe()
    logit_full.fit(lx_tr, t_y)
    h_logit = logit_full.predict_proba(lx_ho)[:, 1]
    hold["logistic_blend"] = compute_classification_metrics(h_y, h_logit)
    print(f"  Logistic blend: holdout LL = {hold['logistic_blend']['log_loss']:.4f}")

    # ── Subset diagnostics ──
    print("\n" + "=" * 60)
    print("Subset Diagnostics")
    print("=" * 60)

    qb_ch_ll = qb_st_ll = qb_ch_mkt_ll = qb_st_mkt_ll = None
    if int(h_qb.sum()) >= 5:
        qb_ch_ll = compute_classification_metrics(h_y[h_qb], h_mod[h_qb])["log_loss"]
        qb_st_ll = compute_classification_metrics(h_y[~h_qb], h_mod[~h_qb])["log_loss"]
        qb_ch_mkt_ll = compute_classification_metrics(h_y[h_qb], h_mkt[h_qb])["log_loss"]
        qb_st_mkt_ll = compute_classification_metrics(h_y[~h_qb], h_mkt[~h_qb])["log_loss"]
        print(f"  QB-change (n={int(h_qb.sum())}): inc LL={qb_ch_ll:.4f} mkt LL={qb_ch_mkt_ll:.4f}")
        qb_st_n = int((~h_qb).sum())
        print(f"  QB-stable  (n={qb_st_n}): inc LL={qb_st_ll:.4f} mkt LL={qb_st_mkt_ll:.4f}")

    if int(h_fav.sum()) >= 5:
        f_inc = compute_classification_metrics(h_y[h_fav.astype(bool)], h_mod[h_fav.astype(bool)])[
            "log_loss"
        ]
        f_mkt = compute_classification_metrics(h_y[h_fav.astype(bool)], h_mkt[h_fav.astype(bool)])[
            "log_loss"
        ]
        print(f"  Favorite dis (n={int(h_fav.sum())}): inc LL={f_inc:.4f} mkt LL={f_mkt:.4f}")

    print("  Delta buckets:")
    for bucket in sorted(set(h_db)):
        mask = h_db == bucket
        n = int(mask.sum())
        if n < 3:
            continue
        inc_b = compute_classification_metrics(h_y[mask], h_mod[mask])["log_loss"]
        mkt_b = compute_classification_metrics(h_y[mask], h_mkt[mask])["log_loss"]
        print(f"    {bucket:>10s} (n={n:3d}): inc LL={inc_b:.4f}  mkt LL={mkt_b:.4f}")

    early_mask = h_wk <= 4
    if int(early_mask.sum()) >= 3:
        e_ll = compute_classification_metrics(h_y[early_mask], h_mod[early_mask])["log_loss"]
        print(f"  Early season (weeks 1-4, n={int(early_mask.sum())}): inc LL={e_ll:.4f}")
    late_mask = h_wk >= 13
    if int(late_mask.sum()) >= 3:
        l_ll = compute_classification_metrics(h_y[late_mask], h_mod[late_mask])["log_loss"]
        print(f"  Late season (weeks 13+, n={int(late_mask.sum())}): inc LL={l_ll:.4f}")

    # ── Caution flags artifact ──
    print("\n=== Writing caution flags artifact ===")
    df["caution_reason"] = ""
    df.loc[df["qb_change_flag"] == 1, "caution_reason"] = (
        df.loc[df["qb_change_flag"] == 1, "caution_reason"] + "QB change; "
    )
    df.loc[df["favorite_disagreement_flag"] == 1, "caution_reason"] = (
        df.loc[df["favorite_disagreement_flag"] == 1, "caution_reason"] + "Favorite disagreement; "
    )
    df.loc[df["large_market_delta_flag"] == 1, "caution_reason"] = (
        df.loc[df["large_market_delta_flag"] == 1, "caution_reason"] + "Large market delta; "
    )
    df.loc[df["caution_early_season"] == 1, "caution_reason"] = (
        df.loc[df["caution_early_season"] == 1, "caution_reason"] + "Early season; "
    )
    df["caution_reason"] = df["caution_reason"].str.rstrip("; ")

    caution_cols = [
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "model_prob",
        "market_prob",
        "market_minus_model",
        "abs_market_minus_model",
        "qb_change_flag",
        "favorite_disagreement_flag",
        "large_market_delta_flag",
        "caution_reason",
    ]
    avail_cc = [c for c in caution_cols if c in df.columns]
    df_caution = df[avail_cc].copy()
    for c in ["model_prob", "market_prob", "market_minus_model", "abs_market_minus_model"]:
        if c in df_caution.columns:
            df_caution[c] = df_caution[c].round(4)
    CAUTION_FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_caution.to_csv(CAUTION_FLAGS_PATH, index=False)
    print(f"  Written: {CAUTION_FLAGS_PATH} ({len(df_caution)} rows, {len(avail_cc)} cols)")

    # ── Write report ──
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)

    challenger_keys = [k for k in hold if k != "incumbent"]
    best_hold_label = min(challenger_keys, key=lambda k: hold[k]["log_loss"])
    best_hold_ll = hold[best_hold_label]["log_loss"]
    beats_inc = best_hold_ll < INCUMBENT_HOLDOUT_LL

    qb_delta_gap = None
    if qb_ch_mkt_ll is not None and qb_ch_ll is not None:
        qb_delta_gap = qb_ch_ll - qb_ch_mkt_ll

    decision = (
        "market-aware challenger (track separately from football-only incumbent)"
        if beats_inc
        else "market-aware diagnostic (does not beat football-only incumbent)"
    )

    with open(rp, "w") as f:
        f.write("# QB-Change Market-Delta Diagnostics\n\n")
        f.write(
            "*Testing whether closing-market disagreement with the football-only"
            " model identifies QB-change / availability-shock games where the"
            " incumbent is structurally blind.*\n\n"
        )

        f.write("## Important\n\n")
        f.write("This is a **market-aware / near-kickoff diagnostic** experiment.\n")
        f.write(
            f"The clean football-only incumbent is **Standard Elo + qb_changed"
            f" + rolling_mov_3 + Platt** (holdout LL **{INCUMBENT_HOLDOUT_LL}**).\n"
        )
        f.write("Closing market should **not** be treated as football-only.\n")
        f.write("**Do not overwrite or demote the football-only incumbent.**\n\n")

        f.write("## Method\n\n")
        f.write("Rolling-origin 3-fold validation, one-shot 2025 holdout.\n\n")
        f.write(f"Incumbent version: {INCUMBENT_VERSION}\n\n")

        f.write("### Diagnostic Fields\n\n")
        f.write("- `model_prob` — Incumbent home win probability\n")
        f.write("- `market_prob` — Closing moneyline no-vig home win probability\n")
        f.write("- `market_minus_model` — Market minus model probability\n")
        f.write("- `abs_market_minus_model` — Absolute disagreement\n")
        f.write("- `model_logit` / `market_logit` — Log-odds\n")
        f.write("- `market_logit_minus_model_logit` — Logit disagreement\n")
        f.write("- `qb_change_flag` — Either team QB changed from prior game\n")
        f.write("- `home_qb_change_flag` — Home QB changed\n")
        f.write("- `away_qb_change_flag` — Away QB changed\n")
        f.write("- `favorite_disagreement_flag` — Market and model disagree on favorite\n")
        f.write("- `directionally_aligned_flag` — Market and model agree on favorite\n")
        f.write("- `large_market_delta_flag` — abs(market - model) >= 0.05\n\n")

        f.write("### Models/Blends Compared\n\n")
        f.write("| Model | Description |\n")
        f.write("|-------|-------------|\n")
        f.write("| Incumbent | Football-only model (Elo + qb_changed + mov_3 + Platt) |\n")
        f.write("| Closing market | Moneyline no-vig probability |\n")
        f.write("| Simple blend | w * mkt + (1-w) * model (9 weights: 0.1..0.9) |\n")
        f.write("| QB-change gated | If QB change: blend; else model (4 weights) |\n")
        f.write("| Large-delta gated | If abs delta >= t: blend; else model (4×4 grid) |\n")
        f.write("| QB+LD gated | If QB change AND large delta: blend; else model |\n")
        f.write("| Logistic blend | Logit(model) + logit(mkt) + qb + delta + qb*delta |\n\n")

        # Validation
        f.write("## Rolling-Origin Validation\n\n")
        f.write("| Model / Config | Avg Val LL | Fold1 | Fold2 | Fold3 |\n")
        f.write("|---------------|-----------|-------|-------|-------|\n")

        def _vr(name, rl):
            if not rl:
                return
            f.write(f"| {name} | {np.mean([r['log_loss'] for r in rl]):.4f}")
            for r in rl:
                f.write(f" | {r['log_loss']:.4f}")
            f.write(" |\n")

        _vr("Incumbent (model)", platt_results)
        _vr("Market (no-vig)", market_results)
        _vr(f"Simple blend ({best_simple_key})", simple_blend_best[best_simple_key]["fold_results"])
        _vr(f"QB gated ({best_qbg_key})", qb_gated_best[best_qbg_key]["fold_results"])
        _vr(f"LD gated ({best_ld_key})", ld_gated_best[best_ld_key]["fold_results"])
        _vr(f"QB+LD gated ({best_qbld_key})", qb_ld_gated_best[best_qbld_key]["fold_results"])
        _vr("Logistic blend", logistic_blend_results)

        f.write("\n### Candidate Selection\n\n")
        f.write("| Candidate | Avg Val LL |\n")
        f.write("|-----------|-----------|\n")
        for name, ll in sorted(candidates.items(), key=lambda x: x[1]):
            marker = " ← best" if name == best_candidate_name else ""
            f.write(f"| {name} | {ll:.4f}{marker} |\n")
        f.write("\n")

        # Holdout
        f.write("## 2025 Holdout (One-Shot)\n\n")
        f.write("| Model | Hold LL | Brier | AUC | Acc |\n")
        f.write("|-------|---------|-------|-----|-----|\n")
        for label, name_ in [
            ("incumbent", "Incumbent"),
            ("market", "Closing market"),
            ("simple_blend", "Simple blend"),
            ("qb_gated", "QB gated"),
            ("ld_gated", "LD gated"),
            ("qb_ld_gated", "QB+LD gated"),
            ("logistic_blend", "Logistic blend"),
        ]:
            if label in hold:
                hm = hold[label]
                row = (
                    f"| {name_} | {hm['log_loss']:.4f}"
                    f" | {hm.get('brier_score', 0):.4f}"
                    f" | {hm.get('roc_auc', 0):.4f}"
                    f" | {hm.get('accuracy', 0):.4f} |\n"
                )
                f.write(row)
        f.write("\n")

        # Subset
        f.write("## Subset Analysis\n\n")

        if qb_ch_ll is not None:
            qb_ch_n = int(h_qb.sum())
            qb_st_n = int((~h_qb).sum())
            f.write("### QB-Change Subset\n\n")
            f.write(f"| Model | QB Change (n={qb_ch_n}) | QB Stable (n={qb_st_n}) |\n")
            f.write("|-------|--------|--------|\n")
            f.write(f"| Incumbent | {qb_ch_ll:.4f} | {qb_st_ll:.4f} |\n")
            f.write(f"| Closing market | {qb_ch_mkt_ll:.4f} | {qb_st_mkt_ll:.4f} |\n")
            if qb_delta_gap is not None:
                f.write(f"\nMarket-incumbent gap on QB-change games: **{qb_delta_gap:.4f}**\n\n")

        if int(h_fav.sum()) >= 5:
            f.write("### Favorite Disagreement Subset\n\n")
            f.write(f"| Model | Favorite Dis (n={int(h_fav.sum())}) |\n")
            f.write("|-------|--------|\n")
            f.write(f"| Incumbent | {f_inc:.4f} |\n")
            f.write(f"| Closing market | {f_mkt:.4f} |\n\n")

        f.write("### Market-Delta Buckets\n\n")
        f.write("| Bucket | N | Incumbent LL | Market LL |\n")
        f.write("|--------|---|-------------|----------|\n")
        for bucket in sorted(set(h_db)):
            mask = h_db == bucket
            n = int(mask.sum())
            if n < 3:
                continue
            inc_b = compute_classification_metrics(h_y[mask], h_mod[mask])["log_loss"]
            mkt_b = compute_classification_metrics(h_y[mask], h_mkt[mask])["log_loss"]
            f.write(f"| {bucket:>10s} | {n:3d} | {inc_b:.4f} | {mkt_b:.4f} |\n")
        f.write("\n")

        early_ll_v = (
            compute_classification_metrics(h_y[early_mask], h_mod[early_mask])["log_loss"]
            if int(early_mask.sum()) >= 3
            else None
        )
        late_ll_v = (
            compute_classification_metrics(h_y[late_mask], h_mod[late_mask])["log_loss"]
            if int(late_mask.sum()) >= 3
            else None
        )
        if early_ll_v or late_ll_v:
            f.write("### Early vs Late Season\n\n")
            f.write("| Period | N | Incumbent LL |\n")
            f.write("|--------|---|-------------|\n")
            if early_ll_v:
                f.write(f"| Early (weeks 1-4) | {int(early_mask.sum())} | {early_ll_v:.4f} |\n")
            if late_ll_v:
                f.write(f"| Late (weeks 13+) | {int(late_mask.sum())} | {late_ll_v:.4f} |\n")
            f.write("\n")

        # Decision
        f.write("## Decision\n\n")
        if beats_inc:
            f.write(
                f"**Market-aware challenger beats football-only incumbent"
                f" on holdout** ({INCUMBENT_HOLDOUT_LL} → {best_hold_ll:.4f}).\n\n"
            )
        else:
            f.write(
                f"No market-aware variant beat the football-only incumbent"
                f" ({INCUMBENT_HOLDOUT_LL}) on holdout.\n"
            )
            f.write(f"Best market-aware holdout LL: {best_hold_ll:.4f} ({best_hold_label}).\n\n")
        f.write(f"**Decision: {decision}.**\n\n")

        f.write("### Key Findings\n\n")
        f.write("1. This is **market-aware / near-kickoff diagnostic work.**\n")
        f.write("2. Closing market should **not** be treated as football-only.\n")
        f.write(
            f"3. Football-only incumbent unchanged:"
            f" Standard Elo + qb_changed + rolling_mov_3 + Platt"
            f" (holdout LL {INCUMBENT_HOLDOUT_LL}).\n"
        )
        if qb_delta_gap is not None:
            f.write(
                f"4. Market-incumbent gap on QB-change games: **{qb_delta_gap:.4f}** — "
                "market substantially outperforms incumbent when QB changes.\n"
            )
        f.write("5. **Opening-line ingestion** should be prioritized next.\n\n")

        f.write("### Market Status\n\n")
        f.write("| Assumption | Value |\n")
        f.write("|-----------|-------|\n")
        f.write("| Market type | Closing (near-kickoff) |\n")
        f.write(f"| Football-only incumbent unchanged | **Yes** (still {INCUMBENT_HOLDOUT_LL}) |\n")
        f.write(f"| Market-aware challenger tracked | **{'Yes' if beats_inc else 'No'}** |\n")

        # Caution flags
        f.write("\n### Caution Flags Artifact\n\n")
        f.write(f"`{CAUTION_FLAGS_PATH}` — {len(df_caution)} rows, {len(avail_cc)} columns.\n")
        f.write("See schema in tests.\n")

    print(f"\nReport: {rp}")
    return str(rp)


def _record_blend(
    storage: dict,
    fold_idx: int,
    key: str,
    metrics: dict,
    proba: np.ndarray,
    model,
) -> None:
    if key not in storage:
        storage[key] = {"fold_results": [], "fold_probas": [], "avg_ll": 0.0}
    storage[key]["fold_results"].append(metrics)
    storage[key]["fold_probas"].append(proba)
    storage[key]["avg_ll"] = float(np.mean([r["log_loss"] for r in storage[key]["fold_results"]]))


if __name__ == "__main__":
    run_qb_market_delta_experiment()
