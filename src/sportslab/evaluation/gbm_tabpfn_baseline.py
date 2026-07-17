"""LightGBM + TabPFN baselines for NFL game-outcome prediction.

Purpose: confirm whether a strong gradient-boosted tree (LightGBM) or the
foundation-model tabular classifier (TabPFN) can beat the v3.1.0 MLP
calibrator (holdout 0.6151) on this small pre-2021-expanded dataset.

All hyperparameters selected on rolling-origin validation (3 folds). 2025
holdout scored exactly once. Promotion requires beating the v3.1.0 baseline on
BOTH validation and holdout by >= MIN_PROMOTION_DELTA.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss

import sportslab.evaluation.deep_mlp as dm

INCUMBENT_VAL_LL = 0.6279
INCUMBENT_HOLDOUT_LL = 0.6151


def _run_gbm(df, yv, xf, gate, ha, aa, params, do_holdout):
    import lightgbm as lgb

    def _fit(train_idx, all_idx):
        if train_idx.dtype == bool:
            train_idx = np.where(train_idx)[0]
        if isinstance(all_idx, slice):
            all_idx = np.arange(len(xf))
        # 20% eval split for early stopping (does not leak into all_idx preds)
        n = len(train_idx)
        n_eval = max(1, int(0.2 * n))
        perm = np.random.RandomState(0).permutation(n)
        tr_idx, ev_idx = train_idx[perm[n_eval:]], train_idx[perm[:n_eval]]
        dtr = lgb.Dataset(xf[tr_idx], yv[tr_idx].astype(int))
        dev = lgb.Dataset(xf[ev_idx], yv[ev_idx].astype(int))
        bst = lgb.train(
            params, dtr, num_boost_round=400, valid_sets=[dev],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        raw = bst.predict(xf[all_idx])
        return dm._apply_overlay(raw, gate, ha, aa)

    if do_holdout:
        tr = (df["season"] < dm.HOLDOUT_SEASON).values
        va = (df["season"] == dm.HOLDOUT_SEASON).values
        pp = _fit(tr, slice(None))
        vy = yv[va]
        valid = ~np.isnan(vy)
        return float(sk_log_loss(vy[valid].astype(int), pp[va][valid]))
    fold_lls = []
    for train_s, val_s in dm.ROLLING_FOLDS:
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values
        if tr.sum() == 0 or va.sum() == 0:
            fold_lls.append(1.0)
            continue
        pp = _fit(tr, slice(None))
        vy = yv[va]
        valid = ~np.isnan(vy)
        fold_lls.append(float(sk_log_loss(vy[valid].astype(int), pp[va][valid])))
    return round(float(np.mean(fold_lls)), 4)


def _run_tabpfn(df, yv, xf, gate, ha, aa, do_holdout):
    from tabpfn import TabPFNClassifier

    def _fit(train_idx, all_idx):
        if train_idx.dtype == bool:
            train_idx = np.where(train_idx)[0]
        if isinstance(all_idx, slice):
            all_idx = np.arange(len(xf))
        clf = TabPFNClassifier(
            n_estimators=4, device="cuda" if dm.DEVICE.type == "cuda" else "cpu",
            fit_mode="fit_preprocessors", ignore_pretraining_limits=True,
        )
        clf.fit(xf[train_idx], yv[train_idx].astype(int))
        raw = clf.predict_proba(xf[all_idx])[:, 1]
        return dm._apply_overlay(raw, gate, ha, aa)

    if do_holdout:
        tr = (df["season"] < dm.HOLDOUT_SEASON).values
        va = (df["season"] == dm.HOLDOUT_SEASON).values
        pp = _fit(tr, slice(None))
        vy = yv[va]
        valid = ~np.isnan(vy)
        return float(sk_log_loss(vy[valid].astype(int), pp[va][valid]))
    fold_lls = []
    for train_s, val_s in dm.ROLLING_FOLDS:
        tr = df["season"].isin(train_s).values
        va = (df["season"] == val_s).values
        if tr.sum() == 0 or va.sum() == 0:
            fold_lls.append(1.0)
            continue
        pp = _fit(tr, slice(None))
        vy = yv[va]
        valid = ~np.isnan(vy)
        fold_lls.append(float(sk_log_loss(vy[valid].astype(int), pp[va][valid])))
    return round(float(np.mean(fold_lls)), 4)


def run_gbm_baseline_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/gbm_tabpfn_baseline.md",
) -> str:
    print("=== LightGBM + TabPFN Baseline ===")
    df = dm.build_feature_table(ft_path)
    yv = df[dm.TARGET_COLUMN].astype(float).values
    gate = dm._build_gate(df)
    ha = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    aa = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    print(f"  Eligible games: {len(df)}")

    base = "v3.1.0_mlp"
    configs = {
        base: ("mlp", None),
        "lgbm_default": ("lgbm", {
            "objective": "binary", "metric": "binary_logloss", "verbose": -1,
            "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 20,
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
        }),
        "lgbm_deep": ("lgbm", {
            "objective": "binary", "metric": "binary_logloss", "verbose": -1,
            "learning_rate": 0.02, "num_leaves": 63, "min_data_in_leaf": 10,
            "max_depth": -1, "feature_fraction": 1.0, "bagging_fraction": 1.0,
        }),
        "tabpfn": ("tabpfn", None),
    }

    xf_inc = dm.feature_matrix(df, "incumbent")
    results = {}
    for name, (kind, params) in configs.items():
        if kind == "mlp":
            # faithful v3.1.0 reproduction (already validated = ~0.6151)
            fn, _ = dm.train_mlp(
                xf_inc[df["season"] < dm.HOLDOUT_SEASON], yv[df["season"] < dm.HOLDOUT_SEASON],
                hidden=(16, 16, 16), weight_decay=1e-4, schedule="none", optimizer="adam",
                n_epochs=200, early_stopping=False, scaler="standard",
                full_batch=True, init="default",
            )
            pp = dm._apply_overlay(fn(xf_inc), gate, ha, aa)
            va = (df["season"] == dm.HOLDOUT_SEASON).values
            vy = yv[va]
            valid = ~np.isnan(vy)
            results[name] = {
                "val": INCUMBENT_VAL_LL,
                "hold": float(sk_log_loss(vy[valid].astype(int), pp[va][valid])),
            }
            continue
        if kind == "tabpfn":
            # TabPFN v8 requires a PriorLabs API token — not available offline.
            results[name] = {"val": float("nan"), "hold": float("nan"), "blocked": True}
            continue
        xfk = dm.feature_matrix(df, "incumbent")
        val = _run_gbm(df, yv, xfk, gate, ha, aa, params, False)
        hold = _run_gbm(df, yv, xfk, gate, ha, aa, params, True)
        results[name] = {"val": val, "hold": hold}

    iv = results[base]["val"]
    ih = results[base]["hold"]
    print("\n--- Validation (avg rolling-origin) ---")
    for name in configs:
        print(f"  {name:18s} val={results[name]['val']:.4f} Δ={results[name]['val']-iv:+.4f}")
    print("\n--- 2025 Holdout ---")
    for name in configs:
        print(f"  {name:18s} hold={results[name]['hold']:.4f} Δ={results[name]['hold']-ih:+.4f}")

    best_v = min(results[n]["val"] for n in configs)
    best_h = min(results[n]["hold"] for n in configs)
    best_vn = min(configs, key=lambda n: results[n]["val"])
    best_hn = min(configs, key=lambda n: results[n]["hold"])
    promoted = (best_v < iv - dm.MIN_PROMOTION_DELTA) and (best_h < ih - dm.MIN_PROMOTION_DELTA)

    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        w = f.write
        w("# LightGBM + TabPFN Baseline\n\n")
        w("Confirms whether a strong GBM or foundation-model tabular classifier beats the "
          f"v3.1.0 MLP (baseline val {iv:.4f}, holdout {ih:.4f}).\n\n")
        w("| Model | Val LL | Δ Val | Holdout LL | Δ Holdout |\n")
        w("|-------|--------|-------|-----------|-----------|\n")
        for name in configs:
            r = results[name]
            if r.get("blocked"):
                w(f"| {name} | BLOCKED | — | BLOCKED | — |\n")
            else:
                w(f"| {name} | {r['val']:.4f} | {r['val']-iv:+.4f} | "
                  f"{r['hold']:.4f} | {r['hold']-ih:+.4f} |\n")
        w("\n## Decision\n\n")
        if promoted:
            w(f"**✅ PROMOTED** — {best_vn} / {best_hn} beat the MLP on both.\n")
        else:
            w("**❌ REJECTED** — no runnable model beats the MLP on both val and holdout.\n\n")
            w(f"Best validation: {best_vn} ({best_v:.4f}, Δ={best_v-iv:+.4f})\n")
            w(f"Best holdout: {best_hn} ({best_h:.4f}, Δ={best_h-ih:+.4f})\n")
        w("\n## Interpretation\n\n")
        w("- If LightGBM ≈ MLP: the MLP gain is real (not just 'logistic was weak').\n")
        w("- If LightGBM > MLP: trees are the better calibrator here; promote GBM.\n")
        w("- TabPFN v8 requires a PriorLabs API token (not available offline) — diagnostic "
          "blocked in this environment.\n")
    print(f"  Report: {rp}")
    return str(report_path)


if __name__ == "__main__":
    run_gbm_baseline_experiment()
