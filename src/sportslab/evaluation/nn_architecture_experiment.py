"""Neural-network architecture exploration vs the v3.1.0 MLP calibrator.

Genuinely-different architectures (not just MLP hyperparameter tuning):

  1. ResNet-tabular (Gorishniy 2025): residual blocks + LayerNorm. Tests
     whether skip connections / normalization help at this data size.
  2. RealMLP-style seed ensemble: 5 seeds, kaiming init, cosine LR, full-batch.
     RealMLP's main edge is ensembling a few well-init'd MLPs.
  3. Deeper/wider MLPs (4x32, 5x64) to test whether capacity helps or hurts.

All selection on rolling-origin validation (3 folds); 2025 holdout scored once.
Promotion requires beating v3.1.0 on BOTH validation and holdout by >= 0.001.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss

import sportslab.evaluation.deep_mlp as dm

INCUMBENT_VAL_LL = 0.6279
INCUMBENT_HOLDOUT_LL = 0.6151


# (label, kind, hidden, arch, activation, dropout, wd, sched, opt, ensemble, init, full_batch)
CONFIGS = [
    # faithful v3.1.0 baseline
    ("v3.1.0_mlp", "incumbent", (16, 16, 16), "mlp", "relu", 0.1, 1e-4, "none", "adam", False, "default", True),  # noqa: E501
    # ResNet-tabular variants  # noqa: E501
    ("resnet_3x64", "incumbent", (64, 64, 64), "resnet", "relu", 0.1, 1e-4, "cosine", "adamw", False, "kaiming", True),  # noqa: E501
    ("resnet_4x128", "incumbent", (128, 128, 128, 128), "resnet", "relu", 0.1, 1e-4, "cosine", "adamw", False, "kaiming", True),  # noqa: E501
    ("resnet_gelu_3x64", "incumbent", (64, 64, 64), "resnet", "gelu", 0.1, 1e-4, "cosine", "adamw", False, "kaiming", True),  # noqa: E501
    # RealMLP 5-seed ensemble (proper init + cosine + full-batch)  # noqa: E501
    ("realmlp_ensemble5", "incumbent", (16, 16, 16), "mlp", "relu", 0.1, 1e-4, "cosine", "adamw", True, "kaiming", True),  # noqa: E501
    ("realmlp_ensemble7", "incumbent", (32, 32), "mlp", "relu", 0.1, 1e-4, "cosine", "adamw", True, "kaiming", True),  # noqa: E501
    # deeper / wider plain MLPs
    ("mlp_4x32", "incumbent", (32, 32, 32, 32), "mlp", "relu", 0.1, 1e-4, "cosine", "adamw", False, "kaiming", True),  # noqa: E501
    ("mlp_5x64", "incumbent", (64, 64, 64, 64, 64), "mlp", "relu", 0.1, 1e-4, "cosine", "adamw", False, "kaiming", True),  # noqa: E501
]


def _run_config(df, yv, xf, gate, ha, aa, cfg, do_holdout):
    (label, kind, hidden, arch, act, drop, wd, sched, opt, ensemble,
     init, full_batch) = cfg
    xk = dm.feature_matrix(df, kind) if kind != "incumbent" else xf

    def _fit(train_idx, all_idx):
        if ensemble:
            raw = dm.predict_mlp_ensemble(
                xk[train_idx], yv[train_idx], xk[all_idx],
                seeds=(0, 1, 2, 3, 4, 5, 6)[:7] if "7" in label else (0, 1, 2, 3, 4),
                hidden=hidden, activation=act, dropout=drop, weight_decay=wd,
                schedule=sched, optimizer=opt, n_epochs=200, early_stopping=False,
                scaler="standard", full_batch=full_batch, init=init, arch=arch,
            )
        else:
            fn, _ = dm.train_mlp(
                xk[train_idx], yv[train_idx], hidden=hidden, activation=act,
                dropout=drop, weight_decay=wd, schedule=sched, optimizer=opt,
                n_epochs=200, early_stopping=False, scaler="standard",
                full_batch=full_batch, init=init, arch=arch,
            )
            raw = fn(xk[all_idx])
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


def run_nn_architecture_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/nn_architecture.md",
) -> str:
    print("=== NN Architecture Exploration ===")
    df = dm.build_feature_table(ft_path)
    yv = df[dm.TARGET_COLUMN].astype(float).values
    gate = dm._build_gate(df)
    ha = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    aa = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    xf = dm.feature_matrix(df, "incumbent")
    print(f"  Eligible games: {len(df)}")

    base = "v3.1.0_mlp"
    val, hold = {}, {}
    for cfg in CONFIGS:
        val[cfg[0]] = _run_config(df, yv, xf, gate, ha, aa, cfg, do_holdout=False)
    for cfg in CONFIGS:
        hold[cfg[0]] = _run_config(df, yv, xf, gate, ha, aa, cfg, do_holdout=True)

    iv, ih = val[base], hold[base]
    print("\n--- Validation (avg rolling-origin) ---")
    for cfg in CONFIGS:
        print(f"  {cfg[0]:18s} val={val[cfg[0]]:.4f} Δ={val[cfg[0]]-iv:+.4f}")
    print("\n--- 2025 Holdout ---")
    for cfg in CONFIGS:
        print(f"  {cfg[0]:18s} hold={hold[cfg[0]]:.4f} Δ={hold[cfg[0]]-ih:+.4f}")

    best_v = min(val.values())
    best_h = min(hold.values())
    best_vn = min(val, key=val.get)
    best_hn = min(hold, key=hold.get)
    promoted = (best_v < iv - dm.MIN_PROMOTION_DELTA) and (best_h < ih - dm.MIN_PROMOTION_DELTA)

    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        w = f.write
        w("# NN Architecture Exploration\n\n")
        w("Tests genuinely-different NN architectures (ResNet-tabular, RealMLP "
          "seed ensemble, deeper/wider MLPs) vs the v3.1.0 MLP. Baseline = "
          f"v3.1.0 (val {iv:.4f}, holdout {ih:.4f}).\n\n")
        w("| Config | Arch | Val LL | Δ Val | Holdout LL | Δ Holdout |\n")
        w("|---------|------|--------|-------|-----------|-----------|\n")
        for cfg in CONFIGS:
            w(f"| {cfg[0]} | {cfg[3]} | {val[cfg[0]]:.4f} | {val[cfg[0]]-iv:+.4f} | "
              f"{hold[cfg[0]]:.4f} | {hold[cfg[0]]-ih:+.4f} |\n")
        w("\n## Decision\n\n")
        if promoted:
            w(f"**✅ PROMOTED** — {best_vn} / {best_hn} beat the MLP on both.\n")
        else:
            w("**❌ REJECTED** — no architecture beats the v3.1.0 MLP on both val "
              "and holdout by >= 0.001.\n\n")
            w(f"Best validation: {best_vn} ({best_v:.4f}, Δ={best_v-iv:+.4f})\n")
            w(f"Best holdout: {best_hn} ({best_h:.4f}, Δ={best_h-ih:+.4f})\n")
        w("\n## Interpretation\n\n")
        w("- ResNet-tabular: tests skip-connections + LayerNorm vs plain MLP.\n")
        w("- RealMLP ensemble: 5-7 seed ensemble with kaiming init + cosine LR.\n")
        w("- Deeper/wider MLPs: capacity test (overfit risk at ~7k rows).\n")
    print(f"  Report: {rp}")
    return str(report_path)


if __name__ == "__main__":
    run_nn_architecture_experiment()
