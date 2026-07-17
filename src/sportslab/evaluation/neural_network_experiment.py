"""Neural network challenger experiment.

Tests whether a small feed-forward neural network (PyTorch) can learn a more
flexible mapping from Elo/qb/mov features to win probability than the
logistic Platt calibration used by the incumbent.

Follows the canonical fold-safe rolling-origin pattern:
  - Features built identically to the incumbent pipeline.
  - Frozen QB overlay (gated logit-space adjustment) applied on top.
  - Selection by average rolling-origin validation log loss.
  - Final 2025 holdout scored ONCE.

Hypothesis: a 1-2 hidden layer MLP with dropout/weight-decay may capture
nonlinear interactions that logistic regression cannot, at this dataset size.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.preprocessing import StandardScaler

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, TARGET_COLUMN
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features


def _ece(probs, y_true, n_bins=10):
    """Expected calibration error (weighted mean |confidence - accuracy|)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += abs(probs[mask].mean() - y_true[mask].astype(float).mean()) * mask.sum()
    return ece / total

BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2
MIN_PROMOTION_DELTA = 0.001
SEED = 42
QBG = 1.0
QBC = 40
E2L = np.log(10) / 400.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INCUMBENT_FEATS = [
    "home_qb_changed",
    "away_qb_changed",
    "home_rolling_mov_3",
    "away_rolling_mov_3",
]

BASE_FEATS = ["elo_prob"] + INCUMBENT_FEATS

# (name, description, hidden sizes, dropout, weight_decay)
NN_VARIANTS = [
    ("incumbent", "Logistic Platt (champion)", None, 0.0, 0.0),
    ("mlp_16", "MLP 1 hidden (16) dp=0.1", (16,), 0.1, 1e-4),
    ("mlp_32_16", "MLP 2 hidden (32,16) dp=0.1", (32, 16), 0.1, 1e-4),
    ("mlp_64", "MLP 1 hidden (64) dp=0.2", (64,), 0.2, 1e-3),
    ("mlp_32_16_wd", "MLP 2 hidden (32,16) dp=0.2 wd=1e-3", (32, 16), 0.2, 1e-3),
    ("mlp_16_16_16", "MLP 3 hidden (16,16,16) dp=0.1 wd=1e-4", (16, 16, 16), 0.1, 1e-4),
    ("mlp_64_32", "MLP 2 hidden (64,32) dp=0.2 wd=1e-3", (64, 32), 0.2, 1e-3),
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def _build_gate(df):
    hc = df.get("home_qb_changed", pd.Series(0)).values.astype(float)
    ac = df.get("away_qb_changed", pd.Series(0)).values.astype(float)
    hs = df.get("home_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    aws = df.get("away_qb_team_starts_pre", pd.Series(0.0)).fillna(0).values.astype(float)
    return (hc == 1) | (ac == 1) | (hs < 17) | (aws < 17)


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(x, y_in, hidden, dropout, weight_decay, epochs=200):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    scaler = StandardScaler()
    x_s = scaler.fit_transform(x.astype(np.float64))
    xt = torch.tensor(x_s, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_in.astype(np.float64), dtype=torch.float32, device=DEVICE)
    model = _MLP(xt.shape[1], hidden, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, scaler


def _mlp_proba(model, scaler, x):
    x_s = scaler.transform(x.astype(np.float64))
    xt = torch.tensor(x_s, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model(xt).cpu().numpy()
    return _sigmoid(logits)


def _train_logistic(x, y):
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    x_s = scaler.fit_transform(x.astype(np.float64))
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(x_s, y.astype(int))
    return lr, scaler


def _logistic_proba(lr, scaler, x):
    x_s = scaler.transform(x.astype(np.float64))
    return lr.predict_proba(x_s)[:, 1]


def _apply_overlay(probs, gate, ha, aa):
    fl = _logit(np.clip(probs, 1e-15, 1 - 1e-15))
    fl = fl + QBG * (np.clip(ha, -QBC, QBC) - np.clip(aa, -QBC, QBC)) * E2L * gate.astype(float)
    return _sigmoid(fl)


def run_neural_network_experiment(
    ft_path: str = "data/features/nfl/feature_table.parquet",
    report_path: str = "reports/experiments/neural_network.md",
) -> str:
    print("=== Neural Network Challenger ===")
    fp = Path(ft_path)
    df_raw = pd.read_parquet(fp)

    overrides = build_team_regression_overrides(
        df_raw, preseason_regression=BEST_REG, qb_change_bonus=BEST_QB_BONUS
    )
    df = compute_elo_features(
        df_raw,
        k_factor=BEST_K,
        home_advantage=BEST_HFA,
        preseason_regression=BEST_REG,
        team_regression_overrides=overrides,
        decay_half_life=BEST_DECAY,
    )
    df = compute_qb_features(df)
    df = compute_qb_adjustments(df)
    df = compute_situational_features(df)

    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df["is_neutral"].fillna(False).values
    df = df[mask].copy().reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(float).values
    print(f"  Eligible games: {len(df)}")

    gate = _build_gate(df)
    ha = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    aa = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    elo_prob = df["elo_prob"].values.astype(float)

    def _feature_matrix(idx):
        return np.column_stack([elo_prob[idx]] + [df[c].values[idx] for c in INCUMBENT_FEATS])

    def _fit_predict(train_idx, all_idx, hidden, dropout, weight_decay):
        xtr = _feature_matrix(train_idx)
        xall = _feature_matrix(all_idx)
        if hidden is None:
            lr, scaler = _train_logistic(xtr, y[train_idx])
            raw = _logistic_proba(lr, scaler, xall)
        else:
            model, scaler = _train_mlp(xtr, y[train_idx], hidden, dropout, weight_decay)
            raw = _mlp_proba(model, scaler, xall)
        return _apply_overlay(raw, gate, ha, aa)

    val_results = {}
    hold_results = {}

    # Rolling-origin validation
    for name, _, hidden, dropout, wd in NN_VARIANTS:
        fold_lls = []
        for train_s, val_s in ROLLING_FOLDS:
            tr = df["season"].isin(train_s).values
            va = (df["season"] == val_s).values
            if tr.sum() == 0 or va.sum() == 0:
                fold_lls.append(1.0)
                continue
            pp = _fit_predict(tr, slice(None), hidden, dropout, wd)
            vy = y[va]
            valid = ~np.isnan(vy)
            fold_lls.append(float(sk_log_loss(vy[valid].astype(int), pp[va][valid])))
        val_results[name] = round(float(np.mean(fold_lls)), 4)

    # 2025 holdout (single fit on all pre-2025)
    tr = (df["season"] < HOLDOUT_SEASON).values
    va = (df["season"] == HOLDOUT_SEASON).values
    ece_results = {}
    for name, _, hidden, dropout, wd in NN_VARIANTS:
        pp = _fit_predict(tr, slice(None), hidden, dropout, wd)
        vy = y[va]
        valid = ~np.isnan(vy)
        hold_results[name] = float(sk_log_loss(vy[valid].astype(int), pp[va][valid]))
        ece_results[name] = float(_ece(pp[va][valid], vy[valid].astype(float)))

    iv = val_results["incumbent"]
    ih = hold_results["incumbent"]
    print("\n--- Validation (avg rolling-origin) ---")
    for name, _, _, _, _ in NN_VARIANTS:
        print(f"  {name:16s}  val={val_results[name]:.4f}  Δ={val_results[name]-iv:+.4f}")
    print("\n--- 2025 Holdout ---")
    for name, _, _, _, _ in NN_VARIANTS:
        print(f"  {name:16s}  hold={hold_results[name]:.4f}  Δ={hold_results[name]-ih:+.4f}")

    best_v = min(v for k, v in val_results.items() if k != "incumbent")
    best_v_name = min((k for k in val_results if k != "incumbent"), key=lambda k: val_results[k])
    best_h = min(v for k, v in hold_results.items() if k != "incumbent")
    best_h_name = min((k for k in hold_results if k != "incumbent"), key=lambda k: hold_results[k])

    promoted = (best_v < iv - MIN_PROMOTION_DELTA) and (best_h < ih - MIN_PROMOTION_DELTA)

    print(f"\n=== Writing report -> {report_path} ===")
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        w = f.write
        w("# Neural Network Challenger\n\n")
        w("Small feed-forward MLP (PyTorch) replacing the logistic Platt calibration, "
          "with the same features + frozen QB overlay.\n\n")
        w("| Model | Val LL | Δ Val | Holdout LL | Δ Holdout | Holdout ECE |\n")
        w("|-------|--------|-------|-----------|-----------|-------------|\n")
        for name, desc, _, _, _ in NN_VARIANTS:
            row = (
                f"| {name} ({desc}) | {val_results[name]:.4f} | "
                f"{val_results[name]-iv:+.4f} | {hold_results[name]:.4f} | "
                f"{hold_results[name]-ih:+.4f} | {ece_results[name]:.4f} |\n"
            )
            w(row)
        w("\n## Decision\n\n")
        if promoted:
            w("**✅ PROMOTED** — a neural variant beat the incumbent on both "
              f"validation ({best_v_name}) and holdout ({best_h_name}).\n\n")
        else:
            w("**❌ REJECTED** — no neural variant beat the incumbent on both "
              "validation and holdout by >= 0.001.\n\n")
            w(f"Best validation: {best_v_name} ({best_v:.4f}, Δ={best_v-iv:+.4f})\n")
            w(f"Best holdout: {best_h_name} ({best_h:.4f}, Δ={best_h-ih:+.4f})\n")
    print(f"  Report: {rp}")
    return str(report_path)


if __name__ == "__main__":
    run_neural_network_experiment()
