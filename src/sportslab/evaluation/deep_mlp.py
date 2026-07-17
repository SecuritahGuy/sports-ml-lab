"""Deep MLP research toolkit for NFL game-outcome prediction.

A properly-engineered PyTorch MLP trainer built per the small-data tabular-DL
literature (RealMLP / "Better by Default", 2024; Gorishniy et al. survey, 2025):
  - RobustScaler preprocessing (robust to outliers, smooth-clipped to ±3)
  - Kaiming/He init for ReLU, GELU option
  - AdamW with weight-decay sweep (1e-4 .. 1e-2)
  - cosine / one-cycle LR schedules + early stopping
  - deep ensembles (multiple seeds) for accuracy + calibration
  - temperature scaling on the validation fold (Platt/logistic special case)

All selection is done on rolling-origin validation folds; the 2025 holdout is
scored exactly once per experiment. Pregame-safe features only.

This module is the shared core for the PyTorch deep-dive experiment and the
LightGBM / TabPFN baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sportslab.evaluation.experiment_config import HOLDOUT_SEASON, ROLLING_FOLDS  # noqa: F401
from sportslab.evaluation.season_regression_experiment import (
    build_team_regression_overrides,
)
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, TARGET_COLUMN  # noqa: F401
from sportslab.features.qb import compute_qb_features
from sportslab.features.qb_adjustment import compute_qb_adjustments
from sportslab.features.ratings import compute_elo_features
from sportslab.features.situational import compute_situational_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Elo spine (matches incumbent / deep-dive branch)
BEST_K = 36
BEST_HFA = 40
BEST_REG = 0.1
BEST_DECAY = 32
BEST_QB_BONUS = 0.2

# Frozen QB overlay (unchanged from incumbent)
QBG = 1.0
QBC = 40
E2L = np.log(10) / 400.0

MIN_PROMOTION_DELTA = 0.001


# ── Feature definitions ────────────────────────────────────────────────

# Base incumbent 5 features (as raw columns; elo_prob added separately).
INCUMBENT_FEATS = [
    "home_qb_changed",
    "away_qb_changed",
    "home_rolling_mov_3",
    "away_rolling_mov_3",
]

# Elo-rich: add the raw rating gap (elo_diff) so the net can correct the
# logistic assumption baked into elo_prob.
ELO_RICH_FEATS = INCUMBENT_FEATS + ["elo_diff"]

# Antisymmetric: home-minus-away differences (no team identity leakage).
# Built at runtime from per-team columns.


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


def _apply_overlay(probs, gate, ha, aa):
    fl = _logit(np.clip(probs, 1e-15, 1 - 1e-15))
    fl = fl + QBG * (np.clip(ha, -QBC, QBC) - np.clip(aa, -QBC, QBC)) * E2L * gate.astype(float)
    return _sigmoid(fl)


def build_feature_table(ft_path: str = "data/features/nfl/feature_table.parquet") -> pd.DataFrame:
    """Build the eligible, non-neutral feature table with model inputs."""
    df_raw = pd.read_parquet(ft_path)
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
    return df


def feature_columns(kind: str) -> list[str]:
    """Return the list of pregame-safe model input columns for a feature set."""
    if kind == "incumbent":
        return ["elo_prob"] + INCUMBENT_FEATS
    if kind == "elo_rich":
        return ["elo_prob", "elo_diff"] + INCUMBENT_FEATS
    if kind == "antisymmetric":
        return [
            "elo_prob",
            "elo_diff",
            "qb_changed_diff",
            "rolling_mov_3_diff",
        ]
    raise ValueError(f"Unknown feature kind: {kind}")


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add antisymmetric / elo_diff derived columns if missing."""
    out = df.copy()
    if "elo_diff" not in out.columns and {"home_elo_pre", "away_elo_pre"}.issubset(out.columns):
        out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    if "qb_changed_diff" not in out.columns:
        out["qb_changed_diff"] = (
            out.get("home_qb_changed", 0).astype(float)
            - out.get("away_qb_changed", 0).astype(float)
        )
    if "rolling_mov_3_diff" not in out.columns and {
        "home_rolling_mov_3",
        "away_rolling_mov_3",
    }.issubset(out.columns):
        out["rolling_mov_3_diff"] = (
            out["home_rolling_mov_3"].fillna(0) - out["away_rolling_mov_3"].fillna(0)
        )
    return out


def feature_matrix(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Build the [N, D] feature matrix for a feature set kind."""
    df = _add_derived(df)
    cols = feature_columns(kind)
    avail = [c for c in cols if c in df.columns]
    missing = set(cols) - set(avail)
    if missing:
        raise KeyError(f"Missing feature columns for {kind}: {missing}")
    return np.column_stack([df[c].values.astype(np.float64) for c in avail])


# ── RobustScaler (RealMLP-style) ───────────────────────────────────────


def _fit_robust_scaler(x, clip=3.0):
    med = np.median(x, axis=0)
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    iqr = q3 - q1
    scales = 0.7413 * iqr  # MAD-style robust std (IQR/1.349)
    scales = np.where(scales == 0, 1.0, scales)
    return med, scales, clip


def _apply_robust_scaler(x, med, scales, clip):
    x = (x - med) / scales
    return np.clip(x, -clip, clip)


# ── MLP model + trainer ────────────────────────────────────────────────


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, activation="relu", dropout=0.1, init="default"):
        super().__init__()
        layers = []
        prev = in_dim
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights(init)

    def _init_weights(self, init):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _make_lr_schedule(optimizer, n_epochs, kind, base_lr):
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
    if kind == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=base_lr, total_steps=max(n_epochs, 1), pct_start=0.3
        )
    return None


def train_mlp(
    x_train,
    y_train,
    hidden=(16, 16, 16),
    activation="relu",
    dropout=0.1,
    weight_decay=1e-4,
    lr=1e-3,
    n_epochs=200,
    schedule="cosine",
    optimizer="adamw",
    batch_size=128,
    seed=42,
    patience=40,
    clip=3.0,
    early_stopping=True,
    scaler="robust",
    full_batch=False,
    init="default",
):
    """Train a single MLP with scaling, LR schedule, early stopping.

    scaler: "robust" (RealMLP-style) or "standard" (matches v3.1.0 incumbent).
    Returns (proba_fn, scaler_params) where proba_fn(X) -> probabilities.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if scaler == "standard":
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(x_train.astype(np.float64))
        med, scales, clip_used = sc.mean_, sc.scale_, None

        def _apply(x):
            return (x.astype(np.float64) - med) / scales
    else:
        med, scales, clip_used = _fit_robust_scaler(x_train, clip)

        def _apply(x):
            return _apply_robust_scaler(x.astype(np.float64), med, scales, clip)

    xs = _apply(x_train)
    xt = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train.astype(np.float64), dtype=torch.float32, device=DEVICE)

    model = _MLP(xt.shape[1], hidden, activation, dropout, init).to(DEVICE)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    sched = _make_lr_schedule(opt, n_epochs, schedule, lr)

    n = len(xt)
    eff_batch = n if full_batch else batch_size
    if early_stopping:
        n_val = max(1, int(0.2 * n))
        perm = torch.randperm(n)
        tr_idx, va_idx = perm[n_val:], perm[:n_val]
        xtr, ytr = xt[tr_idx], yt[tr_idx]
        xva, yva = xt[va_idx], yt[va_idx]
    else:
        xtr, ytr = xt, yt
        xva = yva = None

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    for epoch in range(n_epochs):
        model.train()
        for i in range(0, len(xtr), eff_batch):
            xb = xtr[i : i + eff_batch]
            yb = ytr[i : i + eff_batch]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        if early_stopping:
            model.eval()
            with torch.no_grad():
                vloss = crit(model(xva), yva).item()
            if vloss < best_loss - 1e-5:
                best_loss = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def proba_fn(x):
        xt = torch.tensor(_apply(x), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits = model(xt).cpu().numpy().reshape(-1)
        return _sigmoid(logits)

    return proba_fn, (med, scales, clip_used)


def predict_mlp_ensemble(x_train, y_train, x_pred, seeds=(0, 1, 2, 3, 4), **kwargs):
    """Train an ensemble of MLPs and average their predicted probabilities."""
    probas = []
    for s in seeds:
        fn, _ = train_mlp(x_train, y_train, seed=s, **kwargs)
        probas.append(fn(x_pred))
    return np.mean(probas, axis=0)


# ── Temperature scaling (calibration) ──────────────────────────────────


def temperature_scale(logits, y_true):
    """Fit a single temperature on logits via grid search (validation fold)."""
    import torch.nn.functional as fnl

    t_best, loss_best = 1.0, float("inf")
    y_t = torch.tensor(y_true.astype(np.float64), dtype=torch.float32, device=DEVICE)
    for t in np.linspace(0.3, 5.0, 48):
        t_t = torch.tensor(float(t), device=DEVICE)
        with torch.no_grad():
            loss = fnl.binary_cross_entropy_with_logits(logits / t_t, y_t).item()
        if loss < loss_best:
            loss_best = loss
            t_best = t
    return float(t_best)
