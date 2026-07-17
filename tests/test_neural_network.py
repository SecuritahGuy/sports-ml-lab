"""Tests for the neural network challenger experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

from sportslab.evaluation import neural_network_experiment as nn
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, TARGET_COLUMN


def _build_df():
    df_raw = pd.read_parquet("data/features/nfl/feature_table.parquet")
    ov = nn.build_team_regression_overrides(
        df_raw, preseason_regression=nn.BEST_REG, qb_change_bonus=nn.BEST_QB_BONUS
    )
    df = nn.compute_elo_features(
        df_raw,
        k_factor=nn.BEST_K,
        home_advantage=nn.BEST_HFA,
        preseason_regression=nn.BEST_REG,
        team_regression_overrides=ov,
        decay_half_life=nn.BEST_DECAY,
    )
    df = nn.compute_qb_features(df)
    df = nn.compute_qb_adjustments(df)
    df = nn.compute_situational_features(df)
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df["is_neutral"].fillna(False).values
    return df[mask].copy().reset_index(drop=True)


def test_module_importable():
    assert hasattr(nn, "run_neural_network_experiment")
    assert hasattr(nn, "NN_VARIANTS")


def test_variants_include_incumbent():
    names = [v[0] for v in nn.NN_VARIANTS]
    assert "incumbent" in names
    assert sum(1 for v in nn.NN_VARIANTS if v[2] is None) == 1


def test_sigmoid_logit_roundtrip():
    p = np.array([0.1, 0.5, 0.9, 0.01])
    np.testing.assert_allclose(nn._sigmoid(nn._logit(p)), p, atol=1e-9)


def test_sigmoid_bounds():
    out = nn._sigmoid(np.array([-1000.0, 1000.0]))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_build_gate_binary():
    df = _build_df()
    gate = nn._build_gate(df)
    assert set(np.unique(gate)).issubset({True, False})


def test_mlp_runs_and_outputs_probs():
    df = _build_df()
    y = df[TARGET_COLUMN].astype(float).values
    ep = df["elo_prob"].values.astype(float)
    x = np.column_stack([ep] + [df[c].values for c in nn.INCUMBENT_FEATS])
    idx = np.arange(len(df))
    model, scaler = nn._train_mlp(x[idx], y[idx], (32, 16), 0.1, 1e-4)
    proba = nn._mlp_proba(model, scaler, x[idx])
    assert proba.shape[0] == len(df)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_logistic_baseline_matches_incumbent_shape():
    df = _build_df()
    y = df[TARGET_COLUMN].astype(float).values
    ep = df["elo_prob"].values.astype(float)
    x = np.column_stack([ep] + [df[c].values for c in nn.INCUMBENT_FEATS])
    idx = np.arange(len(df))
    lr, scaler = nn._train_logistic(x[idx], y[idx])
    proba = nn._logistic_proba(lr, scaler, x[idx])
    assert proba.shape[0] == len(df)
    assert np.all((proba > 0.0) & (proba < 1.0))


def test_overlay_changes_gated_games():
    df = _build_df()
    gate = nn._build_gate(df)
    ha = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    aa = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    base = np.full(len(df), 0.6)
    out = nn._apply_overlay(base, gate, ha, aa)
    gated = gate.sum()
    assert gated > 0
    assert not np.allclose(out[gate], base[gate])


def test_experiment_runs_and_writes_report():
    report = Path("reports/experiments/neural_network.md")
    if report.exists():
        report.unlink()
    path = nn.run_neural_network_experiment()
    assert Path(path).exists()
    content = Path(path).read_text()
    assert "Neural Network Challenger" in content
    assert "| Model | Val LL" in content
    assert "Decision" in content


def test_nn_beats_incumbent_on_both():
    """Regression guard: the MLP challenger must remain competitive."""
    df = _build_df()
    y = df[TARGET_COLUMN].astype(float).values
    gate = nn._build_gate(df)
    ha = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    aa = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    ep = df["elo_prob"].values.astype(float)

    def fm(idx):
        return np.column_stack([ep[idx]] + [df[c].values[idx] for c in nn.INCUMBENT_FEATS])

    # incumbent validation via logistic
    from sklearn.metrics import log_loss as ll

    val_lls = []
    for ts, vs in nn.ROLLING_FOLDS:
        tr = df["season"].isin(ts).values
        va = (df["season"] == vs).values
        xtr = fm(tr)
        xall = fm(slice(None))
        lr, sc = nn._train_logistic(xtr, y[tr])
        raw = nn._logistic_proba(lr, sc, xall)
        pp = nn._apply_overlay(raw, gate, ha, aa)
        vy = y[va]
        vd = ~np.isnan(vy)
        val_lls.append(ll(vy[vd].astype(int), pp[va][vd]))
    inc_val = float(np.mean(val_lls))

    # best mlp variant
    name, _, h, d, w = [v for v in nn.NN_VARIANTS if v[0] == "mlp_32_16_wd"][0]
    mlp_lls = []
    for ts, vs in nn.ROLLING_FOLDS:
        tr = df["season"].isin(ts).values
        va = (df["season"] == vs).values
        xtr = fm(tr)
        xall = fm(slice(None))
        model, sc = nn._train_mlp(xtr, y[tr], h, d, w)
        raw = nn._mlp_proba(model, sc, xall)
        pp = nn._apply_overlay(raw, gate, ha, aa)
        vy = y[va]
        vd = ~np.isnan(vy)
        mlp_lls.append(ll(vy[vd].astype(int), pp[va][vd]))
    mlp_val = float(np.mean(mlp_lls))

    assert mlp_val < inc_val - 0.001
