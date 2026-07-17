"""Tests for the neural network + expanded features experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

from sportslab.evaluation import neural_network_features_experiment as nnf
from sportslab.features.build_features import MODEL_ELIGIBLE_COLUMN, TARGET_COLUMN


def _build_df():
    df_raw = pd.read_parquet("data/features/nfl/feature_table.parquet")
    ov = nnf.build_team_regression_overrides(
        df_raw, preseason_regression=nnf.BEST_REG, qb_change_bonus=nnf.BEST_QB_BONUS
    )
    df = nnf.compute_elo_features(
        df_raw,
        k_factor=nnf.BEST_K,
        home_advantage=nnf.BEST_HFA,
        preseason_regression=nnf.BEST_REG,
        team_regression_overrides=ov,
        decay_half_life=nnf.BEST_DECAY,
    )
    df = nnf.compute_qb_features(df)
    df = nnf.compute_qb_adjustments(df)
    df = nnf.compute_situational_features(df)
    mask = df[MODEL_ELIGIBLE_COLUMN].values & ~df["is_neutral"].fillna(False).values
    return df[mask].copy().reset_index(drop=True)


def test_module_importable():
    assert hasattr(nnf, "run_neural_network_features_experiment")
    assert hasattr(nnf, "FEATURE_VARIANTS")


def test_feature_variants_include_incumbent():
    names = [v[0] for v in nnf.FEATURE_VARIANTS]
    assert "incumbent_only" in names
    # Every variant must be a superset of the incumbent features
    for name, cols in nnf.FEATURE_VARIANTS:
        for c in nnf.INCUMBENT_FEATS:
            assert c in cols, f"{name} missing {c}"


def test_extra_candidates_pregame_safe():
    # None of the extra candidates should be score/result/market derived.
    forbidden = {"home_score", "away_score", "result", "home_win", "is_tie"}
    for c in nnf.EXTRA_CANDIDATES:
        assert c not in forbidden


def test_mlp_runs_with_expanded_features():
    df = _build_df()
    y = df[TARGET_COLUMN].astype(float).values
    ep = df["elo_prob"].values.astype(float)
    cols = nnf.INCUMBENT_FEATS + nnf.EXTRA_CANDIDATES
    avail = [c for c in cols if c in df.columns]
    x = np.column_stack([ep] + [df[c].values for c in avail])
    model, scaler = nnf._train_mlp(x, y, nnf.MLP_HIDDEN, nnf.MLP_DROPOUT, nnf.MLP_WD)
    proba = nnf._mlp_proba(model, scaler, x)
    assert proba.shape[0] == len(df)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_experiment_runs_and_writes_report():
    report = Path("reports/experiments/neural_network_features.md")
    if report.exists():
        report.unlink()
    path = nnf.run_neural_network_features_experiment()
    assert Path(path).exists()
    content = Path(path).read_text()
    assert "Neural Network + Expanded Features" in content
    assert "| Feature set | Val LL" in content
    assert "Decision" in content


def test_expanded_features_do_not_beat_incumbent_on_both():
    """Regression guard: the MLP win is the calibrator, not new features."""
    df = _build_df()
    y = df[TARGET_COLUMN].astype(float).values
    gate = nnf._build_gate(df)
    ha = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    aa = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    ep = df["elo_prob"].values.astype(float)

    def fm(cols, idx):
        avail = [c for c in cols if c in df.columns]
        return np.column_stack([ep[idx]] + [df[c].values[idx] for c in avail])

    def fit_predict(cols, train_idx, all_idx):
        xtr = fm(cols, train_idx)
        xall = fm(cols, all_idx)
        model, scaler = nnf._train_mlp(
            xtr, y[train_idx], nnf.MLP_HIDDEN, nnf.MLP_DROPOUT, nnf.MLP_WD
        )
        raw = nnf._mlp_proba(model, scaler, xall)
        return nnf._apply_overlay(raw, gate, ha, aa)

    from sklearn.metrics import log_loss as ll

    iv_folds = []
    ih_folds = []
    for name, cols in nnf.FEATURE_VARIANTS:
        v = []
        for ts, vs in nnf.ROLLING_FOLDS:
            tr = df["season"].isin(ts).values
            va = (df["season"] == vs).values
            pp = fit_predict(cols, tr, slice(None))
            vy = y[va]
            vd = ~np.isnan(vy)
            v.append(ll(vy[vd].astype(int), pp[va][vd]))
        iv = float(np.mean(v))
        tr = (df["season"] < nnf.HOLDOUT_SEASON).values
        va = (df["season"] == nnf.HOLDOUT_SEASON).values
        pp = fit_predict(cols, tr, slice(None))
        vy = y[va]
        vd = ~np.isnan(vy)
        ih = float(ll(vy[vd].astype(int), pp[va][vd]))
        iv_folds.append((name, iv))
        ih_folds.append((name, ih))

    inc_v = dict(iv_folds)["incumbent_only"]
    inc_h = dict(ih_folds)["incumbent_only"]
    for name, v in iv_folds:
        if name == "incumbent_only":
            continue
        h = dict(ih_folds)[name]
        # No expanded variant should beat BOTH val and holdout by >= 0.001
        assert not (v < inc_v - 0.001 and h < inc_h - 0.001), (
            f"{name} unexpectedly beat incumbent on both: val {v:.4f} hold {h:.4f}"
        )
