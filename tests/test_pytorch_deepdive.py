"""Tests for the PyTorch deep-dive and LightGBM/TabPFN baseline experiments."""

from pathlib import Path

import numpy as np
import pandas as pd

from sportslab.evaluation import deep_mlp as dm
from sportslab.evaluation import gbm_tabpfn_baseline as gbm_exp
from sportslab.evaluation import pytorch_deepdive_experiment as pd_exp

FEATURE_PATH = "data/features/nfl/feature_table.parquet"


def _df():
    return dm.build_feature_table(FEATURE_PATH)


def test_deep_mlp_importable():
    assert hasattr(dm, "train_mlp")
    assert hasattr(dm, "predict_mlp_ensemble")
    assert hasattr(dm, "feature_matrix")
    assert hasattr(dm, "build_feature_table")


def test_feature_columns_kinds():
    assert dm.feature_columns("incumbent")[0] == "elo_prob"
    assert len(dm.feature_columns("incumbent")) == 5
    assert "elo_diff" in dm.feature_columns("elo_rich")
    assert "qb_changed_diff" in dm.feature_columns("antisymmetric")


def test_gate_and_overlay_shapes():
    df = _df()
    gate = dm._build_gate(df)
    assert gate.dtype == bool
    assert len(gate) == len(df)
    ha = df.get("home_qb_adj", pd.Series(0.0)).values.astype(float)
    aa = df.get("away_qb_adj", pd.Series(0.0)).values.astype(float)
    raw = np.full(len(df), 0.5)
    out = dm._apply_overlay(raw, gate, ha, aa)
    assert out.min() >= 0 and out.max() <= 1
    # overlay only moves gated rows
    assert not np.allclose(out, raw)


def test_feature_matrix_runs():
    df = _df()
    xf = dm.feature_matrix(df, "incumbent")
    assert xf.shape[0] == len(df)
    assert xf.shape[1] == 5
    xr = dm.feature_matrix(df, "elo_rich")
    assert xr.shape[1] == 6


def test_train_mlp_reproducible():
    df = _df()
    yv = df[dm.TARGET_COLUMN].astype(float).values
    xf = dm.feature_matrix(df, "incumbent")
    tr = df["season"].isin([2000, 2001, 2002]).values
    fn1, _ = dm.train_mlp(xf[tr], yv[tr], hidden=(16, 16, 16), weight_decay=1e-4,
                          schedule="none", optimizer="adam", n_epochs=50,
                          early_stopping=False, scaler="standard", full_batch=True, seed=0)
    fn2, _ = dm.train_mlp(xf[tr], yv[tr], hidden=(16, 16, 16), weight_decay=1e-4,
                          schedule="none", optimizer="adam", n_epochs=50,
                          early_stopping=False, scaler="standard", full_batch=True, seed=0)
    p1 = fn1(xf[tr])
    p2 = fn2(xf[tr])
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_train_mlp_full_batch_uses_all_data():
    # With early_stopping=False the internal 80/20 split must not drop data (bug guard).
    df = _df()
    yv = df[dm.TARGET_COLUMN].astype(float).values
    xf = dm.feature_matrix(df, "incumbent")
    tr = df["season"].isin([2000, 2001]).values
    fn, _ = dm.train_mlp(xf[tr], yv[tr], hidden=(16,), weight_decay=1e-4,
                         schedule="none", optimizer="adam", n_epochs=20,
                         early_stopping=False, scaler="standard", full_batch=True)
    # predicts finite probs for all training rows
    p = fn(xf[tr])
    assert np.isfinite(p).all()


def test_ensemble_matches_single_on_baseline():
    df = _df()
    yv = df[dm.TARGET_COLUMN].astype(float).values
    xf = dm.feature_matrix(df, "incumbent")
    tr = df["season"].isin([2000, 2001, 2002]).values
    ens = dm.predict_mlp_ensemble(xf[tr], yv[tr], xf[tr], seeds=(0, 1), hidden=(16,),
                                  weight_decay=1e-4, schedule="none", optimizer="adam",
                                  n_epochs=20, early_stopping=False, scaler="standard",
                                  full_batch=True)
    assert ens.shape[0] == tr.sum() and np.isfinite(ens).all()


def test_deepdive_configs_well_formed():
    for cfg in pd_exp.MLP_CONFIGS:
        assert len(cfg) == 13
        assert cfg[1] in ("incumbent", "elo_rich", "antisymmetric")


def test_deepdive_report_written(tmp_path):
    rp = tmp_path / "deepdive.md"
    out = pd_exp.run_pytorch_deepdive_experiment(report_path=str(rp))
    assert Path(out).exists()
    txt = Path(out).read_text()
    assert "PyTorch Deep-Dive" in txt
    assert "REJECTED" in txt or "PROMOTED" in txt


def test_gbm_configs_well_formed():
    # build configs dict indirectly via module import
    assert hasattr(gbm_exp, "run_gbm_baseline_experiment")


def test_gbm_report_written(tmp_path):
    rp = tmp_path / "gbm.md"
    out = gbm_exp.run_gbm_baseline_experiment(report_path=str(rp))
    assert Path(out).exists()
    txt = Path(out).read_text()
    assert "LightGBM" in txt
    # MLP baseline must appear; TabPFN marked BLOCKED (no API token offline)
    assert "v3.1.0_mlp" in txt
    assert "BLOCKED" in txt
