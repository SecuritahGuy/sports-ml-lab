"""Tests for the NN architecture exploration experiment."""

from pathlib import Path

import numpy as np

from sportslab.evaluation import deep_mlp as dm
from sportslab.evaluation import nn_architecture_experiment as arch_exp


def test_module_importable():
    assert hasattr(arch_exp, "run_nn_architecture_experiment")
    assert hasattr(arch_exp, "CONFIGS")


def test_configs_well_formed():
    for cfg in arch_exp.CONFIGS:
        assert len(cfg) == 12
        assert cfg[3] in ("mlp", "resnet")


def test_resnet_model_runs():
    df = dm.build_feature_table()
    yv = df[dm.TARGET_COLUMN].astype(float).values
    xf = dm.feature_matrix(df, "incumbent")
    tr = df["season"].isin([2000, 2001, 2002]).values
    fn, _ = dm.train_mlp(
        xf[tr], yv[tr], hidden=(64, 64, 64), arch="resnet", activation="relu",
        dropout=0.1, weight_decay=1e-4, schedule="none", optimizer="adam",
        n_epochs=20, early_stopping=False, scaler="standard", full_batch=True,
        init="kaiming",
    )
    p = fn(xf[tr])
    assert np.isfinite(p).all() and p.shape[0] == tr.sum()


def test_ensemble_runs():
    df = dm.build_feature_table()
    yv = df[dm.TARGET_COLUMN].astype(float).values
    xf = dm.feature_matrix(df, "incumbent")
    tr = df["season"].isin([2000, 2001, 2002]).values
    ens = dm.predict_mlp_ensemble(
        xf[tr], yv[tr], xf[tr], seeds=(0, 1, 2), hidden=(16, 16, 16),
        arch="mlp", activation="relu", dropout=0.1, weight_decay=1e-4,
        schedule="none", optimizer="adam", n_epochs=20, early_stopping=False,
        scaler="standard", full_batch=True, init="kaiming",
    )
    assert ens.shape[0] == tr.sum() and np.isfinite(ens).all()


def test_report_written(tmp_path):
    rp = tmp_path / "nn_arch.md"
    out = arch_exp.run_nn_architecture_experiment(report_path=str(rp))
    assert Path(out).exists()
    txt = Path(out).read_text()
    assert "NN Architecture" in txt
    assert "REJECTED" in txt or "PROMOTED" in txt
