"""Tests for the logistic regression baseline model."""

import numpy as np
import pandas as pd

from sportslab.models.logistic import build_baseline_pipeline


class TestBuildBaselinePipeline:
    def test_pipeline_steps(self):
        pipe = build_baseline_pipeline()
        assert len(pipe.steps) == 3
        assert "imputer" in pipe.named_steps
        assert "scaler" in pipe.named_steps
        assert "classifier" in pipe.named_steps

    def test_pipeline_fit_and_predict(self):
        np.random.seed(42)
        x = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = (x["a"] + x["b"] > 0).astype(int)
        pipe = build_baseline_pipeline()
        pipe.fit(x, y)
        proba = pipe.predict_proba(x)[:, 1]
        assert proba.shape == (100,)
        assert all(0 <= p <= 1 for p in proba)

    def test_pipeline_handles_nans(self):
        x = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
        y = np.array([1, 0, 1])
        pipe = build_baseline_pipeline()
        pipe.fit(x, y)
        proba = pipe.predict_proba(x)[:, 1]
        assert proba.shape == (3,)
        assert not np.any(np.isnan(proba))
