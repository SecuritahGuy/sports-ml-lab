"""Logistic regression baseline model."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def build_baseline_pipeline() -> Pipeline:
    """Build the baseline logistic regression pipeline.

    Pipeline steps:
        1. SimpleImputer(strategy="mean") — fill missing numeric values.
        2. StandardScaler() — standardise features to zero mean, unit variance.
        3. LogisticRegression(max_iter=1000, random_state=42) — L2-regularised
           logistic regression.

    Returns:
        A scikit-learn Pipeline ready for .fit() and .predict_proba().
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            ),
        ]
    )
