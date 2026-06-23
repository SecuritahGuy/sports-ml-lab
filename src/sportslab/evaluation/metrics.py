"""Binary classification metrics for home-win prediction."""

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def compute_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute all relevant metrics for binary home-win prediction.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred_proba: Predicted probabilities for the positive class (home win).

    Returns:
        dict with log_loss, brier_score, accuracy, roc_auc, and calibration bucket
        summary.
    """
    # Handle single-class case: sklearn needs explicit labels
    n_classes = len(np.unique(y_true))

    if n_classes > 1:
        log_loss_val = float(log_loss(y_true, y_pred_proba))
        roc_auc_val = float(roc_auc_score(y_true, y_pred_proba))
    else:
        log_loss_val = float(log_loss(y_true, y_pred_proba, labels=[0, 1]))
        roc_auc_val = None

    metrics = {
        "log_loss": log_loss_val,
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred_proba >= 0.5)),
        "roc_auc": roc_auc_val,
    }

    # Calibration buckets: deciles
    buckets = {}
    for i in range(10):
        lo, hi = i * 0.1, (i + 1) * 0.1
        mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(y_pred_proba[mask].mean())
            mean_actual = float(y_true[mask].mean())
            buckets[f"[{lo:.1f}, {hi:.1f})"] = {
                "count": count,
                "mean_predicted_prob": round(mean_pred, 4),
                "mean_actual_rate": round(mean_actual, 4),
                "calibration_error": round(abs(mean_pred - mean_actual), 4),
            }

    metrics["calibration_buckets"] = buckets
    return metrics
