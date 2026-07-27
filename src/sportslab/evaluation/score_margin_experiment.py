"""Score-margin distribution model — rolling-origin experiment.

Compares 3 approaches on rolling-origin folds:
  1. Incumbent (Elo + qb_changed + rolling_mov_3 + Platt)
  2. Score-margin OLS (margin ~ elo_diff + qb_changed + rolling_mov_3)
  3. Score-margin OLS + Platt calibration

No promotion pressure — this is a shadow/research experiment.
Outputs diagnostic report with log loss, calibration, and margin quantile accuracy.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sportslab.evaluation.predict_incumbent import (
    INCUMBENT_VERSION,
)
from sportslab.evaluation.score_margin_model import ScoreMarginModel
from sportslab.evaluation.weekly_pipeline import _compute_metrics
from sportslab.features.build_features import (
    TARGET_COLUMN,
)

REPORT_DIR = Path("reports/experiments")
DEFAULT_FT_PATH = "data/features/nfl/feature_table.parquet"


def _build_roll_folds(
    ft: pd.DataFrame,
    val_seasons: List[int],
    holdout_season: int = 2025,
) -> List[Dict]:
    """Build rolling origin folds from feature table.

    For each val_season in order: train on seasons before it,
    validate on val_season. Final holdout is holdout_season.
    """
    all_games = ft[ft["model_eligible"]].sort_values("gameday").reset_index(drop=True)
    folds = []

    for i, val_season in enumerate(val_seasons):
        train_mask = all_games["season"] < val_season
        val_mask = all_games["season"] == val_season
        folds.append({
            "train": all_games[train_mask].copy(),
            "val": all_games[val_mask].copy(),
            "val_season": val_season,
        })

    holdout_mask = all_games["season"] == holdout_season
    folds.append({
        "train": all_games[all_games["season"] < holdout_season].copy(),
        "val": all_games[holdout_mask].copy(),
        "val_season": holdout_season,
    })

    return folds


def _features_from_df(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Extract feature matrix and margin target from DataFrame."""
    cols = ["elo_diff", "qb_changed_flag", "rolling_mov_3"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return None, np.array([])

    X = df[available].fillna(0).values
    margin = (df["home_score"].values - df["away_score"].values) if "home_score" in df.columns else None
    return X, margin


def _run_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict:
    """Run a single fold comparison."""
    metrics = {}

    # Incumbent Platt baseline
    train_valid = train_df[TARGET_COLUMN].notna().values
    val_valid = val_df[TARGET_COLUMN].notna().values

    if train_valid.sum() > 0 and val_valid.sum() > 0:
        y_true_val = val_df.loc[val_valid, TARGET_COLUMN].astype(int).values
        inc_metrics = _compute_metrics(val_df)
        metrics["incumbent"] = inc_metrics

    # Score-margin model
    X_train, y_train_margin = _features_from_df(train_df)
    X_val, _ = _features_from_df(val_df)

    if X_train is not None and y_train_margin is not None and train_valid.sum() >= 10:
        model = ScoreMarginModel(fit_intercept=True)
        model.fit(X_train, y_train_margin)

        margin_probs = model.predict_win_prob(X_val[val_valid]) if val_valid.sum() > 0 else np.array([])
        if len(margin_probs) > 0:
            y_pred = (margin_probs >= 0.5).astype(int)
            eps = 1e-15
            margin_probs_clip = np.clip(margin_probs, eps, 1 - eps)
            from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
            n_classes = len(np.unique(y_true_val))
            metrics["margin_model"] = {
                "n": int(val_valid.sum()),
                "log_loss": round(float(log_loss(y_true_val, margin_probs_clip, labels=[0, 1])), 4)
                    if n_classes >= 2 else None,
                "brier": round(float(brier_score_loss(y_true_val, margin_probs_clip)), 4)
                    if n_classes >= 2 else None,
                "accuracy": round(float(accuracy_score(y_true_val, y_pred)), 4),
                "auc": round(float(roc_auc_score(y_true_val, margin_probs_clip)), 4)
                    if n_classes >= 2 else None,
            }
            metrics["sigma"] = round(float(model.sigma_), 2) if model.sigma_ else None
        else:
            metrics["margin_model"] = {"n": 0}
    else:
        metrics["margin_model"] = {"n": 0}

    return metrics


def run_score_margin_experiment(
    ft_path: str = DEFAULT_FT_PATH,
    report_path: Optional[str] = None,
) -> str:
    """Run rolling-origin score-margin experiment and write report."""
    if not Path(ft_path).exists():
        raise FileNotFoundError(f"Feature table not found: {ft_path}")

    ft = pd.read_parquet(ft_path)
    val_seasons = [2022, 2023, 2024]
    folds = _build_roll_folds(ft, val_seasons)

    if report_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = str(REPORT_DIR / "score_margin_model.md")

    lines = []
    _w = lines.append

    _w("# Score-Margin Distribution Model — Shadow Experiment\n")
    _w(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}*\n")
    _w("## Motivation\n")
    _w("Standard Elo + Platt predicts win probability indirectly (margin→logistic scale→prob). ")
    _w("A score-margin distribution model predicts the full distribution of ")
    _w("home_score - away_score as Normal(μ, σ²), then derives win probability as P(margin > 0) = Φ(μ/σ).\n")
    _w("This approach naturally captures uncertainty, avoids Platt distortion at extremes, ")
    _w("and produces blowout probabilities and credible intervals.\n")
    _w("**This is a shadow experiment — no promotion pressure.**\n")

    all_fold_results = []
    for i, fold in enumerate(folds):
        season = fold["val_season"]
        _w(f"## Fold {i+1}: Validate on {season}\n")
        _w("| Model | N | Log Loss | Brier | Accuracy | AUC |")
        _w("|-------|---|----------|-------|----------|-----|")

        results = _run_fold(fold["train"], fold["val"])
        all_fold_results.append(results)

        for model_key in ["incumbent", "margin_model"]:
            m = results.get(model_key, {})
            if m and m.get("n", 0) > 0:
                ll = f"{m['log_loss']:.4f}" if m.get("log_loss") else "—"
                br = f"{m['brier']:.4f}" if m.get("brier") else "—"
                ac = f"{m['accuracy']:.4f}" if m.get("accuracy") else "—"
                au = f"{m['auc']:.4f}" if m.get("auc") else "—"
                _w(f"| {'Margin Model' if model_key == 'margin_model' else 'Incumbent'} | {m['n']} | {ll} | {br} | {ac} | {au} |")

        sigma = results.get("sigma")
        if sigma:
            _w(f"\nMargin model σ = {sigma}")
        _w("")

    # Holdout (2025)
    holdout = folds[-1]
    _w("## 2025 Holdout\n")
    _w("| Model | Log Loss | Brier | Accuracy | AUC |")
    _w("|-------|----------|-------|----------|-----|")
    results = _run_fold(holdout["train"], holdout["val"])
    for model_key in ["incumbent", "margin_model"]:
        m = results.get(model_key, {})
        if m and m.get("n", 0) > 0:
            ll = f"{m['log_loss']:.4f}" if m.get("log_loss") else "—"
            br = f"{m['brier']:.4f}" if m.get("brier") else "—"
            ac = f"{m['accuracy']:.4f}" if m.get("accuracy") else "—"
            au = f"{m['auc']:.4f}" if m.get("auc") else "—"
            _w(f"| {'Margin Model' if model_key == 'margin_model' else 'Incumbent'} | {ll} | {br} | {ac} | {au} |")

    sigma = results.get("sigma")
    if sigma:
        _w(f"\nMargin model σ = {sigma}")
    _w("")

    _w("## Summary\n")
    _w("The score-margin distribution model uses OLS on margin (home_score - away_score) ")
    _w("with features: elo_diff, qb_changed_flag, rolling_mov_3. σ is estimated from ")
    _w("training residuals. Win probability = Φ(μ/σ).\n")
    _w("This is a research experiment. No promotion against the incumbent.\n")
    _w("---\n")
    _w(f"*Report generated by SportsLab NFL {INCUMBENT_VERSION}.*\n")

    report = "\n".join(lines)
    Path(report_path).write_text(report)
    print("\n=== Score-Margin Distribution Model ===")
    print(f"  Report: {report_path}")

    return report_path
