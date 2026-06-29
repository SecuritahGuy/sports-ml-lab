"""Weekly prediction pipeline — snapshot, grade, season dashboard.

Workflow:
  1. sportslab predict-week --season 2026 --week 1
     → Fits Elo on all historical data, predicts week 1 games,
       saves timestamped snapshot + weekly report.
  2. sportslab grade-week --season 2026 --week 1
     → Loads snapshot, merges actual results, computes metrics,
       appends to prediction history.
  3. sportslab season-report --season 2026
     → Generates cumulative dashboard from prediction history.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sportslab.evaluation.predict_incumbent import (
    INCUMBENT_DATE,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VAL_LL,
    INCUMBENT_VERSION,
)
from sportslab.features.build_features import TARGET_COLUMN

# ── Paths ──
SNAPSHOT_DIR = Path("reports/predictions/snapshots")
HISTORY_PATH = Path("reports/predictions/prediction_history.csv")
REPORT_DIR = Path("reports/predictions")
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"

NOW = datetime.now(timezone.utc)


def _timestamp() -> str:
    return NOW.strftime("%Y%m%d_%H%M%S")


def _snapshot_path(season: int, week: int) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOT_DIR / f"week_{season}_{week:02d}_{_timestamp()}.csv"


def _latest_prediction(season: int, week: int) -> Optional[Path]:
    """Find most recent snapshot for given season/week."""
    if not SNAPSHOT_DIR.exists():
        return None
    pattern = f"week_{season}_{week:02d}_*.csv"
    matches = sorted(SNAPSHOT_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def predict_week(
    season: int,
    week: int,
    qb_input: Optional[str] = None,
    snapshot_path: Optional[str] = None,
) -> Dict[str, str]:
    """Predict a single week, save snapshot + report.

    Args:
        season: Season year (e.g. 2026).
        week: Week number (1-22).
        qb_input: Optional path to QB input CSV.
        snapshot_path: Override snapshot output path.

    Returns:
        Dict with paths to snapshot, report.
    """
    from sportslab.evaluation.predict_future import predict_future

    out = snapshot_path or str(_snapshot_path(season, week))
    pred_result = predict_future(
        season=season,
        week=week,
        qb_input_path=qb_input,
        output_path=out,
    )
    if not pred_result:
        print(f"  No games found for {season} week {week}.")
        return {}

    # Generate weekly report
    from sportslab.evaluation.weekly_report import generate_weekly_report

    rpt = Path(REPORT_DIR) / f"week_{season}_{week:02d}_report.md"
    try:
        report_path = generate_weekly_report(
            season=season,
            week=week,
            output=str(rpt),
            input_path=str(snapshot_path) if snapshot_path else str(pred_result["predictions"]),
        )
    except FileNotFoundError as e:
        print(f"  Report generation skipped: {e}")
        report_path = str(rpt)

    print(f"\n=== Week {week}, {season} Season ===")
    print(f"  Snapshot: {out}")
    print(f"  Report:   {report_path}")

    return {"snapshot": out, "report": report_path}


def _load_actuals(df_snapshot: pd.DataFrame) -> pd.DataFrame:
    """Merge actual results from feature table into snapshot."""
    ft_path = Path(FEATURE_TABLE_PATH)
    if not ft_path.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURE_TABLE_PATH}")

    ft = pd.read_parquet(ft_path)
    needed = ["game_id", TARGET_COLUMN]
    ft_actuals = ft[needed].copy()
    ft_actuals.columns = ["game_id", "actual_home_win"]
    merged = df_snapshot.merge(ft_actuals, on="game_id", how="left")
    return merged


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute grading metrics from snapshot with actuals."""
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    valid = df["actual_home_win"].notna().values
    if valid.sum() == 0:
        return {"n": 0}

    y_true = df.loc[valid, "actual_home_win"].astype(int).values
    y_prob = df.loc[valid, "incumbent_home_win_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    n_classes = len(np.unique(y_true))
    ll = float(log_loss(y_true, y_prob, labels=[0, 1])) if n_classes >= 2 else float("nan")
    brier = float(brier_score_loss(y_true, y_prob)) if n_classes >= 2 else float("nan")
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if n_classes >= 2 else float("nan")

    return {
        "n": int(valid.sum()),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
    }


def _read_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(
        columns=[
            "season",
            "week",
            "n",
            "log_loss",
            "brier",
            "accuracy",
            "auc",
            "model_version",
            "snapshot",
            "graded_at",
        ]
    )


def _write_history(df: pd.DataFrame) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_PATH, index=False)


def grade_week(
    season: int,
    week: int,
    snapshot: Optional[str] = None,
) -> Dict:
    """Grade a completed week's predictions.

    Args:
        season: Season year.
        week: Week number.
        snapshot: Path to snapshot CSV. Auto-detects if None.

    Returns:
        Dict with metrics and history path.
    """
    # Find snapshot
    if snapshot:
        snap_path = Path(snapshot)
    else:
        found = _latest_prediction(season, week)
        if not found:
            print(f"  No snapshot found for {season} week {week}.")
            return {}
        snap_path = found

    if not snap_path.exists():
        print(f"  Snapshot not found: {snap_path}")
        return {}

    df = pd.read_csv(snap_path)
    if TARGET_COLUMN not in df.columns and "actual_home_win" not in df.columns:
        df = _load_actuals(df)

    # Check if actuals are available
    has_col = "actual_home_win" in df.columns
    has_actuals = has_col and df["actual_home_win"].notna().sum() > 0
    if not has_actuals:
        print(f"  No actual results found for {season} week {week}.")
        print("  Games may not be played yet, or feature table needs rebuilding.")
        return {}

    metrics = _compute_metrics(df)
    if metrics["n"] == 0:
        print("  No graded games found.")
        return {}

    # Append to history
    history = _read_history()
    existing = history[(history["season"] == season) & (history["week"] == week)]
    if not existing.empty:
        history = history[~((history["season"] == season) & (history["week"] == week))]

    row = {
        "season": season,
        "week": week,
        "n": metrics["n"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "accuracy": metrics["accuracy"],
        "auc": metrics["auc"],
        "model_version": INCUMBENT_VERSION,
        "snapshot": str(snap_path),
        "graded_at": _timestamp(),
    }
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    _write_history(history)

    print(f"\n=== Week {week}, {season} Season — Grade ===")
    print(f"  Games graded: {metrics['n']}")
    print(f"  Log loss:     {metrics['log_loss']}")
    print(f"  Brier:        {metrics['brier']}")
    print(f"  Accuracy:     {metrics['accuracy']}")
    print(f"  AUC:          {metrics['auc']}")
    print(f"  History:      {HISTORY_PATH}")

    return {"metrics": metrics, "history": str(HISTORY_PATH)}


def _season_report_content(
    df_history: pd.DataFrame,
    season: int,
) -> str:
    """Generate season dashboard markdown."""
    lines = []
    _w = lines.append
    _w(f"# Season Report — {season} Season\n")
    _w(f"*Generated by `sportslab season-report` ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*\n")
    _w("")

    season_df = df_history[df_history["season"] == season].sort_values("week")
    if season_df.empty:
        _w("No graded weeks found for this season.\n")
        return "\n".join(lines)

    _w("## Season Summary\n")
    _w("")
    tot_n = int(season_df["n"].sum())
    mean_ll = round(float(season_df["log_loss"].mean()), 4)
    mean_brier = round(float(season_df["brier"].mean()), 4)
    mean_acc = round(float(season_df["accuracy"].mean()), 4)
    mean_auc = round(float(season_df["auc"].mean()), 4)

    _w("| Metric | Value |")
    _w("|--------|-------|")
    _w(f"| Games graded | {tot_n} |")
    _w(f"| Weeks graded | {len(season_df)} |")
    _w(f"| Mean log loss | {mean_ll} |")
    _w(f"| Mean Brier | {mean_brier} |")
    _w(f"| Mean accuracy | {mean_acc} |")
    _w(f"| Mean AUC | {mean_auc} |")
    _w(f"| Model | {INCUMBENT_VERSION} |")
    _w(f"| Holdout LL | {INCUMBENT_HOLDOUT_LL} |")
    _w("")

    _w("## Per-Week Breakdown\n")
    _w("")
    _w("| Week | Games | Log Loss | Brier | Accuracy | AUC |")
    _w("|------|-------|----------|-------|----------|-----|")
    for _, r in season_df.iterrows():
        _w(
            f"| {int(r['week'])} | {int(r['n'])} | {r['log_loss']:.4f}"
            if not pd.isna(r["log_loss"])
            else f" | — | {r['brier']:.4f}"
            if not pd.isna(r["brier"])
            else f" | — | {r['accuracy']:.4f} | {r['auc']:.4f}"
            if not pd.isna(r["auc"])
            else " | — |"
        )
    _w("")

    _w("## Model Metadata\n")
    _w("")
    _w("| Attribute | Value |")
    _w("|-----------|-------|")
    _w(f"| Model version | {INCUMBENT_VERSION} |")
    _w("| Feature set | qb_changed + rolling_mov_3 |")
    _w("| Calibration | Platt (logistic on Elo prob + features) |")
    _w(f"| Validation LL | {INCUMBENT_VAL_LL} |")
    _w(f"| Holdout LL (2025) | {INCUMBENT_HOLDOUT_LL} |")
    _w("")

    _w("---\n")
    _w(f"*Report generated by SportsLab NFL Incumbent {INCUMBENT_VERSION}.*\n")
    return "\n".join(lines)


def season_report(season: int) -> Dict[str, str]:
    """Generate season dashboard from prediction history.

    Args:
        season: Season year.

    Returns:
        Dict with report path.
    """
    if not HISTORY_PATH.exists():
        print("  No prediction history found. Run `sportslab grade-week` first.")
        return {}

    history = _read_history()
    content = _season_report_content(history, season)

    rpt = REPORT_DIR / f"season_{season}_report.md"
    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text(content)

    print(f"\n=== Season Report: {season} ===")
    print(f"  Report: {rpt}")
    print(f"  Weeks graded: {len(history[history['season'] == season])}")
    print(f"  Prediction history: {HISTORY_PATH}")

    return {"report": str(rpt)}


def prediction_history() -> pd.DataFrame:
    """Return the prediction history DataFrame."""
    return _read_history()
