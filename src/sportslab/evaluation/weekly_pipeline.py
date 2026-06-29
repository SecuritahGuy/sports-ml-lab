"""Weekly prediction pipeline — snapshot, grade, season dashboard.

Workflow:
  1. sportslab predict-week --season 2026 --week 1
     → Fits Elo on all historical data, predicts week 1 games,
       saves timestamped snapshot + weekly report + manifest entry.
  2. sportslab grade-week --season 2026 --week 1
     → Loads snapshot from manifest, merges actual results,
       computes metrics, updates manifest + history.
  3. sportslab season-report --season 2026
     → Generates cumulative dashboard from prediction history.
  4. sportslab prediction-audit --season 2026
     → Full audit: calibration, confidence buckets, QB-source breakdown,
       worst prediction ledger, GitHub Pages output.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sportslab.evaluation.predict_incumbent import (
    BEST_DECAY,
    BEST_HFA,
    BEST_K,
    BEST_QB_BONUS,
    BEST_REG,
    INCUMBENT_DATE,
    INCUMBENT_FEATURE_SET,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VAL_LL,
    INCUMBENT_VERSION,
)
from sportslab.features.build_features import TARGET_COLUMN

# ── Paths ──
SNAPSHOT_DIR = Path("reports/predictions/snapshots")
HISTORY_PATH = Path("reports/predictions/prediction_history.csv")
MANIFEST_PATH = Path("reports/predictions/snapshot_manifest.json")
REPORT_DIR = Path("reports/predictions")
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"

NOW = datetime.now(timezone.utc)
MANIFEST_VERSION = 1


def _timestamp() -> str:
    return NOW.strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return NOW.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Snapshot manifest ──


def _read_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"manifest_version": MANIFEST_VERSION, "snapshots": []}


def _write_manifest(manifest: Dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def _snapshot_id(season: int, week: int, ts: str) -> str:
    return f"week_{season}_{week:02d}_{ts}"


def _snapshot_path(season: int, week: int) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    return SNAPSHOT_DIR / f"{_snapshot_id(season, week, ts)}.csv"


def _file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


def _register_snapshot(
    path: Path, season: int, week: int, qb_source: str, n_games: int,
) -> str:
    """Add a snapshot entry to the manifest."""
    manifest = _read_manifest()
    sid = path.stem
    entry = {
        "snapshot_id": sid,
        "path": str(path),
        "season": season,
        "week": week,
        "created_at": _iso_now(),
        "model_version": INCUMBENT_VERSION,
        "feature_set": INCUMBENT_FEATURE_SET,
        "calibration": "Platt (logistic on Elo prob + features)",
        "training_seasons": "2021-2024",
        "qb_source": qb_source,
        "n_games": n_games,
        "checksum": _file_checksum(path),
        "data_cutoff": _iso_now(),
        "elo_params": {
            "k": BEST_K,
            "hfa": BEST_HFA,
            "reg": BEST_REG,
            "decay": BEST_DECAY,
            "qb_bonus": BEST_QB_BONUS,
        },
        "graded": False,
        "grade_metrics": None,
        "graded_at": None,
    }
    # Remove any prior manifest entry for same season/week (keep latest)
    manifest["snapshots"] = [
        s for s in manifest["snapshots"]
        if not (s["season"] == season and s["week"] == week)
    ]
    manifest["snapshots"].append(entry)
    _write_manifest(manifest)
    return sid


def _get_snapshot_from_manifest(season: int, week: int) -> Optional[Dict]:
    """Get the latest snapshot entry for a season/week from manifest."""
    manifest = _read_manifest()
    matches = [s for s in manifest["snapshots"]
               if s["season"] == season and s["week"] == week]
    if not matches:
        return None
    return max(matches, key=lambda s: s["created_at"])


def _update_manifest_grade(season: int, week: int, metrics: Dict) -> None:
    """Mark a snapshot as graded in the manifest."""
    manifest = _read_manifest()
    for s in manifest["snapshots"]:
        if s["season"] == season and s["week"] == week:
            s["graded"] = True
            s["grade_metrics"] = {
                "n": metrics["n"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
            }
            s["graded_at"] = _iso_now()
    _write_manifest(manifest)


# ── Prediction ──


def predict_week(
    season: int,
    week: int,
    qb_input: Optional[str] = None,
    snapshot_path: Optional[str] = None,
) -> Dict[str, str]:
    """Predict a single week, save snapshot + report + manifest entry."""
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

    # Detect QB source from snapshot
    snap_df = pd.read_csv(out)
    qb_source = snap_df["qb_source"].iloc[0] if "qb_source" in snap_df.columns else "oracle"
    n_games = len(snap_df)

    # Register in manifest
    _register_snapshot(Path(out), season, week, qb_source, n_games)
    print(f"  Manifest: {MANIFEST_PATH}")

    # Generate weekly report
    from sportslab.evaluation.weekly_report import generate_weekly_report

    rpt = REPORT_DIR / f"week_{season}_{week:02d}_report.md"
    try:
        report_path = generate_weekly_report(
            season=season,
            week=week,
            output=str(rpt),
            input_path=out,
        )
    except FileNotFoundError as e:
        print(f"  Report generation skipped: {e}")
        report_path = str(rpt)

    print(f"\n=== Week {week}, {season} Season ===")
    print(f"  Snapshot: {out}")
    print(f"  Report:   {report_path}")
    print(f"  QB source: {qb_source}")
    print(f"  Games:     {n_games}")

    return {"snapshot": out, "report": report_path}


# ── Grading with guardrail ──


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
    return pd.DataFrame(columns=[
        "season", "week", "n", "log_loss",
        "brier", "accuracy", "auc",
        "model_version", "snapshot", "graded_at",
    ])


def _write_history(df: pd.DataFrame) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_PATH, index=False)


def grade_week(
    season: int,
    week: int,
    snapshot: Optional[str] = None,
) -> Dict:
    """Grade a completed week's predictions.

    Guardrail: must grade from a manifest-registered snapshot.
    Refuses to grade if no snapshot exists (prevents retroactive grading).

    Args:
        season: Season year.
        week: Week number.
        snapshot: Path to snapshot CSV. Auto-detects from manifest if None.

    Returns:
        Dict with metrics and history path.
    """
    # Guardrail: find snapshot via manifest
    manifest_entry = _get_snapshot_from_manifest(season, week)

    if snapshot:
        snap_path = Path(snapshot)
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_path}")
    elif manifest_entry:
        snap_path = Path(manifest_entry["path"])
        if not snap_path.exists():
            raise FileNotFoundError(
                f"Manifest references {snap_path} but file is missing."
                f" Data may have been moved or deleted."
            )
    else:
        raise FileNotFoundError(
            f"No snapshot found for {season} week {week} in manifest."
            f" Run `sportslab predict-week --season {season} --week {week}` first."
            f" Grading from retroactively regenerated predictions is not allowed."
        )

    # Verify snapshot checksum against manifest
    if manifest_entry:
        actual_checksum = _file_checksum(snap_path)
        expected = manifest_entry["checksum"]
        if actual_checksum != expected:
            raise ValueError(
                f"Snapshot checksum mismatch for {snap_path}:\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual_checksum}\n"
                f"  File may have been modified after prediction. Aborting."
            )

    df = pd.read_csv(snap_path)
    if TARGET_COLUMN not in df.columns and "actual_home_win" not in df.columns:
        df = _load_actuals(df)

    has_col = "actual_home_win" in df.columns
    has_actuals = has_col and df["actual_home_win"].notna().sum() > 0
    if not has_actuals:
        raise ValueError(
            f"No actual results found for {season} week {week}."
            f" Games may not be played yet, or feature table needs rebuilding."
        )

    metrics = _compute_metrics(df)
    if metrics["n"] == 0:
        raise ValueError("No graded games found.")

    # Update manifest
    _update_manifest_grade(season, week, metrics)

    # Append to history
    history = _read_history()
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
        "graded_at": _iso_now(),
    }
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    _write_history(history)

    print(f"\n=== Week {week}, {season} Season — Grade ===")
    print(f"  Snapshot: {snap_path}")
    print(f"  Games graded: {metrics['n']}")
    print(f"  Log loss:     {metrics['log_loss']}")
    print(f"  Brier:        {metrics['brier']}")
    print(f"  Accuracy:     {metrics['accuracy']}")
    print(f"  AUC:          {metrics['auc']}")
    print(f"  History:      {HISTORY_PATH}")

    return {"metrics": metrics, "history": str(HISTORY_PATH)}


# ── Season report ──


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
        ll = f"{r['log_loss']:.4f}" if not pd.isna(r["log_loss"]) else "—"
        br = f"{r['brier']:.4f}" if not pd.isna(r["brier"]) else "—"
        ac = f"{r['accuracy']:.4f}"
        au = f"{r['auc']:.4f}" if not pd.isna(r["auc"]) else "—"
        _w(f"| {int(r['week'])} | {int(r['n'])} | {ll} | {br} | {ac} | {au} |")
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
    """Generate season dashboard from prediction history."""
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
