"""Prediction audit dashboard — calibration, confidence buckets, QB breakdown, worst predictions.

Generates comprehensive audit reports for graded seasons. Outputs to
both reports/predictions/ and docs/predictions/ for GitHub Pages.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sportslab.evaluation.predict_incumbent import (
    INCUMBENT_DATE,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
)
from sportslab.evaluation.weekly_pipeline import _read_manifest

REPORT_DIR = Path("reports/predictions")
DOCS_DIR = Path("docs/predictions")

INCUMBENT_NAME = "Standard Elo + qb_changed + rolling_mov_3 + Platt"
INCUMBENT_FEATURES = "qb_changed + rolling_mov_3"
INCUMBENT_VAL_LL = "0.6334"


def _load_graded_snapshots(season: int) -> pd.DataFrame:
    """Load all graded snapshots for a season into a single DataFrame."""
    manifest = _read_manifest()
    graded = [s for s in manifest["snapshots"]
              if s["season"] == season and s.get("graded")]
    if not graded:
        return pd.DataFrame()

    frames = []
    for entry in graded:
        sp = Path(entry["path"])
        if not sp.exists():
            continue
        df = pd.read_csv(sp)
        df["week"] = entry["week"]
        # Merge actuals from feature table
        from sportslab.evaluation.weekly_pipeline import _load_actuals
        if "actual_home_win" not in df.columns:
            df = _load_actuals(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Log loss, safe for single-class buckets. Clips at 1e-15."""
    eps = 1e-15
    y_p = np.clip(y_prob, eps, 1 - eps)
    return float(np.mean(-(y_true * np.log(y_p) + (1 - y_true) * np.log(1 - y_p))))


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score, safe for single-class buckets."""
    return float(np.mean((y_prob - y_true) ** 2))


def _fmt(val) -> str:
    """Format a metric value, replacing None/nan with em-dash."""
    if val is None:
        return "—"
    if isinstance(val, float) and np.isnan(val):
        return "—"
    return f"{val:.4f}"


def _calibration_buckets(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> List[Dict]:
    """Compute calibration buckets (deciles)."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    labels = [f"{i*10}-{(i+1)*10}%" for i in range(10)]
    indices = np.clip(np.floor(y_p * 10).astype(int), 0, 9)
    results = []
    for i in range(10):
        mask = indices == i
        if mask.sum() == 0:
            continue
        y_bin = y_t[mask]
        p_bin = y_p[mask]
        results.append({
            "bucket": labels[i],
            "n": int(mask.sum()),
            "mean_pred": round(float(p_bin.mean()), 4),
            "mean_actual": round(float(y_bin.mean()), 4),
            "cal_error": round(float(abs(p_bin.mean() - y_bin.mean())), 4),
            "log_loss": round(_safe_log_loss(y_bin, p_bin), 4),
            "brier": round(_safe_brier(y_bin, p_bin), 4),
        })
    return results


def _confidence_buckets(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> List[Dict]:
    """Compute confidence buckets (0-20, 20-40, ..., 80-100)."""
    valid = ~np.isnan(y_true)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    conf = np.abs(y_p - 0.5) * 2
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    indices = np.clip(np.floor(conf / 0.2).astype(int), 0, 4)
    results = []
    for i in range(5):
        mask = indices == i
        if mask.sum() == 0:
            continue
        y_bin = y_t[mask]
        p_bin = y_p[mask]
        from sklearn.metrics import accuracy_score
        results.append({
            "bucket": labels[i],
            "n": int(mask.sum()),
            "log_loss": round(_safe_log_loss(y_bin, p_bin), 4),
            "brier": round(_safe_brier(y_bin, p_bin), 4),
            "accuracy": round(float(accuracy_score(y_bin, p_bin >= 0.5)), 4),
        })
    return results


def _worst_predictions(
    df: pd.DataFrame, n: int = 20,
) -> List[Dict]:
    """Find the worst predictions by log loss contribution."""
    if df.empty or "actual_home_win" not in df.columns:
        return []
    valid = df["actual_home_win"].notna().values
    y_t = df.loc[valid, "actual_home_win"].astype(int).values
    y_p = np.clip(df.loc[valid, "incumbent_home_win_prob"].values, 1e-15, 1 - 1e-15)
    contrib = -(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p))
    worst = np.argsort(-contrib)[:n]
    results = []
    for i in worst:
        idx = df.index[valid][i]
        results.append({
            "game_id": str(df.loc[idx, "game_id"]),
            "season": int(df.loc[idx, "season"]) if "season" in df.columns else 0,
            "week": int(df.loc[idx, "week"]) if "week" in df.columns else 0,
            "away_team": str(df.loc[idx, "away_team"]),
            "home_team": str(df.loc[idx, "home_team"]),
            "actual_home_win": int(y_t[i]),
            "prob": round(float(y_p[i]), 4),
            "log_loss_contrib": round(float(contrib[i]), 4),
        })
    return results


def _qb_source_breakdown(
    manifest: Dict, season: int,
) -> Optional[Dict]:
    """Compare oracle vs live_pregame graded performance."""
    graded = [s for s in manifest["snapshots"]
              if s["season"] == season and s.get("graded")]
    if not graded:
        return None

    oracle = [s for s in graded if s["qb_source"] == "oracle" and s.get("grade_metrics")]
    live = [s for s in graded if s["qb_source"] == "live_pregame" and s.get("grade_metrics")]

    def _avg_metrics(items: List[Dict]) -> Dict:
        if not items:
            return {"n_weeks": 0, "n_games": 0, "log_loss": None,
                    "brier": None, "accuracy": None}
        m = [s["grade_metrics"] for s in items]
        return {
            "n_weeks": len(m),
            "n_games": sum(g["n"] for g in m),
            "log_loss": round(
                float(np.mean([g["log_loss"] for g in m if g["log_loss"] is not None])), 4),
            "brier": round(
                float(np.mean([g["brier"] for g in m if g["brier"] is not None])), 4),
            "accuracy": round(float(np.mean([g["accuracy"] for g in m])), 4),
        }

    return {
        "oracle": _avg_metrics(oracle),
        "live_pregame": _avg_metrics(live),
    }


def _write_audit_pages(
    season: int, df_all: pd.DataFrame,
    calibration: List, confidence: List, worst: List,
    qb_breakdown: Optional[Dict],
    manifest: Dict,
    mode: str = "live",
) -> Dict[str, str]:
    """Write audit report to reports/ and docs/."""
    lines = []
    _w = lines.append

    label = "Historical Rehearsal" if mode == "rehearsal" else "Prediction Audit"
    _w(f"# {label} — {season} Season\n")
    _w(f"*Generated by `sportslab prediction-audit` ({INCUMBENT_VERSION}, {INCUMBENT_DATE})*\n")
    _w("")

    m_entry = _get_last_manifest_entry(manifest)
    _w("## Manifest\n")
    _w("")
    _w("| Field | Value |")
    _w("|-------|-------|")
    _w(f"| Manifest version | {manifest.get('manifest_version', '?')} |")
    _w(f"| Snapshots tracked | {len(manifest.get('snapshots', []))} |")
    _w(f"| Graded weeks | {len([s for s in manifest['snapshots'] if s.get('graded')])} |")
    _w(f"| Model | {INCUMBENT_VERSION} |")
    _w(f"| Holdout LL | {INCUMBENT_HOLDOUT_LL} |")
    if m_entry:
        _w(f"| Last snapshot | {m_entry.get('snapshot_id', '?')} |")
        _w(f"| Last graded | {m_entry.get('graded_at', '?')} |")
    _w("")

    if df_all.empty:
        _w("No graded snapshots found. Run `sportslab grade-week` first.\n")
        _w("\n---\n")
        _w(f"*{INCUMBENT_VERSION}*\n")
        content = "\n".join(lines)
        return _write_both(content, season)

    _w("## Season Calibration\n")
    _w("")
    _w("Calibration buckets across all graded weeks.\n")
    _w("")
    if calibration:
        _w("| Bucket | N | Mean Pred | Mean Actual | Cal Error | Log Loss | Brier |")
        _w("|--------|---|-----------|-------------|-----------|----------|-------|")
        for b in calibration:
            _w(f"| {b['bucket']} | {b['n']} | {b['mean_pred']:.4f}"
               f" | {b['mean_actual']:.4f} | {b['cal_error']:.4f}"
               f" | {_fmt(b['log_loss'])} | {_fmt(b['brier'])} |")
        _w("")

    _w("## Confidence Buckets\n")
    _w("")
    _w("Confidence = 2 × |prob − 0.5|. Higher = more confident.\n")
    _w("")
    if confidence:
        _w("| Confidence | N | Log Loss | Brier | Accuracy |")
        _w("|-----------|---|----------|-------|----------|")
        for b in confidence:
            _w(f"| {b['bucket']} | {b['n']} | {_fmt(b['log_loss'])}"
               f" | {_fmt(b['brier'])} | {b['accuracy']} |")
        _w("")

    if qb_breakdown:
        _w("## QB Source Performance\n")
        _w("")
        _w("| QB Source | Weeks | Games | Log Loss | Brier | Accuracy |")
        _w("|-----------|-------|-------|----------|-------|----------|")
        for src, m in qb_breakdown.items():
            if m["n_weeks"] == 0:
                continue
            ll_str = f" {m['log_loss']}" if m["log_loss"] is not None else " —"
            br_str = f" {m['brier']}" if m["brier"] is not None else " —"
            _w(f"| {src} | {m['n_weeks']} | {m['n_games']}"
               f" |{ll_str} |{br_str} | {m['accuracy']} |")
        _w("")

    _w("## Season-Week Performance\n")
    _w("")
    _w("| Week | Games | Log Loss | Brier | Accuracy | AUC |")
    _w("|------|-------|----------|-------|----------|-----|")
    graded = sorted([s for s in manifest["snapshots"]
                     if s["season"] == season and s.get("graded")],
                    key=lambda s: s["week"])
    for s in graded:
        gm = s.get("grade_metrics", {})
        _w(f"| {s['week']} | {gm.get('n', '?')}"
           f" | {gm.get('log_loss', '—')}"
           f" | {gm.get('brier', '—')}"
           f" | {gm.get('accuracy', '—')}"
           f" | {gm.get('auc', '—')} |")
    _w("")

    _w("## Worst Predictions\n")
    _w("")
    if worst:
        _w("| # | Game ID | Wk | Away | Home | Actual | Prob | Log Loss |")
        _w("|---|---------|----|------|------|--------|------|----------|")
        for i, w in enumerate(worst, 1):
            _w(f"| {i} | {w['game_id']} | {w['week']}"
               f" | {w['away_team']} | {w['home_team']}"
               f" | {w['actual_home_win']} | {w['prob']:.4f}"
               f" | {w['log_loss_contrib']:.4f} |")
        _w("")

    _w("## Prediction History Validation\n")
    _w("")
    _w("| Check | Status |")
    _w("|-------|--------|")
    total_snapshots = len(manifest.get("snapshots", []))
    total_graded = len([s for s in manifest["snapshots"] if s.get("graded")])

    from sportslab.evaluation.weekly_pipeline import _read_history
    hist = _read_history()
    hist_weeks = len(hist)
    _w(f"| Total snapshots in manifest | {total_snapshots} |")
    _w(f"| Graded snapshots | {total_graded} |")
    _w(f"| History entries | {hist_weeks} |")
    _w(f"| All graded in history | {'✓' if hist_weeks == total_graded else '✗'} |")
    missing = []
    for s in manifest["snapshots"]:
        if s.get("graded"):
            sp = Path(s["path"])
            if not sp.exists():
                missing.append(s["snapshot_id"])
    if missing:
        _w(f"| Missing snapshot files | {len(missing)} |")
    else:
        _w("| All snapshot files exist | ✓ |")

    checksum_ok = 0
    for s in manifest["snapshots"]:
        sp = Path(s["path"])
        if sp.exists():
            from sportslab.evaluation.weekly_pipeline import _file_checksum
            if _file_checksum(sp) == s["checksum"]:
                checksum_ok += 1
    _w(f"| Snapshots with matching checksums | {checksum_ok}/{total_snapshots} |")
    _w("")

    mode_label = " [Rehearsal]" if mode == "rehearsal" else ""
    _w("---\n")
    _w(f"*Audit generated by SportsLab NFL Incumbent {INCUMBENT_VERSION}{mode_label}.*\n")

    return _write_both("\n".join(lines), season, mode=mode)


def _write_both(content: str, season: int, mode: str = "live") -> Dict[str, str]:
    """Write report to reports/ and optionally docs/."""
    paths = {}
    for base in ([REPORT_DIR, DOCS_DIR] if mode == "live" else [REPORT_DIR]):
        dest = base / f"audit_{season}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        paths[str(base.name)] = str(dest)
    return paths


def _get_last_manifest_entry(manifest: Dict) -> Optional[Dict]:
    snaps = manifest.get("snapshots", [])
    return max(snaps, key=lambda s: s.get("created_at", "")) if snaps else None


def run_prediction_audit(season: int, mode: str = "live") -> Dict[str, str]:
    """Run full prediction audit for a season.

    Generates: calibration, confidence buckets, QB-source breakdown,
    worst predictions, week-by-week table, history validation.

    Args:
        season: Season year.
        mode: ``"live"`` (default, writes to reports/ and docs/)
              or ``"rehearsal"`` (writes to rehearsal dir only, labels as rehearsal).

    Returns:
        Dict mapping output location to file path.
    """
    manifest = _read_manifest()

    df_all = _load_graded_snapshots(season)

    calibration = []
    confidence = []
    worst = []
    qb_breakdown = None

    if not df_all.empty:
        y_true = df_all["actual_home_win"].values.astype(float)
        y_prob = df_all["incumbent_home_win_prob"].values
        valid = ~np.isnan(y_true)

        calibration = _calibration_buckets(y_true[valid], y_prob[valid])
        confidence = _confidence_buckets(y_true[valid], y_prob[valid])
        worst = _worst_predictions(df_all)
        qb_breakdown = _qb_source_breakdown(manifest, season)

    paths = _write_audit_pages(
        season, df_all, calibration, confidence, worst,
        qb_breakdown, manifest, mode=mode,
    )

    print(f"\n=== Prediction Audit: {season} ===")
    n_graded = len([s for s in manifest["snapshots"]
                     if s["season"] == season and s.get("graded")])
    print(f"  Graded snapshots: {n_graded}")
    print("  Reports:")
    for loc, p in paths.items():
        print(f"    {loc}: {p}")

    return paths


def build_prediction_index() -> str:
    """Generate docs/predictions/index.md from manifest state."""
    manifest = _read_manifest()
    snaps = manifest.get("snapshots", [])

    seasons = sorted(set(s["season"] for s in snaps))
    graded = [s for s in snaps if s.get("graded")]
    n_graded_weeks = len(graded)
    n_graded_games = sum(
        s.get("grade_metrics", {}).get("n", 0) for s in graded
    )

    lines = []
    _w = lines.append

    _w("# Predictions Dashboard\n")
    _w("*Automatically generated by `sportslab build-prediction-index`*\n")
    _w("")

    _w("## Season Status\n")
    _w("")
    _w("| Metric | Value |")
    _w("|--------|-------|")
    _w(f"| Seasons tracked | {len(seasons)} ({', '.join(str(s) for s in seasons)}) |")
    _w(f"| Graded weeks | {n_graded_weeks} |")
    _w(f"| Graded games | {n_graded_games} |")
    _w(f"| Incumbent model | {INCUMBENT_NAME} |")
    _w(f"| Version | {INCUMBENT_VERSION} |")
    _w(f"| Holdout LL | {INCUMBENT_HOLDOUT_LL} |")
    _w(f"| Feature set | {INCUMBENT_FEATURES} |")
    _w("")

    _w("## Quick Links\n")
    _w("")
    _w("- [Weekly Runbook]({{ site.baseurl }}/weekly_runbook)")
    _w("- [Full Predictions CSV (GitHub)]"
       "(https://github.com/SecuritahGuy/sports-ml-lab/blob/main/"
       "reports/predictions/incumbent_predictions.csv)")
    _w("- [2025 Holdout CSV (GitHub)]"
       "(https://github.com/SecuritahGuy/sports-ml-lab/blob/main/"
       "reports/predictions/incumbent_predictions_2025_holdout.csv)")
    _w("")

    _w("## Audit Reports\n")
    _w("")
    audit_files = sorted(DOCS_DIR.glob("audit_*.md"))
    if audit_files:
        _w("| Season | Report |")
        _w("|--------|--------|")
        for af in audit_files:
            season = af.stem.replace("audit_", "")
            _w(f"| {season} | [Audit Report](./{af.name}) |")
    else:
        _w("No audit reports found. Run `sportslab prediction-audit` after grading.\n")
    _w("")

    _w("## Prediction Schema\n")
    _w("")
    _w("Each snapshot contains per-game predictions with the following columns:\n")
    _w("")
    _w("| Column | Description |")
    _w("|--------|-------------|")
    _w("| `game_id` | Unique game identifier |")
    _w("| `incumbent_home_win_prob` | Predicted home win probability (0–1) |")
    _w("| `confidence_bucket` | Probability range (50-55, 55-60, …, 80+) |")
    _w("| `caution_qb_change` | QB did not start prior game |")
    _w("| `caution_early_season` | Week ≤ 4 |")
    _w("| `qb_source` | `oracle` or `live_pregame` |")
    _w("| `model_version` | Incumbent version |")
    _w("")

    _w("## Confidence Buckets\n")
    _w("")
    _w("| Bucket | Range | Description |")
    _w("|--------|-------|-------------|")
    _w("| 50-55 | 0.50–0.55 | Near coin flip |")
    _w("| 55-60 | 0.55–0.60 | Slight favorite |")
    _w("| 60-65 | 0.60–0.65 | Moderate favorite |")
    _w("| 65-70 | 0.65–0.70 | Solid favorite |")
    _w("| 70-80 | 0.70–0.80 | Strong favorite |")
    _w("| 80+ | 0.80+ | Heavy favorite |")
    _w("")

    _w("---\n")
    _w(f"*Page generated by SportsLab NFL Incumbent {INCUMBENT_VERSION}.*\n")

    content = "\n".join(lines)
    dest = DOCS_DIR / "index.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    print(f"  Prediction index: {dest}")
    return str(dest)
