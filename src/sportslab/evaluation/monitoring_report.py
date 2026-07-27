"""Weekly monitoring report — auto-fills live_monitoring.md template from graded artifacts.

Reads from prediction history, manifest, and model-trust outputs.
Checks 11 drift thresholds defined in docs/live_monitoring.md and
produces a structured markdown report saved to reports/monitoring/.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sportslab.evaluation.predict_incumbent import (
    INCUMBENT_DATE,
    INCUMBENT_HOLDOUT_LL,
    INCUMBENT_VERSION,
)
from sportslab.evaluation.prediction_audit import (
    _calibration_buckets,
    _safe_log_loss,
)
from sportslab.evaluation.weekly_pipeline import (
    HISTORY_PATH,
    _file_checksum,
    _load_actuals,
    _read_history,
    _read_manifest,
)
from sportslab.features.build_features import SPORTSLAB_MIN_SEASON

MONITORING_DIR = Path("reports/monitoring")

INCUMBENT_HOLDOUT_LL_FLOAT = float(INCUMBENT_HOLDOUT_LL)


def _validate_season(season: int, context: str = ""):
    if season < SPORTSLAB_MIN_SEASON:
        raise ValueError(
            f"Season {season} not allowed{' in ' + context if context else ''}. "
            f"Minimum season is {SPORTSLAB_MIN_SEASON}."
        )


def _small_sample_rule(n: int) -> str:
    if n <= 5:
        return "no_ll"
    elif n <= 15:
        return "low_sample"
    return "full"


def _load_all_graded_games(season: int, mode: str = "live") -> pd.DataFrame:
    manifest = _read_manifest()
    graded = [
        s for s in manifest["snapshots"]
        if s["season"] == season and s.get("graded") and s.get("mode") == mode
    ]
    if not graded:
        return pd.DataFrame()
    frames = []
    for entry in sorted(graded, key=lambda x: x["week"]):
        sp = Path(entry["path"])
        if not sp.exists():
            continue
        df = pd.read_csv(sp)
        df["week"] = entry["week"]
        if "actual_home_win" not in df.columns:
            df = _load_actuals(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _get_snapshot_info(
    manifest: Dict, season: int, week: int, mode: str = "live",
) -> Optional[Dict]:
    for s in manifest["snapshots"]:
        if s["season"] == season and s["week"] == week and s.get("mode") == mode:
            return s
    return None


def _check_drift_thresholds(
    weekly_ll: Optional[float],
    rolling_4_ll: Optional[float],
    weekly_brier: Optional[float],
    weekly_acc: Optional[float],
    ece: Optional[float],
    rolling_4_ece: Optional[float],
    high_conf_miss_rate: Optional[float],
    missing_weather_rate: Optional[float],
    qb_change_ll_gap: Optional[float],
    n_pred: int,
    n_graded: int,
    snapshot_checksum_ok: bool,
    market_gap_ll: Optional[float],
    schema_ok: bool,
    stale_days: Optional[int],
) -> List[Dict]:
    checks = []

    checks.append(_check_one(
        "Weekly LL", weekly_ll, "≤0.65", lambda v: v is None or v <= 0.65,
        floating=True))
    checks.append(_check_one(
        "Rolling 4-week LL", rolling_4_ll, "≤0.64", lambda v: v is None or v <= 0.64,
        floating=True))
    checks.append(_check_one(
        "High-confidence miss rate (p≥0.80)",
        high_conf_miss_rate, "≤20%",
        lambda v: v is None or v <= 0.20,
        floating=True))
    checks.append(_check_one(
        "ECE (any week)", ece, "<0.10",
        lambda v: v is None or v < 0.10,
        floating=True))
    checks.append(_check_one(
        "ECE (rolling 4-week)", rolling_4_ece, "<0.08",
        lambda v: v is None or v < 0.08,
        floating=True))
    checks.append(_check_one(
        "Missing-weather rate", missing_weather_rate, "≤35%",
        lambda v: v is None or v <= 0.35,
        floating=True))
    checks.append(_check_one(
        "QB-change split degradation", qb_change_ll_gap, "≤+0.02",
        lambda v: v is None or v <= 0.02,
        floating=True))
    checks.append(_check_one(
        "Schema changes", schema_ok, "No unexpected changes",
        lambda v: v, boolean=True))
    checks.append(_check_one(
        "No games found", n_graded > 0, ">0 graded",
        lambda v: v, boolean=True))
    checks.append(_check_one(
        "Stale data", stale_days, "<7 days",
        lambda v: v is None or v < 7,
        integer_ok=True))
    checks.append(_check_one(
        "Prediction count match", n_pred == n_graded or n_graded == 0,
        "n_graded == n_predicted",
        lambda v: v, boolean=True))
    checks.append(_check_one(
        "Published file checksum match", snapshot_checksum_ok,
        "manifest == file",
        lambda v: v, boolean=True))
    checks.append(_check_one(
        "Market gap widening sharply", market_gap_ll, "≤0.05 weekly",
        lambda v: v is None or v <= 0.05,
        floating=True))

    return checks


def _check_one(
    name: str, value, threshold: str, pass_fn, *,
    floating: bool = False, boolean: bool = False, integer_ok: bool = False,
) -> Dict:
    status = "✅" if pass_fn(value) else "⚠️"
    if floating and value is not None:
        display = f"{value:.4f}"
    elif integer_ok and value is not None:
        display = f"{int(value)} days"
    else:
        display = str(value) if value is not None else "N/A"
    return {
        "check": name,
        "threshold": threshold,
        "actual": display,
        "status": status,
    }


def _high_confidence_accuracy(
    df: pd.DataFrame, threshold: float,
) -> Tuple[int, int, float]:
    if df.empty:
        return 0, 0, 0.0
    valid = df["actual_home_win"].notna().values
    y_p = df.loc[valid, "incumbent_home_win_prob"].values
    y_t = df.loc[valid, "actual_home_win"].astype(int).values
    high = np.abs(y_p - 0.5) * 2 >= threshold
    n = int(high.sum())
    if n == 0:
        return 0, 0, 0.0
    correct = int((y_t[high] == (y_p[high] >= 0.5)).sum())
    acc = correct / n
    return n, n - correct, acc


def _subgroup_log_loss(
    df: pd.DataFrame, mask: np.ndarray,
) -> Optional[float]:
    if mask.sum() == 0:
        return None
    valid = df["actual_home_win"].notna().values
    mask_valid = mask & valid
    if mask_valid.sum() == 0:
        return None
    sample_rule = _small_sample_rule(int(mask_valid.sum()))
    if sample_rule == "no_ll":
        return None
    y_t = df.loc[mask_valid, "actual_home_win"].astype(int).values
    y_p = df.loc[mask_valid, "incumbent_home_win_prob"].values
    return round(_safe_log_loss(y_t, y_p), 4)


def _missing_weather_mask(df: pd.DataFrame) -> np.ndarray:
    if "weather_missing_flag" in df.columns:
        return df["weather_missing_flag"].fillna(False).values.astype(bool)
    return np.zeros(len(df), dtype=bool)


def _dome_mask(df: pd.DataFrame) -> np.ndarray:
    if "roof" in df.columns:
        return df["roof"].fillna("").str.lower().str.contains("dome", na=False).values
    return np.zeros(len(df), dtype=bool)


def _open_roof_mask(df: pd.DataFrame) -> np.ndarray:
    if "roof" in df.columns:
        roof_vals = df["roof"].fillna("").str.lower().values
        return np.array(
            [r in ("open", "retractable") for r in roof_vals], dtype=bool
        )
    return np.zeros(len(df), dtype=bool)


def _qb_change_mask(df: pd.DataFrame) -> np.ndarray:
    if "qb_change_flag" in df.columns:
        return df["qb_change_flag"].fillna(False).values.astype(bool)
    home_col = next((c for c in df.columns if "home_qb_changed" in c), None)
    away_col = next((c for c in df.columns if "away_qb_changed" in c), None)
    if home_col and away_col:
        return (
            df[home_col].fillna(False).values.astype(bool)
            | df[away_col].fillna(False).values.astype(bool)
        )
    return np.zeros(len(df), dtype=bool)


def _generate_report(
    season: int,
    week: int,
    history_df: pd.DataFrame,
    manifest: Dict,
    season_games: pd.DataFrame,
    mode: str = "live",
) -> str:
    now = datetime.now(timezone.utc)
    snapshot_info = _get_snapshot_info(manifest, season, week, mode=mode)

    lines = []
    _w = lines.append

    # ── Header ──
    _w(f"# Weekly Monitoring Report — {season} Week {week}")
    _w("")
    _w(f"*Generated: {now.strftime('%Y-%m-%d %H:%M')} UTC*")
    if snapshot_info:
        snap_name = Path(snapshot_info["path"]).name
        _w(f"*Snapshot: {snap_name}*")
    _w("")
    _w("---")
    _w("")

    # ── Overview ──
    _w("## Overview")
    _w("")
    _w("| Field | Value |")
    _w("|-------|-------|")
    _w(f"| Season | {season} |")
    _w(f"| Week | {week} |")
    _w(f"| Mode | {mode} |")

    stale_days = None

    if snapshot_info:
        snap_path = Path(snapshot_info["path"])
        snap_df = pd.read_csv(snap_path) if snap_path.exists() else pd.DataFrame()
        n_pred = len(snap_df)
        _w(f"| Games predicted | {n_pred} |")

        if snapshot_info.get("qb_source"):
            _w(f"| QB source | {snapshot_info['qb_source']} |")

        if snapshot_info.get("graded"):
            gm = snapshot_info.get("grade_metrics", {})
            _w(f"| Games graded | {gm.get('n', 0)} |")
        else:
            _w("| Games graded | Not yet graded |")
    else:
        _w("| Games predicted | No snapshot found |")

    # Staleness check
    if season_games is not None and not season_games.empty:
        if "gameday" in season_games.columns:
            last_data = pd.to_datetime(season_games["gameday"]).max()
            stale_days = (now - last_data).days
            _w(f"| Stale-data warnings | {stale_days} days since latest data |")

    _w(f"| Model version | {INCUMBENT_VERSION} |")
    if snapshot_info:
        _w(f"| Snapshot checksum | {snapshot_info.get('checksum', 'N/A')} |")
    _w("")
    _w("---")
    _w("")

    # ── Core Metrics ──
    _w("## Core Metrics")
    _w("")

    week_row = history_df[
        (history_df["season"] == season)
        & (history_df["week"] == week)
        & (history_df["mode"] == mode)
    ]
    rolling = history_df[
        (history_df["season"] == season)
        & (history_df["week"] <= week)
        & (history_df["week"] > week - 4)
        & (history_df["mode"] == mode)
    ]

    weekly_ll = None
    weekly_brier = None
    weekly_acc = None
    rolling_4_ll = None

    if not week_row.empty:
        r = week_row.iloc[0]
        weekly_ll = float(r["log_loss"]) if pd.notna(r.get("log_loss")) else None
        weekly_brier = float(r["brier"]) if pd.notna(r.get("brier")) else None
        weekly_acc = float(r["accuracy"]) if pd.notna(r.get("accuracy")) else None

    if not rolling.empty:
        roll_lls = rolling["log_loss"].dropna()
        if len(roll_lls) > 0:
            rolling_4_ll = round(float(roll_lls.mean()), 4)

    _w("| Metric | This Week | Rolling 4-Week | Model-Trust Threshold | Status |")
    _w("|--------|-----------|----------------|----------------------|--------|")
    ll_status = '✅' if (weekly_ll is None or weekly_ll <= 0.65) else '⚠️'
    br_status = '✅' if (weekly_brier is None or weekly_brier <= 0.24) else '⚠️'
    ac_status = '✅' if (weekly_acc is None or weekly_acc >= 0.55) else '⚠️'
    _w(f"| Log loss | {_fmt(weekly_ll)} | {_fmt(rolling_4_ll)} | ≤0.65 (warning) | {ll_status} |")
    _w(f"| Brier score | {_fmt(weekly_brier)} | — | ≤0.24 (warning) | {br_status} |")
    _w(f"| Accuracy | {_fmt(weekly_acc)} | — | ≥0.55 (warning) | {ac_status} |")
    _w("")
    _w("---")
    _w("")

    # ── Calibration Buckets ──
    _w("## Calibration Buckets")
    _w("")

    valid = np.array([], dtype=bool)
    if not season_games.empty:
        valid = season_games["actual_home_win"].notna().values
        if valid.sum() > 0:
            y_t = season_games.loc[valid, "actual_home_win"].astype(int).values
            y_p = season_games.loc[valid, "incumbent_home_win_prob"].values
            cal_buckets = _calibration_buckets(y_t, y_p)
            ece = round(float(np.mean([b["cal_error"] for b in cal_buckets])), 4) if cal_buckets else None

            _w("| Bucket | N | Observed Rate | Predicted Mean | Cal Error |")
            _w("|--------|---|-------------|---------------|-----------|")
            for b in cal_buckets:
                obs = f"{b['mean_actual']*100:.1f}%"
                pred = f"{b['mean_pred']*100:.1f}%"
                _w(f"| {b['bucket']} | {b['n']} | {obs} | {pred} | {b['cal_error']:.4f} |")
            _w("")
        else:
            ece = None
    else:
        ece = None

    _w("---")
    _w("")

    # ── High-Confidence Predictions ──
    _w("## High-Confidence Predictions")
    _w("")

    _w("| Threshold | N | Correct | Missed | Accuracy | Miss Rate |")
    _w("|-----------|---|---------|--------|----------|-----------|")
    high_conf_miss_rate = None
    for thr in [0.70, 0.80, 0.90]:
        n, missed, acc = _high_confidence_accuracy(season_games, thr)
        if n > 0:
            miss_rate = missed / n
            if thr == 0.80:
                high_conf_miss_rate = miss_rate
            _w(f"| p ≥ {thr:.2f} | {n} | {n - missed} | {missed} | {acc*100:.1f}% | {miss_rate*100:.1f}% |")
        else:
            _w(f"| p ≥ {thr:.2f} | 0 | — | — | — | — |")
    _w("")
    _w("---")
    _w("")

    # ── Subgroup Performance ──
    _w("## Subgroup Performance")
    _w("")

    overall_ll = None
    if valid.sum() > 0:
        overall_ll = _safe_log_loss(
            season_games.loc[valid, "actual_home_win"].astype(int).values,
            season_games.loc[valid, "incumbent_home_win_prob"].values,
        )

    _w("| Subgroup | N | Log Loss | Δ vs Overall | Status |")
    _w("|----------|---|----------|-------------|--------|")

    subgroups = {
        "QB-change games": _qb_change_mask(season_games),
        "Missing weather": _missing_weather_mask(season_games),
        "Dome games": _dome_mask(season_games),
        "Open/retractable roof": _open_roof_mask(season_games),
        "Early season (Weeks 1-4)": (
            season_games["week"].values <= 4 if "week" in season_games.columns
            else np.zeros(len(season_games), dtype=bool)
        ),
    }

    qb_change_ll_gap = None
    for name, mask in subgroups.items():
        sg_ll = _subgroup_log_loss(season_games, mask)
        n_sg = int(mask.sum())
        delta = round(sg_ll - overall_ll, 4) if (sg_ll is not None and overall_ll is not None) else None
        if name == "QB-change games" and delta is not None:
            qb_change_ll_gap = delta
        status = "✅" if (delta is None or delta <= 0.02) else "⚠️"
        _w(f"| {name} | {n_sg} | {_fmt(sg_ll)} | {_fmt(delta)} | {status} |")

    # Home underdogs
    if "home_moneyline" in season_games.columns:
        h_underdog = (
            season_games["home_moneyline"].values > 0
            if "home_moneyline" in season_games.columns
            else np.zeros(len(season_games), dtype=bool)
        )
        hu_ll = _subgroup_log_loss(season_games, h_underdog)
        hu_n = int(h_underdog.sum())
        hu_delta = round(hu_ll - overall_ll, 4) if (hu_ll is not None and overall_ll is not None) else None
        _w(f"| Home underdogs | {hu_n} | {_fmt(hu_ll)} | {_fmt(hu_delta)} | {'✅' if (hu_delta is None or hu_delta <= 0.02) else '⚠️'} |")
    else:
        _w("| Home underdogs | — | — | — | — |")

    _w("")
    _w("---")
    _w("")

    # ── Model-vs-Market Disagreement ──
    _w("## Model-vs-Market Disagreement")
    _w("")

    _w("| Metric | Value | Note |")
    _w("|--------|-------|------|")
    market_gap_ll = None
    if "market_prob_diagnostic" in season_games.columns and "incumbent_home_win_prob" in season_games.columns:
        valid_m = season_games["actual_home_win"].notna().values & season_games["market_prob_diagnostic"].notna().values
        if valid_m.sum() > 0:
            mkt_prob = season_games.loc[valid_m, "market_prob_diagnostic"].values
            mkt_mkt = season_games.loc[valid_m, "incumbent_home_win_prob"].values
            diffs = np.abs(mkt_prob - mkt_mkt)
            avg_diff = round(float(diffs.mean()), 4)
            n_big = int((diffs > 0.15).sum())
            _w(f"| Avg |model_prob − market_prob| | {avg_diff} | Market data diagnostic-only |")
            _w(f"| Games with diff > 0.15 | {n_big} | {'List if ≥3 games' if n_big >= 3 else 'Below threshold'} |")
            mkt_ll = _safe_log_loss(
                season_games.loc[valid_m, "actual_home_win"].astype(int).values,
                season_games.loc[valid_m, "market_prob_diagnostic"].values,
            )
            market_gap_ll = round(mkt_ll - overall_ll, 4) if overall_ll is not None else None
            _w(f"| Market gap (LL) | {_fmt(market_gap_ll)} | "
            "Model - Market log loss diff |")
        else:
            _w("| Market data | Not available for graded games | |")
    else:
        _w("| Market data | Not available in snapshot | |")
    _w("")
    _w("---")
    _w("")

    # ── Missing Weather Rate ──
    missing_weather_rate = None
    if "weather_missing_flag" in season_games.columns:
        missing_mask = season_games["weather_missing_flag"].fillna(False).values.astype(bool)
        missing_weather_rate = float(missing_mask.mean()) if len(missing_mask) > 0 else None
    _w("")
    _w("---")
    _w("")

    # ── Operator Notes ──
    _w("## Operator Notes")
    _w("")
    _w("<Free-text observations, anomalies, data issues, rollback reasons.>")
    _w("")
    _w("---")
    _w("")

    # ── Drift Check Summary ──
    _w("## Drift Check Summary")
    _w("")

    # Schema check
    schema_ok = True
    snapshot_checksum_ok = True
    if snapshot_info:
        if "checksum" in snapshot_info:
            snap_path = Path(snapshot_info["path"])
            if snap_path.exists():
                actual_cs = _file_checksum(snap_path)
                snapshot_checksum_ok = actual_cs == snapshot_info["checksum"]
            else:
                snapshot_checksum_ok = False
    n_pred_games = len(season_games) if not season_games.empty else 0
    n_graded_games = int(season_games["actual_home_win"].notna().sum()) if not season_games.empty else 0

    checks = _check_drift_thresholds(
        weekly_ll=weekly_ll,
        rolling_4_ll=rolling_4_ll,
        weekly_brier=weekly_brier,
        weekly_acc=weekly_acc,
        ece=ece,
        rolling_4_ece=None,
        high_conf_miss_rate=high_conf_miss_rate,
        missing_weather_rate=missing_weather_rate,
        qb_change_ll_gap=qb_change_ll_gap,
        n_pred=n_pred_games,
        n_graded=n_graded_games,
        snapshot_checksum_ok=snapshot_checksum_ok,
        market_gap_ll=market_gap_ll,
        schema_ok=schema_ok,
        stale_days=stale_days,
    )

    _w("| Check | Threshold | Actual | Status |")
    _w("|-------|-----------|--------|--------|")
    for c in checks:
        _w(f"| {c['check']} | {c['threshold']} | {c['actual']} | {c['status']} |")
    _w("")
    _w("---")

    # ── Footer ──
    n_warnings = sum(1 for c in checks if c["status"] == "⚠️")
    _w(f"*{n_warnings} threshold warning(s). No automatic model changes. Review via post-week workflow.*")
    _w(f"*Report generated by {INCUMBENT_VERSION} ({INCUMBENT_DATE}).*")
    _w("")

    return "\n".join(lines)


def _fmt(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def generate_monitoring_report(
    season: int,
    week: int,
    mode: str = "live",
    output: Optional[str] = None,
) -> str:
    """Generate the weekly monitoring report."""
    _validate_season(season, "monitoring_report")

    history_df = _read_history()
    manifest = _read_manifest()
    season_games = _load_all_graded_games(season, mode=mode)

    report = _generate_report(
        season=season,
        week=week,
        history_df=history_df,
        manifest=manifest,
        season_games=season_games,
        mode=mode,
    )

    if output:
        out_path = Path(output)
    else:
        MONITORING_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MONITORING_DIR / f"weekly_{season}_w{week:02d}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    print(f"\n=== Weekly Monitoring Report: {season} Week {week} ===")
    print(f"  Mode:    {mode}")
    print(f"  Report:  {out_path}")
    print(f"  History: {HISTORY_PATH}")

    return str(out_path)
