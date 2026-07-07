"""Model trust diagnostics — structured trust report for the incumbent.

Produces a comprehensive trust report covering incumbent reproduction,
failure-mode splits, market benchmark comparison, high-confidence
analysis, and reproducibility verification.

No network access required. Reads from existing artifacts.
"""

import importlib.util
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Lazy dependency loading ──────────────────────────────────────────

_LIBS: Dict[str, bool] = {}

try:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    _LIBS["sklearn"] = True
except ImportError:
    _LIBS["sklearn"] = False

_LIBS["metrics"] = importlib.util.find_spec(
    "sportslab.evaluation.metrics"
) is not None

try:
    from sportslab.evaluation.season_regression_experiment import (
        build_team_regression_overrides,
    )

    _LIBS["season_regression"] = True
except ImportError:
    _LIBS["season_regression"] = False

try:
    from sportslab.features.ratings import compute_elo_features

    _LIBS["ratings"] = True
except ImportError:
    _LIBS["ratings"] = False

# ── Constants ────────────────────────────────────────────────────────

INCUMBENT_HOLDOUT_LL = 0.6200
INCUMBENT_VERSION = "v3.0.0"

PREDICTIONS_PATH = "reports/predictions/incumbent_predictions.csv"
FEATURE_TABLE_PATH = "data/features/nfl/feature_table.parquet"
REPORT_PATH = "reports/experiments/model_trust.md"

HOLDOUT_SEASON = 2025

CONFIDENCE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

# Sensitivity threshold for over/underconfidence flag
OVER_UNDER_THRESHOLD = 0.01


def _check_libs(*names: str) -> None:
    """Raise RuntimeError if any required library is missing."""
    missing = [n for n in names if not _LIBS.get(n)]
    if missing:
        raise RuntimeError(
            f"Required dependencies not available: {missing}. "
            f"Try: pip install scikit-learn sportslab"
        )


# ── ECE Computation ──────────────────────────────────────────────────


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Expected Calibration Error with 10 equal-width bins.

    Formula: sum(n_bin * |p_pred - p_actual|) / N
    """
    n_total = len(y_true)
    if n_total == 0:
        return {"ece": 0.0, "n_bins": n_bins, "n": 0, "buckets": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    bucket_details: List[Dict[str, Any]] = []

    for i in range(n_bins):
        mask = bin_indices == i
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        p_pred = float(y_prob[mask].mean())
        p_actual = float(y_true[mask].mean())
        cal_err = abs(p_pred - p_actual)
        ece += n_bin * cal_err
        bucket_details.append(
            {
                "bin": f"[{bins[i]:.1f}, {bins[i + 1]:.1f})",
                "n": n_bin,
                "mean_pred": round(p_pred, 4),
                "mean_actual": round(p_actual, 4),
                "cal_error": round(cal_err, 4),
            }
        )

    ece = ece / n_total
    return {"ece": round(ece, 6), "n_bins": n_bins, "n": n_total, "buckets": bucket_details}


# ── Metric Helpers ───────────────────────────────────────────────────


def _split_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Any]:
    """Compute metrics for a subset of predictions."""
    _check_libs("sklearn")
    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "log_loss": None,
            "brier_score": None,
            "accuracy": None,
            "roc_auc": None,
            "avg_pred_prob": None,
            "actual_win_rate": None,
            "over_under": "n/a",
            "ece": None,
        }

    n_classes = len(np.unique(y_true))
    if n_classes > 1:
        ll = float(log_loss(y_true, y_prob))
        auc_val = float(roc_auc_score(y_true, y_prob))
    else:
        ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
        auc_val = None

    brier = float(brier_score_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_prob >= 0.5))
    avg_prob = float(np.mean(y_prob))
    actual_rate = float(np.mean(y_true))

    if avg_prob > actual_rate + OVER_UNDER_THRESHOLD:
        over_under = "overconfident"
    elif actual_rate > avg_prob + OVER_UNDER_THRESHOLD:
        over_under = "underconfident"
    else:
        over_under = "neutral"

    ece_result = compute_ece(y_true, y_prob)

    return {
        "n": n,
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "roc_auc": round(auc_val, 4) if auc_val is not None else None,
        "avg_pred_prob": round(avg_prob, 4),
        "actual_win_rate": round(actual_rate, 4),
        "over_under": over_under,
        "ece": ece_result["ece"],
    }


# ── Data Loading ─────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Load and merge incumbent predictions with feature table context."""
    pred_path = Path(PREDICTIONS_PATH)
    ft_path = Path(FEATURE_TABLE_PATH)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    if not ft_path.exists():
        raise FileNotFoundError(f"Feature table not found: {ft_path}")

    pred = pd.read_csv(str(pred_path))
    ft = pd.read_parquet(str(ft_path))

    # Merge on game_id, keep pred columns on conflict
    merge_cols = ["game_id", "season", "week", "gameday", "home_team", "away_team"]
    df = pred.merge(
        ft.drop(columns=[c for c in merge_cols if c in ft.columns and c != "game_id"]),
        on="game_id",
        how="left",
        suffixes=("", "_ft"),
    )

    # Compute short_week flag from rest columns
    if "home_rest" in df.columns and "away_rest" in df.columns:
        df["short_week_flag"] = (
            (df["home_rest"].fillna(7) < 7) | (df["away_rest"].fillna(7) < 7)
        ).astype(int)
    else:
        df["short_week_flag"] = 0

    # Compute Elo features for gap analysis (incumbent parameters)
    if _LIBS.get("ratings") and _LIBS.get("season_regression"):
        try:
            overrides = build_team_regression_overrides(
                ft, preseason_regression=0.1, qb_change_bonus=0.2
            )
            elo_df = compute_elo_features(
                ft,
                k_factor=36,
                home_advantage=40,
                preseason_regression=0.1,
                team_regression_overrides=overrides,
                decay_half_life=32,
            )
            elo_map = elo_df[["game_id", "home_elo_pre", "away_elo_pre"]].set_index("game_id")
            df["home_elo_pre"] = df["game_id"].map(elo_map["home_elo_pre"])
            df["away_elo_pre"] = df["game_id"].map(elo_map["away_elo_pre"])
        except Exception:
            df["home_elo_pre"] = np.nan
            df["away_elo_pre"] = np.nan
    else:
        df["home_elo_pre"] = np.nan
        df["away_elo_pre"] = np.nan

    return df


# ── Section 1: Incumbent Reproduction ────────────────────────────────


def _compute_incumbent_reproduction(df: pd.DataFrame) -> Dict[str, Any]:
    """Reproduce holdout metrics and verify against documented values."""
    _check_libs("sklearn")

    holdout = df[df["season"] == HOLDOUT_SEASON].copy()
    all_games = df[df["incumbent_home_win_prob"].notna()].copy()

    if holdout.empty:
        return {"error": f"No holdout ({HOLDOUT_SEASON}) data found"}

    y_true = holdout["home_win_actual"].values.astype(float)
    y_prob = holdout["incumbent_home_win_prob"].values.astype(float)

    holdout_metrics = _split_metrics(y_true, y_prob)

    # Overall metrics (all seasons)
    all_y_true = all_games["home_win_actual"].values.astype(float)
    all_y_prob = all_games["incumbent_home_win_prob"].values.astype(float)
    overall_metrics = _split_metrics(all_y_true, all_y_prob)

    # Per-season metrics
    season_metrics = {}
    for season in sorted(df["season"].unique()):
        mask = df["season"] == season
        sub = df[mask]
        sub_y_true = sub["home_win_actual"].values.astype(float)
        sub_y_prob = sub["incumbent_home_win_prob"].values.astype(float)
        season_metrics[int(season)] = _split_metrics(sub_y_true, sub_y_prob)

    ll_match = (
        abs(holdout_metrics["log_loss"] - INCUMBENT_HOLDOUT_LL) < 0.001
        if holdout_metrics["log_loss"] is not None
        else False
    )

    return {
        "model_version": (
            str(df["model_version"].iloc[0])
            if "model_version" in df.columns else "unknown"
        ),
        "holdout_season": HOLDOUT_SEASON,
        "holdout_n": holdout_metrics["n"],
        "holdout_log_loss": holdout_metrics["log_loss"],
        "holdout_brier": holdout_metrics["brier_score"],
        "holdout_accuracy": holdout_metrics["accuracy"],
        "holdout_auc": holdout_metrics["roc_auc"],
        "holdout_ece": holdout_metrics["ece"],
        "holdout_avg_prob": holdout_metrics["avg_pred_prob"],
        "holdout_actual_rate": holdout_metrics["actual_win_rate"],
        "documented_holdout_ll": INCUMBENT_HOLDOUT_LL,
        "holdout_ll_matches": ll_match,
        "overall": overall_metrics,
        "per_season": season_metrics,
    }


# ── Section 2: Failure-Mode Splits ───────────────────────────────────


def _split_label(label: str, mask: np.ndarray, total: int) -> str:
    """Generate a human-readable split label with count."""
    count = int(mask.sum())
    pct = 100.0 * count / total if total > 0 else 0.0
    return f"{label} (n={count}, {pct:.1f}%)"


def _compute_splits_from_masks(
    df: pd.DataFrame, split_specs: List[Tuple[str, np.ndarray]]
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics for each split mask."""
    total = len(df)
    results = {}
    for label, mask in split_specs:
        sub = df[mask]
        y_true = sub["home_win_actual"].values.astype(float)
        y_prob = sub["incumbent_home_win_prob"].values.astype(float)
        metrics = _split_metrics(y_true, y_prob)
        results[_split_label(label, mask, total)] = metrics
    return results


def _compute_all_splits(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all failure-mode split analyses."""
    total = len(df)
    results: Dict[str, Any] = {}

    # 1. QB changed vs not
    if "qb_change_flag" in df.columns:
        qb_mask = df["qb_change_flag"] == 1
        results["qb_changed"] = _compute_splits_from_masks(
            df,
            [
                ("QB changed", qb_mask.values),
                ("No QB change", (~qb_mask).values),
            ],
        )

    # 2. Roof type
    if "roof" in df.columns:
        dome_mask = df["roof"].str.lower().str.contains("dome", na=False)
        outdoor_mask = df["roof"].str.lower().str.contains("outdoor", na=False)
        retract_mask = df["roof"].str.lower().str.contains("open|retract", na=False)
        results["roof_type"] = _compute_splits_from_masks(
            df,
            [
                ("Dome", dome_mask.values),
                ("Outdoor", outdoor_mask.values),
                ("Retractable/open", retract_mask.values),
            ],
        )

    # 3. Rest advantage
    if "rest_diff" in df.columns:
        rest_pos = df["rest_diff"].fillna(0) > 0
        rest_neg = df["rest_diff"].fillna(0) < 0
        rest_zero = df["rest_diff"].fillna(0) == 0
        results["rest_advantage"] = _compute_splits_from_masks(
            df,
            [
                ("Home rest advantage", rest_pos.values),
                ("Away rest advantage", rest_neg.values),
                ("Equal rest", rest_zero.values),
            ],
        )

    # 4. Short week
    if "short_week_flag" in df.columns:
        sw_mask = df["short_week_flag"] == 1
        results["short_week"] = _compute_splits_from_masks(
            df,
            [
                ("Short week", sw_mask.values),
                ("Normal rest", (~sw_mask).values),
            ],
        )

    # 5. Large Elo gap
    if "home_elo_pre" in df.columns and "away_elo_pre" in df.columns:
        elo_gap = (df["home_elo_pre"] - df["away_elo_pre"]).abs()
        large_gap = elo_gap > 50
        close_gap = (~large_gap) & elo_gap.notna()
        no_elo = elo_gap.isna()
        results["elo_gap"] = _compute_splits_from_masks(
            df,
            [
                ("Large Elo gap (>50)", large_gap.values),
                ("Close Elo gap (<=50)", close_gap.values),
            ],
        )
        if no_elo.any():
            results["elo_gap"]["_note"] = (
                f"{int(no_elo.sum())} games without Elo data"
            )

    # 6. Home favorites / underdogs
    home_fav = df["incumbent_home_win_prob"] > 0.5
    home_dog = df["incumbent_home_win_prob"] <= 0.5
    results["home_status"] = _compute_splits_from_masks(
        df,
        [
            ("Home favorite", home_fav.values),
            ("Home underdog", home_dog.values),
        ],
    )

    # 7. Road favorites / underdogs (inverse of home)
    results["road_status"] = _compute_splits_from_masks(
        df,
        [
            ("Road favorite", home_dog.values),
            ("Road underdog", home_fav.values),
        ],
    )

    # 8. Missing weather data
    if "weather_missing_flag" in df.columns:
        w_miss = df["weather_missing_flag"] == 1
        results["missing_weather"] = _compute_splits_from_masks(
            df,
            [
                ("Missing weather data", w_miss.values),
                ("Weather data present", (~w_miss).values),
            ],
        )

    # 9. Missing QB metadata
    qb_cols = ["home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name"]
    avail_qb = [c for c in qb_cols if c in df.columns]
    if avail_qb:
        qb_miss = df[avail_qb].isna().any(axis=1)
        results["missing_qb_metadata"] = _compute_splits_from_masks(
            df,
            [
                ("Missing QB metadata", qb_miss.values),
                ("QB metadata present", (~qb_miss).values),
            ],
        )

    # 10. Season phase
    if "game_type" in df.columns:
        playoff = df["game_type"].isin(["WC", "DIV", "CON", "SB"])
        early = (df["week"] <= 4) & (~playoff)
        mid = (df["week"] >= 5) & (df["week"] <= 12) & (~playoff)
        late = (df["week"] >= 13) & (~playoff)
        results["season_phase"] = _compute_splits_from_masks(
            df,
            [
                ("Early season (weeks 1-4)", early.values),
                ("Mid season (weeks 5-12)", mid.values),
                ("Late season (weeks 13+)", late.values),
                ("Playoffs", playoff.values),
            ],
        )
    else:
        # Fallback: use week only
        early = df["week"] <= 4
        mid = (df["week"] >= 5) & (df["week"] <= 12)
        late = df["week"] >= 13
        results["season_phase"] = _compute_splits_from_masks(
            df,
            [
                ("Early season (weeks 1-4)", early.values),
                ("Mid season (weeks 5-12)", mid.values),
                ("Late season (weeks 13+)", late.values),
            ],
        )

    # 11. Neutral-site games
    if "is_neutral" in df.columns:
        neutral = df["is_neutral"] == 1
        results["neutral_site"] = _compute_splits_from_masks(
            df,
            [
                ("Neutral site", neutral.values),
                ("Non-neutral", (~neutral).values),
            ],
        )

    results["_total_games"] = total
    return results


# ── Section 3: Market Benchmark Comparison ───────────────────────────


def _moneyline_to_implied(moneyline: float) -> float:
    """Convert moneyline odds to implied probability (before vig)."""
    if pd.isna(moneyline) or moneyline == 0:
        return np.nan
    if moneyline > 0:
        return 100.0 / (moneyline + 100.0)
    else:
        return abs(moneyline) / (abs(moneyline) + 100.0)


def _no_vig_prob(home_ml: float, away_ml: float) -> Optional[float]:
    """Compute de-vigged home win probability from moneyline odds."""
    if pd.isna(home_ml) or pd.isna(away_ml):
        return None
    p_home = _moneyline_to_implied(home_ml)
    p_away = _moneyline_to_implied(away_ml)
    if p_home is None or p_away is None or (p_home + p_away) == 0:
        return None
    return p_home / (p_home + p_away)


def _compute_market_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare incumbent vs market no-vig probabilities."""
    _check_libs("sklearn")

    required = ["home_moneyline", "away_moneyline"]
    if not all(c in df.columns for c in required):
        return {"error": "Market columns not available"}

    df = df.copy()
    df["market_no_vig_home"] = df.apply(
        lambda r: _no_vig_prob(r["home_moneyline"], r["away_moneyline"]), axis=1
    )

    has_market = df["market_no_vig_home"].notna() & df["incumbent_home_win_prob"].notna()
    market_df = df[has_market].copy()

    if market_df.empty:
        return {"error": "No games with both market and model predictions"}

    y_true = market_df["home_win_actual"].values.astype(float)
    incumbent_prob = market_df["incumbent_home_win_prob"].values.astype(float)
    market_prob = market_df["market_no_vig_home"].values.astype(float)

    incumbent_market_metrics = _split_metrics(y_true, incumbent_prob)
    market_metrics = _split_metrics(y_true, market_prob)

    # Per-season market comparison
    season_comparison = {}
    for season in sorted(market_df["season"].unique()):
        sub = market_df[market_df["season"] == season]
        sub_y_true = sub["home_win_actual"].values.astype(float)
        sub_inc = sub["incumbent_home_win_prob"].values.astype(float)
        sub_mkt = sub["market_no_vig_home"].values.astype(float)
        season_comparison[int(season)] = {
            "n": len(sub),
            "incumbent_ll": _split_metrics(sub_y_true, sub_inc)["log_loss"],
            "market_ll": _split_metrics(sub_y_true, sub_mkt)["log_loss"],
        }

    # Week bucket comparison
    week_buckets = {}
    for label, wk_mask in [
        ("Early (1-4)", market_df["week"] <= 4),
        ("Mid (5-12)", (market_df["week"] >= 5) & (market_df["week"] <= 12)),
        ("Late (13+)", market_df["week"] >= 13),
    ]:
        sub = market_df[wk_mask]
        if len(sub) > 0:
            sub_y_true = sub["home_win_actual"].values.astype(float)
            sub_inc = sub["incumbent_home_win_prob"].values.astype(float)
            sub_mkt = sub["market_no_vig_home"].values.astype(float)
            week_buckets[label] = {
                "n": len(sub),
                "incumbent_ll": _split_metrics(sub_y_true, sub_inc)["log_loss"],
                "market_ll": _split_metrics(sub_y_true, sub_mkt)["log_loss"],
            }

    return {
        "games_with_market": market_metrics["n"],
        "incumbent_vs_market_gap": (
            round(incumbent_market_metrics["log_loss"] - market_metrics["log_loss"], 4)
            if incumbent_market_metrics["log_loss"] is not None
            and market_metrics["log_loss"] is not None
            else None
        ),
        "incumbent_on_market_subset": incumbent_market_metrics,
        "market_no_vig": market_metrics,
        "per_season": season_comparison,
        "per_week_bucket": week_buckets,
    }


# ── Section 4: High-Confidence Analysis ──────────────────────────────


def _compute_high_confidence(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze predictions at increasing confidence thresholds."""
    _check_libs("sklearn")

    results = {}
    y_prob_all = df["incumbent_home_win_prob"].values.astype(float)

    for threshold in CONFIDENCE_THRESHOLDS:
        high_conf = y_prob_all >= threshold
        low_conf = y_prob_all <= (1.0 - threshold)
        either = high_conf | low_conf

        # High-confidence home wins
        hw = df[high_conf]
        # High-confidence away wins (home prob <= threshold)
        aw = df[y_prob_all <= (1.0 - threshold)]
        # Either
        either_df = df[either]

        entry: Dict[str, Any] = {
            "threshold": threshold,
            "high_conf_all": _split_metrics(
                either_df["home_win_actual"].values.astype(float),
                either_df["incumbent_home_win_prob"].values.astype(float),
            ) if len(either_df) > 0 else {"n": 0},
            "high_conf_home_wins": _split_metrics(
                hw["home_win_actual"].values.astype(float),
                hw["incumbent_home_win_prob"].values.astype(float),
            ) if len(hw) > 0 else {"n": 0},
            "high_conf_away_wins": _split_metrics(
                aw["home_win_actual"].values.astype(float),
                aw["incumbent_home_win_prob"].values.astype(float),
            ) if len(aw) > 0 else {"n": 0},
        }

        # Actual win rate for home favorites
        if len(hw) > 0:
            entry["home_fav_actual_win_rate"] = round(
                float(hw["home_win_actual"].mean()), 4
            )
            entry["home_fav_avg_prob"] = round(
                float(hw["incumbent_home_win_prob"].mean()), 4
            )

        results[f"p_{threshold:.2f}"] = entry

    return results


# ── Section 5: Reproducibility ───────────────────────────────────────


def _compute_reproducibility() -> Dict[str, Any]:
    """Verify reproducibility by double-loading predictions."""
    df1 = pd.read_csv(PREDICTIONS_PATH)
    df2 = pd.read_csv(PREDICTIONS_PATH)

    # Sort both by game_id for deterministic comparison
    df1 = df1.sort_values("game_id").reset_index(drop=True)
    df2 = df2.sort_values("game_id").reset_index(drop=True)

    if len(df1) != len(df2):
        return {
            "deterministic": False,
            "note": "Row count mismatch between loads",
            "n1": len(df1),
            "n2": len(df2),
        }

    prob_col = "incumbent_home_win_prob"
    if prob_col not in df1.columns or prob_col not in df2.columns:
        return {
            "deterministic": False,
            "note": f"Column '{prob_col}' not found",
            "n": len(df1),
        }

    probs_match = np.allclose(
        df1[prob_col].values.astype(float),
        df2[prob_col].values.astype(float),
        equal_nan=True,
    )

    result: Dict[str, Any] = {
        "deterministic": bool(probs_match),
        "n_games": len(df1),
        "n_columns": len(df1.columns),
        "note": (
            "Predictions are from a static CSV file — "
            "deterministic by construction."
        ),
    }

    if not probs_match:
        diff_count = int(
            (df1[prob_col].values != df2[prob_col].values).sum()
        )
        result["mismatch_count"] = diff_count
        result[
            "note"
        ] = f"MISMATCH: {diff_count} games differ between loads"

    return result


# ── Report Writer ────────────────────────────────────────────────────


def _fmt_metric_row(name: str, metrics: Dict[str, Any]) -> str:
    """Format a single metric row for markdown table."""
    if metrics.get("n", 0) == 0:
        return f"| {name} | 0 | — | — | — | — | — | — |"
    return (
        f"| {name} | {metrics['n']} "
        f"| {metrics['log_loss']:.4f} "
        f"| {metrics['brier_score']:.4f} "
        f"| {metrics['accuracy']:.4f} "
        f"| {metrics['avg_pred_prob']:.4f} "
        f"| {metrics['actual_win_rate']:.4f} "
        f"| {metrics['over_under']} |"
    )


def _split_table(title: str, splits: Dict[str, Any]) -> str:
    """Build a markdown table for a split analysis."""
    hdr = "| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |"
    sep = "|------|---|---|---|---|---|---|---|"
    out_lines = [f"\n### {title}\n", hdr, sep]
    for label in sorted(splits.keys()):
        if label.startswith("_"):
            continue
        metrics = splits[label]
        if isinstance(metrics, dict) and "n" in metrics:
            if metrics["n"] == 0:
                out_lines.append(f"| {label} | 0 | — | — | — | — | — | — |")
            else:
                ece_str = (
                    f"{metrics.get('ece', 0):.4f}"
                    if metrics.get('ece') is not None else "—"
                )
                out_lines.append(
                    f"| {label} | {metrics['n']} "
                    f"| {metrics['log_loss']:.4f} "
                    f"| {metrics['brier_score']:.4f} "
                    f"| {metrics['accuracy']:.4f} "
                    f"| {metrics['avg_pred_prob']:.4f} "
                    f"| {metrics['actual_win_rate']:.4f} "
                    f"| {ece_str} |"
                )
    note = splits.get("_note")
    if note:
        out_lines.append(f"\n*Note: {note}*")
    return "\n".join(out_lines)


def _build_report(sections: Dict[str, Any], output_path: str) -> str:
    """Assemble the full trust report markdown."""
    lines = [
        "# Model Trust Diagnostics Report",
        "",
        f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*",
        "*Incumbent: "
        f"{sections.get('incumbent_reproduction', {}).get('model_version', 'unknown')}*",
        "",
        "---",
        "",
        "## 1. Incumbent Reproduction",
        "",
    ]

    inc = sections.get("incumbent_reproduction", {})
    if "error" in inc:
        lines.append(f"**Error:** {inc['error']}")
    else:
        ll_match = "PASS" if inc.get("holdout_ll_matches") else "FAIL"
        ll_icon = "✅" if inc.get("holdout_ll_matches") else "❌"
        lines.extend(
            [
                "| Metric | Value | Documented | Match |",
                "|---|---|---|---|",
                "| Holdout LL | "
                f"{inc.get('holdout_log_loss', 'N/A'):.4f} | "
                f"{INCUMBENT_HOLDOUT_LL:.4f} | {ll_icon} {ll_match} |",
                "| Holdout Brier | "
                f"{inc.get('holdout_brier', 'N/A'):.4f} | — | — |",
                "| Holdout Accuracy | "
                f"{inc.get('holdout_accuracy', 'N/A'):.4f} | — | — |",
                "| Holdout AUC | "
                f"{inc.get('holdout_auc', 'N/A')} | — | — |",
                "| Holdout ECE | "
                f"{inc.get('holdout_ece', 'N/A')} | — | — |",
                "| Holdout N | "
                f"{inc.get('holdout_n', 0)} | — | — |",
                "",
                f"**All games ({inc.get('overall', {}).get('n', 0)}):** "
                f"LL={inc.get('overall', {}).get('log_loss', 'N/A')}, "
                f"Brier={inc.get('overall', {}).get('brier_score', 'N/A')}, "
                f"Acc={inc.get('overall', {}).get('accuracy', 'N/A')}, "
                f"AUC={inc.get('overall', {}).get('roc_auc', 'N/A')}",
                "",
                "**Season breakdown:**",
                "| Season | N | Log Loss | Brier | Acc | AUC |",
                "|---|---|---|---|---|---|",
            ]
        )
        for season, sm in sorted(inc.get("per_season", {}).items()):
            lines.append(
                f"| {season} | {sm.get('n', 0)} "
                f"| {sm.get('log_loss', 'N/A')} "
                f"| {sm.get('brier_score', 'N/A')} "
                f"| {sm.get('accuracy', 'N/A')} "
                f"| {sm.get('roc_auc', 'N/A')} |"
            )

    # Section 2: Failure-Mode Splits
    lines.extend(["", "---", "", "## 2. Failure-Mode Splits", ""])
    splits = sections.get("failure_mode_splits", {})
    lines.append(f"*Total games analyzed: {splits.get('_total_games', 0)}*")

    for key in [
        "qb_changed",
        "roof_type",
        "rest_advantage",
        "short_week",
        "elo_gap",
        "home_status",
        "road_status",
        "missing_weather",
        "missing_qb_metadata",
        "season_phase",
        "neutral_site",
    ]:
        if key in splits:
            lines.append(_split_table(key.replace("_", " ").title(), splits[key]))

    # Section 3: Market Comparison
    lines.extend(["", "---", "", "## 3. Market Benchmark Comparison", ""])
    mkt = sections.get("market_comparison", {})
    if "error" in mkt:
        lines.append(f"*{mkt['error']}*")
    else:
        lines.extend(
            [
                "| Model | N | Log Loss | Brier | Acc | AUC |",
                "|---|---|---|---|---|---|",
                "| Incumbent | "
                f"{mkt.get('incumbent_on_market_subset', {}).get('n', 0)} "
                f"| {mkt.get('incumbent_on_market_subset', {}).get('log_loss', 'N/A')} "
                f"| {mkt.get('incumbent_on_market_subset', {}).get('brier_score', 'N/A')} "
                f"| {mkt.get('incumbent_on_market_subset', {}).get('accuracy', 'N/A')} "
                f"| {mkt.get('incumbent_on_market_subset', {}).get('roc_auc', 'N/A')} |",
                "| Market (no-vig) | "
                f"{mkt.get('market_no_vig', {}).get('n', 0)} "
                f"| {mkt.get('market_no_vig', {}).get('log_loss', 'N/A')} "
                f"| {mkt.get('market_no_vig', {}).get('brier_score', 'N/A')} "
                f"| {mkt.get('market_no_vig', {}).get('accuracy', 'N/A')} "
                f"| {mkt.get('market_no_vig', {}).get('roc_auc', 'N/A')} |",
                "",
                "**Incumbent vs Market gap (log loss):** "
                f"{mkt.get('incumbent_vs_market_gap', 'N/A')}",
                "",
                "**Per-season comparison:**",
                "| Season | N | Incumbent LL | Market LL |",
                "|---|---|---|---|",
            ]
        )
        for season, sc in sorted(mkt.get("per_season", {}).items()):
            lines.append(
                f"| {season} | {sc.get('n', 0)} "
                f"| {sc.get('incumbent_ll', 'N/A')} "
                f"| {sc.get('market_ll', 'N/A')} |"
            )

        lines.extend(
            [
                "",
                "**Per week-bucket comparison:**",
                "| Bucket | N | Incumbent LL | Market LL |",
                "|---|---|---|---|",
            ]
        )
        for bucket, bc in sorted(mkt.get("per_week_bucket", {}).items()):
            lines.append(
                f"| {bucket} | {bc.get('n', 0)} "
                f"| {bc.get('incumbent_ll', 'N/A')} "
                f"| {bc.get('market_ll', 'N/A')} |"
            )

    # Section 4: High-Confidence Analysis
    lines.extend(["", "---", "", "## 4. High-Confidence Analysis", ""])
    hc = sections.get("high_confidence", {})
    lines.append(
        "| Threshold | N (eith) | LL (eith) | Acc (eith) "
        "| N (home) | Home win% | Home avg prob |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(hc.keys()):
        entry = hc[key]
        thr = entry.get("threshold", 0)
        either = entry.get("high_conf_all", {})
        hw = entry.get("high_conf_home_wins", {})
        lines.append(
            f"| p ≥ {thr:.2f} | {either.get('n', 0)} "
            f"| {either.get('log_loss', '—')} "
            f"| {either.get('accuracy', '—')} "
            f"| {hw.get('n', 0)} "
            f"| {entry.get('home_fav_actual_win_rate', '—')} "
            f"| {entry.get('home_fav_avg_prob', '—')} |"
        )

    # Section 5: Reproducibility
    lines.extend(["", "---", "", "## 5. Reproducibility", ""])
    rep = sections.get("reproducibility", {})
    status = "✅ Deterministic" if rep.get("deterministic") else "❌ Non-deterministic"
    lines.extend(
        [
            f"**Status:** {status}",
            f"**Games compared:** {rep.get('n_games', 'N/A')}",
            f"**Columns:** {rep.get('n_columns', 'N/A')}",
            f"**Note:** {rep.get('note', '')}",
        ]
    )

    # Section 6: Model Trust Thresholds
    lines.extend(["", "---", "", "## 6. Model Trust Thresholds", ""])
    inc_data = sections.get("incumbent_reproduction", {})
    ece_val = inc_data.get("holdout_ece", 0.0)
    ece_flag = ece_val is not None and ece_val >= 0.10
    lines.append(
        "| Check | Value | Threshold | Status |"
    )
    lines.append(
        "|---|---|---|---|"
    )
    ece_symbol = "⚠️" if ece_flag else "✅"
    ece_status = "CAUTION: ECE ≥ 0.10" if ece_flag else "PASS"
    lines.append(
        f"| ECE (holdout) | {ece_val} | < 0.10 | {ece_symbol} {ece_status} |"
    )

    # High-confidence calibration: p > 0.90
    hc = sections.get("high_confidence", {})
    hc_90 = hc.get("p_0.90", {})
    hc_all = hc_90.get("high_conf_all", {})
    hc_n = hc_all.get("n", 0)
    hc_acc = hc_all.get("accuracy", 0.0)
    calib_90_ok = hc_n == 0 or hc_acc >= 0.80
    calib_symbol = "⚠️" if not calib_90_ok else "✅"
    calib_status = "CAUTION: acc < 0.80" if not calib_90_ok else "PASS"
    lines.append(
        f"| High-confidence acc (p≥0.90) | {hc_acc} | ≥ 0.80 | {calib_symbol} {calib_status} |"
    )
    lines.append(
        f"| High-confidence games (p≥0.90) | {hc_n} | — | — |"
    )

    # Market gap
    mkt = sections.get("market_comparison", {})
    gap_val = mkt.get("incumbent_vs_market_gap")
    if gap_val is not None:
        gap_flag = gap_val > 0.05
        gap_symbol = "⚠️" if gap_flag else "✅"
        gap_status = "CAUTION: gap > 0.05" if gap_flag else "PASS"
    else:
        gap_flag = False
        gap_symbol = "—"
        gap_status = "N/A (no market data)"
    lines.append(
        f"| Market gap (LL) | {gap_val} | ≤ 0.05 | {gap_symbol} {gap_status} |"
    )

    lines.extend(
        [
            "",
            "---",
            "",
            "*Report generated by sportslab.evaluation.model_trust*",
            "*No network access required*",
            "",
        ]
    )

    report = "\n".join(lines)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    return report


# ── Main Entry Point ─────────────────────────────────────────────────


def run_model_trust(output_path: Optional[str] = None) -> str:
    """Run comprehensive model trust diagnostics.

    Args:
        output_path: Path for the report markdown (default: REPORT_PATH).

    Returns:
        Path to the generated report.
    """
    _check_libs("sklearn")

    output_path = output_path or REPORT_PATH
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    df = load_data()

    sections: Dict[str, Any] = {
        "incumbent_reproduction": _compute_incumbent_reproduction(df),
        "failure_mode_splits": _compute_all_splits(df),
        "market_comparison": _compute_market_comparison(df),
        "high_confidence": _compute_high_confidence(df),
        "reproducibility": _compute_reproducibility(),
    }

    _build_report(sections, output_path)
    return str(Path(output_path).resolve())
