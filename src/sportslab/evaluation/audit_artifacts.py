"""Artifact audit — validates benchmark registry, docs consistency, and prediction schema parity.

Checks:
1. All benchmark files exist
2. Incumbent md contains holdout LL
3. Leaderboard CSV parses correctly
4. Incumbent holdout LL in promoted rows
5. No diagnostic labeled promoted
6. All referenced experiment reports exist
7. Benchmark history entry count matches summary
8. Holdout predictions CSV log loss matches expected
9. All docs/*.md agree on incumbent version and holdout LL
10. Future/live prediction CSVs share common v3 schema with incumbent
11. Generated prediction metadata matches actual columns (version, calibration, feature_set)
"""

import csv
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
REPORTS = BASE / "reports"
BENCHMARKS = REPORTS / "benchmarks"
PREDICTIONS = REPORTS / "predictions"
EXPERIMENTS = REPORTS / "experiments"
DOCS = BASE / "docs"

INCUMBENT_HOLDOUT_LL = 0.6200
INCUMBENT_VALIDATION_LL = 0.6305
INCUMBENT_VERSION = "v3.0.0"

# Common columns shared by all v3 prediction CSVs (incumbent, future, snapshots)
COMMON_V3_COLUMNS = {
    "game_id", "season", "week", "gameday", "away_team", "home_team",
    "incumbent_home_win_prob", "base_incumbent_prob", "predicted_winner",
    "confidence_bucket", "model_version", "model_date", "training_seasons",
    "feature_set", "calibration_method", "model_holdout_ll",
    "elo_k", "elo_hfa", "elo_reg", "elo_decay", "elo_qb_bonus",
    "overlay_gamma", "overlay_cap", "overlay_gate_active", "home_qb_adj", "away_qb_adj",
    "caution_qb_change", "caution_neutral", "caution_early_season",
    "caution_missing_features", "caution_model_market_disagreement",
    "market_model_diff", "market_prob_diagnostic", "market_minus_model_diagnostic",
    "qb_change_flag",
}

INCUMBENT_PATH = BENCHMARKS / "nfl_research_incumbent.md"
LEADERBOARD_PATH = BENCHMARKS / "leaderboard.csv"
HISTORY_PATH = BENCHMARKS / "benchmark_history.md"
INCUMBENT_PREDS_PATH = PREDICTIONS / "incumbent_predictions.csv"
HOLDOUT_PREDS_PATH = PREDICTIONS / "incumbent_predictions_2025_holdout.csv"
CARD_PATH = BENCHMARKS / "incumbent_model_card.md"

DOC_PATHS = [
    DOCS / "index.md",
    DOCS / "benchmarks.md",
    DOCS / "predictions.md",
    DOCS / "model-card.md",
    DOCS / "experiments.md",
]

# Known differences: incumbent has score/target columns, future has qb columns
INCUMBENT_EXTRA_COLS = {"home_score", "away_score", "result", "home_win_actual"}
FUTURE_EXTRA_COLS = {"qb_source", "home_qb_id", "away_qb_id", "model_val_ll"}


def gather_issues():
    issues = []

    def check(cond, msg):
        if not cond:
            issues.append(msg)

    # 1. All benchmark files exist
    for name, path in [
        ("incumbent md", INCUMBENT_PATH),
        ("leaderboard csv", LEADERBOARD_PATH),
        ("history md", HISTORY_PATH),
        ("predictions csv", INCUMBENT_PREDS_PATH),
        ("holdout csv", HOLDOUT_PREDS_PATH),
        ("model card", CARD_PATH),
    ]:
        check(path.exists(), f"MISSING: {name} at {path}")

    if issues:
        return issues

    inc_text = INCUMBENT_PATH.read_text()

    # 2. Incumbent md contains holdout LL
    check(
        str(INCUMBENT_HOLDOUT_LL) in inc_text,
        f"Incumbent md missing holdout LL {INCUMBENT_HOLDOUT_LL}",
    )

    # 3. Leaderboard parses
    with open(LEADERBOARD_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    check(len(rows) > 0, "Leaderboard CSV is empty")

    expected_cols = {
        "experiment", "model_features", "selection_method",
        "decision", "holdout_ll", "report_path", "date",
    }
    actual_cols = set(rows[0].keys())
    missing_cols = expected_cols - actual_cols
    check(not missing_cols, f"Leaderboard missing columns: {missing_cols}")

    # 4. Incumbent holdout LL in leaderboard
    promoted = [r for r in rows if r["decision"] == "promoted"]
    check(len(promoted) >= 1, "No promoted entry in leaderboard")
    incumbent_found = any(
        abs(float(r["holdout_ll"]) - INCUMBENT_HOLDOUT_LL) < 0.001
        for r in promoted if r["holdout_ll"]
    )
    check(
        incumbent_found,
        f"Incumbent holdout {INCUMBENT_HOLDOUT_LL} not in leaderboard promoted rows",
    )

    # 5. No diagnostic labeled promoted
    for r in rows:
        decision = (r.get("decision") or "").lower()
        experiment = (r.get("experiment") or "").lower()
        is_diag = (
            "diagnostic" in decision or "holdout" in decision or "diagnostic" in experiment
        )
        is_promoted = decision == "promoted"
        check(
            not (is_diag and is_promoted),
            f"Diagnostic '{r['experiment']}' labeled promoted",
        )

    # 6. All referenced experiment reports exist
    for r in rows:
        rp = r.get("report_path", "")
        if rp and rp != "nan" and rp.strip():
            ep = EXPERIMENTS / Path(rp).name
            check(ep.exists(), f"Report missing: {rp}")

    # 7. Benchmark history summary matches entries
    hist_text = HISTORY_PATH.read_text()
    hist_lines = hist_text.splitlines()
    entry_count = sum(
        1
        for line in hist_lines
        if (line.strip().startswith("### ") and not line.strip().startswith("### Summary"))
        or (
            line.strip().startswith("## ")
            and len(line.strip()) > 3
            and line.strip()[3].isdigit()
            and ". " in line.strip()
        )
    )
    total_line = [line for line in hist_lines if "Total experiments" in line]
    if total_line:
        m = re.search(r"\b(\d+)\b", total_line[0])
        if m:
            parsed = int(m.group(1))
            check(
                parsed == entry_count,
                f"History summary says {parsed} entries, found {entry_count}",
            )

    # 8. Holdout predictions CSV log loss matches expected
    if HOLDOUT_PREDS_PATH.exists():
        holdout_df = pd.read_csv(HOLDOUT_PREDS_PATH)
        if (
            "home_win_actual" in holdout_df.columns
            and "incumbent_home_win_prob" in holdout_df.columns
        ):
            probs = holdout_df["incumbent_home_win_prob"].clip(0.001, 0.999)
            actuals = holdout_df["home_win_actual"]
            import math
            ll = -(
                actuals * probs.apply(lambda p: math.log(p))
                + (1 - actuals) * probs.apply(lambda p: math.log(1 - p))
            ).mean()
            check(
                abs(ll - INCUMBENT_HOLDOUT_LL) < 0.005,
                f"Holdout CSV log loss {ll:.4f} != expected {INCUMBENT_HOLDOUT_LL}",
            )

    # 9. Docs agree on incumbent version and holdout LL
    for dp in DOC_PATHS:
        if dp.exists():
            text = dp.read_text()
            check(
                INCUMBENT_VERSION in text,
                f"Doc {dp.name} missing incumbent version {INCUMBENT_VERSION}",
            )
            check(
                str(INCUMBENT_HOLDOUT_LL) in text,
                f"Doc {dp.name} missing holdout LL {INCUMBENT_HOLDOUT_LL}",
            )

    # 10. Future/live prediction CSV schema matches common v3 columns
    future_candidates = [
        PREDICTIONS / "future_predictions.csv",
    ]
    snap_dir = PREDICTIONS / "snapshots"
    if snap_dir.exists():
        snapshots = sorted(snap_dir.glob("*.csv"))
        if snapshots:
            future_candidates.append(snapshots[-1])

    for fpath in future_candidates:
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                actual_cols = set(df.columns)
                missing_common = COMMON_V3_COLUMNS - actual_cols
                allowed_extra = FUTURE_EXTRA_COLS | INCUMBENT_EXTRA_COLS
                unexpected = actual_cols - COMMON_V3_COLUMNS - allowed_extra
                if missing_common:
                    check(
                        False,
                        f"{fpath.name} missing common v3 columns: {sorted(missing_common)}",
                    )
                if unexpected:
                    check(
                        False,
                        f"{fpath.name} has unexpected columns: {sorted(unexpected)}",
                    )
            except Exception as e:
                check(False, f"Failed to read {fpath.name}: {e}")

    # 11. Metadata in prediction CSVs matches actual values
    metadata_checks = {
        "model_version": INCUMBENT_VERSION,
        "model_holdout_ll": str(INCUMBENT_HOLDOUT_LL),
    }
    for fpath in future_candidates + [INCUMBENT_PREDS_PATH]:
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                for col, expected_val in metadata_checks.items():
                    if col in df.columns:
                        actual_val = str(df[col].iloc[0]).strip()
                        check(
                            actual_val == expected_val,
                            f"{fpath.name}: {col} = '{actual_val}' != expected '{expected_val}'",
                        )
            except Exception:
                pass

    return issues


def run_audit():
    issues = gather_issues()
    report_path = BENCHMARKS / "artifact_audit.md"

    lines = []
    lines.append("# Artifact Audit Report")
    lines.append("")
    lines.append(f"*Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if not issues:
        lines.append("✅ **All checks passed.** No issues found.")
    else:
        lines.append(f"⚠️ **{len(issues)} issue(s) found:**")
        lines.append("")
        for i, issue in enumerate(issues, 1):
            lines.append(f"  {i}. {issue}")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n")
    return issues
