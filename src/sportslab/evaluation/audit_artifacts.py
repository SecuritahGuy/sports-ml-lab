import csv
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
REPORTS = BASE / "reports"
BENCHMARKS = REPORTS / "benchmarks"
PREDICTIONS = REPORTS / "predictions"
EXPERIMENTS = REPORTS / "experiments"

INCUMBENT_HOLDOUT_LL = 0.6200
INCUMBENT_VALIDATION_LL = 0.6334

INCUMBENT_PATH = BENCHMARKS / "nfl_research_incumbent.md"
LEADERBOARD_PATH = BENCHMARKS / "leaderboard.csv"
HISTORY_PATH = BENCHMARKS / "benchmark_history.md"
INCUMBENT_PREDS_PATH = PREDICTIONS / "incumbent_predictions.csv"
HOLDOUT_PREDS_PATH = PREDICTIONS / "incumbent_predictions_2025_holdout.csv"
CARD_PATH = BENCHMARKS / "incumbent_model_card.md"


def gather_issues():
    issues = []

    def check(cond, msg):
        if not cond:
            issues.append(msg)

    # 1. All files exist
    for name, path in [
        ("incumbent md", INCUMBENT_PATH),
        ("leaderboard csv", LEADERBOARD_PATH),
        ("history md", HISTORY_PATH),
        ("predictions csv", INCUMBENT_PREDS_PATH),
        ("holdout csv", HOLDOUT_PREDS_PATH),
        ("model card", CARD_PATH),
    ]:
        check(path.exists(), f"MISSING: {name} at {path}")

    if not issues:
        # 2. Incumbent md contains holdout LL
        inc_text = INCUMBENT_PATH.read_text()
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
            "experiment",
            "model_features",
            "selection_method",
            "decision",
            "holdout_ll",
            "report_path",
            "date",
        }
        actual_cols = set(rows[0].keys())
        missing_cols = expected_cols - actual_cols
        check(not missing_cols, f"Leaderboard missing columns: {missing_cols}")

        # 4. Incumbent holdout LL in leaderboard
        promoted = [r for r in rows if r["decision"] == "promoted"]
        check(len(promoted) >= 1, "No promoted entry in leaderboard")
        incumbent_found = any(
            abs(float(r["holdout_ll"]) - INCUMBENT_HOLDOUT_LL) < 0.001
            for r in promoted
            if r["holdout_ll"]
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

        # 8. Holdout predictions CSV contains correct holdout LL
        if HOLDOUT_PREDS_PATH.exists():
            import pandas as pd

            holdout_df = pd.read_csv(HOLDOUT_PREDS_PATH)
            if (
                "home_win_actual" in holdout_df.columns
                and "incumbent_home_win_prob" in holdout_df.columns
            ):
                probs = holdout_df["incumbent_home_win_prob"].clip(0.001, 0.999)
                actuals = holdout_df["home_win_actual"]
                ll = -(
                    actuals * probs.apply(lambda p: __import__("math").log(p))
                    + (1 - actuals) * probs.apply(lambda p: __import__("math").log(1 - p))
                ).mean()
                check(
                    abs(ll - INCUMBENT_HOLDOUT_LL) < 0.005,
                    f"Holdout CSV log loss {ll:.4f} != expected {INCUMBENT_HOLDOUT_LL}",
                )

    return issues


def run_audit():
    issues = gather_issues()
    report_path = BENCHMARKS / "artifact_audit.md"

    lines = []
    lines.append("# Artifact Audit Report")
    lines.append("")
    lines.append("*Generated: automatic*")
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
