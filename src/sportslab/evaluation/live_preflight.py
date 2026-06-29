"""Live preflight — pre-week checklist before live predictions."""

from pathlib import Path
from typing import List, Optional

from sportslab.evaluation.data_audit import (
    FEATURE_TABLE_PATH,
    run_data_audit,
)
from sportslab.features.qb_input import parse_qb_input_csv


def validate_qb_csv(qb_input_path: Optional[str]) -> List[str]:
    """Validate a QB input CSV for common issues.

    Returns a list of warning/error messages (empty = valid).
    """
    issues: List[str] = []
    if not qb_input_path:
        return issues

    path = Path(qb_input_path)
    if not path.exists():
        issues.append(f"QB input CSV not found: {qb_input_path}")
        return issues

    try:
        df = parse_qb_input_csv(qb_input_path)
        n = len(df)
        n_home_ok = df["home_qb_id"].notna().sum()
        n_away_ok = df["away_qb_id"].notna().sum()
        print(f"  QB input CSV: {path} ({n} games, {n_home_ok} home, {n_away_ok} away)")

        # Check game_ids match the feature table
        if Path(FEATURE_TABLE_PATH).exists():
            import pandas as pd
            ft = pd.read_parquet(FEATURE_TABLE_PATH)
            feature_ids = set(ft["game_id"].unique())
            csv_ids = set(df["game_id"].unique())
            missing = csv_ids - feature_ids
            if missing:
                issues.append(
                    f"QB CSV game_ids not found in feature table: {sorted(missing)[:5]}..."
                )
        else:
            issues.append("Cannot validate QB game_ids: no feature table")
    except ValueError as e:
        issues.append(f"QB input CSV error: {e}")
    except Exception as e:
        issues.append(f"QB input CSV unexpected error: {e}")

    return issues


def dry_run_smoke_test() -> List[str]:
    """Run a basic dry-run predict as a smoke test.

    Returns a list of issues (empty = success).
    """
    issues: List[str] = []
    try:
        from sportslab.evaluation.predict_future import predict_future

        result = predict_future(season=2026, week=1, mode="dry_run")
        if result:
            print(f"  Dry-run predict: {len(result)} output files created")
        else:
            issues.append("Dry-run predict produced no output")
    except FileNotFoundError as e:
        issues.append(f"Dry-run predict failed (data missing): {e}")
    except Exception as e:
        issues.append(f"Dry-run predict error: {e}")
    return issues


def run_live_preflight(
    qb_input_path: Optional[str] = None,
    seasons: Optional[List[int]] = None,
) -> List[str]:
    """Run the live preflight checklist.

    Checks:
    1. Data audit (structure, staleness, partial ingest)
    2. QB input CSV validity (if provided)
    3. Dry-run predict smoke test
    4. Summary report

    Args:
        qb_input_path: Path to live QB starter CSV (optional).
        seasons: Seasons to validate in data audit.

    Returns:
        List of all issues found (empty = all clear).
    """
    all_issues: List[str] = []

    print("\n" + "="*60)
    print("  LIVE PREFLIGHT CHECKLIST")
    print("="*60 + "\n")

    # 1. Data audit
    print("\n[1/3] Data Audit")
    print("-" * 40)
    audit_issues = run_data_audit(seasons=seasons)
    all_issues.extend(audit_issues)

    # 2. QB input validation
    print("\n[2/3] QB Input Validation")
    print("-" * 40)
    if qb_input_path:
        qb_issues = validate_qb_csv(qb_input_path)
        all_issues.extend(qb_issues)
    else:
        print("  No QB input CSV provided (skipped)")

    # 3. Dry-run smoke test
    print("\n[3/3] Dry-Run Predict Smoke Test")
    print("-" * 40)
    dry_issues = dry_run_smoke_test()
    all_issues.extend(dry_issues)

    # Summary
    print("\n" + "="*60)
    total = len(all_issues)
    if total == 0:
        print("  ✅ PREFLIGHT PASSED — all checks clear")
    else:
        print(f"  ❌ PREFLIGHT FAILED — {total} issue(s) found")
        for i, issue in enumerate(all_issues, 1):
            print(f"     {i}. {issue}")
    print("="*60 + "\n")

    return all_issues
