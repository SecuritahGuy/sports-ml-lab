# RALPH Loop 5: Weekly Live-Ops Hardening

## Goal

Prove that a human operator can safely run a real NFL prediction week from
start to finish — weekly live-ops simulation, 2026 readiness, prediction
publishing, grading workflow, and failure recovery.

## Scope

No model challenger research. No changes to v3.0.0 incumbent (holdout LL
0.6200). Canonical promotion policy unchanged (must beat BOTH validation AND
holdout with Δ≥0.001).

## Changes Made

| File | Change |
|------|--------|
| `src/sportslab/evaluation/live_preflight.py` | Fixed `dry_run_smoke_test()` — no longer hardcodes 2026 week 1; auto-detects latest season with future games from feature table. Moved inline imports to top-level. |
| `src/sportslab/evaluation/weekly_pipeline.py` | Added `--force` parameter to `grade_week()` — bypasses checksum and manifest guardrails for legitimate data recovery. |
| `src/sportslab/evaluation/prediction_audit.py` | Created `publish_predictions()` function with `dry_run` parameter. Moved inline imports (`_load_actuals`, `_read_history`, `_file_checksum`, `accuracy_score`) to top-level. Fixed f-string lint issues. |
| `src/sportslab/cli.py` | Added `sportslab publish-predictions` CLI command with `--dry-run` flag. Added `--force` option to `grade-week` command. |
| `Makefile` | Updated `publish-predictions` target to call `sportslab publish-predictions`. Added `publish-predictions-dry-run` target. |
| `tests/test_weekly_operations.py` | **New** — 31 tests for Week 1 cold-start, QB input validation (missing columns, empty file, duplicates, all-null, whitespace stripping, missing file, valid v1 schema, apply preserves non-matching rows), season/mode validation, pre-2021 rejection, prediction publishing safety, grading safety (empty metrics, output keys), failure injection (pre-2021 CLI rejection, malformed QB CSV, duplicate game IDs). |
| `docs/weekly_runbook.md` | Added 3 new failure recovery sections: Force-Grading (Section 6), Publishing Failure (Section 7), Artifact Audit Failure (Section 8). Updated pre-2026 launch checklist with new commands. |

## Verification Results

| Check | Result |
|-------|--------|
| `pytest tests/` | **955 passed, 1 skipped** |
| `ruff check src/ tests/` (all .py files) | **Clean** |
| `sportslab data-audit` | **All checks passed** (1696 games, 2021-2026) |
| `sportslab audit-artifacts` | **All checks OK** |
| `sportslab model-trust` | **Report generated** (no errors) |
| `sportslab publish-predictions --dry-run` | **Clean** — no files written |
| `sportslab live-preflight --qb-input` | **✅ Preflight passed** — all checks clear |

## Key Decisions

1. **`publish-predictions --dry-run` as CLI command**, not just Makefile target.
   The Makefile target delegates to the CLI, ensuring consistent behavior.

2. **`grade-week --force` added** for legitimate data recovery (snapshot
   regenerated from identical data, manifest corruption, machine rebuild).
   Normal grading with guardrails remains the intended workflow — force mode
   is opt-in via explicit `--force` flag.

3. **Live preflight smoke test now auto-detects** the latest season with
   future games, rather than hardcoding 2026 or guessing `max_season + 1`.
   This is year-agnostic and will work correctly after the 2026 season ends.

4. **No changes to v3.0.0 incumbent** (holdout LL 0.6200). All work was
   operational hardening, not model research.

## Current State

| Metric | Value |
|--------|-------|
| Tests | 955 passed, 1 skipped |
| Lint | Clean (ruff) |
| Incumbent | v3.0.0 Frozen QB Overlay, holdout LL 0.6200 |
| Preflight | Passes (data audit + QB validation + dry-run predict) |
| Publishing | Dry-run verified, manual push required |

## Relevant Files

- `src/sportslab/evaluation/live_preflight.py` — dynamic dry-run smoke test + QB CSV validation
- `src/sportslab/evaluation/weekly_pipeline.py` — `grade_week()` with force parameter
- `src/sportslab/evaluation/prediction_audit.py` — `publish_predictions()` with dry-run
- `src/sportslab/cli.py` — `publish-predictions` and `grade-week --force` commands
- `docs/weekly_runbook.md` — failure recovery sections 6-8 added
- `tests/test_weekly_operations.py` — 31 live-ops tests

## Next Steps

1. Supply actual pregame QB starter data CSV for live-pregame comparison vs oracle
2. Any model must beat **Frozen QB Overlay (holdout LL 0.6200)** to become new incumbent
3. Enable GitHub Pages from repo settings (Settings → Pages → Deploy from `main` `/docs`)
