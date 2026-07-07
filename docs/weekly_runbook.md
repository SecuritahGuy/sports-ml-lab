# Weekly Prediction Runbook — v3

## Overview

Weekly workflow for generating, saving, and grading NFL predictions using the research incumbent model.

**Model:** Elo + qb_changed + rolling_mov_3 + Platt (v3.0.0)
**Validated on:** 2021-2024, holdout LL 0.6200 (2025)
**Mode safety:** Live mode blocks oracle QB. Dry-run accepts oracle. Rehearsal fully isolated.

---

## Snapshot Modes

| Mode | QB Source | Can Grade | Contamination Risk | Use Case |
|------|-----------|-----------|-------------------|----------|
| `live` (default) | `live_pregame` only | Yes | None | Production predictions |
| `dry_run` | `oracle` or `live_pregame` | No (filtered) | None | Test predictions before live week |
| `rehearsal` | `oracle` or `live_pregame` | Yes (isolated) | None (separate dir) | Historical replay |

**Live mode blocks oracle QB data.** If you run `predict-week --mode live` without `--qb-input`, it raises an error. Use `--mode dry_run` for oracle-QB test predictions.

---

## Workflow

### Thursday morning (before TNF): Predict the week

Live mode (requires `--qb-input`):

```bash
sportslab predict-week --season 2026 --week 1 --mode live --qb-input data/live/qb_2026_w1.csv
```

Dry-run mode (oracle QB allowed, for testing before QB starters available):

```bash
sportslab predict-week --season 2026 --week 1 --mode dry_run
```

This does:
1. Loads all historical data (2021-2025) from the feature table
2. Fits Elo chronologically
3. Computes QB change and rolling MOV features
4. Fits Platt calibration on historical games
5. Predicts the specified week's games
6. Saves timestamped snapshot to `reports/predictions/snapshots/`
7. Generates weekly report to `reports/predictions/`

### Tuesday morning (after MNF): Grade the week

```bash
sportslab grade-week --season 2026 --week 1 --mode live
```

This does:
1. Finds the latest non-superseded live-mode snapshot for that week
2. Verifies SHA-256 checksum against manifest (blocks modified files)
3. Merges actual results from the feature table
4. Computes log loss, Brier, accuracy, AUC
5. Marks snapshot status as "graded" in manifest
6. Appends to `reports/predictions/prediction_history.csv`

### End of season: Generate dashboard

```bash
make season-report SEASON=2026
# or:
sportslab season-report --season 2026
```

---

## File Locations

| Artifact | Path |
|----------|------|
| Week snapshot | `reports/predictions/snapshots/week_{season}_{week}_{mode}_{timestamp}.csv` |
| Week report | `reports/predictions/week_{season}_{week}_report.md` |
| Snapshot manifest | `reports/predictions/snapshot_manifest.json` |
| Prediction history | `reports/predictions/prediction_history.csv` |
| Season report | `reports/predictions/season_{season}_report.md` |
| Audit report | `reports/predictions/audit_{season}.md` + `docs/predictions/audit_{season}.md` |
| Prediction index | `docs/predictions/index.md` |
| Feature table | `data/features/nfl/feature_table.parquet` |
| Schedule data | `data/raw/nfl/schedules.parquet` |
| Schedule metadata | `data/raw/nfl/schedules_metadata.json` |

---

## Snapshot Schema

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `season`, `week`, `gameday` | Game time context |
| `home_team`, `away_team` | Teams |
| `incumbent_home_win_prob` | Model prediction (0-1) |
| `predicted_winner` | Team with prob ≥ 0.5 |
| `confidence_bucket` | Probability range label |
| `model_version` | `v3.0.0` |
| `model_date` | Incumbent freeze date |
| `training_seasons` | `2021-2024` |
| `feature_set` | `qb_changed + rolling_mov_3` |
| `calibration_method` | Platt logistic |
| `model_val_ll` | 0.6305 |
| `model_holdout_ll` | 0.6200 |
| `elo_k`, `elo_hfa`, `elo_reg`, `elo_decay`, `elo_qb_bonus` | Elo parameters |
| `qb_source` | `oracle` or `live_pregame` |
| `home_qb_id`, `away_qb_id` | QB identifiers |
| `caution_qb_change` | 1 if either QB changed |
| `caution_early_season` | 1 if week ≤ 4 |
| `data_cutoff` | Date of data used for fitting |

---

## Caution Conditions

| Flag | Meaning |
|------|---------|
| `caution_qb_change` | Either team has a different QB than prior game |
| `caution_early_season` | Week 1-4 (higher error observed) |

---

## Caveats

- **QB data is oracle by default.** Use `--qb-input` for live-safe pregame-announced starters.
- **The feature table must be rebuilt** after the season ends (or new data ingested) before grades will include actual results.
- **Market data is not used in predictions.** The model is football-only.
- **This is research output, not betting advice.**

---

## Data Cutoff

Each snapshot includes a `data_cutoff` field. This is the date of the feature table used for fitting. The model does not use any information after this date.

---

## Historical Rehearsal

Replay a completed season through the weekly pipeline to validate the
operational flow and produce audit reports without touching live artifacts:

```bash
make rehearsal-2025
# or:
sportslab rehearsal-season --season 2025
```

This does:
1. Iterates through each week of the season
2. Fits Elo on all available data before each week (no future leakage)
3. Generates immutable prediction snapshots
4. Grades each week using known actual results
5. Produces season report and prediction audit

All output is isolated to `reports/predictions/rehearsal/`:

| Artifact | Path |
|----------|------|
| Manifest | `reports/predictions/rehearsal/manifest.json` |
| History | `reports/predictions/rehearsal/prediction_history.csv` |
| Snapshots | `reports/predictions/rehearsal/snapshots/` |
| Season report | `reports/predictions/rehearsal/season_{season}_report.md` |
| Audit | `reports/predictions/rehearsal/audit_{season}.md` |

### Live vs Rehearsal

| Aspect | Live | Dry Run | Rehearsal |
|--------|------|---------|-----------|
| Mode value | `live` | `dry_run` | `rehearsal` |
| Path | `reports/predictions/` | `reports/predictions/` | `reports/predictions/rehearsal/` |
| QB source | `live_pregame` only | `oracle` allowed | `oracle` or live |
| Actuals | From feature table at grade time | Not graded | Pre-populated in snapshot |
| Oracle blocking | Yes (raises error) | No | No |
| Contamination risk | None | None (not gradable as live) | None (separate dir) |
| Audit label | "Prediction Audit" | N/A | "Historical Rehearsal" |
| GitHub Pages | `docs/predictions/` | Not published | Not published |

---

---

## Safety & Guardrails

### Snapshot Lifecycle

Each snapshot in the manifest has a `status` field:

| Status | Meaning |
|--------|---------|
| `initial` | First snapshot for a season/week/mode |
| `superseded` | Replaced by a newer snapshot for same season/week/mode |
| `graded` | Graded against actual results (final) |

When a new snapshot is created for the same season/week/mode, the old one is
automatically marked `superseded`. Only non-superseded snapshots are considered
for grading.

### Oracle QB Blocking (Live Mode)

In `live` mode, `predict-week` rejects oracle QB data:

```bash
# This raises ValueError:
sportslab predict-week --season 2026 --week 1 --mode live

# This works:
sportslab predict-week --season 2026 --week 1 --mode live --qb-input qb.csv
```

Use `--mode dry_run` for oracle-QB test predictions.

### Mode Filtering in Grading

`grade-week` only finds snapshots matching the specified mode. A dry_run
snapshot cannot be graded as live, and vice versa:

```bash
# Dry-run snapshots are invisible to live grading:
sportslab grade-week --season 2026 --week 1 --mode live
# → "No live snapshot found for 2026 week 1"
```

### Ingestion Safety

`ingest-nfl` appends new seasons by default — it never silently drops
historical data:

```bash
# Safe: adds 2026 to existing 2021-2025
sportslab ingest-nfl 2026

# Destructive: overwrites everything with just 2026
sportslab ingest-nfl --replace-all 2026
```

The default behavior merges new seasons with existing data and deduplicates
by `game_id`. Only use `--replace-all` when you intentionally want to rebuild
from scratch.

### Data Audit

Validate schedule and feature table health before each live week:

```bash
make data-audit
# or:
sportslab data-audit

# Check specific seasons:
sportslab data-audit --seasons 2021,2022,2023,2024,2025,2026
```

Checks performed:
- Schedule file exists and has rows
- All requested seasons present
- Required columns present (`game_id`, `season`, `week`, `scores`)
- No duplicate `game_id` values
- All seasons >= 2021
- Feature table exists with expected columns
- Incumbent feature columns present
- Market columns preserved but NOT used by incumbent
- Data integrity (completed games have scores, future games don't)

### Preseason Fire Drill

Full end-to-end operational test using dry-run mode:

```bash
make preseason-fire-drill
```

This runs: `build-features` → `data-audit` → `predict-week` (dry_run) →
`prediction-audit`. Confirms the pipeline is healthy before the first live week.

---

## Failure Recovery Workflow

### 1. Live Preflight Failure

If `sportslab live-preflight` fails, do NOT proceed to live predict-week.
Resolve each check in order:

| Check | Failure | Resolution |
|-------|---------|------------|
| Data audit | Stale feature table | Run `make build-features` to rebuild |
| Data audit | Partial ingest | Run `sportslab ingest-nfl <missing_season>` then `make build-features` |
| Data audit | Past-dated games missing scores | Re-run `sportslab ingest-nfl <season>` and rebuild features |
| QB input | CSV not found | Verify path, ensure CSV exists with columns `game_id,home_qb_id,away_qb_id` |
| QB input | Duplicate game_ids | Remove duplicate rows from CSV, each game_id must appear once |
| QB input | Missing columns | Add all required columns (`game_id,home_qb_id,away_qb_id`) |
| QB input | All-null QB IDs | Ensure every row has a valid QB identifier (not empty/NaN) |
| Dry-run predict | No output | Check feature table exists, check seasons are correct |

Run `sportslab live-preflight --qb-input qb.csv` again after resolving issues.

### 2. Lost or Corrupted Snapshot

If a live snapshot is lost or corrupted, re-running `predict-week` for the same
season/week creates a new entry. The old entry is automatically superseded:

```bash
sportslab predict-week --season 2026 --week 1 --mode live --qb-input qb.csv
```

Then re-grade using the new snapshot:

```bash
sportslab grade-week --season 2026 --week 1 --mode live
```

The checksum guardrail will pass because the new snapshot is freshly generated
and registered in the manifest.

### 3. Stale Data After Ingest

If you re-ingest a season (e.g., after final scores are posted):

```bash
# Safe: appends new data without overwriting existing
sportslab ingest-nfl 2025

# Or replace a specific season entirely:
sportslab ingest-nfl --replace-seasons 2025

# Rebuild feature table after any ingest change:
make build-features
```

The `_check_partial_ingest` function will verify that schedule and feature table
row counts match per season after rebuild.

### 4. Malformed QB Input

If `predict-week` or `live-preflight` rejects your QB input CSV:

- **"Duplicate game_id(s) found"**: Each game must appear at most once.
- **"All home_qb_id values are missing"**: All cells in home_qb_id column are null.
- **"Missing required columns"**: CSV needs `game_id`, `home_qb_id`, `away_qb_id`.
- **"QB input CSV is empty"**: File has header but no data rows.

Fix the CSV, then re-run `predict-week` or `live-preflight`.

### 5. Grading Failures

| Error | Cause | Fix |
|-------|-------|-----|
| "No graded games found" | Games not yet played | Wait until after MNF |
| "Checksum mismatch" | Snapshot modified after creation | Re-run `predict-week` |
| "No live snapshot found for ..." | No snapshot for this mode+season+week | Run `predict-week --mode live` first |
| "Oracle QB data not allowed" | Live mode without QB input | Add `--qb-input` or use `--mode dry_run` |
| "No actual results found" | Feature table needs rebuild | `make build-features` after games finish |

### 6. Force-Grading a Week (Data Recovery)

If grading fails due to checksum mismatch or missing manifest entries, use
`--force` to bypass guardrails:

```bash
# Force-grade a live-mode snapshot even if checksum mismatches
sportslab grade-week --season 2026 --week 1 --mode live --force
```

This skips:
- SHA-256 checksum verification (the snapshot was not modified, merely
  the original is unavailable and was regenerated)
- Manifest guardrail (the original entry may be missing)

**Only use `--force` when you have verified the snapshot is legitimate.**
Normal grading with guardrails is the intended workflow. Force mode is for
data recovery scenarios where the snapshot was regenerated from identical
data (e.g., machine rebuild, manifest corruption).

### 7. Publishing Failure

If `sportslab publish-predictions` or `make publish-predictions` fails:

| Error | Cause | Fix |
|-------|-------|-----|
| "Prediction index" not generated | No audit reports exist | Run `sportslab prediction-audit --season <YEAR>` first |
| "Docs directory" missing | `docs/predictions/` does not exist | Run `sportslab build-prediction-index` to create it |
| Dry-run reports "no files written" | Dry-run mode is informational | Re-run without `--dry-run` to actually write files |

First, always use dry-run to verify what will happen:

```bash
sportslab publish-predictions --dry-run
```

Then run without dry-run to publish:

```bash
sportslab publish-predictions
# git add docs/predictions/
# git commit -m "Update prediction artifacts"
# git push origin main
```

### 8. Artifact Audit Failure

If `sportslab audit-artifacts` reports issues:

| Check | Failure | Resolution |
|-------|---------|------------|
| Incumbent prediction files exist | Missing CSV | Re-run `sportslab predict-incumbent` |
| Holdout CSV exist | Missing holdout file | Re-run `sportslab predict-incumbent` |
| Summary report exists | Missing index | Re-run `sportslab build-prediction-index` |
| Docs directory exists | Missing predictions page | Run `sportslab build-prediction-index` |

### 9. Full Pipeline Reset

To rebuild from scratch (destructive — only when intentionally resetting):

```bash
# 1. Re-ingest all seasons (overwrite existing)
sportslab ingest-nfl --replace-all 2021 2022 2023 2024 2025 2026

# 2. Rebuild feature table
make build-features

# 3. Verify health
sportslab data-audit

# 4. Dry-run smoke test
sportslab predict-week --season 2026 --week 1 --mode dry_run
```

---

## Model Trust Diagnostics

Run the comprehensive trust diagnostic at any time to verify model quality:

```bash
sportslab model-trust
# or:
make model-trust
```

This produces `reports/experiments/model_trust.md` with:
- Incumbent reproduction (holdout LL, Brier, accuracy, AUC, ECE)
- Failure-mode splits across 11+ dimensions (QB change, roof type, rest, short week, Elo gap, home/road status, missing weather, season phase, neutral site)
- Market benchmark comparison (incumbent vs no-vig market by season and week bucket)
- High-confidence analysis (5 thresholds from 0.70 to 0.90)
- Reproducibility verification

No network access required. All data reads from existing prediction artifacts.

## Reproducing Incumbent Metrics

To verify the reported incumbent metrics:

```bash
# 1. Calibration audit (reports ECE, MCE, Brier decomposition, subset calibration)
sportslab calibration-audit

# 2. Backtest 2025 (per-game predictions, calibration buckets, weekly breakdown)
sportslab backtest-2025

# 3. Model trust report (comprehensive: splits, market comparison, high-confidence)
sportslab model-trust
```

All three commands work with existing data — no network access required. The
holdout LL should be **0.6200** (v3.0.0 Frozen QB Overlay). If any command
reports a different value, the feature table or prediction artifacts may have
been regenerated with different parameters.

## Model Promotion Rules

1. A challenger must beat the incumbent's **holdout log loss** AND have **better average rolling validation log loss** (both with minimum improvement delta of 0.001) to be promoted.
2. If the challenger uses a logit-space overlay (e.g., frozen QB overlay), the non-gated subset must also not degrade (equality check).
3. Selection uses average rolling validation log loss only. The 2025 holdout is for final evaluation only, never for model selection.
4. Every feature must be **pregame-safe** and **explainable**.
5. Do not promote based on AUC, accuracy, or ROI alone.
6. Promoted models are documented in `reports/benchmarks/benchmark_history.md` and the registry at `reports/benchmarks/nfl_research_incumbent.md`.

## Why Market Data Is Diagnostic-Only

The feature table includes moneyline and spread data from nflreadpy schedules.
This data is **diagnostic-only** and must never be used as model features:

- **Timing mismatch**: Closing lines are near-kickoff. The model should produce
  pregame predictions usable before game day.
- **Circularity**: Using market odds as features means the model is learning
  to replicate the market rather than discovering independent signals.
- **Research integrity**: The goal is football-only predictions. Market data
  sets the improvement ceiling (0.6090 holdout LL) but is not a production
  candidate.

The feature table includes a `MARKET_COLUMNS` constant that lists all market
columns. The `LEAKAGE_COLUMNS` list in `build_features.py` does NOT include
market columns — they are tracked separately and excluded from all feature
pipelines by convention.

## Why Oracle QB Fields Are Blocked From Live Mode

The feature table's `home_qb_id` / `away_qb_id` reflect the **final actual
starter** (backtest research oracle), not the pregame-announced starter:

- **Live mode** (`--mode live`) rejects oracle QB data. You must provide
  `--qb-input CSV` with pregame-announced starters.
- **Dry-run mode** (`--mode dry_run`) allows oracle QB for test predictions.
- **Rehearsal mode** uses oracle QB by default (historical replay only).

This guardrail prevents accidentally using post-hoc knowledge in a live
prediction context. The `predict-future` command defaults to `live` mode
for the same reason.

## Data / Artifact Policy

### What Is Committed (Source of Truth)

| Artifact | Tracked | Rationale |
|----------|---------|-----------|
| `data/raw/nfl/schedules.parquet` | Yes | Raw ingest — irreplaceable source data |
| `data/features/nfl/feature_table.parquet` | Yes | Built features — expensive to rebuild; base for all experiments |
| `reports/experiments/*.md` | Yes | Experiment reports — canonical documentation |
| `reports/benchmarks/*.{md,csv}` | Yes | Benchmark registry — governance artifacts |
| `reports/predictions/incumbent_predictions*.csv` | Yes | Canonical prediction artifacts — test reproducibility |

### What Is NOT Committed (Generated at Runtime)

| Artifact | Reason |
|----------|--------|
| Any new `*.parquet` not already tracked | Build artifact; gitignore prevents accidental commit |
| `reports/predictions/snapshots/` | Hundreds of weekly snapshots |
| `reports/predictions/rehearsal/` | Historical rehearsal runs |
| `mlruns/` | MLflow tracking data |

### Generated Report Drift Policy

Experiment reports (`reports/experiments/*.md`) are **source-of-truth documents**.
They are committed once after review and never regenerated by CI. If the code
evolves such that a report becomes inaccurate:

1. A test should catch the drift (e.g., holdout LL mismatch).
2. A human must update the report text to match.
3. CI runs tests but does NOT regenerate reports.

This prevents silent report drift: if the code changes but the report doesn't,
a test will fail, forcing human reconciliation.

### CI Pipeline

The CI workflow (`.github/workflows/ci.yml`) runs on push/PR to `main`:
1. `ruff check` — lint verification
2. `pytest` — 500+ tests including incumbent metric verification
3. Feature table integrity check
4. Incumbent holdout LL verification (±0.01 tolerance)

CI does NOT:
- Regenerate experiment reports
- Run compute-intensive experiments (grid searches, Optuna)
- Require network access or API keys
- Modify any tracked files

## Publishing Audit Reports

Live audit reports are automatically written to `docs/predictions/audit_{season}.md`
for GitHub Pages. Push the repository to trigger Pages rebuild:

```bash
git push origin main
```

Rehearsal audit reports are not published to Pages.

### Prediction Index

After audits are generated, rebuild the prediction index to link all
published audits on the predictions page:

```bash
make prediction-index
```

This generates `docs/predictions/index.md` with links to all available
season audit reports. Run after every new audit.

### Publishing All Prediction Artifacts

```bash
make publish-predictions
```

This runs `prediction-index` and prints a reminder to push. Equivalent to:
```bash
sportslab build-prediction-index
git push origin main  # manual reminder
```

---

---

## Pre-2026 Launch Checklist

Before the 2026 season starts, run through this checklist to confirm
the pipeline is launch-ready:

- [ ] **Feature table built**: `make build-features` — confirms data up to 2025 season
- [ ] **Full test suite**: `python -m pytest tests/` — 989+ tests passing
- [ ] **Lint clean**: `make lint` — no new errors
- [ ] **Rehearsal passes**: `make rehearsal-2025` — 21 weeks, LL matches 0.6200
- [ ] **Audit generates cleanly**: `sportslab prediction-audit --season 2025` — no nan metrics
- [ ] **Prediction index built**: `make prediction-index` — `docs/predictions/index.md` generated
- [ ] **GitHub Pages configured**: Settings → Pages → Deploy from `main` `/docs`
- [ ] **Pages renders**: Visit `https://<user>.github.io/sports-ml-lab/predictions/`
- [ ] **Runbook printed**: This doc is the reference for weekly operations
- [ ] **QB starter CSV template ready**: 3-column CSV with game_id, home_qb_id, away_qb_id
- [ ] **Incumbent frozen**: v3.0.0, holdout LL 0.6200 — no model changes planned
- [ ] **Preseason fire drill passes**: `make preseason-fire-drill` — full dry-run cycle
- [ ] **Data audit clean**: `make data-audit` — all scheduled seasons present
- [ ] **Ingest safety confirmed**: `sportslab ingest-nfl 2026` does not drop 2021-2025
- [ ] **Oracle QB blocked**: `sportslab predict-week --season 2026 --week 1 --mode live` raises error
- [ ] **Live-preflight passes**: `sportslab live-preflight --qb-input data/samples/sample_qb_input_2025_w1.csv` — all checks clear
- [ ] **Grade-week --force confirmed**: `sportslab grade-week --help` shows `--force` flag
- [ ] **Publish-predictions --dry-run passes**: `sportslab publish-predictions --dry-run` — reports no files written
- [ ] **Audit-artifacts passes**: `sportslab audit-artifacts` — all artifact checks clean

### First Live Week (2026 Week 1)

```bash
# Thursday before TNF
sportslab live-preflight --qb-input data/live/qb_2026_w1.csv   # Full preflight
sportslab predict-week --season 2026 --week 1 --mode live --qb-input data/live/qb_2026_w1.csv

# Tuesday after MNF
sportslab grade-week --season 2026 --week 1 --mode live

# Optional: run audit
sportslab prediction-audit --season 2026
make prediction-index
```

### Weekly Cadence

| Day | Action |
|-----|--------|
| Thursday (before TNF) | `data-audit`, `live-preflight`, `predict-week --mode live` |
| Tuesday (after MNF) | `grade-week --mode live`, `model-trust`, weekly monitoring report, post-week review |
| End of season | `season-report`, `prediction-audit`, `prediction-index`, full backtest |

*See `docs/live_monitoring.md` for drift thresholds and monitoring templates.*

---

## Post-Week Review Workflow

Run after grading each week (Tuesday after MNF).

### Steps

1. **Grade completed games**
   ```bash
   sportslab grade-week --season <Y> --week <W> --mode live
   ```

2. **Regenerate model trust report**
   ```bash
   sportslab model-trust
   ```

3. **Run prediction audit**
   ```bash
   sportslab prediction-audit --season <Y> --mode live
   ```

4. **Fill monitoring report**
   Copy template from `docs/live_monitoring.md` to `reports/monitoring/weekly_<Y>_w<W>.md`.
   Fill fields from:
   - `grade-week` output for core metrics
   - `model-trust` report for ECE, threshold checks, subgroup splits
   - `prediction-audit` report for calibration buckets, confidence buckets

5. **Check drift thresholds**
   Compare against thresholds in `docs/live_monitoring.md` (Section 2).
   If any threshold is breached, note it in the operator notes section.
   Follow the escalation rules:
   - Single breach → note only
   - 2+ consecutive weeks breaching same threshold → schedule review
   - 4+ consecutive weeks breaching any threshold → escalate to model research

6. **Record operator notes**
   Document:
   - Any data issues encountered (missing scores, ingest failures)
   - Any operator errors (wrong QB input, wrong mode)
   - Any schedule/QB anomalies (last-minute changes, rescheduled games)
   - Any unexpected behavior from the model
   - Any games where prediction was clearly wrong and why (if identifiable)

7. **Classify issues**
   Determine whether any observed issue is:
   - **Data issue**: Missing/malformed input data, stale feature table, incomplete ingest
   - **Operator issue**: Wrong command, wrong mode, wrong QB input, wrong season/week
   - **Schedule/QB issue**: Last-minute starter change, rescheduled game, neutral-site change
   - **Model weakness**: Known weakness manifesting (early season, QB change, roof type)
   - **Expected variance**: Normal statistical fluctuation within backtest range

8. **Decide next action**
   Based on classification:
   - **Data issue**: Fix data pipeline, re-ingest, rebuild features, re-predict
   - **Operator issue**: Fix workflow, update runbook, retrain operator
   - **Schedule/QB issue**: Note in operator log; no model change
   - **Model weakness**: Note in operator log. Do NOT change model from one week.
     If same weakness repeats across 4+ weeks, follow research trigger policy below.
   - **Expected variance**: No action. Continue monitoring.

9. **Update season report**
   ```bash
   sportslab season-report --season <Y>
   ```

---

## Future Research Trigger Policy

Model research (new challenger experiments) must remain frozen unless **one or more** of the following triggers is met.

### Triggers

| # | Trigger | Evidence Required | Action |
|---|---------|------------------|--------|
| 1 | **Repeated live underperformance** | 4+ consecutive weeks where weekly LL > 0.65 across 16+ games/week | Run comparison backtest; design challenger if underperformance is consistent |
| 2 | **Known failure mode repeats across multiple weeks** | Same subgroup (e.g., QB change, early season) shows LL > 0.70 in 3+ separate weeks | Design targeted challenger for that subgroup |
| 3 | **New reliable pregame data source** | Source is documented, pregame-safe, available for all 2021+ games, and passed leakage audit | Run feature experiment following canonical promotion policy |
| 4 | **Governance allows expanded seasons** | Explicit override of pre-2021 data ban | Run expanded-Elo validation on larger dataset |
| 5 | **Operational needs change** | Live prediction format, frequency, or scope fundamentally changes | Update pipeline, then consider model changes |
| 6 | **Model-trust thresholds are breached persistently** | Any threshold (ECE ≥ 0.10, high-confidence acc < 0.80, market gap > 0.05) breached for 4+ consecutive weeks | Run full diagnostic; consider calibration or model update |

### Non-Triggers

The following are **not** triggers for model research:

- One bad week (expected variance)
- Single-game extreme miss (happens in holdout too)
- Market outperforming model (always does — known gap)
- A new idea with no data or hypothesis (must have clear expected improvement)
- Pressure to improve during the season (freeze is intentional)
- Another researcher's model or approach that cannot be tested with our data

### Process

1. Trigger condition observed → operator notes it in monitoring report
2. If condition persists for full trigger duration → schedule research review
3. Review produces a challenger hypothesis with specific target and success criteria
4. Challenger tested via isolated experiment (rolling-origin 3-fold, Platt per fold)
5. If challenger beats incumbent on BOTH val and holdout with Δ ≥ 0.001 → promote
6. If challenger fails → document in experiment ledger, close the direction
7. After promotion or closure → resume freeze until next trigger

### Do Not Resume Model Research from One Bad Week

A single week with high log loss does not indicate model failure. The 2025 holdout contains individual weeks with LL > 0.70. A meaningful degradation signal requires sustained underperformance across multiple weeks (4+ consecutive weeks or repeated same-subgroup failure). Weekly variance is expected and does not justify breaking the freeze.
