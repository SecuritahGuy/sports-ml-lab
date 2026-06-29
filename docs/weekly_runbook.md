# Weekly Prediction Runbook — v2

## Overview

Weekly workflow for generating, saving, and grading NFL predictions using the research incumbent model.

**Model:** Elo + qb_changed + rolling_mov_3 + Platt (v2.0.0)
**Validated on:** 2021-2024, holdout LL 0.6262 (2025)
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
| `model_version` | `v2.0.0` |
| `model_date` | Incumbent freeze date |
| `training_seasons` | `2021-2024` |
| `feature_set` | `qb_changed + rolling_mov_3` |
| `calibration_method` | Platt logistic |
| `model_val_ll` | 0.6334 |
| `model_holdout_ll` | 0.6262 |
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

### 6. Full Pipeline Reset

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

## Publishing Audit Reports

Live audit reports are automatically written to `docs/predictions/audit_{season}.md`
for GitHub Pages. Push the repository to trigger Pages rebuild:

```bash
git push origin main
```

Rehearsal audit reports are not published to Pages.

---

---

## Pre-2026 Launch Checklist

Before the 2026 season starts, run through this checklist to confirm
the pipeline is launch-ready:

- [ ] **Feature table built**: `make build-features` — confirms data up to 2025 season
- [ ] **Full test suite**: `make test` — 364+ tests passing
- [ ] **Lint clean**: `make lint` — no new errors
- [ ] **Rehearsal passes**: `make rehearsal-2025` — 21 weeks, LL matches 0.6262
- [ ] **Audit generates cleanly**: `sportslab prediction-audit --season 2025` — no nan metrics
- [ ] **Prediction index built**: `make prediction-index` — `docs/predictions/index.md` generated
- [ ] **GitHub Pages configured**: Settings → Pages → Deploy from `main` `/docs`
- [ ] **Pages renders**: Visit `https://<user>.github.io/sports-ml-lab/predictions/`
- [ ] **Runbook printed**: This doc is the reference for weekly operations
- [ ] **QB starter CSV template ready**: 3-column CSV with game_id, home_qb_id, away_qb_id
- [ ] **Incumbent frozen**: v2.0.0, holdout LL 0.6262 — no model changes planned
- [ ] **Preseason fire drill passes**: `make preseason-fire-drill` — full dry-run cycle
- [ ] **Data audit clean**: `make data-audit` — all scheduled seasons present
- [ ] **Ingest safety confirmed**: `sportslab ingest-nfl 2026` does not drop 2021-2025
- [ ] **Oracle QB blocked**: `sportslab predict-week --season 2026 --week 1 --mode live` raises error
- [ ] **Live-preflight passes**: `sportslab live-preflight --qb-input data/samples/sample_qb_input_2025_w1.csv` — all checks clear

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
| Thursday (before TNF) | `live-preflight`, `predict-week --mode live` |
| Tuesday (after MNF) | `grade-week --mode live` |
| End of season | `prediction-audit`, `prediction-index` |

*See `reports/benchmarks/feature_family_status.md` for the research governance doc.*
