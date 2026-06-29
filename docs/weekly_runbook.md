# Weekly Prediction Runbook

## Overview

Weekly workflow for generating, saving, and grading NFL predictions using the research incumbent model.

**Model:** Elo + qb_changed + rolling_mov_3 + Platt
**Validated on:** 2021-2024, holdout LL 0.6262 (2025)

---

## Workflow

### Thursday morning (before TNF): Predict the week

```bash
make predict-week SEASON=2026 WEEK=1
# or:
sportslab predict-week --season 2026 --week 1
```

This does:
1. Loads all historical data (2021-2025) from the feature table
2. Fits Elo chronologically
3. Computes QB change and rolling MOV features
4. Fits Platt calibration on historical games
5. Predicts the specified week's games
6. Saves timestamped snapshot to `reports/predictions/snapshots/week_2026_01_YYYYMMDD_HHMMSS.csv`
7. Generates weekly report to `reports/predictions/week_2026_01_report.md`

### With live-safe QB starters (recommended)

```bash
sportslab predict-week --season 2026 --week 1 --qb-input data/samples/sample_qb_input_2025_w1.csv
```

Provide a CSV with columns: `game_id,home_qb_id,away_qb_id`

### Tuesday morning (after MNF): Grade the week

```bash
make grade-week SEASON=2026 WEEK=1
# or:
sportslab grade-week --season 2026 --week 1
```

This does:
1. Finds the latest snapshot for that week
2. Merges actual results from the feature table
3. Computes log loss, Brier, accuracy, AUC
4. Appends to `reports/predictions/prediction_history.csv`

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
| Week snapshot | `reports/predictions/snapshots/week_{season}_{week}_{timestamp}.csv` |
| Week report | `reports/predictions/week_{season}_{week}_report.md` |
| Prediction history | `reports/predictions/prediction_history.csv` |
| Season report | `reports/predictions/season_{season}_report.md` |
| Feature table | `data/features/nfl/feature_table.parquet` |

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

| Aspect | Live | Rehearsal |
|--------|------|-----------|
| Path | `reports/predictions/` | `reports/predictions/rehearsal/` |
| QB source | Live or oracle | Oracle (or live with `--qb-input`) |
| Actuals | From feature table at grade time | Pre-populated in snapshot |
| Contamination risk | None | None — isolated top-level dir |
| Audit label | "Prediction Audit" | "Historical Rehearsal" |
| GitHub Pages | `docs/predictions/` | Not published |

---

## Recovery

If a live snapshot is lost or corrupted, re-running `predict-week` for the same
season/week replaces the old snapshot with a new one (manifest keeps latest):

```bash
sportslab predict-week --season 2026 --week 1
```

Then re-grade using the new snapshot:

```bash
sportslab grade-week --season 2026 --week 1
```

The checksum guardrail will pass because the new snapshot is freshly generated
and registered in the manifest.

### If grading fails

- **"No graded games found"**: Games may not have been played yet. Wait
  until after MNF.
- **"Checksum mismatch"**: The snapshot file was modified after creation.
  Re-run `predict-week` to create a fresh snapshot.
- **"No snapshot found"**: Run `predict-week` first.

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
- [ ] **Full test suite**: `make test` — 327+ tests passing
- [ ] **Lint clean**: `make lint` — no new errors
- [ ] **Rehearsal passes**: `make rehearsal-2025` — 21 weeks, LL matches 0.6262
- [ ] **Audit generates cleanly**: `sportslab prediction-audit --season 2025` — no nan metrics
- [ ] **Prediction index built**: `make prediction-index` — `docs/predictions/index.md` generated
- [ ] **GitHub Pages configured**: Settings → Pages → Deploy from `main` `/docs`
- [ ] **Pages renders**: Visit `https://<user>.github.io/sports-ml-lab/predictions/`
- [ ] **Runbook printed**: This doc is the reference for weekly operations
- [ ] **QB starter CSV template ready**: 3-column CSV with game_id, home_qb_id, away_qb_id
- [ ] **Incumbent frozen**: v2.0.0, holdout LL 0.6262 — no model changes planned

### First Live Week (2026 Week 1)

```bash
# Thursday before TNF
make build-features                  # Ensure latest data
sportslab predict-week --season 2026 --week 1 --qb-input data/live/qb_2026_w1.csv

# Tuesday after MNF
sportslab grade-week --season 2026 --week 1

# Optional: run audit
sportslab prediction-audit --season 2026
make prediction-index
```

### Weekly Cadence

| Day | Action |
|-----|--------|
| Thursday (before TNF) | `predict-week` |
| Tuesday (after MNF) | `grade-week` |
| End of season | `prediction-audit`, `prediction-index` |

*See `reports/benchmarks/feature_family_status.md` for the research governance doc.*
