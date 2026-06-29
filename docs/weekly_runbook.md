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

## Recovery

If a snapshot is lost or corrupted:

```bash
sportslab predict-week --season 2026 --week 1 --output reports/predictions/snapshots/week_2026_01_recovery.csv
```

Then re-grade using the recovery snapshot:

```bash
sportslab grade-week --season 2026 --week 1 --snapshot reports/predictions/snapshots/week_2026_01_recovery.csv
```

---

*See `reports/benchmarks/feature_family_status.md` for the research governance doc.*
