# Release v3.0.0 — Frozen QB Overlay

*Release date: 2026-07-06*
*RALPH Loop completed: 9*

---

## Model Information

| Attribute | Value |
|-----------|-------|
| Model version | v3.0.0 |
| Release label | Frozen QB Overlay |
| Release date | 2026-07-06 |
| Feature set | `elo_prob`, `home_qb_changed`, `away_qb_changed`, `home_rolling_mov_3`, `away_rolling_mov_3` |
| Calibration | Platt (logistic regression on Elo prob + features) |
| Elo parameters | K=36, HFA=40, preseason_reg=0.1, decay_hl=32 |
| QB overlay parameters | changed OR starts<17 → cap=40, gamma=1.0 |
| Training seasons | 2021–2024 |
| Holdout season | 2025 |
| Holdout LL | 0.6200 |
| Validation LL | 0.6305 (rolling-origin 3-fold avg) |
| Holdout Brier | 0.2157 |
| Holdout AUC | 0.7098 |
| Holdout accuracy | 0.6630 |
| Holdout ECE | 0.0628 |

---

## Artifact Locations

| Artifact | Path |
|----------|------|
| Production freeze checklist | `docs/production_freeze.md` |
| Weekly runbook | `docs/weekly_runbook.md` |
| Live monitoring template | `docs/live_monitoring.md` |
| Research integrity audit | `docs/research_integrity_audit.md` |
| Incumbent predictions (all) | `reports/predictions/incumbent_predictions.csv` |
| Incumbent predictions (holdout) | `reports/predictions/incumbent_predictions_2025_holdout.csv` |
| Prediction snapshots dir | `reports/predictions/snapshots/` |
| Prediction history | `reports/predictions/prediction_history.csv` |
| Snapshot manifest | `reports/predictions/snapshot_manifest.json` |
| Model-trust report | `reports/experiments/model_trust.md` |
| Backtest report | `reports/backtests/2025_backtest_report.md` |
| Experiment report (combined features) | `reports/experiments/combined_features.md` |
| Experiment report (frozen QB overlay) | `reports/experiments/frozen_qb_overlay_foldsafe.md` |
| Experiment ledger | `reports/benchmarks/experiment_ledger.csv` |
| Leaderboard | `reports/benchmarks/leaderboard.csv` |
| Research backlog | `reports/benchmarks/research_backlog.md` |

---

## Live-Operations Commands

| Step | Command | Timing |
|------|---------|--------|
| Ingest | `sportslab ingest-nfl --seasons <Y>` | After new schedule |
| Build features | `make build-features` | After ingest |
| Preflight | `sportslab live-preflight --qb-input <path>` | Before predicting |
| Predict live | `sportslab predict-week --season <Y> --week <W> --mode live --qb-input <path>` | Thursday |
| Predict dry-run | `sportslab predict-week --season <Y> --week <W> --mode dry_run` | Any time |
| Grade week | `sportslab grade-week --season <Y> --week <W>` | Tuesday |
| Season report | `sportslab season-report --season <Y>` | End of season |
| Prediction audit | `sportslab prediction-audit --season <Y> --mode live` | After grading |
| Model trust | `sportslab model-trust` | Weekly |
| Data audit | `sportslab data-audit` | Before each cycle |

---

## Required Validation Commands

| Check | Command | Expected |
|-------|---------|----------|
| Full test suite | `python -m pytest tests/` | 989+ passed, 0 failures |
| Lint | `ruff check src/ tests/` | All checks passed |
| Data audit | `sportslab data-audit` | 0 issues |
| Artifact audit | `sportslab audit-artifacts` | All OK |
| Live preflight | `sportslab live-preflight` | All OK |
| Model trust | `sportslab model-trust` | Thresholds pass |
| Dry-run predict | `sportslab predict-week --season <Y> --week <W> --mode dry_run` | Succeeds |

---

## Known Limitations

1. **Early-season weakness**: Weeks 1-4 LL=0.6727 — Elo requires 3+ games to stabilize rolling MOV
2. **QB source oracle bias**: Backtest uses final actual starters; live prediction requires external QB input CSV
3. **Retractable/open roof**: Cal ECE=0.2141 but only 32 games (too small for any action)
4. **Market gap**: Market closes at 0.6090 vs incumbent 0.6200 — Elo is purely pregame
5. **Training sample**: ~1,000 games (2021-2024) limits model complexity
6. **Weather/missing data**: Missing weather LL=0.6497; dome imputation is safe but imperfect
7. **Tie handling**: Win-streak technically incorrect for ties (excluded from evaluation)

---

## Latest RALPH Loop

| Loop | Experiment | Result | Date |
|------|-----------|--------|------|
| 8 | Preseason Elo Prior | ❌ Rejected — all 6 variants worse on both val and holdout | 2026-07-06 |
| 9 | Production Freeze | ✅ Complete — freeze, monitoring, post-week review, research triggers | 2026-07-06 |

---

## Test & CI Status

| Metric | Value |
|--------|-------|
| Test count | 989 passed, 1 skipped |
| Lint | Clean (ruff) |
| Data audit | Pass |
| Artifact audit | Pass |
| Model trust thresholds | All pass |
| Live preflight | Pass |

---

## Git Tag

Proposed tag for this release:

```
v3.0.0
```

Tag this commit with `git tag -a v3.0.0 -m "v3.0.0 Frozen QB Overlay — production freeze"` after operator confirms release readiness.
