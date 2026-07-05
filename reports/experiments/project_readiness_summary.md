# Project Readiness Summary

*Generated: 2026-07-02 by cleanup pass (calibration audit)*

---

## Current Incumbent

| Metric | Value |
|--------|-------|
| Model | v3.0.0 Frozen QB Overlay |
| Feature set | Elo + qb_changed + rolling_mov_3 + frozen QB overlay (logit-space) |
| Calibration | Platt (logistic on Elo prob + features) |
| **Val LL** | **0.6305** |
| **Holdout LL** | **0.6200** |
| Brier | 0.2157 |
| AUC | 0.7098 |
| Accuracy | 0.6630 |

---

## What Is Ready

| Area | Status | Notes |
|------|--------|-------|
| **Research incumbent** | ✅ Complete | v3.0.0 Frozen QB Overlay champion frozen; reproducibility verified |
| **Live weekly prediction** | ✅ Ready | `sportslab predict-week --season X --week Y --mode live --weekly-qb` with weekly tracker (88% accuracy). Manual `--qb-input` for 100% accuracy |
| **Live oracle blocking** | ✅ Verified | `weekly_pipeline.py:256` raises ValueError in live mode without QB input. Confirmed by `test_live_mode_rejects_no_qb` and `test_live_requires_qb_input` |
| **Dry_run oracle access** | ✅ Verified | `weekly_pipeline.py` and `predict_future.py` allow `mode=dry_run` without QB input. Confirmed by `test_dry_run_allows_no_qb` and `test_dry_run_allows_oracle` |
| **Future prediction** | ✅ Verified | 264 future games for 2026 season predicted. No NaN, no ties leaked, all unplayed games |
| **Benchmark registry** | ✅ Complete | `reports/benchmarks/leaderboard.csv`, `benchmark_history.md`, `nfl_research_incumbent.md` with promotion rules |
| **Prediction artifacts** | ✅ Complete | Full predictions CSV, 2025 holdout CSV, prediction cards, snapshot manifest |
| **Model card** | ✅ Complete | `reports/benchmarks/incumbent_model_card.md` with integrity section |
| **Dashboard (GitHub Pages)** | ✅ Built | `make build-dashboard` generates docs/; enable in repo Settings → Pages |
| **Research integrity audit** | ✅ Complete | `docs/research_integrity_audit.md` with leakage, timing, tie-handling verification |
| **Weekly pipeline** | ✅ Complete | `predict-week` → `grade-week` → `season-report` → `prediction-audit` with snapshot manifest, checksums, grading guardrails |
| **Quality gates** | ✅ Passing | `ruff check` clean, 725 tests passing, `audit-artifacts` passing, `data-audit` passing |
| **Calibration audit** | ✅ Complete | ECE=0.0628, MCE=0.1343, QB-change ECE=0.2097 (N=55), fold ECE stable (range 0.0142) |

---

## What Is Not Ready / Still Caveated

| Area | Caveat | Mitigation |
|------|--------|------------|
| **Market-beating claims** | ❌ Market (no-vig 0.6090) beats incumbent (0.6200) on holdout. Incumbent is football-only champion, not market-beating | Report states market is diagnostic benchmark; never promote on ROI alone |
| **Oracle QB in live mode** | ❌ Oracle QB is blocked in live mode. Incumbent's backtest accuracy (0.6200) is validated on oracle data, which live mode cannot use | Weekly tracker at 88% closes most of the gap; manual CSV for 100% |
| **Snapshot/report lifecycle** | ⚠️ Generated snapshots, rehearsal data, and manifests accumulate timestamps. No automated cleanup | Manual management; present state has 4 uncommitted timestamp-only changes |
| **AutoGluon/tree models** | ❌ All tree-based models rejected (expressive models, HGB, RF, AutoGluon) | Complexity does not help at ~1,000 training rows |
| **Feature hunting** | ❌ Closed as of 2026-06-29. 30+ families tested, 27 rejected, 4 watchlist. Do Not Retest rule in effect | Two additional seasons (260+ games) or new data source required to reopen |
| **Cross-validation standard** | ⚠️ Rolling-origin 3-fold (2021→2022, 2021-2022→2023, 2021-2023→2024) is used; not k-fold CV | Chronological CV prevents leakage by design |
| **New season 2026** | ⚠️ Feature table has 2026 schedule loaded but no completed games yet | 272 upcoming games ready for live prediction |

---

## Recent Rejected Models

| Experiment | Date | Decision | Best Val LL | Best Holdout LL |
|-----------|------|----------|-------------|-----------------|
| Regularized logistic meta-model | 2026-07-02 | ❌ Rejected | 0.6327 | 0.6244 |
| Expanded Elo spine (840 combos) | Prior | ❌ Rejected | 0.6299 | 0.6302 |
| QB × roster interaction overlay | Prior | ❌ Rejected | 0.6305 | 0.6200 |
| Roster availability overlay | Prior | ❌ Rejected | 0.6341 | 0.6259 |
| Coach+QB season regression | Prior | ❌ Rejected | 0.6309 | 0.6290 |
| O/D Elo ratings | Prior | ❌ Diagnostic only | 0.6371 | 0.6271 |
| AutoGluon AutoML | Prior | ❌ Rejected | 0.6956 | 0.6404 |

All rejected models and their full reports are listed in `reports/benchmarks/benchmark_history.md`.

---

## Recommended Next Work

| Priority | Task | Trigger |
|----------|------|---------|
| **High** | Operational rehearsal: run the full `predict-week` → `grade-week` → `season-report` → `prediction-audit` workflow on a completed 2025 week in `dry_run` mode | Before 2026 Week 1 |
| **Medium** | Opening-line diagnostic: compare incumbent vs opening moneyline (not closing) if explicitly requested | Diagnostic only; not a model promotion path |
| ~~**Medium**~~ | Calibration audit | ✅ Complete |
| **Low** | New feature hunting: requires governance trigger (2 new seasons / new data source / repeatable failure mode) | Not before 2027 unless trigger met |
| **Low** | Live preflight: run `sportslab live-preflight` with a mock QB CSV to validate the full pipeline | Before 2026 Week 1 |

---

## Commands Run and Results

| Command | Status | Result |
|---------|--------|--------|
| `git status --short` | ✅ | 7 modified, 4 new files (calibration audit + timestamp churn) |
| `git diff --stat` | ✅ | 7 files changed, 156 insertions, 141 deletions |
| `ruff check src/ tests/` | ✅ | All checks passed |
| `python -m pytest tests/ -q` | ✅ | 725 passed |
| `sportslab audit-artifacts` | ✅ | All checks OK |
| `sportslab data-audit` | ✅ | 1696 rows, 2021-2026, 272 upcoming games, all checks passed |
| `predict_future(mode=dry_run)` | ✅ | 264 games, no NaN, no ties, all unplayed |
| `sportslab calibration-audit` | ✅ | ECE=0.0628, MCE=0.1343, report → `calibration_audit.md` |
| Report verification (regularized logistic) | ✅ | All 9 required sections present |
| Report verification (live QB readiness) | ✅ | All 8 required sections present |

---

*No commits or pushes made during this verification pass. Artifact timestamp changes remain unstaged.*
