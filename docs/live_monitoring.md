# Live Monitoring Design — v3.0.0

*Template for weekly monitoring and drift detection.*

---

## 1. Weekly Monitoring Report Template

Generate one report per week after grading. Save to `reports/monitoring/weekly_<YEAR>_w<WEEK>.md`.

```markdown
# Weekly Monitoring Report — YYYY Week W

*Generated: YYYY-MM-DD HH:MM*
*Snapshot: <snapshot_filename>*

## Overview

| Field | Value |
|-------|-------|
| Season | YYYY |
| Week | W |
| Games predicted | N |
| Games graded | N |
| Ungraded games | N (reason) |
| Skipped games | N (reason: neutral/bye/non-eligible) |
| Stale-data warnings | N |
| QB input warnings | N |
| Model version | v3.0.0 |
| Snapshot checksum | <sha256> |
| Published file list | <paths> |
| Operator notes | <free text> |

## Core Metrics

| Metric | This Week | Rolling 4-Week | Model-Trust Threshold | Status |
|--------|-----------|----------------|----------------------|--------|
| Log loss | X.XXXX | X.XXXX | ≤0.65 (warning) | ✅/⚠️/❌ |
| Brier score | X.XXXX | X.XXXX | ≤0.24 (warning) | ✅/⚠️/❌ |
| Accuracy | X.XX | X.XX | ≥0.55 (warning) | ✅/⚠️/❌ |
| ECE | X.XXXX | X.XXXX | <0.10 | ✅/⚠️/❌ |
| Market gap (LL) | X.XXXX | X.XXXX | ≤0.05 | ✅/⚠️/❌ |

## Calibration Buckets

| Bucket | N | Observed Rate | Predicted Mean | Cal Error |
|--------|---|-------------|---------------|-----------|
| 50-55% | N | X.XX% | X.XX% | ±X.XX |
| 55-60% | N | X.XX% | X.XX% | ±X.XX |
| 60-65% | N | X.XX% | X.XX% | ±X.XX |
| 65-70% | N | X.XX% | X.XX% | ±X.XX |
| 70-80% | N | X.XX% | X.XX% | ±X.XX |
| 80%+ | N | X.XX% | X.XX% | ±X.XX |

## High-Confidence Predictions

| Threshold | N | Correct | Missed | Accuracy | Miss Rate |
|-----------|---|---------|--------|----------|-----------|
| p ≥ 0.70 | N | N | N | X.XX% | X.XX% |
| p ≥ 0.80 | N | N | N | X.XX% | X.XX% |
| p ≥ 0.90 | N | N | N | X.XX% | X.XX% |

## Subgroup Performance

| Subgroup | N | Log Loss | Δ vs Overall | Status |
|----------|---|----------|-------------|--------|
| QB-change games | N | X.XXXX | ±X.XXXX | ✅/⚠️/❌ |
| Home underdogs | N | X.XXXX | ±X.XXXX | ✅/⚠️/❌ |
| Missing weather | N | X.XXXX | ±X.XXXX | ✅/⚠️/❌ |
| Dome games | N | X.XXXX | ±X.XXXX | ✅/⚠️/❌ |
| Open/retractable roof | N | X.XXXX | ±X.XXXX | ✅/⚠️/❌ |
| Early season (Weeks 1-4) | N | X.XXXX | ±X.XXXX | ✅/⚠️/❌ |

## Model-vs-Market Disagreement

| Metric | Value | Note |
|--------|-------|------|
| Avg |model_prob − market_prob| | X.XXXX | Market data diagnostic-only |
| Games with diff > 0.15 | N | List if ≥3 games |
| Market gap widening? | Y/N | Compare to running avg |

## Operator Notes

<Free-text observations, anomalies, data issues, rollback reasons.>

## Drift Check Summary

| Check | Threshold | Actual | Status |
|-------|-----------|--------|--------|
| Weekly LL | ≤0.65 | X.XXXX | ✅/⚠️/❌ |
| Rolling 4-week LL | ≤0.64 | X.XXXX | ✅/⚠️/❌ |
| High-confidence miss rate | ≤20% | X.XX% | ✅/⚠️/❌ |
| ECE drift | <0.10 | X.XXXX | ✅/⚠️/❌ |
| Schema unchanged | Yes | Y/N | ✅/⚠️/❌ |
| No stale data | Fresh <7 days | X days | ✅/⚠️/❌ |
| Prediction count match | N_graded == N_predicted | Y/N | ✅/⚠️/❌ |
| Checksum match | manifest == file | Y/N | ✅/⚠️/❌ |
```

---

## 2. Drift Thresholds

### Warning Thresholds (not failure — trigger review)

| Metric | Warning Threshold | Rationale |
|--------|------------------|-----------|
| Weekly log loss | > 0.65 | Expected range 0.55–0.65 based on 2025 holdout (0.6200). Anything above 0.65 is worse than any single 2025 fold. |
| Rolling 4-week log loss | > 0.64 | Smoothed version of weekly threshold. 0.64 would be worst 4-week stretch in backtest. |
| High-confidence miss rate (p≥0.80) | > 20% | Backtest: 80+% confidence games have ~15% miss rate. Above 20% is drift. |
| ECE (any week) | ≥ 0.10 | Model-trust threshold. Breach means calibration is degrading. |
| ECE (rolling 4-week) | ≥ 0.08 | Smoothed calibration drift. Below single-week warning level. |
| Missing-weather rate | > 35% of games | Expected ~30% of games have missing weather data. Above 35% may indicate data source issue. |
| QB-change split degradation | > +0.02 vs same-week overall | If QB-change games are 0.02+ worse log loss than the week average, the QB input source may be wrong. |
| Schema changes | Any unexpected column addition/removal | Should never change mid-season. If it does, backfill check. |
| No games found | 0 graded games for a scheduled week | Likely ingest/scheduling issue. |
| Stale data | Feature table > 7 days old | Schedule or scores may have updated. |
| Prediction count mismatch | graded ≠ predicted | Grading skipped some snapshots or prediction file is missing rows. |
| Published file checksum mismatch | manifest ≠ file | File may be corrupt or manually modified. |
| Market gap widening sharply | > 0.05 weekly or > 0.03 rolling | If market is pulling away significantly, model may be deteriorating or conditions have shifted. |

### Threshold Policies

- **Single-week threshold breaches**: Note in operator notes. Do not change model.
- **2+ consecutive weeks breaching the same threshold**: Schedule a review. Investigate data source, feature pipeline, QB input quality.
- **4+ consecutive weeks breaching any threshold**: Escalate to model-trust diagnostics and backtest comparison. Consider whether to design a future challenger.
- **No automatic model changes**: All thresholds are warnings, not triggers. No model change without canonical promotion policy.

### Small Sample Handling

| Sample Size | Rule |
|-------------|------|
| 0-5 games | Log loss not reported (high variance). Report only accuracy and count. |
| 6-15 games | Log loss reported but flagged as low-sample. Do not compare against thresholds. |
| 16+ games | Full threshold comparison. |
| QB-change subgroup | Report regardless of size but warn if n < 10. |

---

## 3. Weekly Cadence

```
Thursday (before TNF)
├── sportslab data-audit
├── sportslab live-preflight --qb-input <path>
├── sportslab predict-week --season <Y> --week <W> --mode live --qb-input <path>
└── Save weekly report

Tuesday (after MNF)
├── sportslab grade-week --season <Y> --week <W>
├── sportslab prediction-audit --season <Y> --mode live
├── sportslab model-trust
├── Fill monitoring report template
└── Run post-week review

End of season
├── sportslab season-report --season <Y>
└── Run full backtest for analysis
```

---

## 4. Monitoring Report Automation

The monitoring report template is designed to be filled manually by the operator after grading. A future enhancement could automate most fields:

- `grade-week` output provides core metrics (log loss, Brier, accuracy, AUC)
- `prediction-audit` output provides calibration buckets, confidence buckets, worst predictions
- `model-trust` output provides ECE, high-confidence acc, subgroup splits
- Market data is diagnostic-only but can be incorporated manually if available

For now, the operator copies the template above and fills fields from the existing tool outputs. If drift thresholds are breached, the operator follows the post-week review workflow.
