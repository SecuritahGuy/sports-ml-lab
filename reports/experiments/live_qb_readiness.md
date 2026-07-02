# Live QB Readiness Report

*Prepared: 2026-07-02*

## Summary

**Verdict: READY for live weekly prediction with caveats.**

The v3.0.0 Frozen QB Overlay champion (holdout LL 0.6200) can be used for live
weekly prediction. Oracle QB data is blocked in live mode. Three non-oracle QB
sources are available. No network downloads are required after the feature table
is built (except `--auto-qb` which requires nflreadpy).

---

## Available QB Sourcing Modes

| Mode | CLI Flag | Accuracy | Network | Week 1 OK? | Week 2+ OK? |
|------|----------|----------|---------|------------|-------------|
| Manual CSV | `--qb-input` | 100% (if correct) | No | ✅ Yes | ✅ Yes |
| Depth chart snapshot | `--auto-qb` | ~67% | nflreadpy | ✅ Yes | ⚠️ Misses mid-season changes |
| Weekly tracker | `--weekly-qb` | ~88% | No | ⚠️ Falls back to depth chart | ✅ Yes |
| Oracle | `mode=dry_run` only | 100% | No | ✅ Blocked in live | ✅ Blocked in live |

### 1. Manual CSV (`--qb-input`)

- **Required in live mode** when neither `--auto-qb` nor `--weekly-qb` is set.
- **V1 format (3 columns):** `game_id, home_qb_id, away_qb_id`
- **V2 format (9 columns, recommended):**
  ```
  game_id, home_qb_id, away_qb_id, home_qb_name, away_qb_name,
  source, confidence, timestamp, notes
  ```
- Source must be one of: `injury_report`, `depth_chart`, `coach_announcement`,
  `roster_move`, `beat_writer`, `manual`
- Confidence must be one of: `confirmed`, `probable`, `questionable`, `estimated`
- Duplicate `game_id` values raise `ValueError`.
- Missing QB IDs are treated as `pd.NA` — the gate still fires if `qb_changed`
  or `home_qb_team_starts_pre < 17` from the feature table.
- **No network required.**

### 2. Depth Chart Snapshot (`--auto-qb`)

- Reads nflreadpy `load_depth_charts()` — a single preseason snapshot.
- **Accuracy: ~67%** (misses mid-season QB changes due to injury/benching).
- Requires `nflreadpy` (network access).
- Falls back to oracle QB from the schedule when depth chart data is missing.
- Recommended only for **Week 1** when no prior-week data exists.

### 3. Weekly Tracker (`--weekly-qb`)

- Uses prior-week actual starters from the feature table.
- **Accuracy: ~88%** (catches mid-season changes).
- **No network required** after feature table is built.
- For Week 1: falls back to depth chart snapshot (same ~67% accuracy).
- Requires feature table to have been built with completed games.
- **Recommended for Week 2+ predictions.**

### 4. Oracle (dry_run / rehearsal only)

- Uses nflreadpy final-schedule `home_qb_id`/`away_qb_id` (actual starters).
- **Blocked in live mode.** Both `predict_week` and `predict_future` raise
  `ValueError("Oracle QB data not allowed in live mode")` when mode is
  `"live"` and no `--qb-input` is provided.
- Accessible via `mode=dry_run` or `mode=rehearsal` for backtesting.

---

## Expected Failure Modes

| Failure Mode | Likelihood | Impact | Mitigation |
|-------------|-----------|--------|------------|
| **Wrong QB in manual CSV** | Medium | Probability flips on gated games | Use `confirmed` source only; cross-check with injury reports |
| **Weekly tracker misses a change** | Low (12% error rate) | Small prob delta (~0.01) | Acceptable — 88% accuracy catches most changes |
| **Depth chart ignores mid-season change** | High (33% error rate) | Wrong QB identity on gated games | Switch to `--weekly-qb` after Week 1 |
| **Missing QB ID** (NA in CSV) | Low | Gate still fires if `qb_changed` or starts < 17 | Elo learns the gap over time |
| **Feature table stale** | Low | Rolling MOV and Elo drift | Run `sportslab data-audit` and rebuild features before each week |
| **nflreadpy depth chart network failure** | Low | `--auto-qb`/`--weekly-qb` fallback to oracle blocked in live | Always have a manual `--qb-input` CSV as fallback |

---

## Historical Accuracy (auto_qb vs weekly_qb)

| QB Source | 2025 Accuracy | Source of Estimate |
|-----------|--------------|-------------------|
| Depth chart snapshot | 67.2% | `tests/test_qb_auto_source.py:9` |
| Weekly tracker | 87.7% | `tests/test_qb_auto_source.py:9` |
| Improvement | +20.5pp | Weekly tracker catches mid-season changes |

The weekly tracker improves on the depth chart by matching prior-week actual
starters. This catches every mid-season QB change that occurred in a prior
week — injuries, benchings, and returns.

---

## Can Live Mode Reproduce v3.0.0?

**No, live mode cannot reproduce v3.0.0 exactly** because the incumbent's
holdout evaluation uses oracle QB data (actual starters). Live mode must use
pregame-announced starters, which may differ.

**Expected gap:**
- With `--weekly-qb` (88% accuracy): very close to oracle performance.
  The 12% mismatch rate affects only gated games (~50% of games) and the
  probability deltas are small (mean abs delta ~0.01 per the audit module).
- With `--auto-qb` (67% accuracy): larger gap. 33% of games have wrong QB
  identity, which changes the gate and the overlay adjustment.
- With manual CSV (depends on user accuracy): can match oracle if data is
  correct.

**Simulation reference:**
Oracle-based 2025 week-by-week simulation produced overall LL 0.6284
(276 games), very close to the fitted-once incumbent 0.6262.

---

## Required CSV Schema for Manual QB Input

```
Required columns:
  game_id          — str, e.g. "2025_01_PHI_DAL"
  home_qb_id       — str (GSIS ID), NA if unknown
  away_qb_id       — str (GSIS ID), NA if unknown

Optional V2 columns:
  home_qb_name     — str, human-readable name
  away_qb_name     — str, human-readable name
  source           — one of: injury_report, depth_chart, coach_announcement,
                     roster_move, beat_writer, manual
  confidence       — one of: confirmed, probable, questionable, estimated
  timestamp        — ISO 8601 datetime
  notes            — str, free text

Constraints:
  - No duplicate game_id values
  - At least some non-NA QB IDs required (ValueError if all NA)
  - Whitespace is stripped from string columns
```

---

## Pre-Week Checklist

Run this before each week's prediction session:

```bash
# 1. Data audit — checks season coverage, duplicates, score consistency
sportslab data-audit

# 2. Build QB input CSV (from injury reports, beat writers, etc.)
#    Edit the CSV to match the V1 or V2 schema above.

# 3. Dry-run smoke test (uses oracle QB for validation)
sportslab predict-week --season 2026 --week 1 --mode dry_run

# 4. Dry-run QB audit (compare sources before going live)
sportslab weekly-qb-audit --season 2026 --week 1

# 5. Live prediction (blocks oracle, requires QB input)
sportslab predict-week --season 2026 --week 1 --mode live --qb-input qb_2026_w1.csv
```

The `sportslab live-preflight` command automates steps 1, 2 (optional), and 3
in a single pass:

```bash
sportslab live-preflight --qb-input qb_2026_w1.csv
```

Automated preflight checks:
1. **Data audit** — structure, staleness, partial ingest
2. **QB CSV validation** — column presence, no duplicates, no all-NA
3. **Dry-run smoke test** — full Elo fit + prediction without grading

---

## Recommendation for 2026

### Week 1

**Strategy:** `--weekly-qb` (falls back to depth chart for Week 1)

```bash
sportslab predict-week --season 2026 --week 1 --mode live --weekly-qb
```

The weekly tracker falls back to the depth chart snapshot when no prior-week
data exists — identical to `--auto-qb` for Week 1. Either flag produces the
same output for Week 1.

**Backup:** Always prepare a manual `--qb-input` CSV in case `--weekly-qb`'s
nflreadpy fallback fails.

### Week 2+

**Strategy:** `--weekly-qb` (prior-week actual starters)

```bash
sportslab predict-week --season 2026 --week 2 --mode live --weekly-qb
```

The weekly tracker catches any mid-season QB changes that occurred in Week 1
(88% accuracy). No network required after the feature table is built.

**Alternative for higher trust:** Manual CSV. Collect starter info from injury
reports, beat writers, and coach announcements. This gives 100% accuracy on the
games where data is confirmed, but requires manual effort.

### Pre-Week Repeatable Process

1. Run `sportslab data-audit` — confirm feature table is up to date.
2. Run `sportslab live-preflight --qb-input qb.csv` — validate data, QB CSV, and
   dry-run predictions in one command.
3. Run `sportslab predict-week --season 2026 --week X --mode live --weekly-qb`
   (or `--qb-input qb.csv` for manual mode).
4. After the week completes, run `sportslab grade-week` to log metrics.
5. Run `sportslab season-report --season 2026` to view cumulative dashboard.

---

## Gaps & Concrete Tasks

| Gap | Priority | Description | Acceptable? |
|-----|----------|-------------|-------------|
| Weekly tracker uses depth chart fallback in Week 1 | Low | Week 1 QB accuracy ~67% unavoidable without preseason roster data | ✅ Acceptable |
| No automated "stale feature table" detection in predict-week | Medium | predict-week doesn't check feature table freshness | ⚠️ Mitigated by data-audit pre-flight |
| No side-by-side weekly_qb vs oracle comparison in predict-week output | Low | Must run `weekly-qb-audit` separately | ✅ Acceptable for now |
| Manual CSV requires game_id knowledge | Low | User must know the game_id format | ✅ Documented above |
| nflreadpy dependency for auto_qb/weekly_qb fallback | Low | Can always fall back to manual CSV | ✅ Acceptable |

---

## Decision

**✅ READY for live weekly use** with the following configuration:

- **Week 1:** `sportslab predict-week --season 2026 --week 1 --mode live --weekly-qb`
- **Week 2+:** `sportslab predict-week --season 2026 --week X --mode live --weekly-qb`

The weekly tracker at 88% accuracy closes most of the gap between oracle and
live prediction. Manual CSV (`--qb-input`) is available for users who need
100% verified starter data.

*Report generated by `sportslab` readiness assessment. Model: v3.0.0 Frozen QB
Overlay (holdout LL 0.6200).*
