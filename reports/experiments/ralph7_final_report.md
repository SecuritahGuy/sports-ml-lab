# RALPH Loop 7: Rejected Challenger Analysis + Research Infrastructure

*Generated: 2026-07-06*

*Incumbent: v3.0.0 Frozen QB Overlay (holdout LL 0.6200)*

---

## Summary

RALPH Loop 7 was a research-infrastructure and analysis pass. No model experimentation was performed.

**Outcome**: Incumbent unchanged (v3.0.0 Frozen QB Overlay, holdout LL 0.6200). Feature set unchanged. Promotion policy unchanged. Live-workflow unchanged.

---

## What Was Done

### 1. Rejected Challenger Postmortems

All 5 RALPH Loop 6 challengers received structured postmortems:

| Challenger | Disposition | Key Finding |
|-----------|-------------|-------------|
| L1: prior_win_pct | **RETIRE** | Prior-season win% too noisy (NFL year-to-year correlation ~0.3); use Elo instead |
| L2: weather_missing | **RETIRE** | Collinear with existing roof type; retested and failed |
| L3: roof_enc | **RETIRE** | Target subgroup (n=32) too small; 0 games in 2025 holdout |
| L4: games_since_change | **MONITOR** | −0.0102 QB-change improvement (real signal) but net penalty; retry with binned encoding or more data |
| L5: isotonic | **RETIRE** | Overfits on <5000 training rows; Platt is correct |

**Diagnostic finding (L4)**: `games_since_change` improved the QB-change subgroup from LL 0.6674 → 0.6572 (−0.0102) but hurt early-season and non-QB-change games. The signal is real but cannot be net-positive at current sample size.

### 2. Experiment Ledger

Created a centralized experiment ledger in `reports/benchmarks/`:

- **`experiment_ledger.csv`** — Machine-readable (44 rows, 13 columns): val LL, holdout LL, AUC/Brier/Acc, decision, rejection reason, leakage risk, operational risk, report path, date
- **`experiment_ledger.md`** — Human-readable summary with timeline, by-decision grouping, and rejection pattern analysis

**Ledger statistics**:
| Outcome | Count |
|---------|-------|
| Promoted (current champion) | 1 |
| Superseded (former champions) | 5 |
| Rejected | 28 |
| Diagnostic | 10 |
| **Total** | **44** |

**Rejection pattern breakdown**:
- Both val and holdout worse: 12 experiments
- Val worse, holdout better (val rejects): 5
- Val better/neutral, holdout worse (overfit): 5
- No improvement over incumbent: 2
- Below baseline: 2

**Key insight**: 43% of rejected experiments degraded BOTH metrics. The incumbent is at a Pareto frontier.

### 3. Research Backlog

Created a ranked backlog in `reports/benchmarks/research_backlog.md` scoring each candidate lane on 5 axes:

| Rank | Lane | Score | Targets | Status |
|------|------|-------|---------|--------|
| 1 | Preseason Elo Prior | 22/30 | Early-season weakness | **Recommended for RALPH 8** |
| 2 | Calibration Shrinkage | 14/30 | High-confidence | Already rejected (#10) |
| 3 | Market Diagnostics | 11/30 | — | Diagnostic only |
| 4 | Coach/QB Continuity | 10/30 | — | Saturated by overlay |
| 5 | Expanded Seasons | N/A | All weaknesses | Blocked by governance |

### 4. Next Experiment Design

**Top recommendation: Preseason Elo Prior (RALPH Loop 8)**

Hypothesis: The early-season weakness (weeks 1-4 LL=0.6727) exists because the Platt model sees `elo_prob` (noisy in early weeks) and `rolling_mov_3 = 0` for weeks 1-3. Adding the previous season's final Elo rating as an explicit feature gives a stable preseason baseline.

- No new data required — leverages existing Elo pipeline
- Direct improvement on L1 (prior_win_pct) — replaces noisy win% with higher-signal Elo
- Targeted at the largest remaining weakness (early season gap of 0.0721 to late season)
- Full design documented in research_backlog.md

### 5. Feature Chasing Guardrails

Added Research Governance section to AGENTS.md with:
- **Required Pre-Conditions** for running a challenger (clear hypothesis, target weakness, pregame data, leakage audit, validation plan, rejection criteria, operational cost)
- **Rejection Is a Result** — a rejected challenger produces documented negative evidence
- **Feature Chasing Guard** — isolated testing, leakage audit, pregame verification, clear promotion path
- **Closed Directions Table** — 12 closed directions with last-tested reference and why closed

### 6. Report Hygiene

Verification completed:
- AGENTS.md — updated with RALPH Loop 7 session summary, Research Governance section, experiment ledger reference
- Benchmark history — experiment ledger now serves as the canonical reference
- Model-trust report — up to date (no changes needed; all L6 challengers rejected)
- Experiment reports directory — ralph7_analysis.md and ralph7_final_report.md added

---

## Verification Commands

All commands pass:

| Command | Result |
|---------|--------|
| `ruff check src/ tests/` | All checks passed |
| `python -m pytest tests/` | 973 passed, 1 skipped (75s) |
| `sportslab data-audit` | Audit found 1 issue (feature table age, pre-existing) |
| `sportslab audit-artifacts` | ✅ Artifact audit passed |
| `sportslab model-trust` | Report generated (incumbent matches) |
| Promotion/leakage tests (76) | All passed |
| Target safety tests | All passed |

---

## Risks

1. **Feature table age**: 7 days old, at threshold. Not a problem for this research-only loop but will need rebuilding before RALPH Loop 8.
2. **Closed directions table in AGENTS.md**: Must be maintained as new experiments are run. If a direction is retested, the "Last Tested" column must be updated.
3. **Experiment ledger maintenance**: The CSV is manually maintained. A future agent must add rows for new experiments.

---

## Recommended RALPH Loop 8 Task

**Implement and test the Preseason Elo Prior feature** against the v3.0.0 Frozen QB Overlay incumbent.

Design details in `reports/benchmarks/research_backlog.md` (lane #1).

This targets the largest remaining weakness (early season, LL=0.6727) using existing data (prior-season Elo ratings), requires no external dependencies, and directly improves on the rejected L1 challenger (prior_win_pct) by using Elo instead of win%.
