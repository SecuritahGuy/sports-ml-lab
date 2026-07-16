# Research Backlog

*Ranked future experiment lanes for the NFL prediction model.*

*Generated: 2026-07-06*

---

## Ranking Methodology

Each lane is scored on 5 axes (1–5, higher = better):

| Axis | Description |
|------|-------------|
| **Expected value** | How much log loss improvement is plausible given model-trust weaknesses |
| **Leakage safety** | Low risk = 5, high risk = 1 |
| **Implementation complexity** | Simple = 5, complex = 1 |
| **Operational fragility** | Stable = 5, fragile = 1 |
| **Testability** | Easy to validate = 5, hard to isolate = 1 |
| **Sample size** | Adequate = 5, undersized = 1 |

---

## Ranked Lanes

### 1. ~~🥇 Preseason Elo Prior~~ (RETIRED — tested in RALPH Loop 8)

**Status: ❌ REJECTED**

**RALPH Loop 8 result**: 6 variants tested (prior_elo_raw, reg10, reg50, diff, decay). Best variant (prior_elo_diff): val 0.6342 (Δ=+0.0001), holdout 0.6282 (Δ=+0.0023). **All variants worse on both val and holdout.** Early season improved (Weeks 1-4 Δ=−0.0036) but mid/late seasons penalized (Δ=+0.0040/+0.0039). The prior-season Elo signal is already absorbed by `elo_prob` — adding it explicitly adds noise once current-season data accumulates.

**Disposition**: RETIRE — no path to football-only promotion at current sample size.

---

### 2. 🥈 Calibration Shrinkage for High-Confidence (Targeted)

**Total score: 14/30**

| Axis | Score | Rationale |
|------|-------|-----------|
| Expected value | 2 | High-confidence subset (p≥0.90) has only 22 games; calibration ECE=0.0628 already passes threshold |
| Leakage safety | 5 | Can be done within existing Platt framework; no new features |
| Implementation complexity | 3 | Requires fold-safe Platt modification with confidence-based regularization |
| Operational fragility | 4 | Calibration change only; no feature pipeline impact |
| Testability | 2 | Only 22 high-confidence games in holdout — high variance |
| Sample size | 1 | 22 games is too few to train a separate calibration |

**Status**: REJECTED — tested in experiment #10 (confidence calibration). Temperature scaling, Platt variants, and isotonic all failed. Shrinkage toward prior overfit. Not worth retesting.

---

### 3. 🥉 Market Disagreement Diagnostics Enhancement

**Total score: 11/30**

| Axis | Score | Rationale |
|------|-------|-----------|
| Expected value | 1 | Already diagnostic; no path to football-only improvement |
| Leakage safety | 2 | Market is near-kickoff, not pregame |
| Implementation complexity | 4 | Simple diagnostic; already built for QB-change games |
| Operational fragility | 3 | Requires market data (external dependency) |
| Testability | 3 | Easy to compute but hard to act on |
| Sample size | 4 | Market data available for all 1388 games |

**Status**: DIAGNOSTIC ONLY — no path to football-only promotion. Already captured in model-trust and market-benchmark reports.

---

### 4. Coach/QB Continuity (Retry with Better Source)

**Total score: 10/30**

| Axis | Score | Rationale |
|------|-------|-----------|
| Expected value | 1 | Already tested and rejected (experiments #7, #18, #33-36) |
| Leakage safety | 4 | Pregame-available |
| Implementation complexity | 3 | Well-defined feature set |
| Operational fragility | 4 | No external dependencies |
| Testability | 2 | Repeated failures suggest saturated signal |
| Sample size | 0 | n/a |

**Status**: RETIRED — QB continuity signal is saturated by the frozen QB overlay. Coach continuity signal is weak (experiment #18: val 0.6309, holdout 0.6286). Not worth retesting.

---

### 5. ✅ Expanded Seasons (Pre-2021) — COMPLETED

**Status**: ✅ DEPLOYED. Governance override applied 2026-07-16. Min season changed 2021 → 2000 (5,593 additional games added). Now the baseline for all experiments. See branch `research/deep-dive`.

---

## Other Rejected Lanes (Not Worth Testing)

| Lane | Reason for Rejection |
|------|---------------------|
| Injury feature refinements | All 20 injury features rejected (experiment #21); QB injury flag rejected (#23); roster overlay rejected (#37, #38) |
| Weather data quality fix | Weather missing/dome is collinear; retested and rejected in L2 of RALPH 6 |
| Tree/ensemble models | Tested 4 times (expressive models, AutoGluon, random forest diagnostic); consistently overfit |
| Turnover differential (to_net_3) | Watchlist — revisit when more data accumulates |
| Team-specific HFA | Watchlist — too noisy from 1-3 seasons of per-team data |
| Rolling MOV variant | mov_1-10, EWMA, capped, log-signed, std — all worse than mov_3 |
| Separate O/D Elo | k_off/k_def selected using holdout; no clean path to football-only promotion |
| Any market-derived feature | Market benchmark is diagnostic only; not pregame-safe |

---

## Next Experiment (RALPH Loop 9)

**Status**: No remaining high-value experiment lanes that satisfy pregame-safe, football-only constraints.

All ranked lanes have been tested or require triggers (new data, new source, live failure, diagnostic request):

| Lane | Status | Trigger to Revisit |
|------|--------|-------------------|
| Preseason Elo Prior | RETIRED | New data (2+ seasons / 260+ games) |
| Calibration Shrinkage | REJECTED | Already tested, failed |
| Market Diagnostics | DIAGNOSTIC ONLY | Already documented |
| Coach/QB Continuity | RETIRED | Saturated by frozen overlay |
| Expanded Seasons | BLOCKED | Governance override |

**Recommended RALPH Loop 9**: Monitor live prediction logs and accumulated data. If 2+ seasons of new data accumulate or a repeatable failure mode emerges in live logs, diagnose and design targeted fix. Otherwise, the incumbent is at a confirmed Pareto optimum.
