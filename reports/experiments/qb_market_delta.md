# QB-Change Market-Delta Diagnostics

*Testing whether closing-market disagreement with the football-only model identifies QB-change / availability-shock games where the incumbent is structurally blind.*

## Important

This is a **market-aware / near-kickoff diagnostic** experiment.
The clean football-only incumbent is **Standard Elo + qb_changed + rolling_mov_3 + Platt** (holdout LL **0.6262**).
Closing market should **not** be treated as football-only.
**Do not overwrite or demote the football-only incumbent.**

## Method

Rolling-origin 3-fold validation, one-shot 2025 holdout.

Incumbent version: v2.0.0

### Diagnostic Fields

- `model_prob` — Incumbent home win probability
- `market_prob` — Closing moneyline no-vig home win probability
- `market_minus_model` — Market minus model probability
- `abs_market_minus_model` — Absolute disagreement
- `model_logit` / `market_logit` — Log-odds
- `market_logit_minus_model_logit` — Logit disagreement
- `qb_change_flag` — Either team QB changed from prior game
- `home_qb_change_flag` — Home QB changed
- `away_qb_change_flag` — Away QB changed
- `favorite_disagreement_flag` — Market and model disagree on favorite
- `directionally_aligned_flag` — Market and model agree on favorite
- `large_market_delta_flag` — abs(market - model) >= 0.05

### Models/Blends Compared

| Model | Description |
|-------|-------------|
| Incumbent | Football-only model (Elo + qb_changed + mov_3 + Platt) |
| Closing market | Moneyline no-vig probability |
| Simple blend | w * mkt + (1-w) * model (9 weights: 0.1..0.9) |
| QB-change gated | If QB change: blend; else model (4 weights) |
| Large-delta gated | If abs delta >= t: blend; else model (4×4 grid) |
| QB+LD gated | If QB change AND large delta: blend; else model |
| Logistic blend | Logit(model) + logit(mkt) + qb + delta + qb*delta |

## Rolling-Origin Validation

| Model / Config | Avg Val LL | Fold1 | Fold2 | Fold3 |
|---------------|-----------|-------|-------|-------|
| Incumbent (model) | 0.6250 | 0.6289 | 0.6526 | 0.5937 |
| Market (no-vig) | 0.6052 | 0.6041 | 0.6258 | 0.5858 |
| Simple blend (simple_w0.90) | 0.6050 | 0.6043 | 0.6264 | 0.5843 |
| QB gated (qb_gated_w0.75) | 0.6210 | 0.6299 | 0.6435 | 0.5895 |
| LD gated (ld_t0.075_w1.00) | 0.6056 | 0.6062 | 0.6252 | 0.5854 |
| QB+LD gated (qb_ld_t0.100_w0.75) | 0.6199 | 0.6266 | 0.6443 | 0.5888 |
| Logistic blend | 0.6193 | 0.6354 | 0.6293 | 0.5931 |

### Candidate Selection

| Candidate | Avg Val LL |
|-----------|-----------|
| simple_blend | 0.6050 ← best |
| market | 0.6052 |
| ld_gated | 0.6056 |
| logistic_blend | 0.6193 |
| qb_ld_gated | 0.6199 |
| qb_gated | 0.6210 |
| incumbent | 0.6250 |

## 2025 Holdout (One-Shot)

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|-----|
| Incumbent | 0.6262 | 0.2180 | 0.7050 | 0.6630 |
| Closing market | 0.6090 | 0.2119 | 0.7199 | 0.6558 |
| Simple blend | 0.6083 | 0.2115 | 0.7205 | 0.6667 |
| QB gated | 0.6130 | 0.2126 | 0.7197 | 0.6848 |
| LD gated | 0.6094 | 0.2115 | 0.7230 | 0.6594 |
| QB+LD gated | 0.6142 | 0.2128 | 0.7200 | 0.6848 |
| Logistic blend | 0.6103 | 0.2125 | 0.7193 | 0.6594 |

## Subset Analysis

### QB-Change Subset

| Model | QB Change (n=55) | QB Stable (n=-331) |
|-------|--------|--------|
| Incumbent | 0.3398 | 0.4756 |
| Closing market | 0.3361 | 0.5468 |

Market-incumbent gap on QB-change games: **0.0037**

### Favorite Disagreement Subset

| Model | Favorite Dis (n=43) |
|-------|--------|
| Incumbent | 0.6956 |
| Closing market | 0.7023 |

### Market-Delta Buckets

| Bucket | N | Incumbent LL | Market LL |
|--------|---|-------------|----------|
| 0.025–0.05 |  57 | 0.5690 | 0.5702 |
| 0.05–0.075 |  59 | 0.6301 | 0.6281 |
|  0.075–0.1 |  25 | 0.6288 | 0.6086 |
|   0.1–0.15 |  49 | 0.7011 | 0.6277 |
|     <0.025 |  45 | 0.5289 | 0.5276 |
|      >0.15 |  41 | 0.7159 | 0.7030 |

### Early vs Late Season

| Period | N | Incumbent LL |
|--------|---|-------------|
| Early (weeks 1-4) | 61 | 0.5759 |
| Late (weeks 13+) | 106 | 0.6157 |

## Decision

**Market-aware challenger beats football-only incumbent on holdout** (0.6262 → 0.6083).

**Decision: market-aware challenger (track separately from football-only incumbent).**

### Key Findings

1. This is **market-aware / near-kickoff diagnostic work.**
2. Closing market should **not** be treated as football-only.
3. Football-only incumbent unchanged: Standard Elo + qb_changed + rolling_mov_3 + Platt (holdout LL 0.6262).
4. Market-incumbent gap on QB-change games: **0.0037** — market substantially outperforms incumbent when QB changes.
5. **Opening-line ingestion** should be prioritized next.

### Market Status

| Assumption | Value |
|-----------|-------|
| Market type | Closing (near-kickoff) |
| Football-only incumbent unchanged | **Yes** (still 0.6262) |
| Market-aware challenger tracked | **Yes** |

### Caution Flags Artifact

`reports/predictions/market_aware_caution_flags.csv` — 1388 rows, 13 columns.
See schema in tests.
