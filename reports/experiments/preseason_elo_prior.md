# RALPH Loop 8: Preseason Elo Prior

*Incumbent: v3.0.0 Frozen QB Overlay (holdout LL 0.62)*

## Hypothesis

The model underperforms in Weeks 1-4 (LL=0.6727 vs late season 0.6032) because season-start Elo ratings carry only a diluted signal from the prior season (10% regression toward 1500). Adding the pre-regression prior-season final Elo as an explicit Platt feature provides a stronger preseason reference, especially before current-season rolling averages accumulate.

## Variants

| ID | Description |
|----|-------------|
| incumbent | Platt(qb_changed + rolling_mov_3) — no prior_elo |
| prior_elo_raw | Incumbent + raw prior-season final Elo (no regression) |
| prior_elo_reg10 | Incumbent + regressed prior Elo (10%) |
| prior_elo_reg50 | Incumbent + regressed prior Elo (50%) |
| prior_elo_diff | Incumbent + prior elo diff |
| prior_elo_raw_decay | Incumbent + decay-weighted prior elo |

## Rolling-Origin Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 | Δ vs Inc |
|-------|-----------|-------|-------|-------|----------|
| incumbent | 0.6341 | 0.6416 | 0.6577 | 0.6031 | +0.0000 |
| prior_elo_raw | 0.6362 | 0.6416 | 0.6636 | 0.6034 | +0.0021 |
| prior_elo_reg10 | 0.6362 | 0.6416 | 0.6636 | 0.6034 | +0.0021 |
| prior_elo_reg50 | 0.6362 | 0.6416 | 0.6636 | 0.6034 | +0.0021 |
| prior_elo_diff | 0.6342 | 0.6416 | 0.6581 | 0.6029 | +0.0001 |
| prior_elo_raw_decay | 0.6324 | 0.6410 | 0.6552 | 0.6011 | -0.0017 |

## Holdout (2025)

| Model | Hold LL | Brier | Acc | AUC | ECE | Δ vs Inc |
|-------|---------|-------|-----|-----|-----|----------|
| incumbent | 0.6259 | 0.2181 | 0.6594 | 0.7048 | 0.055736 | +0.0000 |
| prior_elo_raw | 0.6283 | 0.2192 | 0.6558 | 0.7025 | 0.037332 | +0.0024 |
| prior_elo_reg10 | 0.6283 | 0.2192 | 0.6558 | 0.7025 | 0.037332 | +0.0024 |
| prior_elo_reg50 | 0.6283 | 0.2192 | 0.6558 | 0.7025 | 0.037332 | +0.0024 |
| prior_elo_diff | 0.6282 | 0.2191 | 0.6594 | 0.703 | 0.050283 | +0.0023 |
| prior_elo_raw_decay | 0.6301 | 0.22 | 0.6449 | 0.6992 | 0.051243 | +0.0042 |

## Season-by-Season (2025 holdout)

| Model | Season | N | Log Loss |
|-------|--------|---|----------|
| incumbent | 2025 | 276 | 0.6259 |
| prior_elo_raw | 2025 | 276 | 0.6283 |
| prior_elo_reg10 | 2025 | 276 | 0.6283 |
| prior_elo_reg50 | 2025 | 276 | 0.6283 |
| prior_elo_diff | 2025 | 276 | 0.6282 |
| prior_elo_raw_decay | 2025 | 276 | 0.6301 |

## Subgroup Analysis (Best: prior_elo_diff)

| Subgroup | N | Incumbent LL | Best LL | Δ |
|----------|---|-------------|---------|---|
| early | 61 | 0.5839 | 0.5803 | -0.0036 |
| mid | 109 | 0.6617 | 0.6657 | +0.0040 |
| late | 106 | 0.6134 | 0.6173 | +0.0039 |
| qb_changed | 55 | 0.6696 | 0.6695 | -0.0001 |
| weather_missing | 92 | 0.6470 | 0.6488 | +0.0018 |

## Decisions

**No challenger beats the incumbent.**

### ❌ prior_elo_raw

Val LL: 0.6362 (Δ=0.0021), Holdout: 0.6283 (Δ=0.0024)

Rejected — val worse (+0.0021) holdout worse (+0.0024).

### ❌ prior_elo_reg10

Val LL: 0.6362 (Δ=0.0021), Holdout: 0.6283 (Δ=0.0024)

Rejected — val worse (+0.0021) holdout worse (+0.0024).

### ❌ prior_elo_reg50

Val LL: 0.6362 (Δ=0.0021), Holdout: 0.6283 (Δ=0.0024)

Rejected — val worse (+0.0021) holdout worse (+0.0024).

### ❌ prior_elo_diff

Val LL: 0.6342 (Δ=0.0001), Holdout: 0.6282 (Δ=0.0023)

Rejected — val better (+0.0001) holdout worse (+0.0023).

### ❌ prior_elo_raw_decay

Val LL: 0.6324 (Δ=-0.0017), Holdout: 0.6301 (Δ=0.0042)

Rejected — val better (-0.0017) holdout worse (+0.0042).

## Leakage Assessment

| Feature | Source | Leakage Risk | Live-Safe |
|---------|--------|--------------|-----------|
| home_prior_elo_raw | Prior-season final Elo | None (pre-kickoff) | Yes |
| away_prior_elo_raw | Prior-season final Elo | None (pre-kickoff) | Yes |

## Operational Impact

* No change to Elo pipeline
* No new external data sources
* No change to live weekly prediction mode
* All prior-season Elo values computed from existing games

## Audit Answers

Best variant: **prior_elo_diff**

1. **Improves Weeks 1-4?** ✅ Improves by -0.0036
2. **Improves both val and holdout by ≥0.001?** ❌ No
3. **Stable across folds?** See fold table above.
4. **Increases overconfidence?** ≈ Flat (Δ=-0.0055)
5. **Worsens QB-change games?** ≈ Flat (Δ=-0.0001)
6. **Worsens missing-weather games?** ❌ Worsens by 0.0018
7. **Data available before kickoff?** Yes — prior-season Elo is known
8. **Changes live weekly operation?** No — feature added to Platt model
9. **Adds operational fragility?** No — static feature
10. **Result large enough?** ❌ No

