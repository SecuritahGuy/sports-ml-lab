# Coach+QB Season Regression Experiment

*Testing whether adding coach-change preseason regression to QB-change regression improves prediction.*

## Motivation

Teams with a new head coach should also get extra preseason regression, similar to QB changes.  A new coach means a new system, new playbook, and potentially different team philosophy — making past Elo ratings less informative.  This experiment tests a combined regression bonus:

```
effective_regression = base + (qb_bonus if QB changed) + (coach_bonus if coach changed)
capped at 1.0
```

## Parameter Grid

| Parameter | Candidates |
|-----------|------------|
| K-factor (frozen) | 36 |
| HFA (frozen) | 40 |
| Decay (frozen) | 32 |
| Base regression | [0.0, 0.1, 0.2] |
| QB change bonus | [0.0, 0.1, 0.2, 0.3] |
| Coach change bonus | [0.0, 0.1, 0.2, 0.3] |
| MOV (frozen) | capped_linear, scale=0.05, cap=2.0 |

Total combinations searched: 48

## Coach Change Counts

Total coach changes across all seasons: 33

| Season | Coach Changes |
|--------|---------------|
| 2021 | 0 () |
| 2022 | 12 (12 teams) |
| 2023 | 7 (ARI, CAR, DEN, HOU, IND, LAC, LV) |
| 2024 | 7 (ATL, CAR, LAC, NE, SEA, TEN, WAS) |
| 2025 | 7 (CHI, DAL, JAX, LV, NE, NO, NYJ) |

## Best Configuration

- Base regression: 0.1
- QB change bonus: 0.3
- Coach change bonus: 0.1
- Average validation log loss: 0.63093
- Holdout raw Elo: 0.6290
- Holdout + Platt: 0.6286
- Incumbent (QB-reg+Platt): 0.6285

## Top 8 Configurations

| Rank | Reg | QB Bonus | Coach Bonus | Avg Val LL | Fold1 | Fold2 | Fold3 |
|------|-----|----------|-------------|-----------|-------|-------|-------|
| 1 | 0.1 | 0.3 | 0.1 | 0.63093 | 0.62937 | 0.65807 | 0.60536 |
| 2 | 0.1 | 0.2 | 0.1 | 0.63111 | 0.62984 | 0.65834 | 0.60515 |
| 3 | 0.1 | 0.3 | 0.2 | 0.63137 | 0.62989 | 0.65817 | 0.60605 |
| 4 | 0.1 | 0.0 | 0.3 | 0.63145 | 0.63086 | 0.65958 | 0.60391 |
| 5 | 0.1 | 0.2 | 0.2 | 0.63147 | 0.63026 | 0.65836 | 0.60579 |
| 6 | 0.1 | 0.2 | 0.0 | 0.63148 | 0.63013 | 0.65892 | 0.6054 |
| 7 | 0.1 | 0.3 | 0.0 | 0.63159 | 0.62989 | 0.65885 | 0.60603 |
| 8 | 0.1 | 0.0 | 0.2 | 0.6316 | 0.6309 | 0.65969 | 0.60421 |

## Validation Comparison

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Best config raw | 0.6309 | 0.6294 | 0.6581 | 0.6054 |
| Best config + Platt | 0.6365 | 0.6425 | 0.6575 | 0.6096 |

## Holdout (2025) Comparison

| Model | Hold LL | Brier | Acc | AUC |
|-------|---------|-------|-----|-----|
| Random | 0.6931 | — | — | — |
| Home prior (0.548) | 0.6908 | — | — | — |
| Incumbent (QB-reg+Platt) | 0.6285 | — | — | — |
| Best raw | 0.6290 | 0.2196 | 0.6703 | 0.7010 |
| Best + Platt | 0.6286 | 0.2191 | 0.6703 | 0.7010 |

## Decision

❌ **Coach+QB regression does not beat the incumbent.**

Best + Platt holdout: 0.6286
Incumbent holdout: 0.6285

## Leakage Prevention

- QB/coach changes detected from all data (purely past information).
- Rolling-origin folds prevent 2025 holdout access.
- Team regression overrides only applied at season boundaries.
- Platt calibration fitted only on training data.

