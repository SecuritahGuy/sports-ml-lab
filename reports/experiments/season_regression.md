# Season-Specific Regression Experiment

*Testing whether extra preseason regression for teams with QB changes improves prediction.*

## Motivation

The residual diagnostics showed QB changes were the largest failure mode.
Teams entering a new season with a different starting QB should have more
regression toward 1500, because their past Elo rating was earned with a
different QB.  Season-specific regression = base + qb_change_bonus.

## Parameter Grid

| Parameter | Candidates |
|-----------|------------|
| K-factor | [20, 28, 36, 44] |
| HFA | [30, 40] |
| Regression | [0.0, 0.1, 0.2, 0.3] |
| Decay half-life | [32] |
| QB change bonus | [0.0, 0.05, 0.1, 0.2, 0.35, 0.5] |
| MOV (frozen) | capped_linear, scale=0.05, cap=2.0 |

Total combinations searched: 192

## Best Configuration

- K=36, HFA=40, reg=0.1
- Decay half-life: 32
- QB change bonus: 0.2
- Average validation log loss: 0.63148
- Holdout raw Elo: 0.6290
- Holdout + Platt: 0.6285
- Incumbent (decayed+Platt): 0.6298

## Top 8 Configurations

| Rank | K | HFA | Reg | Decay | QB Bonus | Avg Val LL | Fold1 | Fold2 | Fold3 |
|------|---|-----|-----|-------|----------|-----------|-------|-------|-------|
| 1 | 36 | 40 | 0.1 | 32 | 0.2 | 0.63148 | 0.63013 | 0.65892 | 0.6054 |
| 2 | 36 | 40 | 0.1 | 32 | 0.1 | 0.63176 | 0.63072 | 0.65943 | 0.60512 |
| 3 | 36 | 40 | 0.1 | 32 | 0.35 | 0.63178 | 0.6299 | 0.65897 | 0.60647 |
| 4 | 36 | 40 | 0.2 | 32 | 0.1 | 0.63184 | 0.63035 | 0.65908 | 0.60611 |
| 5 | 36 | 40 | 0.2 | 32 | 0.05 | 0.63193 | 0.6306 | 0.65927 | 0.60592 |
| 6 | 36 | 40 | 0.2 | 32 | 0.2 | 0.63196 | 0.6301 | 0.65901 | 0.60675 |
| 7 | 36 | 40 | 0.1 | 32 | 0.05 | 0.63204 | 0.63115 | 0.65985 | 0.60513 |
| 8 | 36 | 30 | 0.1 | 32 | 0.2 | 0.63208 | 0.63159 | 0.65997 | 0.60468 |

## Validation Comparison

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Best config raw | 0.6315 | 0.6301 | 0.6589 | 0.6054 |
| Best config + Platt | 0.6368 | 0.6425 | 0.6576 | 0.6103 |

## Holdout (2025) Comparison

| Model | Hold LL | Brier | Acc | AUC |
|-------|---------|-------|-----|-----|
| Random | 0.6931 | — | — | — |
| Home prior (0.548) | 0.6908 | — | — | — |
| Incumbent (decayed+Platt) | 0.6298 | — | — | — |
| Best raw | 0.6290 | 0.2195 | 0.6667 | 0.7024 |
| Best + Platt | 0.6285 | 0.2191 | 0.6667 | 0.7024 |

## Decision

✅ **Season-specific regression (qb_bonus=0.2) + Platt beats the incumbent.**

Holdout log loss 0.6285 vs incumbent 0.6298.

## Leakage Prevention

- QB change detected from training seasons only (per-fold).
- Rolling-origin folds prevent 2025 holdout access.
- Team regression overrides only applied to teams with confirmed QB change.
- Platt calibration fitted only on training data.

