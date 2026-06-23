# Decayed Elo Experiment

*Testing exponential decay toward mean for Elo ratings.*

## Motivation

Standard Elo weights all past games equally across a team's history.
Exponential decay decreases the weight of older games, so recent form
matters more.  After each game, both teams' ratings decay toward the
mean (capped_linear = 36).

## Method

After each game's rating update, apply:

```
decay = 2^(-1 / half_life)
rating = 1500 + (rating - 1500) * decay
```

`half_life` = number of games to halve deviation from 1500.
Lower half_life = faster mean reversion.

## Parameter Grid

| Parameter | Candidates |
|-----------|------------|
| K-factor | [20, 24, 28, 32, 36, 40, 44, 48] |
| HFA | [30, 40] |
| Regression | [0.0, 0.2] |
| Decay half-life | [None, 32, 16, 8, 4] (games) |
| MOV (frozen) | capped_linear, scale=0.05, cap=2.0 |

Total combinations searched: 160

## Best Configuration (by avg val LL across folds)

- **K=36, HFA=40, reg=0.2**
- **Decay half-life: 32 games**
- Average validation log loss: 0.63211
- Holdout (2025) raw Elo: 0.6301
- Holdout (2025) + Platt: 0.6298
- Incumbent (MOV+Platt) holdout: 0.6373

## Top 8 Configurations

| Rank | K | HFA | Reg | Decay | Avg Val LL | Fold1 | Fold2 | Fold3 |
|------|---|-----|-----|-------|-----------|-------|-------|-------|
| 1 | 36 | 40 | 0.2 | 32 | 0.63211 | 0.63094 | 0.65958 | 0.60582 |
| 2 | 32 | 40 | 0.2 | 32 | 0.63215 | 0.63172 | 0.65728 | 0.60745 |
| 3 | 40 | 40 | 0.2 | 32 | 0.63259 | 0.63069 | 0.66223 | 0.60485 |
| 4 | 32 | 40 | 0.0 | 32 | 0.63269 | 0.63307 | 0.65882 | 0.60618 |
| 5 | 36 | 30 | 0.2 | 32 | 0.63271 | 0.63239 | 0.66064 | 0.60511 |
| 6 | 32 | 30 | 0.2 | 32 | 0.63272 | 0.63314 | 0.65832 | 0.60671 |
| 7 | 28 | 40 | 0.0 | 32 | 0.63273 | 0.63395 | 0.65626 | 0.60796 |
| 8 | 28 | 40 | 0.2 | 32 | 0.63281 | 0.63312 | 0.65543 | 0.60989 |

## Validation Comparison

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Decayed Elo raw | 0.6321 | 0.6309 | 0.6596 | 0.6058 |
| Decayed + Platt | 0.6371 | 0.6427 | 0.6574 | 0.6111 |

## Holdout (2025) Comparison

| Model | Hold LL | Brier | Acc | AUC |
|-------|---------|-------|-----|-----|
| Random | 0.6931 | — | — | — |
| Home prior (0.548) | 0.6908 | — | — | — |
| Incumbent (MOV+Platt) | 0.6373 | — | — | — |
| Best decayed raw Elo | 0.6301 | 0.2201 | 0.6630 | 0.7024 |
| Best decayed + Platt | 0.6298 | 0.2197 | 0.6558 | 0.7024 |

## Decision

✅ **Decayed Elo (K=36, decay=32) + Platt beats the incumbent.**

Holdout log loss 0.6298 vs incumbent 0.6373.

## Leakage Prevention

- Decay is applied post-game, never pregame.
- Pregame Elo features are recorded before the update.
- Rolling-origin folds prevent 2025 holdout access.
- Platt calibration fitted only on training data.

