# Player Recovery Experiment

## Data

- Seasons: 2021-2025
- Return events identified: 425
- Games with non-zero recovery adjustment: 319/1388
- Rolling-origin 3-fold: 2022/2023/2024 val, 2025 holdout

## Variants

| ID | Model | Description |
|---|-------|-------------|
| A | Incumbent (Pi-Ratings + qb_changed + mov_3 + Platt) | Base model |
| B | Incumbent + Recovery | Logit-space adjustment from player recovery curves |
| C | Recovery only | Recovery adjustment on 0.5 baseline (diagnostic) |

## Validation (Rolling-Origin 3-Fold)

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| Incumbent (Pi only) | 0.6266 | 0.6313 | 0.6485 | 0.5999 |
| Incumbent + Recovery (logit adj) | 0.6282 | 0.6362 | 0.6519 | 0.5966 |
| Recovery only | 0.6955 | 0.6993 | 0.6956 | 0.6915 |

## Holdout (2025)

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|------|
| Incumbent (Pi only) | 0.6350 | 0.2217 | 0.6962 | 0.6268 |
| Incumbent + Recovery (logit adj) | 0.6305 | 0.2201 | 0.6996 | 0.6268 |
| Recovery only | 0.6898 | 0.2482 | 0.5450 | 0.5362 |

## Comparison vs Incumbent

Incumbent (A): val=0.6266, hold=0.6350

| Model | Δval | Δhold | Decision |
|-------|------|-------|----------|
| Incumbent + Recovery (logit adj) | +0.0016 | -0.0045 | Loses val, wins hold |
| Recovery only | +0.0689 | +0.0548 | Worse on both |

## Recovery Curve Summary

Key findings from the recovery analysis:

- **QB**: Week 1 bounce (−4.7 fantasy deficit, i.e. *better* than baseline). Week 2 regression (+2.8).
- **RB**: Week 1 slight bounce (−1.0). Week 2 regression (+2.0).
- **WR**: Small persistent deficit (+0.5 week 1, +0.5 week 2).
- **TE**: Small persistent deficit (+0.6 week 1, +0.5 week 2).
- **Compounding**: WR repeat injuries (+1.86) worse than single (+0.16).

## Decision

**No recovery variant beats incumbent on both val and holdout by ≥ 0.001.**

---
Report: player_recovery_experiment.py