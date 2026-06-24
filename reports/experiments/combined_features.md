# Combined Feature Experiment

## Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| Platt | 0.6406 | 0.6471 | 0.6621 | 0.6126 |
| Platt + qb_changed | 0.6334 | 0.6388 | 0.6598 | 0.6017 |
| Platt + rolling_mov_3 | 0.6406 | 0.6495 | 0.6593 | 0.6131 |
| Platt + qb_changed + mov3 | 0.6334 | 0.6413 | 0.6573 | 0.6016 |
| Platt + coach_win_pct | 0.6417 | 0.6481 | 0.6643 | 0.6127 |
| Platt + coach_tenure | 0.6456 | 0.6439 | 0.6675 | 0.6253 |
| Platt + all_coach | 0.6542 | 0.6610 | 0.6774 | 0.6242 |
| Platt + coach + qb | 0.6469 | 0.6536 | 0.6745 | 0.6126 |

## Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|------|
| Platt | 0.6315 | 0.2204 | 0.6983 | 0.6739 |
| Platt + qb_changed | 0.6314 | 0.2205 | 0.6974 | 0.6558 |
| Platt + rolling_mov_3 | 0.6255 | 0.2176 | 0.7075 | 0.6667 |
| Platt + qb_changed + mov3 | 0.6262 | 0.2180 | 0.7050 | 0.6630 |
| Platt + coach_win_pct | 0.6326 | 0.2207 | 0.6991 | 0.6703 |
| Platt + coach_tenure | 0.6452 | 0.2262 | 0.6795 | 0.6449 |
| Platt + all_coach | 0.6756 | 0.2375 | 0.6564 | 0.6341 |
| Platt + coach + qb | 0.6771 | 0.2383 | 0.6577 | 0.6087 |

## Decision

Best on validation: Platt + qb_changed + mov3 (0.6334).
Incumbent holdout: 0.6315
**Promoted: Platt + qb_changed, Platt + qb_changed + mov3**
