# StatSpace FDR Experiment (vs Full Champion)

## Methods

FDR (Fraud Detector Rating) is computed per team-season from nflverse PBP, schedule results, and Elo ratings. It blends record strength, underlying quality, luck gap, close-game luck, turnover luck, and schedule suspicion into a z-scored composite where positive = overachieving (regression risk) and negative = underachieving (upside).

The champion model is:
  `Platt(elo_prob + qb_changed + rolling_mov_3) + QB overlay`
  (gate = qb_changed OR starts<17, gamma=1.0, cap=40)

## Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| A. Platt (no overlay) | 0.6342 | 0.6416 | 0.6577 | 0.6031 |
| B. Platt + overlay (champion) | 0.6317 | 0.6377 | 0.6627 | 0.5947 |
| C. Platt + FDR | 0.6203 | 0.6263 | 0.6540 | 0.5805 |
| D. Platt + FDR + overlay | 0.6172 | 0.6241 | 0.6558 | 0.5715 |

## Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|------|
| A. Platt (no overlay) | 0.6259 | 0.2181 | 0.7048 | 0.6594 |
| B. Platt + overlay (champion) | 0.6228 | 0.2169 | 0.7067 | 0.6594 |
| C. Platt + FDR | 0.6011 | 0.2078 | 0.7329 | 0.6703 |
| D. Platt + FDR + overlay | 0.5972 | 0.2065 | 0.7356 | 0.6775 |

## Decision

Champion: val=0.6317, hold=0.6228

  A. Platt (no overlay): Δval=+0.0025, Δhold=+0.0031
  C. Platt + FDR: Δval=-0.0114, Δhold=-0.0217
  D. Platt + FDR + overlay: Δval=-0.0145, Δhold=-0.0256

**Promoted: C. Platt + FDR, D. Platt + FDR + overlay**

---
Report: statspace_fdr_experiment.py
