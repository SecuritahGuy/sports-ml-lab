# StatSpace DOBA Experiment

## Methods

DOBA (sustainable Offensive efficiency composite) is computed per team-season from nflverse PBP. It blends offensive EPA/play, success rate, early-down efficiency, third/fourth-down efficiency, explosive rate, red zone efficiency, negative play rate, turnover rate, and a dependency penalty into a z-scored composite where higher = more sustainable offensive quality.

## Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| A. Platt (no features) | 0.6342 | 0.6416 | 0.6577 | 0.6031 |
| B. Platt + FDR (incumbent) | 0.6203 | 0.6263 | 0.6540 | 0.5805 |
| C. Platt + DOBA | 0.6034 | 0.5951 | 0.6424 | 0.5729 |
| D. Platt + FDR + DOBA | 0.5853 | 0.5949 | 0.6226 | 0.5382 |
| E. DOBA only | 0.6068 | 0.5937 | 0.6430 | 0.5838 |

## Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|------|
| A. Platt (no features) | 0.6259 | 0.2181 | 0.7048 | 0.6594 |
| B. Platt + FDR (incumbent) | 0.6011 | 0.2078 | 0.7329 | 0.6703 |
| C. Platt + DOBA | 0.6241 | 0.2171 | 0.7100 | 0.6413 |
| D. Platt + FDR + DOBA | 0.5945 | 0.2052 | 0.7458 | 0.6449 |
| E. DOBA only | 0.6301 | 0.2196 | 0.7000 | 0.6449 |

## Decision

Incumbent (FDR): val=0.6203, hold=0.6011

  A. Platt (no features): Δval=+0.0139, Δhold=+0.0249
  C. Platt + DOBA: Δval=-0.0168, Δhold=+0.0230
  D. Platt + FDR + DOBA: Δval=-0.0350, Δhold=-0.0066
  E. DOBA only: Δval=-0.0135, Δhold=+0.0290

**Promoted: D. Platt + FDR + DOBA**

---
Report: statspace_doba_experiment.py
