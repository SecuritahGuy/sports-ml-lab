# StatSpace Chaos Rate Experiment

## Methods

Chaos Rate is a defensive disruption composite computed per team-season from nflverse PBP. It blends defensive EPA/play allowed, success rate allowed, negative EPA forced rate, sack rate, turnover forced rate, explosive rate allowed, third/fourth-down stop rate, and penalty first-down rate allowed into a z-scored composite where higher = more disruptive defense.

## Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| A. Platt + FDR + DOBA (incumbent) | 0.5853 | 0.5949 | 0.6226 | 0.5382 |
| B. Platt + FDR + DOBA + Chaos | 0.5609 | 0.5753 | 0.6025 | 0.5049 |
| C. Platt + Chaos | 0.6258 | 0.6347 | 0.6483 | 0.5943 |
| D. Platt + FDR + Chaos | 0.6129 | 0.6149 | 0.6572 | 0.5667 |
| E. Chaos only | 0.6416 | 0.6422 | 0.6586 | 0.6239 |

## Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|------|
| A. Platt + FDR + DOBA (incumbent) | 0.5945 | 0.2052 | 0.7458 | 0.6449 |
| B. Platt + FDR + DOBA + Chaos | 0.5548 | 0.1886 | 0.7871 | 0.6884 |
| C. Platt + Chaos | 0.6072 | 0.2099 | 0.7276 | 0.6703 |
| D. Platt + FDR + Chaos | 0.5884 | 0.2025 | 0.7467 | 0.6739 |
| E. Chaos only | 0.6232 | 0.2169 | 0.7056 | 0.6558 |

## Decision

Incumbent (FDR+DOBA): val=0.5853, hold=0.5945

  B. Platt + FDR + DOBA + Chaos: Δval=-0.0244, Δhold=-0.0397
  C. Platt + Chaos: Δval=+0.0405, Δhold=+0.0127
  D. Platt + FDR + Chaos: Δval=+0.0276, Δhold=-0.0061
  E. Chaos only: Δval=+0.0563, Δhold=+0.0287

**Promoted: B. Platt + FDR + DOBA + Chaos**

---
Report: statspace_chaos_experiment.py
