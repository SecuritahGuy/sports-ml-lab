# StatSpace QB Lift Experiment

## Methods

QB Lift Index measures a QB's value beyond their supporting cast. It blends EPA/dropback, CPOE, 3rd/4th-down EPA, scramble EPA, pressure avoidance, YAC dependency, sack rate, and garbage-time inflation into a z-scored composite. Higher = QB provides more value beyond what the supporting cast explains.

We take the primary QB (highest qb_lift_index) per team-season.

## Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| A. FDR+DOBA+Chaos (incumbent) | 0.5609 | 0.5753 | 0.6025 | 0.5049 |
| B. Incumbent + QB Lift | 0.5612 | 0.5740 | 0.6012 | 0.5082 |
| C. Incumbent + QB Lift (support) | 0.5645 | 0.5823 | 0.6004 | 0.5109 |
| D. FDR+DOBA+Chaos+QB Lift+Support | 0.5649 | 0.5819 | 0.6016 | 0.5111 |
| E. QB Lift only | 0.6329 | 0.6248 | 0.6436 | 0.6304 |

## Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|------|
| A. FDR+DOBA+Chaos (incumbent) | 0.5548 | 0.1886 | 0.7871 | 0.6884 |
| B. Incumbent + QB Lift | 0.5550 | 0.1887 | 0.7862 | 0.6884 |
| C. Incumbent + QB Lift (support) | 0.5572 | 0.1896 | 0.7862 | 0.6884 |
| D. FDR+DOBA+Chaos+QB Lift+Support | 0.5565 | 0.1893 | 0.7839 | 0.6884 |
| E. QB Lift only | 0.6288 | 0.2197 | 0.6946 | 0.6196 |

## Decision

Incumbent (FDR+DOBA+Chaos): val=0.5609, hold=0.5548

  B. Incumbent + QB Lift: Δval=+0.0002, Δhold=+0.0002
  C. Incumbent + QB Lift (support): Δval=+0.0036, Δhold=+0.0024
  D. FDR+DOBA+Chaos+QB Lift+Support: Δval=+0.0040, Δhold=+0.0017
  E. QB Lift only: Δval=+0.0720, Δhold=+0.0740

**No model beats incumbent on both val and holdout by ≥ 0.001.**

---
Report: statspace_qb_lift_experiment.py
