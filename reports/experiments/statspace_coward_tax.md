# StatSpace Coward Tax Experiment

## Methods

Coward Tax is a 4th-down aggressiveness composite computed per team-season from nflverse PBP. It measures WP left on the table by conservative fourth-down decisions. The `aggression_score` is a z-scored composite blending: +0.50 decision edge, +0.35 aggression credit, +0.15 aggressive decisions, −0.25 coward tax, −0.15 conservative decisions.

## Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| A. FDR+DOBA+Chaos (incumbent) | 0.5609 | 0.5753 | 0.6025 | 0.5049 |
| B. Incumbent + Coward Tax (agg_score) | 0.5620 | 0.5778 | 0.6034 | 0.5047 |
| C. FDR+DOBA+Coward Tax | 0.5872 | 0.5976 | 0.6258 | 0.5383 |
| D. Incumbent + Coward Tax (coward_tax_per_game) | 0.5609 | 0.5753 | 0.6025 | 0.5049 |

## Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|------|
| A. FDR+DOBA+Chaos (incumbent) | 0.5548 | 0.1886 | 0.7871 | 0.6884 |
| B. Incumbent + Coward Tax (agg_score) | 0.5543 | 0.1883 | 0.7876 | 0.6884 |
| C. FDR+DOBA+Coward Tax | 0.5882 | 0.2024 | 0.7513 | 0.6703 |
| D. Incumbent + Coward Tax (coward_tax_per_game) | 0.5548 | 0.1886 | 0.7871 | 0.6884 |

## Decision

Incumbent (FDR+DOBA+Chaos): val=0.5609, hold=0.5548

  B. Incumbent + Coward Tax (agg_score): Δval=+0.0011, Δhold=-0.0004
  C. FDR+DOBA+Coward Tax: Δval=+0.0263, Δhold=+0.0334
  D. Incumbent + Coward Tax (coward_tax_per_game): Δval=+0.0000, Δhold=+0.0000

**No model beats incumbent on both val and holdout by ≥ 0.001.**

---
Report: statspace_coward_tax_experiment.py
