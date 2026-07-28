# StatSpace Features on Pi-Ratings Base

Testing whether StatSpace PBP composites (FDR, DOBA, Chaos) improve
on the Pi-Ratings football-only champion.

## Configs

| ID | Model | Base Rating | StatSpace Features |
|---|-------|-------------|-------------------|
| A | Current overall champion | Standard Elo (K=36) | FDR + DOBA + Chaos |
| B | Football-only champion | Pi-Ratings (α=0.5) | None |
| C | Pi + FDR | Pi-Ratings | FDR |
| D | Pi + FDR + DOBA | Pi-Ratings | FDR + DOBA |
| E | Pi + FDR + DOBA + Chaos | Pi-Ratings | FDR + DOBA + Chaos |

## Validation (Rolling-Origin 3-Fold)

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| A. Elo + FDR + DOBA + Chaos (champion) | 0.5609 | 0.5753 | 0.6025 | 0.5049 |
| B. Pi-Ratings only | 0.6266 | 0.6313 | 0.6485 | 0.5999 |
| C. Pi + FDR | 0.6074 | 0.6155 | 0.6397 | 0.5671 |
| D. Pi + FDR + DOBA | 0.5775 | 0.5812 | 0.6204 | 0.5308 |
| E. Pi + FDR + DOBA + Chaos | 0.5557 | 0.5625 | 0.6015 | 0.5030 |

## Holdout (2025)

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|------|
| A. Elo + FDR + DOBA + Chaos (champion) | 0.5548 | 0.1886 | 0.7871 | 0.6884 |
| B. Pi-Ratings only | 0.6350 | 0.2217 | 0.6962 | 0.6268 |
| C. Pi + FDR | 0.5998 | 0.2070 | 0.7368 | 0.6703 |
| D. Pi + FDR + DOBA | 0.5913 | 0.2039 | 0.7475 | 0.6630 |
| E. Pi + FDR + DOBA + Chaos | 0.5532 | 0.1886 | 0.7874 | 0.6957 |

## Comparison vs Current Champion (Elo + FDR + DOBA + Chaos)

Incumbent (A): val=0.5609, hold=0.5548

| Model | Δval | Δhold | Decision |
|-------|------|-------|----------|
| B. Pi-Ratings only | +0.0657 | +0.0803 | Worse on both |
| C. Pi + FDR | +0.0465 | +0.0450 | Worse on both |
| D. Pi + FDR + DOBA | +0.0165 | +0.0365 | Worse on both |
| E. Pi + FDR + DOBA + Chaos | -0.0053 | -0.0015 | ✅ PROMOTED |

## Comparison vs Pi-Ratings Only (football-only champion)

Pi-only (B): val=0.6266, hold=0.6350

| Model | Δval | Δhold | Decision |
|-------|------|-------|----------|
| C. Pi + FDR | -0.0192 | -0.0353 | ✅ PROMOTED over Pi-only |
| D. Pi + FDR + DOBA | -0.0491 | -0.0438 | ✅ PROMOTED over Pi-only |
| E. Pi + FDR + DOBA + Chaos | -0.0709 | -0.0818 | ✅ PROMOTED over Pi-only |

## Decision

**Promoted: E. Pi + FDR + DOBA + Chaos**

StatSpace features improve on Pi-Ratings base.

---
Report: pi_statspace_experiment.py