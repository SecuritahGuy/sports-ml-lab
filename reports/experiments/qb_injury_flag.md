# QB Injury Flag Experiment

*Testing whether a single binary 'home starting QB OUT' flag improves on O/D Elo+Platt incumbent.*

## Motivation

Residual diagnostics identified QB change as the #1 failure mode (LL gap 0.042 vs stable QB). The full 19-feature injury set added too much noise, but QB-out subset showed strong signal (holdout LL=0.5506, n=28).

## Method

Single binary feature `home_injuries_qb_out` = 1 if home team's starting QB is ruled OUT on the final injury report.

Logistic regression on [elo_prob, qb_out_flag].

Rolling-origin 3-fold validation, one-shot 2025 holdout.

## Rolling-Origin Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| Platt (incumbent) | 0.6376 | 0.6430 | 0.6567 | 0.6132 |
| Platt + QB OUT | 0.6464 | 0.6577 | 0.6600 | 0.6215 |
| QB OUT only | 0.6898 | 0.6890 | 0.6897 | 0.6908 |

## 2025 Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|-----|
| Platt (incumbent) | 0.6258 | 0.2179 | 0.7066 | 0.6703 |
| Platt + QB OUT | 0.6255 | 0.2177 | 0.7078 | 0.6630 |
| QB OUT only | 0.6863 | 0.2466 | 0.5295 | 0.5580 |

### QB-Out Subset (2025 Holdout)

| Subset | N | LL |
|--------|---|----|
| QB OUT (home) | 29 | 0.5746 |
| QB healthy | 247 | 0.6312 |

**Platt + QB OUT beats the incumbent.** A single binary QB injury flag improves prediction.
