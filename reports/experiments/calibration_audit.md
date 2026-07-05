# Calibration & Uncertainty Audit

*Model: v3.0.0 Frozen QB Overlay*
*Holdout: 2025 season*

## Summary

| Metric | Value |
|--------|-------|
| Holdout LL | 0.6200 |
| Holdout Brier | 0.2157 |
| Accuracy | 0.6630 |
| AUC | 0.7098 |
| **ECE** | **0.0628** |
| **MCE** | **0.1343** |
| N (holdout) | 276 |

## ECE & MCE

Expected Calibration Error (ECE): weighted average of absolute difference between mean predicted probability and observed frequency across 10 equal-width bins. Maximum Calibration Error (MCE): max over bins.

- **ECE** = 0.0628
- **MCE** = 0.1343
- Bins = 10

Interpretation:
- ECE < 0.02: well-calibrated (avg within 2% of true frequency)
- MCE > 0.10: some bins have meaningful miscalibration
- Check reliability diagram and per-bucket table below for which bins.

## Reliability Diagram

```
Reliability Diagram
  (bars show fraction of positives per bucket; ideal = diagonal)

Bucket               N   Pred  Actual   Err  Chart
--------------------------------------------------------------------------------------------
[0.1, 0.2)          14 0.164  0.214   0.050 ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
[0.2, 0.3)          22 0.252  0.318   0.066 ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
[0.3, 0.4)          34 0.358  0.265   0.093 █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
[0.4, 0.5)          49 0.450  0.469   0.020 ███████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
[0.5, 0.6)          41 0.553  0.634   0.081 ███████████████████████████████░░░░░░░░░░░░░░░░░░░
[0.6, 0.7)          50 0.653  0.600   0.053 ██████████████████████████████░░░░░░░░░░░░░░░░░░░░
[0.7, 0.8)          31 0.747  0.613   0.134 ██████████████████████████████░░░░░░░░░░░░░░░░░░░░
[0.8, 0.9)          29 0.847  0.862   0.015 ███████████████████████████████████████████░░░░░░░
[0.9, 1.0)           6 0.921  1.000   0.079 ██████████████████████████████████████████████████

Ideal (p=actual)                        █████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░
```

## Per-Bucket Calibration Table

| Bucket | N | Mean Pred | Mean Actual | Cal Error |
|--------|---|-----------|-------------|-----------|
| [0.1, 0.2) | 14 | 0.164 | 0.214 | 0.0504 |
| [0.2, 0.3) | 22 | 0.252 | 0.318 | 0.0662 |
| [0.3, 0.4) | 34 | 0.358 | 0.265 | 0.0931 |
| [0.4, 0.5) | 49 | 0.450 | 0.469 | 0.0197 |
| [0.5, 0.6) | 41 | 0.553 | 0.634 | 0.0814 |
| [0.6, 0.7) | 50 | 0.653 | 0.600 | 0.0527 |
| [0.7, 0.8) | 31 | 0.747 | 0.613 | 0.1343 |
| [0.8, 0.9) | 29 | 0.847 | 0.862 | 0.0146 |
| [0.9, 1.0) | 6 | 0.921 | 1.000 | 0.0787 |

## Brier Score Decomposition

| Component | Value | Description |
|-----------|-------|-------------|
| Brier score | 0.2157 | Raw mean-squared error |
| Uncertainty | 0.2487 | Base-rate variance (p̄(1-p̄)); upper bound if always predicting 0.5 |
| Resolution | 0.0376 | How much predictions deviate from base rate by subgroup |
| Reliability | 0.0053 | Calibration error component; lower is better |
| Decomposed Brier | 0.2164 | = Uncertainty - Resolution + Reliability |

## Sharpness (Confidence Distribution)

How spread out are the predicted probabilities? A well-sharpened model concentrates predictions away from 0.5.

| Bin | Count | % of Predictions |
|-----|-------|-----------------|
| [0.0, 0.1) | 0 |   0.0% █
| [0.1, 0.2) | 14 |   5.1% █
| [0.2, 0.3) | 22 |   8.0% ██
| [0.3, 0.4) | 34 |  12.3% ████
| [0.4, 0.5) | 49 |  17.8% █████
| [0.5, 0.6) | 41 |  14.9% ████
| [0.6, 0.7) | 50 |  18.1% ██████
| [0.7, 0.8) | 31 |  11.2% ███
| [0.8, 0.9) | 29 |  10.5% ███
| [0.9, 1.0) | 6 |   2.2% █

## Home-Favorite Directional Error

WARNING: This is NOT a general over/underconfidence metric.

Only predictions > 0.5 (model favors home team) are counted.

| Metric | Value |
|--------|-------|
| Overconfident (pred > actual) | 51 / 276 (18.5%) |
| Underconfident (pred < actual) | 106 / 276 (38.4%) |
| Correct | 183 / 276 (66.3%) |

## Subset-Specific Calibration

| Subset | N | LL | Brier | Acc | AUC | ECE | MCE | Over% | Under% |
|--------|---|----|-------|-----|-----|-----|-----|-------|--------|
| All (2025 holdout) | 276 | 0.6200 | 0.2157 | 0.6630 | 0.7098 | 0.0628 | 0.1343 | 18.5 | 38.4 |
| QB-change games | 55 | 0.6674 | 0.2377 | 0.5455 | 0.6753 | 0.2097 | 0.5690 | 29.1 | 41.8 |
| Non-QB-change games | 221 | 0.6082 | 0.2102 | 0.6923 | 0.7282 | 0.0714 | 0.1517 | 15.8 | 37.6 |
| Early season (W1-4) | 61 | 0.5844 | 0.1995 | 0.6885 | 0.7590 | 0.0771 | 0.1554 | 13.1 | 42.6 |
| Mid-late season (W5+) | 215 | 0.6301 | 0.2202 | 0.6558 | 0.7008 | 0.0741 | 0.2484 | 20.0 | 37.2 |
| Gate active | 199 | 0.5888 | 0.2014 | 0.6935 | 0.7626 | 0.0726 | 0.1692 | 18.1 | 37.2 |
| Gate inactive | 77 | 0.7007 | 0.2524 | 0.5844 | 0.5364 | 0.1006 | 0.7278 | 19.5 | 41.6 |
| High confidence (>=0.8) | 35 | 0.3769 | 0.1072 | 0.8857 | 0.2742 | 0.0256 | 0.0787 | 11.4 | 88.6 |
| Low confidence (<=0.2) | 14 | 0.5227 | 0.1696 | 0.7857 | 0.5152 | 0.0504 | 0.0504 | 0.0 | 0.0 |
| Mid confidence (0.4-0.6) | 90 | 0.6824 | 0.2447 | 0.5778 | 0.5839 | 0.0478 | 0.0814 | 16.7 | 28.9 |

## Fold Stability (Calibration)

| Validation Season | N | Val LL | ECE | MCE |
|-----------------|---|--------|-----|-----|
| 2022 | 275 | 0.6360 | 0.0663 | 0.1938 |
| 2023 | 279 | 0.6596 | 0.0602 | 0.3210 |
| 2024 | 278 | 0.5960 | 0.0521 | 0.1675 |

ECE across folds: mean=0.0595 std=0.0058 range=[0.0521, 0.0663]
MCE across folds: mean=0.2274 std=0.067 range=[0.1675, 0.3210]

## Key Findings

### Calibration Quality

- **ECE = 0.0628**: moderate miscalibration detected.
- **MCE = 0.1343**: meaningful miscalibration in at least one bin.

### Sharpness

- Extreme predictions (<0.1 or >=0.9): 2.2% of predictions
- Near-50/50 predictions (0.4-0.6): 32.7% of predictions

### Subset Gaps

- **QB-change games**: ECE=0.2097 (LL=0.6674)
- **Gate inactive**: ECE=0.1006 (LL=0.7007)
- **Early season (W1-4)**: ECE=0.0771 (LL=0.5844)
- **Mid-late season (W5+)**: ECE=0.0741 (LL=0.6301)
- **Gate active**: ECE=0.0726 (LL=0.5888)
- **Non-QB-change games**: ECE=0.0714 (LL=0.6082)
- **All (2025 holdout)**: ECE=0.0628 (LL=0.6200)
- **Low confidence (<=0.2)**: ECE=0.0504 (LL=0.5227)

### Fold Stability

- ECE stable across folds (range 0.0142) — calibration generalizes

### Known Limitations

- ASCII reliability diagram is text-based; no matplotlib dependency.
- ECE/MCE use equal-width bins (10). Adaptive binning may give different results.
- Subset analysis on small N (<50 games) may be noisy.
- No isotonic or temperature-scaled comparison — this audit covers the incumbent only.

---
*Report generated by `sportslab calibration-audit`. Model: v3.0.0, Holdout: 2025.*
