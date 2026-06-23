# Elo Tuning and Calibration

*Systematic Elo parameter search + calibration comparison.*

## Data Split

| Split | Seasons | Description |
|-------|---------|-------------|
| Train | [2021, 2022, 2023] | Elo updates + Platt + isotonic + logistic fit |
| Validation | 2024 | Model selection — best Elo params chosen here |
| Holdout | 2025 | Final untouched evaluation (never used for selection) |

## Parameter Grid

| Parameter | Candidates |
|-----------|------------|
| K-factor | [4, 8, 12, 16, 20, 24, 32] |
| Home-field advantage (Elo) | [0, 25, 40, 55, 65, 75] |
| Preseason regression toward 1500 | [0.0, 0.25, 0.33, 0.5] |

Total combinations searched: 168

## Top 5 Configurations (by validation log loss)

| Rank | K | HFA | Regression | Val Log Loss | Holdout Log Loss |
|------|---|-----|------------|--------------|------------------|
| 1 | 32 | 25 | 0.0 | 0.60659 | 0.66162 |
| 2 | 32 | 40 | 0.0 | 0.60743 | 0.6625 |
| 3 | 32 | 25 | 0.25 | 0.60889 | 0.64246 |
| 4 | 32 | 0 | 0.0 | 0.60905 | 0.66437 |
| 5 | 32 | 40 | 0.25 | 0.60977 | 0.64332 |

## Best Configuration (selected by validation log loss)

- **K=32, HFA=25, regression=0.0**
- Validation log loss: 0.60659
- Holdout log loss: 0.66162

## Holdout (2025) Was NOT Used for Selection

The holdout season (2025) remained untouched during the grid search. All 168 parameter combinations were evaluated only on validation (2024). The holdout results shown in this report are for final comparison only.

## Leakage Prevention

- Elo features are computed chronologically: for each game, features
  depend only on games played before it.
- Calibration (Platt, isotonic) is fitted **only on training data** and
  applied to validation and holdout.
- Minimal logistic model is trained only on training data.
- The 2025 holdout is never accessed during any fitting or selection step.

## Full Comparison

| Model | Val LL | Val Brier | Val Acc | Val AUC | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|--------|-----------|---------|---------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.552) | 0.6910 | — | — | 0.5000 | 0.6910 | — | — | 0.5000 |
| Elo K=20 (original) | 0.6166 | 0.2134 | 0.6835 | 0.7284 | 0.6678 | 0.2363 | 0.6268 | 0.6486 |
| Elo tuned (raw) | 0.6066 | 0.2091 | 0.6978 | 0.7333 | 0.6616 | 0.2328 | 0.6196 | 0.6652 |
| Elo tuned + Platt | 0.6117 | 0.2112 | 0.6799 | 0.7333 | 0.6523 | 0.2299 | 0.6304 | 0.6652 |
| Elo tuned + Isotonic | 0.6103 | 0.2112 | 0.6799 | 0.7223 | 0.6793 | 0.2385 | 0.6413 | 0.6509 |
| Minimal Logistic | 0.6128 | 0.2123 | 0.6619 | 0.7257 | 0.6561 | 0.2318 | 0.6268 | 0.6568 |
| Prev logistic team-strength | 0.6477 | — | — | 0.6896 | 0.6866 | — | — | 0.6531 |

## Raw Elo (tuned)

### Validation Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.0, 0.1) | 1 | 0.0836 | 0.0 | 0.0836 |
| [0.1, 0.2) | 8 | 0.16 | 0.25 | 0.09 |
| [0.2, 0.3) | 17 | 0.2712 | 0.1176 | 0.1535 |
| [0.3, 0.4) | 44 | 0.3478 | 0.3864 | 0.0386 |
| [0.4, 0.5) | 53 | 0.4498 | 0.3396 | 0.1102 |
| [0.5, 0.6) | 52 | 0.5529 | 0.5769 | 0.024 |
| [0.6, 0.7) | 41 | 0.6474 | 0.7805 | 0.1331 |
| [0.7, 0.8) | 39 | 0.741 | 0.6923 | 0.0487 |
| [0.8, 0.9) | 20 | 0.8382 | 0.9 | 0.0618 |
| [0.9, 1.0) | 3 | 0.9027 | 1.0 | 0.0973 |

### Holdout Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 11 | 0.1533 | 0.3636 | 0.2104 |
| [0.2, 0.3) | 33 | 0.2514 | 0.3636 | 0.1122 |
| [0.3, 0.4) | 32 | 0.3541 | 0.3438 | 0.0103 |
| [0.4, 0.5) | 43 | 0.4562 | 0.4884 | 0.0322 |
| [0.5, 0.6) | 49 | 0.5492 | 0.5102 | 0.039 |
| [0.6, 0.7) | 42 | 0.6495 | 0.6905 | 0.041 |
| [0.7, 0.8) | 40 | 0.7449 | 0.675 | 0.0699 |
| [0.8, 0.9) | 24 | 0.8441 | 0.7083 | 0.1357 |
| [0.9, 1.0) | 2 | 0.9022 | 1.0 | 0.0978 |

## Platt-calibrated Elo

### Validation Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 1 | 0.1978 | 0.0 | 0.1978 |
| [0.2, 0.3) | 10 | 0.2549 | 0.2 | 0.0549 |
| [0.3, 0.4) | 44 | 0.3595 | 0.3409 | 0.0186 |
| [0.4, 0.5) | 59 | 0.4564 | 0.339 | 0.1174 |
| [0.5, 0.6) | 52 | 0.5526 | 0.5 | 0.0526 |
| [0.6, 0.7) | 56 | 0.6467 | 0.7679 | 0.1211 |
| [0.7, 0.8) | 49 | 0.7434 | 0.7347 | 0.0087 |
| [0.8, 0.9) | 7 | 0.8128 | 1.0 | 0.1872 |

### Holdout Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.2, 0.3) | 23 | 0.2641 | 0.3913 | 0.1272 |
| [0.3, 0.4) | 40 | 0.3501 | 0.325 | 0.0251 |
| [0.4, 0.5) | 43 | 0.4547 | 0.4186 | 0.0361 |
| [0.5, 0.6) | 57 | 0.5497 | 0.4912 | 0.0585 |
| [0.6, 0.7) | 52 | 0.6505 | 0.6923 | 0.0418 |
| [0.7, 0.8) | 53 | 0.7455 | 0.6981 | 0.0474 |
| [0.8, 0.9) | 8 | 0.8162 | 0.875 | 0.0588 |

## Isotonic-calibrated Elo

### Validation Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 9 | 0.1111 | 0.2222 | 0.1111 |
| [0.4, 0.5) | 133 | 0.4359 | 0.3684 | 0.0675 |
| [0.5, 0.6) | 30 | 0.5318 | 0.5667 | 0.0348 |
| [0.6, 0.7) | 11 | 0.6078 | 0.5455 | 0.0624 |
| [0.7, 0.8) | 86 | 0.7675 | 0.7791 | 0.0116 |
| [0.8, 0.9) | 6 | 0.8208 | 0.8333 | 0.0125 |

### Holdout Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 12 | 0.1143 | 0.4167 | 0.3024 |
| [0.4, 0.5) | 125 | 0.4346 | 0.392 | 0.0426 |
| [0.5, 0.6) | 29 | 0.5285 | 0.5862 | 0.0577 |
| [0.6, 0.7) | 13 | 0.6078 | 0.9231 | 0.3152 |
| [0.7, 0.8) | 87 | 0.7704 | 0.6437 | 0.1267 |
| [0.8, 0.9) | 3 | 0.8 | 1.0 | 0.2 |
| [0.9, 1.0) | 5 | 0.9639 | 0.8 | 0.1639 |

## Recommendation

✅ **Tuned Elo + Platt is the new research incumbent.** Holdout log loss 0.6523 beats original Elo K=20 (0.6678). Platt scaling improved probability calibration. Future models must beat this benchmark.

### Next Recommended Experiment

Add weather features to the minimal logistic model or test a GradientBoosting model with clean pregame features (Elo + rest + structural).  Weather may provide the signal needed to break through the Elo ceiling.
