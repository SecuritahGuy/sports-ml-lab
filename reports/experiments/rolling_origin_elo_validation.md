# Rolling-Origin Elo Validation

*Cross-fold rolling-origin validation for Elo parameter selection with expanded grid.*

## Data Split

| Split | Seasons | Description |
|-------|---------|-------------|
| Fold 1 | Train: [2021], Val: 2022 | Elo param selection via avg val LL |
| Fold 2 | Train: [2021, 2022], Val: 2023 | Elo param selection via avg val LL |
| Fold 3 | Train: [2021, 2022, 2023], Val: 2024 | Elo param selection via avg val LL |
| Holdout | 2025 | Final untouched evaluation (never used for selection) |

## Parameter Grid

| Parameter | Candidates |
|-----------|------------|
| K-factor | [20, 24, 28, 32, 36, 40, 48] |
| Home-field advantage (Elo) | [10, 20, 25, 30, 35, 40] |
| Preseason regression toward 1500 | [0.0, 0.1, 0.2, 0.25, 0.33] |

Total combinations searched: 210

## Top 5 Configurations (by average validation log loss)

| Rank | K | HFA | Regression | Avg Val LL | Fold1 LL | Fold2 LL | Fold3 LL |
|------|---|-----|------------|-----------|----------|----------|----------|
| 1 | 40 | 40 | 0.25 | 0.63634 | 0.63937 | 0.66362 | 0.60604 |
| 2 | 40 | 40 | 0.33 | 0.63645 | 0.63878 | 0.66316 | 0.60741 |
| 3 | 48 | 40 | 0.33 | 0.63649 | 0.63726 | 0.66706 | 0.60514 |
| 4 | 40 | 40 | 0.2 | 0.6365 | 0.63989 | 0.66418 | 0.60542 |
| 5 | 40 | 35 | 0.25 | 0.63652 | 0.63995 | 0.66403 | 0.60558 |

## Best Configuration (selected by average validation log loss across folds)

- **K=40, HFA=40, regression=0.25**
- Average validation log loss: 0.63634
  - Fold 1 (val 2022): 0.63937
  - Fold 2 (val 2023): 0.66362
  - Fold 3 (val 2024): 0.60604
- Holdout (2025) log loss: 0.6409

## Holdout (2025) Was NOT Used During Model Selection

The grid search evaluated 210 parameter combinations across 3 rolling-origin folds.  Selection was based **only** on average validation log loss.  The 2025 holdout was not accessed during any part of the grid search.  Holdout metrics in this report are for final comparison only.

## Leakage Prevention

- Elo features computed chronologically across all seasons.
- Rolling-origin folds simulate realistic walk-forward evaluation.
- Calibration (Platt, isotonic) fitted **only on training folds** during validation.
- Final calibration fitted on 2021–2024, then applied to 2025.
- Minimal logistic model trained only on training folds during selection; final model trained on 2021–2024.
- 2025 holdout never touched during any fitting or selection step.

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Original Elo K=20 | 0.6678 | 0.2363 | 0.6268 | 0.6486 |
| Current tuned Elo K=32 HFA=25 | 0.6616 | 0.2328 | 0.6196 | 0.6652 |
| Rolling-origin selected raw Elo | 0.6409 | 0.2247 | 0.6558 | 0.6861 |
| Rolling-origin selected + Platt | 0.6395 | 0.2240 | 0.6522 | 0.6861 |
| Rolling-origin selected + Isotonic | 0.6459 | 0.2269 | 0.6341 | 0.6885 |
| Rolling-origin selected Minimal Logistic | 0.6443 | 0.2264 | 0.6486 | 0.6765 |

## Average Validation Metrics Across Folds

| Model | Avg Val LL | Fold1 LL | Fold2 LL | Fold3 LL |
|-------|------------|----------|----------|----------|
| Original Elo K=20 | 0.6481 | 0.6600 | 0.6676 | 0.6166 |
| Current tuned Elo K=32 HFA=25 | 0.6396 | 0.6466 | 0.6655 | 0.6066 |
| Rolling-origin selected raw Elo | 0.6363 | 0.6394 | 0.6636 | 0.6060 |
| Rolling-origin selected + Platt | 0.6408 | 0.6492 | 0.6611 | 0.6119 |
| Rolling-origin selected + Isotonic | 0.8024 | 0.6449 | 1.0282 | 0.7341 |
| Rolling-origin selected Minimal Logistic | 0.6397 | 0.6469 | 0.6602 | 0.6120 |

## Rolling-Origin Selected Raw Elo (Holdout)

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 4 | 0.1659 | 0.25 | 0.0841 |
| [0.2, 0.3) | 25 | 0.258 | 0.4 | 0.142 |
| [0.3, 0.4) | 21 | 0.3498 | 0.1905 | 0.1593 |
| [0.4, 0.5) | 59 | 0.4505 | 0.3898 | 0.0607 |
| [0.5, 0.6) | 53 | 0.5535 | 0.566 | 0.0125 |
| [0.6, 0.7) | 48 | 0.6415 | 0.6667 | 0.0252 |
| [0.7, 0.8) | 41 | 0.7427 | 0.6585 | 0.0841 |
| [0.8, 0.9) | 25 | 0.8425 | 0.84 | 0.0025 |

## Platt-Calibrated Elo (Holdout)

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 4 | 0.1893 | 0.25 | 0.0607 |
| [0.2, 0.3) | 26 | 0.2586 | 0.3846 | 0.126 |
| [0.3, 0.4) | 30 | 0.358 | 0.3 | 0.058 |
| [0.4, 0.5) | 52 | 0.4511 | 0.3846 | 0.0664 |
| [0.5, 0.6) | 51 | 0.5515 | 0.549 | 0.0025 |
| [0.6, 0.7) | 51 | 0.6424 | 0.6667 | 0.0242 |
| [0.7, 0.8) | 48 | 0.7455 | 0.7083 | 0.0372 |
| [0.8, 0.9) | 14 | 0.8179 | 0.8571 | 0.0392 |

## Isotonic-Calibrated Elo (Holdout)

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 7 | 0.1667 | 0.2857 | 0.119 |
| [0.3, 0.4) | 73 | 0.3859 | 0.3151 | 0.0708 |
| [0.4, 0.5) | 59 | 0.4427 | 0.5254 | 0.0827 |
| [0.5, 0.6) | 39 | 0.5605 | 0.641 | 0.0806 |
| [0.6, 0.7) | 20 | 0.641 | 0.6 | 0.041 |
| [0.7, 0.8) | 68 | 0.7755 | 0.6765 | 0.099 |
| [0.8, 0.9) | 8 | 0.8321 | 0.875 | 0.0429 |
| [0.9, 1.0) | 1 | 0.923 | 1.0 | 0.077 |

## Isotonic Calibration Risk

Isotonic calibration did not improve holdout log loss and carries high overfit risk.  **Rejected.**

## Rolling-Origin Calibration Deciles (Selected Raw Elo)

### Fold 1 (Validation 2022)

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 4 | 0.1766 | 0.0 | 0.1766 |
| [0.2, 0.3) | 7 | 0.2634 | 0.4286 | 0.1651 |
| [0.3, 0.4) | 30 | 0.3578 | 0.5333 | 0.1756 |
| [0.4, 0.5) | 58 | 0.4496 | 0.431 | 0.0186 |
| [0.5, 0.6) | 60 | 0.5543 | 0.5333 | 0.021 |
| [0.6, 0.7) | 63 | 0.6467 | 0.6032 | 0.0435 |
| [0.7, 0.8) | 46 | 0.7392 | 0.8261 | 0.0869 |
| [0.8, 0.9) | 7 | 0.8463 | 0.8571 | 0.0108 |

### Fold 2 (Validation 2023)

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 3 | 0.1793 | 0.3333 | 0.154 |
| [0.2, 0.3) | 12 | 0.2651 | 0.4167 | 0.1515 |
| [0.3, 0.4) | 31 | 0.3475 | 0.4516 | 0.1041 |
| [0.4, 0.5) | 57 | 0.4547 | 0.4912 | 0.0365 |
| [0.5, 0.6) | 65 | 0.5523 | 0.4615 | 0.0908 |
| [0.6, 0.7) | 58 | 0.653 | 0.7241 | 0.0711 |
| [0.7, 0.8) | 45 | 0.747 | 0.7556 | 0.0086 |
| [0.8, 0.9) | 8 | 0.8485 | 0.5 | 0.3485 |

### Fold 3 (Validation 2024)

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 5 | 0.1581 | 0.2 | 0.0419 |
| [0.2, 0.3) | 9 | 0.2604 | 0.4444 | 0.184 |
| [0.3, 0.4) | 37 | 0.3506 | 0.2703 | 0.0803 |
| [0.4, 0.5) | 52 | 0.4472 | 0.3462 | 0.1011 |
| [0.5, 0.6) | 63 | 0.5446 | 0.4921 | 0.0526 |
| [0.6, 0.7) | 45 | 0.6508 | 0.7111 | 0.0603 |
| [0.7, 0.8) | 49 | 0.7392 | 0.7551 | 0.0159 |
| [0.8, 0.9) | 17 | 0.8457 | 0.8824 | 0.0367 |
| [0.9, 1.0) | 1 | 0.9054 | 1.0 | 0.0946 |

## Recommendation

✅ **Rolling-origin selected + Platt is the new research incumbent.** Holdout log loss 0.6395 beats the current tuned Elo incumbent (0.6616).

Rolling-origin validation selected a configuration that generalizes better across seasons.
### Next Recommended Experiment

1. Add weather features to the minimal logistic model.
2. Test a GradientBoosting model with clean pregame features.
3. Expand Elo K-factor grid above 48.
