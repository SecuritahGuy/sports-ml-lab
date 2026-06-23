# Residual Diagnostics: MOV Elo + Platt Incumbent

Diagnostic analysis of where the incumbent fails systematically.

## 1. Overall Performance

| Metric | Train (2021-2024) | Holdout (2025) |
|--------|-------------------|----------------|
| Log loss | 0.6429 | 0.6373 |
| Brier score | 0.2259 | 0.2230 |
| Accuracy | 0.6268 | 0.6522 |
| ROC AUC | — | 0.6907 |
| N | 1112 | 276 |

## 2. Calibration (Holdout 2025)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 3 | 0.1928 | 0.3333 | 0.1406 |
| [0.2, 0.3) | 30 | 0.2529 | 0.3667 | 0.1138 |
| [0.3, 0.4) | 26 | 0.3567 | 0.2692 | 0.0875 |
| [0.4, 0.5) | 53 | 0.4446 | 0.3962 | 0.0484 |
| [0.5, 0.6) | 53 | 0.5532 | 0.6038 | 0.0505 |
| [0.6, 0.7) | 48 | 0.6502 | 0.5833 | 0.0669 |
| [0.7, 0.8) | 47 | 0.7473 | 0.7021 | 0.0452 |
| [0.8, 0.9) | 16 | 0.8111 | 0.9375 | 0.1264 |

## 3. Residuals by Team

Average residual (predicted - actual). Negative = model was too pessimistic (underpredicted home wins).

### Worst-predicted teams (highest mean |residual|)

| Team | Side | N | Mean Residual | Mean |Residual| |
|------|------|---|---------------|-----------------|
| CLE | home | 34 | -0.0861 | 0.5131 |
| CIN | away | 38 | +0.0896 | 0.5024 |
| BUF | away | 35 | +0.0055 | 0.4990 |
| JAX | home | 31 | +0.0611 | 0.4984 |
| GB | away | 35 | +0.0118 | 0.4955 |
| SF | away | 37 | +0.0087 | 0.4940 |
| DEN | home | 34 | -0.0460 | 0.4921 |
| ATL | home | 33 | +0.0092 | 0.4914 |
| NYJ | home | 34 | +0.0478 | 0.4903 |
| PIT | home | 33 | -0.0531 | 0.4876 |
| NO | away | 34 | -0.0022 | 0.4849 |
| BAL | away | 35 | +0.0482 | 0.4845 |
| NO | home | 32 | +0.1207 | 0.4819 |
| TEN | home | 34 | +0.0695 | 0.4807 |
| LV | away | 35 | -0.0791 | 0.4802 |

## 4. Residuals by Game Context

### By Season

| Group | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|-------|---|-----------|-------------|----------|-----------------|
| 2021 | 280 | 0.5525 | 0.5143 | 0.6744 | 0.4747 |
| 2023 | 279 | 0.5458 | 0.5663 | 0.6568 | 0.4559 |
| 2024 | 278 | 0.5442 | 0.5360 | 0.6042 | 0.4283 |
| 2022 | 275 | 0.5480 | 0.5745 | 0.6359 | 0.4482 |

### By Game Type

| Group | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|-------|---|-----------|-------------|----------|-----------------|
| Regular | 1065 | 0.5442 | 0.5399 | 0.6449 | 0.4527 |
| Wild Card | 23 | 0.6254 | 0.7826 | 0.5613 | 0.4121 |
| Divisional | 16 | 0.6504 | 0.6250 | 0.6128 | 0.4346 |
| Conference | 8 | 0.5848 | 0.7500 | 0.6755 | 0.4785 |

### By Weekday

| Group | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|-------|---|-----------|-------------|----------|-----------------|
| Sunday | 899 | 0.5435 | 0.5406 | 0.6453 | 0.4536 |
| Monday | 80 | 0.5692 | 0.5375 | 0.6935 | 0.4728 |
| Thursday | 75 | 0.5502 | 0.5467 | 0.6326 | 0.4475 |
| Saturday | 52 | 0.5866 | 0.6923 | 0.5677 | 0.4091 |
| Friday | 2 | 0.5927 | 0.5000 | 0.3140 | 0.2636 |
| Tuesday | 2 | 0.6344 | 1.0000 | 0.4559 | 0.3656 |
| Wednesday | 2 | 0.3388 | 0.0000 | 0.4142 | 0.3388 |

### By Roof

| Group | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|-------|---|-----------|-------------|----------|-----------------|
| outdoors | 734 | 0.5561 | 0.5640 | 0.6409 | 0.4487 |
| dome | 207 | 0.5456 | 0.5507 | 0.6481 | 0.4583 |
| closed | 139 | 0.5169 | 0.4748 | 0.6281 | 0.4487 |
| open | 32 | 0.5016 | 0.4688 | 0.7206 | 0.4950 |

### By Week

| Week | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|------|---|-----------|-------------|----------|-----------------|
| 1 | 61 | 0.5457 | 0.4754 | 0.6726 | 0.4717 |
| 2 | 64 | 0.5522 | 0.5000 | 0.6850 | 0.4788 |
| 3 | 64 | 0.5538 | 0.5000 | 0.7034 | 0.4807 |
| 4 | 62 | 0.5369 | 0.5000 | 0.6769 | 0.4761 |
| 5 | 56 | 0.5503 | 0.5179 | 0.6521 | 0.4585 |
| 6 | 54 | 0.5247 | 0.4630 | 0.6413 | 0.4558 |
| 7 | 54 | 0.5512 | 0.6296 | 0.6256 | 0.4461 |
| 8 | 61 | 0.5381 | 0.5410 | 0.6344 | 0.4466 |
| 9 | 55 | 0.5375 | 0.6182 | 0.6502 | 0.4602 |
| 10 | 52 | 0.5773 | 0.5577 | 0.7005 | 0.4757 |
| 11 | 55 | 0.5380 | 0.5273 | 0.6162 | 0.4409 |
| 12 | 60 | 0.5311 | 0.5167 | 0.6306 | 0.4446 |
| 13 | 57 | 0.5328 | 0.4912 | 0.5767 | 0.4203 |
| 14 | 55 | 0.5879 | 0.6000 | 0.6222 | 0.4342 |
| 15 | 64 | 0.5063 | 0.5156 | 0.6204 | 0.4410 |
| 16 | 64 | 0.5471 | 0.6094 | 0.6219 | 0.4391 |
| 17 | 63 | 0.5683 | 0.6190 | 0.5542 | 0.4028 |
| 18 | 64 | 0.5229 | 0.5469 | 0.7169 | 0.4738 |
| 19 | 23 | 0.6254 | 0.7826 | 0.5613 | 0.4121 |
| 20 | 16 | 0.6504 | 0.6250 | 0.6128 | 0.4346 |
| 21 | 8 | 0.5848 | 0.7500 | 0.6755 | 0.4785 |

### Short Week (Home Team)

| Short Week | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|------------|---|-----------|-------------|----------|-----------------|
| No | 940 | 0.5447 | 0.5362 | 0.6486 | 0.4546 |
| Yes | 172 | 0.5640 | 0.6105 | 0.6120 | 0.4363 |

### Off Bye (Home Team)

| Off Bye | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|---------|---|-----------|-------------|----------|-----------------|
| No | 1037 | 0.5470 | 0.5419 | 0.6469 | 0.4538 |
| Yes | 75 | 0.5568 | 0.6267 | 0.5873 | 0.4238 |

### Primetime Games

| Game Type | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|-----------|---|-----------|-------------|----------|-----------------|
| Not Thursday | 1037 | 0.5475 | 0.5477 | 0.6437 | 0.4521 |
| Thursday | 75 | 0.5502 | 0.5467 | 0.6326 | 0.4475 |

### Bad Weather

| Weather | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|---------|---|-----------|-------------|----------|-----------------|
| Normal | 914 | 0.5443 | 0.5350 | 0.6441 | 0.4535 |
| Bad weather | 198 | 0.5632 | 0.6061 | 0.6374 | 0.4439 |

### Indoor vs Outdoor

| Venue | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|-------|---|-----------|-------------|----------|-----------------|
| Outdoor | 766 | 0.5538 | 0.5601 | 0.6442 | 0.4506 |
| Indoor | 346 | 0.5341 | 0.5202 | 0.6401 | 0.4545 |

### QB Change (Home Team)

| QB Changed | N | Mean Pred | Mean Actual | Log Loss | Mean |Residual| |
|------------|---|-----------|-------------|----------|-----------------|
| No | 983 | 0.5503 | 0.5554 | 0.6381 | 0.4494 |
| Yes | 129 | 0.5277 | 0.4884 | 0.6799 | 0.4704 |

## 5. Residuals vs Elo Confidence

Do residuals grow when Elo is more confident (extreme probabilities)?

| Elo Bucket | N | Mean Pred | Mean Actual | Mean Residual | Log Loss |
|------------|---|-----------|-------------|---------------|----------|
| [0.0,0.1) | 4 | 0.1849 | 0.2500 | -0.0651 | 0.5862 |
| [0.1,0.2) | 20 | 0.2223 | 0.3500 | -0.1277 | 0.6716 |
| [0.2,0.3) | 79 | 0.2977 | 0.4177 | -0.1200 | 0.7076 |
| [0.3,0.4) | 132 | 0.3765 | 0.2955 | +0.0810 | 0.6281 |
| [0.4,0.5) | 183 | 0.4592 | 0.4590 | +0.0002 | 0.6926 |
| [0.5,0.6) | 220 | 0.5502 | 0.4818 | +0.0684 | 0.7035 |
| [0.6,0.7) | 216 | 0.6301 | 0.6435 | -0.0135 | 0.6515 |
| [0.7,0.8) | 165 | 0.7101 | 0.7818 | -0.0717 | 0.5355 |
| [0.8,0.9) | 81 | 0.7700 | 0.7284 | +0.0416 | 0.5853 |
| [0.9,1.0) | 12 | 0.8242 | 1.0000 | -0.1758 | 0.1934 |

## 6. Extreme Prediction Errors (Holdout 2025)

Games where the model was most confidently wrong.

| Predicted | Actual | Residual | Home | Away | Week |
|-----------|--------|----------|------|------|------|
| 0.811 | 0 | +0.8111 | BUF | NE | 5 |
| 0.189 | 1 | -0.8110 | NYG | PHI | 6 |
| 0.799 | 0 | +0.7990 | PHI | WAS | 18 |
| 0.215 | 1 | -0.7855 | TEN | KC | 16 |
| 0.785 | 0 | +0.7851 | GB | CAR | 9 |
| 0.777 | 0 | +0.7773 | TB | NO | 14 |
| 0.233 | 1 | -0.7667 | CAR | LA | 13 |
| 0.238 | 1 | -0.7617 | MIA | BUF | 10 |
| 0.761 | 0 | +0.7605 | PHI | DEN | 5 |
| 0.240 | 1 | -0.7603 | LV | KC | 18 |
| 0.758 | 0 | +0.7585 | BAL | CIN | 13 |
| 0.242 | 1 | -0.7582 | CLE | GB | 3 |
| 0.250 | 1 | -0.7503 | DAL | PHI | 12 |
| 0.749 | 0 | +0.7488 | MIN | ATL | 2 |
| 0.749 | 0 | +0.7486 | CIN | NYJ | 8 |
| 0.252 | 1 | -0.7479 | CLE | PIT | 17 |
| 0.254 | 1 | -0.7459 | NYG | LAC | 4 |
| 0.743 | 0 | +0.7433 | ARI | TEN | 5 |
| 0.259 | 1 | -0.7415 | ATL | LA | 17 |
| 0.739 | 0 | +0.7390 | PHI | CHI | 13 |

## 7. Directional Bias

- Games where model confidently predicted home win but away won: 56
- Games where model confidently predicted away win but home won: 40
- Mean residual (holdout): +0.0059
- Model is optimistic about home teams: YES

## 8. Best vs Worst Predicted Teams (Holdout)

### Worst Predicted Teams (highest MAE)

| Team | N | Mean |Residual| | Mean Residual |
|------|---|-----------------|---------------|
| KC | 7 | 0.5911 | -0.5261 |
| DAL | 7 | 0.5607 | -0.0993 |
| ATL | 8 | 0.5521 | -0.0792 |
| ATL | 8 | 0.5496 | +0.0676 |
| NE | 9 | 0.5389 | +0.5389 |
| PIT | 8 | 0.5386 | -0.0133 |
| MIN | 8 | 0.5240 | +0.0468 |
| PHI | 9 | 0.5182 | +0.0335 |
| WAS | 8 | 0.5175 | -0.0718 |
| CAR | 9 | 0.5132 | -0.1828 |

### Best Predicted Teams (lowest MAE)

| Team | N | Mean |Residual| | Mean Residual |
|------|---|-----------------|---------------|
| LV | 8 | 0.2536 | -0.1220 |
| NYG | 9 | 0.2937 | -0.1792 |
| ARI | 9 | 0.3408 | -0.0989 |
| NYJ | 8 | 0.3408 | -0.1536 |
| TEN | 9 | 0.3422 | +0.1676 |
| HOU | 9 | 0.3573 | -0.1434 |
| TEN | 8 | 0.3619 | -0.0030 |
| CLE | 8 | 0.3682 | -0.0584 |
| LA | 8 | 0.3722 | -0.2024 |
| WAS | 8 | 0.3771 | +0.2575 |

## 9. Market Efficiency Check

Correlation between incumbent residual and market probability:
- Correlation(residual, spread_line): r=-0.097, p=0.1073
- Interpretation: Residuals are independent of market spread

## 10. Summary & Recommendations

### Where the model works well

- Overall log loss 0.6373 is 0.00 away from market (0.6373).
- Calibration is reasonable (check decile table above).
- Performance is consistent across most game contexts.

### Where the model struggles

- Team prediction quality varies (worst: CLE MAE=0.5131)
- Check high-error seasons/game types from tables above.
- Extreme predictions (very confident) still miss sometimes.

### Recommended Next Steps

1. **DVOA/EPA features** — check if nflreadpy provides advanced metrics.
2. **Move to market-relative modeling** — use market odds as a baseline
   and model the residual (market vs actual).
3. **Per-team Elo initialization** — teams may need different starting Elos.
4. **Coach/coordinators features** — system changes affect team performance.
