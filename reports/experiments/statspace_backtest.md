# StatSpace Metric Backtest

Each metric is tested as a standalone predictor using prior-season team values (lagged by 1 year). Raw = sigmoid(home - away diff) as probability. Platt = logistic regression fit on 2022-2024 diffs.

| Metric | Raw LL | Platt LL | Raw Brier | Platt Brier | Raw AUC | Platt AUC |
|--------|--------|----------|-----------|-------------|---------|-----------|
| Raw Elo | 0.7062 | 0.6918 | 0.2549 | 0.2487 | 0.575 | 0.575 |
| FDR | 1.0431 | 0.6912 | 0.3300 | 0.2490 | 0.540 | 0.540 |
| DOBA | 0.8920 | 0.6986 | 0.3117 | 0.2521 | 0.547 | 0.547 |
| Chaos Rate | 0.7625 | 0.6924 | 0.2705 | 0.2496 | 0.579 | 0.421 |
| Coward Tax | 0.9724 | 0.6936 | 0.3280 | 0.2501 | 0.509 | 0.509 |
| QB Lift | 0.7662 | 0.7087 | 0.2816 | 0.2572 | 0.489 | 0.489 |
| Chaos Rate (2y avg) | 0.7413 | 0.6900 | 0.2690 | 0.2484 | 0.544 | 0.544 |
| Chaos Rate (3y avg) | 0.7182 | 0.6898 | 0.2606 | 0.2484 | 0.552 | 0.552 |
| Combined (all) | 0.7220 | 0.7062 | 0.2618 | 0.2553 | 0.559 | 0.538 |

## Rankings (by Platt-calibrated log loss)

1. Chaos Rate (3y avg): 0.6898 (AUC=0.552)
2. Chaos Rate (2y avg): 0.6900 (AUC=0.544)
3. FDR: 0.6912 (AUC=0.540)
4. Raw Elo: 0.6918 (AUC=0.575)
5. Chaos Rate: 0.6924 (AUC=0.421)
6. Coward Tax: 0.6936 (AUC=0.509)
7. DOBA: 0.6986 (AUC=0.547)
8. Combined (all): 0.7062 (AUC=0.538)
9. QB Lift: 0.7087 (AUC=0.489)

---
Report: statspace_backtest.py
