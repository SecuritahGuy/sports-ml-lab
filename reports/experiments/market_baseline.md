# Market Baseline Comparison

Comparing the MOV Elo+Platt incumbent against moneyline-implied probabilities.

## Methodology

- Moneyline odds converted to implied probabilities:
  - Negative odds (favorite): `prob = -odds / (-odds + 100)`
  - Positive odds (underdog): `prob = 100 / (odds + 100)`
- Vig (overround) removed via multiplicative normalization:
  `fair_home_prob = home_implied / (home_implied + away_implied)`
- All comparisons use rolling-origin validation (2025 untouched until final).
- Elo + Market combines both signals via logistic regression.

## Data

| Stat | Value |
|------|-------|
| Games (filtered) | 1388 |
| Market prob range | [0.0876, 0.9258] |
| Market prob mean | 0.5526 |
| Avg overround | 0.0369' |
## Incumbent Params

| Parameter | Value |
|-----------|-------|
| K-factor | 36 |
| HFA | 40 |
| Preseason regression | 0.2 |
| MOV type | capped_linear |
| MOV scale | 0.05 |
| MOV cap | 2.0 |

## Data Split

| Fold | Training | Validation |
|------|----------|------------|
| 1 | [2021] | 2022 |
| 2 | [2021, 2022] | 2023 |
| 3 | [2021, 2022, 2023] | 2024 |
| Holdout | 2021–2024 | 2025 |

## Models Compared

| Model | Description |
|-------|-------------|
| **Platt (incumbent)** | MOV Elo + Platt scaling |
| **Market (raw)** | De-vigged moneyline implied probability |
| **Market + Platt** | Platt-calibrated market (tests favorite-longshot bias) |
| **Elo + Market** | Logistic regression on both signals |

## Average Validation Log Loss

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Platt (incumbent) | 0.6363 | 0.6438 | 0.6564 | 0.6088 |
| Market (raw) | 0.6052 | 0.6042 | 0.6258 | 0.5858 |
| Market + Platt | 0.6088 | 0.6147 | 0.6268 | 0.5848 |
| Elo + Market | 0.6189 | 0.6359 | 0.6234 | 0.5975 |

## 2025 Holdout Comparison

| Model | Hold LL | Brier | Acc | AUC |
|-------|---------|-------|-----|-----|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6373 | 0.2230 | 0.6522 | 0.6907 |
| Market (raw) | 0.6090 | 0.2119 | 0.6558 | 0.7199 |
| Market + Platt | 0.6127 | 0.2131 | 0.6594 | 0.7199 |
| Elo + Market | 0.6119 | 0.2128 | 0.6522 | 0.7204 |

## Market (Raw, Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 10 | 0.1689 | 0.1 | 0.0689 |
| [0.2, 0.3) | 24 | 0.2477 | 0.25 | 0.0023 |
| [0.3, 0.4) | 33 | 0.3451 | 0.4242 | 0.0791 |
| [0.4, 0.5) | 46 | 0.4424 | 0.413 | 0.0293 |
| [0.5, 0.6) | 46 | 0.5532 | 0.5652 | 0.0121 |
| [0.6, 0.7) | 48 | 0.6451 | 0.5417 | 0.1034 |
| [0.7, 0.8) | 40 | 0.7491 | 0.725 | 0.0241 |
| [0.8, 0.9) | 26 | 0.8574 | 0.9231 | 0.0657 |
| [0.9, 1.0) | 3 | 0.9054 | 1.0 | 0.0946 |

## Market + Platt (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 17 | 0.176 | 0.1765 | 0.0005 |
| [0.2, 0.3) | 27 | 0.2518 | 0.2593 | 0.0074 |
| [0.3, 0.4) | 36 | 0.3454 | 0.4444 | 0.0991 |
| [0.4, 0.5) | 36 | 0.4403 | 0.4167 | 0.0237 |
| [0.5, 0.6) | 41 | 0.5536 | 0.561 | 0.0074 |
| [0.6, 0.7) | 44 | 0.6474 | 0.6136 | 0.0338 |
| [0.7, 0.8) | 45 | 0.7511 | 0.6444 | 0.1066 |
| [0.8, 0.9) | 30 | 0.8429 | 0.9333 | 0.0905 |

## Platt Incumbent (Holdout)

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

## Elo + Market (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 13 | 0.1714 | 0.1538 | 0.0175 |
| [0.2, 0.3) | 33 | 0.2512 | 0.303 | 0.0518 |
| [0.3, 0.4) | 36 | 0.3532 | 0.3889 | 0.0356 |
| [0.4, 0.5) | 34 | 0.4453 | 0.4706 | 0.0252 |
| [0.5, 0.6) | 39 | 0.553 | 0.5128 | 0.0402 |
| [0.6, 0.7) | 46 | 0.6423 | 0.6087 | 0.0336 |
| [0.7, 0.8) | 45 | 0.7476 | 0.6667 | 0.081 |
| [0.8, 0.9) | 30 | 0.841 | 0.9333 | 0.0923 |

## Recommendation

✅ **Market (raw) beats the incumbent.**

Holdout log loss 0.6090 vs incumbent 0.6373.

⚠️ **Elo does not add information beyond market odds.**
Elo + Market (hold LL=0.6119) does not beat Market alone (0.6090).

### Favorite-Longshot Bias

Market + Platt calibration result indicates whether the market has systematic favorite-longshot bias:
- ⚠️ Platt calibration does not improve market: no strong favorite-longshot bias.

### Next Recommended Experiment

1. Residual diagnostics — where does the incumbent fail systematically?
2. DVOA/EPA features if available.
3. Expand Elo K > 48 if needed.
