# Market Benchmark and Elo-vs-Market Diagnostics

*Determining whether MOV Elo+Platt has independent signal relative to market-implied probabilities.*

## Market Data Audit

| Column | Coverage | Type | Source |
|--------|----------|------|--------|
| `home_moneyline` | 100% | int32 | nflreadpy |
| `away_moneyline` | 100% | int32 | nflreadpy |
| `spread_line` | 100% | float64 | nflreadpy |
| `home_spread_odds` | 100% | int32 | nflreadpy |
| `away_spread_odds` | 100% | int32 | nflreadpy |
| `total_line` | 100% | float64 | nflreadpy |
| `over_odds` | 100% | int32 | nflreadpy |
| `under_odds` | 100% | int32 | nflreadpy |
| opening/home_moneyline | 0% | — | Not available in nflreadpy |
| opening/away_moneyline | 0% | — | Not available in nflreadpy |

### Data Quality Notes

- All closing moneylines and spreads are 100% available.
- **No opening line data available.** nflreadpy provides only closing market data.  Opening lines require a separate source.
- Spread odds vary (not always -110), indicating real market variation.
- All odds are American format.

## Methodology

### Moneyline Conversion

- Negative odds (favorite): `prob = -odds / (-odds + 100)`
- Positive odds (underdog): `prob = 100 / (odds + 100)`
- Vig removed via multiplicative normalization:
  `fair_home_prob = home_implied / (home_implied + away_implied)`

### Spread→Implied Probability

- A logistic regression is fit per fold mapping spread line → home win prob.
- Fitted on training data only (no validation/holdout access).
- Tests whether spread line alone matches moneyline information.

### Blend Methods

- **Logistic blend**: Logistic regression on Elo prob + market prob.
- **Average blend**: Simple `(Elo + Market) / 2`.
- Neither blend is a production champion candidate.

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

| Split | Seasons | Role |
|-------|---------|------|
| Fold 1 | Train: [2021], Val: 2022 | Selection |
| Fold 2 | Train: [2021, 2022], Val: 2023 | Selection |
| Fold 3 | Train: [2021, 2022, 2023], Val: 2024 | Selection |
| Holdout | 2021–2024 → 2025 | Final eval |

## Models Compared

| Model | Description | Timing |
|-------|-------------|--------|
| **Raw Elo** | MOV Elo probability (no calibration) | Pregame |
| **Platt (incumbent)** | MOV Elo + Platt scaling | Pregame |
| **Market (no-vig)** | De-vigged moneyline implied prob | Closing line |
| **Market + Platt** | Platt-calibrated market | Closing line |
| **Elo + Market (logit)** | Logistic blend | Pregame + closing |
| **Elo + Market (avg)** | Simple average blend | Pregame + closing |
| **Spread→prob** | Logistic from spread line | Closing line |
> **Timing note:** Closing lines are near-kickoff and may reflect late-breaking information. Elo is purely pregame (previous games only). These are not directly comparable as production strategies.

## Average Validation Log Loss

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Raw Elo | 0.6345 | 0.6347 | 0.6670 | 0.6019 |
| Platt (incumbent) | 0.6363 | 0.6438 | 0.6564 | 0.6088 |
| Market (no-vig) | 0.6052 | 0.6042 | 0.6258 | 0.5858 |
| Market + Platt | 0.6088 | 0.6147 | 0.6268 | 0.5848 |
| Elo + Market (logit) | 0.6189 | 0.6359 | 0.6234 | 0.5975 |
| Elo + Market (avg) | 0.6127 | 0.6130 | 0.6398 | 0.5853 |
| Spread→prob | 0.6076 | 0.6134 | 0.6224 | 0.5870 |

## 2025 Holdout Comparison

| Model | Hold LL | Brier | Acc | AUC |
|-------|---------|-------|-----|-----|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Raw Elo | 0.6464 | 0.2258 | 0.6522 | 0.6907 |
| Platt (incumbent) | 0.6373 | 0.2230 | 0.6522 | 0.6907 |
| Market (no-vig) | 0.6090 | 0.2119 | 0.6558 | 0.7199 |
| Market + Platt | 0.6127 | 0.2131 | 0.6594 | 0.7199 |
| Elo + Market (logit) | 0.6119 | 0.2128 | 0.6522 | 0.7204 |
| Elo + Market (avg) | 0.6203 | 0.2163 | 0.6703 | 0.7107 |
| Spread→prob | 0.6092 | 0.2122 | 0.6558 | 0.7173 |

## Residual Correlation

| Comparison | r | p-value |
|------------|---|--------|
| Elo residual vs Market residual | 0.9768 | 0.0000 |
| Elo prob vs Market prob | 0.8780 | 0.0000 |
| Elo edge (Elo − Market) range | [-0.3007, 0.2924] | — |

### Elo Edge Analysis

Elo edge = Elo probability − Market probability. Positive means Elo is more confident in a home win than the market.

| Bucket | Count | Mean Actual Win Rate | Mean Elo LL | Mean Market LL |
|--------|-------|---------------------|-------------|----------------|
| Elo higher than mkt (0.05 to 0.15) | 63 | 0.651 | 0.5850 | 0.5921 |
| Elo lower than mkt (-0.15 to -0.05) | 62 | 0.516 | 0.6667 | 0.6422 |
| Elo much higher than mkt (> 0.15) | 20 | 0.300 | 0.9046 | 0.6420 |
| Elo much lower than mkt (< -0.15) | 14 | 0.786 | 0.9155 | 0.5833 |
| Near agreement (-0.05 to 0.05) | 117 | 0.496 | 0.5923 | 0.5981 |

## Calibration Deciles

### Platt (Incumbent, Holdout)

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

### Market (No-Vig, Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 10 | 0.1689 | 0.1000 | 0.0689 |
| [0.2, 0.3) | 24 | 0.2477 | 0.2500 | 0.0023 |
| [0.3, 0.4) | 33 | 0.3451 | 0.4242 | 0.0791 |
| [0.4, 0.5) | 46 | 0.4424 | 0.4130 | 0.0293 |
| [0.5, 0.6) | 46 | 0.5532 | 0.5652 | 0.0121 |
| [0.6, 0.7) | 48 | 0.6451 | 0.5417 | 0.1034 |
| [0.7, 0.8) | 40 | 0.7491 | 0.7250 | 0.0241 |
| [0.8, 0.9) | 26 | 0.8574 | 0.9231 | 0.0657 |
| [0.9, 1.0) | 3 | 0.9054 | 1.0000 | 0.0946 |

### Elo + Market (Logit, Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 13 | 0.1714 | 0.1538 | 0.0175 |
| [0.2, 0.3) | 33 | 0.2512 | 0.3030 | 0.0518 |
| [0.3, 0.4) | 36 | 0.3532 | 0.3889 | 0.0356 |
| [0.4, 0.5) | 34 | 0.4453 | 0.4706 | 0.0252 |
| [0.5, 0.6) | 39 | 0.5530 | 0.5128 | 0.0402 |
| [0.6, 0.7) | 46 | 0.6423 | 0.6087 | 0.0336 |
| [0.7, 0.8) | 45 | 0.7476 | 0.6667 | 0.0810 |
| [0.8, 0.9) | 30 | 0.8410 | 0.9333 | 0.0923 |

## Subset Analysis (2025 Holdout)

| Subset | N | Elo LL | Market LL | EM Logit LL |
|--------|---|--------|-----------|-------------|
| Big away fav (spread < -7) | 19 | 0.5931 | 0.4929 | 0.4947 |
| Big home fav (spread > 7) | 45 | 0.4598 | 0.4270 | 0.4382 |
| Close game (|spread| ≤ 3) | 99 | 0.7041 | 0.6795 | 0.6793 |
| Early season (W1-4) | 61 | 0.5679 | 0.5640 | 0.5637 |
| High confidence (>0.9) | 9 | 0.3390 | 0.2942 | 0.3186 |
| Home favorite | 168 | 0.6219 | 0.5992 | 0.6031 |
| Home underdog | 108 | 0.6844 | 0.6244 | 0.6257 |
| Late season (W5+) | 215 | 0.6686 | 0.6218 | 0.6256 |
| Low confidence (<=0.6) | 162 | 0.6736 | 0.6462 | 0.6499 |
| QB changed (home) | 24 | 0.8309 | 0.6662 | 0.6725 |
| QB stable (home) | 252 | 0.6288 | 0.6036 | 0.6062 |

### QB-Change Deep Dive

**On QB-change games:**

| Metric | Elo | Market | EM Logit |
|--------|-----|--------|----------|
| Log Loss | 0.8309 | 0.6662 | 0.6725 |
| Brier | 0.3064 | 0.2400 | 0.2423 |
| Accuracy | 0.5417 | 0.5417 | 0.5417 |
| AUC | 0.4545 | 0.6224 | 0.6434 |

Market beats Elo on QB-change games by 0.1647 log loss. The market prices in QB-change information that Elo misses.

### Favorite/Underdog Buckets

| Bucket | N | Elo LL | Market LL |
|--------|---|--------|-----------|
| Home favorite | 168 | 0.6219 | 0.5992 |
| Home underdog | 108 | 0.6844 | 0.6244 |
| Big home fav (spread > 7) | 45 | 0.4598 | 0.4270 |
| Big away fav (spread < -7) | 19 | 0.5931 | 0.4929 |
| Close game (|spread| ≤ 3) | 99 | 0.7041 | 0.6795 |

## Recommendation

✅ **Market (no-vig) beats the incumbent on holdout.**

Holdout log loss 0.6090 vs incumbent 0.6373.

> ⚠️ Market data (closing lines) reflects near-kickoff information, not purely pregame conditions. This comparison is diagnostic, not a direct apples-to-apples comparison of modeling strategies.

⚠️ **Elo does not add independent information beyond market odds.**
Elo + Market (logit, hold LL=0.6119) does not beat Market alone (0.6090).

### Favorite-Longshot Bias

⚠️ Platt calibration does not improve market (hold LL 0.6127 vs 0.6090). No strong favorite-longshot bias.

### QB-Change Recommendation

Since market beats the incumbent, and QB-change is the #1 failure mode, recommend QB-change market-delta as a feature: market_prob − elo_prob at QB-change games.

### Next Recommended Experiment

1. QB-change market-delta feature: `market_prob − elo_prob` at QB-change games.
2. Test if market odds at QB-change games alone explain the gap.
3. Stacked model: Elo predicts residual of market-only model.

## Appendix: Elo vs Market by Season

| Season | N | Elo LL | Market LL | EM Logit LL |
|--------|---|--------|-----------|-------------|
| 2021 | 280 | 0.6789 | 0.6223 | 0.6432 |
| 2022 | 275 | 0.6347 | 0.6042 | 0.6130 |
| 2023 | 279 | 0.6670 | 0.6258 | 0.6398 |
| 2024 | 278 | 0.6019 | 0.5858 | 0.5853 |
| 2025 | 276 | 0.6464 | 0.6090 | 0.6203 |

