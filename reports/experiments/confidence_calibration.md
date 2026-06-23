# Confidence Calibration and Probability Shrinkage

*Testing post-processing methods to reduce overconfidence in MOV Elo+Platt probabilities.*

## Methods Tested

| Method | Variants | Description |
|--------|----------|-------------|
| Baseline (raw Elo) | — | Uncalibrated MOV Elo probability |
| Probability clipping | 5 thresholds (0.01–0.10) | Clip to [lo, hi] range |
| Temperature scaling | 6 temperatures (1.05–1.50) | Soften logit by dividing by T > 1 |
| Global shrinkage (p=0.5) | 5 strengths (0.02–0.15) | Shrink toward 0.5 |
| Global shrinkage (home prior) | 5 strengths | Shrink toward home win rate |
| High-confidence-only shrinkage | 3 thresholds × 5 strengths | Shrink only p ≤ lo or p ≥ hi |
| Early-season shrinkage | 5 strengths | Shrink weeks 1–4 only |

## Incumbent MOV Elo Params

| Parameter | Value |
|-----------|-------|
| K-factor | 36 |
| Home-field advantage | 40 |
| Preseason regression | 0.2 |
| MOV type | capped_linear |
| MOV scale | 0.05 |
| MOV cap | 2.0 |

## Rolling-Origin Selection

| Split | Seasons | Role |
|-------|---------|------|
| Fold 1 | Train: [2021], Val: 2022 | Selection |
| Fold 2 | Train: [2021, 2022], Val: 2023 | Selection |
| Fold 3 | Train: [2021, 2022, 2023], Val: 2024 | Selection |
| Holdout | 2025 | Final eval |

## Top 10 Methods (Avg Validation Log Loss)

| Method | Avg Val LL |
|--------|------------|
| shrink_home_prior_a=0.020_p=0.552 | 0.6018 |
| shrink_home_prior_a=0.050_p=0.552 | 0.6018 |
| shrink_home_prior_a=0.080_p=0.552 | 0.6021 |
| shrink_home_prior_a=0.100_p=0.552 | 0.6024 |
| shrink_home_prior_a=0.150_p=0.552 | 0.6037 |
| shrink_50_a=0.150 | 0.6322 |
| temperature_t=1.20 | 0.6323 |
| shrink_50_a=0.100 | 0.6324 |
| temperature_t=1.15 | 0.6324 |
| early_season_shrink_a=0.150 | 0.6324 |

## 2025 Holdout Comparison

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Raw Elo (no Platt) | 0.6464 | 0.2258 | 0.6522 | 0.6907 |
| **Best: shrink_home_prior_a=0.020_p=0.552** | 0.6448 | 0.2254 | 0.6522 | 0.6907 |
| MOV Elo + Platt (incumbent) | 0.6373 | — | — | — |

### All Clip Variants

| Clip | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|------|---------|------------|----------|----------|
| [clip, 0.01, 0.99] | 0.6464 | 0.2258 | 0.6522 | 0.6907 |
| [clip, 0.03, 0.97] | 0.6464 | 0.2258 | 0.6522 | 0.6907 |
| [clip, 0.05, 0.95] | 0.6464 | 0.2258 | 0.6522 | 0.6907 |
| [clip, 0.08, 0.92] | 0.6464 | 0.2258 | 0.6522 | 0.6907 |
| [clip, 0.1, 0.9] | 0.6464 | 0.2258 | 0.6522 | 0.6905 |

### All Temperature Variants

| Temp | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|------|---------|------------|----------|----------|
| T=1.05 | 0.6439 | 0.2251 | 0.6522 | 0.6907 |
| T=1.1 | 0.6419 | 0.2246 | 0.6522 | 0.6907 |
| T=1.15 | 0.6404 | 0.2241 | 0.6522 | 0.6907 |
| T=1.2 | 0.6393 | 0.2238 | 0.6522 | 0.6907 |
| T=1.3 | 0.6379 | 0.2233 | 0.6522 | 0.6907 |
| T=1.5 | 0.6373 | 0.2232 | 0.6522 | 0.6907 |

### All Global Shrinkage Variants

| Shrinkage | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-----------|---------|------------|----------|----------|
| shrink50_0.02 | 0.6448 | 0.2254 | 0.6522 | 0.6907 |
| shrink50_0.05 | 0.6427 | 0.2248 | 0.6522 | 0.6907 |
| shrink50_0.08 | 0.6410 | 0.2243 | 0.6522 | 0.6907 |
| shrink50_0.1 | 0.6401 | 0.2240 | 0.6522 | 0.6907 |
| shrink50_0.15 | 0.6384 | 0.2234 | 0.6522 | 0.6907 |
| shrink_hp_0.02 | 0.6448 | 0.2254 | 0.6522 | 0.6907 |
| shrink_hp_0.05 | 0.6428 | 0.2248 | 0.6522 | 0.6907 |
| shrink_hp_0.08 | 0.6411 | 0.2243 | 0.6522 | 0.6907 |
| shrink_hp_0.1 | 0.6402 | 0.2241 | 0.6522 | 0.6907 |
| shrink_hp_0.15 | 0.6385 | 0.2235 | 0.6486 | 0.6907 |

## Subset Analysis (2025 Holdout)

| Subset | N | Raw Elo | Best Method |
|--------|---|---------|-------------|
| Early season (W1-4) | 61 | 0.5679 | 0.5684 |
| High confidence (>0.9) | 6 | 0.0838 | 0.0918 |
| Late season (W5+) | 215 | 0.6686 | 0.6665 |
| Low confidence (<=0.6) | 162 | 0.6736 | 0.6720 |
| QB changed (home) | 24 | 0.8309 | 0.8250 |
| QB stable (home) | 252 | 0.6288 | 0.6276 |

## Calibration Deciles

### Raw Elo (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.0, 0.1) | 2 | 0.0932 | 0.5000 | 0.4068 |
| [0.1, 0.2) | 18 | 0.1712 | 0.4444 | 0.2733 |
| [0.2, 0.3) | 20 | 0.2426 | 0.1500 | 0.0926 |
| [0.3, 0.4) | 28 | 0.3640 | 0.3571 | 0.0069 |
| [0.4, 0.5) | 44 | 0.4437 | 0.4091 | 0.0346 |
| [0.5, 0.6) | 50 | 0.5545 | 0.5800 | 0.0255 |
| [0.6, 0.7) | 35 | 0.6452 | 0.6571 | 0.0119 |
| [0.7, 0.8) | 41 | 0.7484 | 0.6341 | 0.1143 |
| [0.8, 0.9) | 29 | 0.8511 | 0.7586 | 0.0925 |
| [0.9, 1.0) | 9 | 0.9144 | 0.8889 | 0.0256 |

### Best: shrink_home_prior_a=0.020_p=0.552 (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 16 | 0.1632 | 0.4375 | 0.2743 |
| [0.2, 0.3) | 23 | 0.2385 | 0.2174 | 0.0211 |
| [0.3, 0.4) | 26 | 0.3613 | 0.3462 | 0.0151 |
| [0.4, 0.5) | 47 | 0.4431 | 0.4043 | 0.0388 |
| [0.5, 0.6) | 50 | 0.5545 | 0.5800 | 0.0255 |
| [0.6, 0.7) | 39 | 0.6490 | 0.6410 | 0.0080 |
| [0.7, 0.8) | 40 | 0.7530 | 0.6250 | 0.1280 |
| [0.8, 0.9) | 29 | 0.8555 | 0.7931 | 0.0624 |
| [0.9, 1.0) | 6 | 0.9123 | 1.0000 | 0.0877 |

## Recommendation

⚠️ **MOV Elo + Platt remains the research incumbent.**

No calibration method beat the incumbent on holdout. Best: shrink_home_prior_a=0.020_p=0.552 (val LL=0.6018, hold LL=0.6448) vs incumbent (0.6373).

### High-Confidence Assessment

Raw Elo high-confidence (6 games): LL=0.0838
Best method high-confidence: LL=0.0918

### QB-Change Assessment

Raw Elo: QB-changed LL=0.8309 | QB-stable LL=0.6288 | gap=0.2021
Best: QB-changed LL=0.8250 | QB-stable LL=0.6276 | gap=0.1974

### Early vs Late Season

Raw Elo: Early LL=0.5679 | Late LL=0.6686
Best: Early LL=0.5684 | Late LL=0.6665

### Next Recommended Experiment

1. Test isotonic regression or Platt-only recalibration.
2. Ensemble methods combining multiple shrinkage variants.
3. Early-season feature enrichment to reduce initial uncertainty.
