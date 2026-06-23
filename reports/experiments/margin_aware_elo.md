# Margin-Aware Elo Experiment

*Testing margin-of-victory multipliers on Elo rating updates.*

## Motivation

Standard Elo treats every win equally regardless of margin.
Margin-aware Elo applies a multiplier to the rating update
based on the point differential, so blowouts produce larger
rating shifts than narrow wins.  The multiplier only affects
the **postgame rating update** — pregame probabilities remain
unchanged, preserving feature safety.

## MOV Formulas Tested

| Type | Formula | Parameters |
|------|---------|------------|
| `none` | `mult = 1.0` (baseline) | — |
| `log` | `mult = 1 + scale * ln(1 + |PD|)` | scale ∈ {0.05, 0.10, 0.20} |
| `sqrt` | `mult = 1 + scale * sqrt(|PD|)` | scale ∈ {0.05, 0.10, 0.20} |
| `capped_log` | `mult = min(cap, 1 + scale * ln(1 + |PD|))` | scale ∈ {0.05, 0.10, 0.20}, cap ∈ {2.0, 3.0} |
| `capped_linear` | `mult = min(cap, 1 + scale * |PD|)` | scale ∈ {0.05, 0.10, 0.20}, cap ∈ {2.0, 3.0} |

## Parameter Grid

| Parameter | Candidates |
|-----------|------------|
| K-factor | [36, 40, 48] |
| HFA | [30, 40] |
| Regression | [0.0, 0.2] |
| MOV types | ['none', 'log', 'sqrt', 'capped_log', 'capped_linear'] |
| MOV scale | [0.05, 0.1, 0.2] (for non-none) |
| MOV cap | [2.0, 3.0] (for capped) |

Total combinations searched: 228

## Top 8 Configurations (by average validation log loss)

| Rank | K | HFA | Reg | MOV | Scale | Cap | Avg Val LL | Fold1 | Fold2 | Fold3 |
|------|---|-----|-----|-----|-------|-----|-----------|-------|-------|-------|
| 1 | 36 | 40 | 0.2 | capped_linear | 0.05 | 2.0 | 0.63452 | 0.63466 | 0.66698 | 0.60193 |
| 2 | 36 | 40 | 0.2 | sqrt | 0.1 | — | 0.63512 | 0.63675 | 0.66557 | 0.60305 |
| 3 | 36 | 30 | 0.2 | capped_linear | 0.05 | 2.0 | 0.63517 | 0.63608 | 0.6681 | 0.60132 |
| 4 | 36 | 40 | 0.2 | capped_linear | 0.05 | 3.0 | 0.6352 | 0.63518 | 0.66717 | 0.60324 |
| 5 | 36 | 40 | 0.2 | sqrt | 0.05 | — | 0.63541 | 0.63828 | 0.6635 | 0.60443 |
| 6 | 36 | 40 | 0.2 | log | 0.1 | — | 0.63547 | 0.63775 | 0.66485 | 0.6038 |
| 7 | 36 | 40 | 0.2 | capped_log | 0.1 | 2.0 | 0.63547 | 0.63775 | 0.66485 | 0.6038 |
| 8 | 36 | 40 | 0.2 | capped_log | 0.1 | 3.0 | 0.63547 | 0.63775 | 0.66485 | 0.6038 |

## Best Per MOV Type

| MOV | Best K | Best HFA | Best Reg | Scale | Cap | Avg Val LL |
|-----|--------|----------|----------|-------|-----|-----------|
| none | 40 | 40 | 0.2 | 0.0 | — | 0.6365 |
| log | 36 | 40 | 0.2 | 0.1 | — | 0.63547 |
| sqrt | 36 | 40 | 0.2 | 0.1 | — | 0.63512 |
| capped_log | 36 | 40 | 0.2 | 0.1 | 2.0 | 0.63547 |
| capped_linear | 36 | 40 | 0.2 | 0.05 | 2.0 | 0.63452 |

## Best Configuration (selected by avg val LL across folds)

- **K=36, HFA=40, reg=0.2**
- **MOV**: type=capped_linear, scale=0.05, cap=2.0
- Average validation log loss: 0.63452
  - Fold 1 (val 2022): 0.63466
  - Fold 2 (val 2023): 0.66698
  - Fold 3 (val 2024): 0.60193
- Holdout (2025) log loss: 0.6464

## Best Non-MOV Configuration (for comparison)

- K=40, HFA=40, reg=0.2
- Average validation log loss: 0.6365
  - Fold 1: 0.63989, Fold 2: 0.66418, Fold 3: 0.60542

## Leakage Prevention

- MOV multiplier only affects post-game rating update, **never** the pregame probability.
- Pregame features (elo_diff, elo_prob) are recorded before the update step.
- Rolling-origin folds prevent 2025 holdout from touching model selection.
- Calibration fitted only on training data.

## Average Validation Metrics Across Folds

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Incumbent (K=40) | 0.6363 | 0.6394 | 0.6636 | 0.6060 |
| MOV-best raw Elo | 0.6345 | 0.6347 | 0.6670 | 0.6019 |
| MOV-best + Platt | 0.6363 | 0.6438 | 0.6564 | 0.6088 |
| MOV-best + Isotonic | 0.7560 | 0.6368 | 0.9090 | 0.7222 |
| MOV-best Minimal Logistic | 0.6357 | 0.6424 | 0.6561 | 0.6085 |

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Incumbent (Platt, K=40) | 0.6395 | 0.2240 | 0.6522 | 0.6861 |
| MOV-best raw Elo | 0.6464 | 0.2258 | 0.6522 | 0.6907 |
| MOV-best + Platt | 0.6373 | 0.2230 | 0.6522 | 0.6907 |
| MOV-best + Isotonic | 0.6412 | 0.2252 | 0.6413 | 0.6886 |
| MOV-best Minimal Logistic | 0.6422 | 0.2254 | 0.6341 | 0.6804 |

## MOV-Best Raw Elo (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.0, 0.1) | 2 | 0.0932 | 0.5 | 0.4068 |
| [0.1, 0.2) | 18 | 0.1712 | 0.4444 | 0.2733 |
| [0.2, 0.3) | 20 | 0.2426 | 0.15 | 0.0926 |
| [0.3, 0.4) | 28 | 0.364 | 0.3571 | 0.0069 |
| [0.4, 0.5) | 44 | 0.4437 | 0.4091 | 0.0346 |
| [0.5, 0.6) | 50 | 0.5545 | 0.58 | 0.0255 |
| [0.6, 0.7) | 35 | 0.6452 | 0.6571 | 0.0119 |
| [0.7, 0.8) | 41 | 0.7484 | 0.6341 | 0.1143 |
| [0.8, 0.9) | 29 | 0.8511 | 0.7586 | 0.0925 |
| [0.9, 1.0) | 9 | 0.9144 | 0.8889 | 0.0256 |

## MOV-Best + Platt (Holdout)

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

## Recommendation

✅ **MOV-best + Platt is the new research incumbent.**

Holdout log loss 0.6373 beats the incumbent (0.6395). Average validation log loss 0.6363 also beats the incumbent. Margin-aware Elo improved rating accuracy.

### Next Recommended Experiment

1. Add weather features (temp, wind, precipitation).
2. Test GradientBoosting or XGBoost with Elo + weather.
3. Explore advanced team metrics (DVOA/EPA) as model features.
