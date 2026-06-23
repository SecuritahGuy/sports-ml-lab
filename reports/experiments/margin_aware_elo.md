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
| K-factor | [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80] |
| HFA | [30, 40] |
| Regression | [0.0, 0.2] |
| MOV types | ['none', 'log', 'sqrt', 'capped_log', 'capped_linear'] |
| MOV scale | [0.05, 0.1, 0.2] (for non-none) |
| MOV cap | [2.0, 3.0] (for capped) |

Total combinations searched: 1292

## Top 8 Configurations (by average validation log loss)

| Rank | K | HFA | Reg | MOV | Scale | Cap | Avg Val LL | Fold1 | Fold2 | Fold3 |
|------|---|-----|-----|-----|-------|-----|-----------|-------|-------|-------|
| 1 | 20 | 40 | 0.2 | capped_linear | 0.1 | 3.0 | 0.63238 | 0.63453 | 0.65784 | 0.60477 |
| 2 | 20 | 30 | 0.2 | capped_linear | 0.1 | 3.0 | 0.63298 | 0.63591 | 0.65895 | 0.60409 |
| 3 | 28 | 40 | 0.2 | capped_linear | 0.05 | 2.0 | 0.63298 | 0.63523 | 0.66022 | 0.60347 |
| 4 | 16 | 40 | 0.2 | capped_linear | 0.1 | 3.0 | 0.63302 | 0.63644 | 0.65453 | 0.60807 |
| 5 | 24 | 40 | 0.2 | capped_linear | 0.05 | 3.0 | 0.63302 | 0.63611 | 0.65712 | 0.60583 |
| 6 | 24 | 40 | 0.2 | capped_linear | 0.1 | 3.0 | 0.63308 | 0.63393 | 0.66202 | 0.60329 |
| 7 | 28 | 40 | 0.2 | capped_linear | 0.05 | 3.0 | 0.63312 | 0.63518 | 0.66008 | 0.6041 |
| 8 | 24 | 40 | 0.2 | capped_linear | 0.05 | 2.0 | 0.63315 | 0.63646 | 0.65745 | 0.60554 |

## Best Per MOV Type

| MOV | Best K | Best HFA | Best Reg | Scale | Cap | Avg Val LL |
|-----|--------|----------|----------|-------|-----|-----------|
| none | 40 | 40 | 0.2 | 0.0 | — | 0.6365 |
| log | 28 | 40 | 0.2 | 0.2 | — | 0.63464 |
| sqrt | 24 | 40 | 0.2 | 0.2 | — | 0.6339 |
| capped_log | 28 | 40 | 0.2 | 0.2 | 2.0 | 0.63464 |
| capped_linear | 20 | 40 | 0.2 | 0.1 | 3.0 | 0.63238 |

## Best Configuration (selected by avg val LL across folds)

- **K=20, HFA=40, reg=0.2**
- **MOV**: type=capped_linear, scale=0.1, cap=3.0
- Average validation log loss: 0.63238
  - Fold 1 (val 2022): 0.63453
  - Fold 2 (val 2023): 0.65784
  - Fold 3 (val 2024): 0.60477
- Holdout (2025) log loss: 0.6490

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
| Incumbent (K=36) | 0.6345 | 0.6347 | 0.6670 | 0.6019 |
| MOV-best raw Elo | 0.6324 | 0.6345 | 0.6578 | 0.6048 |
| MOV-best + Platt | 0.6357 | 0.6439 | 0.6543 | 0.6090 |
| MOV-best + Isotonic | 0.7558 | 0.6405 | 0.9022 | 0.7246 |
| MOV-best Minimal Logistic | 0.6350 | 0.6422 | 0.6539 | 0.6089 |

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Incumbent (Platt, K=40) | 0.6373 | 0.2230 | 0.6522 | 0.6907 |
| MOV-best raw Elo | 0.6490 | 0.2276 | 0.6413 | 0.6812 |
| MOV-best + Platt | 0.6438 | 0.2258 | 0.6377 | 0.6812 |
| MOV-best + Isotonic | 0.7676 | 0.2283 | 0.6341 | 0.6794 |
| MOV-best Minimal Logistic | 0.6483 | 0.2279 | 0.6522 | 0.6748 |

## MOV-Best Raw Elo (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 10 | 0.1574 | 0.4 | 0.2426 |
| [0.2, 0.3) | 23 | 0.2332 | 0.3478 | 0.1146 |
| [0.3, 0.4) | 28 | 0.3536 | 0.3571 | 0.0035 |
| [0.4, 0.5) | 50 | 0.4465 | 0.38 | 0.0665 |
| [0.5, 0.6) | 45 | 0.5468 | 0.5556 | 0.0087 |
| [0.6, 0.7) | 48 | 0.645 | 0.625 | 0.02 |
| [0.7, 0.8) | 44 | 0.75 | 0.6591 | 0.0909 |
| [0.8, 0.9) | 25 | 0.8561 | 0.8 | 0.0561 |
| [0.9, 1.0) | 3 | 0.9115 | 1.0 | 0.0885 |

## MOV-Best + Platt (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 5 | 0.1828 | 0.4 | 0.2172 |
| [0.2, 0.3) | 28 | 0.2464 | 0.3571 | 0.1107 |
| [0.3, 0.4) | 29 | 0.3554 | 0.3448 | 0.0106 |
| [0.4, 0.5) | 52 | 0.4464 | 0.4038 | 0.0426 |
| [0.5, 0.6) | 46 | 0.55 | 0.5435 | 0.0065 |
| [0.6, 0.7) | 48 | 0.6447 | 0.6042 | 0.0405 |
| [0.7, 0.8) | 49 | 0.74 | 0.7143 | 0.0257 |
| [0.8, 0.9) | 19 | 0.8185 | 0.8421 | 0.0236 |

## Recommendation

⚠️ **Incumbent (Platt-calibrated rolling-origin Elo) remains the research incumbent.**

No margin-aware configuration beat the incumbent on holdout.  Closest: MOV-best + Platt (val LL=0.6357, hold LL=0.6438) vs incumbent hold LL=0.6373.

MOV multipliers did not meaningfully improve the Elo rating signal on this NFL dataset (2021–2025).

### Next Recommended Experiment

1. Add weather features (temp, wind, precipitation).
2. Test GradientBoosting or XGBoost with Elo + weather.
3. Explore advanced team metrics (DVOA/EPA) as model features.
