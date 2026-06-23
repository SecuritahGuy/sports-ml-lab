# Scheduling/Rest Feature Experiment

*Adding pregame scheduling and rest features on top of the Elo+Platt incumbent.*

## Feature Definitions

| Feature | Source | Description |
|---------|--------|-------------|
| `home_short_week` | `home_rest` | Home on short rest (≤6d) |
| `away_short_week` | `away_rest` | Away on short rest (≤6d) |
| `home_off_bye` | `home_rest` | Home off bye (≥13d rest) |
| `away_off_bye` | `away_rest` | Away off bye (≥13d rest) |
| `thursday_flag` | `weekday` | Thursday game |
| `monday_flag` | `weekday` | Monday game |
| `is_neutral` | `location` | Neutral site |
| `is_international` | `stadium_id` | Outside US |
| `home_consecutive_road` | chronological | Home consecutive road/neutral games |
| `away_consecutive_road` | chronological | Away consecutive road games |

## Incumbent Elo Params

- K=40, HFA=40, preseason regression=0.25

## Data Split

| Split | Seasons | Description |
|-------|---------|-------------|
| Fold 1 | Train: [2021], Val: 2022 | Rolling-origin selection |
| Fold 2 | Train: [2021, 2022], Val: 2023 | Rolling-origin selection |
| Fold 3 | Train: [2021, 2022, 2023], Val: 2024 | Rolling-origin selection |
| Holdout | 2025 | One-shot final evaluation |

## Leakage Prevention

- Elo features computed chronologically across all seasons.
- Scheduling features computed chronologically in a single pass.
- Consecutive road game counts reflect the streak **before** each game.
- Rest days (`home_rest`, `away_rest`) are provided by nflreadpy as pregame
  data (days since each team's prior game).
- Day-of-week flags are determined solely from the `weekday` column
  (known at schedule release).
- International flag is based on `stadium_id` (known at schedule release).
- Rolling-origin folds ensure no 2025 data touches model selection.

## Average Validation Metrics Across Folds

| Model | Avg Val LL | Fold1 LL | Fold2 LL | Fold3 LL |
|-------|------------|----------|----------|----------|
| Raw Elo | 0.6363 | 0.6394 | 0.6636 | 0.6060 |
| Platt (incumbent) | 0.6408 | 0.6492 | 0.6611 | 0.6119 |
| Incumbent + Scheduling | 0.6599 | 0.6793 | 0.6683 | 0.6320 |
| Raw Elo + Scheduling | 0.6596 | 0.6781 | 0.6691 | 0.6316 |
| Scheduling only | 0.7055 | 0.7096 | 0.6955 | 0.7114 |

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Raw Elo | 0.6409 | 0.2247 | 0.6558 | 0.6861 |
| Platt (incumbent) | 0.6395 | 0.2240 | 0.6522 | 0.6861 |
| Incumbent + Scheduling | 0.6401 | 0.2244 | 0.6341 | 0.6835 |
| Raw Elo + Scheduling | 0.6408 | 0.2248 | 0.6341 | 0.6828 |
| Scheduling only | 0.6922 | 0.2495 | 0.5326 | 0.5003 |

## Incumbent (Platt, Holdout)

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

## Incumbent + Scheduling (Holdout)

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.2, 0.3) | 30 | 0.2511 | 0.3 | 0.0489 |
| [0.3, 0.4) | 36 | 0.3559 | 0.4444 | 0.0886 |
| [0.4, 0.5) | 39 | 0.4508 | 0.359 | 0.0919 |
| [0.5, 0.6) | 55 | 0.5499 | 0.5273 | 0.0226 |
| [0.6, 0.7) | 53 | 0.6438 | 0.6038 | 0.04 |
| [0.7, 0.8) | 52 | 0.7447 | 0.7308 | 0.0139 |
| [0.8, 0.9) | 11 | 0.8272 | 0.9091 | 0.0818 |

## Raw Elo + Scheduling (Holdout)

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 3 | 0.1898 | 0.3333 | 0.1435 |
| [0.2, 0.3) | 25 | 0.2489 | 0.32 | 0.0711 |
| [0.3, 0.4) | 33 | 0.3501 | 0.3939 | 0.0439 |
| [0.4, 0.5) | 44 | 0.4478 | 0.3864 | 0.0614 |
| [0.5, 0.6) | 58 | 0.5504 | 0.5517 | 0.0014 |
| [0.6, 0.7) | 50 | 0.6421 | 0.58 | 0.0621 |
| [0.7, 0.8) | 47 | 0.7452 | 0.7234 | 0.0218 |
| [0.8, 0.9) | 16 | 0.8314 | 0.875 | 0.0436 |

## Recommendation

⚠️ **Platt incumbent remains the research incumbent.**

Average validation log loss: Platt=0.6408, best challenger=Raw Elo + Scheduling (0.6596).  No challenger achieved meaningfully lower validation log loss.

Scheduling features did not provide a material improvement over the Elo+Platt baseline on this dataset.

### Next Recommended Experiment

1. Add weather features (temp, wind, precipitation).
2. Test a GradientBoosting model with Elo + scheduling + weather.
3. Explore advanced team metrics (DVOA, EPA).
