# Constrained Expressive Models Experiment

Comparing constrained tree-based and logistic models on a curated feature set
built around the incumbent MOV Elo probability signal.

## Curated Feature Set

| Feature | Source | Rationale |
|---------|--------|----------|
| `elo_prob` | MOV Elo incumbent | Core signal — home win probability |
| `elo_logit` | logit(elo_prob) | Linearize probability for logistic models |
| `elo_diff` | Elo ratings | Home minus away pregame rating |
| `home_short_week`, `away_short_week` | Rest diff ≤6 | Short-rest scheduling disadvantage |
| `home_off_bye`, `away_off_bye` | Rest diff ≥13 | Extra-rest scheduling advantage |
| `thursday_flag`, `monday_flag` | Weekday | Primetime scheduling effects |
| `home_consecutive_road`, `away_consecutive_road` | Location history | Travel fatigue |
| `is_international` | Stadium location | International travel |
| `home_qb_changed`, `away_qb_changed` | QB tracking | QB continuity disruption |
| `qb_starts_diff` | QB starts this season | Experience gap |
| `qb_win_pct_diff` | QB win pct | Winning experience gap |
| `games_since_qb_change_diff` | QB change history | Stability gap |
| `new_qb_diff` | First start on this team | Novelty gap |
| `cold_flag` | weather_tmin/tmax ≤32°F | Cold weather |
| `windy_flag` | wind ≥15 mph | Windy conditions |
| `bad_weather_flag` | Cold OR windy OR precip | Combined adverse weather |
| `outdoor_game_flag` | Roof ∈ {outdoors, open} | Outdoor venue |
| `is_dome` | Roof ∈ {dome, closed} | Indoor venue |
| `weather_missing_flag` | Weather data null | Missing weather indicator |
| `week_norm` | Week / max(week) | Season timing (0–1) |
| `rest_diff` | home_rest − away_rest | Rest advantage |
| `div_game` | Divisional game flag | Familiarity/rivalry |

## Leakage Prevention

- All features are pregame-safe (known before kickoff).
- Rolling-origin folds prevent 2025 from influencing model selection.
- QB change features computed chronologically from prior games only.
- Scheduling features computed chronologically.
- Weather features dome-neutralized; missing values imputed with dataset median.
- No target, score, or result columns in feature set.
- No raw team identity, QB identity OHE, or stadium identity.

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

| Model | Type | Grid Size |
|-------|------|-----------|
| Platt (incumbent) | Platt-scaled MOV Elo | N/A |
| LogisticRegression | Linear on curated features | N/A |
| HistGradientBoosting | Constrained boosting | 576 combos |
| GradientBoosting | Constrained boosting | 144 combos |
| RandomForest | Diagnostic only | 36 combos |

## Model Grids

### HistGradientBoosting
| Parameter | Values |
|-----------|--------|
| max_leaf_nodes | 4, 8, 12, 16 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| max_iter | 50, 100, 200 |
| min_samples_leaf | 20, 40, 60 |
| l2_regularization | 0.0, 0.1, 0.5, 1.0 |

### GradientBoosting
| Parameter | Values |
|-----------|--------|
| max_leaf_nodes | 4, 8, 12, 16 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| n_estimators | 50, 100, 200 |
| min_samples_leaf | 20, 40, 60 |
| subsample | 0.8 |

### RandomForest (diagnostic)
| Parameter | Values |
|-----------|--------|
| max_leaf_nodes | 4, 8, 12, 16 |
| n_estimators | 50, 100, 200 |
| min_samples_leaf | 20, 40, 60 |

## Average Validation Log Loss

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Platt (incumbent) | 0.6363 | 0.6438 | 0.6564 | 0.6088 |
| LogisticRegression | 0.6744 | 0.7237 | 0.6894 | 0.6100 |
| HistGradientBoosting | 0.6361 | 0.6446 | 0.6562 | 0.6075 |
| GradientBoosting | 0.6366 | 0.6451 | 0.6523 | 0.6123 |
| RandomForest | 0.6329 | 0.6366 | 0.6473 | 0.6146 |

## Best Model Selected: HistGradientBoosting

Selected by lowest average validation log loss among challengers.

### Per-Fold Best Parameters

**Fold 1** (train=[2021], val=2022):
- max_leaf_nodes: 4
- learning_rate: 0.01
- max_iter: 200
- min_samples_leaf: 20
- l2_regularization: 0.0
**Fold 2** (train=[2021, 2022], val=2023):
- max_leaf_nodes: 4
- learning_rate: 0.01
- max_iter: 100
- min_samples_leaf: 60
- l2_regularization: 1.0
**Fold 3** (train=[2021, 2022, 2023], val=2024):
- max_leaf_nodes: 4
- learning_rate: 0.1
- max_iter: 50
- min_samples_leaf: 20
- l2_regularization: 1.0

## 2025 Holdout Comparison

| Model | Holdout LL | Brier | Acc | AUC |
|-------|-----------|-------|-----|-----|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| **Platt (incumbent) [target: 0.6373]** | 0.6373 | 0.2230 | 0.6522 | — |
| LogisticRegression | 0.6422 | 0.2252 | 0.6522 | 0.6842 |
| HistGradientBoosting | 0.6638 | 0.2356 | 0.6087 | 0.6554 |
| GradientBoosting | 0.6610 | 0.2340 | 0.6087 | 0.6533 |
| RandomForest | 0.6456 | 0.2269 | 0.6377 | 0.6755 |

## Calibration

### HistGradientBoosting + Platt

Holdout log loss: **0.7091**

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.0, 0.1) | 1 | 0.0956 | 0.0 | 0.0956 |
| [0.1, 0.2) | 24 | 0.1631 | 0.4167 | 0.2536 |
| [0.2, 0.3) | 27 | 0.2488 | 0.2963 | 0.0475 |
| [0.3, 0.4) | 35 | 0.3451 | 0.4571 | 0.112 |
| [0.4, 0.5) | 33 | 0.4455 | 0.4848 | 0.0393 |
| [0.5, 0.6) | 21 | 0.548 | 0.4762 | 0.0718 |
| [0.6, 0.7) | 25 | 0.6461 | 0.56 | 0.0861 |
| [0.7, 0.8) | 30 | 0.7556 | 0.5333 | 0.2223 |
| [0.8, 0.9) | 49 | 0.8558 | 0.6327 | 0.2231 |
| [0.9, 1.0) | 31 | 0.918 | 0.871 | 0.047 |

### HistGradientBoosting + Isotonic

Holdout log loss: **1.0851**

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.0, 0.1) | 4 | 0.0 | 0.75 | 0.75 |
| [0.1, 0.2) | 9 | 0.1154 | 0.3333 | 0.218 |
| [0.2, 0.3) | 55 | 0.2421 | 0.3455 | 0.1034 |
| [0.3, 0.4) | 6 | 0.3558 | 0.3333 | 0.0224 |
| [0.4, 0.5) | 26 | 0.4507 | 0.5769 | 0.1262 |
| [0.5, 0.6) | 33 | 0.5288 | 0.4545 | 0.0742 |
| [0.6, 0.7) | 38 | 0.6459 | 0.5526 | 0.0932 |
| [0.7, 0.8) | 56 | 0.7482 | 0.5714 | 0.1768 |
| [0.8, 0.9) | 10 | 0.865 | 0.7 | 0.165 |
| [0.9, 1.0) | 28 | 0.963 | 0.7143 | 0.2487 |

## Feature Importance

Feature importance not available for HistGradientBoosting.
Logistic regression coefficients and permutation importance:


## Recommendation

⚠️ **Challenger wins validation but not holdout.**

**HistGradientBoosting** (avg val LL=0.6361) won rolling-origin selection but holdout (0.7091) did not beat incumbent (0.6373). Keeping MOV Elo + Platt as incumbent.


### Next Recommended Experiment

1. Market-baseline comparison (moneyline implied probabilities).
2. Residual diagnostics — where does the incumbent fail systematically?
3. DVOA/EPA features if available.
