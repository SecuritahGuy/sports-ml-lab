# QB Starter/Change Features Experiment

*Adding pregame QB continuity and starter-change features on top of MOV Elo+Platt.*

## QB Data Audit

| Column | Type | Nulls | Unique Values | Source |
|--------|------|-------|---------------|--------|
| `home_qb_id` | str | 0 | 108 | nflreadpy via nfl_schedule |
| `away_qb_id` | str | 0 | 113 | nflreadpy via nfl_schedule |
| `home_qb_name` | str | 0 | 111 | nflreadpy via nfl_schedule |
| `away_qb_name` | str | 0 | 116 | nflreadpy via nfl_schedule |

Total games: 1424.  QB data is **complete** — no missing QB starters in 2021–2025.

## Feature Definitions

All features are computed from games **before** the current game.

| Feature | Description |
|---------|-------------|
| `home_qb_changed` / `away_qb_changed` | 1 if QB differs from prior game |
| `qb_change_diff` | home_changed − away_changed |
| `home_qb_starts_this_season_pre` / `away_qb_starts_this_season_pre` | QB starts this season before this game |
| `qb_starts_diff` | home_starts − away_starts |
| `home_qb_team_starts_pre` / `away_qb_team_starts_pre` | QB career starts for this team (2021+) |
| `home_qb_win_pct_pre` / `away_qb_win_pct_pre` | QB win rate with this team this season (prior) |
| `qb_win_pct_diff` | home_win_pct − away_win_pct |
| `home_games_since_qb_change` / `away_games_since_qb_change` | Consecutive prior games same QB |
| `games_since_qb_change_diff` | home − away |
| `home_new_qb_flag` / `away_new_qb_flag` | 1 if QB has zero prior starts for this team ever |
| `new_qb_diff` | home_new − away_new |
| `home_qb_missing_flag` / `away_qb_missing_flag` | 1 if QB id is null (not observed) |

## Missingness Summary

No missing values in QB feature columns.

## Leakage Prevention

- QB features computed in a single chronological pass.
- For each game, QB features use only data from **prior** games.
- The current game result does not affect its own feature values.
- Season boundaries reset: first game of each season has no `qb_changed` flag.
- QB team starts (`qb_team_starts_pre`) persist across seasons (career count).
- Rolling-origin folds prevent 2025 holdout from touching model selection.

## Incumbent MOV Elo Params

| Parameter | Value |
|-----------|-------|
| K-factor | 36 |
| Home-field advantage | 40 |
| Preseason regression | 0.2 |
| MOV type | capped_linear |
| MOV scale | 0.05 |
| MOV cap | 2.0 |
| Calibration | Platt scaling |

## Data Split

| Split | Seasons | Description |
|-------|---------|-------------|
| Fold 1 | Train: [2021], Val: 2022 | Rolling-origin selection |
| Fold 2 | Train: [2021, 2022], Val: 2023 | Rolling-origin selection |
| Fold 3 | Train: [2021, 2022, 2023], Val: 2024 | Rolling-origin selection |
| Holdout | 2025 | One-shot final evaluation |

## Models Compared

| Model | Description |
|-------|-------------|
| **Platt (incumbent)** | MOV Elo + Platt scaling (K=36, HFA=40, reg=0.2, capped_linear) |
| **MOV Elo + QB features** | Logistic regression on Elo prob + QB features |
| **QB features only** | Logistic regression on QB features alone |
| **QB identity (OHE)** | Logistic regression on one-hot encoded QB IDs (experimental) |

## Average Validation Metrics Across Folds

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Platt (incumbent) | 0.6363 | 0.6438 | 0.6564 | 0.6088 |
| MOV Elo + QB features | 0.6436 | 0.6432 | 0.7052 | 0.5823 |
| QB features only | 0.6666 | 0.6583 | 0.7369 | 0.6047 |
| QB identity (OHE) | 1.0658 | 1.1049 | 1.1325 | 0.9599 |

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6373 | 0.2230 | 0.6522 | 0.6907 |
| MOV Elo + QB features | 0.6459 | 0.2256 | 0.6486 | 0.6906 |
| QB features only | 0.6615 | 0.2322 | 0.6486 | 0.6702 |
| QB identity (OHE) | 0.8717 | 0.2861 | 0.5543 | 0.5565 |

## Platt (Incumbent, Holdout)

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

## MOV Elo + QB Features (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.0, 0.1) | 2 | 0.0916 | 0.5 | 0.4084 |
| [0.1, 0.2) | 17 | 0.1568 | 0.2941 | 0.1373 |
| [0.2, 0.3) | 35 | 0.2439 | 0.3143 | 0.0704 |
| [0.3, 0.4) | 36 | 0.3477 | 0.3611 | 0.0135 |
| [0.4, 0.5) | 35 | 0.4581 | 0.4857 | 0.0276 |
| [0.5, 0.6) | 30 | 0.545 | 0.5667 | 0.0217 |
| [0.6, 0.7) | 44 | 0.6545 | 0.6364 | 0.0182 |
| [0.7, 0.8) | 40 | 0.7519 | 0.7 | 0.0519 |
| [0.8, 0.9) | 30 | 0.8472 | 0.7333 | 0.1139 |
| [0.9, 1.0) | 7 | 0.9293 | 0.8571 | 0.0721 |

## Recommendation

⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**

No QB-augmented model beat the incumbent on holdout. Closest: MOV Elo + QB features (val LL=0.6436, hold LL=0.6459) vs incumbent hold LL=0.6373.

QB starter/change features did not meaningfully improve over MOV Elo + Platt on this dataset (2021–2025).

### Next Recommended Experiment

1. Add weather features (temp, wind, precipitation).
2. Test GradientBoosting or XGBoost with Elo + available features.
3. Explore DVOA/EPA as model features if available.
