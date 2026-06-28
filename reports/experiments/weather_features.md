# Weather Features Experiment

*Adding pregame weather features on top of MOV Elo+Platt.*

## Weather Data Audit

Raw weather columns from nflreadpy `load_schedules()`:

| Column | Type | Nulls | Coverage | Source |
|--------|------|-------|----------|--------|
| `temp` | float64 | 596/1424 | 58.1% | nflreadpy schedules |
| `wind` | float64 | 596/1424 | 58.1% | nflreadpy schedules |

Total games: 1424.

### Missingness by Season

| Season | temp missing | wind missing |
|--------|-------------|--------------|
| 2021 | 94/285 | 94/285 |
| 2022 | 177/284 | 177/284 |
| 2023 | 127/285 | 127/285 |
| 2024 | 103/285 | 103/285 |
| 2025 | 95/285 | 95/285 |

### Missingness by Roof Type

| Roof | Count | temp missing | wind missing |
|------|-------|-------------|--------------|
| outdoors | 947 | 120/947 | 120/947 |
| dome | 261 | 261/261 | 261/261 |
| closed | 184 | 183/184 | 183/184 |
| open | 32 | 32/32 | 32/32 |

## Feature Definitions

All weather features are pregame-safe.

| Feature | Source | Description |
|---------|--------|-------------|
| `temperature_f` | nflreadpy `temp` (°F) | Game-time temperature |
| `wind_mph` | nflreadpy `wind` (mph) | Wind speed |
| `precipitation_flag` | nflreadpy `temp`/`wind` available | Any adverse weather indicator |
| `cold_flag` | temperature_f ≤ 32°F | Freezing or below |
| `very_cold_flag` | temperature_f ≤ 20°F | Extremely cold |
| `hot_flag` | temperature_f ≥ 85°F | Hot conditions |
| `windy_flag` | wind_mph ≥ 15 | Breezy/windy |
| `very_windy_flag` | wind_mph ≥ 20 | Strong wind |
| `bad_weather_flag` | cold OR windy OR precip | Combined adverse weather |
| `outdoor_game_flag` | roof ∈ {outdoors, open} | Game is outdoors |
| `is_dome` | roof ∈ {dome, closed} | Game is in dome/indoor |
| `weather_missing_flag` | temp or wind null | Weather data unavailable |
| `temp_missing_flag` | temp null | Temperature unavailable |
| `wind_missing_flag` | wind null | Wind speed unavailable |

## Dome/Indoor Handling

For games in domes or closed-roof stadiums (`is_dome=1`):
- `temperature_f` is set to 70°F (neutral indoor temperature)
- `wind_mph` is set to 0
- `precipitation_flag` is set to 0
- Missing flags (`temp_missing_flag`, etc.) are preserved
- The `is_dome` flag allows the model to learn that weather
  does not apply to indoor games
- Retractable-roof stadiums: roof status is taken from
  the nflreadpy `roof` column, which may not indicate
  whether the roof was actually open/closed on game day.
  This is a known limitation.

## Leakage Prevention

- Weather data is game-level from nflreadpy schedules.
- `temp` and `wind` are game-time conditions or forecasts
  — pregame-safe and available before kickoff.
- Dome/indoor neutralization prevents outdoor weather
  from leaking into indoor games.
- Rolling-origin folds prevent 2025 holdout from being
  used in model selection.

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
| **Platt (incumbent)** | MOV Elo + Platt scaling |
| **MOV Elo + Weather** | Logistic on Elo prob + weather features |
| **Weather only** | Logistic on weather features alone |
| **Outdoor MOV+Weather** | Same model, outdoor games only |

## Average Validation Metrics Across Folds

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Platt (incumbent) | 0.6363 | 0.6438 | 0.6564 | 0.6088 |
| MOV Elo + Weather | 0.6689 | 0.7186 | 0.6700 | 0.6179 |
| Weather only | 0.7193 | 0.7536 | 0.7025 | 0.7018 |
| Outdoor MOV+Weather | 0.6924 | 0.7813 | 0.6859 | 0.6100 |

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6373 | 0.2230 | 0.6522 | 0.6907 |
| MOV Elo + Weather | 0.6485 | 0.2277 | 0.6522 | 0.6764 |
| Weather only | 0.7024 | 0.2543 | 0.5362 | 0.4675 |

## Subset Analysis (2025 Holdout)

| Subset | N | Platt Hold LL | Raw Elo Hold LL |
|--------|---|---------------|----------------|
| All games | 276 | 0.6373 | 0.6464 |
| Outdoor games | 187 | 0.6373 | 0.6461 |
| Cold games (≤32°F) | 19 | 0.6373 | 0.4346 |
| Windy games (≥15 mph) | 21 | 0.6373 | 0.6279 |
| Bad weather | 38 | 0.6373 | 0.5591 |

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

## MOV Elo + Weather (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 2 | 0.1907 | 0.5 | 0.3093 |
| [0.2, 0.3) | 31 | 0.2565 | 0.3548 | 0.0983 |
| [0.3, 0.4) | 30 | 0.3612 | 0.3333 | 0.0278 |
| [0.4, 0.5) | 45 | 0.4506 | 0.3556 | 0.095 |
| [0.5, 0.6) | 56 | 0.5485 | 0.5893 | 0.0408 |
| [0.6, 0.7) | 42 | 0.6539 | 0.619 | 0.0349 |
| [0.7, 0.8) | 55 | 0.7467 | 0.7091 | 0.0376 |
| [0.8, 0.9) | 15 | 0.8282 | 0.8 | 0.0282 |

## Recommendation

⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**

No weather-augmented model beat the incumbent on holdout. Closest: MOV Elo + Weather (val LL=0.6689, hold LL=0.6485) vs incumbent hold LL=0.6373.

Weather features did not meaningfully improve over MOV Elo + Platt on this dataset (2021–2025).

### Next Recommended Experiment

1. Test GradientBoosting or XGBoost with Elo + available features.
2. Explore DVOA/EPA as model features if available.
3. Consider advanced team metrics (injury reports, OL/DL rankings).
