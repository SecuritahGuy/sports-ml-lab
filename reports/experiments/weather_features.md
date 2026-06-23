# Weather Features Experiment

*Adding pregame weather features on top of MOV Elo+Platt.*

## Weather Data Audit

| Column | Type | Nulls | Coverage | Source |
|--------|------|-------|----------|--------|
| `weather_temp` | float64 | 1424/1424 | 0.0% | Meteostat Daily |
| `weather_tmin` | float64 | 59/1424 | 95.9% | Meteostat Daily |
| `weather_tmax` | float64 | 59/1424 | 95.9% | Meteostat Daily |
| `weather_humidity` | float64 | 1424/1424 | 0.0% | Meteostat Daily |
| `weather_precip` | float64 | 267/1424 | 81.2% | Meteostat Daily |
| `weather_wind_speed` | float64 | 59/1424 | 95.9% | Meteostat Daily |
| `weather_pressure` | float64 | 67/1424 | 95.3% | Meteostat Daily |
| `weather_cloud_cover` | float64 | 1424/1424 | 0.0% | Meteostat Daily |

Total games: 1424.

### Missingness by Season

| Season | weather_tmin | weather_wind_speed | weather_precip |
|--------|-------------|-------------------|----------------|
| 2021 | 33/285 | 33/285 | 92/285 |
| 2022 | 9/284 | 9/284 | 63/284 |
| 2023 | 8/285 | 8/285 | 60/285 |
| 2024 | 8/285 | 8/285 | 46/285 |
| 2025 | 1/285 | 1/285 | 6/285 |

### Missingness by Roof Type

| Roof | Count | weather_tmin missing | weather_wind_speed missing |
|------|-------|---------------------|--------------------------|
| outdoors | 947 | 43/947 | 43/947 |
| dome | 261 | 7/261 | 7/261 |
| closed | 184 | 8/184 | 8/184 |
| open | 32 | 1/32 | 1/32 |

## Feature Definitions

All weather features are pregame-safe.

| Feature | Source | Description |
|---------|--------|-------------|
| `temperature_f` | avg(weather_tmin, weather_tmax) °C→°F | Approximate game temperature |
| `wind_mph` | weather_wind_speed, km/h→mph | Wind speed |
| `precipitation_flag` | weather_precip > 0 | Any precipitation |
| `cold_flag` | temperature_f ≤ 32°F | Freezing or below |
| `very_cold_flag` | temperature_f ≤ 20°F | Extremely cold |
| `hot_flag` | temperature_f ≥ 85°F | Hot conditions |
| `windy_flag` | wind_mph ≥ 15 | Breezy/windy |
| `very_windy_flag` | wind_mph ≥ 20 | Strong wind |
| `bad_weather_flag` | cold OR windy OR precip | Combined adverse weather |
| `outdoor_game_flag` | roof ∈ {outdoors, open} | Game is outdoors |
| `is_dome` | roof ∈ {dome, closed} | Game is in dome/indoor |
| `weather_missing_flag` | tmin or wind_speed null | Weather data unavailable |
| `temp_missing_flag` | weather_tmin null | Temperature unavailable |
| `wind_missing_flag` | weather_wind_speed null | Wind speed unavailable |

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

- Weather data is daily historical data from Meteostat.
- Temperature is approximated as the average of daily min/max
  — this is a pregame-safe approximation.
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
| MOV Elo + Weather | 0.6445 | 0.6554 | 0.6655 | 0.6125 |
| Weather only | 0.6941 | 0.6947 | 0.6954 | 0.6923 |
| Outdoor MOV+Weather | 0.6546 | 0.6785 | 0.6822 | 0.6031 |

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6373 | 0.2230 | 0.6522 | 0.6907 |
| MOV Elo + Weather | 0.6439 | 0.2258 | 0.6486 | 0.6803 |
| Weather only | 0.6973 | 0.2518 | 0.5362 | 0.5031 |

## Subset Analysis (2025 Holdout)

| Subset | N | Platt Hold LL | Raw Elo Hold LL |
|--------|---|---------------|----------------|
| All games | 276 | 0.6373 | 0.6464 |
| Outdoor games | 187 | 0.6373 | 0.6461 |
| Cold games (≤32°F) | 26 | 0.6373 | 0.5777 |
| Windy games (≥15 mph) | 15 | 0.6373 | 0.6521 |
| Bad weather | 90 | 0.6373 | 0.6359 |

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
| [0.1, 0.2) | 3 | 0.1867 | 0.3333 | 0.1466 |
| [0.2, 0.3) | 28 | 0.2522 | 0.3929 | 0.1407 |
| [0.3, 0.4) | 29 | 0.358 | 0.2759 | 0.0821 |
| [0.4, 0.5) | 43 | 0.4456 | 0.3721 | 0.0735 |
| [0.5, 0.6) | 60 | 0.5511 | 0.5833 | 0.0322 |
| [0.6, 0.7) | 50 | 0.6534 | 0.62 | 0.0334 |
| [0.7, 0.8) | 44 | 0.7493 | 0.7045 | 0.0447 |
| [0.8, 0.9) | 19 | 0.8237 | 0.7895 | 0.0343 |

## Recommendation

⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**

No weather-augmented model beat the incumbent on holdout. Closest: MOV Elo + Weather (val LL=0.6445, hold LL=0.6439) vs incumbent hold LL=0.6373.

Weather features did not meaningfully improve over MOV Elo + Platt on this dataset (2021–2025).

### Next Recommended Experiment

1. Test GradientBoosting or XGBoost with Elo + available features.
2. Explore DVOA/EPA as model features if available.
3. Consider advanced team metrics (injury reports, OL/DL rankings).
