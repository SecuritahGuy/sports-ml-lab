# Baseline Logistic (identity features)

*Pregame-only NFL home-win baseline.*

## Data Split

| Split | Seasons | Rows |
|-------|---------|------|
| Train | [2021, 2022, 2023] | 834 |
| Validation | 2024 | 278 |
| Holdout | 2025 | 276 |

## Filtering Applied

- `model_eligible == True` (ties excluded)
- `is_neutral == False` (neutral-site games excluded)

## Leakage Prevention

- Elo features: ratings updated **after** computing pregame features for each game. Games processed chronologically.
- Rolling features: computed from games **before** the current game only. The current game's result is never included.
- All features are pregame-safe: no future information leaks into any row.

## Excluded Column Groups

| Reason | Columns |
|--------|---------|
| leakage (score/result/overtime) | away_score, home_score, result, total, overtime |
| market / odds | away_moneyline, home_moneyline, spread_line, away_spread_odds, home_spread_odds, total_line, under_odds, over_odds |
| weather (deferred) | weather_temp, weather_tmin, weather_tmax, weather_humidity, weather_precip, weather_wind_speed, weather_pressure, weather_cloud_cover |
| row identifiers | game_id, gameday, gametime, stadium, old_game_id, gsis, nfl_detail_id, pfr, pff, espn, ftn |
| raw string columns | away_team, home_team, away_qb_id, home_qb_id, away_qb_name, home_qb_name, away_coach, home_coach, referee, stadium_id, game_type, weekday, roof, surface, location |
| target / flags | home_win, model_eligible, is_tie, is_neutral |

## Included Features (19)

| Feature | Coefficient |
|---------|-------------|
| season | 0.1523 |
| week | 0.1264 |
| away_team_enc | -0.0735 |
| home_team_enc | -0.0367 |
| away_rest | -0.0133 |
| home_rest | 0.0698 |
| rest_diff | 0.0738 |
| div_game | 0.0172 |
| is_dome | 0.1157 |
| is_neutral | 0.0 |
| away_qb_id_enc | 0.0468 |
| home_qb_id_enc | -0.1679 |
| away_coach_enc | -0.0105 |
| home_coach_enc | -0.0386 |
| stadium_id_enc | -0.0767 |
| game_type_enc | 0.1104 |
| weekday_enc | 0.0007 |
| roof_enc | 0.3204 |
| surface_enc | 0.1382 |

## Train Metrics

| Metric | Value |
|--------|-------|
| Log loss | 0.6709 |
| Brier score | 0.2392 |
| Accuracy | 0.5935 |
| ROC AUC | 0.5999 |

### Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.2, 0.3) | 3 | 0.2926 | 0.0 | 0.2926 |
| [0.3, 0.4) | 43 | 0.3648 | 0.4186 | 0.0538 |
| [0.4, 0.5) | 173 | 0.4603 | 0.4277 | 0.0326 |
| [0.5, 0.6) | 380 | 0.551 | 0.5632 | 0.0122 |
| [0.6, 0.7) | 194 | 0.6404 | 0.6134 | 0.027 |
| [0.7, 0.8) | 40 | 0.7346 | 0.85 | 0.1154 |
| [0.8, 0.9) | 1 | 0.8057 | 1.0 | 0.1943 |

## Validation Metrics

| Metric | Value |
|--------|-------|
| Log loss | 0.7213 |
| Brier score | 0.2625 |
| Accuracy | 0.5252 |
| ROC AUC | 0.5222 |

### Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.3, 0.4) | 4 | 0.373 | 0.75 | 0.377 |
| [0.4, 0.5) | 17 | 0.4548 | 0.5294 | 0.0746 |
| [0.5, 0.6) | 86 | 0.56 | 0.4884 | 0.0717 |
| [0.6, 0.7) | 126 | 0.648 | 0.5635 | 0.0845 |
| [0.7, 0.8) | 41 | 0.7341 | 0.5122 | 0.2219 |
| [0.8, 0.9) | 4 | 0.8215 | 0.75 | 0.0715 |

## Holdout Metrics

| Metric | Value |
|--------|-------|
| Log loss | 0.7486 |
| Brier score | 0.2735 |
| Accuracy | 0.5326 |
| ROC AUC | 0.4900 |

### Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.3, 0.4) | 1 | 0.3952 | 0.0 | 0.3952 |
| [0.4, 0.5) | 10 | 0.4619 | 0.6 | 0.1381 |
| [0.5, 0.6) | 44 | 0.5679 | 0.5 | 0.0679 |
| [0.6, 0.7) | 140 | 0.6553 | 0.5571 | 0.0981 |
| [0.7, 0.8) | 73 | 0.7414 | 0.5205 | 0.2209 |
| [0.8, 0.9) | 8 | 0.8301 | 0.5 | 0.3301 |

## Comparison Baselines

| Baseline | Val LL | Val Brier | Val Acc | Val AUC | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|----------|-------------|----------|---------|---------|---------|------------|----------|----------|
| random | 0.6931 | 0.2500 | 0.5360 | 0.5000 | 0.6931 | 0.2500 | 0.5362 | 0.5000 |
| home_prior | 0.6910 | 0.2489 | 0.5360 | 0.5000 | 0.6910 | 0.2489 | 0.5362 | 0.5000 |
| **Logistic (Baseline Logistic (identity features))** | 0.7213 | 0.2625 | 0.5252 | 0.5222 | 0.7486 | 0.2735 | 0.5326 | 0.4900 |

## Calibration Notes

Validation: max decile calibration error 0.3770, mean decile error 0.1502
Holdout: max decile calibration error 0.3952, mean decile error 0.2084

## Leakage Check

✅ No suspiciously large coefficients detected.

## Recommendation

❌ **Do not champion.** Holdout log loss 0.7486 does not beat the best simple baseline (0.6910). Consider adding weather features, team-strength ratings with different K-factors, or switching to a more expressive model (GradientBoosting, RandomForest, AutoGluon).
