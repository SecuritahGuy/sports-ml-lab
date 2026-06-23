# Team Strength Logistic (Elo + rolling)

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

## Included Features (25)

| Feature | Coefficient |
|---------|-------------|
| season | 0.1311 |
| week | 0.1102 |
| home_elo_pre | 0.0109 |
| away_elo_pre | -0.1813 |
| elo_diff | 0.1385 |
| home_rolling_win_pct | -0.0585 |
| away_rolling_win_pct | -0.0236 |
| rolling_win_pct_diff | -0.0252 |
| home_rolling_point_diff | 0.1395 |
| away_rolling_point_diff | -0.0277 |
| rolling_point_diff_diff | 0.121 |
| home_rolling_points_for | 0.173 |
| away_rolling_points_for | -0.1287 |
| home_rolling_points_against | 0.0025 |
| away_rolling_points_against | -0.1053 |
| home_rest | 0.066 |
| away_rest | -0.0025 |
| rest_diff | 0.0616 |
| div_game | 0.0319 |
| is_dome | 0.1271 |
| is_neutral | 0.0 |
| game_type_enc | 0.092 |
| weekday_enc | -0.0247 |
| roof_enc | 0.3053 |
| surface_enc | 0.119 |

## Train Metrics

| Metric | Value |
|--------|-------|
| Log loss | 0.6429 |
| Brier score | 0.2256 |
| Accuracy | 0.6319 |
| ROC AUC | 0.6764 |

### Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 11 | 0.1781 | 0.2727 | 0.0946 |
| [0.2, 0.3) | 26 | 0.2714 | 0.3462 | 0.0747 |
| [0.3, 0.4) | 92 | 0.3572 | 0.3587 | 0.0015 |
| [0.4, 0.5) | 184 | 0.4546 | 0.4239 | 0.0307 |
| [0.5, 0.6) | 204 | 0.5516 | 0.5098 | 0.0418 |
| [0.6, 0.7) | 171 | 0.6466 | 0.7193 | 0.0727 |
| [0.7, 0.8) | 113 | 0.7424 | 0.7257 | 0.0167 |
| [0.8, 0.9) | 33 | 0.8324 | 0.8485 | 0.0161 |

## Validation Metrics

| Metric | Value |
|--------|-------|
| Log loss | 0.6477 |
| Brier score | 0.2288 |
| Accuracy | 0.6367 |
| ROC AUC | 0.6896 |

### Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.1, 0.2) | 2 | 0.1755 | 0.5 | 0.3245 |
| [0.2, 0.3) | 6 | 0.2545 | 0.3333 | 0.0789 |
| [0.3, 0.4) | 17 | 0.3413 | 0.2941 | 0.0472 |
| [0.4, 0.5) | 49 | 0.4612 | 0.3061 | 0.1551 |
| [0.5, 0.6) | 57 | 0.5522 | 0.5088 | 0.0434 |
| [0.6, 0.7) | 59 | 0.6541 | 0.5593 | 0.0947 |
| [0.7, 0.8) | 58 | 0.7562 | 0.6552 | 0.101 |
| [0.8, 0.9) | 21 | 0.8412 | 0.8571 | 0.016 |
| [0.9, 1.0) | 9 | 0.9075 | 0.8889 | 0.0186 |

## Holdout Metrics

| Metric | Value |
|--------|-------|
| Log loss | 0.6866 |
| Brier score | 0.2459 |
| Accuracy | 0.5688 |
| ROC AUC | 0.6531 |

### Calibration Buckets

| Bucket | Count | Mean Predicted | Mean Actual | Calibration Error |
|--------|-------|----------------|-------------|-------------------|
| [0.2, 0.3) | 4 | 0.2793 | 0.75 | 0.4707 |
| [0.3, 0.4) | 18 | 0.351 | 0.2778 | 0.0732 |
| [0.4, 0.5) | 33 | 0.4517 | 0.4545 | 0.0028 |
| [0.5, 0.6) | 47 | 0.5535 | 0.3617 | 0.1918 |
| [0.6, 0.7) | 57 | 0.6448 | 0.4912 | 0.1536 |
| [0.7, 0.8) | 63 | 0.7503 | 0.5873 | 0.163 |
| [0.8, 0.9) | 47 | 0.8447 | 0.766 | 0.0788 |
| [0.9, 1.0) | 7 | 0.9185 | 1.0 | 0.0815 |

## Comparison Baselines

| Baseline | Val LL | Val Brier | Val Acc | Val AUC | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|----------|-------------|----------|---------|---------|---------|------------|----------|----------|
| random | 0.6931 | 0.2500 | 0.5360 | 0.5000 | 0.6931 | 0.2500 | 0.5362 | 0.5000 |
| home_prior | 0.6910 | 0.2489 | 0.5360 | 0.5000 | 0.6910 | 0.2489 | 0.5362 | 0.5000 |
| elo_only | 0.6166 | 0.2134 | 0.6835 | 0.7284 | 0.6678 | 0.2363 | 0.6268 | 0.6486 |
| **Logistic (Team Strength Logistic (Elo + rolling))** | 0.6477 | 0.2288 | 0.6367 | 0.6896 | 0.6866 | 0.2459 | 0.5688 | 0.6531 |

## Calibration Notes

Validation: max decile calibration error 0.3245, mean decile error 0.0977
Holdout: max decile calibration error 0.4707, mean decile error 0.1519

## Leakage Check

✅ No suspiciously large coefficients detected.

## Recommendation

❌ **Do not champion.** Holdout log loss 0.6866 does not beat the best simple baseline (0.6678). Consider adding weather features, team-strength ratings with different K-factors, or switching to a more expressive model (GradientBoosting, RandomForest, AutoGluon).
