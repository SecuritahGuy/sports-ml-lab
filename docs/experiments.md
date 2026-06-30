# Experiments

[Home](index) | [2026 Schedule](2026-schedule) | [Benchmarks](benchmarks) | [Predictions](predictions) | [Model Card](model-card) | **Experiments** | [Backtests](backtests)

---


All 39 experiments conducted during feature research (2026-06).


### Promoted / Superseded

| Experiment | Decision | Val LL | Holdout LL |
|-----------|----------|--------|------------|
| tuned_elo | superseded | 0.6500 | 0.6616 |
| rolling_origin_elo | superseded | 0.6363 | 0.6395 |
| margin_aware_elo | superseded | 0.6363 | 0.6373 |
| decayed_elo | promoted | 0.6321 | 0.6298 |
| season_regression | promoted | 0.6315 | 0.6285 |
| combined_features | promoted | 0.6334 | 0.6262 |
| frozen_qb_overlay_foldsafe_v3 | promoted | 0.6305 | 0.6200 |


### Rejected

| Experiment | Decision | Val LL | Holdout LL |
|-----------|----------|--------|------------|
| identity_logistic | rejected | 0.6800 | 0.6900 |
| team_strength_logistic | rejected | 0.6700 | 0.6700 |
| scheduling_rest | rejected | 0.6599 | 0.6401 |
| qb_features | rejected | 0.6436 | 0.6459 |
| weather_features | rejected | 0.6445 | 0.6439 |
| epa_features | rejected | 0.6654 | 0.6495 |
| confidence_calibration | rejected | 0.6374 | 0.6373 |
| expressive_models | rejected | 0.6361 | 0.6638 |
| team_hfa | rejected | 0.6355 | 0.6263 |
| residual_blending | rejected | 0.6368 | 0.6303 |
| coach_season_regression | rejected | 0.6309 | 0.6286 |
| autogluon | rejected | 0.6956 | 0.6404 |
| injury_features | rejected | 0.6406 | 0.6514 |
| optuna_elo_search | rejected | 0.6342 | 0.6318 |
| qb_injury_flag | rejected | 0.6464 | 0.6255 |
| glicko_rating | rejected | 0.6513 | 0.7013 |
| home_away_elo | rejected | 0.6622 | 0.6634 |
| team_stats | rejected | 0.6541 | 0.6415 |
| comprehensive_efficiency | rejected | 0.6368 | 0.6313 |
| qb_adjusted_elo_v0 | rejected | 0.6338 | 0.6299 |
| gated_qb_adjusted_elo_v1 | rejected | 0.6341 | 0.6255 |
| roster_overlay_foldsafe | rejected | 0.6341 | 0.6255 |
| qb_roster_interaction | rejected | 0.6305 | 0.6195 |
| expanded_elo_spine | rejected | 0.6299 | 0.6302 |


### Diagnostic Only

| Experiment | Decision | Val LL | Holdout LL |
|-----------|----------|--------|------------|
| market_benchmark | diagnostic | 0.6052 | 0.6090 |
| residual_diagnostics | diagnostic | — | 0.6373 |
| od_elo | diagnostic | 0.6376 | 0.6258 |
| qb_market_delta | diagnostic | 0.6052 | 0.6090 |
| feature_selection | diagnostic | 0.6334 | 0.6314 |
| optuna_feature_selection | diagnostic | 0.6334 | 0.6347 |
| frozen_qb_overlay_v2 | diagnostic | 0.6238 | 0.6200 |


---
*Full details: [`leaderboard.csv`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main//Users/tim/dev/sports-ml-lab/reports/benchmarks/leaderboard.csv)*