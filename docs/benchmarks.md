# Benchmarks & Leaderboard

## Current Football-Only Incumbent

**Standard Elo + qb_changed + rolling_mov_3 + Platt**

- Version: v2.0.0
- Holdout log loss: **0.6262**
- Average validation log loss: 0.6334
- Feature set: qb_changed + rolling_mov_3
- Full details: [Model Card](model-card)

### Promotion Rules

1. A challenger must beat **0.6262** holdout log loss
   to become the new football-only incumbent.
2. The challenger must also have **better average rolling validation
   log loss** than the incumbent.
3. Selection must use average rolling validation log loss only.
4. 2025 holdout is for final evaluation only, never for model selection.
5. Every feature must be pregame-safe, explainable, and leakage-safe.
6. Do not promote based on AUC or ROI alone.

---

### Promoted / Superseded Models

These models were promoted as the research incumbent at some point:

| Experiment | Decision | Val LL | Holdout LL | Holdout AUC | Report |
| --- | --- | --- | --- | --- | --- |
| tuned_elo | superseded | 0.6500 | 0.6616 | 0.6800 | [elo_tuning_calibration.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/elo_tuning_calibration.md) |
| rolling_origin_elo | superseded | 0.6363 | 0.6395 | 0.6900 | [rolling_origin_elo_validation.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/rolling_origin_elo_validation.md) |
| margin_aware_elo | superseded | 0.6363 | 0.6373 | 0.6907 | [margin_aware_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/margin_aware_elo.md) |
| decayed_elo | promoted | 0.6321 | 0.6298 | 0.7024 | [decayed_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/decayed_elo.md) |
| season_regression | promoted | 0.6315 | 0.6285 | 0.7024 | [season_regression.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/season_regression.md) |
| combined_features | promoted | 0.6334 | 0.6262 | — | [combined_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/combined_features.md) |
| frozen_qb_overlay_foldsafe_v3 | promoted | 0.6305 | 0.6200 | 0.7098 | [frozen_qb_overlay_foldsafe.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/frozen_qb_overlay_foldsafe.md) |


### Rejected Challengers

These models failed to beat the incumbent:

| Experiment | Decision | Val LL | Holdout LL | Holdout AUC | Report |
| --- | --- | --- | --- | --- | --- |
| identity_logistic | rejected | 0.6800 | 0.6900 | — | — |
| team_strength_logistic | rejected | 0.6700 | 0.6700 | — | — |
| scheduling_rest | rejected | 0.6599 | 0.6401 | — | [schedule_rest_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/schedule_rest_features.md) |
| qb_features | rejected | 0.6436 | 0.6459 | — | [qb_starter_change_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_starter_change_features.md) |
| weather_features | rejected | 0.6445 | 0.6439 | — | [weather_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/weather_features.md) |
| epa_features | rejected | 0.6654 | 0.6495 | 0.6800 | [epa_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/epa_features.md) |
| confidence_calibration | rejected | 0.6374 | 0.6373 | 0.6907 | [confidence_calibration.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/confidence_calibration.md) |
| expressive_models | rejected | 0.6361 | 0.6638 | 0.6800 | [expressive_models.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/expressive_models.md) |
| team_hfa | rejected | 0.6355 | 0.6263 | 0.7063 | [team_hfa.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/team_hfa.md) |
| residual_blending | rejected | 0.6368 | 0.6303 | 0.6600 | [residual_blending.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/residual_blending.md) |
| coach_season_regression | rejected | 0.6309 | 0.6286 | 0.7010 | [coach_season_regression.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/coach_season_regression.md) |
| autogluon | rejected | 0.6956 | 0.6404 | 0.6848 | [autogluon.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/autogluon.md) |
| injury_features | rejected | 0.6406 | 0.6514 | — | [injury_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/injury_features.md) |
| optuna_elo_search | rejected | 0.6342 | 0.6318 | 0.7000 | [optuna_elo_search.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/optuna_elo_search.md) |
| qb_injury_flag | rejected | 0.6464 | 0.6255 | 0.7060 | [qb_injury_flag.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_injury_flag.md) |
| glicko_rating | rejected | 0.6513 | 0.7013 | 0.2520 | [glicko_rating.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/glicko_rating.md) |
| home_away_elo | rejected | 0.6622 | 0.6634 | — | [home_away_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/home_away_elo.md) |
| team_stats | rejected | 0.6541 | 0.6415 | 0.6730 | [team_stats.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/team_stats.md) |
| comprehensive_efficiency | rejected | 0.6368 | 0.6313 | 0.6895 | [comprehensive_efficiency.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/comprehensive_efficiency.md) |
| qb_adjusted_elo_v0 | rejected | 0.6338 | 0.6299 | 0.6972 | [qb_adjusted_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_adjusted_elo.md) |
| gated_qb_adjusted_elo_v1 | rejected | 0.6341 | 0.6255 | 0.7020 | [gated_qb_adjusted_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/gated_qb_adjusted_elo.md) |


### Diagnostics

These experiments produced diagnostic insights but were not promoted:

| Experiment | Decision | Val LL | Holdout LL | Holdout AUC | Report |
| --- | --- | --- | --- | --- | --- |
| market_benchmark | diagnostic | 0.6052 | 0.6090 | 0.7199 | [market_benchmark.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/market_benchmark.md) |
| residual_diagnostics | diagnostic | — | 0.6373 | 0.6907 | [residual_diagnostics.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/residual_diagnostics.md) |
| od_elo | diagnostic | 0.6376 | 0.6258 | 0.7066 | [od_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/od_elo.md) |
| qb_market_delta | diagnostic | 0.6052 | 0.6090 | 0.7199 | [qb_market_delta.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_market_delta.md) |
| feature_selection | diagnostic | 0.6334 | 0.6314 | — | [feature_selection.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/feature_selection.md) |
| optuna_feature_selection | diagnostic | 0.6334 | 0.6347 | — | [optuna_feature_selection.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/optuna_feature_selection.md) |
| frozen_qb_overlay_v2 | diagnostic | 0.6238 | 0.6200 | 0.7098 | [frozen_qb_overlay.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/frozen_qb_overlay.md) |


### Market-Aware Diagnostics

Market-relative benchmarks. Not football-only:

| Experiment | Decision | Val LL | Holdout LL | Holdout AUC | Report |
| --- | --- | --- | --- | --- | --- |
| qb_market_delta | market-aware diagnostic | 0.6050 | 0.6083 | 0.7205 | [qb_market_delta.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_market_delta.md) |


### Note on Market Benchmark

Market (no-vig closing moneyline) achieves holdout log loss 0.6090,
significantly better than the football-only incumbent (0.6262).
The market is the true performance ceiling for pregame NFL prediction.
The incumbent is a purely pregame, market-free benchmark.

---

*Source: [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/leaderboard.csv`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/leaderboard.csv) and [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/benchmark_history.md`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/benchmark_history.md)*
