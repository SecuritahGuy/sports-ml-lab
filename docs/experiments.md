# Experiments

All 36 experiments grouped by outcome.

---

### Promoted / Superseded

- **tuned_elo** (superseded) — val 0.6500, holdout 0.6616 — [elo_tuning_calibration.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/elo_tuning_calibration.md)
- **rolling_origin_elo** (superseded) — val 0.6363, holdout 0.6395 — [rolling_origin_elo_validation.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/rolling_origin_elo_validation.md)
- **margin_aware_elo** (superseded) — val 0.6363, holdout 0.6373 — [margin_aware_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/margin_aware_elo.md)
- **decayed_elo** (promoted) — val 0.6321, holdout 0.6298 — [decayed_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/decayed_elo.md)
- **season_regression** (promoted) — val 0.6315, holdout 0.6285 — [season_regression.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/season_regression.md)
- **combined_features** (promoted) — val 0.6334, holdout 0.6262 — [combined_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/combined_features.md)
- **frozen_qb_overlay_foldsafe_v3** (promoted) — val 0.6305, holdout 0.6200 — [frozen_qb_overlay_foldsafe.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/frozen_qb_overlay_foldsafe.md)

---

### Rejected

- **identity_logistic** (rejected) — val 0.6800, holdout 0.6900 — —
- **team_strength_logistic** (rejected) — val 0.6700, holdout 0.6700 — —
- **scheduling_rest** (rejected) — val 0.6599, holdout 0.6401 — [schedule_rest_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/schedule_rest_features.md)
- **qb_features** (rejected) — val 0.6436, holdout 0.6459 — [qb_starter_change_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_starter_change_features.md)
- **weather_features** (rejected) — val 0.6445, holdout 0.6439 — [weather_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/weather_features.md)
- **epa_features** (rejected) — val 0.6654, holdout 0.6495 — [epa_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/epa_features.md)
- **confidence_calibration** (rejected) — val 0.6374, holdout 0.6373 — [confidence_calibration.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/confidence_calibration.md)
- **expressive_models** (rejected) — val 0.6361, holdout 0.6638 — [expressive_models.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/expressive_models.md)
- **team_hfa** (rejected) — val 0.6355, holdout 0.6263 — [team_hfa.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/team_hfa.md)
- **residual_blending** (rejected) — val 0.6368, holdout 0.6303 — [residual_blending.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/residual_blending.md)
- **coach_season_regression** (rejected) — val 0.6309, holdout 0.6286 — [coach_season_regression.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/coach_season_regression.md)
- **autogluon** (rejected) — val 0.6956, holdout 0.6404 — [autogluon.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/autogluon.md)
- **injury_features** (rejected) — val 0.6406, holdout 0.6514 — [injury_features.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/injury_features.md)
- **optuna_elo_search** (rejected) — val 0.6342, holdout 0.6318 — [optuna_elo_search.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/optuna_elo_search.md)
- **qb_injury_flag** (rejected) — val 0.6464, holdout 0.6255 — [qb_injury_flag.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_injury_flag.md)
- **glicko_rating** (rejected) — val 0.6513, holdout 0.7013 — [glicko_rating.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/glicko_rating.md)
- **home_away_elo** (rejected) — val 0.6622, holdout 0.6634 — [home_away_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/home_away_elo.md)
- **team_stats** (rejected) — val 0.6541, holdout 0.6415 — [team_stats.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/team_stats.md)
- **comprehensive_efficiency** (rejected) — val 0.6368, holdout 0.6313 — [comprehensive_efficiency.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/comprehensive_efficiency.md)
- **qb_adjusted_elo_v0** (rejected) — val 0.6338, holdout 0.6299 — [qb_adjusted_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_adjusted_elo.md)
- **gated_qb_adjusted_elo_v1** (rejected) — val 0.6341, holdout 0.6255 — [gated_qb_adjusted_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/gated_qb_adjusted_elo.md)

---

### Diagnostic

- **market_benchmark** (diagnostic) — val 0.6052, holdout 0.6090 — [market_benchmark.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/market_benchmark.md)
- **residual_diagnostics** (diagnostic) — val —, holdout 0.6373 — [residual_diagnostics.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/residual_diagnostics.md)
- **od_elo** (diagnostic) — val 0.6376, holdout 0.6258 — [od_elo.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/od_elo.md)
- **qb_market_delta** (diagnostic) — val 0.6052, holdout 0.6090 — [qb_market_delta.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_market_delta.md)
- **feature_selection** (diagnostic) — val 0.6334, holdout 0.6314 — [feature_selection.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/feature_selection.md)
- **optuna_feature_selection** (diagnostic) — val 0.6334, holdout 0.6347 — [optuna_feature_selection.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/optuna_feature_selection.md)
- **frozen_qb_overlay_v2** (diagnostic) — val 0.6238, holdout 0.6200 — [frozen_qb_overlay.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/frozen_qb_overlay.md)

---

### Market-Aware

- **qb_market_delta** (market-aware diagnostic) — val 0.6050, holdout 0.6083 — [qb_market_delta.md](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments/qb_market_delta.md)

---

### Artifact / Productization

These experiments were about registry, prediction artifacts, and productization:

- **benchmark_registry** — Created `nfl_research_incumbent.md`, `benchmark_history.md`, `leaderboard.csv`
- **predict_incumbent** — Created `incumbent_predictions.csv`, holdout file, prediction cards
- **weekly_report** — Weekly game report generation
- **build_dashboard** — This GitHub Pages dashboard

---

*Source: [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/leaderboard.csv`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/leaderboard.csv) and [`reports/experiments/`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/experiments)*
