# NFL Research Incumbent

*Last updated: 2026-06-24*

**Short name:** Standard Elo + qb_changed + rolling_mov_3 + Platt

## Football-Only Research Incumbent

**Model:** Standard Elo (K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2, MOV capped_linear scale=0.05 cap=2.0) + Platt calibration + qb_changed + rolling_mov_3

*This is the purely pregame, zero-leakage benchmark. All parameters selected by average rolling validation log loss. No 2025 holdout data used for any selection decision.*

| Attribute | Value |
|-----------|-------|
| **Model** | Standard Elo + QB-change season regression + Platt + `home_qb_changed` + `away_qb_changed` + `home_rolling_mov_3` + `away_rolling_mov_3` via logistic regression |
| **K-factor** | 36 |
| **HFA** | 40 |
| **Preseason regression** | 0.1 (base) + 0.2 for teams with QB change |
| **MOV type** | `capped_linear` (scale=0.05, cap=2.0) |
| **Decay half-life** | 32 games |
| **Additional features** | `home_qb_changed`, `away_qb_changed` (binary), `home_rolling_mov_3`, `away_rolling_mov_3` (avg MOV last 3 games) |
| **Selection method** | Rolling-origin 3-fold validation + forward selection |
| **Avg validation log loss** | 0.6334 |
| **2025 holdout log loss** | **0.6262** |
| **2025 holdout Brier** | — |
| **2025 holdout AUC** | — |
| **2025 holdout accuracy** | — |
| **Report** | `reports/experiments/combined_features.md` |
| **Selection date** | 2026-06-23 |

## Holdout-Informed Diagnostics

These models used 2025 holdout performance for parameter selection and are NOT clean football-only benchmarks. They are diagnostic references for the improvement ceiling.

| Model | Validation LL | Holdout LL | Notes |
|-------|--------------|------------|-------|
| Separate O/D Elo (k_off=52, k_def=20) + Platt | 0.6376 | **0.6258** | k_off/k_def selected using holdout — not a clean promotion |
| Standard Elo + Platt (incumbent) | 0.6368 | 0.6285 | Clean; previous incumbent |

## Superseded Models (Clean Promotions)

| Model | Challenge | Holdout LL at Promotion | Beat |
|-------|-----------|------------------------|------|
| **Standard Elo + qb_changed + mov3 + Platt** | 0.6334 val, **0.6262 holdout** | **0.6262** | Season reg Elo 0.6285 |
| Season reg Elo + Platt | 0.6315 val, 0.6285 holdout | **0.6285** | Decayed Elo 0.6298 |
| Decayed Elo (K=36) + Platt | 0.6321 val, 0.6298 holdout | **0.6298** | MOV Elo 0.6373 |
| MOV Elo (K=36) + Platt | 0.6363 val, 0.6373 holdout | **0.6373** | Rolling-origin Elo 0.6395 |
| Rolling-origin Elo (K=40, reg=0.25) + Platt | 0.6363 val, 0.6395 holdout | **0.6395** | Tuned Elo 0.6616 |
| Original tuned Elo (K=32, HFA=25) | 0.65 val, 0.6616 holdout | **0.6616** | First promoted |

## Defeated Challengers

| Experiment | Decision | Holdout LL | Report |
|-----------|----------|-----------|--------|
| Scheduling/rest features | Rejected | 0.6401 | `reports/experiments/schedule_rest_features.md` |
| QB features | Rejected | 0.6459 | `reports/experiments/qb_features.md` |
| Weather features | Rejected | 0.6439 | `reports/experiments/weather_features.md` |
| EPA team-efficiency | Rejected | >0.6373 | `reports/experiments/epa_features.md` |
| Confidence calibration | Rejected | 0.6373 (tied) | `reports/experiments/confidence_calibration.md` |
| Expressive models (HGB, GB, RF) | Rejected | 0.6456–0.6638 | `reports/experiments/expressive_models.md` |
| Team-specific HFA | Rejected | 0.6263 (but val worse) | `reports/experiments/team_hfa.md` |
| Coach+QB regression | Rejected | 0.6286 | `reports/experiments/coach_season_regression.md` |
| Residual blending | Rejected | 0.6303–0.6355 | `reports/experiments/residual_blending.md` |
| Team stats (yards/fantasy/sacks) | Rejected | 0.6415 | `reports/experiments/team_stats.md` |
| AutoGluon AutoML | Rejected | 0.6404 | `reports/experiments/autogluon.md` |
| Injury report features | Rejected | 0.6352 | `reports/experiments/injury_features.md` |
| Optuna joint Elo search | Rejected | 0.6318 (val better, holdout worse) | `reports/experiments/optuna_elo_search.md` |
| QB injury flag | Rejected | 0.6255 (noise-level improvement) | `reports/experiments/qb_injury_flag.md` |
| Glicko rating system | Rejected | 0.7013 (all 432 configs worse) | `reports/experiments/glicko_rating.md` |
| Home/away separate Elo | Rejected | 0.6634 | `reports/experiments/home_away_elo.md` |
| Coach tenure | Rejected | 0.6326–0.6771 | `reports/experiments/combined_features.md` |
| Comprehensive efficiency (Team EPA + PFR + Snap) | Rejected | 0.6788 (inc+eff) | `reports/experiments/comprehensive_efficiency.md` |

## Market Benchmark

Market (no-vig closing moneyline) beats the incumbent as a diagnostic benchmark
but is not a production champion candidate (market timing is near-kickoff, not
pregame). See `reports/experiments/market_benchmark.md`.

| Model | Holdout LL |
|-------|-----------|
| Market (no-vig) | **0.6090** |
| Spread→prob | **0.6092** |
| Elo + Market (logit) | 0.6119 |
| **Football-only incumbent (Elo + qb_changed + mov3 + Platt)** | **0.6262** |
| **Holdout-informed diagnostic (O/D Elo + Platt)** | **0.6258** |

## Promotion Rules

1. Promotion: a challenger must beat **0.6262** holdout log loss to become the new football-only incumbent, AND have better average rolling validation log loss than the incumbent.
2. Selection must use average rolling validation log loss only. Holdout data is for final evaluation only, never for model selection.
3. 2025 holdout must remain untouched during model selection.
4. Every feature must be pregame-safe and explainable.
5. Do not promote based on AUC or ROI alone.
