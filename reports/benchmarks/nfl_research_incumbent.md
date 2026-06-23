# NFL Research Incumbent

*Last updated: 2026-06-23*

## Current Champion

| Attribute | Value |
|-----------|-------|
| **Model** | Separate O/D Elo (k_off=52, k_def=20) + QB-change season regression + Platt calibration |
| **k_off** | 52 |
| **k_def** | 20 |
| **K-factor (baseline)** | 36 (overridden by k_off/k_def) |
| **HFA** | 40 |
| **Preseason regression** | 0.1 (base) + 0.2 for teams with QB change |
| **MOV type** | `capped_linear` |
| **MOV scale** | 0.05 |
| **MOV cap** | 2.0 |
| **Decay half-life** | 32 games |
| **Selection method** | Rolling-origin 3-fold validation (user override for holdout-leading split) |
| **Avg validation log loss** | 0.6376 |
| **2025 holdout log loss** | **0.6258** |
| **2025 holdout Brier** | 0.2179 |
| **2025 holdout AUC** | 0.7066 |
| **2025 holdout accuracy** | 0.6703 |
| **Report** | `reports/experiments/od_elo.md` |
| **Selection date** | 2026-06-23 |

## Runner-Up Models

| Model | Validation LL | Holdout LL | Notes |
|-------|--------------|------------|-------|
| Standard Elo (K=36, reg=0.1, decay=32, qb_bonus=0.2) + Platt | 0.6368 | 0.6285 | Previous incumbent |
| Decayed Elo (K=36) + Platt | 0.6321 | 0.6298 | Superseded by season-regression |
| MOV Elo (K=36) + Platt | 0.6363 | 0.6373 | Superseded by decayed Elo |
| Rolling-origin Elo (K=40, reg=0.25) + Platt | 0.6363 | 0.6395 | Superseded by MOV Elo |
| Original tuned Elo (K=32, HFA=25) | — | 0.6616 | Pre-MOV incumbent |

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

## Market Benchmark

Market (no-vig closing moneyline) beats the incumbent as a diagnostic benchmark
but is not a production champion candidate (market timing is near-kickoff, not
pregame). See `reports/experiments/market_benchmark.md`.

| Model | Holdout LL |
|-------|-----------|
| Market (no-vig) | **0.6090** |
| Spread→prob | **0.6092** |
| Elo + Market (logit) | 0.6119 |
| **Previous incumbent (Season Reg + Platt)** | **0.6285** |
| **New incumbent (O/D Elo + Platt)** | **0.6258** |

## Promotion Rules

1. A challenger must beat **0.6258** holdout log loss to become the new incumbent.
2. Selection should use average rolling validation log loss (user may override for strong holdout signal).
3. 2025 holdout must remain untouched during model selection.
4. Every feature must be pregame-safe and explainable.
5. Do not promote based on AUC or ROI alone.
