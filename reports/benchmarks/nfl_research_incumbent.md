# NFL Research Incumbent

*Last updated: 2026-06-23*

## Current Champion

| Attribute | Value |
|-----------|-------|
| **Model** | Exponential-decay margin-aware rolling-origin Elo + Platt calibration |
| **K-factor** | 36 |
| **HFA** | 40 |
| **Preseason regression** | 0.20 |
| **MOV type** | `capped_linear` |
| **MOV scale** | 0.05 |
| **MOV cap** | 2.0 |
| **Decay half-life** | 32 games |
| **Selection method** | Rolling-origin 3-fold validation |
| **Avg validation log loss** | 0.6321 |
| **2025 holdout log loss** | **0.6298** |
| **2025 holdout Brier** | 0.2197 |
| **2025 holdout AUC** | 0.7024 |
| **2025 holdout accuracy** | 0.6558 |
| **Report** | `reports/experiments/decayed_elo.md` |
| **Selection date** | 2026-06-23 |

## Runner-Up Models

| Model | Validation LL | Holdout LL | Notes |
|-------|--------------|------------|-------|
| MOV Elo (K=36) + Platt | 0.6363 | 0.6373 | Previous incumbent |
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

## Market Benchmark

Market (no-vig closing moneyline) beats the incumbent as a diagnostic benchmark
but is not a production champion candidate (market timing is near-kickoff, not
pregame). See `reports/experiments/market_benchmark.md`.

| Model | Holdout LL |
|-------|-----------|
| Market (no-vig) | **0.6090** |
| Spread→prob | **0.6092** |
| Elo + Market (logit) | 0.6119 |
| **Previous incumbent (MOV Elo + Platt)** | **0.6373** |
| **New incumbent (Decayed Elo + Platt)** | **0.6298** |

## Promotion Rules

1. A challenger must beat **0.6298** holdout log loss to become the new incumbent.
2. Selection must use average rolling validation log loss only.
3. 2025 holdout must remain untouched during model selection.
4. Every feature must be pregame-safe and explainable.
5. Do not promote based on AUC or ROI alone.
