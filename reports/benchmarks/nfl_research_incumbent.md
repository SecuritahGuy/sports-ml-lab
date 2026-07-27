# NFL Research Incumbent

*Last updated: 2026-07-26*

**Short name:** Standard Elo + qb_changed + rolling_mov_3 + FDR + DOBA + Chaos Rate + Platt

## Football-Only Research Incumbent

**Model:** Standard Elo (K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2) + rolling_mov_3 + qb_changed + StatSpace FDR (Fraud Detector Rating) + StatSpace DOBA (sustainable offensive efficiency) + StatSpace Chaos Rate (defensive disruption composite) + Platt calibration

*FDR is a team-season composite blending record strength, underlying quality (EPA, success rate, Elo), luck gap, close-game luck, turnover luck, and schedule suspicion. DOBA is a team-season composite blending offensive EPA/play, success rate, early-down efficiency, third/fourth-down efficiency, explosive rate, red zone efficiency, negative play rate, turnover rate, and dependency penalty. Chaos Rate is a defensive disruption composite blending defensive EPA/play allowed, success rate allowed, negative EPA forced rate, sack rate, turnover forced rate, explosive rate allowed, third/fourth-down stop rate, and penalty first-down rate allowed. Together they capture overall team quality, offensive efficiency, and defensive disruption.*

| Attribute | Value |
|-----------|-------|
| **Model** | Standard Elo + QB-change season regression + Platt + `home_qb_changed` + `away_qb_changed` + `home_rolling_mov_3` + `away_rolling_mov_3` + `home_fdr_fraud_detector_rating` + `away_fdr_fraud_detector_rating` + `home_doba_doba_score` + `away_doba_doba_score` + `home_chaos_rate` + `away_chaos_rate` |
| **K-factor** | 36 |
| **HFA** | 40 |
| **Preseason regression** | 0.1 (base) + 0.2 for teams with QB change |
| **Elo MOV multiplier** | None (standard point-differential) |
| **Decay half-life** | 32 games |
| **Base features** | `home_qb_changed`, `away_qb_changed`, `home_rolling_mov_3`, `away_rolling_mov_3`, FDR, DOBA, Chaos Rate |
| **FDR data source** | nflverse PBP (nflreadpy) + schedule results + Elo ratings |
| **DOBA data source** | nflverse PBP (nflreadpy) |
| **Chaos Rate data source** | nflverse PBP (nflreadpy) |
| **Selection method** | Fold-safe rolling-origin 3-fold validation (Platt fit per fold) |
| **Avg validation log loss** | **0.5609** |
| **2025 holdout log loss** | **0.5548** |
| **2025 holdout Brier** | 0.1886 |
| **2025 holdout AUC** | 0.7871 |
| **2025 holdout accuracy** | 0.6884 |
| **Report** | `reports/experiments/statspace_chaos.md` |
| **Selection date** | 2026-07-26 |

## Held-Out-Informed Diagnostics

| Model | Validation LL | Holdout LL | Notes |
|-------|--------------|------------|-------|
| **FDR + QB overlay** | **0.6172** | **0.5972** | FDR + frozen QB overlay on top |
| Market (no-vig closing moneyline) | 0.6052 | **0.6090** | Diagnostic only — near-kickoff timing |
| Spread→prob | 0.6076 | 0.6092 | Diagnostic only |
| Separate O/D Elo (k_off=52, k_def=20) + Platt | 0.6376 | 0.6258 | Holdout-informed parameter selection |

## Holdout-Informed Diagnostics

These models used 2025 holdout performance for parameter selection and are NOT clean football-only benchmarks. They are diagnostic references for the improvement ceiling.

| Model | Validation LL | Holdout LL | Notes |
|-------|--------------|------------|-------|
| Separate O/D Elo (k_off=52, k_def=20) + Platt | 0.6376 | **0.6258** | k_off/k_def selected using holdout — not a clean promotion |
| Standard Elo + Platt (incumbent) | 0.6368 | 0.6285 | Clean; previous incumbent |

## Superseded Models (Clean Promotions)

| Model | Challenge | Holdout LL at Promotion | Beat |
|-------|-----------|------------------------|------|
| **Standard Elo + qb_changed + mov3 + FDR + DOBA + Chaos (current)** | 0.5853 val, **0.5548 holdout** | **0.5548** | FDR + DOBA 0.5945 |
| Standard Elo + qb_changed + mov3 + FDR + DOBA | 0.5853 val, 0.5945 holdout | **0.5945** | FDR + QB overlay 0.5972 |
| Standard Elo + qb_changed + mov3 + FDR | 0.6203 val, 0.6011 holdout | **0.6011** | Frozen QB overlay 0.6228 |
| Frozen QB overlay v3.0.0 | 0.6317 val, 0.6228 holdout | **0.6228** | Qb_changed + mov3 0.6262 |
| Standard Elo + qb_changed + mov3 + Platt | 0.6334 val, 0.6262 holdout | **0.6262** | Season reg Elo 0.6285 |
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
| Glicko rating system | Rejected *(bug fixed 2026-07-21, retested)* | 0.6338 (best after fix, still worse) | `reports/experiments/glicko_rating.md` |
| Home/away separate Elo | Rejected | 0.6634 | `reports/experiments/home_away_elo.md` |
| Coach tenure | Rejected | 0.6326–0.6771 | `reports/experiments/combined_features.md` |
| Comprehensive efficiency (Team EPA + PFR + Snap) | Rejected | 0.6788 (inc+eff) | `reports/experiments/comprehensive_efficiency.md` |

## Market Benchmark

Market (no-vig closing moneyline) is now BEATEN by 0.0542 log loss by the
FDR+DOBA+Chaos incumbent (0.5548 vs 0.6090). StatSpace composites now
substantially outperform market odds.

| Model | Holdout LL |
|-------|-----------|
| **Football-only incumbent (Elo + qb_changed + mov3 + FDR + DOBA + Chaos + Platt)** | **0.5548** |
| Market (no-vig) | 0.6090 |
| Spread→prob | 0.6092 |
| Elo + Market (logit) | 0.6119 |

## Promotion Rules

1. A challenger must beat the incumbent's **holdout log loss** AND have **better average rolling validation log loss** (both with minimum improvement delta of 0.001) to be promoted.
2. If the challenger uses a logit-space overlay (e.g., QB overlay), the non-gated subset must also not degrade (equality check).
3. Selection uses average rolling validation log loss only. Holdout data is for final evaluation only, never for model selection.
4. Every feature must be pregame-safe and explainable.
5. Do not promote based on AUC or ROI alone.
6. The first policy document explicitly setting these rules is RALPH Loop 4 (2026-07-06). Earlier experiments used varying thresholds.
