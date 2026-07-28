# NFL Research Incumbent

*Last updated: 2026-07-28*

## Overall Champion (Pi-Ratings + StatSpace PBP Composites)

**Short name:** Pi-Ratings + rolling_mov_3 + qb_changed + FDR + DOBA + Chaos Rate + Platt

**Model:** Pi-Ratings (α=0.5 power-law MOV, base_k=28, hk_ratio=1.25, HFA=30, reg=0.0) + rolling_mov_3 + qb_changed + StatSpace FDR (Fraud Detector Rating) + StatSpace DOBA (sustainable offensive efficiency) + StatSpace Chaos Rate (defensive disruption composite) + Platt calibration

Pi-Ratings is a coupled home/away rating system with power-law MOV (`|margin|^α` where α=0.5 compresses blowouts via square root) and asymmetric home/away K (hk_ratio=1.25 gives home teams faster updates). When α=1.0 and hk_ratio=1.0, this reduces to standard capped_linear Elo.

*FDR, DOBA, and Chaos Rate are team-season PBP composites capturing overall team quality, offensive efficiency, and defensive disruption. See `reports/experiments/pi_statspace.md` for details.*

| Attribute | Value |
|-----------|-------|
| **Avg validation log loss** | **0.5557** |
| **2025 holdout log loss** | **0.5532** |
| **2025 holdout Brier** | 0.1886 |
| **2025 holdout AUC** | 0.7874 |
| **Δvs old champion** | val=−0.0053, hold=−0.0016 |
| **Selection date** | 2026-07-28 |
| **Report** | `reports/experiments/pi_statspace.md` |

## Football-Only Champion (Pi-Ratings v4.0.0)

**Short name:** Pi-Ratings + qb_changed + rolling_mov_3 + QB overlay + Platt

**Model:** Pi-Ratings (α=0.5 power-law MOV, base_k=28, hk_ratio=1.25, HFA=30, reg=0.0) + Platt(`home_qb_changed`, `away_qb_changed`, `home_rolling_mov_3`, `away_rolling_mov_3`) + frozen QB overlay (gate: changed OR starts<17, cap=40, gamma=1.0)

Pi-Ratings is a coupled home/away rating system with two innovations vs standard Elo:
1. Power-law MOV: `mov = |margin|^alpha` where α=0.5 compresses blowouts via square root
2. Asymmetric K: `k_home = base_k × hk_ratio`, `k_away = base_k × (2 − hk_ratio)` where hk_ratio=1.25 gives home teams faster updates

When α=1.0 and hk_ratio=1.0, this reduces to standard capped_linear Elo.

| Attribute | Value |
|-----------|-------|
| **α (power-law MOV)** | 0.5 |
| **base_k** | 28 |
| **hk_ratio** | 1.25 |
| **HFA** | 30 |
| **Preseason regression** | 0.0 |
| **Features** | `home_qb_changed`, `away_qb_changed`, `home_rolling_mov_3`, `away_rolling_mov_3` |
| **QB overlay** | Gate: changed OR starts<17, cap=40, gamma=1.0 |
| **Calibration** | Platt per fold (logistic fit on pi_prob + features) |
| **Selection method** | Standalone champion comparison vs v3.0.0 in same pipeline |
| **Avg validation log loss** | **0.6260** |
| **2025 holdout log loss** | **0.5918** |
| **Δ vs v3.0.0** | val=−0.0046, hold=−0.0022 |
| **Selection date** | 2026-07-28 |
| **Report** | `reports/experiments/pi_ratings_champion_comparison.md` |

**Note:** StatSpace composites have been tested on the Pi-Ratings base (see `reports/experiments/pi_statspace.md`). They improve Pi-Ratings by the same pattern as standard Elo — the composites are feature-orthogonal to the rating system.

## Superseded Models (Clean Promotions)

| Model | Challenge | Holdout LL at Promotion | Beat |
|-------|-----------|------------------------|------|
| **Pi-Ratings + FDR + DOBA + Chaos (overall champion)** | 0.5557 val, **0.5532 holdout** | **0.5532** | Standard Elo+FDR+DOBA+Chaos 0.5548 |
| **Standard Elo + qb_changed + mov3 + FDR + DOBA + Chaos** | 0.5609 val, **0.5548 holdout** | **0.5548** | FDR + DOBA 0.5945 |
| Standard Elo + qb_changed + mov3 + FDR + DOBA | 0.5853 val, 0.5945 holdout | **0.5945** | FDR + QB overlay 0.5972 |
| Standard Elo + qb_changed + mov3 + FDR | 0.6203 val, 0.6011 holdout | **0.6011** | Frozen QB overlay 0.6228 |
| Frozen QB overlay v3.0.0 | 0.6317 val, 0.6228 holdout | **0.6228** | Pi-Ratings v4.0.0 (below) |
| **Pi-Ratings v4.0.0 (football-only champion)** | 0.6260 val, **0.5918 holdout** | **0.5918** | Frozen QB overlay 0.6228 |
| Standard Elo + qb_changed + mov3 + Platt | 0.6334 val, 0.6262 holdout | **0.6262** | Season reg Elo 0.6285 |
| Season reg Elo + Platt | 0.6315 val, 0.6285 holdout | **0.6285** | Decayed Elo 0.6298 |
| Decayed Elo (K=36) + Platt | 0.6321 val, 0.6298 holdout | **0.6298** | MOV Elo 0.6373 |
| MOV Elo (K=36) + Platt | 0.6363 val, 0.6373 holdout | **0.6373** | Rolling-origin Elo 0.6395 |
| Rolling-origin Elo (K=40, reg=0.25) + Platt | 0.6363 val, 0.6395 holdout | **0.6395** | Tuned Elo 0.6616 |
| Original tuned Elo (K=32, HFA=25) | 0.65 val, 0.6616 holdout | **0.6616** | First promoted |

## Held-Out-Informed Diagnostics

| Model | Validation LL | Holdout LL | Notes |
|-------|--------------|------------|-------|
| **FDR + QB overlay** | **0.6172** | **0.5972** | FDR + frozen QB overlay on top |
| Market (no-vig closing moneyline) | 0.6052 | **0.6090** | Diagnostic only — near-kickoff timing |
| Spread→prob | 0.6076 | 0.6092 | Diagnostic only |
| Separate O/D Elo (k_off=52, k_def=20) + Platt | 0.6376 | 0.6258 | Holdout-informed parameter selection |

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
| **Overall champion (Standard Elo + FDR + DOBA + Chaos + Platt)** | **0.5548** |
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
