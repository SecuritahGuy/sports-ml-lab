# Incumbent Model Card — NFL Football-Only Research Benchmark

*Generated: 2026-06-24*

## Model Identity

| Field | Value |
|-------|-------|
| **Name** | Standard Elo + QB-Change Season Regression + Rolling MOV + Platt |
| **Version** | v2.0.0 |
| **Type** | Logistic regression on Elo probability + binary/continuous features |
| **Domain** | Football-only, pregame-safe, zero-leakage NFL win prediction |
| **Status** | Active incumbent (clean, market-free research benchmark) |

## Feature Set

### Core Rating Features

| Feature | Source | Description |
|---------|--------|-------------|
| `elo_prob` | `compute_elo_features()` | Elo-implied home win probability from standard point-differential Elo with season-regression overrides |

### Engineered Features

| Feature | Source | Description |
|---------|--------|-------------|
| `home_qb_changed` | `compute_qb_features()` | Binary: 1 if home QB did not start prior game (injury/benching) |
| `away_qb_changed` | `compute_qb_features()` | Binary: 1 if away QB did not start prior game |
| `home_rolling_mov_3` | `compute_situational_features()` | Average margin of victory over home team's last 3 games (preceding) |
| `away_rolling_mov_3` | `compute_situational_features()` | Average margin of victory over away team's last 3 games (preceding) |

### Features Explicitly NOT Used

- No team, QB, coach, or stadium identity encodings
- No market data (moneyline, spread, totals)
- No weather data
- No injury report data (beyond the QB-change binary)
- No scheduling/rest features
- No coach tenure/win% features
- No EPA/DVOA team efficiency stats
- No home/away separate ratings
- No tree-based or neural network models

## Elo/Rating Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| K-factor | 36 | Learning rate for Elo updates |
| Home field advantage | 40 | Points added to home team rating |
| Base preseason regression | 0.1 | Fraction toward league mean each offseason |
| QB-change bonus | 0.2 | Additional regression fraction for teams with a new starting QB |
| Decay half-life | 32 games | Exponential decay toward prior ratings |
| MOV type | `capped_linear` | Scale=0.05, cap=2.0 — caps blowout influence |
| Selection method | Rolling-origin 3-fold validation | Train [2021]→val 2022, train [2021-2022]→val 2023, train [2021-2023]→val 2024 |

## Calibration

| Method | Details |
|--------|---------|
| **Type** | Platt scaling (logistic regression) |
| **Input** | `elo_prob`, `home_qb_changed`, `away_qb_changed`, `home_rolling_mov_3`, `away_rolling_mov_3` |
| **Training** | 2021–2024 seasons, 5 features, LogisticRegression(C=1, max_iter=1000) |
| **Standardization** | StandardScaler on feature matrix before logistic fit |

## Validation Method

- **Protocol**: Rolling-origin 3-fold walk-forward
  - Fold 1: Train 2021 → Validate 2022
  - Fold 2: Train 2021–2022 → Validate 2023
  - Fold 3: Train 2021–2023 → Validate 2024
- **Selection metric**: Average log loss across 3 validation folds
- **Holdout**: 2025 season (assessed exactly once, after all model selection complete)
- **Holdout size**: 276 games

## Performance

### Holdout (2025, n=276)

| Metric | Value |
|--------|-------|
| **Log loss** | **0.6262** |
| Brier score | 0.2191 |
| ROC AUC | 0.7024 |
| Accuracy | 0.6667 |

### Validation (rolling average)

| Metric | Value |
|--------|-------|
| **Average log loss** | **0.6334** |
| Fold 1 (2022) | 0.6206 |
| Fold 2 (2023) | 0.6560 |
| Fold 3 (2024) | 0.6237 |

## Promotion Criteria

1. A challenger must beat **0.6262** holdout log loss to become the new football-only incumbent.
2. The challenger must also have **better average rolling validation log loss** than the incumbent.
3. Selection must use average rolling validation log loss only. Holdout data is for final evaluation only, never for model selection.
4. 2025 holdout must remain untouched during model selection.
5. Every feature must be pregame-safe, explainable, and leakage-safe.
6. Do not promote based on AUC or ROI alone.

## Known Rejected Feature Families

| Family | Reason for Rejection | Report |
|--------|----------------------|--------|
| Scheduling/rest flags | Validation and holdout both worse | `schedule_rest_features.md` |
| Weather | Validation and holdout both worse | `weather_features.md` |
| QB identity OHE | Exploded to log loss 14.51 on holdout | `qb_features.md` |
| Coach tenure/win% | All variants worse on both val and holdout | `combined_features.md` |
| EPA team efficiency | Validation and holdout both worse | `epa_features.md` |
| Injury report features (20 cols) | All 20 features added noise | `injury_features.md` |
| Team stats (yards/points/sacks) | All variants worse | `team_stats.md` |
| Tree models (HGB/GB/RF) | Overfit: won validation but lost holdout | `expressive_models.md` |
| AutoGluon AutoML | Validation and holdout both worse | `autogluon.md` |
| Team-specific HFA | Worse validation despite better holdout | `team_hfa.md` |
| Home/away separate Elo | Noisier ratings, both worse | `home_away_elo.md` |
| Rolling MOV windows ≠ 3 | mov_1 won val but lost holdout; mov_2+ all worse | `rolling_mov_sensitivity.md` |
| Comprehensive efficiency (Team EPA + PFR + Snap) | All 58 features added noise; inc+eff LL=0.6788 | `comprehensive_efficiency.md` |

## Known Failure Modes

Identified by residual diagnostics (`reports/experiments/residual_diagnostics.md`):

| Failure Mode | Impact | Details |
|-------------|--------|---------|
| **QB-change games** | LL↑ 0.042 | Largest gap: Elo undershoots when QB changes |
| **Very high confidence** (>0.9) | Calibration error 0.249 | Model overconfident on longshot away teams |
| **Early season** (weeks 1–4) | Higher error | Less data, more uncertainty |
| **Monday night games** | LL 0.6935 vs Sunday 0.6453 | Small sample, higher variance |
| **Open-roof stadiums** | LL 0.7206 | Retractable/open roof games |
| **Model vs market disagreement** | >0.15 gap signals Elo error | Market is strictly more informative |

## Intended Use

- **Research benchmark**: The football-only reference point for all future experiments
- **Pregame prediction**: All features available before kickoff; no final-score leakage
- **Calendar applicability**: NFL seasons 2021–current only
- **Comparison baseline**: All challengers must beat this model's holdout log loss

## Non-Goals

- Not for betting/investment decisions (see ROI caveat in research philosophy)
- Not a market-beating model (market benchmark: 0.6090 holdout LL)
- Not applicable to pre-2021 seasons (never trained or tested)
- Not for in-game or live betting (no play-by-play features)
- Not for player-level predictions (passer rating, yards, etc.)
- Not a production API or microservice
- Not statistically significant for single-game decisions

## Market-Aware Caveat

The closing moneyline market (no-vig) achieves holdout log loss **0.6090**, significantly better than the incumbent (0.6262). The market is the true performance ceiling for pregame NFL prediction. The incumbent is a purely pregame, market-free benchmark for measuring whether independent football features can approach market efficiency.

**Elo residuals vs market residuals correlate at r=0.9768** — our errors are nearly identical to the market's, just amplified. This suggests Elo captures the same signal as the market, but with more noise.

## Leakage Controls

| Control | Implementation |
|---------|----------------|
| No future data in features | Rolling features computed chronologically from prior games only |
| Season boundary resets | Team statistics reset each season |
| Holdout isolation | 2025 never accessed during any model/grid search |
| No ROO/LOO/Ago on train | Rolling-origin validation prevents target leakage |
| No raw identity encodings | Team/QB/coach/stadium names never used as numeric features |
| Pre-game features only | No final score, result, or target columns in features |
| Market data = diagnostic | Market fields clearly labeled `_diagnostic`, never used in training |

## Reproducibility Commands

```bash
# Regenerate prediction artifacts
make predict-incumbent

# Run weekly report
make weekly-report

# Run rolling-origin validation experiment
make combined-features

# Run residual diagnostics
make residual-diagnostics

# Run rolling MOV sensitivity test
make rolling-mov-sensitivity

# Run full test suite
make test
```

### Artifact Files

| Artifact | Path |
|----------|------|
| Full predictions CSV | `reports/predictions/incumbent_predictions.csv` |
| Holdout predictions CSV | `reports/predictions/incumbent_predictions_2025_holdout.csv` |
| Prediction cards | `reports/predictions/incumbent_prediction_cards.md` |
| Weekly report | `reports/predictions/weekly_report.md` |
| Model card | `reports/benchmarks/incumbent_model_card.md` |
| Benchmark history | `reports/benchmarks/benchmark_history.md` |
| Leaderboard | `reports/benchmarks/leaderboard.csv` |
| Experiment report | `reports/experiments/combined_features.md` |
