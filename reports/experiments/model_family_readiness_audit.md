# Model Family Readiness Audit

*Generated: 2026-06-30*
*Purpose: Determine the best next non-Elo or hybrid model-family experiments to try against the v3.0.0 Frozen QB Overlay incumbent.*

---

## 1. Current Incumbent Summary

| Attribute | Value |
|-----------|-------|
| **Model** | Standard Elo + qb_changed + rolling_mov_3 + Platt + Frozen QB overlay (gated) |
| **Version** | v3.0.0 |
| **Base Elo** | K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2 |
| **Base features** | elo_prob, home_qb_changed, away_qb_changed, home_rolling_mov_3, away_rolling_mov_3 |
| **Overlay** | Logit-space: `final_logit = logit(base_prob) + gamma * clip(net_adj, -cap, cap) * ln(10)/400` gated on `qb_changed OR starts<17` |
| **Overlay params** | gamma=1.0, cap=40 |
| **Calibration** | Platt (StandardScaler + LogisticRegression(C=1.0, L2)) per fold |
| **Incumbent file** | `src/sportslab/evaluation/predict_incumbent.py` |
| **Promotion file** | `src/sportslab/evaluation/frozen_qb_overlay_foldsafe_experiment.py` |
| **Validation LL** | 0.6305 (3-fold rolling-origin avg) |
| **Holdout LL** | **0.6200** |
| **Holdout Brier** | 0.2157 |
| **Holdout AUC** | 0.7098 |

### Promotion Chain

| Step | Model | Holdout LL | Δ |
|------|-------|-----------|---|
| 1 | Tuned Elo (K=32, HFA=25) | 0.6616 | — |
| 2 | Rolling-origin Elo + Platt | 0.6395 | −0.0221 |
| 3 | MOV Elo + Platt | 0.6373 | −0.0022 |
| 4 | Decayed Elo + Platt | 0.6298 | −0.0075 |
| 5 | Season-regression Elo + Platt | 0.6285 | −0.0013 |
| 6 | Elo + qb_changed + mov_3 + Platt | 0.6262 | −0.0023 |
| 7 | **Frozen QB Overlay (v3.0.0)** | **0.6200** | **−0.0062** |

---

## 2. Available Pregame-Safe Features

These features are computed by `src/sportslab/features/` modules and are available in the feature table or at experiment runtime. All are pregame-safe (no future data, no scores).

| Feature Family | Module | Columns | Status in Incumbent |
|---------------|--------|---------|---------------------|
| **Elo prob** | `ratings.py` | `elo_prob`, `elo_diff`, `home_elo_pre`, `away_elo_pre` | Used (spine) |
| **QB change flag** | `qb.py` | `home_qb_changed`, `away_qb_changed`, `qb_change_diff` | Used (feature) |
| **Rolling MOV 3** | `situational.py` | `home_rolling_mov_3`, `away_rolling_mov_3` | Used (feature) |
| **QB adjustment** | `qb_adjustment.py` | `home_qb_adj`, `away_qb_adj` | Used (overlay) |
| **QB starts** | `qb.py` | `home_qb_team_starts_pre`, `away_qb_team_starts_pre` | Used (gate) |
| **Rest diff** | `build_features.py` | `rest_diff`, `home_rest`, `away_rest` | Not used |
| **Dome flag** | `build_features.py` | `is_dome` | Not used |
| **Division game** | `build_features.py` | `div_game` | Not used |
| **Game type** | `build_features.py` | `game_type_enc` (0=REG, 1=WC, etc.) | Not used |
| **Week** | `build_features.py` | `week` | Not used |
| **Weekday** | `build_features.py` | `weekday_enc` | Not used |
| **Rolling MOV 1/2/5** | `situational.py` | `home_rolling_mov_1`, `home_rolling_mov_2`, etc. | Not used (3 selected) |
| **Rolling win %** | `situational.py` | `home_rolling_win_pct`, `away_rolling_win_pct` | Not used |
| **Win streak** | `situational.py` | `home_win_streak`, `away_win_streak` | Not used (bug fixed) |
| **YTD win %** | `situational.py` | `home_ytd_win_pct`, `away_ytd_win_pct` | Not used |
| **Scheduling** | `scheduling.py` | `short_week`, `off_bye`, `thursday_flag`, `monday_flag`, `consecutive_road` | Not used (rejected) |
| **QB depth** | `qb_depth.py` | `qb_rust_games`, `qb_first_season_start`, `career_starts`, `career_win_pct` | Not used (rejected) |
| **Roof/surface** | `build_features.py` | `roof_enc`, `surface_enc` | Not used |
| **Coach features** | `coach.py` | `coach_tenure`, `coach_win_pct`, `coach_career_wins`, etc. | Not used (rejected) |
| **Coach-QB tenure** | `coach_qb_tenure.py` | `home_coach_qb_games`, `away_coach_qb_games` | Not used (rejected) |
| **Rolling turnover** | `turnovers.py` | `to_net_3`, `to_net_5` | Not used (watchlist) |
| **Situational micro** | `situational_micro.py` | `div_x_qb_changed`, `first_year_coach`, `surface_mismatch` | Not used (rejected) |
| **Stadium ID** | `build_features.py` | `stadium_id_enc` | Not used |

---

## 3. Features Excluded Due to Leakage Risk

These columns are present in the feature table but MUST NOT be used as model features. They are preserved for diagnostic/audit purposes only.

| Column Group | Columns | Risk |
|-------------|---------|------|
| **Scores** | `away_score`, `home_score`, `result`, `total` | Contain game outcome |
| **Derived outcomes** | `home_win` (target), `is_tie` | Contains target |
| **Market** | `home_moneyline`, `away_moneyline`, `spread_line`, `home_spread_odds`, `away_spread_odds`, `total_line`, `under_odds`, `over_odds` | Closing lines are near-kickoff, not pregame-safe |
| **Weather raw** | `weather_temp`, `weather_tmin`, `weather_tmax`, `weather_precip`, `weather_wind_speed` | Raw weather from API may use post-game data; use processed weather features only |

---

## 4. Diagnostic-Only Features

These features are available but marked as **diagnostic only**. They should not be used in promotion-eligible experiments without explicit re-evaluation.

| Feature | Reason for Diagnostic Status |
|---------|------------------------------|
| **Market odds** (`market_home_prob_novig`, `market_away_prob_novig`) | Closing lines — are near-kickoff, not purely pregame. Market beats incumbent by 0.028 LL (0.6090 vs 0.6373). Adding market as a feature would be circular (model should learn independent signals). |
| **QB raw adjustment** (`home_qb_adj`, `away_qb_adj`) | These are used in the frozen QB overlay (the incumbent mechanism). Using them as standalone features would double-count the QB adjustment signal. |
| **Spread→prob** (`spread_home_prob`, `spread_away_prob`) | Same closing-line timing issue. |
| **Referee** (`referee_enc`) | Only 1 referee has missing values. Diagnostic-only without penalty data. |

---

## 5. Existing Validation Protocol

### Rolling-Origin Fold Structure

Defined in `src/sportslab/evaluation/experiment_config.py`:

```python
ALL_SEASONS = [2021, 2022, 2023, 2024]
ROLLING_FOLDS = [
    ([2021],               2022),   # Fold 1: train 2021, val 2022
    ([2021, 2022],         2023),   # Fold 2: train 2021-2022, val 2023
    ([2021, 2022, 2023],   2024),   # Fold 3: train 2021-2023, val 2024
]
```

### Fold-Safe Pattern

Established in `frozen_qb_overlay_foldsafe_experiment.py`:

1. Compute features chronologically on full dataset (Elo is inherently chronological — no future leakage)
2. For each fold: fit model (Platt/overlay) using **only** the fold's training seasons
3. Generate predictions for ALL data using that fold's model
4. Score ONLY on the fold's validation season
5. Average validation log loss across folds for model selection
6. Holdout (2025) is NEVER accessed during validation

### Selection Criteria

- Variants are selected by **average validation log loss** across all 3 folds
- 2025 holdout is strictly diagnostic — only evaluated once after selection
- `MIN_PROMOTION_DELTA = 0.001` for promotion decisions

---

## 6. Existing Calibration Protocol

### Current Incumbent Calibration

The v3.0.0 incumbent uses a 2-stage calibration:

1. **Stage 1 — Platt scaling** (StandardScaler + LogisticRegression, C=1.0, L2, solver='lbfgs')
   - Input: `[elo_prob, home_qb_changed, away_qb_changed, home_rolling_mov_3, away_rolling_mov_3]`
   - Output: `base_incumbent_prob`

2. **Stage 2 — Frozen QB overlay** (logit-space additive adjustment)
   - Input: `base_incumbent_prob` + `home_qb_adj` + `away_qb_adj` + gate mask
   - Formula: `final_logit = logit(base_prob) + gamma * clip(net_adj, -cap, cap) * ln(10)/400`
   - Only applied where gate is active (qb_changed OR starts<17)

### Available Calibration Infrastructure

| Tool | Location | Description |
|------|----------|-------------|
| `fit_platt(x, y)` | `experiment_utils.py:39` | StandardScaler + LogisticRegression(C=1.0, L2) — basic Platt |
| `build_baseline_pipeline()` | `models/logistic.py:11` | SimpleImputer + StandardScaler + LR — adds imputation |
| `compute_classification_metrics()` | `metrics.py:7` | Log loss, Brier, AUC, Acc, calibration buckets (10 deciles) |
| `calibration_buckets()` | `experiment_utils.py:91` | 10 equal-width decile buckets |
| `confidence_buckets()` | `experiment_utils.py:115` | 5 confidence bins (0-20 through 80-100) |
| `bootstrap_delta()` | `experiment_utils.py:59` | Bootstrap CI for Δ log loss |
| Era-split Platt | `calibration_improvements_experiment.py` | Separate Platt for weeks 1-4 vs 5+ |
| Temperature scaling | `confidence_calibration_experiment.py` | Divide logit by T > 1 |
| Global shrinkage | `confidence_calibration_experiment.py` | Shrink toward prior (0.5 or home win rate) |

### Gaps in Calibration Infrastructure

1. **No reusable Platt tuner**: `fit_platt()` uses fixed C=1.0. Every experiment that wants C tuning duplicates the grid search.
2. **No reusable calibration comparator**: Each experiment duplicates the "fit Platt, compare against incumbent" pattern.
3. **No elastic-net wrapper**: `LogisticRegression` with `penalty='elasticnet'` needs `solver='saga'` and `l1_ratio`. Not wrapped anywhere.
4. **No built-in log-loss decomposition**: Adding decomposition (calibration loss + refinement loss) would help diagnose whether calibration or discrimination is the issue.
5. **No shrinkage calibration as reusable function**: The shrinkage methods tested in `confidence_calibration_experiment.py` are not factored into a reusable utility.

---

## 7. Existing Holdout Protocol

### Standard Pattern

1. Fit final model on ALL 2021–2024 data (training + validation seasons combined)
2. Evaluate ONCE on 2025 holdout season
3. Report log loss, Brier, AUC, accuracy
4. The holdout is never used for model selection

### Incumbent-Specific Protocol

Per `predict_incumbent.py`:

```python
TRAIN_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025

# Fit on 2021-2024
pipe.fit(x_train, y_train)

# Score on 2025
hold_y, hold_prob = ...
metrics = compute_classification_metrics(hold_y, hold_prob)
```

Verified: `INCUMBENT_HOLDOUT_LL = 0.6200` matches the holdout CSV artifact.

---

## 8. Gaps in Reusable Evaluation Infrastructure

### What Exists

| Component | Status | Location |
|-----------|--------|----------|
| Shared config (folds, holdout) | ✅ | `experiment_config.py` |
| Metrics (LL, Brier, AUC, Acc) | ✅ | `metrics.py`, `experiment_utils.py` |
| Platt fitting | ✅ | `experiment_utils.py` (fixed C=1.0) |
| Feature matrix builder | ✅ | `experiment_utils.py` |
| Bootstrap CI | ✅ | `experiment_utils.py` |
| Calibration buckets | ✅ | `experiment_utils.py`, `metrics.py` |
| Worst-predictions finder | ✅ | `experiment_utils.py` |
| Fold-safe rolling-origin pattern | ✅ | Multiple experiments |
| Holdout scoring | ✅ | Multiple experiments |

### What's Missing

| Component | Description | Impact |
|-----------|-------------|--------|
| **Reusable fold-safe runner** | No shared function that takes a model-building callable and runs it across folds. Each experiment reimplements the fold loop. | High — each new experiment duplicates ~50 lines of fold logic |
| **C-tunable Platt** | `fit_platt()` uses fixed C=1.0. Tuning C requires manual grid search. | Medium — easy to copy-paste but violates DRY |
| **Model comparator** | No shared function to compare two models (incumbent vs challenger) across folds with consistent reporting. | Medium — each experiment writes its own comparison table |
| **Slice analyzer** | No reusable slice analysis (QB-change, confidence, week, etc.). Each experiment reimplements slice logic. | Low — slices vary by experiment |
| **Report template** | No standardized report format. Each experiment writes unique Markdown. | Low — variety is acceptable |
| **Promotion checker** | No shared function that checks MIN_PROMOTION_DELTA across val + holdout. | Low — simple logic, easy to duplicate |

**Key takeaway**: The existing infrastructure is functional but repetitive. New experiments duplicate ~80 lines of boilerplate (fold loop, Platt fitting, report writing). This is acceptable for 1–3 new experiments but should be refactored if the project scales.

---

## 9. Recommended Model Families to Test Next

### Priority 1: Regularized Logistic Meta-Model (Promotion-Eligible)

**Why**: The v3.0.0 incumbent uses LogisticRegression(C=1.0, L2) as its Platt calibrator. Tuning C (regularization strength) and testing L1 vs L2 vs ElasticNet could improve calibration. Additionally, adding low-cardinality pregame features (rest_diff, is_dome, div_game) on top of the incumbent logit may capture residual signal.

**Risk**: Low. This is a minor extension of the existing Platt approach. The incumbent already uses logistic regression — this just tunes the hyperparameters.

**Features**: incumbent_logit (from v3.0.0), rest_diff, is_dome, div_game, game_type, week sin/cos

**Required infrastructure**: Fold-safe per-fold incumbent fitting + logistic regression grid search (C, penalty, l1_ratio). The fold-safe pattern exists in `frozen_qb_overlay_foldsafe_experiment.py`.

### Priority 2: GAM/Spline-Based Logistic Model (Diagnostic)

**Why**: The incumbent assumes a linear relationship between the logit of elo_prob and outcome. Nonlinear transformations (splines, binning, polynomial features) could capture threshold effects (e.g., "blowout threshold", "close game compression").

**Risk**: Medium. Splines on only ~1000 training rows risk overfitting. Must use strong regularization.

**Features**: Spline/binned elo_prob, spline/binned overlay delta, spline/binned rest_diff

**Required infrastructure**: `sklearn.preprocessing.SplineTransformer` or manual binning. Requires scipy (already in optional deps).

### Priority 3: Fold-Safe Gradient Boosting Diagnostic (Diagnostic)

**Why**: Boosted trees (HistGradientBoostingClassifier) with strong regularization (low learning rate, early stopping, min samples leaf) could capture nonlinear interactions that Elo + logistic miss.

**Risk**: Medium-High. Previous tree experiments (expressive_models, 2026-06-29) showed consistent overfit patterns. This experiment would use fold-safe validation, strict early stopping within each fold, and diagnostic-only status by default.

**Features**: All pregame-safe features (20–30 columns)

**Required infrastructure**: `sklearn.ensemble.HistGradientBoostingClassifier` with `early_stopping`. No extra dependencies.

### Priority 4: Small Ensemble Blend (Diagnostic)

**Why**: A validation-weighted blend of the incumbent and one challenger could marginally improve performance if the two models have uncorrelated errors.

**Risk**: Low-Medium. Very easy to overfit by using holdout to choose blend weights. Must select weights on validation only.

**Method**: Linear blend of incumbent_prob and challenger_prob, weight chosen by grid search on validation log loss.

### Priority 5: Dynamic Calibration (Promotion-Eligible)

**Why**: The incumbent uses fixed Platt scaling. Dynamic methods (season-aware Platt, high-confidence shrinkage, temperature scaling) could improve calibration without changing the base model.

**Risk**: Low. These are post-hoc adjustments that cannot leak data (if fold-safe). Previous tests (confidence_calibration, calibration_improvements) were rejected but on older incumbents. Worth re-testing against v3.0.0.

---

## 10. Model Families to Avoid (With Reasons)

| Model Family | Reason for Exclusion |
|-------------|---------------------|
| **AutoGluon/AutoML** | Already tested and rejected. All variants underperformed Platt on both val and holdout. Only sklearn ensemble models available (no LightGBM/XGBoost). |
| **Full tree ensembles (RF, GB, XGB)** | Already tested (expressive_models): RF won validation (0.6329) but lost holdout (0.6456). HGB tied val (0.6361) but lost holdout (0.6638). Consistent overfit pattern on ~1000 training rows. |
| **Neural network** | Sample size (~1000 games) is far too small. Would require massive regularization and risk severe overfit. No infrastructure. |
| **SVM with RBF kernel** | No obvious advantage over logistic regression. Kernel methods are hard to calibrate and explain. No infrastructure. |
| **Team-level OHE** | 32+ team one-hot features on ~1000 rows = severe overfit risk. Team identity is already captured by Elo ratings. |
| **QB identity OHE** | Already rejected with holdout LL 14.51. 93+ classes on 376 rows. |
| **Market-based model** | Market closing lines beat the incumbent by 0.028. But market is a diagnostic benchmark only — using it as a feature would be circular (model should learn independent signals). |
| **Expanded Elo spine** | Already tested (840 combos, expanded_elo_spine, rejected). 0/840 combos beat v3.0.0 by >= 0.001. QB overlay compresses Elo differences below the promotion threshold. |
| **Roster overlay** | Already tested (roster_overlay_foldsafe, rejected). Position-group availability from injury OUT counts too noisy. |
| **QB × roster interaction** | Already tested (qb_roster_interaction, rejected). No signal beyond QB overlay alone. |

---

## 11. Recommended First Experiment

### Regularized Logistic Meta-Model (Promotion-Eligible)

**Rationale**:
1. Lowest risk — extends existing Platt approach with hyperparameter tuning
2. New features on top of v3.0.0 — rest_diff, is_dome, div_game, game_type were tested on older incumbents but never on v3.0.0
3. Infrastructure already exists — fold-safe pattern, logistic regression, grid search
4. Quick to run — ~200 hyperparameter combos × 3 folds = fast
5. If it fails, it eliminates the most conservative challenger and justifies trying more complex models

**Design**:
- Base: v3.0.0 incumbent (fold-safe: fit Platt + overlay per fold)
- Meta-model: LogisticRegression with C grid + L1/L2 penalty
- Features: incumbent_logit, rest_diff, is_dome, div_game, week_sin, week_cos, game_type_is_playoff
- Calibration: The logistic regression is inherently calibrated (Platt scaling)
- Selection: Average validation log loss across 3 folds
- Holdout: One-shot evaluation on 2025
- Promotion: Requires >= 0.001 improvement on BOTH val and holdout

---

## 12. Risks and Safeguards

| Risk | Likelihood | Impact | Safeguard |
|------|-----------|--------|-----------|
| Meta-model overfits to ~1000 training rows | Medium | High — holdout LL could regress | Strong L1/L2 regularization (C grid from 0.001 to 100); fold-safe validation catches overfit patterns |
| Features add noise, not signal | Medium | Low — val LL stays same or degrades | Reject if val doesn't improve; no promotion without both val AND holdout improvement |
| Nested fitting introduces leakage | Low | Critical — invalid results | Each fold fits Platt + overlay from scratch using only training seasons; no data from validation or holdout leaks into meta-features |
| C hyperparameter chosen by noise | Low | Medium — val LL may be misleading | 3-fold average reduces noise; MIN_PROMOTION_DELTA=0.001 prevents noise-level promotion |
| incumbent_logit dominates all other features | High | Low — experiment confirms that v3.0.0 is optimal | If other features get near-zero coefficients, the conclusion is valid: v3.0.0 extracts all available signal |
| Report not comparable to registry | Low | Low — follow existing experiment template | Use standard metrics (LL, Brier, AUC, Acc) and structure from frozen_qb_overlay_foldsafe |

---

## Files Inspected

See exploration results from 2026-06-30 session. Key files:

- `src/sportslab/evaluation/predict_incumbent.py` — v3.0.0 production pipeline
- `src/sportslab/evaluation/frozen_qb_overlay_foldsafe_experiment.py` — Gold-standard fold-safe pattern
- `src/sportslab/evaluation/combined_features_experiment.py` — Feature combination (v2.0.0 creator)
- `src/sportslab/evaluation/expanded_elo_spine_experiment.py` — 840-combo Elo grid (rejected)
- `src/sportslab/evaluation/experiment_utils.py` — Shared utilities
- `src/sportslab/evaluation/experiment_config.py` — Shared config
- `src/sportslab/evaluation/metrics.py` — Classification metrics
- `src/sportslab/models/logistic.py` — Basic logistic pipeline
- `src/sportslab/features/build_features.py` — Feature table builder
- `src/sportslab/features/` (all 26 files) — Feature modules
- `src/sportslab/cli.py` — CLI commands
- `reports/benchmarks/nfl_research_incumbent.md` — Incumbent registry
- `reports/benchmarks/incumbent_model_card.md` — Model card
- `reports/benchmarks/leaderboard.csv` — 40-row leaderboard
- `reports/benchmarks/benchmark_history.md` — 39 experiments
- `tests/test_incumbent_schema.py` — Schema validation
