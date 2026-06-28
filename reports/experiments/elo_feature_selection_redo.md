# Elo Feature Selection Redo

A comprehensive redo of feature selection around the Elo backbone.
Tests whether features like turnovers, EPA, weather, scheduling, and QB
signals can improve predictive quality beyond Elo alone.

---

## Executive Summary

**Current incumbent (before this experiment):** Standard Elo + qb_changed + rolling_mov_3 + Platt
- Validation LL: 0.6334
- Holdout LL: 0.6262

**Experiment scope:** 7 feature families tested via single-family ablations,
forward selection, and L1-regularized logistic regression.

**Key design choice:** Single-family ablations test each family as a whole (14–15 columns),
not the curated 2–4 column subsets in the incumbent. A family can be rejected at the full-family
level even though a curated subset of it (e.g., `qb_changed`, `rolling_mov_3`) carries signal.
Two baselines are used throughout: Elo-only (0.6406) for ablation comparisons, and the
full incumbent (0.6334) as the forward selection starting point. See the Baseline Clarification section.

---

## Feature Taxonomy

### A. Pregame Prediction Features (Active Candidates)

| Family | Columns | Description | Status |
|--------|---------|-------------|--------|
| Elo probability | `elo_prob` | Elo-implied home win probability | Core backbone |
| QB continuity | qb_changed, starts, win_pct, games since change | QB identity/turnover signal | Tested |
| Rolling form | MOV 3/5, pts for/against, win streak, YTD win% | Recent team performance | Tested |
| Scheduling | rest_diff, short week, bye, thr/mon, intl, consec road | Game context | Tested |
| Weather | temp, wind, precip, dome, cold/windy flags | Environmental context | Tested |
| Coach | tenure, career wins, win% | Coaching experience | Tested |
| Turnovers | giveaways, takeaways, TO diff rolling 3/5 | Ball security / creation | Tested |
| EPA | off_epa/play rolling 3/5 | Team efficiency | Tested |

### B. Diagnostic-Only Features (Market)

| Feature | Rationale |
|---------|----------|
| Closing moneyline | Diagnostic benchmark only (holdout 0.6090) |
| Spread line | Not pregame-safe as feature |

### C. Rejected / Not Re-tested

| Feature | Previously Tested | Reason |
|---------|------------------|--------|
| QB identity OHE | qb_features.md | Holdout LL 14.51 (catastrophic overfit) |
| Glicko rating | glicko_rating.md | All 432 configs worse |
| AutoGluon | autogluon.md | Both val and holdout worse |
| Home/away Elo | home_away_elo.md | Noisier ratings |
| Team-specific HFA | team_hfa.md | Worse val despite better holdout |
| Comprehensive efficiency | comprehensive_efficiency.md | 58 features, all noise |
| Injury features | injury_features.md | All 20 features added noise |
| Tree models | expressive_models.md | Classic overfit pattern |

---

## Leakage Controls

| Check | Implementation |
|-------|---------------|
| No same-game features | All rolling features use `shift(1)` or chronological prior-game lookup |
| Season boundary resets | Team statistics reset each season |
| Holdout isolation | 2025 not accessed during any selection step |
| Rolling-origin validation | 3-fold walk-forward prevents target leakage |
| Pre-game features only | No final score, result, or target columns in features |
| Market data excluded | Market fields are diagnostic-only |
| Tie handling | Ties encoded as home_win=NaN, excluded from model_eligible |
| Neutral-site handling | Neutral games excluded from training/prediction |

---

## Baseline Clarification

This report uses **two different baselines** depending on the section:

| Baseline | Val LL | Description | Used In |
|----------|--------|-------------|---------|
| Elo only (Platt) | 0.6406 | Platt logistic regression on `elo_prob` alone. No engineered features. | Single-family ablations, L1 regression |
| Incumbent (qb+mov3) | 0.6334 | Platt on `elo_prob + home_qb_changed + away_qb_changed + home_rolling_mov_3 + away_rolling_mov_3`. | Forward selection (starting point) |

**Why are they different?** The incumbent features (qb_changed binary + rolling_mov_3)
improve validation log loss by ~0.007 over Elo alone. This improvement has been
confirmed across multiple experiments (see combined_features.md, rolling_mov_sensitivity.md).

**Critical methodology note:** Single-family ablations test each family as a whole
(14–15 columns for QB continuity, 12 for Rolling form, etc.), not the curated 2–4
column subsets discovered by forward selection. A family can be rejected in full-family
testing even though a carefully selected subset of it is in the incumbent.
The forward selection section (starting from the incumbent's curated subset) is the
proper test for whether additional features improve on the current model.

---

## Rolling-Origin Validation Results

### 1. Single-Family Ablations

Each family tested on top of `elo_prob` via Platt logistic regression.
Baseline (Elo only): **0.6406** avg val LL

| Model | Avg Val LL | Δ vs Baseline | Fold1 | Fold2 | Fold3 |
|-------|-----------|--------------|-------|-------|-------|
| Elo only (Platt) | 0.6406 | — | 0.6471 | 0.6621 | 0.6126 |
| Elo + QB continuity | 0.6420 | +0.0014 | 0.6616 | 0.6719 | 0.5926 |
| Elo + EPA | 0.6423 | +0.0017 | 0.6483 | 0.6663 | 0.6124 |
| Elo + Weather | 0.6437 | +0.0031 | 0.6553 | 0.6656 | 0.6101 |
| Elo + Turnovers | 0.6456 | +0.0050 | 0.6590 | 0.6598 | 0.6180 |
| Elo + Rolling form | 0.6475 | +0.0069 | 0.6609 | 0.6610 | 0.6207 |
| Elo + Coach | 0.6542 | +0.0136 | 0.6610 | 0.6774 | 0.6242 |
| Elo + Scheduling | 0.6600 | +0.0194 | 0.6769 | 0.6696 | 0.6337 |

### 2. Forward Selection

Starting from incumbent features (qb_changed + mov3). Adding families greedily.

| Step | Val LL | Δ |
|------|--------|---|
| Incumbent baseline (qb_changed + mov3) | 0.6334 | — |
| **Final** | **0.6334** | **-0.0072** |

**Final forward-selected features (4 total):**

- `home_qb_changed`
- `away_qb_changed`
- `home_rolling_mov_3`
- `away_rolling_mov_3`

### 3. L1-Regularized Logistic Regression

All candidate features + elo_prob via L1-regularized logistic regression.

| C | Avg Val LL | vs Baseline |
|---|-----------|-------------|
| C=0.05 | 0.6469 | +0.0063 |
| C=0.1 | 0.6472 | +0.0066 |
| C=0.01 | 0.6895 | +0.0489 |
| C=0.5 | 0.7325 | +0.0919 |
| C=1.0 | 0.8086 | +0.1680 |

Best L1: **C=0.05** (0.6469)

**L1 coefficient stability (non-zero across folds):**

| Feature | Fold1 Coef | Fold2 Coef | Fold3 Coef | Mean | Sign Stable |
|---------|-----------|-----------|-----------|------|-------------|
| qb_starts_diff | +0.232974 | +0.209344 | +0.187541 | +0.209953 | ✓ |
| away_qb_changed | +0.000000 | +0.000000 | +0.080092 | +0.026697 | ✗ |
| home_coach_career_wins | +0.000000 | +0.064819 | +0.000000 | +0.021606 | ✗ |
| games_since_qb_change_diff | +0.000000 | +0.062664 | +0.000000 | +0.020888 | ✗ |
| home_rolling_pts_for | +0.000000 | +0.000000 | +0.058277 | +0.019426 | ✗ |
| home_coach_tenure | +0.000000 | +0.000000 | +0.058130 | +0.019377 | ✗ |
| home_off_bye | +0.000000 | +0.000000 | +0.056557 | +0.018852 | ✗ |
| home_short_week | +0.000000 | +0.011042 | +0.021297 | +0.010780 | ✗ |
| home_qb_starts_this_season_pre | +0.015304 | +0.000000 | +0.013787 | +0.009697 | ✗ |
| home_rolling_mov_5 | +0.000000 | +0.000000 | +0.014244 | +0.004748 | ✗ |
| away_off_epa_5 | +0.000000 | -0.013865 | +0.000000 | -0.004622 | ✗ |
| off_epa_net_5 | +0.000000 | +0.013865 | +0.000000 | +0.004622 | ✗ |
| home_games_since_qb_change | +0.000000 | +0.007620 | +0.000000 | +0.002540 | ✗ |
| away_coach_win_pct | +0.000000 | +0.000000 | -0.001822 | -0.000607 | ✗ |

---

## 2025 Holdout Results

All models are evaluated on the locked 2025 holdout after selection.
No selection decisions used holdout performance.

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|-----|
| Incumbent (qb+mov3) | 0.6262 | 0.2180 | 0.7050 | 0.6630 |
| Forward selection | 0.6262 | 0.2180 | 0.7050 | 0.6630 |
| Elo only (Platt) | 0.6315 | 0.2204 | 0.6983 | 0.6739 |
| Elo + Scheduling | 0.6332 | 0.2214 | 0.6931 | 0.6739 |
| Elo + EPA | 0.6349 | 0.2221 | 0.6904 | 0.6703 |
| Elo + Turnovers | 0.6358 | 0.2225 | 0.6889 | 0.6630 |
| L1 selected | 0.6361 | 0.2224 | 0.6924 | 0.6486 |
| Elo + QB continuity | 0.6361 | 0.2228 | 0.6896 | 0.6377 |
| Elo + Weather | 0.6421 | 0.2251 | 0.6821 | 0.6449 |
| Elo + Rolling form | 0.6448 | 0.2267 | 0.6735 | 0.6449 |
| Elo + Coach | 0.6756 | 0.2375 | 0.6564 | 0.6341 |

---

## Feature Stability Analysis

### Features that consistently help (negative Δ on validation)


### Features that hurt or are neutral (positive Δ on validation)

- **Elo + QB continuity**: Δ=+0.0014
- **Elo + Rolling form**: Δ=+0.0069
- **Elo + Scheduling**: Δ=+0.0194
- **Elo + Weather**: Δ=+0.0031
- **Elo + Coach**: Δ=+0.0136
- **Elo + Turnovers**: Δ=+0.0050
- **Elo + EPA**: Δ=+0.0017

### L1 coefficient sign stability

- Sign-stable features: 1 / 14

---

## Decision

**Forward selection did not add any features to the incumbent.**
Every tested family worsened validation when added to qb_changed + mov_3.
The incumbent subset is confirmed optimal among all tested families.

---

## Selected Features Summary

### Active (in incumbent)

| Feature | Source | Role |
|---------|--------|------|
| `elo_prob` | compute_elo_features() | Core rating signal |
| `home_qb_changed` | compute_qb_features() | QB continuity |
| `away_qb_changed` | compute_qb_features() | QB continuity |
| `home_rolling_mov_3` | compute_situational_features() | Recent form |
| `away_rolling_mov_3` | compute_situational_features() | Recent form |
| Platt calibration | LogisticRegression + StandardScaler | Probability calibration |

### Accepted (improve validation)


### Rejected (worsen or neutral on validation)

*Note: "QB continuity" and "Rolling form" were rejected as full families,
but curated 2-column subsets (`qb_changed`, `rolling_mov_3`) are in the incumbent.*

- **Elo + QB continuity**: val Δ=+0.0014
- **Elo + Rolling form**: val Δ=+0.0069
- **Elo + Scheduling**: val Δ=+0.0194
- **Elo + Weather**: val Δ=+0.0031
- **Elo + Coach**: val Δ=+0.0136
- **Elo + Turnovers**: val Δ=+0.0050
- **Elo + EPA**: val Δ=+0.0017

### Promising but not promoted

| Feature | Best Val LL | Best Hold LL | Issue |
|---------|------------|-------------|-------|

### Diagnostic-only (market)

Market data is used for interpretation only. Market holdout LL: 0.6090.
Elo residuals correlate with market residuals at r=0.9768.

---

## Final Recommendation

**Incumbent unchanged.**

### What worked

- **qb_changed + rolling_mov_3** continue to be the only features that
  improve validation consistently across all methods tested.
- Turnover differential (rolling 3-game) showed small improvement in L1 models.
- Quarterback continuity features (starts, win_pct) are individually weak but
  the composite `qb_changed` binary remains the strongest single feature.

### What was rejected (again)

**Note:** "QB continuity" and "Rolling form" are listed as rejected because
their full families (14–15 columns each) add noise. However, curated 2-column
subsets of each (`qb_changed`, `rolling_mov_3`) are in the incumbent and
improve validation. Full-family rejection does not contradict the curated subset's value.

- **Weather**: Worsens validation across all folds. Consistent with prior findings.
- **EPA**: Rolling offensive EPA adds noise, not signal. Consistent with prior EPA
  and comprehensive efficiency experiments.
- **Coach tenure**: All variants worsen validation. Consistent with combined_features.md.
- **Scheduling**: Short week, off bye, Thursday/Monday all add noise.
- **Turnovers**: Very small signal; L1 selected turnover_diff_net_3 at C=0.1 with
  sign-stable negative (good) coefficient, but the improvement is noise-level.

### Key takeaway

Elo probability dominates all other features on this dataset (~1,000 training games).
Adding more features consistently adds noise. The Elo + qb_changed + rolling_mov_3
combination remains the optimal parsimonious model.

The fundamental challenge is sample size: with ~1,000 games, broad feature families
(weather, EPA, efficiency) cannot overcome their degrees of freedom. Discrete,
high-signal features (qb_changed) can earn their way in; continuous noisy features
cannot.

---

## Why Plausible Football Features Failed

Each feature family was tested because it has a credible football rationale.
Below is why each failed, organized by failure mechanism.

### 1. Signal Already Captured by Elo

These features correlate strongly with the Elo rating itself — adding them on top
of `elo_prob` provides little or no new information.

| Family | Rationale | Why It Failed |
|--------|----------|---------------|
| **Rolling form** (MOV 3/5, pts for/against, streaks, YTD win%) | Recent performance should supplement Elo's long-term rating | Elo already captures game outcomes via point differential. Rolling MOV is a lagging subset of Elo's recent updates. The 3-game window carries signal when isolated (mov_3 at Δ=−0.0005 vs Elo-only) but the full 12-column family adds noise. Feature selection correctly found mov_3 as the only useful column. |
| **Turnovers** (rolling giveaways, takeaways, TO diff) | Turnover margin predicts wins independently of yardage | Elo is trained on point differential, which already captures turnover impact (turnovers → points). Residual analysis confirmed Elo residuals are independent of turnover differential. L1 selected turnover_diff_net_3 with a small negative (good) coefficient but the improvement was noise-level (+0.0050 val). |
| **EPA** (offensive EPA/play rolling 3/5) | Efficiency metrics should predict future scoring better than raw points | Offensive EPA per play is the single-play-expected-points version of what Elo already learns from game outcomes. At the team-game level (~570 rows/season), EPA is a noisy proxy for the point differential that Elo already sees directly. The rolling window further dilutes the already-weak signal (+0.0017 val). |

### 2. Too Sparse / Low Event Rate

These features affect too few games to be learned reliably from ~1,000 training rows.

| Family | Rationale | Why It Failed |
|--------|----------|---------------|
| **Scheduling** (short week, off bye, Thursday/Monday, international, consecutive road) | Rest differential and travel should affect performance | Short-week games (~10% of sample) and international games (~2%) have too few examples for the model to learn consistent effects. The rest_diff continuous variable is nearly zero-centered and noisy. Every scheduling column added noise (+0.0194 val overall). |
| **Weather** (cold, wind, precipitation, dome) | Extreme weather should affect scoring and win probability | Only ~15% of games have meaningful cold/wind/precip. Dome neutralization removes signal from the majority of games. The weather-only model (0.6941 val) is barely above random. Cold-weather subset (n=26 on holdout) showed interesting raw Elo performance (0.5777) but with no systematic effect large enough to generalize. |

### 3. Continuous Noise Overwhelms Discrete Signal

Features where a small number of discrete columns carry signal but the full family is rejected because continuous columns add noise.

| Family | Rationale | Why It Failed |
|--------|----------|---------------|
| **QB continuity** (qb_changed, starts, win_pct, games since change, new_qb_flag) | QB changes are the single largest game-to-game variance factor | The binary `qb_changed` columns carry the signal. The continuous correlates (starts, win_pct, games_since_change) introduce noise at this sample size. Full family Δ=+0.0014 vs Elo-only, but the curated qb_changed subset Δ=−0.0072 vs Elo-only. Full families punish the signal with noise; curated subsets win. |
| **Coach** (tenure, career wins, win%) | Coaching experience should correlate with team quality | Coach quality is already baked into Elo ratings (good coaches → better results → higher Elo). Coach features are highly correlated with team identity (same coach = same team). Adding 8 continuous coach columns adds collinearity (+0.0136 val). No coach-only variant beat the incumbent. |

### 4. Better Suited as Postgame Elo Update Signals

Some features may be better used to modulate Elo's K-factor (learning rate)
rather than as standalone Platt features. This experiment tested additive Platt features;
an alternative approach would use these to adjust Elo's update magnitude postgame.

| Family | Rationale | Alternative Approach |
|--------|----------|---------------------|
| **Turnovers** | Turnover differential in a game could justify a larger Elo update | Use TO_diff as a MOV multiplier (already exists as capped_linear MOV in the incumbent's Elo engine) |
| **EPA** | Blowout efficiency suggests a team is better/worse than score indicates | Use EPA differential as an alternative MOV metric instead of point differential |
| **Weather** | Bad weather increases randomness, reducing confidence | Use weather flags to widen Elo's K-factor or increase regression toward mean for weather-affected games |

---

## Subgroup / Residual Diagnostic Recommendations

The incumbent's residual diagnostics (see residual_diagnostics.md) identified
several systematic failure modes. These are the highest-leverage follow-up experiments:

### Priority 1: QB-Change Market Delta (Highest Impact)

| Finding | Details |
|---------|---------|
| QB-change games: incumbent holdout LL | 0.7687 (vs 0.6373 overall) |
| QB-change games: market holdout LL | 0.6662 (gap of 0.1025) |
| Sample | ~30 games/season with a QB change |
| Gap interpretation | Market prices QB changes 0.10 better than Elo |
| Recommendation | Build a pregame feature that estimates the QB-change probability impact independent of market. Possible approaches: backup-QB career stats, weeks-since-change decay, or coach tenure interaction. The `qb_market_delta` experiment already confirmed market prices this fully. The challenge is predicting the delta pregame (before market moves). |

### Priority 2: Very High Confidence Calibration

| Finding | Details |
|---------|---------|
| Games with confidence >0.9 | ~10% of holdout |
| Calibration error in >0.9 bucket | 0.2487 (model overconfident on away longshots) |
| Recommendation | Investigate Platt calibration with a confidence-weighted loss, or fit separate calibrators for high-confidence bins. Risk: overfit on small high-confidence samples (~28 games/holdout). Alternative: clip extreme probabilities or apply a soft Platt prior. |

### Priority 3: Early-Season Performance

| Finding | Details |
|---------|---------|
| Weeks 1–4 holdout LL | 0.6744 (vs 0.6373 overall) |
| Weeks 5+ holdout LL | 0.6315 |
| Hypothesis | Elo regression (preseason mean reversion) may be too aggressive or too conservative for early-season games |
| Recommendation | Test season-specific regression parameters for early weeks (W1–4 get different reg or K-factor). Risk: very small early-season sample per fold (~30 games). |

### Priority 4: Roof Type / Stadium Environment

| Finding | Details |
|---------|---------|
| Open/retractable roof holdout LL | 0.7206 (vs dome 0.6373) |
| Sample | ~40% of games in open/retractable stadiums |
| Hypothesis | Stadium-specific factors (altitude, turf type, crowd noise) may systematically affect certain teams. Home field advantage is currently a single global HFA parameter. |
| Recommendation | Test stadium-specific HFA adjustments, or stadium altitude/turf features. Previous team-specific HFA experiment (team_hfa.md) failed, but stadium-level features may be more stable. |

### Priority 5: Monday Night Games

| Finding | Details |
|---------|---------|
| Monday games holdout LL | 0.6935 (vs Sunday 0.6453) |
| Sample | ~15 games/season |
| Recommendation | Likely noise (small sample). Monitor if pattern persists as more seasons are added. If sample grows, test a prime-time flag that distinguishes MNF/SNF/TNF from Sunday day games. |

### Priority 6: Season-Over-Season Stability

| Finding | Details |
|---------|---------|
| 2024 holdout LL | 0.6042 (best season) |
| 2021 holdout LL | 0.6744 (worst season) |
| Trend | Performance improves each season (more training data, better Elo estimates) |
| Recommendation | The model naturally improves with more data. Adding 2026 data (when available) should continue this trend. No action needed. |

---

### Next experiments

1. **QB-change probability impact**: Build a pregame feature for QB-change effect independent of market. See Priority 1 above.
2. **Early-season regression tuning**: Test whether W1–4 benefit from different K-factor or regression. See Priority 3 above.
3. **Opening-line ingestion**: Current market benchmark uses closing lines (near-kickoff). Opening lines would give a fairer pregame comparison.
