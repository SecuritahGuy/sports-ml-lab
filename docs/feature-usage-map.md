# Feature Usage Map

*A comprehensive reference for how features are used, tested, and rejected in this project.*

*Source: [`sports-ml-lab`](https://github.com/SecuritahGuy/sports-ml-lab)*

---

## A. Active Incumbent Features

The current research incumbent (**Standard Elo + qb_changed + rolling_mov_3 + Platt**) uses exactly 5 features in its Platt calibration layer:

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `elo_prob` | Continuous (0–1) | `compute_elo_features()` | Elo-implied home win probability from point-differential Elo |
| `home_qb_changed` | Binary | `compute_qb_features()` | 1 if home QB did not start team's prior game |
| `away_qb_changed` | Binary | `compute_qb_features()` | 1 if away QB did not start team's prior game |
| `home_rolling_mov_3` | Continuous | `compute_situational_features()` | Avg margin of victory over home team's last 3 games |
| `away_rolling_mov_3` | Continuous | `compute_situational_features()` | Avg margin of victory over away team's last 3 games |

The Elo rating engine itself has tunable parameters that act as implicit features:

| Parameter | Value | Role |
|-----------|-------|------|
| K-factor | 36 | Learning rate for rating updates |
| Home field advantage | 40 Elo points | Added to home team rating |
| Base preseason regression | 0.1 | Shrink toward league mean each offseason |
| QB-change bonus regression | 0.2 | Additional regression for teams with new starting QB |
| Decay half-life | 32 games | Exponential decay toward prior rating |
| MOV type | capped_linear (scale=0.05, cap=2.0) | Caps blowout influence on rating updates |

---

## B. Promising but Disputed Features

These features showed some signal but did not earn promotion under the strict validation+holdout rule:

| Feature | Best Validation LL | Best Holdout LL | Why Not Promoted |
|---------|-------------------|-----------------|------------------|
| `rolling_mov_3` alone (no qb_changed) | 0.6406 | **0.6255** | Wins holdout but not validation. The 0.6255 holdout beats the incumbent 0.6262, but validation is 0.6406 vs incumbent 0.6334. |
| `rolling_mov_1` (1-game window) | **0.6338** | 0.6302 | Wins validation but loses holdout — classic overfit. Not promoted. |
| Coach+QB season regression | 0.6309 | 0.6286 | Tiny validation win (0.0006) erased on holdout (-0.0001). |
| O/D Elo (ko52_kd20) + Platt | 0.6376 | 0.6258 | Better holdout but worse validation. Demoted to holdout-informed diagnostic. |
| QB injury flag (single binary) | 0.6464 | 0.6255 | Noise-level holdout improvement; validation 0.0088 worse. |

---

## C. Rejected Feature Families

All 20 rejected experiments. Each was tested via rolling-origin 3-fold validation
and the 2025 one-shot holdout.

### Weather (4 columns)

- **Features**: temperature, wind speed, precipitation flag, dome flag, cold/windy/bad-weather thresholds
- **Validation**: 0.6445 (vs incumbent 0.6363)
- **Holdout**: 0.6439 (vs incumbent 0.6373)
- **Decision**: Rejected — both worse
- **Report**: `reports/experiments/weather_features.md`

### Scheduling / Rest (6+ columns)

- **Features**: short week, off bye, Thursday/Monday flags, consecutive road games, international
- **Validation**: 0.6599 (vs incumbent 0.6363)
- **Holdout**: 0.6401 (vs incumbent 0.6373)
- **Decision**: Rejected — both worse
- **Report**: `reports/experiments/schedule_rest_features.md`

### QB Identity OHE

- **Features**: 93 one-hot encoded QB names
- **Holdout**: 14.51 log loss (catastrophic overfit)
- **Decision**: Rejected — 93 classes for 376 training rows
- **Report**: `reports/experiments/qb_features.md`

### Coach Tenure / Win Percentage

- **Features**: coach tenure (games/years), career wins, career win%
- **Decision**: Rejected — all variants worse on both val and holdout
- **Report**: `reports/experiments/combined_features.md`

### EPA / Team Efficiency (18 columns)

- **Features**: rolling 3/5 avg of passing/rushing/receiving/total EPA per play
- **Validation**: 0.6654 (vs incumbent 0.6363)
- **Holdout**: 0.6495 (vs incumbent 0.6373)
- **Decision**: Rejected — both worse; made QB-change failure mode worse
- **Report**: `reports/experiments/epa_features.md`

### Comprehensive Efficiency (58 columns, 3 sources)

- **Sources**: Team Stats Total EPA, PFR Advanced Stats (pass/rush/rec/def), Snap Counts
- **Validation**: 0.6597 (vs incumbent 0.6368)
- **Holdout**: 0.6788 (vs incumbent 0.6313)
- **Decision**: Rejected — 58 features added noise at this sample size
- **Report**: `reports/experiments/comprehensive_efficiency.md`

### Injury Report Features (20 columns)

- **Features**: QB OUT flags, position-group injury counts (RB/WR/TE/OL/DL/LB/DB), injury-driven QB change, net differentials
- **Validation**: 0.6486 (vs incumbent 0.6406)
- **Holdout**: 0.6514 (vs incumbent 0.6315/0.6285)
- **Decision**: Rejected — all worse
- **Report**: `reports/experiments/injury_features.md`

### Team Stats (yards / fantasy / sacks)

- **Features**: rolling aggregates from nflreadpy load_player_stats
- **Validation**: 0.6541 (vs incumbent 0.6368)
- **Holdout**: 0.6415 (vs incumbent 0.6285)
- **Decision**: Rejected — all variants worse
- **Report**: `reports/experiments/team_stats.md`

### Tree-Based Models (HGB / GB / RF)

- **Best validation**: RandomForest at 0.6329 on curated 27 features
- **Holdout**: 0.6456 (RF) — classic overfit pattern
- **Decision**: Rejected — won validation but lost holdout
- **Report**: `reports/experiments/expressive_models.md`

### AutoGluon AutoML

- **Features**: sklearn-only ensemble (RF, ExtraTrees) on 47 pregame features
- **Validation**: 0.6956 (vs Platt 0.6376)
- **Holdout**: 0.6404 (vs Platt 0.6362)
- **Decision**: Rejected — both worse
- **Report**: `reports/experiments/autogluon.md`

### Glicko Rating System

- **Configurations**: 432 (4 HFA × 6 init_RD × 6 sys_c × 3 QB bonus)
- **Best validation**: 0.6513 (worse than Elo)
- **Best holdout**: 0.7013 (far worse than Elo)
- **Decision**: Rejected — all 432 configs worse
- **Report**: `reports/experiments/glicko_rating.md`

### Team-Specific HFA

- **Validation**: 0.6355 (worse than global HFA 0.6321)
- **Holdout**: 0.6263 (better than incumbent, but val rules)
- **Decision**: Rejected — worse validation despite better holdout
- **Report**: `reports/experiments/team_hfa.md`

### Home/Away Separate Elo

- **Validation**: 0.6622 (vs standard 0.6410)
- **Holdout**: 0.6634
- **Decision**: Rejected — noisier per-split ratings
- **Report**: `reports/experiments/home_away_elo.md`

### Rolling MOV Windows ≠ 3

- **mov_1**: Won validation (0.6338) but lost holdout (0.6302 vs 0.6262)
- **mov_2+**: All worse on validation
- **Decision**: Rejected — mov_3 confirmed optimal
- **Report**: `reports/experiments/rolling_mov_sensitivity.md`

### Confidence Calibration (temperature, isotonic, shrinkage)

- **Best**: Temperature T=1.50 — tied incumbent on val (0.6374) and holdout (0.6373)
- **Decision**: Rejected — no method beat Platt on both val and holdout
- **Report**: `reports/experiments/confidence_calibration.md`

### Residual Blending

- **Approach**: Logistic blend on elo_prob + week/rest/early-season features
- **Decision**: Rejected — all blends worse
- **Report**: `reports/experiments/residual_blending.md`

### Coach+QB Season Regression

- **Validation**: 0.6309 (wins by 0.0006)
- **Holdout**: 0.6286 (loses by 0.0001)
- **Decision**: Rejected — signal too weak
- **Report**: `reports/experiments/coach_season_regression.md`

### QB Injury Flag (single binary)

- **Holdout**: 0.6255 (noise-level 0.0003 improvement)
- **Validation**: 0.6464 (0.0088 worse)
- **Decision**: Rejected — noise-level improvement
- **Report**: `reports/experiments/qb_injury_flag.md`

---

## D. Diagnostic-Only Features (Market Data)

These are NOT part of the football-only model. They are used for benchmarking and diagnostic comparison only:

| Feature | Source | Description |
|---------|--------|-------------|
| `market_prob_diagnostic` | Closing moneyline (no-vig) | Market-implied home win probability |
| `market_minus_model_diagnostic` | Derived | Model error relative to market |
| Caution flag: model-market disagreement | Derived | Gap > 0.15 triggers caution flag |

**Market holdout log loss: 0.6090** — significantly better than the football-only incumbent (0.6262). The market is the true performance ceiling. Our Elo residuals correlate with market residuals at r=0.9768.

---

## E. Key Lesson

This project has tested **14+ feature families** (58 columns in the largest). Nearly all were rejected because:

1. **Small sample**: ~1,000 training games (2021–2024). Broad features (EPA, efficiency, weather) add noise, not signal.
2. **Elo already captures most signal**: Elo probability correlates with market at r=0.9768. Adding weak features on top is difficult.
3. **Discrete signals > continuous noise**: The two features that earned promotion (qb_changed, rolling_mov_3) are discrete or simple rolling aggregates. Broad continuous feature groups consistently overfit.
4. **Validation before holdout**: Multiple features (mov_1, team HFA, O/D Elo) looked good on holdout but failed on validation. The strict validation-first rule prevents false promotions.

Every feature must earn its way in through chronological validation. Features are not ignored — they are tested systematically and rejected honestly.
