# Optuna Feature Selection

*Combinatorial search over 16 feature groups using Optuna TPE.*

## Method

Rolling-origin 3-fold validation. Each feature group is a boolean inclusion variable (0/1). 500 trials with TPE sampler, MedianPruner. Objective: avg val log loss.

Incumbent: qb_changed + rolling_mov_3 (val LL = 0.6334).

### Feature Groups

| Group | Columns |
|-------|---------|
| `coach_tenure` | home_coach_tenure, away_coach_tenure |
| `coach_win_pct` | home_coach_win_pct, away_coach_win_pct |
| `games_since_change` | home_games_since_qb_change, away_games_since_qb_change |
| `high_altitude` | high_altitude_flag |
| `new_qb` | home_new_qb_flag, away_new_qb_flag |
| `prime_time` | prime_time_flag |
| `qb_changed` | home_qb_changed, away_qb_changed |
| `qb_win_pct` | home_qb_win_pct_pre, away_qb_win_pct_pre |
| `rest_diff_squared` | rest_diff_squared |
| `rolling_mov_3` | home_rolling_mov_3, away_rolling_mov_3 |
| `rolling_mov_5` | home_rolling_mov_5, away_rolling_mov_5 |
| `rolling_pts_against` | home_rolling_pts_against, away_rolling_pts_against |
| `rolling_pts_for` | home_rolling_pts_for, away_rolling_pts_for |
| `turf_flag` | turf_flag |
| `win_streak` | home_win_streak, away_win_streak |
| `ytd_win_pct` | home_ytd_win_pct, away_ytd_win_pct |

### Elo Params

K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2

## Results

**Trials completed:** 500
**Pruned trials:** 5

### Best Configuration

| Group | Included |
|-------|----------|
| `coach_tenure` | ✗ |
| `coach_win_pct` | ✓ |
| `games_since_change` | ✗ |
| `high_altitude` | ✗ |
| `new_qb` | ✗ |
| `prime_time` | ✗ |
| `qb_changed` | ✓ |
| `qb_win_pct` | ✗ |
| `rest_diff_squared` | ✗ |
| `rolling_mov_3` | ✗ |
| `rolling_mov_5` | ✗ |
| `rolling_pts_against` | ✗ |
| `rolling_pts_for` | ✗ |
| `turf_flag` | ✗ |
| `win_streak` | ✓ |
| `ytd_win_pct` | ✗ |

**Active groups:** coach_win_pct, qb_changed, win_streak

**Total feature columns:** 6

**Validation LL:** 0.6334

### Validation Comparison

| Model | Avg Val LL |
|-------|-----------|
| Platt baseline | 0.6406 |
| Incumbent | 0.6334 |
| Optuna best | 0.6334 |

### Holdout Comparison

| Model | Holdout LL |
|-------|-----------|
| Platt baseline | 0.6315 |
| Incumbent | 0.6262 |
| Optuna best | 0.6347 |

### Feature Importance (TPE param importance)

| Group | Importance |
|-------|-----------|
| `rolling_pts_for` | 0.2277 |
| `qb_changed` | 0.1354 |
| `coach_tenure` | 0.1292 |
| `rolling_mov_5` | 0.1265 |
| `games_since_change` | 0.1095 |
| `turf_flag` | 0.0479 |
| `rolling_pts_against` | 0.0425 |
| `qb_win_pct` | 0.0398 |
| `ytd_win_pct` | 0.0329 |
| `high_altitude` | 0.0326 |
| `win_streak` | 0.0273 |
| `rest_diff_squared` | 0.0136 |
| `prime_time` | 0.0118 |
| `coach_win_pct` | 0.0114 |
| `new_qb` | 0.0062 |
| `rolling_mov_3` | 0.0057 |

## Decision

**No improvement.** Optuna best (0.6334) does not beat incumbent (0.6334) on validation.
Greedy forward selection result is validated.
