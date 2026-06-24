# Feature Selection Experiment

*Systematic forward feature selection on top of Elo+Platt.*

## Method

Rolling-origin 3-fold validation. Each feature group tested individually on top of Elo probability + logistic regression. Forward selection builds up the best combination.

### Elo Params

K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2

### Candidate Features

| Group | Columns | Description |
|-------|---------|-------------|
| `rolling_mov_3` | home/away | Avg MOV last 3 games |
| `rolling_mov_5` | home/away | Avg MOV last 5 games |
| `rolling_pts_for` | home/away | Season avg points scored |
| `rolling_pts_against` | home/away | Season avg points allowed |
| `win_streak` | home/away | Current win/loss streak |
| `ytd_win_pct` | home/away | Season win % YTD |
| `turf_flag` | binary | Artificial turf surface |
| `high_altitude` | binary | DEN/MEX altitude stadiums |
| `prime_time` | binary | Mon/Thu/Sun night or 8PM+ |
| `rest_diff_squared` | numeric | Nonlinear rest edge |

## Individual Feature Results

| Feature | Avg Val LL | Δ vs Platt | Fold1 | Fold2 | Fold3 |
|--------|-----------|-----------|-------|-------|-------|
| Platt (incumbent) | 0.6406 | — | 0.6471 | 0.6621 | 0.6126 |
| rolling_mov_3 | 0.6406 | +0.0000 | 0.6495 | 0.6593 | 0.6131 |
| rolling_pts_against | 0.6409 | +0.0003 | 0.6419 | 0.6661 | 0.6146 |
| high_altitude | 0.6409 | +0.0003 | 0.6477 | 0.6624 | 0.6126 |
| win_streak | 0.6413 | +0.0007 | 0.6492 | 0.6638 | 0.6108 |
| prime_time | 0.6413 | +0.0007 | 0.6482 | 0.6631 | 0.6126 |
| ytd_win_pct | 0.6419 | +0.0013 | 0.6482 | 0.6652 | 0.6124 |
| turf_flag | 0.6422 | +0.0016 | 0.6492 | 0.6653 | 0.6121 |
| rest_diff_squared | 0.6423 | +0.0017 | 0.6442 | 0.6639 | 0.6188 |
| rolling_mov_5 | 0.6434 | +0.0028 | 0.6567 | 0.6589 | 0.6145 |
| rolling_pts_for | 0.6434 | +0.0028 | 0.6587 | 0.6567 | 0.6148 |
| All situational | 0.6554 | +0.0148 | 0.6731 | 0.6669 | 0.6263 |

## QB Subset Results

| Feature | Avg Val LL | Δ vs Platt |
|--------|-----------|-----------|
| qb_changed | 0.6334 | -0.0072 |
| new_qb | 0.6393 | -0.0013 |
| games_since_change | 0.6393 | -0.0013 |
| qb_win_pct | 0.6436 | +0.0030 |

## Forward Selection

Starting from Platt baseline (0.6406).

| Round | Added | Val LL | Δ |
|-------|-------|--------|---|
| 1 | qb_changed | — | — |
| Final | qb_changed | 0.6334 | -0.0072 |

## 2025 Holdout

| Model | Holdout LL |
|-------|-----------|
| Platt (incumbent) | 0.6315 |
| Best (rolling_mov_3) | 0.6255 |
| Forward selection | 0.6314 |

## Decision

**No feature beats the incumbent on validation.**

Forward selection converged on: qb_changed.
