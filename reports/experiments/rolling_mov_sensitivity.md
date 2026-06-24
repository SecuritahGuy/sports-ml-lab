# Rolling MOV Sensitivity Experiment

Tests whether the promoted `rolling_mov_3` window is optimal vs other window sizes and functional forms.

## Method

Rolling-origin 3-fold validation. One-shot 2025 holdout.

### Elo Spine

K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2

### Candidate Features

| Name | Description | Columns |
|------|-------------|---------|
| mov_1 | Rolling avg MOV last 1 games | home/away_rolling_mov_1 |
| mov_2 | Rolling avg MOV last 2 games | home/away_rolling_mov_2 |
| mov_3 | Rolling avg MOV last 3 games | home/away_rolling_mov_3 |
| mov_4 | Rolling avg MOV last 4 games | home/away_rolling_mov_4 |
| mov_5 | Rolling avg MOV last 5 games | home/away_rolling_mov_5 |
| mov_6 | Rolling avg MOV last 6 games | home/away_rolling_mov_6 |
| mov_8 | Rolling avg MOV last 8 games | home/away_rolling_mov_8 |
| mov_10 | Rolling avg MOV last 10 games | home/away_rolling_mov_10 |
| mov_diff | Home MOV - Away MOV | rolling_mov_diff |
| mov_capped | Capped diff [-14, 14] | rolling_mov_capped |
| mov_log_signed | Signed log(1+|diff|) | rolling_mov_log_signed |
| mov_ewma | EWMA MOV (alpha=0.5) | rolling_mov_ewma |
| mov_std_3 | MOV volatility 3-game | rolling_mov_std_3 |
| mov_std_5 | MOV volatility 5-game | rolling_mov_std_5 |

### Leakage Prevention

- Rolling MOV uses only previous games (shifted)
- Current game result excluded
- Season boundaries: team history reset per season
- Early-season: defaults to 0.0 (no prior games)
- EWMA computed from historical games only

### Rolling-Origin Folds

- Train [2021] → Validate 2022
- Train [2021, 2022] → Validate 2023
- Train [2021, 2022, 2023] → Validate 2024
- Holdout: 2025

## Validation Results

| Model | Alone | +qb_changed |
|------|-------|-------------|
| Platt | — | — |
| qb_changed only | — | 0.6334 |
| Incumbent (qb+mov_3) | — | 0.6348 |
| mov_1 | 0.6411 | 0.6338 |
| mov_10 | 0.6471 | 0.6403 |
| mov_2 | 0.6424 | 0.6348 |
| mov_3 | 0.6419 | 0.6348 |
| mov_4 | 0.6449 | 0.6381 |
| mov_5 | 0.6460 | 0.6392 |
| mov_6 | 0.6452 | 0.6384 |
| mov_8 | 0.6468 | 0.6400 |
| mov_capped | 0.6420 | 0.6351 |
| mov_diff | 0.6422 | 0.6355 |
| mov_ewma | 0.6416 | 0.6346 |
| mov_log_signed | 0.6428 | 0.6362 |
| mov_std_3 | 0.6420 | 0.6349 |
| mov_std_5 | 0.6420 | 0.6350 |
| qb+mov_3+mov_5 | — | 0.6403 |

### Best Validation Configurations

Best with qb_changed: **mov_1** (0.6338)

Best without qb_changed: **mov_1** (0.6411)

## 2025 Holdout (one-shot)

| Model | Holdout LL |
|-------|-----------|
| Platt baseline | 0.6315 |
| qb_changed only | 0.6314 |
| Incumbent (qb+mov_3) | 0.6262 |
| Selected (mov_1) | 0.6302 |

## Decision

**Diagnostic only:** mov_1 beats incumbent on val (0.6338 vs 0.6348) but not holdout (0.6302 vs 0.6262).
