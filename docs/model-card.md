# Model Card: v3.0.0

[Home](index) | [2026 Schedule](2026-schedule) | [Benchmarks](benchmarks) | [Predictions](predictions) | **Model Card** | [Experiments](experiments) | [Backtests](backtests)

---


## Model Identity

| Attribute | Value |
|-----------|-------|
| **Name** | Standard Elo + qb_changed + rolling_mov_3 + Platt + frozen QB overlay |
| **Version** | v3.0.0 |
| **Date** | 2026-06-29 |
| **Type** | Elo ratings + Platt logistic + frozen QB overlay |

## Architecture (Two-Layer)

```
Layer 0: Elo ratings (K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2)
         → elo_prob (raw Elo home win probability)

Layer 1: Platt logistic on [elo_prob, qb_changed, rolling_mov_3]
         → base probability

Layer 2: Frozen QB overlay (logit-space additive, gated)
         Gate: qb_changed OR career_starts < 17 (either side)
         Adjustment: gamma=1.0 * clip(qb_adj, ±40) * ln(10)/400
         Applied only when gate is active
         → final probability
```

## Performance

| Metric | Value |
|--------|-------|
| Avg validation log loss | 0.6305 |
| 2025 holdout log loss | **0.6200** |
| 2025 holdout Brier | 0.2157 |
| 2025 holdout AUC | 0.7098 |
| 2025 holdout Accuracy | 0.6630 |

## Feature Set

Exactly 5 features enter the Platt logistic:
1. `elo_prob` — raw Elo home win probability
2. `home_qb_changed` — home QB changed from prior game
3. `away_qb_changed` — away QB changed from prior game
4. `home_rolling_mov_3` — home margin of victory, last 3 games
5. `away_rolling_mov_3` — away margin of victory, last 3 games

The frozen QB overlay uses additional computed signals:
- `home_qb_adj`, `away_qb_adj` — Bayesian-shrunken per-QB Elo ratings
- `home_qb_team_starts_pre`, `away_qb_team_starts_pre` — career starts
- `home_qb_changed`, `away_qb_changed` — QB change flags

## Leakage Prevention

- Elo computed chronologically: emit probability before updating rating
- Rolling MOV excludes current game from its window
- QB change detected from previous game only (no future knowledge)
- Preseason regression resets at season boundaries
- Decay halflife = 32 games, applied chronologically
- Ties excluded from Platt training, treated as 0.5 in Elo updates
- 2025 holdout never accessed during model selection

## Training Data

- Seasons: 2021–2024 (1,356 eligible games)
- Rolling-origin 3-fold validation
- 2025 season: locked holdout (276 games)

## Known Limitations

1. **Cold start**: Elo decayed after offseason; rolling_mov_3 = 0 in week 1
2. **QB change oracle**: Uses final actual starter (not pregame-announced)
3. **Market gap**: Market benchmark 0.6090 vs incumbent 0.6200
4. **Small sample**: ~1,000 training games limits feature complexity
5. **No roster features**: Injuries, depth chart changes only via Elo decay
