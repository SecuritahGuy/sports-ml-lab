# Optuna Joint Elo Parameter Search

*Jointly optimizing all Elo parameters simultaneously with rolling-origin 3-fold validation.*

## Method

- **Optimizer**: Optuna TPESampler, 200 trials
- **Search space** (10 parameters):
  - K: 20–60
  - HFA: 10–50
  - reg (base): 0.0–0.5
  - decay half-life: 16–64 games
  - qb_bonus: 0.0–0.5
  - k_off: 20–80
  - k_def: 10–60
  - MOV type: ['none', 'capped_linear', 'log', 'sqrt', 'capped_log']
  - MOV scale: 0.01–0.15 (log, only if MOV != none)
  - MOV cap: 1.5–5.0 (only for capped types)
- **Objective**: Average validation log loss across 3 rolling-origin folds
- **Calibration**: Platt scaling on each fold

## Best Params

| Parameter | Value |
|-----------|-------|
| decay | 56 |
| hfa | 16 |
| k | 44 |
| k_def | 34 |
| k_off | 24 |
| mov_cap | 4.122188198671257 |
| mov_scale | 0.09583901047677328 |
| mov_type | capped_linear |
| qb_bonus | 0.010067202192372502 |
| reg | 0.15825691858772073 |
| best_val_ll | 0.6342 |

## Holdout Results

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|-----|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | 0.5000 | — |
| Incumbent (O/D Elo + Platt) | 0.6258 | 0.2179 | 0.7066 | 0.6703 |
| **Optuna best + Platt** | 0.6318 | 0.2206 | 0.7002 | 0.6486 |

**Incumbent retains champion.** Optuna search could not beat incumbent (0.0060 worse on holdout).

## Top 10 Trials

| Trial | Avg Val LL | K | HFA | Reg | Decay | QB Bonus | k_off | k_def | MOV |
|-------|-----------|----|-----|------|-------|---------|-------|-------|-----|
| 185 | 0.6342 | 44 | 16 | 0.16 | 56 | 0.01 | 24 | 34 | capped_linear |
