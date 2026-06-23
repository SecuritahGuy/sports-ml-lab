# Residual Blending Experiment

*Testing whether adding simple game-context features to Elo probability improves on the incumbent.*

## Motivation

Residual diagnostics showed systematic errors by week, rest, and confidence.  The question is whether a tiny logistic model on Elo prob + 1-2 context features can reduce these errors without overfitting.

## Feature Sets Tested

| Setup | Features |
|-------|----------|
| Platt (incumbent) | elo_prob only |
| Elo + week | elo_prob, week |
| Elo + week + rest_diff | elo_prob, week, rest_diff |
| Elo + early_season | elo_prob, flag(week <= 5) |
| Elo + week (no Platt) | elo_prob + week via LR (no Platt step) |

## Rolling-Origin Results

| Model | Avg Val Platt LL | Avg Val Blend LL |
|-------|-----------------|-----------------|
| Platt (incumbent) | 0.63684 | 0.64463 |
| Elo + week | 0.63684 | 0.64219 |
| Elo + week + rest_diff | 0.63684 | 0.64279 |
| Elo + early_season | 0.63684 | 0.64256 |

## Holdout (2025) Results

| Model | Hold LL | Brier | Acc | AUC | vs Platt |
|-------|---------|-------|-----|-----|----------|
| Platt (incumbent) | 0.6303 | 0.2199 | 0.6703 | 0.7024 | -0.0018 |
| Elo + early_season | 0.6330 | 0.2213 | 0.6486 | 0.6948 | -0.0045 |
| Elo + week + rest_diff | 0.6355 | 0.2225 | 0.6377 | 0.6893 | -0.0069 |
| Elo + week | 0.6355 | 0.2224 | 0.6377 | 0.6905 | -0.0070 |
| Elo + week (no Platt) | 0.6355 | — | — | — | — |

## Incumbent Comparison

Incumbent (season-reg + Platt) holdout: 0.6285

## Decision

❌ **Residual blending does not beat the incumbent.**

## Leakage Prevention

- All features are pregame-safe (week, rest diff, elo_prob).
- Rolling-origin folds prevent 2025 holdout access.
- Blend models fitted only on training data per fold.
- Holdout evaluated only once after model selection.

