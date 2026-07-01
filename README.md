# SportsLab — NFL Prediction Research

A reproducible ML research lab for pregame NFL win prediction. Feature-governed, leakage-safe, market-free.

**Champion:** v3.0.0 Frozen QB Overlay (2025 holdout LL: 0.6200)

## Quick Start

```bash
make install        # install dependencies
make test           # run 669 tests
make lint           # ruff check
sportslab build-dashboard  # generate docs site
```

## Weekly Prediction Pipeline

```bash
# Preseason / Week 1: depth chart snapshot
sportslab predict-week --season 2026 --week 1 --mode dry_run

# Week 2+: week-over-week QB tracking (88% vs 67% accuracy)
sportslab predict-week --season 2026 --week 2 --weekly-qb --mode live

# Grade after games are played
sportslab grade-week --season 2026 --week 1 --mode live

# Audit QB sourcing before locking predictions
sportslab weekly-qb-audit --season 2026 --week 2
```

## Key Results

| Model | Holdout LL | Brier | AUC |
|-------|-----------|-------|-----|
| v3.0.0 Frozen QB Overlay | **0.6200** | 0.2157 | 0.7098 |
| v2.0.0 Elo + qb_changed + mov3 | 0.6262 | 0.2180 | 0.7050 |
| Raw Elo + Platt | 0.6373 | — | — |
| Market (no-vig) | 0.6090 | — | — |

Market is diagnostic-only baseline. All football-only champions are market-free.

## Architecture

```
Two-layer:
  Layer 1: Elo (K=36, HFA=40, reg=0.1, decay=32) + qb_changed + mov_3 + Platt
  Layer 2: Frozen QB overlay — logit-space adjustment gated on changed QB or <17 starts
```

43 experiments completed, 30+ feature families tested and documented.
