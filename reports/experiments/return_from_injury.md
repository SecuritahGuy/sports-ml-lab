# Return-from-Injury Rust Features

Testing whether players returning from multi-game absences cause teams
to underperform their Elo expectation.

## Data

- Injury sources: 5 seasons (2021-2025), nflreadpy import_injuries()
- Return events: players with 2+ consecutive "Out" weeks, then not Out
- Rust weight: position_weight × sqrt(games_missed)
- Weights: QB=5.0, RB=3.0, WR/TE=2.0, OL=1.5, DL/LB/DB=1.0

## Variants

| ID | Model | Rust Features |
|---|-------|--------------|
| A | Incumbent (Pi-Ratings + qb_changed + mov_3 + Platt) | None |
| B | Incumbent + All Rust | 8 columns: score, QB, skill, games_missed (H/A) |
| C | Incumbent + QB Rust | home_rust_qb, away_rust_qb |
| D | Incumbent + Skill Rust | home_rust_skill, away_rust_skill |

## Rust Coverage

Games with any rust feature active: 601/1388

Injury data rows loaded: from [2021, 2022, 2023, 2024, 2025]


## Validation (Rolling-Origin 3-Fold)

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| Incumbent (Pi only) | 0.6266 | 0.6313 | 0.6485 | 0.5999 |
| Incumbent + Rust (all 8) | 0.6535 | 0.7030 | 0.6513 | 0.6062 |
| Incumbent + Rust (QB only) | 0.6395 | 0.6686 | 0.6486 | 0.6012 |
| Incumbent + Rust (Skill only) | 0.6410 | 0.6687 | 0.6497 | 0.6047 |

## Holdout (2025)

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|------|
| Incumbent (Pi only) | 0.6350 | 0.2217 | 0.6962 | 0.6268 |
| Incumbent + Rust (all 8) | 0.6351 | 0.2221 | 0.6971 | 0.6304 |
| Incumbent + Rust (QB only) | 0.6326 | 0.2206 | 0.6994 | 0.6304 |
| Incumbent + Rust (Skill only) | 0.6412 | 0.2241 | 0.6906 | 0.6232 |

## Comparison vs Incumbent

Incumbent (A): val=0.6266, hold=0.6350

| Model | Δval | Δhold | Decision |
|-------|------|-------|----------|
| Incumbent + Rust (all 8) | +0.0269 | +0.0000 | Worse on both |
| Incumbent + Rust (QB only) | +0.0129 | -0.0024 | Loses val, wins hold |
| Incumbent + Rust (Skill only) | +0.0145 | +0.0062 | Worse on both |

## Decision

**No rust variant beats incumbent on both val and holdout by ≥ 0.001.**

---
Report: return_from_injury_experiment.py