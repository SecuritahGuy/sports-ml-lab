# Comprehensive Efficiency Features Experiment

*Testing pregame efficiency features from 3 nflreadpy sources (Team Stats Total EPA, PFR Advanced Stats, Snap Counts) on top of the incumbent.*

## Data Sources

| Source | Description | Rows/Season | Level |
|--------|-------------|-------------|------|
| `load_team_stats` | Game-level passing_epa, rushing_epa, receiving_epa (totals) | ~570 | team-game |
| `load_pfr_advstats` (pass/rush/rec/def) | Pressure rate, bad throws, YAC, broken tackles, def passer rating, missed tackles | ~700-8000 | player-week |
| `load_snap_counts` | OL snap%, top RB snap% | ~26000 | player-week |

## Feature Groups

| Group | Features | Count |
|-------|----------|-------|
| Team Stats Total EPA | Rolling 3/5 of pass_epa, rush_epa, rec_epa, total_epa + net diffs | 18 |
| PFR Advanced Stats | Pressure rate, bad throw rate, YAC/rush, broken tackles/rush, def passer rating, def missed tackle % + net diffs | 30 |
| Snap Counts | OL snap%, top RB snap% + net diffs | 10 |
| **Total** | | **58** |

## Leakage Prevention

- All features computed chronologically, shifted (current game excluded)
- Rolling windows reset at season boundaries
- New season games use 0 imputation + missing flags
- Rolling-origin validation prevents 2025 holdout from influencing selection

## Incumbent Params

| Parameter | Value |
|-----------|-------|
| K-factor | 36 |
| Home-field advantage | 40 |
| MOV type | capped_linear |
| Features | elo_prob + qb_changed + rolling_mov_3 |
| Calibration | Platt scaling |
| Holdout LL | 0.6262 |

## Models Compared

| Model | Description |
|-------|-------------|
| **Platt (incumbent)** | Elo + qb_changed + mov_3 + Platt |
| **Efficiency only** | Logistic on all efficiency features |
| **Incumbent + Efficiency** | Logistic on elo + qb + mov_3 + efficiency features |
| **Team EPA only** | Logistic on team_stats total EPA features |
| **PFR only** | Logistic on PFR advanced stats |
| **Snap only** | Logistic on snap count features |

## Rolling-Origin Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Platt (incumbent) | 0.6368 | 0.6427 | 0.6568 | 0.6110 |
| Efficiency only | 0.7082 | 0.7159 | 0.7102 | 0.6984 |
| Incumbent + Efficiency | 0.6597 | 0.6714 | 0.6845 | 0.6232 |
| Team EPA only | 0.6889 | 0.6819 | 0.7020 | 0.6827 |
| PFR only | 0.7047 | 0.7108 | 0.7030 | 0.7004 |
| Snap only | 0.6918 | 0.6956 | 0.6784 | 0.7013 |

## 2025 Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|-----|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6313 | 0.2204 | 0.6995 | 0.6449 |
| Efficiency only | 0.7171 | 0.2552 | 0.5921 | 0.5725 |
| Incumbent + Efficiency | 0.6788 | 0.2363 | 0.6570 | 0.6377 |
| Team EPA only | 0.6842 | 0.2457 | 0.5608 | 0.5217 |
| PFR only | 0.7056 | 0.2511 | 0.5429 | 0.5616 |
| Snap only | 0.7141 | 0.2553 | 0.5814 | 0.5543 |

## Subset Analysis (2025 Holdout)

| Subset | N | Platt | Eff only | Inc+Eff |
|--------|---|-------|----------|---------|
| All games | 276 | 0.6313 | 0.7171 | 0.6788 |
| QB changed (home) | 24 | 0.7687 | 0.6290 | 0.8018 |
| QB stable (home) | 252 | 0.6182 | 0.7255 | 0.6671 |

## Recommendation

⚠️ **Incumbent remains research incumbent.**

No efficiency-augmented model beat incumbent on holdout. Closest: Incumbent + Efficiency (val LL=0.6597, hold LL=0.6788) vs incumbent hold LL=0.6313.

> **Note:** The Platt baseline in this experiment (0.6313 holdout) differs from the official incumbent (0.6262, v2.0.0) because this experiment runs its own Elo pipeline with simplified parameters (no season-specific QB-change regression). The 0.6313 is the correct comparison baseline for this experiment's setup. The official 0.6262 is referenced in `reports/benchmarks/nfl_research_incumbent.md`.

### QB-Change Failure Mode

| Model | QB changed (n=24) | QB stable (n=252) |
|-------|--------|--------|
| Platt | 0.7687 | 0.6182 |
| Inc+Eff | 0.8018 | 0.6671 |
