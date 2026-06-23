# EPA Team-Efficiency Features Experiment

*Testing whether pregame rolling EPA features improve on MOV Elo+Platt, with focus on QB-change failure mode.*

## PBP Data Source

| Source | Columns | Coverage |
|--------|---------|----------|
| nflreadpy (nflverse) | epa, success, pass/run splits | 2021–2025 |

## Feature Definitions

All features are pregame-safe (chronological rolling windows, shifted).

| Feature Group | Windows | Description |
|--------------|---------|-------------|
| Offensive EPA/play | 3, 5 | Rolling avg of home/away offense EPA per play |
| Offensive success rate | 3, 5 | Rolling avg of EPA success rate |
| Defensive EPA/play | 3, 5 | Rolling avg of opponent EPA per play |
| Defensive success rate | 3, 5 | Rolling avg of EPA success rate against |
| Passing splits | 3, 5 | Pass-only EPA and success rate |
| Rushing splits | 3, 5 | Rush-only EPA and success rate |
| Net differentials | 3, 5 | Home offense − away defense |
| Missing flags | — | Games available and missingness indicator |

## Leakage Prevention

- Rolling windows are shifted by 1 game (current game excluded)
- Stats reset at season boundaries (no prior-season carryover)
- Week 1 games use 0 (neutral) imputation + missingness flags
- Rolling-origin folds prevent 2025 holdout from influencing model selection

## Incumbent MOV Elo Params

| Parameter | Value |
|-----------|-------|
| K-factor | 36 |
| Home-field advantage | 40 |
| Preseason regression | 0.1 |
| MOV type | capped_linear |
| MOV scale | 0.05 |
| MOV cap | 2.0 |
| Calibration | Platt scaling |

## Data Split

| Split | Seasons | Description |
|-------|---------|-------------|
| Fold 1 | Train: [2021], Val: 2022 | Rolling-origin selection |
| Fold 2 | Train: [2021, 2022], Val: 2023 | Rolling-origin selection |
| Fold 3 | Train: [2021, 2022, 2023], Val: 2024 | Rolling-origin selection |
| Holdout | 2025 | One-shot final evaluation |

## Models Compared

| Model | Description |
|-------|-------------|
| **Platt (incumbent)** | MOV Elo + Platt scaling |
| **EPA only** | Logistic on EPA features alone |
| **MOV Elo + EPA** | Logistic on Elo prob + EPA features |
| **Raw Elo + EPA** | Logistic on raw Elo prob + EPA (diagnostic) |

## Average Validation Metrics Across Folds

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Platt (incumbent) | 0.6368 | 0.6425 | 0.6576 | 0.6103 |
| EPA only | 0.6898 | 0.6853 | 0.7336 | 0.6505 |
| MOV Elo + EPA | 0.6600 | 0.6674 | 0.7110 | 0.6015 |
| Raw Elo + EPA | 0.6600 | 0.6674 | 0.7110 | 0.6015 |

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6285 | 0.2191 | 0.6667 | 0.7024 |
| EPA only | 0.6704 | 0.2390 | 0.5797 | 0.6229 |
| MOV Elo + EPA | 0.6590 | 0.2322 | 0.6449 | 0.6638 |
| Raw Elo + EPA | 0.6590 | 0.2322 | 0.6449 | 0.6638 |

## Subset Analysis (2025 Holdout)

| Subset | N | Platt | EPA only | MOV+EPA |
|--------|---|-------|----------|--------|
| All games | 276 | 0.6285 | 0.6704 | 0.6590 |
| QB changed (home) | 24 | 0.7722 | 0.8509 | 0.9062 |
| QB stable (home) | 252 | 0.6149 | 0.6533 | 0.6354 |
| High confidence (>0.9) | 0 | insufficient | insufficient | insufficient |
| Low confidence (<=0.6) | 170 | 0.6532 | 0.7023 | 0.6802 |

## Recommendation

⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**

No EPA-augmented model beat the incumbent on holdout. Closest: MOV Elo + EPA (val LL=0.6600, hold LL=0.6590) vs incumbent hold LL=0.6285.


### QB-Change Failure Mode Assessment

Platt incumbent on QB-changed games: 0.7722 | QB-stable: 0.6149 | QB-change gap: 0.1574
EPA features did not close the QB-change gap on holdout.

### Next Recommended Experiment

1. If EPA features beat incumbent, test with more expressive models.
2. DVOA/EPA from external sources if nflfastR is insufficient.
3. Consider team-injury feature engineering.
