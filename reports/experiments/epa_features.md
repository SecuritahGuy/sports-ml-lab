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
| Preseason regression | 0.2 |
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
| Platt (incumbent) | 0.6363 | 0.6438 | 0.6564 | 0.6088 |
| EPA only | 0.6898 | 0.6853 | 0.7336 | 0.6505 |
| MOV Elo + EPA | 0.6593 | 0.6683 | 0.7085 | 0.6013 |
| Raw Elo + EPA | 0.6593 | 0.6683 | 0.7085 | 0.6013 |

## Full Comparison (2025 Holdout)

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6373 | 0.2230 | 0.6522 | 0.6907 |
| EPA only | 0.6704 | 0.2390 | 0.5797 | 0.6229 |
| MOV Elo + EPA | 0.6662 | 0.2353 | 0.6522 | 0.6555 |
| Raw Elo + EPA | 0.6662 | 0.2353 | 0.6522 | 0.6555 |

## Subset Analysis (2025 Holdout)

| Subset | N | Platt | EPA only | MOV+EPA |
|--------|---|-------|----------|--------|
| All games | 276 | 0.6373 | 0.6704 | 0.6662 |
| QB changed (home) | 24 | 0.7836 | 0.8509 | 0.9093 |
| QB stable (home) | 252 | 0.6234 | 0.6533 | 0.6431 |
| High confidence (>0.9) | 0 | insufficient | insufficient | insufficient |
| Low confidence (<=0.6) | 165 | 0.6602 | 0.6993 | 0.6899 |

## Recommendation

⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**

No EPA-augmented model beat the incumbent on holdout. Closest: MOV Elo + EPA (val LL=0.6593, hold LL=0.6662) vs incumbent hold LL=0.6373.


### QB-Change Failure Mode Assessment

Platt incumbent on QB-changed games: 0.7836 | QB-stable: 0.6234 | QB-change gap: 0.1601
EPA features did not close the QB-change gap on holdout.

### Next Recommended Experiment

1. If EPA features beat incumbent, test with more expressive models.
2. DVOA/EPA from external sources if nflfastR is insufficient.
3. Consider team-injury feature engineering.
