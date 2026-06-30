# QB-Adjusted Elo Experiment — V0

## Research Question

Can we improve team Elo by adjusting pregame strength based on expected QB availability? QB-specific ratings (computed with Bayesian shrinkage toward replacement level) may capture the magnitude of QB quality differences that the binary `qb_changed` flag misses.

## Models Compared

| Model | Description |
|-------|-------------|
| **A. Incumbent** | Elo + qb_changed + rolling_mov_3 + Platt |
| **B. QB-adjusted (raw)** | QB-adjusted Elo prob, no calibration |
| **C. QB-adjusted + Platt** | QB-adjusted Elo prob + Platt |
| **D. QB-adj + qb_changed + mov3 + Platt** | QB-adj + existing feats + Platt |
| **X. Diagnostic: aggressive QB adj** | 3× scaled QB adjustments + Platt |

## QB Adjustment Formula

```
qb_adj = Elo-point adjustment for each QB start

Per-game impact = actual_win - elo_expected_win_prob

Shrunken impact = (observed_impact * n + PRIOR_IMPACT * PRIOR_STARTS)
                  / (n + PRIOR_STARTS)

qb_adj = 400 * log10((0.5 + shrunken_impact) / (0.5 - shrunken_impact))

team_effective_elo = team_elo + qb_adjustment
adjusted_prob = 1/(1 + 10^(-(h_elo+h_adj - a_elo-a_adj + HFA)/400))
```

Hyperparameters: PRIOR_STARTS=17 (~1 season), PRIOR_IMPACT=-0.03 (replacement ~3% below avg), MAX_ADJUSTMENT=120.0 Elo points.

## Data Used

- 2021–2025 NFL seasons (285 games/season, all non-neutral regular + postseason)
- QB starter IDs from nflreadpy (`home_qb_id`, `away_qb_id` GSIS IDs)
- Team Elo ratings (K=36, HFA=40, reg=0.1, decay=32, MOV capped_linear)
- 108 unique QBs across all seasons

## Data Excluded

- Seasons before 2021
- Neutral-site games
- Ties and games with missing scores
- 2026 games (no scores)
- Post-game stats, final scores, market data

## Leakage Safeguards

1. QB ratings computed chronologically — only prior starts used
2. No post-game data (scores, result) used in rating computation
3. Season-boundary reset for rolling features
4. Holdout (2025) untouched during training/validation
5. Bayesian shrinkage prevents small-sample overfitting
6. QB IDs with missing data assigned 0.0 adjustment (no signal)

## Backtest Setup

- Rolling-origin folds: [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
- Holdout: 2025
- Platt: StandardScaler + LogisticRegression (C=1.0, lbfgs)
- Feature set (incumbent): elo_prob + qb_changed (2) + rolling_mov_3 (2)

## Rolling-Origin Validation Log Loss

| Model | Avg LL | Fold1 | Fold2 | Fold3 |
|-------|--------|-------|-------|-------|
| A. Incumbent (Elo + qb_changed + mov3 + Platt) | 0.6341 | 0.6416 | 0.6577 | 0.6031 |
| B. QB-adjusted Elo (raw) | 0.6349 | 0.6397 | 0.6638 | 0.6012 |
| C. QB-adjusted Elo + Platt | 0.6405 | 0.6500 | 0.6566 | 0.6149 |
| D. QB-adjusted Elo + qb_changed + mov3 + Platt | 0.6338 | 0.6436 | 0.6531 | 0.6046 |
| X. Diagnostic: aggressive QB adj + Platt | 0.6440 | 0.6541 | 0.6545 | 0.6235 |

## 2025 Holdout

| Model | Log Loss | Brier | AUC | Accuracy |
|-------|----------|-------|-----|----------|
| A. Incumbent (Elo + qb_changed + mov3 + Platt) | 0.6259 | 0.2181 | 0.7048 | 0.6594 |
| B. QB-adjusted Elo (raw) | 0.6392 | 0.2240 | 0.6848 | 0.6667 |
| C. QB-adjusted Elo + Platt | 0.6376 | 0.2233 | 0.6848 | 0.6667 |
| D. QB-adjusted Elo + qb_changed + mov3 + Platt | 0.6299 | 0.2201 | 0.6972 | 0.6522 |
| X. Diagnostic: aggressive QB adj + Platt | 0.6473 | 0.2276 | 0.6664 | 0.6486 |

## Slice Performance (2025 Holdout)

| Slice | N | Incumbent LL | Challenger LL | Δ |
|-------|---|-------------|---------------|---|
| All games | 276 | 0.6346 | 0.6376 | +0.0030 |
| QB change (either) | 55 | 0.6579 | 0.6544 | -0.0035 |
| No QB change | 221 | 0.6288 | 0.6334 | +0.0046 |
| Home favorite (>=0.5) | 181 | 0.6259 | 0.6337 | +0.0078 |
| Away underdog (<0.5) | 95 | 0.6512 | 0.6449 | -0.0063 |
| High confidence (>=0.7) | 44 | 0.5152 | 0.5190 | +0.0038 |
| Medium confidence (0.5-0.7) | 137 | 0.6614 | 0.6706 | +0.0092 |

## Biggest QB Adjustments (2025 Holdout Games)

| Game | Team | QB | Adjustment | Starts |
|------|------|----|------------|--------|
| 2025_01_NYG_WAS | WAS | Jayden Daniels | +90.6 | 15 |
| 2025_06_CHI_WAS | WAS | Jayden Daniels | +71.8 | 19 |
| 2025_02_PHI_KC | KC | Patrick Mahomes | +71.6 | 80 |
| 2025_05_DEN_PHI | PHI | Jalen Hurts | +69.3 | 76 |
| 2025_08_WAS_KC | KC | Patrick Mahomes | +69.3 | 86 |
| ... | ... | ... | ... | ... |
| 2025_05_MIA_CAR | CAR | Bryce Young | -98.3 | 28 |
| 2025_03_ATL_CAR | CAR | Bryce Young | -103.4 | 26 |
| 2025_08_TB_NO | NO | Spencer Rattler | -109.4 | 11 |
| 2025_05_NYG_NO | NO | Spencer Rattler | -111.8 | 8 |
| 2025_18_NYJ_BUF | BUF | Mitchell Trubisky | -116.2 | 7 |

## Decision

**No model beats incumbent on both validation and holdout.**

Best validation: D. QB-adjusted Elo + qb_changed + mov3 + Platt (0.6338)
Best holdout: A. Incumbent (Elo + qb_changed + mov3 + Platt) (0.6259)

### Failure Modes

1. **Small-sample QBs**: QBs with <17 starts are strongly shrunk toward replacement. Adjustments for seldom-seen backups are near zero — correct but may miss real signal.
2. **QB adjustment independence**: The adjustment assumes QB impact is additive and independent of the rest of the team.
3. **No position-group interaction**: The adjustment ignores offensive line, skill-position talent, and defensive support.
4. **Oracle QB data**: Uses final actual starter IDs, not pregame-announced. Live-pregame requires `--qb-input CSV`.

### Recommended Next Experiment

1. **Position-group roster strength (V1)**: Extend shrinkage-based rating to OL, skill, and defensive units.
2. **QB-adjustment + market delta**: Test whether QB ratings add signal beyond the market benchmark for QB-change games.
3. **Season-expanded QB ratings**: Confirm whether 2021–2025 has enough QB-start data for stable ratings.

---
*Report generated by `sportslab qb-adjusted-elo`. PRIOR_STARTS=17, PRIOR_IMPACT=-0.03, MAX_ADJUSTMENT=120.0.*
