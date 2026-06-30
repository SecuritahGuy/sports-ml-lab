# Gated QB-Adjusted Elo Experiment — V1

## Research Question

Can QB-adjusted Elo improve games with meaningful QB uncertainty or QB changes without degrading games where the starting QB situation is stable?

## Why V0 Was Rejected

The first QB-adjusted Elo experiment (V0) showed that QB adjustments helped QB-change games (holdout Δ = −0.0035) but hurt non-QB-change games (Δ = +0.0046). The overall holdout was worse (0.6376 vs incumbent 0.6259). The adjustment introduces noise for stable-QB situations where the Elo rating already captures team quality.

Key insight: the adjustment is useful specifically when the QB situation is uncertain (change, low continuity, rookie/backup), not for established starters who have a long track record.

## Gating Strategy

Instead of applying the QB adjustment uniformly to all games, a pregame-safe **gate** determines the adjustment multiplier for each side of each game:

```
gate_multiplier = f(qb_changed, qb_team_starts, games_since_change)
gated_adj = full_qb_adj * gate_multiplier
```

All gating features (`qb_changed`, `qb_team_starts_pre`, `games_since_qb_change`) are computed chronologically from games before the current game — no leakage.

### Gating Variants Tested

| Letter | Gate Mode | Description |
|--------|-----------|-------------|
| B | full | V0: no gating, full adjustment |
| C | qb_changed_only | Adjust only when QB changed (gate=0 otherwise) |
| D | low_continuity | Adjust when QB changed OR <N starts with team |
| E | shrunk_stable | Full adjust for changed, scaled (0.1-0.5x) for stable |
| F | capped_only | Same as V0 but lower max cap (40-80) |
| G | aggressive_diagnostic | DIAGNOSTIC: 2x changed, 0x stable |
| H | recency_weighted | Decayed older games (HL=32), standard shrinkage |
| I | combined | Low continuity (starts<8) + cap=60 |

## Hyperparameters Tested

| Parameter | Values Tested |
|-----------|---------------|
| gate_mode | full, qb_changed_only, low_continuity, shrunk_stable, capped_only |
| stable_shrink | 0.1, 0.3, 0.5 |
| min_starts_for_stable | 4, 8, 17 |
| max_adj_cap | 40, 60, 80 |
| decay_half_life | 32 (recency only) |

## Data Used

- 2021–2025 NFL seasons (non-neutral regular + postseason)
- QB starter IDs from nflreadpy (`home_qb_id`, `away_qb_id` GSIS IDs)
- Team Elo ratings (K=36, HFA=40, reg=0.1, decay=32, MOV capped_linear)
- Incumbent feature set: elo_prob + qb_changed (2) + rolling_mov_3 (2) + Platt
- Gated variants: same Elo spine, QB adj prob + Platt

## Data Excluded

- Seasons before 2021
- Neutral-site games
- Ties and games with missing scores
- 2026 games (no scores)
- Post-game stats, final scores, market data

## Leakage Safeguards

1. QB adjustments computed chronologically — only prior starts used
2. Gate conditions (qb_changed, starts, continuity) are pregame only
3. No post-game data used in any rating or gate computation
4. Season-boundary reset for rolling features
5. Holdout (2025) untouched during training/validation
6. Bayesian shrinkage prevents small-sample overfitting
7. Missing QB IDs assigned 0.0 adjustment (no signal)

## Backtest Setup

- Rolling-origin folds: [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
- Holdout: 2025
- Platt: StandardScaler + LogisticRegression (C=1.0, lbfgs)
- Incumbent feature set: elo_prob + qb_changed (2) + rolling_mov_3 (2)
- Gated variants: qb-adjusted prob + Platt (no extra features)

## Rolling-Origin Validation Log Loss

| Model | Avg LL | Fold1 | Fold2 | Fold3 |
|-------|--------|-------|-------|-------|
| A. Incumbent baseline | 0.6341 | 0.6416 | 0.6577 | 0.6031 |
| H. Diagnostic: aggressive gated | 0.6359 | 0.6408 | 0.6603 | 0.6067 |
| C. Gated QB (qb_changed_only) | 0.6367 | 0.6425 | 0.6603 | 0.6074 |
| E. Gated QB (shrunk_stable (shrink=0.1)) | 0.6368 | 0.6433 | 0.6595 | 0.6076 |
| E. Gated QB (shrunk_stable (shrink=0.3)) | 0.6372 | 0.6448 | 0.6582 | 0.6087 |
| E. Gated QB (shrunk_stable (shrink=0.5)) | 0.6380 | 0.6464 | 0.6574 | 0.6102 |
| J. Combined (low cont+cap=60) | 0.6386 | 0.6434 | 0.6596 | 0.6127 |
| D. Gated QB (low_continuity (starts<4)) | 0.6389 | 0.6420 | 0.6635 | 0.6113 |
| D. Gated QB (low_continuity (starts<8)) | 0.6390 | 0.6434 | 0.6604 | 0.6131 |
| F. Gated QB (capped_only (cap=40)) | 0.6395 | 0.6483 | 0.6567 | 0.6134 |
| F. Gated QB (capped_only (cap=60)) | 0.6397 | 0.6499 | 0.6561 | 0.6131 |
| F. Gated QB (capped_only (cap=80)) | 0.6400 | 0.6497 | 0.6565 | 0.6137 |
| B. Full QB adj (V0) | 0.6405 | 0.6500 | 0.6566 | 0.6149 |
| I. Recency-weighted QB adj | 0.6405 | 0.6484 | 0.6583 | 0.6149 |
| D. Gated QB (low_continuity (starts<17)) | 0.6414 | 0.6518 | 0.6586 | 0.6138 |

## 2025 Holdout

| Model | Log Loss | Brier | AUC | Accuracy |
|-------|----------|-------|-----|----------|
| D. Gated QB (low_continuity (starts<17)) | 0.6255 | 0.2179 | 0.7020 | 0.6486 |
| A. Incumbent baseline | 0.6259 | 0.2181 | 0.7048 | 0.6594 |
| D. Gated QB (low_continuity (starts<8)) | 0.6280 | 0.2189 | 0.7002 | 0.6449 |
| J. Combined (low cont+cap=60) | 0.6281 | 0.2190 | 0.7010 | 0.6449 |
| D. Gated QB (low_continuity (starts<4)) | 0.6308 | 0.2205 | 0.6950 | 0.6377 |
| F. Gated QB (capped_only (cap=40)) | 0.6326 | 0.2210 | 0.6937 | 0.6667 |
| E. Gated QB (shrunk_stable (shrink=0.1)) | 0.6339 | 0.2218 | 0.6931 | 0.6377 |
| C. Gated QB (qb_changed_only) | 0.6339 | 0.2218 | 0.6940 | 0.6522 |
| E. Gated QB (shrunk_stable (shrink=0.3)) | 0.6343 | 0.2219 | 0.6919 | 0.6449 |
| E. Gated QB (shrunk_stable (shrink=0.5)) | 0.6350 | 0.2222 | 0.6900 | 0.6486 |
| F. Gated QB (capped_only (cap=60)) | 0.6361 | 0.2226 | 0.6883 | 0.6667 |
| I. Recency-weighted QB adj | 0.6366 | 0.2229 | 0.6863 | 0.6522 |
| F. Gated QB (capped_only (cap=80)) | 0.6372 | 0.2231 | 0.6858 | 0.6594 |
| B. Full QB adj (V0) | 0.6376 | 0.2233 | 0.6848 | 0.6667 |
| H. Diagnostic: aggressive gated | 0.6395 | 0.2243 | 0.6869 | 0.6558 |

## Slice Performance by Variant (2025 Holdout)

| Model | QB-change LL | No-QB-change LL | Δ vs inc (QC) | Δ vs inc (noQC) |
|-------|-------------|-----------------|---------------|-----------------|
| F. Gated QB (capped_only (cap=40)) | 0.6480 | 0.6288 | -0.0216 | +0.0137 |
| F. Gated QB (capped_only (cap=60)) | 0.6531 | 0.6319 | -0.0165 | +0.0168 |
| B. Full QB adj (V0) | 0.6544 | 0.6334 | -0.0152 | +0.0183 |
| F. Gated QB (capped_only (cap=80)) | 0.6544 | 0.6329 | -0.0152 | +0.0178 |
| I. Recency-weighted QB adj | 0.6557 | 0.6319 | -0.0139 | +0.0168 |
| D. Gated QB (low_continuity (starts<17)) | 0.6597 | 0.6170 | -0.0099 | +0.0019 |
| E. Gated QB (shrunk_stable (shrink=0.5)) | 0.6635 | 0.6280 | -0.0061 | +0.0129 |
| J. Combined (low cont+cap=60) | 0.6660 | 0.6186 | -0.0036 | +0.0035 |
| D. Gated QB (low_continuity (starts<8)) | 0.6675 | 0.6181 | -0.0021 | +0.0030 |
| E. Gated QB (shrunk_stable (shrink=0.3)) | 0.6689 | 0.6256 | -0.0007 | +0.0105 |
| A. Incumbent baseline | 0.6696 | 0.6151 | +0.0000 | +0.0000 |
| E. Gated QB (shrunk_stable (shrink=0.1)) | 0.6756 | 0.6235 | +0.0060 | +0.0084 |
| D. Gated QB (low_continuity (starts<4)) | 0.6762 | 0.6194 | +0.0066 | +0.0043 |
| C. Gated QB (qb_changed_only) | 0.6794 | 0.6226 | +0.0098 | +0.0075 |
| H. Diagnostic: aggressive gated | 0.7047 | 0.6233 | +0.0351 | +0.0082 |

## Best Challenger Slice Performance

Best gated challenger: **D. Gated QB (low_continuity (starts<17))**

| Slice | N | Incumbent LL | Challenger LL | Δ |
|-------|---|-------------|---------------|---|
| All games | 276 | 0.6346 | 0.6255 | -0.0091 |
| QB change (either) | 55 | 0.6579 | 0.6597 | +0.0018 |
| No QB change | 221 | 0.6288 | 0.6170 | -0.0118 |
| Stable QB (≥4 starts, no change) | 168 | 0.6339 | 0.6217 | -0.0122 |
| Low-sample QB (<4 starts) | 86 | 0.6299 | 0.6215 | -0.0084 |
| High confidence (>=0.7) | 44 | 0.5152 | 0.5197 | +0.0045 |
| Medium confidence (0.5-0.7) | 137 | 0.6614 | 0.6694 | +0.0080 |
| Home favorite (>=0.5) | 181 | 0.6259 | 0.6330 | +0.0071 |
| Away underdog (<0.5) | 95 | 0.6512 | 0.6111 | -0.0401 |

## Full Results Summary

| Model | Val LL | Hold LL | Δ vs inc (val) | Δ vs inc (hold) | QC Δ | NoQC Δ |
|-------|--------|---------|----------------|-----------------|------|--------|
| A. Incumbent baseline | 0.6341 | 0.6259 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| B. Full QB adj (V0) | 0.6405 | 0.6376 | +0.0064 | +0.0117 | -0.0152 | +0.0183 |
| C. Gated QB (qb_changed_only) | 0.6367 | 0.6339 | +0.0026 | +0.0080 | +0.0098 | +0.0075 |
| D. Gated QB (low_continuity (starts<17)) | 0.6414 | 0.6255 | +0.0073 | -0.0005 | -0.0099 | +0.0019 |
| D. Gated QB (low_continuity (starts<4)) | 0.6389 | 0.6308 | +0.0048 | +0.0048 | +0.0066 | +0.0043 |
| D. Gated QB (low_continuity (starts<8)) | 0.6390 | 0.6280 | +0.0048 | +0.0020 | -0.0021 | +0.0030 |
| E. Gated QB (shrunk_stable (shrink=0.1)) | 0.6368 | 0.6339 | +0.0027 | +0.0079 | +0.0060 | +0.0084 |
| E. Gated QB (shrunk_stable (shrink=0.3)) | 0.6372 | 0.6343 | +0.0031 | +0.0083 | -0.0007 | +0.0105 |
| E. Gated QB (shrunk_stable (shrink=0.5)) | 0.6380 | 0.6350 | +0.0039 | +0.0091 | -0.0061 | +0.0129 |
| F. Gated QB (capped_only (cap=40)) | 0.6395 | 0.6326 | +0.0053 | +0.0067 | -0.0216 | +0.0137 |
| F. Gated QB (capped_only (cap=60)) | 0.6397 | 0.6361 | +0.0056 | +0.0102 | -0.0165 | +0.0168 |
| F. Gated QB (capped_only (cap=80)) | 0.6400 | 0.6372 | +0.0058 | +0.0112 | -0.0152 | +0.0178 |
| H. Diagnostic: aggressive gated | 0.6359 | 0.6395 | +0.0018 | +0.0136 | +0.0351 | +0.0082 |
| I. Recency-weighted QB adj | 0.6405 | 0.6366 | +0.0064 | +0.0107 | -0.0139 | +0.0168 |
| J. Combined (low cont+cap=60) | 0.6386 | 0.6281 | +0.0044 | +0.0021 | -0.0036 | +0.0035 |

## Decision

**No gated variant beats incumbent on both validation and holdout.**

All gated variants are either rejected or marked diagnostic only.

Best validation: A. Incumbent baseline (0.6341)
Best holdout: D. Gated QB (low_continuity (starts<17)) (0.6255)

### QB-Change Game Impact

Best QB-change improvement: F. Gated QB (capped_only (cap=40)) (-0.0216)
Worst QB-change degradation: C. Gated QB (qb_changed_only) (+0.0098)

### Non-QB-Change Game Impact

Best non-QB-change improvement: D. Gated QB (low_continuity (starts<17)) (+0.0019)
Worst non-QB-change degradation: B. Full QB adj (V0) (+0.0183)

### Failure Modes

1. **Gating sharpness**: The qb_changed flag is coarse — a QB change from one elite starter to another elite starter triggers the same gate as a change to a backup.
2. **Small-sample QBs**: QBs with <17 starts are strongly shrunk toward replacement. Gating can't fix the fact that tiny-sample adjustments are inherently noisy.
3. **Oracle QB data**: Uses final actual starter IDs, not pregame-announced. Live-pregame requires `--qb-input CSV`.
4. **No position-group interaction**: The adjustment ignores offensive line, skill-position talent, and defensive support.

### Recommended Next Experiment

1. **QB-specific position-group ratings**: Instead of a global QB rating, separate QB rating into passing/timing/decision-making components
2. **Coach-QB interaction features**: The combination of a new QB and new coordinator may be more informative than QB change alone
3. **Expanded Elo K search**: Test K > 48 with the season regression spine
4. **Any model must beat Standard Elo + qb_changed + rolling_mov_3 + Platt (holdout LL 0.6262)** to become the new clean football-only incumbent

---
*Report generated by `sportslab gated-qb-elo`. GATE_MODES tested: ['full', 'qb_changed_only', 'low_continuity', 'shrunk_stable', 'capped_only', 'aggressive_diagnostic', 'recency_decay'].*
