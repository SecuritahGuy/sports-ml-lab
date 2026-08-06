# Experiment Ledger

*Machine-readable: `experiment_ledger.csv`*

*Generated: 2026-07-28*

---

## Summary

| Count | Value |
|-------|-------|
| Total experiments | 48 |
| Promoted (current champion) | 2 |
| Superseded (former champions) | 6 |
| Rejected | 31 |
| Diagnostic | 10 |
| Current overall incumbent | Pi-Ratings + FDR + DOBA + Chaos |
| Current football-only incumbent | Pi-Ratings v4.0.0 |

---

## Timeline (Chronological)

| # | Experiment | Date | Decision | Val LL | Holdout LL |
|---|-----------|------|----------|--------|-----------|
| 1 | Identity Logistic | 2026-06-23 | rejected | 0.68 | 0.69 |
| 2 | Team-Strength Logistic | 2026-06-23 | rejected | 0.67 | 0.67 |
| 3 | Tuned Elo (K=32, HFA=25) | 2026-06-23 | superseded | 0.65 | 0.6616 |
| 4 | Rolling-Origin Elo + Platt | 2026-06-23 | superseded | 0.6363 | 0.6395 |
| 5 | Scheduling/Rest Features | 2026-06-23 | rejected | 0.6599 | 0.6401 |
| 6 | Margin-Aware MOV Elo | 2026-06-23 | superseded | 0.6363 | 0.6373 |
| 7 | QB Starter/Change Features | 2026-06-23 | rejected | 0.6436 | 0.6459 |
| 8 | Weather Features | 2026-06-23 | rejected | 0.6445 | 0.6439 |
| 9 | EPA Team-Efficiency | 2026-06-23 | rejected | 0.6654 | 0.6495 |
| 10 | Confidence Calibration | 2026-06-23 | rejected | 0.6374 | 0.6373 |
| 11 | Constrained Expressive Models | 2026-06-23 | rejected | 0.6361 | 0.6638 |
| 12 | Market Baseline | 2026-06-23 | diagnostic | 0.6052 | 0.6090 |
| 13 | Residual Diagnostics | 2026-06-23 | diagnostic | — | 0.6373 |
| 14 | Decayed Elo | 2026-06-23 | superseded | 0.6321 | 0.6298 |
| 15 | Team-Specific HFA | 2026-06-23 | rejected | 0.6355 | 0.6263 |
| 16 | Season-Specific QB Regression | 2026-06-23 | superseded | 0.6315 | 0.6285 |
| 17 | Residual Blending | 2026-06-23 | rejected | 0.6368 | 0.6303 |
| 18 | Coach+QB Season Regression | 2026-06-23 | rejected | 0.6309 | 0.6286 |
| 19 | Separate O/D Elo | 2026-06-23 | diagnostic | 0.6376 | 0.6258 |
| 20 | AutoGluon AutoML | 2026-06-23 | rejected | 0.6956 | 0.6404 |
| 21 | Injury Features | 2026-06-23 | rejected | 0.6406 | 0.6514 |
| 22 | Optuna Joint Elo Search | 2026-06-23 | rejected | 0.6342 | 0.6318 |
| 23 | QB Injury Flag | 2026-06-23 | rejected | 0.6464 | 0.6255 |
| 24 | Glicko Rating System | 2026-06-23 | rejected | 0.6513 | 0.7013 | *(bug found — formula fixed 2026-07-21)* |
| 46 | Glicko Rating System (retest) | 2026-07-21 | rejected | 0.6415 | 0.6338 | *(fix: 0.7013→0.6338, still can't beat Elo)* |
| 25 | QB Market Delta | 2026-06-23 | diagnostic | 0.6052 | 0.6090 |
| 26 | Forward Feature Selection | 2026-06-23 | diagnostic | 0.6334 | 0.6314 |
| 27 | Combined Features (qb_changed + mov_3) | 2026-06-23 | superseded | 0.6334 | 0.6262 |
| 28 | Home/Away Separate Elo | 2026-06-23 | rejected | 0.6622 | 0.6634 |
| 29 | Team Stats Features | 2026-06-24 | rejected | 0.6541 | 0.6415 |
| 30 | Optuna Feature Selection | 2026-06-24 | diagnostic | 0.6334 | 0.6347 |
| 31 | QB Market Delta V2 | 2026-06-24 | diagnostic | 0.6050 | 0.6083 |
| 32 | Comprehensive Efficiency | 2026-06-24 | rejected | 0.6368 | 0.6313 |
| 33 | QB-Adjusted Elo V0 | 2026-06-29 | rejected | 0.6338 | 0.6299 |
| 34 | Gated QB-Adjusted Elo V1 | 2026-06-29 | rejected | 0.6341 | 0.6255 |
| 35 | Frozen QB Overlay V2 (flawed) | 2026-06-29 | diagnostic | — | 0.6200 |
| 36 | Frozen QB Overlay V3 (fold-safe) | 2026-06-29 | **promoted** | **0.6305** | **0.6200** |
| 37 | Roster Overlay (Position Groups) | 2026-06-29 | rejected | 0.6341 | 0.6255 |
| 38 | QB × Roster Interaction | 2026-06-30 | rejected | 0.6305 | 0.6195 |
| 39 | Expanded Elo Spine (840 combos) | 2026-06-30 | rejected | 0.6299 | 0.6302 |
| 40 | RALPH 6: prior_win_pct | 2026-07-06 | rejected | 0.6316 | 0.6183 |
| 41 | RALPH 6: weather_missing | 2026-07-06 | rejected | 0.6481 | 0.6198 |
| 42 | RALPH 6: roof_enc | 2026-07-06 | rejected | 0.6315 | 0.6202 |
| 43 | RALPH 6: games_since_change | 2026-07-06 | rejected | 0.6321 | 0.6202 |
| 44 | RALPH 6: isotonic | 2026-07-06 | rejected | 0.6312 | 0.6283 |
| 45 | RALPH Loop 8: Preseason Elo Prior | 2026-07-06 | rejected | 0.6342 | 0.6282 |

---

## By Decision

### Promoted (Current Champion)
| # | Experiment | Holdout LL | Report |
|---|-----------|------------|--------|
| 51 | Pi-Ratings v4 | **0.5918** | pi_ratings_champion_comparison.md |

### Superseded (Former Champions)
| # | Experiment | Holdout LL at Promotion | Report |
|---|-----------|------------------------|--------|
| 36 | Frozen QB Overlay V3 | 0.6200 | frozen_qb_overlay_foldsafe.md |
| 27 | Combined Features (qb_changed + mov_3) | 0.6262 | combined_features.md |
| 16 | Season Regression | 0.6285 | season_regression.md |
| 14 | Decayed Elo | 0.6298 | decayed_elo.md |
| 6 | Margin-Aware MOV Elo | 0.6373 | margin_aware_elo.md |
| 4 | Rolling-Origin Elo + Platt | 0.6395 | rolling_origin_elo_validation.md |

### Rejected (30)
All rejected experiments documented with rejection reason in `experiment_ledger.csv`.

### Diagnostic (10)
| # | Experiment | Type | Report |
|---|-----------|------|--------|
| 12 | Market Baseline | Market benchmark | market_benchmark.md |
| 13 | Residual Diagnostics | Error analysis | residual_diagnostics.md |
| 19 | Separate O/D Elo | Holdout-informed | od_elo.md |
| 25 | QB Market Delta | Market-aware | qb_market_delta.md |
| 26 | Forward Feature Selection | Feature audit | feature_selection.md |
| 30 | Optuna Feature Selection | Feature audit | optuna_feature_selection.md |
| 31 | QB Market Delta V2 | Market-aware | qb_market_delta.md |
| 35 | Frozen QB Overlay V2 | Flawed validation | frozen_qb_overlay.md |

---

## Rejection Pattern Analysis

The 28 rejected experiments fall into these categories:

| Rejection Pattern | Count | Examples |
|------------------|-------|----------|
| Both val and holdout worse | 13 | Scheduling, weather, EPA, Glicko *(retested 2026-07-21, still worse)*, AutoGluon, injury features, team stats, expressive models, home/away Elo, comprehensive efficiency, QB features, residual blending, preseason Elo Prior |
| Val worse, holdout better (val rejects) | 5 | Team HFA, prior_win_pct, games_since_change, roof_enc, weather_missing |
| Val better/neutral, holdout worse (overfit) | 5 | Isotonic, Optuna search, QB-adjusted Elo V0, gated QB-adjusted V1, expanded Elo spine |
| No improvement over incumbent | 2 | Confidence calibration (tied), coach-season regression (val 0.0006 better, holdout 0.0001 worse) |
| Below baseline | 2 | Identity logistic, team-strength logistic (both below home prior) |

**Key insight**: 13/30 (43%) of rejected experiments degraded BOTH val and holdout. Only 2/30 were close enough to warrant a second look (coach-season reg, confidence calibration). The incumbent is at a Pareto frontier — changing it almost always makes both metrics worse.

### Distinct Signal Categories with Evidence
- **Strongly rejected** (≥0.005 worse on both): AutoGluon, scheduling, weather, EPA, team stats, comprehensive efficiency, home/away Elo, injury features
- **Strongly rejected on holdout, close on val**: Glicko *(retested 2026-07-21: val Δ=+0.0039, hold Δ=+0.0080)*
- **Close but rejected** (<0.005 on at least one axis): coach-season reg, confidence calibration, isotonic, roof_enc, expanded Elo spine, O/D Elo (val)
- **Mixed signal** (improves one axis but degrades the other): QB-adjusted Elo V0/V1, gated QB-adjusted, prior_win_pct, games_since_change, team HFA

---

## Current Frontier

The incumbent (holdout 0.5532) is at a Pareto optimum for the current sample size, feature set, and modeling approach using Pi-Ratings + StatSpace PBP composites. All rejected experiments demonstrate that adding complexity without new high-signal data sources degrades performance on at least one axis.

The best reference points:
- **Overall champion**: **0.5532** (Pi-Ratings + FDR + DOBA + Chaos)
- **Previous overall champion**: 0.5548 (Standard Elo + FDR + DOBA + Chaos)
- **Football-only champion**: 0.5918 (Pi-Ratings v4.0.0)
- **Market ceiling**: 0.6090 (no-vig closing moneyline)
- **Holdout-informed ceiling**: 0.6258 (O/D Elo, selected using holdout)
