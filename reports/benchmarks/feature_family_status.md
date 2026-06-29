# Feature Family Status Report

*Generated: 2026-06-29 by `sportslab` closure pass*

---

## Summary

This document is the **master inventory** of every feature family tested against the research incumbent. It serves as the governance record: **do not retest any family in the Rejected or Watchlist categories without new data or a documented failure mode.**

### Research Incumbent

```
Model:        Elo + qb_changed + rolling_mov_3 + Platt
Val LL:       0.6334
Holdout LL:   0.6262
Holdout Brier: 0.2180
Holdout AUC:   0.7050
Version:      v2.0.0
```

### Live-Safe Fallback (No Oracle QB)

```
Model:        Elo + rolling_mov_3 + Platt
Val LL:       0.6363
Holdout LL:   0.6298
```

---

## Status Legend

| Status | Meaning |
|--------|---------|
| **PROMOTED** | Active incumbent feature. |
| **REJECTED** | Tested across ≥1 experiments. Consistently worse on both val and holdout. Do not retest without new data or a documented failure mode. |
| **WATCHLIST** | Mixed results (won on one axis but lost the other). A small improvement on holdout but worse on val. Revisit only if new data accumulates. |
| **DIAGNOSTIC** | Not a model input. Used for error analysis, calibration, or benchmarking only. |
| **NOT TESTED** | Requires new data source or pregame-safe pipeline not yet built. |

---

## Feature Families

### PROMOTED

| Family | Features Tested | Best Val LL | Best Hold LL | Reports |
|--------|----------------|-------------|-------------|---------|
| **Elo rating** | K, HFA, reg, decay, MOV type | 0.6363 | 0.6395 | rolling_origin_elo_validation, margin_aware_elo, season_regression |
| **QB change flag** | `home_qb_changed`, `away_qb_changed` | 0.6334 | 0.6262 | feature_selection, combined_features |
| **Rolling MOV 3-game** | `home_rolling_mov_3`, `away_rolling_mov_3` | 0.6334 | 0.6262 | rolling_mov_sensitivity, combined_features |
| **Platt calibration** | Logistic regression on Elo prob + features | 0.6334 | 0.6262 | downstream of all experiments |

### REJECTED

| Family | Features Tested | Worst Δ Val | Worst Δ Hold | Reports |
|--------|----------------|-------------|-------------|---------|
| **Scheduling / Rest** | short_week, off_bye, thursday, monday, intl, consec_road, rest_diff, div_rest | +0.0236 | +0.0006 | schedule_rest_features |
| **Weather** | temp, wind, precip, cold/windy/bad-weather flags, dome neutralize | +0.0082 | +0.0066 | weather_features, feature_selection |
| **QB identity (OHE)** | 93-class one-hot of starter name | +0.4295 | +13.88 | qb_starter_change_features |
| **QB rookie/backup** | is_rookie_qb, is_backup_qb | +0.0073 | +0.0086 | qb_starter_change_features |
| **QB change prior** | qb_change_prior_game | — | — | qb_starter_change_features |
| **QB injury flags** | QB OUT, position-group injury counts (RB/WR/TE/OL/DL/LB/DB), net diffs | +0.0080 | +0.0199 | injury_features |
| **QB depth** | qb_rust_games, qb_first_season_start, career_starts, career_win_pct | +0.0324 | +0.0130 | qb_depth |
| **QB continuity** | qb_experience_global, gated_binary, gated_team_specific, gated_simple_diff | +0.0095 | +0.0060 | qb_continuity, qb_gated_experience |
| **QB magnitude** | rolling_epa, change_magnitude (abs+signed), epa_diff | +0.0051 | +0.0012 | qb_magnitude |
| **Coach features** | coach_tenure, career_wins, career_games, career_win_pct | +0.0065 | +0.0456 | combined_features, feature_selection, coach_qb_tenure |
| **Coach+QB season regression** | coach_bonus on top of qb_bonus | +0.0006 | +0.0001 | coach_season_regression |
| **First-year coach** | home_first_year_coach, away_first_year_coach, coach_change_diff | +0.0015 | +0.0033 | situational_micro |
| **Surface mismatch** | away_surface_mismatch, away_grass_to_turf, away_turf_to_grass | +0.0024 | +0.0025 | situational_micro |
| **Divisional game** | div_game, div×qb_changed interactions | +0.0016 | +0.0053 | situational_micro |
| **Team-specific HFA** | per-team home field advantage from historical data | +0.0034 | −0.0035 | team_hfa |
| **Home/away separate Elo** | independent home/away ratings per team | +0.0212 | +0.0158 | home_away_elo |
| **Turnovers** | to_net_3, to_net_5 | +0.0039 | +0.0025 | turnover_features |
| **Team EPA** | pass_epa, rush_epa, rec_epa, total_epa rolling 3/5 | +0.0521 | +0.0386 | epa_features, comprehensive_efficiency |
| **PFR advanced stats** | pressure_rate, bad_throw_rate, YAC/rush, broken_tackles, def passer_rating | +0.0679 | +0.0633 | comprehensive_efficiency |
| **Snap counts** | ol_snap_pct, top_rb_snap_pct | +0.0551 | +0.0481 | comprehensive_efficiency |
| **Win streak / momentum** | home_win_streak, away_win_streak (signed: pos=win, neg=loss) | +0.0007 | — | feature_selection, optuna_feature_selection |
| **Season regression (standalone)** | qb_change_bonus without Elo tuning | — | — | season_regression |
| **Glicko rating** | Full Glicko replacement for Elo | — | — | glicko_rating |
| **AutoGluon AutoML** | 47 features, all sklearn backends | +0.0580 | +0.0142 | autogluon |
| **Constrained tree models** | HGB, GB, RF on 27 curated features | +0.0381 | +0.0718 | expressive_models |
| **Decayed Elo** | exponential momentum on Elo | — | — | decayed_elo |
| **Residual blending** | Elo prob + week/rest/early-season features | — | +0.0045 | residual_blending |
| **Confidence calibration** | era-split Platt, high-confidence shrinkage | — | — | calibration_improvements |
| **Adaptive K** | Elo K varies by week number | — | — | adaptive_k |

### WATCHLIST

| Family | Why | Condition to Revisit |
|--------|-----|---------------------|
| **Turnover diff (to_net_3)** | −0.0032 holdout improvement but +0.0036 val loss. Small signal in isolated window. | If dataset grows by 2+ seasons OR pregame turnover data source found. |
| **Team-specific HFA** | −0.0035 holdout improvement but +0.0034 val loss. Not stable across folds. | If more stadium/ref/altitude data available. |
| **QB market delta** | Full market information ratio confirmed. Diagnostic only — not a model input. | Only as market benchmark diagnostic. |
| **Rolling MOV 1-game** | Best val (−0.0001) but lost holdout. Noisier than MOV 3. | Revisit if more data stabilizes short windows. |

### DIAGNOSTIC ONLY

| Feature | Used For | Source |
|---------|----------|--------|
| Market moneyline | Benchmark comparison, residual correlation | market_baseline, market_benchmark |
| Market spread | Benchmark comparison, spread→prob conversion | market_benchmark |
| Referee | Sparsity audit (21 unique, 70 games/ref) | situational_micro |
| QB career starts/win% | Noise-level diagnostic | qb_depth |

### NOT TESTED

| Feature | Reason |
|---------|--------|
| Opening lines (market) | Not in nflreadpy schedule data. Would require external data source. |
| Injury report timing | Pregame weekly injury report (not just game-day status). Would require nflreadpy `load_injuries()`. |
| Coach market delta | No data source. |
| Preseason power ratings | No data source. |
| DVOA / Defense-adjusted stats | No data source. |

---

## Do Not Retest Rule

The following families are **permanently closed** for new experiments without a documented trigger:

1. **New data accumulates** — at least 2 additional seasons (260+ games) past the current 2021-2025 span.
2. **New pregame-safe data source** is added to the repository (not just discovered to exist).
3. **Live prediction logs** reveal a repeatable failure mode not already captured by residual diagnostics.
4. **Market benchmark** comparison is requested as a diagnostic (not as a model input).

A closed family may be reopened only with an explicit research note explaining why prior negative results no longer apply.

---

## One-Paragraph Rationale per Rejected Family

| Family | Why It Failed |
|--------|-------------|
| **Scheduling / Rest** | Too few short-week (10%) and international games (2%). `rest_diff` is nearly zero-centered and noisy. |
| **Weather** | Only ~15% of games have meaningful cold/wind/precip. Dome neutralization removes signal. Weather-only model is barely above random. |
| **QB identity OHE** | 93 classes for 1388 rows → severe overfit. Holdout log loss of 14.51 (worse than random 0.69). |
| **QB injury flags** | Injury reports are noisy. Even "QB OUT" hurts performance. Subset analysis shows Elo performs *better* on QB-out games (Elo undershoots → pleasant surprise). |
| **QB depth** | Career starts and win% from nflreadpy `load_players()` have near-zero variance for established QBs. Rust/first-start flags add noise. |
| **Coach features** | Coach quality already baked into Elo ratings. Coach features are highly correlated with team identity. |
| **Team EPA** | Rolling team EPA is a noisy per-game aggregate. Elo already captures game outcomes from point differential. |
| **PFR advanced stats** | High-dimensional player-week data is too sparse to aggregate to team-game level without leakage. |
| **Win streak** | Elo already captures recent form (winning teams → higher Elo). `win_streak` adds multicollinearity without independent signal. |
| **Tree models** | At 1000 training rows, tree ensembles overfit the curated features. Elo+Platt is the optimal complexity for this dataset size. |

---

## Report Index

| Report | Family | Status |
|--------|--------|--------|
| rolling_origin_elo_validation.md | Elo tuning (K, HFA, reg) | PROMOTED (subcomponent) |
| margin_aware_elo.md | MOV type + scale + cap | PROMOTED (subcomponent) |
| season_regression.md | QB-change preseason regression | PROMOTED (subcomponent) |
| combined_features.md | qb_changed + rolling_mov_3 | PROMOTED (final incumbent) |
| feature_selection.md | 10 situational + 4 QB feature groups | REJECTED (all except qb_changed) |
| schedule_rest_features.md | Scheduling flags, rest diff | REJECTED |
| weather_features.md | Temp, wind, precip | REJECTED |
| qb_starter_change_features.md | QB OHE, rookie, backup flags | REJECTED |
| injury_features.md | 20 injury report columns | REJECTED |
| qb_depth.md | Career starts, win%, rust | REJECTED |
| qb_continuity.md | 6 QB-continuity variants | REJECTED |
| qb_gated_experience.md | 5 gated experience variants | REJECTED |
| qb_magnitude.md | Rolling EPA, change magnitude | REJECTED |
| coach_qb_tenure.md | Coach-QB tenure interaction | REJECTED |
| coach_season_regression.md | Coach+QB regression bonus | REJECTED |
| situational_micro.md | Div, first-year coach, surface | REJECTED |
| team_hfa.md | Per-team HFA | WATCHLIST |
| home_away_elo.md | Separate home/away ratings | REJECTED |
| turnover_features.md | TO diff 3/5 game | WATCHLIST |
| epa_features.md | Team EPA rolling | REJECTED |
| comprehensive_efficiency.md | Team EPA + PFR + Snap | REJECTED |
| expressive_models.md | HGB, GB, RF on curated features | REJECTED |
| autogluon.md | AutoML on 47 features | REJECTED |
| glicko_rating.md | Glicko rating system | REJECTED |
| decayed_elo.md | Exponential momentum | REJECTED |
| residual_blending.md | Elo + week/rest blend | REJECTED |
| calibration_improvements.md | Era-split Platt, shrinkage | REJECTED |
| adaptive_k.md | Week-varying K | REJECTED |
| qb_injury_flag.md | Single QB OUT flag | REJECTED |
| qb_magnitude.md | QB quality magnitude | REJECTED |
| qb_market_delta.md | QB-level injury market pricing | DIAGNOSTIC |
| market_baseline.md | Moneyline comparison | DIAGNOSTIC |
| market_benchmark.md | Full market benchmark | DIAGNOSTIC |
| residual_diagnostics.md | Error analysis | DIAGNOSTIC |
| rolling_mov_sensitivity.md | Window sensitivity | PROMOTED (confirms mov_3) |
| optuna_feature_selection.md | L1 + TPE feature search | DIAGNOSTIC |
| elo_feature_selection_redo.md | Exhaustive feature redo | DIAGNOSTIC |

---

## Appendix: Enforced Rules

The following rules apply to all future model research:

1. **No retest without trigger** — Any feature family in REJECTED must not be retested unless one of the four triggers is met (new data, new source, live failure, diagnostic request).

2. **No pre-2021 data** — Allowed seasons are 2021–current only. No training, testing, backtesting, or tuning on earlier seasons.

3. **No market features as model inputs** — Market odds are benchmark-only. Model inputs must be pregame-safe and independent of market pricing.

4. **No ROI-based promotion** — Models are promoted on log loss, Brier score, and calibration. ROI is not a promotion metric.

5. **Promotion requires BOTH val and holdout improvement** — A challenger must beat the incumbent on BOTH rolling validation AND fitted-once holdout log loss. Holdout-only or val-only improvements are not sufficient.

6. **No feature from existing data without isolation** — Any new feature must be tested in isolation (incumbent + feature) and as part of its family. Full-family tests without individual isolation are diagnostic only.
