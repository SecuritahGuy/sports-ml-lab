# StatSpace R&D Metrics Porting Plan

## Status: Investigation Complete

All 5 StatSpace NFL R&D metrics have been ported to `src/sportslab/features/statspace/`.
Imports fixed — all 8 modules compile cleanly. Next step: extract features for all
games and test each as a challenger against the incumbent (v3.0.0, holdout LL 0.6200).

---

## Metric Overview

| # | Metric | Standalone? | Lines | Complexity | Expected Value |
|---|--------|-------------|-------|------------|----------------|
| 1 | Coward Tax | ✅ Yes (PBP only) | 485 | Medium | Low (coaching decisions are rare events) |
| 2 | DOBA | ✅ Yes (PBP only) | 449 | Medium | Medium (efficiency composite) |
| 3 | Chaos Rate | ✅ Yes (PBP only) | 452 | Medium | Medium (defensive disruption) |
| 4 | FDR | ❌ Needs games + Elo | 628 | High | High (regression/fraud detection) |
| 5 | QB Lift Index | ✅ Yes (PBP only) | 676 | High | Low (QB signal already in overlay) |

## Data Requirements

All 5 metrics draw from the same nflverse PBP columns, which we already have
via `nfl_data_py` or `nflreadpy`:

| Column | Used By |
|--------|---------|
| `game_id`, `season`, `week` | All 5 |
| `posteam`, `defteam` | All 5 |
| `epa`, `success` | All 5 |
| `down`, `ydstogo`, `yardline_100` | Coward Tax, DOBA, Chaos |
| `wp`, `wpa`, `game_seconds_remaining`, `score_differential` | Coward Tax |
| `play_type`, `no_play`, `two_point_attempt`, `extra_point_attempt` | Coward Tax |
| `half_seconds_remaining`, `qb_kneel` | Coward Tax |
| `yards_gained`, `touchdown`, `interception`, `fumble_lost`, `sack` | DOBA, Chaos, QB Lift |
| `qb_dropback`, `rush_attempt` | DOBA, Chaos, QB Lift |
| `qb_scramble`, `yards_after_catch`, `passing_yards`, `complete_pass` | QB Lift |
| `cpoe`, `passer_player_id`, `passer_player_name` | QB Lift |
| `first_down_penalty`, `penalty_team` | Chaos |
| Home/away scores, win/loss, margin | FDR |
| Elo ratings | FDR |

## Build Plan

### Phase A: Adapter Layer (1 session)

Create `src/sportslab/features/statspace/adapter.py`:

```python
# One function per metric that:
# 1. Loads PBP via our existing load_pbp_data()
# 2. Calls the StatSpace metric function
# 3. Returns a DataFrame of team-season or team-game features
# 4. All features get SPORTSLAB_MIN_SEASON guard

def compute_statspace_coward_tax(...) -> pd.DataFrame
def compute_statspace_doba(...) -> pd.DataFrame
def compute_statspace_chaos(...) -> pd.DataFrame
def compute_statspace_fdr(...) -> pd.DataFrame
def compute_statspace_qb_lift(...) -> pd.DataFrame

# Also: helper to merge team-level metrics into our
# game-level feature table (home/away columns)
def augment_feature_table(features, metric_df, metric_name, ...) -> pd.DataFrame
```

**Key decisions:**
- All metrics produce per-TEAM-season values. For FDR (positive = fraud), QB Lift,
  DOBA, Chaos, Coward Tax. We'll create home/away columns by merging on team name.
- FDR needs game results + scores. We'll build `NFLHistoricalGame` objects from
  our existing schedule/feature table.
- QB Lift needs per-QB values. We'll aggregate to team-level by taking the
  primary QB's value (most dropbacks).

### Phase B: Individual Validation (5 experiments, ~1 session each)

For each metric, run a **rolling-origin experiment** following our
canonical protocol:

1. **Load PBP** for 2021–2025 (all seasons)
2. **Compute metric** for each season (team-level)
3. **Merge into feature table** — home/away columns for each game
4. **3-fold rolling-origin** with Platt calibration per fold
5. **Compare** vs incumbent (v3.0.0 Frozen QB Overlay)
6. **Promotion check**: Δ ≥ 0.001 on BOTH avg val LL AND holdout LL

**Order (highest expected value first):**

| Priority | Metric | Rationale |
|----------|--------|-----------|
| 1 | **FDR** | Most comprehensive — blends W-L, points, EPA, Elo, schedule. Directly targets the "record vs quality" weakness. |
| 2 | **DOBA** | Efficiency composite that our EPA rolling features don't capture. Includes explosive rate, dependency penalty. |
| 3 | **Chaos Rate** | Defensive disruption — our model has no defensive quality measure beyond what Elo learns from scores. |
| 4 | **Coward Tax** | Coaching aggression — weakest signal (rare events, noisy), but quick to test. |
| 5 | **QB Lift Index** | QB quality beyond Elo. Likely collinear with our existing QB overlay, but worth confirming. |

### Phase C: Combined Challenger (if any individual metric wins)

If 1+ metrics beat the incumbent individually, test them combined:
- FDR + DOBA
- FDR + DOBA + Chaos
- Best combo + Platt

### Phase D: Team-Site Integration (if promoted)

If any metric is promoted to the incumbent:
1. Add to `build_features.py` constants
2. Add to feature pipeline (`build-features` target)
3. Add to team site (team profile cards, standings)
4. Update benchmark registry

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **PBP mismatch**: StatSpace uses nflverse PBP via `nfl_data_py`; we use `nflreadpy` | Low | Both pull from same nflverse source; column names are identical |
| **Performance**: Building all 5 metrics requires loading full PBP for 5 seasons | Medium | Cache PBP locally; compute all 5 in one pass |
| **Leakage**: Any metric must not use future data | Low | All metrics compute per-season aggregates; use pre-game-available stats only |
| **Collinearity**: DOBA/Chaos are composites of features we already tested (EPA, success rate) | Medium | That's the experiment — if they add nothing, we reject |
| **FDR complexity**: Needs Elo + schedule loader; may behave differently with our Elo | Medium | Use our own Elo ratings from `ratings.py` instead of `NFLEloEngine` |
| **QA overfitting**: FDR has 6 weighted components — tuning on past data could overfit | Low | Use default weights from StatSpace; do NOT optimize weights on holdout |

---

## Timeline

| Session | Phase | Estimated Time |
|---------|-------|---------------|
| 1 | A — Adapter layer | 30 min |
| 2 | B1 — FDR experiment | 30 min |
| 3 | B2 — DOBA experiment | 20 min |
| 4 | B3 — Chaos experiment | 20 min |
| 5 | B4 — Coward Tax experiment | 15 min |
| 6 | B5 — QB Lift experiment | 20 min |
| 7 | C — Combined (if needed) | 20 min |
| 8 | D — Promotion (if needed) | 30 min |

**Total: ~3 hours if all phases needed.**

---

## Files

| File | Purpose |
|------|---------|
| `src/sportslab/features/statspace/__init__.py` | Package init |
| `src/sportslab/features/statspace/nfl_coward_tax.py` | Coward Tax (ported from StatSpace) |
| `src/sportslab/features/statspace/nfl_doba.py` | DOBA (ported) |
| `src/sportslab/features/statspace/nfl_chaos_rate.py` | Chaos Rate (ported) |
| `src/sportslab/features/statspace/nfl_qb_lift_index.py` | QB Lift Index (ported) |
| `src/sportslab/features/statspace/nfl_branded_stats.py` | FDR (ported) |
| `src/sportslab/features/statspace/nfl_elo.py` | Elo engine (ported, for FDR) |
| `src/sportslab/features/statspace/nfl_predictive_features.py` | PBP aggregation (ported, for FDR) |
| `src/sportslab/features/statspace/nfl_schedule_loader.py` | Schedule loader (ported, for FDR) |
| `src/sportslab/features/statspace/adapter.py` | Adapter layer (to be built) |
| `reports/benchmarks/statspace_rd_port_plan.md` | This plan |
