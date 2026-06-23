# NFL Schedules Data Audit

**File:** `data/raw/nfl/schedules.parquet`  
**Source:** `nflreadpy.load_schedules()`  
**Ingested:** 2025-06-23  
**Rows:** 1,424  
**Columns:** 46  
**Seasons:** 2021, 2022, 2023, 2024, 2025  

---

## Game Types

| Type | Count | Meaning |
|------|-------|---------|
| REG  | 1,359 | Regular season |
| WC   | 30    | Wild Card |
| DIV  | 20    | Divisional |
| CON  | 10    | Conference Championship |
| SB   | 5     | Super Bowl |

All weeks 1–22 (regular + postseason) are present.

---

## Canonical Column Mapping

| Category | Raw Column | Notes |
|----------|-----------|-------|
| Game ID | `game_id` | Unique key |
| Season | `season` | 2021–2025 |
| Game type | `game_type` | REG/WC/DIV/CON/SB |
| Week | `week` | 1–22 |
| Game date | `gameday` | YYYY-MM-DD |
| Weekday | `weekday` | e.g. Sunday, Monday |
| Kickoff time | `gametime` | HH:MM (ET) |
| Away team | `away_team` | 3-letter abbreviation |
| Home team | `home_team` | 3-letter abbreviation |
| Home/away location | `location` | "Home" for all standard games |
| Away rest | `away_rest` | Days since last game |
| Home rest | `home_rest` | Days since last game |
| Division game | `div_game` | 1 if divisional matchup |
| Roof | `roof` | outdoors / dome / closed / open |
| Surface | `surface` | grass / fieldturf / astroturf |
| Temperature | `temp` | 42% missing (dome games) |
| Wind | `wind` | 42% missing (dome games) |
| Away QB | `away_qb_name` | Starting QB |
| Home QB | `home_qb_name` | Starting QB |
| Away QB ID | `away_qb_id` | nflverse player ID |
| Home QB ID | `home_qb_id` | nflverse player ID |
| Away coach | `away_coach` | Head coach |
| Home coach | `home_coach` | Head coach |
| Referee | `referee` | 1 missing row |
| Stadium | `stadium` / `stadium_id` | Human-readable + ID |
| Cross-ref IDs | `old_game_id`, `gsis`, `pfr`, `pff`, `espn`, `ftn`, `nfl_detail_id` | External IDs (not for features) |

---

## Score / Result Columns (LEAKAGE — exclude from features)

| Column | Description |
|--------|-------------|
| `home_score` | Home team final score |
| `away_score` | Away team final score |
| `result` | home_score - away_score |
| `total` | Combined final score |
| `overtime` | 1 if game went to OT |

**Rule:** These are target variables or direct leakage. Never include in model features.

---

## Market / Odds Columns (exclude from first pure model)

| Column | Description |
|--------|-------------|
| `away_moneyline` | Away ML odds |
| `home_moneyline` | Home ML odds |
| `spread_line` | Closing spread |
| `away_spread_odds` | Away spread odds |
| `home_spread_odds` | Home spread odds |
| `total_line` | Closing total |
| `under_odds` | Under odds |
| `over_odds` | Over odds |

**Rule:** Preserve for later market-aware model comparison. Do not include in the first non-market baseline model.

---

## Missingness Summary

| Column | Missing % | Note |
|--------|-----------|------|
| `nfl_detail_id` | 81% | Sparse — skip for features |
| `pff` | 61% | Sparse — skip for features |
| `temp`, `wind` | 42% | Missing for dome/closed-roof games — impute or flag |
| `ftn` | 21% | Sparse — skip for features |
| `referee` | <1% | 1 row missing — fill or drop |
| All other columns | 0% | Complete |

---

## Recommended First Feature Table Columns

These are the pregame-safe columns suitable for a first pure feature table:

- `game_id` — row key (do not use as a feature)
- `season` — year
- `week` — week number
- `game_type` — REG/WC/DIV/CON/SB
- `gameday`, `weekday`, `gametime` — temporal context
- `away_team`, `home_team` — team identifiers (need encoding)
- `location` — always "Home", effectively constant
- `away_rest`, `home_rest` — rest advantage
- `div_game` — divisional flag
- `roof`, `surface` — stadium characteristics
- `temp`, `wind` — weather (impute dome = NaN)
- `away_qb_id`, `home_qb_id` — starting QB identifiers
- `away_coach`, `home_coach` — head coach identifiers
- `stadium_id` — venue identifier

**TODOs before feature engineering:**

1. Weather: decide imputation for dome/closed-roof games (mean? flag column "is_dome"?)
2. Team IDs: one-hot, ordinal, or target-encode for 32 teams
3. QB/Coach IDs: handle turnover across seasons
4. `ot_flag` from `overtime` — is this available pregame? No — it's a result. Exclude.
5. `referee`: 1 missing row — fill with mode or drop row

---

## Recommendation

**Data is usable for the next step.** The schedules table is clean, well-structured, and contains 46 columns with very low missingness on the core fields. The known leakage columns (scores, result, total, overtime) and market columns (moneyline, spread, totals) are clearly separable from pregame-safe features.

### Next Step

Build the first feature table using the pregame-safe columns listed above, excluding:
- All leakage columns (away_score, home_score, result, total, overtime)
- All market/odds columns (moneyline, spread_line, total_line, odds)
- Sparse cross-reference IDs (nfl_detail_id, pff, ftn)

Create a cleaned feature DataFrame with imputed weather, indicator flags for dome games, and encoded team/coach/QB identifiers.
