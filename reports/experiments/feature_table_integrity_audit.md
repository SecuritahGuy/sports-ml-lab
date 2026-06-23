# Feature Table Integrity Audit

**Audited file:** `data/features/nfl/feature_table.parquet`  
**Audited files:** `src/sportslab/features/build_features.py`, `src/sportslab/features/weather.py`  
**Rows:** 1,424  
**Columns:** 46  

---

## Season Counts

| Season | Rows |
|--------|------|
| 2021   | 285  |
| 2022   | 284  |
| 2023   | 285  |
| 2024   | 285  |
| 2025   | 285  |

No seasons before 2021 are present. Season scope rule enforced.

---

## Target Column: `home_win`

**❌ Does NOT exist in the feature table.**

The feature builder correctly excluded `home_score` and `away_score` (leakage columns), but it did not create the target `home_win`. This must be derived from the raw schedules at modeling time by joining back to `data/raw/nfl/schedules.parquet`.

### Tie Games (4 total)

| Season | Week | Away | Home | Score |
|--------|------|------|------|-------|
| 2021   | 10   | DET  | PIT  | 16–16 |
| 2022   | 1    | IND  | HOU  | 20–20 |
| 2022   | 13   | WAS  | NYG  | 20–20 |
| 2025   | 4    | GB   | DAL  | 40–40 |

**Recommendation:** Exclude tie games from training for the binary baseline. `home_win` derivation: `1 if home_score > away_score else 0`. Drop rows where `home_score == away_score` (4 rows out of 1,424 — negligible).

---

## Leakage Columns Present in Feature Table

**None.** The builder correctly excluded all score/result columns:
- `away_score` ❌ absent
- `home_score` ❌ absent
- `result` ❌ absent
- `total` ❌ absent
- `overtime` ❌ absent
- `home_win` ❌ absent (needs to be added at training time)

---

## Market Columns Present in Feature Table

**None.** The builder correctly excluded all odds/spread columns:
- `away_moneyline` ❌ absent
- `home_moneyline` ❌ absent
- `spread_line` ❌ absent
- `away_spread_odds` ❌ absent
- `home_spread_odds` ❌ absent
- `total_line` ❌ absent
- `under_odds` ❌ absent
- `over_odds` ❌ absent

---

## Categorical Handling

12 columns are label-encoded with `_enc` suffix alongside their raw string versions.

**Issue:** Encoding is fitted on the full dataset (all seasons), which means the holdout set influenced the encoding scheme. For integer label encoding this is a minor concern because:
- All categories (team abbreviations, game types, stadiums) are stable across seasons
- New categories in the future would get a `-1` via `.cat.codes` for unseen values

**Recommendation for first baseline:** Acceptable as-is. For future iterations, move encoding inside a sklearn `Pipeline` with `LabelEncoder` fitted only on the training split.

---

## Weather Handling

8 Meteostat weather columns with the following missingness:

| Column | Missing % | Notes |
|--------|-----------|-------|
| `weather_precip` | 18.8% | Precipitation not recorded at all stations |
| `weather_cloud_cover` | 17.3% | Cloud cover not recorded at all stations |
| `weather_pressure` | 4.7% | Sparse missing |
| `weather_temp/tmin/tmax/humidity/wind` | 4.1% | 59 rows — mostly international venues |

Meteostat returns **historical observed values** — these are past weather readings, not forecasts. For historical modeling this is valid pregame information. For live inference you would need forecasts.

**Recommendation:** Exclude weather from the first pure baseline model. Weather adds missingness and complexity. Add it in a second iteration with imputation (mean, median, or simple flag for dome/indoor).

---

## Neutral Site Games

32 games have `location = "Neutral"` (international games + Super Bowls). For these games the "home" team is designated but has no true home-field advantage.

**Recommendation:** Exclude neutral games from the first baseline, or add a `is_neutral` flag. They represent only 2.2% of data.

---

## Recommended First Baseline Feature Set

Exclude the following from model features:

| Exclusion Reason | Columns |
|----------------|---------|
| Row identifiers (not features) | `game_id`, `gameday`, `gametime` |
| Redundant with encoded versions | Raw string cols (`away_team`, `home_team`, `game_type`, etc.) |
| Constant / near-constant | `location` (always "Home") |
| Weather (second iteration) | All `weather_*` columns |

First baseline feature set:

- `season`, `week` — temporal context
- `away_team_enc`, `home_team_enc` — team strength
- `away_rest`, `home_rest`, `rest_diff` — rest advantage
- `div_game` — divisional matchup flag
- `is_dome` — indoor environment
- `away_qb_id_enc`, `home_qb_id_enc` — starting QB
- `away_coach_enc`, `home_coach_enc` — head coach
- `stadium_id_enc` — venue
- `game_type_enc` — regular/postseason
- `weekday_enc` — day of week
- `roof_enc`, `surface_enc` — stadium characteristics

**Total first baseline features: 17**

---

## Recommended Split

| Split | Seasons | Rows (after tie drop) |
|-------|---------|-----------------------|
| Train | 2021–2023 | 854 (approx 852) |
| Validation | 2024 | 285 (approx 284) |
| Holdout | 2025 | 285 (approx 284) |
| Live forward eval | 2026 | TBD (future) |

---

## Pass/Fail Judgment

**❌ The feature table is NOT directly usable for modeling as-is.**

The table is structurally sound (no leakage, no market columns, seasons enforced), but it is missing the target column `home_win`. The target must be derived from the raw schedules before training.

---

## TODOs Before Training

1. **Add `home_win` target** — derive from raw schedules: `home_win = 1 if home_score > away_score else 0`, drop ties
2. **Decide neutral site handling** — flag or exclude 32 neutral games
3. **Decide weather inclusion** — exclude for first baseline, add later with imputation
4. **Move categorical encoding** into training pipeline for future iterations (acceptable for first baseline)
5. **Build training script** that loads feature table, joins target, selects feature columns, trains classifier

---

## Summary

| Question | Answer |
|----------|--------|
| `home_win` present? | ✅ Yes (patch 1) |
| Ties handled? | ✅ Yes (is_tie + model_eligible)` |
| Score columns excluded from features? | ✅ Yes |
| Market columns excluded from features? | ✅ Yes |
| Encoding before split? | ✅ Yes (minor concern, acceptable) |
| Weather historical? | ✅ Yes (historical observed) |
| Weather in first baseline? | ❌ Recommend exclude |
| 2021–current enforced? | ✅ Yes |
| Pre-2021 seasons present? | ❌ No |
| Split possible? | ✅ Yes (852 train / 284 val / 284 holdout) |
| **Pass for modeling?** | **✅ Yes — after filtering to model_eligible** |

---

## Patch 1 Addendum (2026-06-23)

### Changes Applied

The feature builder was patched to include the target column, tie handling, and neutral-site flag:

| Change | Details |
|--------|---------|
| `home_win` | Derived from raw scores: 1 if home_score > away_score, 0 if home_score < away_score, NA for ties |
| `is_tie` | Boolean column, True when home_score == away_score (4 rows) |
| `model_eligible` | Boolean column, True when home_win is not NA (1,420 rows) |
| `is_neutral` | Boolean column, True when location == "Neutral" (32 rows) |
| Leakage columns | Preserved in feature table (alongside market columns) for audit convenience; model code must exclude them via `BASELINE_FEATURE_COLUMNS` |
| Weather fix | `Daily`/`Stations` capitalisation resolved; weather fetch works with local cache |

### New Constants Exported from `build_features.py`

| Constant | Purpose |
|----------|---------|
| `TARGET_COLUMN = "home_win"` | Primary binary target for model training |
| `MODEL_ELIGIBLE_COLUMN = "model_eligible"` | Filtering flag — drop rows where False |
| `TIE_COLUMN = "is_tie"` | Tie indicator for audit / analysis |
| `NEUTRAL_COLUMN = "is_neutral"` | Neutral-site indicator |
| `WEATHER_COLUMNS` | List of 8 `weather_*` column names |
| `SPARSE_ID_COLUMNS` | List of 7 rarely-filled ID columns (old_game_id, gsis, etc.) |
| `BASELINE_FEATURE_COLUMNS` | 19 recommended features for first baseline model (includes `is_neutral`, excludes weather, leakage, market) |

### Current Feature Table Stats

| Metric | Value |
|--------|-------|
| Shape | 1,424 rows × 63 columns |
| model_eligible | 1,420 (99.7%) |
| is_tie | 4 (0.3%) — excluded from modeling |
| is_neutral | 32 (2.2%) — kept for first baseline |
| Baseline features | 19 available |
| Weather columns | 8 (with 4–18% missingness) |
| Weather missing rows | 59 (4.1%) |
| Leakage columns preserved | `away_score`, `home_score`, `result`, `total`, `overtime`, `home_win`, `is_tie` |
| Market columns preserved | `away_moneyline`, `home_moneyline`, `spread_line`, `away_spread_odds`, `home_spread_odds`, `total_line`, `under_odds`, `over_odds` |

### Validation

- All 42 existing and new tests pass
- `BASELINE_FEATURE_COLUMNS` excludes all leakage, market, and weather columns
- `SPARSE_ID_COLUMNS` are excluded from pregame and model features
- Home/away/aspect patterns remain safe (no final-score features in model inputs)
