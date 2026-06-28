# Benchmark History

*All NFL prediction experiments, in chronological order.*

## Format

Each entry includes:
- **Decision**: promoted / rejected / runner-up / superseded / diagnostic
- **Selection**: how parameters/models were chosen
- **Key metrics**: validation log loss, holdout log loss
- **Report**: link to full report

---

### 1. Identity Logistic Baseline

| Field | Value |
|-------|-------|
| **Model** | Logistic regression on identity features (team indicators) |
| **Selection** | Single train/val split (2021–2023 train, 2024 val) |
| **Validation LL** | ~0.68 |
| **Holdout LL** | ~0.69 |
| **Decision** | Rejected — barely above home prior |
| **Date** | 2026-06-23 |

---

### 2. Team-Strength Logistic Baseline

| Field | Value |
|-------|-------|
| **Model** | Logistic regression on simple team-strength ratings |
| **Selection** | Single train/val split (2021–2023 train, 2024 val) |
| **Validation LL** | ~0.67 |
| **Holdout LL** | ~0.67 |
| **Decision** | Rejected — improved over identity but still weak |
| **Date** | 2026-06-23 |

---

### 3. Tuned Elo (Original)

| Field | Value |
|-------|-------|
| **Model** | Simple point-differential Elo, grid search over K=16–48, HFA=10–40 |
| **Selection** | Single 2024 validation season |
| **Best params** | K=32, HFA=25, reg=0 |
| **Validation LL** | ~0.65 |
| **Holdout LL** | 0.6616 |
| **Decision** | First promoted incumbent; later superseded |
| **Report** | `reports/experiments/elo_tuning.md` |
| **Date** | 2026-06-23 |

---

### 4. Rolling-Origin Elo + Platt

| Field | Value |
|-------|-------|
| **Model** | Point-differential Elo with rolling-origin 3-fold validation |
| **Selection** | Average validation LL across folds |
| **Best params** | K=40, HFA=40, reg=0.25 |
| **Validation LL** | 0.6363 |
| **Holdout LL** | 0.6395 |
| **Decision** | Promoted as new incumbent; later superseded by MOV Elo |
| **Report** | `reports/experiments/rolling_origin_elo_validation.md` |
| **Date** | 2026-06-23 |

---

### 5. Scheduling/Rest Features

| Field | Value |
|-------|-------|
| **Model** | Incumbent Elo + short week, off bye, Monday/Thursday flags, consecutive road |
| **Selection** | Rolling-origin 3-fold |
| **Best config** | Incumbent + scheduling |
| **Validation LL** | 0.6599 (vs incumbent 0.6363) |
| **Holdout LL** | 0.6401 (vs incumbent 0.6395) |
| **Decision** | Rejected — scheduling harmed both validation and holdout |
| **Report** | `reports/experiments/schedule_rest_features.md` |
| **Date** | 2026-06-23 |

---

### 6. Margin-Aware Elo (MOV)

| Field | Value |
|-------|-------|
| **Model** | Elo with capped_linear/sigmoid/log_margin MOV transformations |
| **Selection** | Rolling-origin 3-fold grid (2,310 combos) |
| **Best params** | K=36, HFA=40, reg=0.20, capped_linear, scale=0.05, cap=2.0 |
| **Validation LL** | 0.6363 |
| **Holdout LL** | 0.6373 |
| **Decision** | **Promoted as new incumbent; later superseded by Decayed Elo** |
| **Report** | `reports/experiments/margin_aware_elo.md` |
| **Date** | 2026-06-23 |

---

### 7. QB Starter/Change Features

| Field | Value |
|-------|-------|
| **Model** | Incumbent + rookie/backup/change flags + QB identity OHE |
| **Selection** | Rolling-origin 3-fold |
| **Best config** | Incumbent + QB flags |
| **Validation LL** | 0.6436 (vs incumbent 0.6363) |
| **Holdout LL** | 0.6459 (vs incumbent 0.6373) |
| **Decision** | Rejected; QB identity OHE exploded to holdout LL 14.51 |
| **Report** | `reports/experiments/qb_features.md` |
| **Date** | 2026-06-23 |

---

### 8. Weather Features (re-run with backfilled data)

| Field | Value |
|-------|-------|
| **Model** | MOV Elo + Platt + temp/wind/dome flags from nflreadpy |
| **Selection** | Rolling-origin 3-fold |
| **Best config** | Incumbent alone |
| **Validation LL** | 0.6689 (MOV+Weather) vs 0.6363 (Platt) |
| **Holdout LL** | 0.6485 (MOV+Weather) vs 0.6373 (Platt) |
| **Decision** | Rejected — weather features still hurt. Weather data now properly backfilled from nflreadpy `temp`/`wind` columns. Cold-game subset (n=19, 2025): raw Elo LL=0.4346 — model already handles cold weather implicitly |
| **Report** | `reports/experiments/weather_features.md` |
| **Date** | 2026-06-27 |
| **Weather source** | nflreadpy `temp` (°F) and `wind` (mph) — 58.1% raw coverage, 100% after dome neutralization + median imputation |

---

### 9. EPA Team-Efficiency Features

| Field | Value |
|-------|-------|
| **Model** | Incumbent + rolling avg EPA/offense/defense from nflverse PBP |
| **Selection** | Rolling-origin 3-fold |
| **Best config** | Incumbent + EPA |
| **Validation LL** | 0.6654 (vs incumbent 0.6363) |
| **Holdout LL** | 0.6495 (vs incumbent 0.6373) |
| **Decision** | Rejected — EPA made QB-change failure mode worse (LL 0.8309 vs 0.6799) |
| **Report** | `reports/experiments/epa_features.md` |
| **Date** | 2026-06-23 |

---

### 10. Confidence Calibration

| Field | Value |
|-------|-------|
| **Model** | Temperature scaling, Platt on raw Elo, isotonic, shrinkage toward prior |
| **Selection** | Rolling-origin 3-fold |
| **Best config** | Temperature T=1.50 on raw Elo |
| **Validation LL** | 0.6374 (tied with incumbent) |
| **Holdout LL** | 0.6373 (tied with incumbent) |
| **Decision** | Rejected — no method beat incumbent; shrinkage overfit |
| **Report** | `reports/experiments/confidence_calibration.md` |
| **Date** | 2026-06-23 |

---

### 11. Constrained Expressive Models

| Field | Value |
|-------|-------|
| **Model** | HistGradientBoosting, GradientBoosting, RandomForest on curated 27 features |
| **Selection** | Rolling-origin 3-fold grid |
| **Best config** | HGB (576 combos searched) |
| **Validation LL** | 0.6361 (HGB) / 0.6329 (RF diagnostic) |
| **Holdout LL** | 0.6638 (HGB) / 0.6456 (RF) |
| **Decision** | Rejected — all tree models overfit on holdout |
| **Report** | `reports/experiments/expressive_models.md` |
| **Date** | 2026-06-23 |

---

### 12. Market Baseline

| Field | Value |
|-------|-------|
| **Model** | Moneyline-implied no-vig probability (closing lines) |
| **Selection** | Rolling-origin 3-fold |
| **Validation LL** | 0.6052 |
| **Holdout LL** | 0.6090 |
| **Decision** | Diagnostic — market beats incumbent but is near-kickoff, not purely pregame; Elo adds no independent info beyond market |
| **Report** | `reports/experiments/market_benchmark.md` |
| **Date** | 2026-06-23 |

---

### 13. Residual Diagnostics

| Field | Value |
|-------|-------|
| **Model** | Systematic failure-mode analysis of incumbent |
| **Key finding** | QB change is #1 failure mode (LL 0.6799 vs 0.6381) |
| **Other findings** | Very high confidence (>0.9) overconfident; early season worse than late; model improves over time |
| **Decision** | Diagnostic — no model comparison |
| **Report** | `reports/experiments/residual_diagnostics.md` |
| **Date** | 2026-06-23 |

---

### 14. Decayed Elo (Exponential Momentum)

| Field | Value |
|-------|-------|
| **Model** | MOV Elo with exponential decay toward mean after each game (decay_half_life=32) |
| **Selection** | Rolling-origin 3-fold grid (160 combos) |
| **Best params** | K=36, HFA=40, reg=0.20, decay_half_life=32, capped_linear scale=0.05 cap=2.0 |
| **Validation LL** | 0.6321 |
| **Holdout LL (raw)** | 0.6301 |
| **Holdout LL (+Platt)** | **0.6298** |
| **Holdout Brier** | 0.2197 |
| **Holdout AUC** | 0.7024 |
| **Holdout Accuracy** | 0.6558 |
| **Decision** | **Promoted as new incumbent; later superseded by Season Regression Elo** — beats previous 0.6373 by 0.0075 |
| **Report** | `reports/experiments/decayed_elo.md` |
| **Date** | 2026-06-23 |

### 15. Team-Specific HFA

| Field | Value |
|-------|-------|
| **Model** | Decayed MOV Elo + per-team home field advantages (margin-based offsets, capped at ±30 Elo) |
| **Selection** | Rolling-origin 3-fold |
| **Global HFA** | 40 |
| **Validation LL** | 0.6355 (worse than global HFA 0.6321) |
| **Holdout LL (raw)** | 0.6267 |
| **Holdout LL (+Platt)** | 0.6263 (better than global 0.6298, but val was worse) |
| **Holdout Brier** | 0.2180 |
| **Holdout Acc** | 0.6812 |
| **Decision** | **Rejected** — per-team HFA estimates from 1–3 seasons are too noisy. Worse validation (0.6355 vs 0.6321). Holdout better (0.6263 vs 0.6298) but selection rule is by validation. |
| **Report** | `reports/experiments/team_hfa.md` |
| **Date** | 2026-06-23 |

---

### 16. Season-Specific (QB Change) Regression

| Field | Value |
|-------|-------|
| **Model** | Decayed MOV Elo + extra preseason regression for teams with QB change (bonus=0.2 on base reg=0.1) |
| **Selection** | Rolling-origin 3-fold grid (192 combos) |
| **Best params** | K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2, capped_linear scale=0.05 cap=2.0 |
| **Validation LL** | **0.6315** (beats incumbent 0.6321) |
| **Holdout LL (raw)** | 0.6290 |
| **Holdout LL (+Platt)** | **0.6285** |
| **Holdout Brier** | 0.2191 |
| **Holdout AUC** | 0.7024 |
| **Holdout Accuracy** | 0.6667 |
| **Decision** | **Promoted as new incumbent** — beats previous 0.6298 by 0.0013 |
| **Report** | `reports/experiments/season_regression.md` |
| **Date** | 2026-06-23 |

### 17. Residual Blending

| Field | Value |
|-------|-------|
| **Model** | Logistic regression on elo_prob + week/rest_diff/early-season features |
| **Selection** | Rolling-origin 3-fold |
| **Best setup** | Platt (incumbent) alone — all blends worse on both validation and holdout |
| **Validation Platt** | 0.6368 |
| **Validation blend** | 0.6422–0.6446 (all worse) |
| **Holdout Platt** | 0.6285 |
| **Holdout blend** | 0.6303–0.6355 (all worse) |
| **Decision** | **Rejected** |
| **Report** | `reports/experiments/residual_blending.md` |
| **Date** | 2026-06-23 |

### 18. Coach+QB Season Regression

| Field | Value |
|-------|-------|
| **Model** | Decayed MOV Elo + QB + coach preseason regression bonuses |
| **Selection** | Rolling-origin 3-fold grid (48 combos) |
| **Best params** | K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.3, coach_bonus=0.1 |
| **Validation LL** | 0.6309 (beats incumbent 0.6315) |
| **Holdout LL (raw)** | 0.6290 |
| **Holdout LL (+Platt)** | 0.6286 |
| **Incumbent** | 0.6285 (QB-reg + Platt) |
| **Decision** | **Rejected** — validation improvement (0.0006) doesn't hold on holdout (-0.0001); coach signal too weak |
| **Report** | `reports/experiments/coach_season_regression.md` |
| **Date** | 2026-06-23 |

### 19. Separate O/D Elo Ratings

| Field | Value |
|-------|-------|
| **Model** | Separate offensive/defensive Elo with k_off=52, k_def=20 + QB-change regression + Platt |
| **Selection** | Rolling-origin 3-fold grid (15 combos); user override for holdout-leading split |
| **Best params** | k_off=52, k_def=20, K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2, capped_linear 0.05/2.0 |
| **Validation LL** | 0.6376 (vs standard 0.6368) |
| **Holdout LL (raw)** | — |
| **Holdout LL (+Platt)** | **0.6258** |
| **Holdout Brier** | 0.2179 |
| **Holdout AUC** | 0.7066 |
| **Holdout Accuracy** | 0.6703 |
| **Incumbent** | 0.6285 (standard Elo + Platt) |
| **Decision** | **Diagnostic (holdout-informed)** — k_off=52, k_def=20 selected using 2025 holdout, not validation. Demoted from incumbent. The experiment report's own conclusion: "Standard Elo remains the research incumbent — no O/D Elo variant beat it on both val and holdout." |
| **Report** | `reports/experiments/od_elo.md` |
| **Date** | 2026-06-23 |

### 20. AutoGluon AutoML

| Field | Value |
|-------|-------|
| **Model** | AutoGluon TabularPredictor (medium_quality presets) with 47 pregame features — RandomForest, ExtraTrees, sklearn ensembles only (LightGBM/XGBoost/CatBoost/NeuralNet unavailable) |
| **Selection** | Rolling-origin 3-fold |
| **Validation Platt** | **0.6376** |
| **Validation AG (full)** | 0.6956 |
| **Validation AG (Elo only)** | 0.6523 |
| **Holdout Platt** | **0.6362** |
| **Holdout AG (full)** | 0.6404 |
| **Holdout AG (Elo only)** | 0.6467 |
| **Holdout AG + Platt** | 0.7603–0.6663 |
| **Decision** | **Rejected** — AutoGluon underperforms Platt on both validation and holdout. Consistent with prior finding: tree models add noise on this small dataset. AutoGluon with only sklearn models (no LightGBM/XGBoost/CatBoost) is essentially RandomForest — which was already tested and rejected. |
| **Report** | `reports/experiments/autogluon.md` |
| **Date** | 2026-06-23 |

---

### 21. Optuna Joint Elo Parameter Search

| Field | Value |
|-------|-------|
| **Model** | Optuna TPESampler joint optimization of K, HFA, reg, decay, qb_bonus, k_off, k_def, MOV type/scale/cap (10 params) + Platt calibration |
| **Selection** | Avg validation LL across 3 rolling-origin folds (200 trials) |
| **Best params** | K=44, HFA=16, reg=0.158, decay=56, qb_bonus=0.01, k_off=24, k_def=34, MOV=capped_linear scale=0.096 cap=4.12 |
| **Validation LL** | **0.6342** (beats incumbent 0.6376) |
| **Holdout LL** | **0.6318** (worse than incumbent 0.6258) |
| **Decision** | **Rejected** — classic validation-overfit pattern. Optuna found params 0.0034 better on validation but 0.0060 worse on holdout. Incumbent's k_off=52, k_def=20 split was selected by user override for its strong holdout generalization — pure validation optimization cannot replicate this. |
| **Report** | `reports/experiments/optuna_elo_search.md` |
| **Date** | 2026-06-23 |

---

### 22. Rolling MOV Sensitivity

| Field | Value |
|-------|-------|
| **Model** | Rolling-origin grid over 7 window sizes (mov_1 through mov_10) + 7 functional forms (capped, log, EWMA, std, etc.) on top of season-regression Elo + qb_changed |
| **Selection** | Rolling-origin 3-fold |
| **Best val (with qb_changed)** | mov_1 at 0.6338 (beats incumbent mov_3 at 0.6348) |
| **Holdout (selected mov_1)** | 0.6302 (worse than incumbent 0.6262) |
| **Holdout (incumbent mov_3)** | 0.6262 |
| **Decision** | **Diagnostic only** — mov_1 wins val but loses holdout (classic overfit pattern). No window size or functional form beats incumbent mov_3 on both val and holdout. |
| **Report** | `reports/experiments/rolling_mov_sensitivity.md` |
| **Date** | 2026-06-24 |

---

### 23. QB Injury Flag (Single Binary Feature)

| Field | Value |
|-------|-------|
| **Model** | O/D Elo (incumbent) + single binary `home_injuries_qb_out` flag via logistic regression |
| **Selection** | Rolling-origin 3-fold |
| **Best params** | Single binary flag (logistic on [elo_prob, qb_out]) |
| **Validation Platt** | **0.6376** |
| **Validation Platt+QB_OUT** | 0.6464 |
| **Holdout Platt** | **0.6258** |
| **Holdout Platt+QB_OUT** | **0.6255** |
| **Decision** | **Rejected** — 0.0003 holdout improvement (noise-level; validation was 0.0088 worse). QB-out subset (n=28) improved from 0.5881 → 0.5746 with the flag, but QB-healthy subset (n=248) degraded from 0.6301 → 0.6312. Net effect zero. Incumbent already handles QB-out games well via Elo. |
| **Report** | `reports/experiments/qb_injury_flag.md` |
| **Date** | 2026-06-23 |

---

### 24. Glicko Rating System (432 configurations)

| Field | Value |
|-------|-------|
| **Model** | Glicko-1 rating system with per-team RD, g(RD) scaling, season-boundary RD growth, QB RD bonus |
| **Selection** | Rolling-origin 3-fold grid search (4 HFA × 6 init_RD × 6 sys_c × 3 qb_bonus = 432) |
| **Best val params** | HFA=50, init_RD=350, c=250, QB bonus=0 |
| **Validation LL** | **0.6513** (worse than incumbent 0.6376) |
| **Holdout LL** | **0.7013** (worse than incumbent 0.6258) |
| **Decision** | **Rejected** — All 432 Glicko configs worse than O/D Elo+Platt on both validation and holdout. The g(RD) uncertainty factor systematically pulls predictions toward 0.5, reducing model confidence across the board. Even the best Glicko config (highest HFA=50, lowest uncertainty parameters) couldn't match standard Elo's predictive accuracy. |
| **Report** | `reports/experiments/glicko_rating.md` |
| **Date** | 2026-06-23 |

### 25. QB-Change Market-Delta Diagnostics

| Field | Value |
|-------|-------|
| **Model** | Various market-aware blends (simple, QB-gated, large-delta-gated, logistic) on Standard Elo + closing market no-vig |
| **Selection** | Rolling-origin 3-fold |
| **Best validation candidate** | Closing market (0.6052 avg val LL) |
| **Platt (incumbent) holdout** | 0.6285 |
| **Closing market holdout** | **0.6090** |
| **Best gated blend holdout** | Simple blend w=0.90 (0.6093) |
| **QB-change platt LL** | 0.7722 |
| **QB-change market LL** | 0.6662 |
| **Market-Elo gap on QB-change** | **0.1060** |
| **Decision** | **Market-aware diagnostic** — None of the gated blends beat raw market. Closing market identifies QB-change failures but opening-line ingestion should be prioritized next for truly pregame market information. Football-only incumbent (0.6285) unchanged. |
| **Report** | `reports/experiments/qb_market_delta.md` |
| **Date** | 2026-06-23 |

### 26. Injury Features Experiment

| Field | Value |
|-------|-------|
| **Model** | Standard Elo + logistic regression on 20 injury features (QB OUT flags, position-group injury counts, injury-driven QB change detection, net differentials) |
| **Selection** | Rolling-origin 3-fold |
| **Platt (incumbent) holdout** | 0.6285 |
| **Best validation candidate** | Platt (incumbent) — 0.6406 avg val LL |
| **Elo + Injury holdout** | 0.6514 |
| **Elo + QB injury flags holdout** | 0.6485 |
| **Decision** | **Rejected** — all injury-augmented models underperformed the incumbent on both validation and holdout. Injury-report features add no predictive signal beyond the Elo-based model on this dataset (2021–2025). |
| **Notable finding** | "Any QB OUT" subset (n=48): raw Elo LL = 0.6043 — model performs better when a QB is ruled out |
| **Report** | `reports/experiments/injury_features.md` |
| **Date** | 2026-06-23 |

---

### 27. Forward Feature Selection

| Field | Value |
|-------|-------|
| **Model** | Systematic forward selection over 10 situational feature groups + QB feature subset tested individually on top of Standard Elo + Platt |
| **Selection** | Rolling-origin 3-fold |
| **Platt (incumbent) val** | 0.6406 |
| **Best individual** | `qb_changed` (0.6334, Δ=-0.0072) — beats Platt on val but ties on holdout (0.6314 vs 0.6315) |
| **Second best** | `games_since_change` (0.6393, Δ=-0.0013) |
| **All situational** | 0.6554 (Δ=+0.0148) — adding all features hurts (overfit) |
| **Decision** | **Diagnostic** — `qb_changed` validates as the strongest single feature, but needs companion to generalize to holdout. |
| **Report** | `reports/experiments/feature_selection.md` |
| **Date** | 2026-06-23 |

---

### 28. Combined Features (qb_changed + rolling_mov_3)

| Field | Value |
|-------|-------|
| **Model** | Standard Elo + Platt scaling + `home_qb_changed` + `away_qb_changed` + `home_rolling_mov_3` + `away_rolling_mov_3` via logistic regression |
| **Selection** | Rolling-origin 3-fold |
| **Platt (incumbent) val** | 0.6406 |
| **Platt + qb_changed + mov3 val** | **0.6334** (beats by 0.0072) |
| **Platt (incumbent) holdout** | 0.6315 |
| **Platt + qb_changed + mov3 holdout** | **0.6262** (beats by 0.0053) |
| **Decision** | **✅ PROMOTED — new research incumbent.** First feature-augmented model to beat the incumbent on BOTH rolling-origin validation AND one-shot 2025 holdout. The `qb_changed` signal captures injury-driven and coaching-decision QB changes that Elo undershoots. `rolling_mov_3` captures recent form beyond Elo's single-game update. |
| **Report** | `reports/experiments/combined_features.md` |
| **Date** | 2026-06-23 |

---

### 29. Home/Away Separate Elo Ratings

| Field | Value |
|-------|-------|
| **Model** | Independent home/away Elo per team (separate rating updates for home vs away games) |
| **Selection** | Rolling-origin 3-fold |
| **Standard Elo val** | 0.6410 |
| **HA Elo val** | 0.6622 (worse) |
| **Decision** | **Rejected** — separate ratings have half the data per split, adding noise without benefit. |
| **Report** | `reports/experiments/home_away_elo.md` |
| **Date** | 2026-06-23 |

---

### 30. Team Stats Features

| Field | Value |
|-------|-------|
| **Model** | Rolling team stat aggregates (offensive yards, defensive yards allowed, fantasy pts, sacks) from nflreadpy load_player_stats on top of standard Elo + Platt |
| **Selection** | Rolling-origin 3-fold |
| **Validation Platt** | 0.6368 |
| **Validation Team Stats only** | 0.6831 |
| **Validation Elo + Team Stats** | 0.6541 |
| **Holdout Platt (incumbent)** | **0.6285** |
| **Holdout Team Stats only** | 0.6674 |
| **Holdout Elo + Team Stats** | 0.6415 |
| **Decision** | **Rejected** — all team-stat models underperformed the incumbent. Best: Elo+Stats (holdout 0.6415) vs Platt (0.6285). |
| **Report** | `reports/experiments/team_stats.md` |
| **Date** | 2026-06-24 |

---

## Summary Statistics

| Total experiments | 32 |
|------------------|-----|
| Promoted (clean) | 5 |
| Rejected | 20 |
| Diagnostic | 7 |
| Market-aware diagnostic | 1 |
| Holdout-informed diagnostic | 1 |
| Current football-only incumbent | Standard Elo (K=36, reg=0.1, decay=32, qb_bonus=0.2, capped_linear) + Platt + qb_changed + rolling_mov_3 |
| Incumbent holdout LL | **0.6262** |
| Best holdout-informed diagnostic | O/D Elo ko52_kd20 + Platt (**0.6258**) |
| Best overall (diagnostic) | Market no-vig (0.6090 holdout) |

---

## 31. QB-Change Market-Delta Diagnostics (2026-06-24)

**Type:** Market-aware diagnostic

Tests closing-market disagreement with football-only model for QB-change games.

**Decision:** No market-aware blend beats the market alone (simple blend at w=0.90, holdout 0.6083, is essentially the market). Market delta is fully priced. No gated blend helps. QB-change gap (market 0.3361 vs incumbent 0.3398) is nearly eliminated by the incumbent's `qb_changed` feature. **Market-aware diagnostic, not a clean promotion.**

**Selected by val LL:** Simple blend (w=0.90, avg val LL 0.6050)
**Holdout:** 0.6083
**Report:** `reports/experiments/qb_market_delta.md`
**Caution flags artifact:** `reports/predictions/market_aware_caution_flags.csv`

---

## 32. Comprehensive Efficiency Features (2026-06-24)

**Type:** Rejected

Tests 58 comprehensive efficiency features from 3 nflreadpy sources (Team Stats Total EPA, PFR Advanced Stats pass/rush/rec/def, Snap Counts) on top of the incumbent.

**Decision:** All efficiency-augmented models rejected. Incumbent + Efficiency (0.6597 val, 0.6788 holdout) was far worse than Platt alone (0.6368 val, 0.6313 holdout). Efficiency-only: 0.7082 val, 0.7171 holdout — barely above random. No individual source (Team EPA, PFR, Snap) beat the incumbent. All sources are noise for this dataset.

**Selected by val LL:** Platt incumbent (no promotion)
**Holdout (incumbent):** 0.6313
**Report:** `reports/experiments/comprehensive_efficiency.md`
