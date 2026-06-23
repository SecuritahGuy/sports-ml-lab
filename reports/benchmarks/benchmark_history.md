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
| **Decision** | **Promoted as new incumbent (current)** |
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

### 8. Weather Features

| Field | Value |
|-------|-------|
| **Model** | Incumbent + temp/wind/precip/dome flags |
| **Selection** | Rolling-origin 3-fold |
| **Best config** | Incumbent + weather |
| **Validation LL** | 0.6445 (vs incumbent 0.6363) |
| **Holdout LL** | 0.6439 (vs incumbent 0.6373) |
| **Decision** | Rejected — weather harmed both validation and holdout |
| **Report** | `reports/experiments/weather_features.md` |
| **Date** | 2026-06-23 |

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
| **Decision** | **Promoted as new incumbent (current)** — beats previous 0.6373 by 0.0075 |
| **Report** | `reports/experiments/decayed_elo.md` |
| **Date** | 2026-06-23 |

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
| **Decision** | **Promoted as new incumbent** — beats previous 0.6285 by 0.0027 on holdout; clear monotonic pattern across 15 k_off/k_def combos |
| **Report** | `reports/experiments/od_elo.md` |
| **Date** | 2026-06-23 |

---

## Summary Statistics

| Total experiments | 19 |
|------------------|-----|
| Promoted | 6 |
| Rejected | 11 |
| Diagnostic | 2 |
| Current incumbent | O/D Elo (k_off=52, k_def=20, HFA=40, reg=0.1, decay=32, qb_bonus=0.2, MOV capped_linear) + Platt |
| Incumbent holdout LL | **0.6258** |
| Best challenger (pregame) | None |
| Best overall (diagnostic) | Market no-vig (0.6090 holdout) |
