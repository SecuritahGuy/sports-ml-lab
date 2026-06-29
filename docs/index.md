# SportsLab / StatSpace NFL Research

**A reproducible ML research lab for pregame NFL win prediction.**
This project demonstrates disciplined probability modeling, walk-forward
validation, systematic feature governance, and strict leakage prevention.

> **This is not a betting bot.** Every model is evaluated by log loss,
> calibration, and leakage prevention — not ROI. Market data is used
> for diagnostic comparison only, never for training or selection.

---

## Current Status

| Attribute | Value |
|-----------|-------|
| **True incumbent** | Standard Elo + qb_changed + rolling_mov_3 + Platt |
| **Version** | v2.0.0 |
| **Average validation log loss** | 0.6334 |
| **2025 holdout log loss** | **0.6262** |
| **2025 holdout Brier score** | 0.2191 |
| **2025 holdout ROC AUC** | 0.7024 |
| **Active feature set** | qb_changed + rolling_mov_3 |
| **Validation method** | Rolling-origin 3-fold walk-forward |
| **Seasons used** | 2021–current only |
| **Market data** | Diagnostic only (closing moneyline: holdout 0.6090) |
| **Last updated** | 2026-06-24 |


---

## What This Project Demonstrates

### Rigorous validation
Every model is tested via **rolling-origin 3-fold walk-forward validation**
(2021→val 2022, 2021–2022→val 2023, 2021–2023→val 2024).
Selection uses average validation log loss. The **2025 season is a one-shot
holdout**, never accessed during model selection.

### Feature governance
19 feature families have been
tested and rejected. Each experiment is documented in a reproducible report.
No feature enters the model without proving itself on both validation and holdout.

### Probability focus
Optimized for **log loss, Brier score, and calibration** — not accuracy or ROI.
The project treats NFL prediction as a probabilistic modeling problem first.

### Reproducibility
Full prediction artifacts, benchmark registry, and model card are checked into
the repo. Run `make test` (645+ tests) to validate the entire pipeline.

---

## Current Research Incumbent

| Attribute | Value |
|-----------|-------|
| **Model** | Standard Elo + qb_changed + rolling_mov_3 + Platt |
| **Version** | v2.0.0 |
| **Feature set** | qb_changed + rolling_mov_3 |
| **Selection method** | Rolling-origin 3-fold validation + forward selection |
| **Average validation log loss** | 0.6334 |
| **2025 holdout log loss** | **0.6262** |
| **Calibration** | Platt scaling (logistic on Elo prob + features) |

### Rejected feature families (19 total)

Weather, scheduling/rest, EPA team efficiency, comprehensive efficiency
(58 columns), injury reports, QB identity OHE (catastrophic overfit),
tree models (overfit), AutoGluon AutoML, Glicko (all 432 configs worse),
team-specific HFA, home/away separate Elo, coach tenure, confidence
calibration tweaks, rolling MOV windows ≠ 3. See [Feature Usage Map](feature-usage-map)
and [Experiments](experiments) for full details.

---

## Registry Summary

| Metric | Count |
|--------|-------|
| Total experiments | 32 |
| Promoted / superseded | 6 |
| Rejected | 19 |
| Diagnostic | 7 |

## Pages

- [Benchmarks & Leaderboard](benchmarks) — incumbent, promotion rules, leaderboard
- [Predictions](predictions) — prediction dashboard, audit reports, artifacts, runbook
- [Model Card](model-card) — full model documentation
- [Experiments](experiments) — experiment reports grouped by outcome
- [Backtest Reports](backtests) — season-by-season analysis (2022–2025)
- [Research Roadmap](research-roadmap) — what worked, what failed, next steps
- [Feature Usage Map](feature-usage-map) — complete feature governance reference

## Research Philosophy

1. Predict probabilities, not vibes.
2. Optimize first for log loss, Brier score, calibration, and leakage prevention.
3. Accuracy is secondary.
4. ROI is not a primary model-promotion metric.
5. Do not use future data in features.
6. Every feature must be explainable and pregame-safe.
7. Every experiment report must include leakage risk.

## Quick Links

- Full predictions CSV: [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions.csv`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions.csv)
- Holdout predictions: [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions_2025_holdout.csv`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions_2025_holdout.csv)
- Weekly report: [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/predictions/weekly_report.md`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/predictions/weekly_report.md)
- Benchmark history: [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/benchmark_history.md`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/benchmark_history.md)
- Leaderboard CSV: [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/leaderboard.csv`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/benchmarks/leaderboard.csv)
- Prediction cards: [`https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/predictions/incumbent_prediction_cards.md`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/predictions/incumbent_prediction_cards.md)
