# SportsLab / StatSpace NFL Research

A reproducible ML research lab for pregame NFL win prediction.
**This is not a betting bot.** Every model is evaluated by log loss,
calibration, and leakage prevention — not ROI.

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

## Registry Summary

| Metric | Count |
|--------|-------|
| Total experiments | 32 |
| Promoted / superseded | 6 |
| Rejected | 19 |
| Diagnostic | 7 |

## Pages

- [Benchmarks & Leaderboard](benchmarks) — incumbent, promotion rules, leaderboard
- [Predictions](predictions) — prediction artifacts, holdout file, weekly report
- [Model Card](model-card) — full model documentation
- [Experiments](experiments) — experiment reports grouped by outcome

## Research Philosophy

1. Predict probabilities, not vibes.
2. Optimize first for log loss, Brier score, calibration, and leakage prevention.
3. Accuracy is secondary.
4. ROI is not a primary model-promotion metric.
5. Do not use future data in features.
6. Every feature must be explainable and pregame-safe.
7. Every experiment report must include leakage risk.

## Quick Links

- Full predictions CSV: [`https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions.csv`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions.csv)
- Holdout predictions: [`https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions_2025_holdout.csv`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions_2025_holdout.csv)
- Weekly report: [`https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/weekly_report.md`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/weekly_report.md)
- Benchmark history: [`https://github.com/timdev/sports-ml-lab/blob/main/reports/benchmarks/benchmark_history.md`](https://github.com/timdev/sports-ml-lab/blob/main/reports/benchmarks/benchmark_history.md)
- Leaderboard CSV: [`https://github.com/timdev/sports-ml-lab/blob/main/reports/benchmarks/leaderboard.csv`](https://github.com/timdev/sports-ml-lab/blob/main/reports/benchmarks/leaderboard.csv)
- Prediction cards: [`https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_prediction_cards.md`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_prediction_cards.md)
