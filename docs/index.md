# SportsLab / StatSpace NFL Research

**Home** | [2026 Schedule](2026-schedule) | [Benchmarks](benchmarks) | [Predictions](predictions) | [Model Card](model-card) | [Experiments](experiments) | [Backtests](backtests)

---


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
| **Incumbent** | Standard Elo + qb_changed + rolling_mov_3 + Platt + frozen QB overlay |
| **Version** | v3.0.0 |
| **Avg validation log loss** | 0.6305 |
| **2025 holdout log loss** | **0.6200** |
| **2025 holdout Brier** | 0.2157 |
| **2025 holdout AUC** | 0.7098 |
| **Feature set** | qb_changed + rolling_mov_3 + frozen QB overlay |
| **Validation** | Rolling-origin 3-fold walk-forward |
| **Seasons** | 2021–current only |
| **Market benchmark** | Closing moneyline holdout 0.6090 (diagnostic) |
| **2026 predictions** | 15 games generated |
| **Last updated** | 2026-06-30 20:18 UTC |

---

## What This Project Demonstrates

### Rigorous validation
Every model is tested via **rolling-origin 3-fold walk-forward validation**
(2021→val 2022, 2021–2022→val 2023, 2021–2023→val 2024).
Selection uses average validation log loss. The **2025 season is a one-shot
holdout**, never accessed during model selection.

### Feature governance
24 feature families have been tested and rejected. Each experiment is
documented in a reproducible report. No feature enters the model without
proving itself on both validation and holdout.

### Probability focus
Optimized for **log loss, Brier score, and calibration** — not accuracy or ROI.
The project treats NFL prediction as a probabilistic modeling problem first.

### Reproducibility
Full prediction artifacts, benchmark registry, and model card are checked into
the repo. Run `make test` (650+ tests) to validate the entire pipeline.

---

## Research Philosophy

1. Predict probabilities, not vibes.
2. Optimize first for log loss, Brier score, calibration, and leakage prevention.
3. Accuracy is secondary.
4. ROI is not a primary model-promotion metric.
5. Do not use future data in features.
6. Every feature must be explainable and pregame-safe.
7. Every experiment report must include leakage risk.

---

## Registry Summary

| Metric | Count |
|--------|-------|
| Total experiments | 39 |
| Promoted / superseded | 7 |
| Rejected | 24 |
| Diagnostic | 8 |
