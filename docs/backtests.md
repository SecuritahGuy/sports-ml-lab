# Backtest Reports

[Home](index) | [2026 Schedule](2026-schedule) | [Benchmarks](benchmarks) | [Predictions](predictions) | [Model Card](model-card) | [Experiments](experiments) | **Backtests**

---


The backtest evaluates the incumbent model
(**Standard Elo + qb_changed + rolling_mov_3 + Platt + frozen QB overlay**) across each NFL season.
Seasons 2022–2024 are in-training diagnostics (part of 2021–2024 training data).
Season 2025 is a locked holdout.

## Key Metrics (2025 Holdout)

| Metric | Value |
|--------|-------|
| Games | 276 |
| Log loss | **0.6200** |
| Brier | 0.2157 |
| AUC | 0.7098 |
| Accuracy | 0.6630 |

## Season-by-Season

### 2025 Season (Locked Holdout)

| Artifact | Description |
|----------|-------------|
| [Full Report](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2025_backtest_report.md) | Comprehensive Markdown report |
| [Weekly Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2025_weekly_summary.csv) | Week-by-week breakdown |
| [Team Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2025_team_summary.csv) | Per-team diagnostics |
| [Calibration Buckets](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2025_calibration_buckets.csv) | Confidence bucket analysis |
| [Extreme Games](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2025_extreme_games.csv) | Best/worst predictions |
| [Subgroup Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2025_subgroup_summary.csv) | Game-context breakdown |

### 2024 Season

| Artifact | Description |
|----------|-------------|
| [Full Report](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2024_backtest_report.md) | Comprehensive Markdown report |
| [Weekly Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2024_weekly_summary.csv) | Week-by-week breakdown |
| [Team Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2024_team_summary.csv) | Per-team diagnostics |
| [Calibration Buckets](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2024_calibration_buckets.csv) | Confidence bucket analysis |
| [Extreme Games](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2024_extreme_games.csv) | Best/worst predictions |
| [Subgroup Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2024_subgroup_summary.csv) | Game-context breakdown |

### 2023 Season

| Artifact | Description |
|----------|-------------|
| [Full Report](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2023_backtest_report.md) | Comprehensive Markdown report |
| [Weekly Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2023_weekly_summary.csv) | Week-by-week breakdown |
| [Team Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2023_team_summary.csv) | Per-team diagnostics |
| [Calibration Buckets](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2023_calibration_buckets.csv) | Confidence bucket analysis |
| [Extreme Games](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2023_extreme_games.csv) | Best/worst predictions |
| [Subgroup Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2023_subgroup_summary.csv) | Game-context breakdown |

### 2022 Season

| Artifact | Description |
|----------|-------------|
| [Full Report](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2022_backtest_report.md) | Comprehensive Markdown report |
| [Weekly Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2022_weekly_summary.csv) | Week-by-week breakdown |
| [Team Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2022_team_summary.csv) | Per-team diagnostics |
| [Calibration Buckets](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2022_calibration_buckets.csv) | Confidence bucket analysis |
| [Extreme Games](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2022_extreme_games.csv) | Best/worst predictions |
| [Subgroup Summary](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/reports/backtests/2022_subgroup_summary.csv) | Game-context breakdown |

## Research Caveats

- The model was trained on 2021–2024 data only. 2025 was a locked holdout.
- Seasons 2022–2024 were part of the training window and are diagnostic only.
- Market data is excluded from the model. Market benchmark: 0.6090 holdout LL.
- This is a probabilistic prediction benchmark, not a gambling product.
