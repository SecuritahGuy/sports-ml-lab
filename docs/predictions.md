# Predictions & Artifacts

[Home](index) | [2026 Schedule](2026-schedule) | [Benchmarks](benchmarks) | **Predictions** | [Model Card](model-card) | [Experiments](experiments) | [Backtests](backtests)

---


## Prediction Schema

Each prediction CSV contains:

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `season`, `week`, `gameday` | Game timing |
| `away_team`, `home_team` | Teams |
| `incumbent_home_win_prob` | Predicted home win probability |
| `predicted_winner` | Home or away team based on prob |
| `confidence_bucket` | Probability range bucket |
| `model_version` | Incumbent version at prediction time |
| `feature_set` | Features used |
| `calibration_method` | Calibration type |
| `qb_source` | QB data source (oracle / live_pregame / auto_qb) |
| `caution_qb_change` | QB changed from prior game |
| `caution_early_season` | Weeks 1–4 |

## Confidence Buckets

| Bucket | Range | Description |
|--------|-------|-------------|
| 50-55 | 0.50–0.55 | Near coin flip |
| 55-60 | 0.55–0.60 | Slight favorite |
| 60-65 | 0.60–0.65 | Moderate favorite |
| 65-70 | 0.65–0.70 | Solid favorite |
| 70-80 | 0.70–0.80 | Strong favorite |
| 80+ | 0.80+ | Heavy favorite (rare in NFL) |

## 2025 Holdout Performance

| Metric | Value |
|--------|-------|
| Games | 276 |
| Log loss | **0.6200** |
| Brier | 0.2157 |
| AUC | 0.7098 |
| Accuracy | 0.6630 |

## 2026 Predictions

See [2026 Schedule & Predictions](2026-schedule) for the current season's
predictions. Updated each week via:

```
sportslab predict-week --season 2026 --week <N> --auto-qb
```

---

*Full registry: [`leaderboard.csv`](https://github.com/SecuritahGuy/sports-ml-lab/blob/main//Users/tim/dev/sports-ml-lab/reports/benchmarks/leaderboard.csv)*
