# Predictions

## Latest Prediction Artifacts

| Artifact | Description | Status |
|----------|-------------|--------|
| [`incumbent_predictions.csv`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions.csv) | All eligible games | OK |
| [`incumbent_predictions_2025_holdout.csv`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions_2025_holdout.csv) | 2025 holdout | OK |
| [`weekly_report.md`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/weekly_report.md) | Weekly report | OK |
| [`incumbent_prediction_cards.md`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_prediction_cards.md) | Game cards | OK |

## Prediction Schema

Each prediction CSV contains:

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `season`, `week`, `gameday` | Game timing |
| `away_team`, `home_team` | Teams |
| `home_win_actual` | Actual result (1 = home win) |
| `incumbent_home_win_prob` | Predicted home win probability |
| `predicted_winner` | Home or away team based on prob |
| `confidence_bucket` | Probability range bucket |
| `model_version` | Incumbent version at prediction time |
| `feature_set` | Features used |
| `calibration_method` | Calibration type |
| `caution_qb_change` | QB changed from prior game |
| `caution_neutral` | Near-50% prediction |
| `caution_early_season` | Weeks 1–4 |
| `caution_missing_features` | Some features imputed |
| `caution_model_market_disagreement` | Model vs market gap > 0.15 |
| `market_prob_diagnostic` | Market-implied prob (diagnostic) |
| `market_minus_model_diagnostic` | Market prob minus model prob |

## Confidence Buckets

| Bucket | Range | Description |
|--------|-------|-------------|
| 50-55 | 0.50–0.55 | Near coin flip |
| 55-60 | 0.55–0.60 | Slight favorite |
| 60-65 | 0.60–0.65 | Moderate favorite |
| 65-70 | 0.65–0.70 | Solid favorite |
| 70-80 | 0.70–0.80 | Strong favorite |
| 80+ | 0.80+ | Heavy favorite (rare in NFL) |

## Caution Flags

| Flag | Meaning |
|------|---------|
| QB change | QB did not start prior game |
| Neutral | Near 50% probability |
| Early season | Weeks 1–4 |
| Missing features | Imputed input data |
| Model-market disagreement | Gap > 0.15 in probability |

## Market Fields (Diagnostic Only)

The `market_prob_diagnostic` and `market_minus_model_diagnostic`
columns are for research comparison only. Market data comes from
closing moneyline odds and is NOT used in model training.

## Holdout Performance

The 2025 holdout contains **276 games** (regular season and playoffs).
The current incumbent achieves log loss **0.6262** on this set.

*Holdout file: [`https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions_2025_holdout.csv`](https://github.com/timdev/sports-ml-lab/blob/main/reports/predictions/incumbent_predictions_2025_holdout.csv)*
