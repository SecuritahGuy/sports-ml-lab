# Injury Features Experiment

*Testing whether pregame injury report features improve on O/D Elo+Platt incumbent.*

## Method

Rolling-origin 3-fold validation, one-shot 2025 holdout.

### Competing Models

| Model | Description |
|------|------------|
| **Platt (incumbent)** | O/D Elo (ko52_kd20) + logistic calibration |
| **Elo + Injury** | O/D Elo features + injury features + logistic regression |
| **Injury only** | Injury features only + logistic regression |

### Injury Features

- home_injuries_out
- away_injuries_out
- injuries_out_diff
- home_injuries_qb_out
- away_injuries_qb_out
- injuries_qb_out_diff
- home_injuries_skill_out
- away_injuries_skill_out
- injuries_skill_out_diff
- home_injuries_ol_out
- away_injuries_ol_out
- injuries_ol_out_diff
- home_injuries_def_out
- away_injuries_def_out
- injuries_def_out_diff
- home_injuries_questionable
- away_injuries_questionable
- home_injuries_doubtful
- away_injuries_doubtful

## Rolling-Origin Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| Platt (incumbent) | 0.6376 | 0.6430 | 0.6567 | 0.6132 |
| Elo + Injury | 0.6433 | 0.6558 | 0.6564 | 0.6176 |
| Injury only | 0.6894 | 0.7046 | 0.6792 | 0.6844 |

## 2025 Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|-----|
| Platt (incumbent) | 0.6276 | 0.2187 | 0.7066 | 0.6739 |
| Elo + Injury | 0.6352 | 0.2200 | 0.7008 | 0.6522 |
| Injury only | 0.6922 | 0.2481 | 0.5544 | 0.5616 |

### QB Injury Subset (Elo + Injury)

| Subset | N | Log Loss |
|--------|---|---------|
| QB Out | 28 | 0.5506 |
| QB healthy | 248 | 0.6448 |

**Incumbent retains champion.** Best challenger elo_inj holdout LL 0.6352 vs incumbent 0.6276

