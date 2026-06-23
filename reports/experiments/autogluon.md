# AutoGluon AutoML Experiment

*Testing whether AutoGluon (with all pregame features) beats O/D Elo+Platt incumbent.*

## Method

Rolling-origin 3-fold validation, one-shot 2025 holdout.

### Competing Models

| Model | Description |
|------|------------|
| **Platt (incumbent)** | O/D Elo (ko52_kd20) + logistic calibration |
| **AutoGluon (full)** | All pregame features + AutoGluon medium_quality |
| **AutoGluon (Elo only)** | 6 O/D Elo features only + AutoGluon |
| **AutoGluon + Platt** | AutoGluon outputs recalibrated with Platt |

### AutoGluon Configuration

| Setting | Value |
|--------|-------|
| presets | medium_quality |
| eval_metric | log_loss |
| problem_type | binary |
| time_limit_per_fold | 1800s |
| total_folds | 3 |

### Feature Set (47 features)

- **O/D Elo ratings** (6): home_off_elo, away_off_elo, home_def_elo, away_def_elo, elo_diff, elo_prob
- **Scheduling** (9): home_short_week, away_short_week, home_off_bye, away_off_bye, thursday_flag, monday_flag, is_international, home_consecutive_road, away_consecutive_road
- **QB flags** (19): home_qb_changed, away_qb_changed, qb_change_diff, home_qb_starts_this_season_pre, away_qb_starts_this_season_pre, qb_starts_diff, home_qb_team_starts_pre, away_qb_team_starts_pre, home_qb_win_pct_pre, away_qb_win_pct_pre, qb_win_pct_diff, home_games_since_qb_change, away_games_since_qb_change, games_since_qb_change_diff, home_new_qb_flag, away_new_qb_flag, new_qb_diff, home_qb_missing_flag, away_qb_missing_flag
- **Basic context** (13): week, rest_diff, div_game, is_dome, is_neutral, game_type_enc, roof_enc, surface_enc, weekday_enc, home_team_enc, away_team_enc, home_coach_enc, away_coach_enc

## Rolling-Origin Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| Platt (incumbent) | 0.6376 | 0.6430 | 0.6567 | 0.6132 |
| AutoGluon (full) | 0.6595 | 0.6513 | 0.6901 | 0.6371 |
| AutoGluon (Elo only) | 0.6849 | 0.7387 | 0.6741 | 0.6418 |

## 2025 Holdout

| Model | Hold LL | Brier | AUC | Acc |
|-------|---------|-------|-----|-----|
| Platt (incumbent) | 0.6362 | 0.2226 | 0.6904 | 0.6667 |
| AutoGluon (full) | 0.6439 | 0.2251 | 0.6839 | 0.6449 |
| AutoGluon (Elo only) | 0.6748 | 0.2376 | 0.6478 | 0.6268 |
| AutoGluon (full) + Platt | 0.7488 | 0.2499 | 0.6839 | 0.6413 |
| AutoGluon (Elo only) + Platt | 0.7599 | 0.2599 | 0.6478 | 0.6232 |

**Incumbent retains champion.** Best challenger platt holdout LL 0.6362 vs incumbent 0.6362

AutoGluon did not improve on simple Platt calibration on this dataset.
