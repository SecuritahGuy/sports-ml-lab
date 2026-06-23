# Team Stats Features Experiment

*Testing whether rolling team stat aggregates (yards, fantasy pts, sacks) improve on the incumbent.*

## Data Source

| Source | Description | Coverage |
|--------|-------------|----------|
| nflreadpy.load_player_stats | Weekly player stats aggregated to team level | 2021–2025 |

## Features

| Feature | Windows | Description |
|---------|---------|-------------|
| off_yds | 3, 5 | Team total offensive yards (pass+rush) |
| def_yds_allowed | 3, 5 | Opponent offensive yards |
| fantasy_pts | 3, 5 | Total fantasy points (all players) |
| def_sacks | 3, 5 | Defensive sacks |
| off_yds_net | 3, 5 | Home offense − away defense |

Total feature columns: 20

## Incumbent Params

K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2

## Results

### Rolling-Origin Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Platt (incumbent) | 0.6368 | 0.6425 | 0.6576 | 0.6103 |
| Team stats only | 0.6831 | 0.6841 | 0.6901 | 0.6750 |
| Elo + Team Stats | 0.6541 | 0.6638 | 0.6746 | 0.6240 |

### 2025 Holdout

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6285 | 0.2191 | 0.6667 | 0.7024 |
| Team stats only | 0.6674 | 0.2375 | 0.5797 | 0.6238 |
| Elo + Team Stats | 0.6415 | 0.2260 | 0.6341 | 0.6730 |

### QB-Change Subset (Platt)

QB changed (n=24): 0.7722 | QB stable (n=252): 0.6149

**Incumbent remains champion.** No team-stat model beat it.
