# Injury Features Experiment

*Adding pregame injury report features on top of MOV Elo+Platt.*

## Method

Rolling-origin 3-fold validation, one-shot 2025 holdout.

### Incumbent Params

| Parameter | Value |
|-----------|-------|
| K-factor | 36 |
| Home-field advantage | 40 |
| Preseason regression | 0.1 |
| Decay | 32 |
| QB-change bonus | 0.2 |
| MOV type | capped_linear, scale=0.05, cap=2.0 |
| Calibration | Platt scaling |

### Injury Features

| Feature | Description |
|---------|-------------|
| `home/away_qb_out` | Count of QBs on team ruled OUT |
| `home/away_qb_doubtful_or_out` | Count of QBs OUT or Doubtful |
| `home/away_total_out` | Count of all players ruled OUT |
| `home/away_total_doubtful_or_out` | Count of all players OUT or Doubtful |
| `home/away_skill_out` | Count of QB+RB+WR+TE OUT |
| `home/away_ol_out` | Count of OL (C+G+T) OUT |
| `home/away_def_out` | Count of defensive players OUT |
| `any_qb_out` | Either team has a QB OUT |
| `net_injuries` | home_total_out − away_total_out |
| `net_skill_out` | home_skill_out − away_skill_out |
| `net_def_out` | home_def_out − away_def_out |
| `home/away_qb_injury_change` | QB changed AND old starter was OUT |

### Injury Data Source

nflreadpy `load_injuries()` — official NFL injury reports, weekly pregame.

### Models Compared

| Model | Description |
|-------|-------------|
| **Platt (incumbent)** | MOV Elo + Platt scaling |
| **Elo + Injury** | Logistic on Elo prob + all injury features |
| **Injury only** | Logistic on injury features alone |
| **Elo + QB injury** | Logistic on Elo prob + QB-specific injury flags |

### Data Split

| Split | Seasons |
|-------|---------|
| Fold 1 | Train: [2021], Val: 2022 |
| Fold 2 | Train: [2021, 2022], Val: 2023 |
| Fold 3 | Train: [2021, 2022, 2023], Val: 2024 |
| Holdout | 2025 |

## Rolling-Origin Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Platt (incumbent) | 0.6406 | 0.6471 | 0.6621 | 0.6126 |
| Elo + Injury | 0.6486 | 0.6711 | 0.6627 | 0.6121 |
| Injury only | 0.6964 | 0.7219 | 0.6828 | 0.6846 |
| Elo + QB injury flags | 0.6428 | 0.6591 | 0.6596 | 0.6096 |

## 2025 Holdout

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Platt (incumbent) | 0.6315 | 0.2204 | 0.6739 | 0.6983 |
| Elo + Injury | 0.6514 | 0.2257 | 0.6486 | 0.6876 |
| Injury only | 0.7034 | 0.2521 | 0.5362 | 0.5413 |
| Elo + QB injury flags | 0.6485 | 0.2255 | 0.6377 | 0.6862 |

## Subset Analysis (2025 Holdout)

| Subset | N | Raw Elo LL |
|--------|---|------------|
| QB-change games (home) | 24 | 0.7553 |
| QB-stable games (home) | 252 | 0.6230 |
| Injury-driven QB change | 11 | 0.7041 |
| Any QB OUT | 48 | 0.6043 |
| No QB OUT | 228 | 0.6409 |

## Platt (Incumbent, Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 4 | 0.1780 | 0.2500 | 0.0720 |
| [0.2, 0.3) | 23 | 0.2621 | 0.3478 | 0.0857 |
| [0.3, 0.4) | 27 | 0.3481 | 0.2963 | 0.0518 |
| [0.4, 0.5) | 52 | 0.4421 | 0.3269 | 0.1151 |
| [0.5, 0.6) | 63 | 0.5483 | 0.6032 | 0.0549 |
| [0.6, 0.7) | 49 | 0.6467 | 0.6327 | 0.0140 |
| [0.7, 0.8) | 41 | 0.7470 | 0.7317 | 0.0153 |
| [0.8, 0.9) | 17 | 0.8227 | 0.8824 | 0.0597 |

## Elo + Injury (Holdout)

| Bucket | Count | Mean Pred | Mean Actual | Cal Error |
|--------|-------|-----------|-------------|-----------|
| [0.1, 0.2) | 6 | 0.1797 | 0.3333 | 0.1536 |
| [0.2, 0.3) | 25 | 0.2521 | 0.3200 | 0.0679 |
| [0.3, 0.4) | 29 | 0.3520 | 0.2414 | 0.1107 |
| [0.4, 0.5) | 51 | 0.4548 | 0.4510 | 0.0038 |
| [0.5, 0.6) | 54 | 0.5523 | 0.5370 | 0.0153 |
| [0.6, 0.7) | 45 | 0.6444 | 0.6444 | 0.0000 |
| [0.7, 0.8) | 41 | 0.7476 | 0.8049 | 0.0573 |
| [0.8, 0.9) | 16 | 0.8404 | 0.7500 | 0.0904 |
| [0.9, 1.0) | 9 | 0.9508 | 0.5556 | 0.3952 |

## Decision

⚠️ **Incumbent (MOV Elo + Platt) remains the research incumbent.**

No injury-augmented model beat the incumbent on both validation and holdout.

- Elo + QB injury flags: val LL=0.6428 (trails incumbent 0.6406), hold LL=0.6485 (trails incumbent 0.6315)
- Elo + Injury: val LL=0.6486 (trails incumbent 0.6406), hold LL=0.6514 (trails incumbent 0.6315)
- Injury only: val LL=0.6964 (trails incumbent 0.6406), hold LL=0.7034 (trails incumbent 0.6315)

### Key Conclusions

1. Injury features from nflreadpy official reports were tested.
2. QB injury flags alone (subset) were also tested separately.
3. Injury-driven QB change detection added. Distinguishes injury-forced changes from coaching decisions.
4. The QB-change failure mode is partly an injury signal — if injury data improves performance, the gap narrows.
