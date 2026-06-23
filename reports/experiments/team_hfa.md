# Team-Specific HFA Experiment

*Testing per-team home field advantages vs global HFA.*

## Method

For each team, compute home margin advantage over away margin:
```
HFA_offset = mean(home_margin) - mean(away_margin)
→ scaled to Elo units (1 pt ≈ 25 Elo, capped at ±30)
effective_HFA = global_HFA + team_HFA_offset
```

Global HFA: 40

## Rolling-Origin Validation

| Model | Avg Raw LL | Avg Platt LL |
|-------|-----------|-------------|
| Global HFA | 0.6321 | 0.6371 |
| Team HFA | 0.6355 | 0.6390 |

## 2025 Holdout

| Model | Raw LL | Brier | Acc | AUC |
|-------|--------|-------|-----|-----|
| Global HFA raw | 0.6301 | 0.2201 | 0.6630 | 0.7024 |
| Global HFA + Platt | 0.6298 | 0.2197 | 0.6558 | 0.7024 |
| Team HFA raw | 0.6267 | 0.2184 | 0.6848 | 0.7063 |
| Team HFA + Platt | 0.6263 | 0.2180 | 0.6812 | 0.7063 |

## Decision

❌ **Team-specific HFA rejected.** Worse average validation log loss (0.6355 vs 0.6321).

Holdout was better (0.6263 vs 0.6298) but the project rule is to select by validation. Per-team HFA estimates from only 1–3 seasons are noisy; more data might help.

## Leakage Prevention

- Per-team HFA computed from training seasons only.
- No validation or holdout data used in HFA estimation.
- Rolling-origin folds enforce temporal split.

## Sample Team HFA Values

| Team | Margin Adv | Elo Offset |
|------|-----------|-----------|
| GB | 1.26 | 30.0 |
| MIA | 7.07 | 30.0 |
| JAX | 1.35 | 30.0 |
| CLE | 3.75 | 30.0 |
| PIT | 1.52 | 30.0 |
| DET | 2.54 | 30.0 |
| CAR | 1.90 | 30.0 |
| ATL | 2.05 | 30.0 |
| CHI | 4.96 | 30.0 |
| DEN | 5.54 | 30.0 |

