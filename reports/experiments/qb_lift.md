# QB Lift Experiment

*Model: v3.0.0 + QB Lift*

## Research Question

Do rolling QB efficiency metrics (EPA/dropback, CPOE) improve on the incumbent, especially for QB-change games?

## Governance Trigger

QB Lift is a **new pregame-safe data source** derived from play-by-play quarterback efficiency data. It is not a retest of any rejected feature family. QB depth features (career starts, win pct) tested previously are unrelated to rolling PBP-derived efficiency metrics.

## Methods

| Variant | Features |
|---------|----------|
| baseline | v3.0.0 unchanged |
| qb_lift_3 | home_qb_epa_3, away_qb_epa_3, net_qb_epa_3 |
| qb_lift_5 | home_qb_epa_5, away_qb_epa_5, net_qb_epa_5 |
| qb_lift_all | home_qb_epa_3, away_qb_epa_3, net_qb_epa_3, home_qb_epa_5, away_qb_epa_5, net_qb_epa_5 |
| qb_lift_cpoe | home_qb_cpoe_3, away_qb_cpoe_3 |

## Validation (Rolling-Origin, 3 folds)

| Variant | Avg Val LL | ECE | MCE |
|---------|-----------|-----|-----|
| baseline | 0.6305 | 0.0595 | 0.2274  ← **SELECTED**|
| qb_lift_cpoe | 0.6314 | 0.0586 | 0.2262 |
| qb_lift_5 | 0.6317 | 0.0678 | 0.2025 |
| qb_lift_3 | 0.6331 | 0.0583 | 0.4121 |
| qb_lift_all | 0.6335 | 0.0758 | 0.4166 |

### Fold Details (Best)

**baseline**

| Fold | Val N | LL | ECE | MCE |
|------|-------|-----|-----|-----|
| 2022 | 275 | 0.6360 | 0.0663 | 0.1938 |
| 2023 | 279 | 0.6596 | 0.0602 | 0.3210 |
| 2024 | 278 | 0.5960 | 0.0521 | 0.1675 |

### Fold Details (Baseline)

| Fold | Val N | LL | ECE | MCE |
|------|-------|-----|-----|-----|
| 2022 | 275 | 0.6360 | 0.0663 | 0.1938 |
| 2023 | 279 | 0.6596 | 0.0602 | 0.3210 |
| 2024 | 278 | 0.5960 | 0.0521 | 0.1675 |

## Holdout (2025)

| Metric | Baseline | Selected |
|--------|----------|----------|
| Log loss | 0.6200 | 0.6200 |
| Brier | 0.2157 | 0.2157 |
| AUC | 0.7098 | 0.7098 |
| Accuracy | 0.6630 | 0.6630 |
| ECE | 0.0628 | 0.0628 |
| MCE | 0.1343 | 0.1343 |
| N | 276 | 276 |

### QB-Change Subset

| Metric | Baseline | Selected |
|--------|----------|----------|
| N | 55 | 55 |
| ECE | 0.2097 | 0.2097 |
| MCE | 0.5690 | 0.5690 |

## Leakage Risk

- QB Lift uses only prior-game data (rolling window, no future).
- Minimum 10 dropbacks filters out non-QB trick plays.
- No 2025 holdout data accessed during fold validation.
- No market features used.
- No new feature families from the rejected list.

## Decision

**❌ REJECTED** — no variant beats baseline on both validation and holdout. Val LL: 0.6305 vs 0.6305. Holdout LL: 0.6200 vs 0.6200.

