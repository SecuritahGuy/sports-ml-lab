# Dynamic Bayesian Elo

State-space model with MLE-estimated variance parameters.

**Model:**
- State: latent team strength theta (Nx1) on margin-of-victory scale
- Transition: theta_k = theta_{k-1} + epsilon
- Observation: y_k = theta_k[home] - theta_k[away] + HFA + v_k
- Estimation: Maximum likelihood via prediction-error decomposition
- Pre-game margin predicted from filtered state -> Platt-calibrated

| Variant | Val LL | Hold LL |
|---------|--------|--------|
| Incumbent | 0.6303 | 0.6218 |
| Dynamic Elo | 0.6342 (Delta=+0.0039) | 0.6229 (Delta=+0.0011) |

## Per-Fold Parameters

| Fold | sigma_evolution | sigma_observation | HFA |
|------|----------------|-------------------|-----|
| Fold 2022 | 0.002 | 13.139 | 1.670 |
| Fold 2023 | 0.283 | 12.038 | 1.938 |
| Fold 2024 | 0.225 | 12.409 | 2.220 |
| Holdout (2025) | 0.237 | 12.372 | 2.138 |

## Platt Metrics

| Model | Val LL | Fold1 | Fold2 | Fold3 | Hold LL | Brier | AUC | Acc |
|-------|--------|-------|-------|-------|---------|-------|-----|-----|
| Incumbent | 0.6303 | 0.6357 | 0.6615 | 0.5937 | 0.6218 | 0.2164 | 0.708 | 0.652 |
| Dynamic Elo | 0.6342 | 0.6453 | 0.6488 | 0.6084 | 0.6229 | 0.2168 | 0.707 | 0.652 |

## ❌ NOT PROMOTED — Dval=+0.0039, Dhold=+0.0011

---
Report: dynamic_elo_experiment.py
