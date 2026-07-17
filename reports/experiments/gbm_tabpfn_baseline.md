# LightGBM + TabPFN Baseline

Confirms whether a strong GBM or foundation-model tabular classifier beats the v3.1.0 MLP (baseline val 0.6279, holdout 0.6155).

| Model | Val LL | Δ Val | Holdout LL | Δ Holdout |
|-------|--------|-------|-----------|-----------|
| v3.1.0_mlp | 0.6279 | +0.0000 | 0.6155 | +0.0000 |
| lgbm_default | 0.6381 | +0.0102 | 0.6260 | +0.0106 |
| lgbm_deep | 0.6398 | +0.0119 | 0.6327 | +0.0172 |
| tabpfn | BLOCKED | — | BLOCKED | — |

## Decision

**❌ REJECTED** — no runnable model beats the MLP on both val and holdout.

Best validation: v3.1.0_mlp (0.6279, Δ=+0.0000)
Best holdout: v3.1.0_mlp (0.6155, Δ=+0.0000)

## Interpretation

- If LightGBM ≈ MLP: the MLP gain is real (not just 'logistic was weak').
- If LightGBM > MLP: trees are the better calibrator here; promote GBM.
- TabPFN v8 requires a PriorLabs API token (not available offline) — diagnostic blocked in this environment.
