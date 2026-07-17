# PyTorch Deep-Dive

Systematic PyTorch investigation: featureization, LR schedule/optimizer, weight-decay, ensembles. Baseline = exact v3.1.0 MLP protocol reproduced here (val 0.6291, holdout 0.6155). Literature v3.1.0 = val 0.6279, holdout 0.6151.

| Config | Features | Optimizer | Schedule | Val LL | Δ Val | Holdout LL | Δ Holdout |
|---------|----------|-----------|----------|--------|-------|-----------|-----------|
| v3.1.0_exact | incumbent | adam | none | 0.6291 | +0.0000 | 0.6155 | +0.0000 |
| mlp_cosine_adamw | incumbent | adamw | cosine | 0.6288 | -0.0003 | 0.6160 | +0.0006 |
| elo_rich_cosine | elo_rich | adamw | cosine | 0.6282 | -0.0009 | 0.6167 | +0.0013 |
| antisym_cosine | antisymmetric | adamw | cosine | 0.6318 | +0.0027 | 0.6186 | +0.0031 |
| elo_rich_wd1e3 | elo_rich | adamw | cosine | 0.6291 | +0.0000 | 0.6197 | +0.0042 |
| elo_rich_wd1e2 | elo_rich | adamw | cosine | 0.6286 | -0.0005 | 0.6208 | +0.0053 |
| elo_rich_gelu | elo_rich | adamw | cosine | 0.6301 | +0.0010 | 0.6172 | +0.0017 |
| elo_rich_onecycle | elo_rich | adamw | one_cycle | 0.6286 | -0.0005 | 0.6209 | +0.0054 |
| elo_rich_ensemble5 | elo_rich | adamw | cosine | 0.6301 | +0.0010 | 0.6177 | +0.0022 |

## Decision

**❌ REJECTED** — no PyTorch config beat the v3.1.0 baseline on both validation and holdout by >= 0.001.

Best validation: elo_rich_cosine (0.6282, Δ=-0.0009)
Best holdout: v3.1.0_exact (0.6155, Δ=+0.0000)

## Key findings

- The v3.1.0 MLP (Adam, constant LR, full-batch, StandardScaler, default init, 3x16, wd=1e-4, 200 epochs) is already near-optimal for this dataset.
- Cosine LR + AdamW: marginal, slightly WORSE on holdout (+0.0006). No benefit over constant LR.
- Richer feature sets (elo_rich, antisymmetric) hurt on holdout — the incumbent 5-feature set is optimal.
- Weight-decay sweep (1e-3, 1e-2), GELU, OneCycle, and deep ensembles all degrade holdout vs baseline.
- Methodology note: an earlier run showed a spurious 0.609 holdout "win" caused by an internal 80/20 train/val split silently dropping 20% of training data in the early-stopping path; fixed by training on the full set when early_stopping=False. After the fix no variant beats the baseline.
