# NN Architecture Exploration

Tests genuinely-different NN architectures (ResNet-tabular, RealMLP seed ensemble, deeper/wider MLPs) vs the v3.1.0 MLP. Baseline = v3.1.0 (val 0.6291, holdout 0.6155).

| Config | Arch | Val LL | Δ Val | Holdout LL | Δ Holdout |
|---------|------|--------|-------|-----------|-----------|
| v3.1.0_mlp | mlp | 0.6291 | +0.0000 | 0.6155 | +0.0000 |
| resnet_3x64 | resnet | 0.6422 | +0.0131 | 0.6331 | +0.0176 |
| resnet_4x128 | resnet | 0.6501 | +0.0210 | 0.6371 | +0.0216 |
| resnet_gelu_3x64 | resnet | 0.6417 | +0.0126 | 0.6314 | +0.0159 |
| realmlp_ensemble5 | mlp | 0.6308 | +0.0017 | 0.6195 | +0.0040 |
| realmlp_ensemble7 | mlp | 0.6295 | +0.0004 | 0.6187 | +0.0032 |
| mlp_4x32 | mlp | 0.6294 | +0.0003 | 0.6204 | +0.0049 |
| mlp_5x64 | mlp | 0.6312 | +0.0021 | 0.6183 | +0.0029 |

## Decision

**❌ REJECTED** — no architecture beats the v3.1.0 MLP on both val and holdout by >= 0.001.

Best validation: v3.1.0_mlp (0.6291, Δ=+0.0000)
Best holdout: v3.1.0_mlp (0.6155, Δ=+0.0000)

## Interpretation

- ResNet-tabular: tests skip-connections + LayerNorm vs plain MLP.
- RealMLP ensemble: 5-7 seed ensemble with kaiming init + cosine LR.
- Deeper/wider MLPs: capacity test (overfit risk at ~7k rows).
