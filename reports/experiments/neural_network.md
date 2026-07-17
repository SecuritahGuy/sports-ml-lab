# Neural Network Challenger

Small feed-forward MLP (PyTorch) replacing the logistic Platt calibration, with the same features + frozen QB overlay.

| Model | Val LL | Δ Val | Holdout LL | Δ Holdout | Holdout ECE |
|-------|--------|-------|-----------|-----------|-------------|
| incumbent (Logistic Platt (champion)) | 0.6308 | +0.0000 | 0.6226 | +0.0000 | 0.0506 |
| mlp_16 (MLP 1 hidden (16) dp=0.1) | 0.6306 | -0.0002 | 0.6179 | -0.0047 | 0.0329 |
| mlp_32_16 (MLP 2 hidden (32,16) dp=0.1) | 0.6296 | -0.0012 | 0.6162 | -0.0064 | 0.0359 |
| mlp_64 (MLP 1 hidden (64) dp=0.2) | 0.6297 | -0.0011 | 0.6179 | -0.0047 | 0.0421 |
| mlp_32_16_wd (MLP 2 hidden (32,16) dp=0.2 wd=1e-3) | 0.6280 | -0.0028 | 0.6159 | -0.0067 | 0.0471 |
| mlp_16_16_16 (MLP 3 hidden (16,16,16) dp=0.1 wd=1e-4) | 0.6279 | -0.0029 | 0.6151 | -0.0075 | 0.0331 |
| mlp_64_32 (MLP 2 hidden (64,32) dp=0.2 wd=1e-3) | 0.6293 | -0.0015 | 0.6172 | -0.0054 | 0.0455 |

## Decision

**✅ PROMOTED** — a neural variant beat the incumbent on both validation (mlp_16_16_16) and holdout (mlp_16_16_16).

