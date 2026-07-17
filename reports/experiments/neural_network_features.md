# Neural Network + Expanded Features

Tests whether the MLP calibrator can exploit pregame-safe features that were rejected for the logistic model.

| Feature set | Val LL | Δ Val | Holdout LL | Δ Holdout |
|-------------|--------|-------|-----------|-----------|
| incumbent_only | 0.6279 | +0.0000 | 0.6151 | +0.0000 |
| +rest_div_dome | 0.6335 | +0.0056 | 0.6189 | +0.0038 |
| +week_roof_surface | 0.6335 | +0.0056 | 0.6185 | +0.0034 |
| +mov5 | 0.6326 | +0.0047 | 0.6112 | -0.0039 |
| +all_extra | 0.6356 | +0.0077 | 0.6221 | +0.0070 |

## Decision

**❌ REJECTED** — no expanded feature set beat the MLP incumbent on both validation and holdout by >= 0.001.

Best validation: +mov5 (0.6326, Δ=+0.0047)
Best holdout: +mov5 (0.6112, Δ=-0.0039)
