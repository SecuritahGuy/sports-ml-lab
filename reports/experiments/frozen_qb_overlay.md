# Frozen-Incumbent QB Overlay Experiment — V2

## Research Question

Can a QB adjustment improve only the targeted QB-change / low-continuity slice while leaving all stable-QB games exactly identical to the incumbent?

## Why V1 Was Rejected

The gated QB-adjusted Elo experiment (V1) used Platt recalibration **after** applying the gate. This changed ALL predictions—including non-gated games—because the Platt logistic regression fit shifted with different input probabilities. Even "qb_changed_only" gating degraded non-QB-change games (Δ = +0.0075 on holdout).

Key finding: the calibration step was the primary source of non-QB-change degradation, not the QB adjustment itself.

## Frozen-Incumbent Overlay Design

The incumbent model's Platt-calibrated probability is the frozen base. A QB overlay is applied in logit space ONLY when a pregame gate is active:

```
base_logit = log(incumbent_prob / (1 - incumbent_prob))
if gate_on:
    final_logit = base_logit + gamma * (home_qb_adj - away_qb_adj) * ln(10) / 400
else:
    final_logit = base_logit
final_prob = sigmoid(final_logit)
```

**Critical property**: When `gate_on` is False, `final_prob` equals `incumbent_prob` exactly (within floating-point tolerance). No recalibration is performed after gating.

## Gates Tested

| Letter | Gate Condition | Description |
|--------|----------------|-------------|
| A | (none) | Incumbent baseline, no overlay |
| B | qb_changed | Apply overlay when either QB changed |
| C | qb_differs_prev | Same as B |
| D | starts<4 | Apply when either QB has <4 team starts |
| E | starts<8 | Apply when either QB has <8 team starts |
| F | starts<17 | Apply when either QB has <17 team starts |
| G | changed OR starts<8 | Union of B and E |
| H | changed OR starts<17 | Union of B and F |
| I | DIAG aggressive | Gamma=2.0, diagnostic only |

## Parameter Sweep

| Parameter | Values Tested |
|-----------|---------------|
| gamma | 0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00 |
| cap (Elo pts) | None, 20, 40, 60 |
| Total combos | 8 gates × 6 gammas × 4 caps = 192 + 1 baseline |

## Data Used

- 2021–2025 NFL seasons (non-neutral regular + postseason)
- Team Elo ratings (K=36, HFA=40, reg=0.1, decay=32, MOV capped_linear)
- Incumbent feature set: elo_prob + qb_changed (2) + rolling_mov_3 (2) + Platt
- QB adjustments computed from prior starts with Bayesian shrinkage
- No recalibration after overlay

## Non-Gated Equality

Non-gated games on holdout: 77

**Max absolute diff vs incumbent:** 1.11e-16
**Mean absolute diff vs incumbent:** 1.08e-17
**Equality check: PASSED**

Non-gated games are identical to the incumbent within floating-point tolerance.

## Rolling-Origin Validation Log Loss (Top 20)

| Model | Avg LL | Fold1 | Fold2 | Fold3 |
|-------|--------|-------|-------|-------|
| H. changed OR starts<17 cap=20 | 0.6238 | 0.6269 | 0.6529 | 0.5915 |
| F. starts<17 cap=20 | 0.6240 | 0.6268 | 0.6532 | 0.5919 |
| H. changed OR starts<17 g=0.75 cap=20 | 0.6241 | 0.6273 | 0.6527 | 0.5922 |
| H. changed OR starts<17 g=0.50 cap=60 | 0.6242 | 0.6282 | 0.6536 | 0.5908 |
| H. changed OR starts<17 g=0.75 cap=40 | 0.6242 | 0.6274 | 0.6543 | 0.5909 |
| F. starts<17 g=0.75 cap=20 | 0.6242 | 0.6272 | 0.6530 | 0.5925 |
| H. changed OR starts<17 g=0.50 cap=40 | 0.6243 | 0.6276 | 0.6534 | 0.5919 |
| H. changed OR starts<17 g=0.75 cap=60 | 0.6243 | 0.6285 | 0.6549 | 0.5895 |
| H. changed OR starts<17 g=0.35 cap=60 | 0.6245 | 0.6283 | 0.6532 | 0.5919 |
| H. changed OR starts<17 cap=40 | 0.6245 | 0.6275 | 0.6556 | 0.5903 |
| H. changed OR starts<17 g=0.50 cap=20 | 0.6245 | 0.6278 | 0.6526 | 0.5931 |
| H. changed OR starts<17 g=0.50 | 0.6246 | 0.6286 | 0.6543 | 0.5908 |
| F. starts<17 g=0.50 cap=20 | 0.6246 | 0.6277 | 0.6528 | 0.5933 |
| F. starts<17 g=0.75 cap=40 | 0.6246 | 0.6274 | 0.6549 | 0.5915 |
| H. changed OR starts<17 g=0.35 cap=40 | 0.6246 | 0.6279 | 0.6531 | 0.5928 |
| F. starts<17 g=0.50 cap=60 | 0.6246 | 0.6284 | 0.6541 | 0.5914 |
| G. changed OR starts<8 cap=20 | 0.6246 | 0.6263 | 0.6536 | 0.5940 |
| H. changed OR starts<17 g=0.35 | 0.6246 | 0.6285 | 0.6536 | 0.5918 |
| F. starts<17 g=0.50 cap=40 | 0.6247 | 0.6277 | 0.6539 | 0.5924 |
| F. starts<17 g=0.35 cap=60 | 0.6247 | 0.6284 | 0.6535 | 0.5923 |

## 2025 Holdout (Top 20)

| Model | Log Loss | Brier | AUC | Accuracy |
|-------|----------|-------|-----|----------|
| H. changed OR starts<17 cap=40 | 0.6200 | 0.2157 | 0.7098 | 0.6630 |
| F. starts<17 cap=40 | 0.6205 | 0.2159 | 0.7099 | 0.6630 |
| H. changed OR starts<17 g=0.75 cap=40 | 0.6208 | 0.2160 | 0.7096 | 0.6594 |
| G. changed OR starts<8 cap=40 | 0.6211 | 0.2160 | 0.7097 | 0.6667 |
| H. changed OR starts<17 cap=20 | 0.6211 | 0.2161 | 0.7092 | 0.6594 |
| F. starts<17 g=0.75 cap=40 | 0.6212 | 0.2161 | 0.7097 | 0.6594 |
| F. starts<17 cap=20 | 0.6217 | 0.2163 | 0.7088 | 0.6594 |
| G. changed OR starts<8 g=0.75 cap=40 | 0.6218 | 0.2163 | 0.7098 | 0.6630 |
| F. starts<17 g=0.75 cap=60 | 0.6219 | 0.2164 | 0.7088 | 0.6522 |
| H. changed OR starts<17 g=0.75 cap=60 | 0.6219 | 0.2164 | 0.7080 | 0.6522 |
| F. starts<17 cap=60 | 0.6219 | 0.2164 | 0.7079 | 0.6558 |
| E. starts<8 cap=40 | 0.6219 | 0.2163 | 0.7100 | 0.6667 |
| I. DIAG aggressive (starts<8) cap=40 | 0.6219 | 0.2163 | 0.7100 | 0.6667 |
| G. changed OR starts<8 cap=20 | 0.6219 | 0.2164 | 0.7099 | 0.6594 |
| H. changed OR starts<17 cap=60 | 0.6220 | 0.2164 | 0.7076 | 0.6558 |
| F. starts<17 g=0.75 | 0.6220 | 0.2166 | 0.7083 | 0.6558 |
| H. changed OR starts<17 g=0.50 cap=40 | 0.6220 | 0.2165 | 0.7087 | 0.6594 |
| H. changed OR starts<17 g=0.75 cap=20 | 0.6221 | 0.2165 | 0.7084 | 0.6522 |
| H. changed OR starts<17 g=0.75 | 0.6222 | 0.2167 | 0.7075 | 0.6558 |
| F. starts<17 g=0.50 cap=40 | 0.6223 | 0.2166 | 0.7088 | 0.6594 |

## Best Variant by Gate Family

| Gate | Gamma | Cap | Val LL | Hold LL | QC Δ | NoQC Δ |
|------|-------|-----|--------|---------|------|--------|
| B. qb_changed | 1.00 | 20 | 0.6250 | 0.6250 | -0.0046 | +0.0000 |
| D. starts<4 | 1.00 | 40 | 0.6274 | 0.6235 | +0.0041 | -0.0041 |
| E. starts<8 | 1.00 | 40 | 0.6256 | 0.6219 | +0.0020 | -0.0055 |
| F. starts<17 | 1.00 | 40 | 0.6250 | 0.6205 | +0.0004 | -0.0069 |
| G. changed OR starts<8 | 1.00 | 40 | 0.6252 | 0.6211 | -0.0022 | -0.0055 |
| H. changed OR starts<17 | 1.00 | 40 | 0.6245 | 0.6200 | -0.0022 | -0.0069 |

## Best Challenger Slice Performance

Best challenger (holdout): **H. changed OR starts<17 cap=40**

| Slice | N | Incumbent LL | Challenger LL | Δ |
|-------|---|-------------|---------------|---|
| All games | 276 | 0.6259 | 0.6200 | -0.0059 |
| QB change (either) | 55 | 0.6696 | 0.6674 | -0.0022 |
| No QB change | 221 | 0.6151 | 0.6082 | -0.0069 |
| Stable QB (≥4 starts, no change) | 168 | 0.6244 | 0.6207 | -0.0037 |
| Low-continuity (<4 starts) | 86 | 0.6164 | 0.6086 | -0.0078 |
| High confidence (>=0.7) | 64 | 0.5138 | 0.5044 | -0.0094 |

## Full Results Summary

| Model | Val LL | Hold LL | Δ val | Δ hold | QC Δ | NoQC Δ |
|-------|--------|---------|-------|--------|------|--------|
| A. Incumbent baseline | 0.6259 | 0.6259 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| B. qb_changed | 0.6261 | 0.6275 | +0.0002 | +0.0015 | +0.0077 | +0.0000 |
| B. qb_changed cap=20 | 0.6250 | 0.6250 | -0.0009 | -0.0009 | -0.0046 | +0.0000 |
| B. qb_changed cap=40 | 0.6250 | 0.6255 | -0.0009 | -0.0004 | -0.0022 | +0.0000 |
| B. qb_changed cap=60 | 0.6254 | 0.6271 | -0.0005 | +0.0012 | +0.0061 | +0.0000 |
| B. qb_changed g=0.10 | 0.6257 | 0.6260 | -0.0001 | +0.0000 | +0.0002 | +0.0000 |
| B. qb_changed g=0.10 cap=20 | 0.6257 | 0.6258 | -0.0001 | -0.0001 | -0.0006 | +0.0000 |
| B. qb_changed g=0.10 cap=40 | 0.6257 | 0.6258 | -0.0002 | -0.0001 | -0.0006 | +0.0000 |
| B. qb_changed g=0.10 cap=60 | 0.6257 | 0.6260 | -0.0001 | +0.0000 | +0.0001 | +0.0000 |
| B. qb_changed g=0.20 | 0.6257 | 0.6260 | -0.0002 | +0.0001 | +0.0005 | +0.0000 |
| B. qb_changed g=0.20 cap=20 | 0.6256 | 0.6257 | -0.0002 | -0.0002 | -0.0011 | +0.0000 |
| B. qb_changed g=0.20 cap=40 | 0.6256 | 0.6257 | -0.0003 | -0.0002 | -0.0011 | +0.0000 |
| B. qb_changed g=0.20 cap=60 | 0.6256 | 0.6260 | -0.0002 | +0.0001 | +0.0003 | +0.0000 |
| B. qb_changed g=0.35 | 0.6256 | 0.6262 | -0.0003 | +0.0002 | +0.0011 | +0.0000 |
| B. qb_changed g=0.35 cap=20 | 0.6255 | 0.6255 | -0.0004 | -0.0004 | -0.0019 | +0.0000 |
| B. qb_changed g=0.35 cap=40 | 0.6254 | 0.6256 | -0.0005 | -0.0004 | -0.0017 | +0.0000 |
| B. qb_changed g=0.35 cap=60 | 0.6255 | 0.6261 | -0.0004 | +0.0002 | +0.0008 | +0.0000 |
| B. qb_changed g=0.50 | 0.6256 | 0.6264 | -0.0003 | +0.0004 | +0.0021 | +0.0000 |
| B. qb_changed g=0.50 cap=20 | 0.6253 | 0.6254 | -0.0005 | -0.0005 | -0.0027 | +0.0000 |
| B. qb_changed g=0.50 cap=40 | 0.6253 | 0.6255 | -0.0006 | -0.0004 | -0.0022 | +0.0000 |
| B. qb_changed g=0.50 cap=60 | 0.6253 | 0.6263 | -0.0005 | +0.0003 | +0.0016 | +0.0000 |
| B. qb_changed g=0.75 | 0.6257 | 0.6268 | -0.0001 | +0.0009 | +0.0045 | +0.0000 |
| B. qb_changed g=0.75 cap=20 | 0.6251 | 0.6252 | -0.0007 | -0.0007 | -0.0037 | +0.0000 |
| B. qb_changed g=0.75 cap=40 | 0.6251 | 0.6255 | -0.0008 | -0.0005 | -0.0024 | +0.0000 |
| B. qb_changed g=0.75 cap=60 | 0.6253 | 0.6266 | -0.0006 | +0.0007 | +0.0035 | +0.0000 |
| C. qb_differs_prev | 0.6261 | 0.6275 | +0.0002 | +0.0015 | +0.0077 | +0.0000 |
| C. qb_differs_prev cap=20 | 0.6250 | 0.6250 | -0.0009 | -0.0009 | -0.0046 | +0.0000 |
| C. qb_differs_prev cap=40 | 0.6250 | 0.6255 | -0.0009 | -0.0004 | -0.0022 | +0.0000 |
| C. qb_differs_prev cap=60 | 0.6254 | 0.6271 | -0.0005 | +0.0012 | +0.0061 | +0.0000 |
| C. qb_differs_prev g=0.10 | 0.6257 | 0.6260 | -0.0001 | +0.0000 | +0.0002 | +0.0000 |
| C. qb_differs_prev g=0.10 cap=20 | 0.6257 | 0.6258 | -0.0001 | -0.0001 | -0.0006 | +0.0000 |
| C. qb_differs_prev g=0.10 cap=40 | 0.6257 | 0.6258 | -0.0002 | -0.0001 | -0.0006 | +0.0000 |
| C. qb_differs_prev g=0.10 cap=60 | 0.6257 | 0.6260 | -0.0001 | +0.0000 | +0.0001 | +0.0000 |
| C. qb_differs_prev g=0.20 | 0.6257 | 0.6260 | -0.0002 | +0.0001 | +0.0005 | +0.0000 |
| C. qb_differs_prev g=0.20 cap=20 | 0.6256 | 0.6257 | -0.0002 | -0.0002 | -0.0011 | +0.0000 |
| C. qb_differs_prev g=0.20 cap=40 | 0.6256 | 0.6257 | -0.0003 | -0.0002 | -0.0011 | +0.0000 |
| C. qb_differs_prev g=0.20 cap=60 | 0.6256 | 0.6260 | -0.0002 | +0.0001 | +0.0003 | +0.0000 |
| C. qb_differs_prev g=0.35 | 0.6256 | 0.6262 | -0.0003 | +0.0002 | +0.0011 | +0.0000 |
| C. qb_differs_prev g=0.35 cap=20 | 0.6255 | 0.6255 | -0.0004 | -0.0004 | -0.0019 | +0.0000 |
| C. qb_differs_prev g=0.35 cap=40 | 0.6254 | 0.6256 | -0.0005 | -0.0004 | -0.0017 | +0.0000 |
| C. qb_differs_prev g=0.35 cap=60 | 0.6255 | 0.6261 | -0.0004 | +0.0002 | +0.0008 | +0.0000 |
| C. qb_differs_prev g=0.50 | 0.6256 | 0.6264 | -0.0003 | +0.0004 | +0.0021 | +0.0000 |
| C. qb_differs_prev g=0.50 cap=20 | 0.6253 | 0.6254 | -0.0005 | -0.0005 | -0.0027 | +0.0000 |
| C. qb_differs_prev g=0.50 cap=40 | 0.6253 | 0.6255 | -0.0006 | -0.0004 | -0.0022 | +0.0000 |
| C. qb_differs_prev g=0.50 cap=60 | 0.6253 | 0.6263 | -0.0005 | +0.0003 | +0.0016 | +0.0000 |
| C. qb_differs_prev g=0.75 | 0.6257 | 0.6268 | -0.0001 | +0.0009 | +0.0045 | +0.0000 |
| C. qb_differs_prev g=0.75 cap=20 | 0.6251 | 0.6252 | -0.0007 | -0.0007 | -0.0037 | +0.0000 |
| C. qb_differs_prev g=0.75 cap=40 | 0.6251 | 0.6255 | -0.0008 | -0.0005 | -0.0024 | +0.0000 |
| C. qb_differs_prev g=0.75 cap=60 | 0.6253 | 0.6266 | -0.0006 | +0.0007 | +0.0035 | +0.0000 |
| D. starts<4 | 0.6296 | 0.6257 | +0.0037 | -0.0003 | +0.0072 | -0.0021 |
| D. starts<4 cap=20 | 0.6260 | 0.6238 | +0.0002 | -0.0022 | +0.0009 | -0.0030 |
| D. starts<4 cap=40 | 0.6274 | 0.6235 | +0.0015 | -0.0025 | +0.0041 | -0.0041 |
| D. starts<4 cap=60 | 0.6283 | 0.6247 | +0.0025 | -0.0013 | +0.0078 | -0.0035 |
| D. starts<4 g=0.10 | 0.6260 | 0.6256 | +0.0001 | -0.0003 | +0.0004 | -0.0005 |
| D. starts<4 g=0.10 cap=20 | 0.6258 | 0.6257 | -0.0001 | -0.0003 | +0.0000 | -0.0004 |
| D. starts<4 g=0.10 cap=40 | 0.6259 | 0.6256 | +0.0000 | -0.0004 | +0.0002 | -0.0006 |
| D. starts<4 g=0.10 cap=60 | 0.6259 | 0.6256 | +0.0001 | -0.0003 | +0.0006 | -0.0006 |
| D. starts<4 g=0.20 | 0.6262 | 0.6254 | +0.0003 | -0.0005 | +0.0009 | -0.0009 |
| D. starts<4 g=0.20 cap=20 | 0.6258 | 0.6254 | -0.0000 | -0.0005 | +0.0001 | -0.0007 |
| D. starts<4 g=0.20 cap=40 | 0.6259 | 0.6252 | +0.0001 | -0.0007 | +0.0005 | -0.0011 |
| D. starts<4 g=0.20 cap=60 | 0.6260 | 0.6253 | +0.0002 | -0.0006 | +0.0011 | -0.0011 |
| D. starts<4 g=0.35 | 0.6266 | 0.6252 | +0.0007 | -0.0008 | +0.0018 | -0.0014 |
| D. starts<4 g=0.35 cap=20 | 0.6258 | 0.6251 | -0.0001 | -0.0009 | +0.0001 | -0.0012 |
| D. starts<4 g=0.35 cap=40 | 0.6260 | 0.6247 | +0.0002 | -0.0012 | +0.0009 | -0.0018 |
| D. starts<4 g=0.35 cap=60 | 0.6262 | 0.6250 | +0.0004 | -0.0010 | +0.0021 | -0.0018 |
| D. starts<4 g=0.50 | 0.6271 | 0.6251 | +0.0012 | -0.0009 | +0.0028 | -0.0018 |
| D. starts<4 g=0.50 cap=20 | 0.6258 | 0.6247 | -0.0001 | -0.0012 | +0.0003 | -0.0016 |
| D. starts<4 g=0.50 cap=40 | 0.6263 | 0.6243 | +0.0004 | -0.0016 | +0.0015 | -0.0024 |
| D. starts<4 g=0.50 cap=60 | 0.6266 | 0.6247 | +0.0007 | -0.0012 | +0.0032 | -0.0023 |
| D. starts<4 g=0.75 | 0.6282 | 0.6252 | +0.0023 | -0.0007 | +0.0048 | -0.0021 |
| D. starts<4 g=0.75 cap=20 | 0.6259 | 0.6242 | +0.0000 | -0.0017 | +0.0005 | -0.0023 |
| D. starts<4 g=0.75 cap=40 | 0.6267 | 0.6238 | +0.0009 | -0.0021 | +0.0027 | -0.0033 |
| D. starts<4 g=0.75 cap=60 | 0.6273 | 0.6246 | +0.0014 | -0.0014 | +0.0053 | -0.0031 |
| E. starts<8 | 0.6279 | 0.6238 | +0.0020 | -0.0022 | +0.0061 | -0.0043 |
| E. starts<8 cap=20 | 0.6248 | 0.6227 | -0.0011 | -0.0032 | -0.0007 | -0.0039 |
| E. starts<8 cap=40 | 0.6256 | 0.6219 | -0.0002 | -0.0040 | +0.0020 | -0.0055 |
| E. starts<8 cap=60 | 0.6265 | 0.6227 | +0.0007 | -0.0033 | +0.0062 | -0.0056 |
| E. starts<8 g=0.10 | 0.6257 | 0.6253 | -0.0001 | -0.0007 | +0.0001 | -0.0009 |
| E. starts<8 g=0.10 cap=20 | 0.6257 | 0.6255 | -0.0002 | -0.0004 | -0.0001 | -0.0005 |
| E. starts<8 g=0.10 cap=40 | 0.6256 | 0.6253 | -0.0002 | -0.0007 | -0.0001 | -0.0008 |
| E. starts<8 g=0.10 cap=60 | 0.6257 | 0.6252 | -0.0002 | -0.0007 | +0.0002 | -0.0010 |
| E. starts<8 g=0.20 | 0.6257 | 0.6247 | -0.0002 | -0.0013 | +0.0004 | -0.0017 |
| E. starts<8 g=0.20 cap=20 | 0.6255 | 0.6251 | -0.0003 | -0.0008 | -0.0003 | -0.0009 |
| E. starts<8 g=0.20 cap=40 | 0.6254 | 0.6247 | -0.0004 | -0.0013 | -0.0001 | -0.0016 |
| E. starts<8 g=0.20 cap=60 | 0.6255 | 0.6246 | -0.0004 | -0.0013 | +0.0005 | -0.0018 |
| E. starts<8 g=0.35 | 0.6257 | 0.6240 | -0.0002 | -0.0019 | +0.0009 | -0.0027 |
| E. starts<8 g=0.35 cap=20 | 0.6253 | 0.6246 | -0.0006 | -0.0013 | -0.0005 | -0.0016 |
| E. starts<8 g=0.35 cap=40 | 0.6253 | 0.6239 | -0.0006 | -0.0020 | -0.0001 | -0.0026 |
| E. starts<8 g=0.35 cap=60 | 0.6254 | 0.6239 | -0.0005 | -0.0021 | +0.0011 | -0.0029 |
| E. starts<8 g=0.50 | 0.6259 | 0.6236 | +0.0001 | -0.0024 | +0.0017 | -0.0034 |
| E. starts<8 g=0.50 cap=20 | 0.6251 | 0.6241 | -0.0008 | -0.0019 | -0.0006 | -0.0022 |
| E. starts<8 g=0.50 cap=40 | 0.6252 | 0.6232 | -0.0007 | -0.0027 | +0.0002 | -0.0034 |
| E. starts<8 g=0.50 cap=60 | 0.6254 | 0.6233 | -0.0005 | -0.0027 | +0.0020 | -0.0038 |
| E. starts<8 g=0.75 | 0.6267 | 0.6234 | +0.0008 | -0.0026 | +0.0036 | -0.0041 |
| E. starts<8 g=0.75 cap=20 | 0.6249 | 0.6233 | -0.0010 | -0.0026 | -0.0007 | -0.0031 |
| E. starts<8 g=0.75 cap=40 | 0.6253 | 0.6224 | -0.0006 | -0.0035 | +0.0009 | -0.0046 |
| E. starts<8 g=0.75 cap=60 | 0.6258 | 0.6227 | -0.0001 | -0.0032 | +0.0038 | -0.0050 |
| F. starts<17 | 0.6270 | 0.6225 | +0.0011 | -0.0035 | +0.0060 | -0.0059 |
| F. starts<17 cap=20 | 0.6240 | 0.6217 | -0.0019 | -0.0042 | -0.0017 | -0.0049 |
| F. starts<17 cap=40 | 0.6250 | 0.6205 | -0.0009 | -0.0054 | +0.0004 | -0.0069 |
| F. starts<17 cap=60 | 0.6257 | 0.6219 | -0.0002 | -0.0040 | +0.0057 | -0.0065 |
| F. starts<17 g=0.10 | 0.6255 | 0.6249 | -0.0004 | -0.0010 | +0.0001 | -0.0013 |
| F. starts<17 g=0.10 cap=20 | 0.6256 | 0.6254 | -0.0003 | -0.0005 | -0.0003 | -0.0006 |
| F. starts<17 g=0.10 cap=40 | 0.6255 | 0.6251 | -0.0004 | -0.0009 | -0.0003 | -0.0011 |
| F. starts<17 g=0.10 cap=60 | 0.6254 | 0.6250 | -0.0004 | -0.0009 | +0.0001 | -0.0012 |
| F. starts<17 g=0.20 | 0.6252 | 0.6241 | -0.0007 | -0.0019 | +0.0003 | -0.0024 |
| F. starts<17 g=0.20 cap=20 | 0.6253 | 0.6249 | -0.0006 | -0.0011 | -0.0005 | -0.0012 |
| F. starts<17 g=0.20 cap=40 | 0.6252 | 0.6243 | -0.0007 | -0.0017 | -0.0005 | -0.0020 |
| F. starts<17 g=0.20 cap=60 | 0.6251 | 0.6242 | -0.0008 | -0.0017 | +0.0003 | -0.0023 |
| F. starts<17 g=0.35 | 0.6249 | 0.6231 | -0.0009 | -0.0029 | +0.0008 | -0.0038 |
| F. starts<17 g=0.35 cap=20 | 0.6249 | 0.6242 | -0.0009 | -0.0018 | -0.0009 | -0.0020 |
| F. starts<17 g=0.35 cap=40 | 0.6248 | 0.6232 | -0.0010 | -0.0028 | -0.0007 | -0.0033 |
| F. starts<17 g=0.35 cap=60 | 0.6247 | 0.6232 | -0.0011 | -0.0027 | +0.0009 | -0.0036 |
| F. starts<17 g=0.50 | 0.6250 | 0.6224 | -0.0009 | -0.0036 | +0.0015 | -0.0048 |
| F. starts<17 g=0.50 cap=20 | 0.6246 | 0.6235 | -0.0013 | -0.0024 | -0.0012 | -0.0028 |
| F. starts<17 g=0.50 cap=40 | 0.6247 | 0.6223 | -0.0012 | -0.0036 | -0.0008 | -0.0044 |
| F. starts<17 g=0.50 cap=60 | 0.6246 | 0.6225 | -0.0012 | -0.0034 | +0.0016 | -0.0047 |
| F. starts<17 g=0.75 | 0.6257 | 0.6220 | -0.0002 | -0.0039 | +0.0034 | -0.0058 |
| F. starts<17 g=0.75 cap=20 | 0.6242 | 0.6225 | -0.0016 | -0.0034 | -0.0015 | -0.0039 |
| F. starts<17 g=0.75 cap=40 | 0.6246 | 0.6212 | -0.0013 | -0.0048 | -0.0004 | -0.0059 |
| F. starts<17 g=0.75 cap=60 | 0.6249 | 0.6219 | -0.0010 | -0.0041 | +0.0033 | -0.0059 |
| G. changed OR starts<8 | 0.6277 | 0.6241 | +0.0019 | -0.0019 | +0.0077 | -0.0043 |
| G. changed OR starts<8 cap=20 | 0.6246 | 0.6219 | -0.0012 | -0.0040 | -0.0046 | -0.0039 |
| G. changed OR starts<8 cap=40 | 0.6252 | 0.6211 | -0.0007 | -0.0049 | -0.0022 | -0.0055 |
| G. changed OR starts<8 cap=60 | 0.6261 | 0.6227 | +0.0002 | -0.0033 | +0.0061 | -0.0056 |
| G. changed OR starts<8 g=0.10 | 0.6256 | 0.6253 | -0.0002 | -0.0007 | +0.0002 | -0.0009 |
| G. changed OR starts<8 g=0.10 cap=20 | 0.6257 | 0.6255 | -0.0002 | -0.0005 | -0.0006 | -0.0005 |
| G. changed OR starts<8 g=0.10 cap=40 | 0.6256 | 0.6252 | -0.0003 | -0.0008 | -0.0006 | -0.0008 |
| G. changed OR starts<8 g=0.10 cap=60 | 0.6256 | 0.6252 | -0.0003 | -0.0007 | +0.0001 | -0.0010 |
| G. changed OR starts<8 g=0.20 | 0.6256 | 0.6247 | -0.0003 | -0.0013 | +0.0005 | -0.0017 |
| G. changed OR starts<8 g=0.20 cap=20 | 0.6255 | 0.6250 | -0.0004 | -0.0010 | -0.0011 | -0.0009 |
| G. changed OR starts<8 g=0.20 cap=40 | 0.6254 | 0.6245 | -0.0005 | -0.0014 | -0.0011 | -0.0016 |
| G. changed OR starts<8 g=0.20 cap=60 | 0.6253 | 0.6246 | -0.0005 | -0.0014 | +0.0003 | -0.0018 |
| G. changed OR starts<8 g=0.35 | 0.6255 | 0.6241 | -0.0004 | -0.0019 | +0.0011 | -0.0027 |
| G. changed OR starts<8 g=0.35 cap=20 | 0.6252 | 0.6243 | -0.0006 | -0.0016 | -0.0019 | -0.0016 |
| G. changed OR starts<8 g=0.35 cap=40 | 0.6251 | 0.6236 | -0.0008 | -0.0024 | -0.0017 | -0.0026 |
| G. changed OR starts<8 g=0.35 cap=60 | 0.6252 | 0.6238 | -0.0007 | -0.0022 | +0.0008 | -0.0029 |
| G. changed OR starts<8 g=0.50 | 0.6257 | 0.6237 | -0.0001 | -0.0023 | +0.0021 | -0.0034 |
| G. changed OR starts<8 g=0.50 cap=20 | 0.6250 | 0.6237 | -0.0009 | -0.0023 | -0.0027 | -0.0022 |
| G. changed OR starts<8 g=0.50 cap=40 | 0.6249 | 0.6228 | -0.0009 | -0.0032 | -0.0022 | -0.0034 |
| G. changed OR starts<8 g=0.50 cap=60 | 0.6251 | 0.6232 | -0.0008 | -0.0027 | +0.0016 | -0.0038 |
| G. changed OR starts<8 g=0.75 | 0.6264 | 0.6235 | +0.0006 | -0.0024 | +0.0045 | -0.0041 |
| G. changed OR starts<8 g=0.75 cap=20 | 0.6247 | 0.6227 | -0.0011 | -0.0032 | -0.0037 | -0.0031 |
| G. changed OR starts<8 g=0.75 cap=40 | 0.6249 | 0.6218 | -0.0010 | -0.0042 | -0.0024 | -0.0046 |
| G. changed OR starts<8 g=0.75 cap=60 | 0.6254 | 0.6227 | -0.0005 | -0.0033 | +0.0035 | -0.0050 |
| H. changed OR starts<17 | 0.6263 | 0.6228 | +0.0004 | -0.0031 | +0.0077 | -0.0059 |
| H. changed OR starts<17 cap=20 | 0.6238 | 0.6211 | -0.0021 | -0.0048 | -0.0046 | -0.0049 |
| H. changed OR starts<17 cap=40 | 0.6245 | 0.6200 | -0.0014 | -0.0059 | -0.0022 | -0.0069 |
| H. changed OR starts<17 cap=60 | 0.6250 | 0.6220 | -0.0009 | -0.0040 | +0.0061 | -0.0065 |
| H. changed OR starts<17 g=0.10 | 0.6254 | 0.6249 | -0.0005 | -0.0010 | +0.0002 | -0.0013 |
| H. changed OR starts<17 g=0.10 cap=20 | 0.6256 | 0.6253 | -0.0003 | -0.0006 | -0.0006 | -0.0006 |
| H. changed OR starts<17 g=0.10 cap=40 | 0.6255 | 0.6250 | -0.0004 | -0.0009 | -0.0006 | -0.0011 |
| H. changed OR starts<17 g=0.10 cap=60 | 0.6253 | 0.6250 | -0.0005 | -0.0009 | +0.0001 | -0.0012 |
| H. changed OR starts<17 g=0.20 | 0.6250 | 0.6241 | -0.0009 | -0.0018 | +0.0005 | -0.0024 |
| H. changed OR starts<17 g=0.20 cap=20 | 0.6253 | 0.6248 | -0.0006 | -0.0012 | -0.0011 | -0.0012 |
| H. changed OR starts<17 g=0.20 cap=40 | 0.6251 | 0.6241 | -0.0008 | -0.0018 | -0.0011 | -0.0020 |
| H. changed OR starts<17 g=0.20 cap=60 | 0.6249 | 0.6242 | -0.0010 | -0.0017 | +0.0003 | -0.0023 |
| H. changed OR starts<17 g=0.35 | 0.6246 | 0.6231 | -0.0012 | -0.0028 | +0.0011 | -0.0038 |
| H. changed OR starts<17 g=0.35 cap=20 | 0.6248 | 0.6240 | -0.0010 | -0.0020 | -0.0019 | -0.0020 |
| H. changed OR starts<17 g=0.35 cap=40 | 0.6246 | 0.6230 | -0.0013 | -0.0030 | -0.0017 | -0.0033 |
| H. changed OR starts<17 g=0.35 cap=60 | 0.6245 | 0.6232 | -0.0014 | -0.0027 | +0.0008 | -0.0036 |
| H. changed OR starts<17 g=0.50 | 0.6246 | 0.6225 | -0.0013 | -0.0034 | +0.0021 | -0.0048 |
| H. changed OR starts<17 g=0.50 cap=20 | 0.6245 | 0.6232 | -0.0014 | -0.0027 | -0.0027 | -0.0028 |
| H. changed OR starts<17 g=0.50 cap=40 | 0.6243 | 0.6220 | -0.0016 | -0.0039 | -0.0022 | -0.0044 |
| H. changed OR starts<17 g=0.50 cap=60 | 0.6242 | 0.6225 | -0.0017 | -0.0034 | +0.0016 | -0.0047 |
| H. changed OR starts<17 g=0.75 | 0.6251 | 0.6222 | -0.0008 | -0.0037 | +0.0045 | -0.0058 |
| H. changed OR starts<17 g=0.75 cap=20 | 0.6241 | 0.6221 | -0.0018 | -0.0039 | -0.0037 | -0.0039 |
| H. changed OR starts<17 g=0.75 cap=40 | 0.6242 | 0.6208 | -0.0017 | -0.0052 | -0.0024 | -0.0059 |
| H. changed OR starts<17 g=0.75 cap=60 | 0.6243 | 0.6219 | -0.0016 | -0.0041 | +0.0035 | -0.0059 |
| I. DIAG aggressive (starts<8) | 0.6279 | 0.6238 | +0.0020 | -0.0022 | +0.0061 | -0.0043 |
| I. DIAG aggressive (starts<8) cap=20 | 0.6248 | 0.6227 | -0.0011 | -0.0032 | -0.0007 | -0.0039 |
| I. DIAG aggressive (starts<8) cap=40 | 0.6256 | 0.6219 | -0.0002 | -0.0040 | +0.0020 | -0.0055 |
| I. DIAG aggressive (starts<8) cap=60 | 0.6265 | 0.6227 | +0.0007 | -0.0033 | +0.0062 | -0.0056 |
| I. DIAG aggressive (starts<8) g=0.10 | 0.6257 | 0.6253 | -0.0001 | -0.0007 | +0.0001 | -0.0009 |
| I. DIAG aggressive (starts<8) g=0.10 cap=20 | 0.6257 | 0.6255 | -0.0002 | -0.0004 | -0.0001 | -0.0005 |
| I. DIAG aggressive (starts<8) g=0.10 cap=40 | 0.6256 | 0.6253 | -0.0002 | -0.0007 | -0.0001 | -0.0008 |
| I. DIAG aggressive (starts<8) g=0.10 cap=60 | 0.6257 | 0.6252 | -0.0002 | -0.0007 | +0.0002 | -0.0010 |
| I. DIAG aggressive (starts<8) g=0.20 | 0.6257 | 0.6247 | -0.0002 | -0.0013 | +0.0004 | -0.0017 |
| I. DIAG aggressive (starts<8) g=0.20 cap=20 | 0.6255 | 0.6251 | -0.0003 | -0.0008 | -0.0003 | -0.0009 |
| I. DIAG aggressive (starts<8) g=0.20 cap=40 | 0.6254 | 0.6247 | -0.0004 | -0.0013 | -0.0001 | -0.0016 |
| I. DIAG aggressive (starts<8) g=0.20 cap=60 | 0.6255 | 0.6246 | -0.0004 | -0.0013 | +0.0005 | -0.0018 |
| I. DIAG aggressive (starts<8) g=0.35 | 0.6257 | 0.6240 | -0.0002 | -0.0019 | +0.0009 | -0.0027 |
| I. DIAG aggressive (starts<8) g=0.35 cap=20 | 0.6253 | 0.6246 | -0.0006 | -0.0013 | -0.0005 | -0.0016 |
| I. DIAG aggressive (starts<8) g=0.35 cap=40 | 0.6253 | 0.6239 | -0.0006 | -0.0020 | -0.0001 | -0.0026 |
| I. DIAG aggressive (starts<8) g=0.35 cap=60 | 0.6254 | 0.6239 | -0.0005 | -0.0021 | +0.0011 | -0.0029 |
| I. DIAG aggressive (starts<8) g=0.50 | 0.6259 | 0.6236 | +0.0001 | -0.0024 | +0.0017 | -0.0034 |
| I. DIAG aggressive (starts<8) g=0.50 cap=20 | 0.6251 | 0.6241 | -0.0008 | -0.0019 | -0.0006 | -0.0022 |
| I. DIAG aggressive (starts<8) g=0.50 cap=40 | 0.6252 | 0.6232 | -0.0007 | -0.0027 | +0.0002 | -0.0034 |
| I. DIAG aggressive (starts<8) g=0.50 cap=60 | 0.6254 | 0.6233 | -0.0005 | -0.0027 | +0.0020 | -0.0038 |
| I. DIAG aggressive (starts<8) g=0.75 | 0.6267 | 0.6234 | +0.0008 | -0.0026 | +0.0036 | -0.0041 |
| I. DIAG aggressive (starts<8) g=0.75 cap=20 | 0.6249 | 0.6233 | -0.0010 | -0.0026 | -0.0007 | -0.0031 |
| I. DIAG aggressive (starts<8) g=0.75 cap=40 | 0.6253 | 0.6224 | -0.0006 | -0.0035 | +0.0009 | -0.0046 |
| I. DIAG aggressive (starts<8) g=0.75 cap=60 | 0.6258 | 0.6227 | -0.0001 | -0.0032 | +0.0038 | -0.0050 |

## Decision

**Frozen-incumbent overlay is NOT promoted.**

The experiment design has a critical flaw: the Platt model is fitted once on full 2021-2024 data, but the rolling-origin validation folds include 2023 and 2024 as validation years — the fitted model saw those years during fitting. This means the validation LL numbers are not proper rolling-origin estimates and cannot be used to select a promoted variant.

The 2025 holdout comparison IS valid (fitted on pre-2025 data), and several overlay variants show small improvements (best: ∆ = -0.0059). However, these gains must be confirmed with a per-fold validation design before promotion.

### Suggested Re-run Design

To properly evaluate the frozen overlay, re-run with per-fold fitting:

```
for train_seasons, val_season in ROLLING_FOLDS:
    # Fit incumbent Platt on train_seasons only
    # Apply overlay (same formula)
    # Evaluate on val_season
```

This ensures no future data leaks into val-segment predictions.

Best validation: H. changed OR starts<17 cap=20 (0.6238)
Best holdout: H. changed OR starts<17 cap=40 (0.6200)

Incumbent: val=0.6259, hold=0.6259

### QB-Change Game Impact

Best QB-change improvement: B. qb_changed cap=20 (-0.0046)
Worst QB-change degradation: D. starts<4 cap=60 (+0.0078)

### Non-QB-Change Game Impact

Best No-QB-change improvement: F. starts<17 cap=40 (-0.0069)
Worst No-QB-change degradation: B. qb_changed g=0.10 (+0.0000)

### Gate Coverage

Games with active gate: 1058 (76.2%)
Games with no gate: 330 (23.8%)

### Failure Modes

1. **Oracle QB data**: Uses final actual starter IDs, not pregame-announced.
2. **Binary gate sharpness**: The gate is all-or-nothing per game. A QB change from Mahomes to a backup triggers the same overlay as Mahomes to Allen.
3. **Small-sample QB adjustments**: QBs with <17 starts are strongly shrunk toward replacement. The overlay can't amplify a near-zero adjustment.
4. **No position-group interaction**: Ignores OL, skill players, defense.

### Recommended Next Experiment

1. **Coach-QB interaction features**: The combination of a new QB + new coordinator may be more informative than QB change alone.
2. **Expanded Elo K search**: Test K > 48 with season regression spine.
3. **DVOA/EPA features**: If available from a new pregame-safe data source.
4. **Any model must beat Standard Elo + qb_changed + rolling_mov_3 + Platt (holdout LL 0.6262)** to become the new clean football-only incumbent.
5. **QB adjustment is now diagnostic-only.** Use for slice analysis in residual diagnostics and experiment reports, not as a promoted feature.

---
*Report generated by `sportslab frozen-qb-overlay`. Gates: ['A. Incumbent baseline', 'B. qb_changed', 'C. qb_differs_prev', 'D. starts<4', 'E. starts<8', 'F. starts<17', 'G. changed OR starts<8', 'H. changed OR starts<17', 'I. DIAG aggressive (starts<8)'], Gamma sweep: [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], Caps: [None, 20, 40, 60].*
