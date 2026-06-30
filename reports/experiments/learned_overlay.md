# Learned Overlay Experiment

## Research Question

Can a single regularized logistic model (L1/L2) learn a better combination of base features + QB adjustment signals than the hand-tuned two-layer v3.0.0 (Platt + frozen QB overlay)?

## Architecture

```
v3.0.0 (two-layer):
  Layer 1: Platt logistic on [elo_prob, qb_changed, mov_3]
  Layer 2: Hand-tuned QB overlay (gamma=1, cap=40, gate=starts<17 OR changed)
  Final: Layer 1 -> Layer 2 (logit-space additive)

Challenger (single layer):
  Regularized logistic on [base + QB adj + depth features]
  All signals learned jointly with L1/L2 shrinkage
```

## Feature Sets

- **base**: elo_prob, home_qb_changed, away_qb_changed, home_rolling_mov_3, away_rolling_mov_3
- **base+adj**: elo_prob, home_qb_changed, away_qb_changed, home_rolling_mov_3, away_rolling_mov_3, home_qb_adj, away_qb_adj
- **base+depth**: elo_prob, home_qb_changed, away_qb_changed, home_rolling_mov_3, away_rolling_mov_3, home_qb_team_starts_pre, away_qb_team_starts_pre, home_games_since_qb_change, away_games_since_qb_change
- **all**: elo_prob, home_qb_changed, away_qb_changed, home_rolling_mov_3, away_rolling_mov_3, home_qb_adj, away_qb_adj, home_qb_team_starts_pre, away_qb_team_starts_pre, home_games_since_qb_change, away_games_since_qb_change
- **adj_only**: elo_prob, home_qb_adj, away_qb_adj

## Hyperparameter Grid

C values: [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0]
Penalties: ['l2', 'l1']
Feature sets: 5
Total variants: 100

## Validation Results

| Feature Set | Best Config | Avg Val LL | Delta vs v3.0.0 |
|------------|-------------|-----------|------------------|
| adj_only | C=1.0 l2 | 0.6414 | +0.0109 |
| all | C=0.01 l2 | 0.6303 | -0.0002 |
| base | C=100.0 l2 | 0.6340 | +0.0035 |
| base+adj | C=0.5 l2 | 0.6353 | +0.0048 |
| base+depth | C=0.01 l2 | 0.6320 | +0.0014 |
| v3.0.0 inc | (hand-tuned overlay) | 0.6305 | -- |

### Top 10 Overall

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| all C=0.01 l2 | 0.6303 | 0.6289 | 0.6630 | 0.5991 |
| base+depth C=0.01 l2 | 0.6320 | 0.6307 | 0.6654 | 0.5998 |
| base C=100.0 l2 | 0.6340 | 0.6413 | 0.6579 | 0.6029 |
| base C=100.0 l1 | 0.6340 | 0.6413 | 0.6579 | 0.6029 |
| base C=1000.0 l2 | 0.6340 | 0.6413 | 0.6579 | 0.6029 |
| base C=1000.0 l1 | 0.6340 | 0.6413 | 0.6579 | 0.6029 |
| base C=5.0 l2 | 0.6341 | 0.6414 | 0.6579 | 0.6029 |
| base C=10.0 l2 | 0.6341 | 0.6414 | 0.6579 | 0.6029 |
| base C=10.0 l1 | 0.6341 | 0.6414 | 0.6579 | 0.6029 |
| base C=5.0 l1 | 0.6341 | 0.6415 | 0.6578 | 0.6030 |

... (90 more variants)

## 2025 Holdout Results

| Model | Log Loss | Brier | AUC | Accuracy | Selection |
|-------|----------|-------|-----|----------|-----------|
| v3.0.0 inc | 0.6200 | -- | -- | -- | baseline |
| all C=0.01 l2 | 0.6293 | 0.2199 | 0.6966 | 0.6413 | validation |

### Best per Feature Set (Holdout)

| Feature Set | Best Config | Holdout LL |
|------------|-------------|------------|
| adj_only | adj_only C=0.05 l1 | 0.6329 |
| all | all C=0.05 l1 | 0.6267 |
| base | base C=0.05 l2 | 0.6252 |
| base+adj | base+adj C=0.1 l1 | 0.6278 |
| base+depth | base+depth C=0.01 l2 | 0.6252 |
| v3.0.0 inc | (hand-tuned overlay) | 0.6200 |

### All Variants (Holdout)

| Model | Log Loss | Brier | AUC |
|-------|----------|-------|-----|
| base+depth C=0.01 l2 | 0.6252 | 0.2179 | 0.7037 |
| base C=0.05 l2 | 0.6252 | 0.2179 | 0.7058 |
| base+depth C=0.05 l1 | 0.6253 | 0.2178 | 0.7034 |
| base C=0.1 l2 | 0.6255 | 0.2179 | 0.7056 |
| base C=0.5 l1 | 0.6258 | 0.2181 | 0.7046 |
| base C=0.5 l2 | 0.6259 | 0.2181 | 0.7051 |
| base C=1.0 l1 | 0.6259 | 0.2181 | 0.7049 |
| base C=1.0 l2 | 0.6260 | 0.2181 | 0.7048 |
| base C=5.0 l1 | 0.6260 | 0.2181 | 0.7047 |
| base C=5.0 l2 | 0.6260 | 0.2181 | 0.7047 |
| base C=10.0 l1 | 0.6260 | 0.2181 | 0.7047 |
| base C=10.0 l2 | 0.6260 | 0.2181 | 0.7047 |
| base C=100.0 l1 | 0.6260 | 0.2181 | 0.7046 |
| base C=100.0 l2 | 0.6260 | 0.2181 | 0.7046 |
| base C=1000.0 l1 | 0.6260 | 0.2181 | 0.7046 |
| base C=1000.0 l2 | 0.6260 | 0.2181 | 0.7046 |
| base C=0.1 l1 | 0.6261 | 0.2183 | 0.7034 |
| all C=0.05 l1 | 0.6267 | 0.2186 | 0.7014 |
| base C=0.01 l2 | 0.6274 | 0.2188 | 0.7071 |
| base C=0.05 l1 | 0.6278 | 0.2191 | 0.7019 |
| base+adj C=0.1 l1 | 0.6278 | 0.2192 | 0.6991 |
| base+adj C=0.5 l1 | 0.6281 | 0.2193 | 0.6993 |
| base+adj C=0.05 l2 | 0.6281 | 0.2194 | 0.6983 |
| base+adj C=0.1 l2 | 0.6282 | 0.2194 | 0.6990 |
| base+adj C=1.0 l1 | 0.6282 | 0.2194 | 0.6996 |
| base+adj C=0.5 l2 | 0.6283 | 0.2194 | 0.6999 |
| base+adj C=5.0 l1 | 0.6283 | 0.2194 | 0.6995 |
| base+adj C=1.0 l2 | 0.6283 | 0.2194 | 0.6997 |
| base+adj C=10.0 l1 | 0.6283 | 0.2194 | 0.6994 |
| base+adj C=5.0 l2 | 0.6283 | 0.2194 | 0.6995 |
| base+adj C=10.0 l2 | 0.6283 | 0.2194 | 0.6994 |
| base+adj C=100.0 l1 | 0.6283 | 0.2194 | 0.6994 |
| base+adj C=100.0 l2 | 0.6283 | 0.2194 | 0.6994 |
| base+adj C=1000.0 l1 | 0.6283 | 0.2194 | 0.6994 |
| base+adj C=1000.0 l2 | 0.6283 | 0.2194 | 0.6994 |
| base+adj C=0.05 l1 | 0.6290 | 0.2197 | 0.6987 |
| base+depth C=0.1 l1 | 0.6291 | 0.2193 | 0.7042 |
| all C=0.01 l2 | 0.6293 | 0.2199 | 0.6966 |
| base+adj C=0.01 l2 | 0.6296 | 0.2199 | 0.6985 |
| all C=0.1 l1 | 0.6313 | 0.2204 | 0.7013 |
| base+depth C=0.05 l2 | 0.6327 | 0.2206 | 0.7010 |
| adj_only C=0.05 l1 | 0.6329 | 0.2213 | 0.6950 |
| adj_only C=0.1 l1 | 0.6330 | 0.2214 | 0.6918 |
| adj_only C=0.5 l1 | 0.6338 | 0.2218 | 0.6913 |
| adj_only C=1.0 l1 | 0.6340 | 0.2219 | 0.6903 |
| adj_only C=5.0 l1 | 0.6341 | 0.2219 | 0.6903 |
| adj_only C=10.0 l1 | 0.6341 | 0.2219 | 0.6902 |
| adj_only C=100.0 l1 | 0.6342 | 0.2219 | 0.6903 |
| adj_only C=1000.0 l1 | 0.6342 | 0.2219 | 0.6903 |
| adj_only C=1000.0 l2 | 0.6342 | 0.2219 | 0.6903 |
| adj_only C=100.0 l2 | 0.6342 | 0.2219 | 0.6903 |
| adj_only C=10.0 l2 | 0.6342 | 0.2219 | 0.6904 |
| adj_only C=5.0 l2 | 0.6342 | 0.2219 | 0.6904 |
| adj_only C=1.0 l2 | 0.6342 | 0.2220 | 0.6904 |
| adj_only C=0.5 l2 | 0.6343 | 0.2220 | 0.6901 |
| adj_only C=0.1 l2 | 0.6347 | 0.2222 | 0.6886 |
| base+depth C=0.1 l2 | 0.6350 | 0.2214 | 0.7001 |
| adj_only C=0.05 l2 | 0.6353 | 0.2224 | 0.6880 |
| base+depth C=0.5 l1 | 0.6354 | 0.2215 | 0.7013 |
| base+depth C=1.0 l1 | 0.6366 | 0.2219 | 0.7004 |
| all C=0.05 l2 | 0.6370 | 0.2227 | 0.6957 |
| base+depth C=0.5 l2 | 0.6373 | 0.2222 | 0.7005 |
| base+depth C=1.0 l2 | 0.6376 | 0.2223 | 0.7007 |
| base+depth C=5.0 l1 | 0.6377 | 0.2223 | 0.7006 |
| base+depth C=10.0 l1 | 0.6378 | 0.2223 | 0.7007 |
| base+depth C=5.0 l2 | 0.6379 | 0.2224 | 0.7006 |
| base+depth C=10.0 l2 | 0.6379 | 0.2224 | 0.7006 |
| base+depth C=100.0 l1 | 0.6379 | 0.2224 | 0.7006 |
| base+depth C=100.0 l2 | 0.6379 | 0.2224 | 0.7006 |
| base+depth C=1000.0 l1 | 0.6379 | 0.2224 | 0.7006 |
| base+depth C=1000.0 l2 | 0.6379 | 0.2224 | 0.7006 |
| all C=0.5 l1 | 0.6388 | 0.2232 | 0.6967 |
| adj_only C=0.01 l2 | 0.6389 | 0.2239 | 0.6857 |
| all C=0.1 l2 | 0.6391 | 0.2234 | 0.6963 |
| all C=1.0 l1 | 0.6401 | 0.2237 | 0.6971 |
| all C=0.5 l2 | 0.6411 | 0.2241 | 0.6965 |
| all C=5.0 l1 | 0.6414 | 0.2241 | 0.6965 |
| all C=1.0 l2 | 0.6414 | 0.2242 | 0.6966 |
| all C=10.0 l1 | 0.6415 | 0.2242 | 0.6966 |
| all C=5.0 l2 | 0.6416 | 0.2242 | 0.6966 |
| all C=10.0 l2 | 0.6416 | 0.2242 | 0.6965 |
| all C=100.0 l1 | 0.6416 | 0.2242 | 0.6966 |
| all C=100.0 l2 | 0.6417 | 0.2242 | 0.6966 |
| all C=1000.0 l1 | 0.6417 | 0.2242 | 0.6966 |
| all C=1000.0 l2 | 0.6417 | 0.2242 | 0.6965 |
| all C=0.001 l2 | 0.6452 | 0.2266 | 0.7016 |
| base+depth C=0.001 l2 | 0.6475 | 0.2277 | 0.7062 |
| base+adj C=0.001 l2 | 0.6553 | 0.2314 | 0.6990 |
| all C=0.01 l1 | 0.6574 | 0.2324 | 0.7002 |
| base+adj C=0.01 l1 | 0.6574 | 0.2324 | 0.7002 |
| adj_only C=0.01 l1 | 0.6574 | 0.2324 | 0.7002 |
| base+depth C=0.01 l1 | 0.6574 | 0.2324 | 0.7002 |
| base C=0.01 l1 | 0.6574 | 0.2324 | 0.7002 |
| base C=0.001 l2 | 0.6594 | 0.2334 | 0.7096 |
| adj_only C=0.001 l2 | 0.6637 | 0.2355 | 0.6823 |
| adj_only C=0.001 l1 | 0.6906 | 0.2487 | 0.5000 |
| base C=0.001 l1 | 0.6906 | 0.2487 | 0.5000 |
| base+depth C=0.001 l1 | 0.6906 | 0.2487 | 0.5000 |
| base+adj C=0.001 l1 | 0.6906 | 0.2487 | 0.5000 |
| all C=0.001 l1 | 0.6906 | 0.2487 | 0.5000 |

## Decision

**REJECTED**

Val Delta: +0.0002 (need >= 0.001)
Holdout Delta: -0.0093 (need >= 0.001)
Regularized logistic does NOT beat v3.0.0 on validation.
Val-selected variant does NOT beat v3.0.0 on holdout.

### Post-Hoc (Diagnostic Only)

The best holdout variant (base+depth C=0.01 l2) differs from the validation-selected variant. This is a diagnostic finding only.

---
*Report generated by `sportslab learned-overlay`. Seasons: 2021--2025, Folds: 3, Variants: 100.*
