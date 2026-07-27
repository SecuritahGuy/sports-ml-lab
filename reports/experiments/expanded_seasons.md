# Expanded Seasons Experiment

Compare the v3.0.0 Frozen QB Overlay incument trained on 3 different season ranges:

| Label | Training Seasons | Games | Description |
|-------|-----------------|-------|-------------|
| D (Baseline 2021–2024) | [2021, 2022, 2023, 2024] | 1112 | Current baseline (production freeze) |
| A (Skip 2020) | [2019, 2021, 2022, 2023, 2024] | 1372 | Pre-COVID 2019 added, COVD-19 season excluded |
| C (Include 2020) | [2020, 2021, 2022, 2023, 2024] | 1376 | Full 2019–2024 including COVID season |

## Rolling-Origin Validation

| Flavor | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|-----------|-------|-------|-------|
| D (Baseline 2021–2024) | 0.6334 | 0.6413 | 0.6573 | 0.6016 |
| A (Skip 2020) | 0.6431 | 0.6725 | 0.6384 | 0.6591 | 0.6023 |
| C (Include 2020) | 0.6429 | 0.6684 | 0.6392 | 0.6654 | 0.5985 |

## 2025 Holdout

| Flavor | Holdout LL | Brier | Acc | N_train |
|-------|-----------|-------|-----|--------|
| D (Baseline 2021–2024) | 0.6262 | 0.2180 | 0.6630 | 1112 |
| A (Skip 2020) | 0.6239 | 0.2173 | 0.6630 | 1372 |
| C (Include 2020) | 0.6242 | 0.2172 | 0.6667 | 1376 |

## Δ vs Baseline

| Flavor | Δ Val LL | Δ Holdout LL | Common-fold Val | Beats Both? | Verdict |
|-------|----------|-------------|-----------------|-------------|--------|
| D (Baseline 2021–2024) | — | — | — | — | Baseline |
| A (Skip 2020) | +0.0097 | -0.0023 | 0.6333 | ❌ | REJECTED |
| C (Include 2020) | +0.0095 | -0.0020 | 0.6344 | ❌ | REJECTED |

## Common Folds Only (2022–2024)

Expanded flavors have 4 folds (2021–2024 val) vs baseline's 3 (2022–2024 val). The extra 2021 fold is harder (fewer training seasons), pulling the average down. Comparing only the 3 common folds:

| Flavor | Common-Fold Val LL | Holdout LL |
|-------|-------------------|-----------|
| D (Baseline 2021–2024) | 0.6334 | 0.6262 |
| A (Skip 2020) | 0.6333 | 0.6239 |
| C (Include 2020) | 0.6344 | 0.6242 |

## Decision

**No flavor beats incumbent on both val and holdout. Incumbent unchanged.**

### Key findings

- **A (Skip 2020)**: Common-fold val LL = 0.6333 (-0.0001 vs baseline), Holdout LL = 0.6239 (-0.0023 vs baseline). Val ≈ tied with baseline. 
- **C (Include 2020)**: Common-fold val LL = 0.6344 (+0.0010 vs baseline), Holdout LL = 0.6242 (-0.0020 vs baseline). 
- **Common-fold val (2022-2024)**: Skip 2020 ties baseline (0.6333 vs 0.6334). Include 2020 is slightly worse (0.6344).
- **Holdout**: Both expanded flavors improve slightly (−0.0023 and −0.0020).
- **Verdict**: Neither flavor beats baseline on BOTH val and holdout by ≥ 0.001.
- **Recommendation**: Expanded data does not warrant promotion. The holdout improvement is small (−0.002) and doesn't justify expanding the season range.

Reference: baseline Platt(qb_changed + rolling_mov_3) val LL = 0.6334, holdout LL = 0.6262

