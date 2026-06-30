# Fold-Safe Frozen-Incumbent QB Overlay Experiment

## Research Question

Does the frozen QB overlay still improve the incumbent when the incumbent base probabilities and calibration are generated using strictly fold-safe rolling-origin training?

## Why the Prior Frozen Overlay Was Diagnostic Only

The V2 frozen overlay fitted Platt calibration **once** on full 2021–2024 data, then used that single fit to evaluate ALL rolling-origin folds. This leaked future data into early-fold validation (e.g., 2023 games scored using a Platt model that had seen 2024 games during training). The validation metrics were not fold-safe and could not be used for trustworthy variant selection.

## Fold-Safe Methodology

For each rolling-origin fold, the following steps are run:

1. **Features**: Elo ratings, QB adjustments, and rolling MOV are computed chronologically on the full dataset. Each game's feature uses only data from games before it — no future leakage in features.
2. **Platt calibration**: Fit using **only** the fold's training seasons. The validation season is never seen during calibration.
3. **Incumbent base probability**: Generated using the fold's Platt model. The base probability for all games is produced by a model trained exclusively on pre-validation data.
4. **Frozen overlay**: Applied in logit space only where the pregame gate is active. Non-gated games are unchanged.
5. **Scoring**: Computed only on the validation season.

## Validation Folds

| Fold | Training Seasons | Validation Season |
|------|-----------------|-------------------|
| 1 | 2021 | 2022 |
| 2 | 2021, 2022 | 2023 |
| 3 | 2021, 2022, 2023 | 2024 |

## Holdout

The 2025 season (year 2025) is held out entirely from validation. A final Platt model is fitted on all 2021–2024 data and the validation-selected variant is evaluated once on 2025.

## Variant Selection

The best variant is selected by **average validation log loss** across all 3 rolling folds. The 2025 holdout is NOT used for selection. Best-holdout and cleanest-gate results are reported separately as diagnostic.

## Variants Tested

| Letter | Gate Condition | Gammas | Caps |
|--------|----------------|--------|------|
| A | (none) Incumbent baseline | — | — |
| B | qb_changed (either side) | 0.10–1.00 | 20, 40, 60 |
| C | qb_differs_prev (same as B) | 0.10–1.00 | 20, 40, 60 |
| D | starts<4 (either QB) | 0.10–1.00 | 20, 40, 60 |
| E | starts<8 | 0.10–1.00 | 20, 40, 60 |
| F | starts<17 | 0.10–1.00 | 20, 40, 60 |
| G | changed OR starts<8 | 0.10–1.00 | 20, 40, 60 |
| H | changed OR starts<17 | 0.10–1.00 | 20, 40, 60 |

Total variants: 127 (1 baseline + 126 overlays)

## Fold-Safe Validation Results (Sorted by Avg Log Loss)

| Model | Avg LL | Fold1 | Fold2 | Fold3 |
|-------|--------|-------|-------|-------|
| H. changed OR starts<17 cap=40 | 0.6305 | 0.6360 | 0.6596 | 0.5960 |
| H. changed OR starts<17 g=0.75 cap=60 | 0.6306 | 0.6374 | 0.6589 | 0.5954 |
| H. changed OR starts<17 cap=60 | 0.6306 | 0.6370 | 0.6607 | 0.5942 |
| H. changed OR starts<17 g=0.75 cap=40 | 0.6307 | 0.6368 | 0.6584 | 0.5970 |
| H. changed OR starts<17 cap=20 | 0.6308 | 0.6370 | 0.6573 | 0.5981 |
| F. starts<17 cap=20 | 0.6311 | 0.6370 | 0.6577 | 0.5985 |
| H. changed OR starts<17 g=0.50 cap=60 | 0.6311 | 0.6382 | 0.6579 | 0.5973 |
| F. starts<17 cap=40 | 0.6313 | 0.6363 | 0.6606 | 0.5969 |
| F. starts<17 g=0.75 cap=60 | 0.6313 | 0.6378 | 0.6598 | 0.5963 |
| F. starts<17 g=0.75 cap=40 | 0.6313 | 0.6371 | 0.6592 | 0.5977 |
| H. changed OR starts<17 g=0.75 cap=20 | 0.6314 | 0.6379 | 0.6571 | 0.5991 |
| H. changed OR starts<17 g=0.50 cap=40 | 0.6315 | 0.6380 | 0.6578 | 0.5986 |
| F. starts<17 cap=60 | 0.6316 | 0.6376 | 0.6618 | 0.5953 |
| F. starts<17 g=0.75 cap=20 | 0.6316 | 0.6379 | 0.6575 | 0.5994 |
| F. starts<17 g=0.50 cap=60 | 0.6316 | 0.6385 | 0.6585 | 0.5979 |
| H. changed OR starts<17 g=0.35 cap=60 | 0.6317 | 0.6390 | 0.6575 | 0.5987 |
| F. starts<17 g=0.50 cap=40 | 0.6319 | 0.6382 | 0.6583 | 0.5991 |
| G. changed OR starts<8 cap=20 | 0.6320 | 0.6369 | 0.6581 | 0.6010 |
| G. changed OR starts<8 cap=40 | 0.6320 | 0.6356 | 0.6597 | 0.6008 |
| G. changed OR starts<8 g=0.75 cap=40 | 0.6321 | 0.6366 | 0.6587 | 0.6009 |
| H. changed OR starts<17 g=0.35 cap=40 | 0.6321 | 0.6389 | 0.6576 | 0.5997 |
| F. starts<17 g=0.35 cap=60 | 0.6321 | 0.6392 | 0.6580 | 0.5992 |
| H. changed OR starts<17 g=0.50 cap=20 | 0.6321 | 0.6390 | 0.6572 | 0.6002 |
| F. starts<17 g=0.50 cap=20 | 0.6323 | 0.6390 | 0.6574 | 0.6004 |
| F. starts<17 g=0.35 cap=40 | 0.6323 | 0.6390 | 0.6579 | 0.6001 |
| G. changed OR starts<8 g=0.75 cap=20 | 0.6323 | 0.6379 | 0.6578 | 0.6013 |
| E. starts<8 cap=20 | 0.6324 | 0.6368 | 0.6582 | 0.6021 |
| G. changed OR starts<8 g=0.75 cap=60 | 0.6324 | 0.6378 | 0.6586 | 0.6007 |
| G. changed OR starts<8 g=0.50 cap=40 | 0.6324 | 0.6380 | 0.6580 | 0.6013 |
| G. changed OR starts<8 g=0.50 cap=60 | 0.6325 | 0.6387 | 0.6578 | 0.6010 |
| E. starts<8 g=0.75 cap=20 | 0.6326 | 0.6378 | 0.6579 | 0.6021 |
| E. starts<8 g=0.75 cap=40 | 0.6326 | 0.6366 | 0.6591 | 0.6022 |
| H. changed OR starts<17 g=0.20 cap=60 | 0.6326 | 0.6400 | 0.6575 | 0.6004 |
| H. changed OR starts<17 g=0.35 cap=20 | 0.6327 | 0.6397 | 0.6573 | 0.6010 |
| G. changed OR starts<8 cap=60 | 0.6327 | 0.6373 | 0.6600 | 0.6008 |
| E. starts<8 cap=40 | 0.6327 | 0.6355 | 0.6602 | 0.6025 |
| B. qb_changed cap=40 | 0.6328 | 0.6406 | 0.6569 | 0.6008 |
| C. qb_differs_prev cap=40 | 0.6328 | 0.6406 | 0.6569 | 0.6008 |
| F. starts<17 g=0.35 cap=20 | 0.6328 | 0.6397 | 0.6574 | 0.6012 |
| G. changed OR starts<8 g=0.35 cap=60 | 0.6328 | 0.6394 | 0.6576 | 0.6014 |
| E. starts<8 g=0.50 cap=40 | 0.6328 | 0.6380 | 0.6583 | 0.6022 |
| G. changed OR starts<8 g=0.50 cap=20 | 0.6328 | 0.6390 | 0.6577 | 0.6018 |
| F. starts<17 g=0.20 cap=60 | 0.6328 | 0.6401 | 0.6577 | 0.6007 |
| G. changed OR starts<8 g=0.35 cap=40 | 0.6328 | 0.6390 | 0.6578 | 0.6017 |
| B. qb_changed cap=20 | 0.6329 | 0.6406 | 0.6567 | 0.6013 |
| C. qb_differs_prev cap=20 | 0.6329 | 0.6406 | 0.6567 | 0.6013 |
| H. changed OR starts<17 g=0.20 cap=40 | 0.6329 | 0.6400 | 0.6575 | 0.6011 |
| B. qb_changed g=0.75 cap=40 | 0.6329 | 0.6407 | 0.6569 | 0.6012 |
| C. qb_differs_prev g=0.75 cap=40 | 0.6329 | 0.6407 | 0.6569 | 0.6012 |
| E. starts<8 g=0.50 cap=60 | 0.6329 | 0.6387 | 0.6581 | 0.6020 |
| B. qb_changed cap=60 | 0.6330 | 0.6413 | 0.6566 | 0.6011 |
| C. qb_differs_prev cap=60 | 0.6330 | 0.6413 | 0.6566 | 0.6011 |
| E. starts<8 g=0.50 cap=20 | 0.6330 | 0.6390 | 0.6577 | 0.6023 |
| B. qb_changed g=0.75 cap=60 | 0.6330 | 0.6411 | 0.6566 | 0.6014 |
| C. qb_differs_prev g=0.75 cap=60 | 0.6330 | 0.6411 | 0.6566 | 0.6014 |
| E. starts<8 g=0.75 cap=60 | 0.6330 | 0.6378 | 0.6591 | 0.6022 |
| F. starts<17 g=0.20 cap=40 | 0.6330 | 0.6401 | 0.6577 | 0.6013 |
| E. starts<8 g=0.35 cap=40 | 0.6331 | 0.6390 | 0.6580 | 0.6023 |
| B. qb_changed g=0.75 cap=20 | 0.6331 | 0.6408 | 0.6569 | 0.6017 |
| C. qb_differs_prev g=0.75 cap=20 | 0.6331 | 0.6408 | 0.6569 | 0.6017 |
| E. starts<8 g=0.35 cap=60 | 0.6331 | 0.6394 | 0.6578 | 0.6022 |
| G. changed OR starts<8 g=0.35 cap=20 | 0.6332 | 0.6398 | 0.6576 | 0.6021 |
| B. qb_changed g=0.50 cap=40 | 0.6333 | 0.6409 | 0.6571 | 0.6018 |
| B. qb_changed g=0.50 cap=60 | 0.6333 | 0.6412 | 0.6568 | 0.6018 |
| C. qb_differs_prev g=0.50 cap=40 | 0.6333 | 0.6409 | 0.6571 | 0.6018 |
| C. qb_differs_prev g=0.50 cap=60 | 0.6333 | 0.6412 | 0.6568 | 0.6018 |
| G. changed OR starts<8 g=0.20 cap=60 | 0.6333 | 0.6403 | 0.6575 | 0.6020 |
| H. changed OR starts<17 g=0.20 cap=20 | 0.6333 | 0.6405 | 0.6574 | 0.6019 |
| G. changed OR starts<8 g=0.20 cap=40 | 0.6333 | 0.6400 | 0.6577 | 0.6022 |
| E. starts<8 g=0.35 cap=20 | 0.6333 | 0.6397 | 0.6577 | 0.6025 |
| F. starts<17 g=0.20 cap=20 | 0.6333 | 0.6405 | 0.6575 | 0.6020 |
| H. changed OR starts<17 g=0.10 cap=60 | 0.6333 | 0.6408 | 0.6575 | 0.6017 |
| B. qb_changed g=0.50 cap=20 | 0.6334 | 0.6410 | 0.6571 | 0.6021 |
| C. qb_differs_prev g=0.50 cap=20 | 0.6334 | 0.6410 | 0.6571 | 0.6021 |
| F. starts<17 g=0.10 cap=60 | 0.6334 | 0.6408 | 0.6577 | 0.6018 |
| B. qb_changed g=0.35 cap=40 | 0.6335 | 0.6411 | 0.6572 | 0.6021 |
| B. qb_changed g=0.35 cap=60 | 0.6335 | 0.6412 | 0.6570 | 0.6022 |
| C. qb_differs_prev g=0.35 cap=40 | 0.6335 | 0.6411 | 0.6572 | 0.6021 |
| C. qb_differs_prev g=0.35 cap=60 | 0.6335 | 0.6412 | 0.6570 | 0.6022 |
| E. starts<8 g=0.20 cap=60 | 0.6335 | 0.6403 | 0.6576 | 0.6025 |
| E. starts<8 cap=60 | 0.6335 | 0.6372 | 0.6605 | 0.6027 |
| H. changed OR starts<17 g=0.10 cap=40 | 0.6335 | 0.6408 | 0.6576 | 0.6020 |
| E. starts<8 g=0.20 cap=40 | 0.6335 | 0.6401 | 0.6578 | 0.6026 |
| F. starts<17 g=0.10 cap=40 | 0.6335 | 0.6408 | 0.6577 | 0.6021 |
| G. changed OR starts<8 g=0.20 cap=20 | 0.6336 | 0.6405 | 0.6577 | 0.6025 |
| B. qb_changed g=0.35 cap=20 | 0.6336 | 0.6412 | 0.6573 | 0.6024 |
| C. qb_differs_prev g=0.35 cap=20 | 0.6336 | 0.6412 | 0.6573 | 0.6024 |
| E. starts<8 g=0.20 cap=20 | 0.6336 | 0.6405 | 0.6577 | 0.6027 |
| G. changed OR starts<8 g=0.10 cap=60 | 0.6337 | 0.6409 | 0.6576 | 0.6025 |
| G. changed OR starts<8 g=0.10 cap=40 | 0.6337 | 0.6408 | 0.6577 | 0.6026 |
| H. changed OR starts<17 g=0.10 cap=20 | 0.6337 | 0.6410 | 0.6576 | 0.6025 |
| B. qb_changed g=0.20 cap=40 | 0.6337 | 0.6413 | 0.6574 | 0.6025 |
| B. qb_changed g=0.20 cap=60 | 0.6337 | 0.6414 | 0.6573 | 0.6025 |
| C. qb_differs_prev g=0.20 cap=40 | 0.6337 | 0.6413 | 0.6574 | 0.6025 |
| C. qb_differs_prev g=0.20 cap=60 | 0.6337 | 0.6414 | 0.6573 | 0.6025 |
| F. starts<17 g=0.10 cap=20 | 0.6337 | 0.6411 | 0.6576 | 0.6025 |
| E. starts<8 g=0.10 cap=40 | 0.6338 | 0.6408 | 0.6577 | 0.6028 |
| D. starts<4 g=0.75 cap=20 | 0.6338 | 0.6400 | 0.6575 | 0.6039 |
| E. starts<8 g=0.10 cap=60 | 0.6338 | 0.6409 | 0.6577 | 0.6028 |
| D. starts<4 g=0.50 cap=20 | 0.6338 | 0.6404 | 0.6575 | 0.6036 |
| B. qb_changed g=0.20 cap=20 | 0.6339 | 0.6414 | 0.6575 | 0.6027 |
| C. qb_differs_prev g=0.20 cap=20 | 0.6339 | 0.6414 | 0.6575 | 0.6027 |
| D. starts<4 cap=20 | 0.6339 | 0.6396 | 0.6576 | 0.6044 |
| G. changed OR starts<8 g=0.10 cap=20 | 0.6339 | 0.6411 | 0.6577 | 0.6028 |
| E. starts<8 g=0.10 cap=20 | 0.6339 | 0.6411 | 0.6577 | 0.6029 |
| B. qb_changed g=0.10 cap=60 | 0.6339 | 0.6415 | 0.6575 | 0.6028 |
| C. qb_differs_prev g=0.10 cap=60 | 0.6339 | 0.6415 | 0.6575 | 0.6028 |
| D. starts<4 g=0.35 cap=20 | 0.6339 | 0.6408 | 0.6576 | 0.6034 |
| B. qb_changed g=0.10 cap=40 | 0.6340 | 0.6415 | 0.6576 | 0.6028 |
| C. qb_differs_prev g=0.10 cap=40 | 0.6340 | 0.6415 | 0.6576 | 0.6028 |
| B. qb_changed g=0.10 cap=20 | 0.6340 | 0.6415 | 0.6576 | 0.6029 |
| C. qb_differs_prev g=0.10 cap=20 | 0.6340 | 0.6415 | 0.6576 | 0.6029 |
| D. starts<4 g=0.20 cap=20 | 0.6340 | 0.6411 | 0.6576 | 0.6033 |
| D. starts<4 g=0.20 cap=40 | 0.6340 | 0.6409 | 0.6579 | 0.6033 |
| D. starts<4 g=0.35 cap=40 | 0.6341 | 0.6405 | 0.6581 | 0.6036 |
| D. starts<4 g=0.10 cap=20 | 0.6341 | 0.6414 | 0.6577 | 0.6032 |
| D. starts<4 g=0.10 cap=40 | 0.6341 | 0.6413 | 0.6578 | 0.6032 |
| D. starts<4 g=0.20 cap=60 | 0.6341 | 0.6411 | 0.6579 | 0.6033 |
| A. Incumbent baseline | 0.6341 | 0.6416 | 0.6577 | 0.6031 |
| D. starts<4 g=0.10 cap=60 | 0.6341 | 0.6414 | 0.6578 | 0.6032 |
| D. starts<4 g=0.50 cap=40 | 0.6342 | 0.6401 | 0.6585 | 0.6039 |
| D. starts<4 g=0.35 cap=60 | 0.6342 | 0.6409 | 0.6582 | 0.6035 |
| D. starts<4 g=0.50 cap=60 | 0.6344 | 0.6407 | 0.6585 | 0.6039 |
| D. starts<4 g=0.75 cap=40 | 0.6344 | 0.6396 | 0.6591 | 0.6046 |
| D. starts<4 cap=40 | 0.6349 | 0.6392 | 0.6600 | 0.6054 |
| D. starts<4 g=0.75 cap=60 | 0.6349 | 0.6406 | 0.6594 | 0.6047 |
| D. starts<4 cap=60 | 0.6357 | 0.6408 | 0.6606 | 0.6057 |

**Incumbent baseline val LL:** 0.6341
**Best validation variant:** H. changed OR starts<17 cap=40 (0.6305)

## Best Variant by Gate Family (Validation)

| Gate | Best Config | Val LL |
|------|------------|--------|
| B. qb_changed | B. qb_changed cap=40 | 0.6328 |
| C. qb_differs_prev | C. qb_differs_prev cap=40 | 0.6328 |
| D. starts<4 | D. starts<4 g=0.75 cap=20 | 0.6338 |
| E. starts<8 | E. starts<8 cap=20 | 0.6324 |
| F. starts<17 | F. starts<17 cap=20 | 0.6311 |
| G. changed OR starts<8 | G. changed OR starts<8 cap=20 | 0.6320 |
| H. changed OR starts<17 | H. changed OR starts<17 cap=40 | 0.6305 |
| B. qb_changed (clean) | B. qb_changed cap=20 | 0.6329 |

## 2025 Holdout Results

| Model | Log Loss | Brier | AUC | Accuracy | Selection |
|-------|----------|-------|-----|----------|-----------|
| A. Incumbent baseline | 0.6259 | 0.2181 | 0.7048 | 0.6594 | baseline |
| H. changed OR starts<17 cap=40 | 0.6200 | 0.2157 | 0.7098 | 0.6630 | **validation-selected** |
| B. qb_changed cap=20 | 0.6250 | 0.2177 | 0.7055 | 0.6630 | diagnostic (clean gate) |

## QB-Change Slice Results (2025 Holdout)

QB-change games: 55 | Non-QB-change games: 221

| Variant | QB-Change LL | No-QB-Change LL | QC Δ | NoQC Δ |
|--------|-------------|-----------------|------|--------|
| A. Incumbent baseline | 0.6696 | 0.6151 | — | — |
| H. changed OR starts<17 cap=40 | 0.6674 | 0.6082 | -0.0022 | -0.0069 |
| H. changed OR starts<17 cap=40 (D) | 0.6674 | 0.6082 | -0.0022 | -0.0069 |
| B. qb_changed cap=20 (G) | 0.6650 | 0.6151 | -0.0046 | +0.0000 |

## Non-Gated Equality

**Max absolute diff vs incumbent:** 1.11e-16
**Mean absolute diff vs incumbent:** 5.23e-18
**Equality check: PASSED**

Non-gated games are identical to the incumbent within floating-point tolerance.

## Full 2025 Holdout Results (All Variants)

| Model | Log Loss | Δ vs incumbent |
|-------|----------|----------------|
| H. changed OR starts<17 cap=40 | 0.6200 | -0.0059 |
| F. starts<17 cap=40 | 0.6205 | -0.0054 |
| H. changed OR starts<17 g=0.75 cap=40 | 0.6208 | -0.0052 |
| G. changed OR starts<8 cap=40 | 0.6211 | -0.0049 |
| H. changed OR starts<17 cap=20 | 0.6211 | -0.0048 |
| F. starts<17 g=0.75 cap=40 | 0.6212 | -0.0048 |
| F. starts<17 cap=20 | 0.6217 | -0.0042 |
| G. changed OR starts<8 g=0.75 cap=40 | 0.6218 | -0.0042 |
| F. starts<17 g=0.75 cap=60 | 0.6219 | -0.0041 |
| H. changed OR starts<17 g=0.75 cap=60 | 0.6219 | -0.0041 |
| F. starts<17 cap=60 | 0.6219 | -0.0040 |
| E. starts<8 cap=40 | 0.6219 | -0.0040 |
| G. changed OR starts<8 cap=20 | 0.6219 | -0.0040 |
| H. changed OR starts<17 cap=60 | 0.6220 | -0.0040 |
| H. changed OR starts<17 g=0.50 cap=40 | 0.6220 | -0.0039 |
| H. changed OR starts<17 g=0.75 cap=20 | 0.6221 | -0.0039 |
| F. starts<17 g=0.50 cap=40 | 0.6223 | -0.0036 |
| E. starts<8 g=0.75 cap=40 | 0.6224 | -0.0035 |
| H. changed OR starts<17 g=0.50 cap=60 | 0.6225 | -0.0034 |
| F. starts<17 g=0.50 cap=60 | 0.6225 | -0.0034 |
| F. starts<17 g=0.75 cap=20 | 0.6225 | -0.0034 |
| G. changed OR starts<8 cap=60 | 0.6227 | -0.0033 |
| G. changed OR starts<8 g=0.75 cap=60 | 0.6227 | -0.0033 |
| E. starts<8 cap=60 | 0.6227 | -0.0033 |
| E. starts<8 cap=20 | 0.6227 | -0.0032 |
| E. starts<8 g=0.75 cap=60 | 0.6227 | -0.0032 |
| G. changed OR starts<8 g=0.75 cap=20 | 0.6227 | -0.0032 |
| G. changed OR starts<8 g=0.50 cap=40 | 0.6228 | -0.0032 |
| H. changed OR starts<17 g=0.35 cap=40 | 0.6230 | -0.0030 |
| F. starts<17 g=0.35 cap=40 | 0.6232 | -0.0028 |
| G. changed OR starts<8 g=0.50 cap=60 | 0.6232 | -0.0027 |
| H. changed OR starts<17 g=0.50 cap=20 | 0.6232 | -0.0027 |
| H. changed OR starts<17 g=0.35 cap=60 | 0.6232 | -0.0027 |
| F. starts<17 g=0.35 cap=60 | 0.6232 | -0.0027 |
| E. starts<8 g=0.50 cap=40 | 0.6232 | -0.0027 |
| E. starts<8 g=0.50 cap=60 | 0.6233 | -0.0027 |
| E. starts<8 g=0.75 cap=20 | 0.6233 | -0.0026 |
| D. starts<4 cap=40 | 0.6235 | -0.0025 |
| F. starts<17 g=0.50 cap=20 | 0.6235 | -0.0024 |
| G. changed OR starts<8 g=0.35 cap=40 | 0.6236 | -0.0024 |
| G. changed OR starts<8 g=0.50 cap=20 | 0.6237 | -0.0023 |
| D. starts<4 cap=20 | 0.6238 | -0.0022 |
| G. changed OR starts<8 g=0.35 cap=60 | 0.6238 | -0.0022 |
| D. starts<4 g=0.75 cap=40 | 0.6238 | -0.0021 |
| E. starts<8 g=0.35 cap=60 | 0.6239 | -0.0021 |
| E. starts<8 g=0.35 cap=40 | 0.6239 | -0.0020 |
| H. changed OR starts<17 g=0.35 cap=20 | 0.6240 | -0.0020 |
| E. starts<8 g=0.50 cap=20 | 0.6241 | -0.0019 |
| H. changed OR starts<17 g=0.20 cap=40 | 0.6241 | -0.0018 |
| F. starts<17 g=0.35 cap=20 | 0.6242 | -0.0018 |
| H. changed OR starts<17 g=0.20 cap=60 | 0.6242 | -0.0017 |
| F. starts<17 g=0.20 cap=60 | 0.6242 | -0.0017 |
| D. starts<4 g=0.75 cap=20 | 0.6242 | -0.0017 |
| F. starts<17 g=0.20 cap=40 | 0.6243 | -0.0017 |
| G. changed OR starts<8 g=0.35 cap=20 | 0.6243 | -0.0016 |
| D. starts<4 g=0.50 cap=40 | 0.6243 | -0.0016 |
| G. changed OR starts<8 g=0.20 cap=40 | 0.6245 | -0.0014 |
| G. changed OR starts<8 g=0.20 cap=60 | 0.6246 | -0.0014 |
| D. starts<4 g=0.75 cap=60 | 0.6246 | -0.0014 |
| E. starts<8 g=0.35 cap=20 | 0.6246 | -0.0013 |
| E. starts<8 g=0.20 cap=60 | 0.6246 | -0.0013 |
| D. starts<4 cap=60 | 0.6247 | -0.0013 |
| E. starts<8 g=0.20 cap=40 | 0.6247 | -0.0013 |
| D. starts<4 g=0.50 cap=20 | 0.6247 | -0.0012 |
| D. starts<4 g=0.35 cap=40 | 0.6247 | -0.0012 |
| D. starts<4 g=0.50 cap=60 | 0.6247 | -0.0012 |
| H. changed OR starts<17 g=0.20 cap=20 | 0.6248 | -0.0012 |
| F. starts<17 g=0.20 cap=20 | 0.6249 | -0.0011 |
| D. starts<4 g=0.35 cap=60 | 0.6250 | -0.0010 |
| G. changed OR starts<8 g=0.20 cap=20 | 0.6250 | -0.0010 |
| H. changed OR starts<17 g=0.10 cap=40 | 0.6250 | -0.0009 |
| H. changed OR starts<17 g=0.10 cap=60 | 0.6250 | -0.0009 |
| F. starts<17 g=0.10 cap=60 | 0.6250 | -0.0009 |
| B. qb_changed cap=20 | 0.6250 | -0.0009 |
| C. qb_differs_prev cap=20 | 0.6250 | -0.0009 |
| F. starts<17 g=0.10 cap=40 | 0.6251 | -0.0009 |
| D. starts<4 g=0.35 cap=20 | 0.6251 | -0.0009 |
| E. starts<8 g=0.20 cap=20 | 0.6251 | -0.0008 |
| G. changed OR starts<8 g=0.10 cap=40 | 0.6252 | -0.0008 |
| B. qb_changed g=0.75 cap=20 | 0.6252 | -0.0007 |
| C. qb_differs_prev g=0.75 cap=20 | 0.6252 | -0.0007 |
| D. starts<4 g=0.20 cap=40 | 0.6252 | -0.0007 |
| G. changed OR starts<8 g=0.10 cap=60 | 0.6252 | -0.0007 |
| E. starts<8 g=0.10 cap=60 | 0.6252 | -0.0007 |
| E. starts<8 g=0.10 cap=40 | 0.6253 | -0.0007 |
| D. starts<4 g=0.20 cap=60 | 0.6253 | -0.0006 |
| H. changed OR starts<17 g=0.10 cap=20 | 0.6253 | -0.0006 |
| B. qb_changed g=0.50 cap=20 | 0.6254 | -0.0005 |
| C. qb_differs_prev g=0.50 cap=20 | 0.6254 | -0.0005 |
| F. starts<17 g=0.10 cap=20 | 0.6254 | -0.0005 |
| D. starts<4 g=0.20 cap=20 | 0.6254 | -0.0005 |
| B. qb_changed g=0.75 cap=40 | 0.6255 | -0.0005 |
| C. qb_differs_prev g=0.75 cap=40 | 0.6255 | -0.0005 |
| G. changed OR starts<8 g=0.10 cap=20 | 0.6255 | -0.0005 |
| B. qb_changed cap=40 | 0.6255 | -0.0004 |
| C. qb_differs_prev cap=40 | 0.6255 | -0.0004 |
| B. qb_changed g=0.50 cap=40 | 0.6255 | -0.0004 |
| C. qb_differs_prev g=0.50 cap=40 | 0.6255 | -0.0004 |
| E. starts<8 g=0.10 cap=20 | 0.6255 | -0.0004 |
| B. qb_changed g=0.35 cap=20 | 0.6255 | -0.0004 |
| C. qb_differs_prev g=0.35 cap=20 | 0.6255 | -0.0004 |
| D. starts<4 g=0.10 cap=40 | 0.6256 | -0.0004 |
| B. qb_changed g=0.35 cap=40 | 0.6256 | -0.0004 |
| C. qb_differs_prev g=0.35 cap=40 | 0.6256 | -0.0004 |
| D. starts<4 g=0.10 cap=60 | 0.6256 | -0.0003 |
| D. starts<4 g=0.10 cap=20 | 0.6257 | -0.0003 |
| B. qb_changed g=0.20 cap=20 | 0.6257 | -0.0002 |
| C. qb_differs_prev g=0.20 cap=20 | 0.6257 | -0.0002 |
| B. qb_changed g=0.20 cap=40 | 0.6257 | -0.0002 |
| C. qb_differs_prev g=0.20 cap=40 | 0.6257 | -0.0002 |
| B. qb_changed g=0.10 cap=40 | 0.6258 | -0.0001 |
| C. qb_differs_prev g=0.10 cap=40 | 0.6258 | -0.0001 |
| B. qb_changed g=0.10 cap=20 | 0.6258 | -0.0001 |
| C. qb_differs_prev g=0.10 cap=20 | 0.6258 | -0.0001 |
| A. Incumbent baseline | 0.6259 | +0.0000 |
| B. qb_changed g=0.10 cap=60 | 0.6260 | +0.0000 |
| C. qb_differs_prev g=0.10 cap=60 | 0.6260 | +0.0000 |
| B. qb_changed g=0.20 cap=60 | 0.6260 | +0.0001 |
| C. qb_differs_prev g=0.20 cap=60 | 0.6260 | +0.0001 |
| B. qb_changed g=0.35 cap=60 | 0.6261 | +0.0002 |
| C. qb_differs_prev g=0.35 cap=60 | 0.6261 | +0.0002 |
| B. qb_changed g=0.50 cap=60 | 0.6263 | +0.0003 |
| C. qb_differs_prev g=0.50 cap=60 | 0.6263 | +0.0003 |
| B. qb_changed g=0.75 cap=60 | 0.6266 | +0.0007 |
| C. qb_differs_prev g=0.75 cap=60 | 0.6266 | +0.0007 |
| B. qb_changed cap=60 | 0.6271 | +0.0012 |
| C. qb_differs_prev cap=60 | 0.6271 | +0.0012 |

## Decision

**✅ PROMOTED: H. changed OR starts<17 cap=40**

| Criterion | Met? |
|-----------|------|
| Beats incumbent on val LL (0.6305 < 0.6341) | ✅ |
| Beats incumbent on holdout LL (0.6200 < 0.6259) | ✅ |
| Non-gated equality passes | ✅ |
| Improves QB-change slice (0.6674 < 0.6696) | ✅ |

### Validation-Selected Best Variant

**H. changed OR starts<17 cap=40** — val LL 0.6305, holdout LL 0.6200

### Best-Holdout Diagnostic Variant

**H. changed OR starts<17 cap=40** — holdout LL 0.6200 (selected via validation)

### Cleanest-Gate Diagnostic Variant

**B. qb_changed cap=20** — holdout LL 0.6250 (only applies overlay to QB-change games, NoQC Δ = 0.00, diagnostic only)

### Failure Modes

1. **Oracle QB data**: Uses final actual starter IDs, not pregame-announced.
2. **Binary gate sharpness**: The gate is all-or-nothing per game.
3. **Small-sample QB adjustments**: QBs with <17 starts are strongly shrunk toward replacement.
4. **No position-group interaction**: Ignores OL, skill players, defense.

### Recommended Next Experiment

With this promotion, the new football-only incumbent becomes:

- **H. changed OR starts<17 cap=40** (val LL 0.6305, holdout LL 0.6200)
- Feature set: Standard Elo + qb_changed + rolling_mov_3 + Platt + frozen QB overlay (gated on changed OR starts<17, cap=40)
- Non-gated games match the previous incumbent exactly

Future models must beat **this** incumbent to promote.

1. **Coach-QB interaction features**: New QB + new coordinator may be more informative than QB change alone.
2. **Expanded Elo K search**: Test K > 48 with season regression spine.
3. **DVOA/EPA features**: If a new pregame-safe data source becomes available.
4. **Raw QB adjustment feature** (from qb_adjustment.py) remains diagnostic-only. Use in residual diagnostics, not as a standalone promoted feature.

---
*Report generated by `sportslab frozen-qb-overlay-foldsafe`. Seasons: 2021–2025, Folds: 3, Variants: 127.*
