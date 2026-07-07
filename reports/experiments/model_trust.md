# Model Trust Diagnostics Report

*Generated: 2026-07-07 09:42*
*Incumbent: v3.0.0*

---

## 1. Incumbent Reproduction

| Metric | Value | Documented | Match |
|---|---|---|---|
| Holdout LL | 0.6200 | 0.6200 | ✅ PASS |
| Holdout Brier | 0.2157 | — | — |
| Holdout Accuracy | 0.6630 | — | — |
| Holdout AUC | 0.7098 | — | — |
| Holdout ECE | 0.062757 | — | — |
| Holdout N | 276 | — | — |

**All games (1388):** LL=0.6355, Brier=0.2227, Acc=0.6455, AUC=0.6889

**Season breakdown:**
| Season | N | Log Loss | Brier | Acc | AUC |
|---|---|---|---|---|---|
| 2021 | 280 | 0.6833 | 0.244 | 0.5893 | 0.6292 |
| 2022 | 275 | 0.6275 | 0.2197 | 0.6436 | 0.6901 |
| 2023 | 279 | 0.6556 | 0.231 | 0.6487 | 0.6644 |
| 2024 | 278 | 0.5903 | 0.2027 | 0.6835 | 0.7522 |
| 2025 | 276 | 0.62 | 0.2157 | 0.663 | 0.7098 |

---

## 2. Failure-Mode Splits

*Total games analyzed: 1388*

### Qb Changed

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| No QB change (n=1109, 79.9%) | 1109 | 0.6431 | 0.2260 | 0.6402 | 0.5311 | 0.5329 | 0.0461 |
| QB changed (n=279, 20.1%) | 279 | 0.6052 | 0.2093 | 0.6667 | 0.6121 | 0.5950 | 0.0655 |

### Roof Type

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Dome (n=256, 18.4%) | 256 | 0.6675 | 0.2358 | 0.6055 | 0.5552 | 0.5469 | 0.0516 |
| Outdoor (n=921, 66.4%) | 921 | 0.6289 | 0.2199 | 0.6558 | 0.5508 | 0.5592 | 0.0452 |
| Retractable/open (n=32, 2.3%) | 32 | 0.7353 | 0.2692 | 0.5625 | 0.5065 | 0.4688 | 0.2141 |

### Rest Advantage

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Away rest advantage (n=257, 18.5%) | 257 | 0.6340 | 0.2219 | 0.6576 | 0.5544 | 0.5603 | 0.0500 |
| Equal rest (n=867, 62.5%) | 867 | 0.6433 | 0.2261 | 0.6344 | 0.5405 | 0.5340 | 0.0414 |
| Home rest advantage (n=264, 19.0%) | 264 | 0.6110 | 0.2121 | 0.6705 | 0.5633 | 0.5682 | 0.0749 |

### Short Week

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Normal rest (n=1070, 77.1%) | 1070 | 0.6416 | 0.2256 | 0.6421 | 0.5477 | 0.5346 | 0.0523 |
| Short week (n=318, 22.9%) | 318 | 0.6147 | 0.2127 | 0.6572 | 0.5464 | 0.5818 | 0.0675 |

### Elo Gap

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Close Elo gap (<=50) (n=617, 44.5%) | 617 | 0.6644 | 0.2363 | 0.5932 | 0.5484 | 0.5365 | 0.0399 |
| Large Elo gap (>50) (n=771, 55.5%) | 771 | 0.6123 | 0.2117 | 0.6874 | 0.5466 | 0.5525 | 0.0513 |

### Home Status

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Home favorite (n=837, 60.3%) | 837 | 0.6130 | 0.2128 | 0.6583 | 0.6675 | 0.6583 | 0.0273 |
| Home underdog (n=551, 39.7%) | 551 | 0.6696 | 0.2376 | 0.6261 | 0.3650 | 0.3739 | 0.0683 |

### Road Status

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Road favorite (n=551, 39.7%) | 551 | 0.6696 | 0.2376 | 0.6261 | 0.3650 | 0.3739 | 0.0683 |
| Road underdog (n=837, 60.3%) | 837 | 0.6130 | 0.2128 | 0.6583 | 0.6675 | 0.6583 | 0.0273 |

### Missing Weather

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Missing weather data (n=574, 41.4%) | 574 | 0.6497 | 0.2288 | 0.6307 | 0.5389 | 0.5192 | 0.0562 |
| Weather data present (n=814, 58.6%) | 814 | 0.6254 | 0.2183 | 0.6560 | 0.5534 | 0.5639 | 0.0382 |

### Missing Qb Metadata

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Missing QB metadata (n=0, 0.0%) | 0 | — | — | — | — | — | — |
| QB metadata present (n=1388, 100.0%) | 1388 | 0.6355 | 0.2227 | 0.6455 | 0.5474 | 0.5454 | 0.0436 |

### Season Phase

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Early season (weeks 1-4) (n=312, 22.5%) | 312 | 0.6727 | 0.2391 | 0.5962 | 0.5371 | 0.5160 | 0.0461 |
| Late season (weeks 13+) (n=461, 33.2%) | 461 | 0.6032 | 0.2087 | 0.6855 | 0.5502 | 0.5531 | 0.0563 |
| Mid season (weeks 5-12) (n=556, 40.1%) | 556 | 0.6469 | 0.2277 | 0.6331 | 0.5423 | 0.5414 | 0.0449 |
| Playoffs (n=59, 4.3%) | 59 | 0.5825 | 0.1973 | 0.7119 | 0.6289 | 0.6780 | 0.0848 |

### Neutral Site

| Split | N | Log Loss | Brier | Acc | Avg Prob | Act Win% | Calib |
|------|---|---|---|---|---|---|---|
| Neutral site (n=0, 0.0%) | 0 | — | — | — | — | — | — |
| Non-neutral (n=1388, 100.0%) | 1388 | 0.6355 | 0.2227 | 0.6455 | 0.5474 | 0.5454 | 0.0436 |

---

## 3. Market Benchmark Comparison

| Model | N | Log Loss | Brier | Acc | AUC |
|---|---|---|---|---|---|
| Incumbent | 1388 | 0.6355 | 0.2227 | 0.6455 | 0.6889 |
| Market (no-vig) | 1388 | 0.6095 | 0.2111 | 0.665 | 0.7238 |

**Incumbent vs Market gap (log loss):** 0.026

**Per-season comparison:**
| Season | N | Incumbent LL | Market LL |
|---|---|---|---|
| 2021 | 280 | 0.6833 | 0.6223 |
| 2022 | 275 | 0.6275 | 0.6042 |
| 2023 | 279 | 0.6556 | 0.6258 |
| 2024 | 278 | 0.5903 | 0.5858 |
| 2025 | 276 | 0.62 | 0.609 |

**Per week-bucket comparison:**
| Bucket | N | Incumbent LL | Market LL |
|---|---|---|---|
| Early (1-4) | 312 | 0.6727 | 0.6361 |
| Late (13+) | 520 | 0.6009 | 0.5812 |
| Mid (5-12) | 556 | 0.6469 | 0.621 |

---

## 4. High-Confidence Analysis

| Threshold | N (eith) | LL (eith) | Acc (eith) | N (home) | Home win% | Home avg prob |
|---|---|---|---|---|---|---|
| p ≥ 0.70 | 448 | 0.5685 | 0.7321 | 302 | 0.7848 | 0.7877 |
| p ≥ 0.75 | 269 | 0.5175 | 0.777 | 183 | 0.8087 | 0.8296 |
| p ≥ 0.80 | 153 | 0.4198 | 0.8497 | 115 | 0.8783 | 0.8619 |
| p ≥ 0.85 | 78 | 0.418 | 0.8462 | 66 | 0.8485 | 0.8899 |
| p ≥ 0.90 | 22 | 0.1845 | 0.9545 | 22 | 0.9545 | 0.92 |

---

## 5. Reproducibility

**Status:** ✅ Deterministic
**Games compared:** 1388
**Columns:** 39
**Note:** Predictions are from a static CSV file — deterministic by construction.

---

## 6. Model Trust Thresholds

| Check | Value | Threshold | Status |
|---|---|---|---|
| ECE (holdout) | 0.062757 | < 0.10 | ✅ PASS |
| High-confidence acc (p≥0.90) | 0.9545 | ≥ 0.80 | ✅ PASS |
| High-confidence games (p≥0.90) | 22 | — | — |
| Market gap (LL) | 0.026 | ≤ 0.05 | ✅ PASS |

---

*Report generated by sportslab.evaluation.model_trust*
*No network access required*
