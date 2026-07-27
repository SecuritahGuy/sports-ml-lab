# Kalman-Filter Elo

Uncertainty-aware Elo ratings using Kalman gain (sigma² / total_var) instead of fixed K-factor update proportions.

**Grid:** 162 combos (K∈[32, 36, 40] × HFA∈[35, 40, 45] × init_sigma∈[200, 400] × obs_noise∈[50, 100, 200] × reg∈[0.05, 0.1, 0.2])

| Config | Raw Val LL | Fold1 | Fold2 | Fold3 | Raw Hold LL |
|--------|-----------|-------|-------|-------|-------------|
| Incumbent (Std Elo) | 0.6337 | 0.6344 | 0.6606 | 0.6063 | 0.6264 |
| K=40_HFA=40_is=400_on=50_reg=0.05 | 0.6438 | 0.6470 | 0.6578 | 0.6267 | 0.6422 |
| K=40_HFA=45_is=400_on=50_reg=0.05 | 0.6439 | 0.6467 | 0.6576 | 0.6275 | 0.6428 |
| K=40_HFA=35_is=400_on=50_reg=0.05 | 0.6440 | 0.6476 | 0.6582 | 0.6261 | 0.6418 |
| K=40_HFA=40_is=400_on=50_reg=0.1 | 0.6440 | 0.6469 | 0.6580 | 0.6272 | 0.6416 |
| K=40_HFA=45_is=400_on=50_reg=0.1 | 0.6441 | 0.6465 | 0.6578 | 0.6279 | 0.6422 |
| K=40_HFA=35_is=400_on=50_reg=0.1 | 0.6441 | 0.6474 | 0.6584 | 0.6266 | 0.6412 |
| K=40_HFA=40_is=400_on=50_reg=0.2 | 0.6444 | 0.6466 | 0.6586 | 0.6279 | 0.6407 |
| K=40_HFA=45_is=400_on=50_reg=0.2 | 0.6444 | 0.6463 | 0.6584 | 0.6287 | 0.6413 |
| K=40_HFA=35_is=400_on=50_reg=0.2 | 0.6445 | 0.6471 | 0.6590 | 0.6273 | 0.6402 |
| K=36_HFA=40_is=400_on=50_reg=0.05 | 0.6452 | 0.6480 | 0.6584 | 0.6292 | 0.6434 |

## Platt + Features + QB Overlay

| Model | Val LL | Fold1 | Fold2 | Fold3 | Hold LL | Brier | AUC | Acc |
|-------|--------|-------|-------|-------|---------|-------|-----|-----|
| Incumbent | 0.6303 | 0.6357 | 0.6615 | 0.5937 | 0.6218 | 0.2164 | 0.708 | 0.652 |
| Kalman Elo (best) | 0.6322 (Δ=+0.0019) | 0.6430 | 0.6577 | 0.5958 | 0.6321 (Δ=+0.0103) | 0.2212 | 0.695 | 0.649 |

**Best Kalman config:** K=40_HFA=40_is=400_on=50_reg=0.05

## ❌ NOT PROMOTED — Δval=+0.0019, Δhold=+0.0103

---
Report: kalman_elo_experiment.py
