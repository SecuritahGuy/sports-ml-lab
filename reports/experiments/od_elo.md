# Separate Offensive/Defensive Elo Experiment

*Testing whether independent O/D Elo with different k_off/k_def improves on standard Elo.*

## Method

Each team maintains independent off_elo and def_elo (both start at 1500).
For prediction, ratings are combined: total = off + def (same as standard).
For updates, k_off and k_def can differ. A lopsided win with k_off > k_def
produces a larger total rating update (offense gets extra credit for the blowout).

## Grid

k_off ∈ [20, 28, 36, 44, 52], k_def ∈ [20, 28, 36, 44, 52] (9 combos, excluding k_off=k_def=36 as duplicate)

Other params: K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2
MOV: capped_linear, scale=0.05, cap=2.0

## Rolling-Origin Validation

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 |
|-------|------------|-------|-------|-------|
| Standard (incumbent) | 0.6368 | 0.6425 | 0.6576 | 0.6103 |
| O/D ko20_kd44 | 0.6365 | 0.6429 | 0.6583 | 0.6083 |
| O/D ko20_kd52 | 0.6370 | 0.6431 | 0.6590 | 0.6087 |
| O/D ko28_kd36 | 0.6364 | 0.6426 | 0.6575 | 0.6093 |
| O/D ko28_kd44 | 0.6368 | 0.6427 | 0.6583 | 0.6094 |
| O/D ko28_kd52 | 0.6372 | 0.6429 | 0.6591 | 0.6097 |
| O/D ko36_kd28 | 0.6367 | 0.6425 | 0.6568 | 0.6106 |
| O/D ko36_kd36 | 0.6368 | 0.6425 | 0.6576 | 0.6103 |
| O/D ko36_kd44 | 0.6372 | 0.6427 | 0.6584 | 0.6104 |
| O/D ko44_kd20 | 0.6371 | 0.6428 | 0.6563 | 0.6123 |
| O/D ko44_kd28 | 0.6371 | 0.6427 | 0.6571 | 0.6116 |
| O/D ko44_kd36 | 0.6373 | 0.6427 | 0.6579 | 0.6114 |
| O/D ko52_kd20 | 0.6376 | 0.6430 | 0.6567 | 0.6132 |
| O/D ko52_kd28 | 0.6377 | 0.6429 | 0.6574 | 0.6126 |
| O/D ko20_kd20 | 0.6361 | 0.6432 | 0.6562 | 0.6089 |
| O/D ko52_kd52 | 0.6386 | 0.6435 | 0.6599 | 0.6125 |

## 2025 Holdout

| Model | Hold LL | Hold Brier | Hold Acc | Hold AUC |
|-------|---------|------------|----------|----------|
| Random | 0.6931 | 0.2500 | 0.5000 | 0.5000 |
| Home prior (0.548) | 0.6908 | — | — | 0.5000 |
| Standard (incumbent) | 0.6285 | 0.2191 | 0.6667 | 0.7024 |
| O/D ko20_kd44 | 0.6321 | 0.2206 | 0.6558 | 0.6971 |
| O/D ko20_kd52 | 0.6319 | 0.2206 | 0.6594 | 0.6978 |
| O/D ko28_kd36 | 0.6303 | 0.2198 | 0.6630 | 0.7000 |
| O/D ko28_kd44 | 0.6301 | 0.2198 | 0.6630 | 0.7000 |
| O/D ko28_kd52 | 0.6302 | 0.2198 | 0.6594 | 0.7000 |
| O/D ko36_kd28 | 0.6286 | 0.2191 | 0.6703 | 0.7032 |
| O/D ko36_kd36 | 0.6285 | 0.2191 | 0.6667 | 0.7024 |
| O/D ko36_kd44 | 0.6286 | 0.2191 | 0.6594 | 0.7025 |
| O/D ko44_kd20 | 0.6271 | 0.2185 | 0.6630 | 0.7051 |
| O/D ko44_kd28 | 0.6271 | 0.2185 | 0.6667 | 0.7047 |
| O/D ko44_kd36 | 0.6272 | 0.2185 | 0.6703 | 0.7047 |
| O/D ko52_kd20 | 0.6258 | 0.2179 | 0.6703 | 0.7066 |
| O/D ko52_kd28 | 0.6259 | 0.2179 | 0.6667 | 0.7056 |
| O/D ko20_kd20 | 0.6337 | 0.2213 | 0.6667 | 0.6960 |
| O/D ko52_kd52 | 0.6269 | 0.2184 | 0.6594 | 0.7046 |

**Standard Elo remains the research incumbent.** No O/D Elo variant beat it on both val and holdout.
