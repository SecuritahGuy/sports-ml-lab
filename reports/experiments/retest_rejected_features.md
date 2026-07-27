# Re-test: Weather, Scheduling, Injury Features

Original tests used an older Elo spine (K=40, reg=0.25, no MOV, no season regression, no QB overlay). Re-running on the current incumbent spine:
- K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2
- 3-fold rolling-origin + 2025 holdout
- All models include QB overlay in logit space

| Model | Val LL | Fold1 | Fold2 | Fold3 | Hold LL | Brier | AUC | Acc | Feat |
|-------|--------|-------|-------|-------|---------|-------|-----|-----|------|
| Incumbent (Platt) | 0.6305 | 0.6360 | 0.6596 | 0.5960 | 0.6200 | 0.2157 | 0.710 | 0.663 | 0 |
| Incumbent + Weather | 0.6634 (Δ=+0.0329) | 0.7135 | 0.6722 | 0.6044 | 0.6298 | 0.2202 | 0.697 | 0.641 | 14 |
| Incumbent + Scheduling | 0.6513 (Δ=+0.0208) | 0.6701 | 0.6680 | 0.6159 | 0.6220 | 0.2164 | 0.708 | 0.645 | 10 |
| Incumbent + Injury | 0.6495 (Δ=+0.0190) | 0.6756 | 0.6712 | 0.6016 | 0.6403 | 0.2205 | 0.704 | 0.652 | 29 |
| Incumbent + All three | 0.7156 (Δ=+0.0851) | 0.8254 | 0.6913 | 0.6302 | 0.6519 | 0.2254 | 0.690 | 0.641 | 53 |
| Weather only (no Elo) | 0.7193 (Δ=+0.0888) | 0.7536 | 0.7025 | 0.7018 | 0.7024 | 0.2543 | 0.467 | 0.536 | 14 |

**No model beats incumbent on both val and holdout by ≥ 0.001.**

---
Report: retest_rejected_features.py
