# Rolling StatSpace Experiment

Test whether rolling-window StatSpace composite metrics (DOBA + Chaos Rate from per-game PBP, z-scored, averaged over [3, 5]-game windows) improve on the incumbent.

| Model | Val LL | Fold1 | Fold2 | Fold3 | Hold LL | Brier | AUC | Acc | Feat |
|------|--------|-------|-------|-------|---------|-------|-----|-----|------|
| Platt (incumbent) | 0.6342 | 0.6416 | 0.6577 | 0.6031 | 0.6259 | 0.2181 | 0.705 | 0.659 | 4 |
| Incumbent + RS (w=3) | 0.6417 (Δ=+0.0075) | 0.6364 | 0.6872 | 0.6015 | 0.6254 | 0.2178 | 0.706 | 0.663 | 8 |
| Incumbent + RS (w=5) | 0.6412 (Δ=+0.0071) | 0.6436 | 0.6748 | 0.6053 | 0.6255 | 0.2184 | 0.702 | 0.652 | 8 |
| Incumbent + RS (w=3+5) | 0.6511 (Δ=+0.0169) | 0.6426 | 0.7030 | 0.6076 | 0.6280 | 0.2196 | 0.700 | 0.656 | 12 |
| RS only (w=3) | 0.6664 (Δ=+0.0322) | 0.6497 | 0.7007 | 0.6487 | 0.6501 | 0.2296 | 0.660 | 0.605 | 4 |

### Winners

- **Best val LL**: Platt (incumbent) (0.6342)
- **Best hold LL**: Incumbent + RS (w=3) (0.6254)

**No model beats incumbent on both val and holdout by ≥ 0.001.**

---
Report: rolling_statspace_experiment.py
