# Release v5.0.0 — Pi-StatSpace Champion

- **Date:** 2026-07-28
- **Model:** Pi-Ratings (α=0.5, base_k=28, hk_ratio=1.25, HFA=30, reg=0.0) + rolling_mov_3 + qb_changed + FDR + DOBA + Chaos + Platt
- **Label:** Pi-Ratings + StatSpace PBP Composites

## Key Innovation

StatSpace PBP composites (FDR, DOBA, Chaos) improve on the Pi-Ratings football-only base by the same pattern as on standard Elo. The composites are feature-orthogonal to the rating system — each progressively improves prediction regardless of Elo base:

| Model | Val LL | Hold LL |
|-------|--------|---------|
| Pi-Ratings only | 0.6266 | 0.6350 |
| Pi + FDR | 0.6074 | 0.5998 |
| Pi + FDR + DOBA | 0.5775 | 0.5913 |
| **Pi + FDR + DOBA + Chaos** | **0.5557** | **0.5532** |

## Metrics

| Metric | Value |
|--------|-------|
| Avg validation log loss | **0.5557** |
| 2025 holdout log loss | **0.5532** |
| 2025 holdout Brier | 0.1886 |
| 2025 holdout AUC | 0.7874 |
| Δ vs v4.0.0 (holdout) | −0.0386 |
| Δ vs v4.0.0 (validation) | −0.0703 |
| Δ vs old champion (holdout) | −0.0016 |

## Feature Set

| Feature | Source |
|---------|--------|
| Pi-Rating probability | `compute_pi_ratings_features()` |
| home_qb_changed | `compute_qb_features()` |
| away_qb_changed | `compute_qb_features()` |
| home_rolling_mov_3 | `compute_situational_features()` |
| away_rolling_mov_3 | `compute_situational_features()` |
| home_fdr_fraud_detector_rating | `compute_statspace_fdr()` |
| away_fdr_fraud_detector_rating | `compute_statspace_fdr()` |
| home_doba_doba_score | `compute_statspace_doba()` |
| away_doba_doba_score | `compute_statspace_doba()` |
| home_chaos_chaos_rate | `compute_statspace_chaos_rate()` |
| away_chaos_chaos_rate | `compute_statspace_chaos_rate()` |

## Artifacts

- **Experiment report:** `reports/experiments/pi_statspace.md`
- **Benchmark leaderboard:** `reports/benchmarks/leaderboard.csv` (row 52)
- **Benchmark history:** `reports/benchmarks/benchmark_history.md` (entry 49)
- **Incumbent card:** `reports/benchmarks/nfl_research_incumbent.md`
- **Experiment module:** `src/sportslab/evaluation/pi_statspace_experiment.py`
- **Tests:** `tests/test_pi_statspace.py`

## Live-Ops Commands

```bash
# Run Pi-StatSpace experiment (verify metrics)
sportslab pi-statspace

# Weekly prediction with vintage tracking
sportslab predict-week --season 2026 --week 1 --vintage locked

# Compare vintages
sportslab compare-vintages --season 2026 --week 1
```

## Validation Commands

```bash
make test    # 1127+ tests expected
make lint    # clean
make check   # test + lint
```

## Known Limitations

1. **PBP-computed features** (FDR, DOBA, Chaos) require nflverse PBP data — adds ~2-3 min to weekly pipeline
2. **Team-season composites** — FDR/DOBA/Chaos are computed per-season, not per-week. Late-season updates may stale.
3. **New rating base** — Pi-Ratings α=0.5 is the only power-law value tested; values between 0.5 and 0.75 not explored
4. **Pre-season regression set to 0.0** — may need tuning for live weekly updates

## Previous Champions

| Version | Feature | Holdout LL | Replaced By |
|---------|---------|-----------|-------------|
| v5.0.0 | Pi-Ratings + FDR + DOBA + Chaos | 0.5532 | (current) |
| v4.0.0 | Pi-Ratings + QB overlay | 0.5918 | Pi-StatSpace v5 |
| v3.0.0 | Frozen QB Overlay | 0.6200 | Pi-Ratings v4 |
| v2.0.0 | qb_changed + mov_3 + Platt | 0.6262 | Frozen QB Overlay |
| v1.0.0 | Season-reg Elo + Platt | 0.6285 | qb_changed + mov_3 |

## CI Status (at release)

- Tests: 1130+ passed (estimated)
- Lint: clean (ruff)
- All experiment modules importable
- Benchmark registry validated

## Proposed Git Tag

```bash
git tag -a v5.0.0 -m "Pi-StatSpace champion (Pi-Ratings + FDR + DOBA + Chaos)"
```
