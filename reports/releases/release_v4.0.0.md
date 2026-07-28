# Release v4.0.0 — Pi-Ratings Champion

- **Date:** 2026-07-28
- **Model:** Pi-Ratings (α=0.5, base_k=28, hk_ratio=1.25, HFA=30, reg=0.0) + Platt(qb_changed, mov_3) + QB overlay
- **Label:** Pi-Ratings Frozen QB Overlay

## Key Innovation

Standard Elo has point-differential MOV scaled by a capped-linear multiplier. Pi-Ratings replaces this with:
1. **Power-law MOV**: `|margin|^0.5` — square root compresses blowouts, reducing overreaction to lopsided scores
2. **Asymmetric K**: home K = 28 × 1.25 = 35, away K = 28 × 0.75 = 21 — home teams update faster, reflecting home-field information advantage

When α=1.0 and hk_ratio=1.0, this reduces to standard capped_linear Elo.

## Metrics

| Metric | Value |
|--------|-------|
| Avg validation log loss | 0.6260 |
| 2025 holdout log loss | **0.5918** |
| Δ vs v3.0.0 (holdout) | −0.0022 |
| Δ vs v3.0.0 (validation) | −0.0046 |

## Feature Set

| Feature | Source |
|---------|--------|
| Pi-Rating probability | `compute_pi_ratings_features()` |
| home_qb_changed | `compute_qb_features()` |
| away_qb_changed | `compute_qb_features()` |
| home_rolling_mov_3 | `compute_situational_features()` |
| away_rolling_mov_3 | `compute_situational_features()` |
| QB overlay | Gate: changed OR starts<17, cap=40, gamma=1.0 |

## Artifacts

- **Experiment report:** `reports/experiments/pi_ratings_champion_comparison.md`
- **144-combo grid:** `reports/experiments/pi_ratings.md`
- **Benchmark leaderboard:** `reports/benchmarks/leaderboard.csv` (row 51)
- **Benchmark history:** `reports/benchmarks/benchmark_history.md` (entry 48)
- **Incumbent card:** `reports/benchmarks/nfl_research_incumbent.md`
- **Pi-Ratings feature function:** `src/sportslab/features/ratings.py` — `compute_pi_ratings_features()`
- **Experiment module:** `src/sportslab/evaluation/pi_ratings_experiment.py`
- **Comparison module:** `src/sportslab/evaluation/pi_ratings_champion_comparison.py`
- **Tests:** `tests/test_pi_ratings.py`, `tests/test_pi_ratings_champion_comparison.py`
- **Prediction vintages:** `src/sportslab/evaluation/prediction_vintages.py`

## Live-Ops Commands

```bash
# Run champion comparison (verify metrics)
sportslab pi-ratings-compare

# Run full grid (144 combos, ~3 min)
sportslab pi-ratings

# Weekly prediction with vintage tracking
sportslab predict-week --season 2026 --week 1 --vintage locked

# Compare vintages
sportslab compare-vintages --season 2026 --week 1
```

## Validation Commands

```bash
make test    # 1119+ tests expected
make lint    # clean
make check   # test + lint
```

## Known Limitations

1. **Holdout LL (0.5918) is a re-computed comparison baseline** — differs from registry v3.0.0 (0.6200) due to pipeline differences. Relative Δ (−0.0022) is the reliable metric.
2. **StatSpace PBP composites** (FDR, DOBA, Chaos at 0.5548) have NOT been tested on Pi-Ratings base — they currently sit on standard Elo
3. **Pi-Ratings is re-fit per fold** in the Platt pipeline — single-shot fit may differ slightly
4. **α=0.5 is the only power-law value tested** — values between 0.5 and 0.75 not explored
5. **Pre-season regression set to 0.0** — may need tuning for live weekly updates

## Previous Champions

| Version | Feature | Holdout LL | Replaced By |
|---------|---------|-----------|-------------|
| v4.0.0 | Pi-Ratings + QB overlay | 0.5918 | (current) |
| v3.0.0 | Frozen QB Overlay | 0.6200 | Pi-Ratings v4 |
| v2.0.0 | qb_changed + mov_3 + Platt | 0.6262 | Frozen QB Overlay |
| v1.0.0 | Season-reg Elo + Platt | 0.6285 | qb_changed + mov_3 |

## CI Status (at release)

- Tests: 1119+ passed, 1 skipped
- Lint: clean (ruff)
- All experiment modules importable
- Benchmark registry validated

## Proposed Git Tag

```bash
git tag -a v4.0.0 -m "Pi-Ratings champion (α=0.5, base_k=28, hk_ratio=1.25)"
```
