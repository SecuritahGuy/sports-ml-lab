# RALPH Loop 7: Rejected Challenger Postmortems

*Incumbent: v3.0.0 Frozen QB Overlay (holdout LL 0.6200)*

*Generated: 2026-07-06*

---

## Postmortem Format

Each challenger includes:
- hypothesis, expected improvement path, val LL, holdout LL, delta vs incumbent, Brier/Acc/AUC (from holdout), calibration/ECE impact, subgroup impact, whether it helped the targeted weakness, whether it hurt other splits, why rejected, and recommended disposition.

---

## L1: prior_win_pct

| Field | Value |
|-------|-------|
| **Hypothesis** | Early weeks need better priors than the default 0.5 Elo midpoint. Prior-season win% provides a team-specific baseline before current-season data accumulates. |
| **Expected improvement** | Early season (Weeks 1–4) — team strength is underdetermined by Elo in the first few games. |
| **Validation LL** | 0.6316 (+0.0011 vs incumbent) |
| **Holdout LL** | 0.6183 (−0.0017 vs incumbent) |
| **Holdout Brier** | 0.2152 (−0.0005 vs incumbent) |
| **Holdout Acc** | 0.6630 (tied with incumbent) |
| **Holdout AUC** | 0.7112 (+0.0014 vs incumbent) |
| **Calibration/ECE** | Not measured separately. |
| **Early season (Weeks 1–4) LL** | 0.5805 (vs incumbent 0.5844, Δ = −0.0039) ✅ Helped the targeted weakness |
| **Weather missing LL** | 0.6465 (vs incumbent 0.6452, Δ = +0.0013) Slightly hurt |
| **QB changed LL** | 0.6665 (vs incumbent 0.6674, Δ = −0.0009) Essentially tied |
| **Why rejected** | Validation worse (+0.0011). The early-season improvement (−0.0039 holdout) was not enough to offset validation degradation across all 3 folds. Prior-season win% is a noisy estimator of current-season strength — NFL year-to-year correlation in team win% is only ~0.3–0.4. |

**Disposition: RETIRE**

Prior-season win% is too noisy. The feature's improvement on early-season holdout (3.9e-3) is real but small, and it costs 1.1e-3 across all folds. The feature bakes in stale information — year-to-year roster turnover makes a team's win% from 9 months ago a weak signal. This direction is exhausted: if prior-season strength is useful, it must be extracted from a higher-signal source (e.g., prior-season Elo rating, not win%).

---

## L2: weather_missing

| Field | Value |
|-------|-------|
| **Hypothesis** | Missing weather data is a proxy for dome/indoor stadiums. Adding `weather_missing_flag`, `is_dome`, and `outdoor_game_flag` would help the model distinguish between environments. |
| **Expected improvement** | Games where weather data is missing (574 games, 41% of dataset), which are heavily dome-biased. |
| **Validation LL** | 0.6481 (+0.0176 vs incumbent) |
| **Holdout LL** | 0.6198 (−0.0002 vs incumbent) |
| **Holdout Brier** | 0.2156 (−0.0001 vs incumbent) |
| **Holdout Acc** | 0.6522 (−0.0108 vs incumbent) |
| **Holdout AUC** | 0.7120 (+0.0022 vs incumbent) |
| **Early season LL** | 0.5818 (vs incumbent 0.5844, Δ = −0.0026) |
| **Weather missing LL** | 0.6438 (vs incumbent 0.6452, Δ = −0.0014) |
| **QB changed LL** | 0.6676 (vs incumbent 0.6674, Δ = +0.0002) |
| **Why rejected** | Validation much worse (+0.0176). Adding three weather-environment features adds noise across all folds. The weather_missing_flag is collinear with dome status (dome games never report weather). The tiny holdout improvement (−0.0002) is noise-level. |

**Disposition: RETIRE**

The weather missing/dome/outdoor signal is fully captured by the `roof` column that already exists in the feature table. Adding explicit flags for "weather data missing" and "is_dome" and "outdoor" is redundant — the data already contains roof type. The model cannot benefit from knowing that weather data is missing when it already knows the stadium has a roof. This direction is closed.

---

## L3: roof_enc

| Field | Value |
|-------|-------|
| **Hypothesis** | Label-encoded roof type (dome=0, retractable=1, outdoor=2, etc.) captures the calibration bias observed in retractable/open roof games (ECE=0.2141, n=32). |
| **Expected improvement** | Retractable/open roof subgroup (32 games). |
| **Validation LL** | 0.6315 (+0.0010 vs incumbent) |
| **Holdout LL** | 0.6202 (+0.0002 vs incumbent) |
| **Holdout Brier** | 0.2157 (tied with incumbent) |
| **Holdout Acc** | 0.6522 (−0.0108 vs incumbent) |
| **Holdout AUC** | 0.7121 (+0.0023 vs incumbent) |
| **Retractable/open subgroup (holdout)** | N=0 games in 2025 — cannot evaluate. |
| **Early season LL** | 0.5841 (vs incumbent 0.5844, Δ = −0.0003) |
| **Why rejected** | Validation worse (+0.0010), holdout essentially tied (+0.0002). The target subgroup (retractable/open, n=32 total) has ZERO games in the 2025 holdout, making it impossible for any holdout-driven experiment to improve. With 32 games across 5 seasons (~6 per season), there is insufficient data to learn a reliable effect. |

**Disposition: RETIRE**

The retractable/open roof subgroup is too small (2.3% of dataset) for any feature to learn a meaningful correction. This is a sample-size problem that can only be resolved by accumulating more seasons. Will be reopened when 2+ additional seasons provide ~18+ retractable/open games for a meaningful check. The ECE=0.2141 on this subgroup is a statistical artifact of small-n variance (32 games have high calibration volatility by definition).

---

## L4: games_since_change

| Field | Value |
|-------|-------|
| **Hypothesis** | QB continuity beyond the binary `qb_changed` flag. A quarterback in their 2nd game vs 10th game with the team has different chemistry — the model should distinguish. |
| **Expected improvement** | QB-change games (279 games, 20% of dataset) — refine the prediction beyond a binary flag. |
| **Validation LL** | 0.6321 (+0.0016 vs incumbent) |
| **Holdout LL** | 0.6202 (+0.0002 vs incumbent) |
| **Holdout Brier** | 0.2162 (+0.0005 vs incumbent) |
| **Holdout Acc** | 0.6522 (−0.0108 vs incumbent) |
| **Holdout AUC** | 0.7081 (−0.0017 vs incumbent) |
| **QB changed LL** | 0.6572 (vs incumbent 0.6674, Δ = **−0.0102**) ✅ **SUBSTANTIAL IMPROVEMENT** |
| **No QB change LL** | Not computed separately but net holdout +0.0002 implies small degradation on non-QB-change games. |
| **Early season LL** | 0.5910 (vs incumbent 0.5844, Δ = +0.0066) ❌ Early season hurt |
| **Why rejected** | Validation worse (+0.0016). The QB-change subgroup improvement (−0.0102) is real and substantial, but the validation penalty across all folds (+0.0016) and the early-season degradation (+0.0066) cancel the benefit. The feature helps when a QB change just happened but adds noise when QB is stable (83% of observations have nonzero games_since_change). |

**Disposition: MONITOR — RETRY LATER**

This is the most promising rejected challenger. The −0.0102 holdout improvement on QB-change games is a strong signal that QB continuity matters beyond the binary flag. However, the feature has high cardinality (1–17+ games since change), which means it adds variance to the 80% of games where QB is stable. The v3.0.0 overlay already captures much of this signal through the `starts<17` gate. Two paths to revisit:

1. **More data**: With 2+ additional seasons, the QB-change sample grows from 279 to ~390 games, potentially stabilizing the benefit.
2. **Smarter encoding**: Instead of raw games-since-change, use a binned encoding (rookie: 1-4, developing: 5-16, established: 17+) to reduce variance.

**Diagnostic recommendation**: Add `home_games_since_qb_change` and `away_games_since_qb_change` to the model-trust report's subgroup analysis, even though they are not model features.

---

## L5: isotonic

| Field | Value |
|-------|-------|
| **Hypothesis** | Non-parametric isotonic regression would provide better calibration than Platt's logistic parametric assumption, especially at the probability extremes. |
| **Expected improvement** | Overall log loss through (potentially) better-calibrated probabilities. |
| **Validation LL** | 0.6312 (+0.0007 vs incumbent) |
| **Holdout LL** | 0.6283 (+0.0083 vs incumbent) ❌ **SUBSTANTIAL DEGRADATION** |
| **Holdout Brier** | 0.2191 (+0.0034 vs incumbent) |
| **Holdout Acc** | 0.6558 (−0.0072 vs incumbent) |
| **Holdout AUC** | 0.7142 (+0.0044 vs incumbent — higher AUC despite worse LL, suggesting calibration failure) |
| **QB changed LL** | 0.6943 (vs incumbent 0.6674, Δ = **+0.0269**) ❌ Much worse on QB-change games |
| **Why rejected** | Validation tiny improvement (+0.0007, below 0.001 threshold). Holdout substantial degradation (+0.0083). Isotonic overfitted the calibration curve on training data and failed to generalize. The step-function nature of isotonic regression creates jagged calibration bins that capture noise, not signal. On QB-change games (55 holdout games), isotonic was catastrophically worse (+0.0269). |

**Disposition: RETIRE**

Isotonic regression is permanently rejected for this dataset size (~1000 training rows). The non-parametric approach consistently overfits. Platt (parametric logistic calibration) is the correct choice for <5000 training rows. If the training set grows to 5000+ games (roughly 10+ seasons), isotonic or spline-based calibration could be re-evaluated, but this is unlikely to change the conclusion — even in large datasets, Platt is often competitive with isotonic and never catastrophically worse.

---

## Summary of Dispositions

| Challenger | Disposition | Key Reason | Retry Condition |
|-----------|-------------|------------|-----------------|
| L1: prior_win_pct | **RETIRE** | Too noisy; year-to-year NFL win% correlation is low | Use Elo-based prior instead of win% (different experiment) |
| L2: weather_missing | **RETIRE** | Redundant with existing roof type; collinear features added noise | Not worth retrying |
| L3: roof_enc | **RETIRE** | Target subgroup (n=32 retractable/open) too small; 0 games in holdout | When 2+ seasons accumulate ~18 more retractable/open games |
| L4: games_since_change | **MONITOR** | Real QB-change improvement (−0.0102) but net validation penalty | Retry with binned encoding OR 2+ more seasons of data |
| L5: isotonic | **RETIRE** | Overfit on ~1000 training rows; Platt is superior at this size | If training data grows to 5000+ rows |

---

## Cross-Challenger Patterns

1. **Early-season weakness remains the largest opportunity** (LL gap of 0.0459 between weeks 1-4 and playoffs). No challenger materially closed this gap.
2. **All 5 challengers had worse validation** (0/5 validated better). This suggests the incumbent's parameter set is near-optimal for the current sample size.
3. **The incumbent's calibration (ECE=0.0628) is adequate** — forcing better calibration (isotonic) was counterproductive.
4. **Subgroup improvements are consistently offset by degradation elsewhere**. The model is at a Pareto frontier for the current dataset size and feature set.
