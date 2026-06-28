# Research Roadmap

A forward-looking research agenda based on 32 completed experiments and current NFL analytics best practices.

*Source: [`sports-ml-lab`](https://github.com/SecuritahGuy/sports-ml-lab)*

---

## 1. What Has Worked

These techniques and design decisions survived strict rolling-origin validation and are core to the research incumbent:

| Technique | Evidence |
|-----------|----------|
| **Elo/rating spine** | Tuned point-differential Elo (K=36, HFA=40, capped_linear MOV) forms the core signal. Outperformed Glicko and naive logistic regression. |
| **Season regression / decay** | Preseason regression toward mean (reg=0.1) + exponential decay (half-life=32 games) + QB-change bonus regression (0.2). Each improved validation log loss. |
| **qb_changed binary feature** | Largest single-feature gain: −0.0072 validation LL. Captures injury/benching shocks that Elo undershoots. |
| **rolling_mov_3** | 3-game rolling average margin of victory. Best window size after sensitivity testing (mov_1 won val but lost holdout). |
| **Platt calibration** | Logistic regression on [elo_prob, qb_changed, rolling_mov_3] beats raw Elo or isotonic calibration. |
| **Rolling-origin validation** | 3-fold walk-forward (train 2021 → val 2022, train 2021–2022 → val 2023, etc.). Prevents leakage and mimics real forecasting. |
| **Benchmark registry** | `nfl_research_incumbent.md`, `benchmark_history.md`, `leaderboard.csv`, artifact audit. Ensures reproducible, transparent research. |

## 2. What Has Failed

Every rejected experiment below was tested on rolling-origin validation and 2025 holdout. None beat the incumbent on both:

| Failed Approach | Reason | Worst Metric |
|----------------|--------|-------------|
| **Broad efficiency features** (EPA, PFR advanced stats, snap counts) | 58 features added noise; inc+eff holdout LL = 0.6788 vs incumbent 0.6313 | +0.047 holdout LL |
| **Raw identity encodings** (team/QB OHE) | 93 QB classes for 376 training rows → holdout LL 14.51 | Catastrophic overfit |
| **Weather features** | Inc+weather holdout 0.6439 vs incumbent 0.6373 | +0.007 holdout LL |
| **Scheduling/rest features** | Inc+sched holdout 0.6401 vs incumbent 0.6373 | +0.003 holdout LL |
| **Tree models** (HGB/GB/RF) | Won validation (RF 0.6329) but lost holdout (0.6456) | Classic overfit |
| **Glicko rating system** | Best val 0.6513, holdout 0.7013 — far worse than Elo | +0.064 holdout LL |
| **AutoGluon AutoML** | 47 features with sklearn-only backends. Holdout 0.6404. | +0.009 holdout LL |
| **Generic calibration tweaks** (temperature, isotonic, shrinkage) | None beat Platt on holdout. Best (shrinkage) holdout 0.6448. | +0.008 holdout LL |
| **Team-specific HFA** | Holdout 0.6263 (better) but val 0.6355 (worse). Selection rule: val wins. | +0.003 val LL |
| **Home/away separate Elo** | Noisier per-split ratings, holdout 0.6634. | +0.035 holdout LL |

## 3. Mistakes to Avoid

1. **Chasing accuracy instead of log loss / calibration.** Accuracy is secondary. The project optimizes for probabilistic prediction quality.

2. **Peeking at 2025 holdout.** Selection must use average rolling validation log loss only. Holdout is for one-shot final evaluation.

3. **Silently changing benchmark definitions.** Every incumbent change must be documented in `benchmark_history.md` and `leaderboard.csv`.

4. **Mixing closing-market diagnostics into the football-only incumbent.** Market data (closing moneyline, holdout 0.6090) is strictly diagnostic. The football-only track must remain market-free.

5. **Adding too many noisy features for a tiny NFL sample.** The dataset has ~1,000 training games (2021–2024). Feature-heavy models (58 efficiency cols, tree ensembles) consistently overfit.

6. **Pretending broad team EPA solves QB-change shocks.** Even 58 comprehensive efficiency features failed. QB-change is a discrete availability signal, not a continuous efficiency signal.

7. **Using postgame / result-derived features as pregame inputs.** Rolling features must use only games before the current game, with season-boundary resets.

8. **Using raw identity label encodings as numeric features.** QB name OHE exploded to LL 14.51. Team OHE is acceptable only as a weak baseline (LL ~0.68).

## 4. Next Research Candidates (Ranked)

| Rank | Candidate | Rationale | Priority |
|------|-----------|-----------|----------|
| 1 | **QB-change market-delta diagnostics** | QB-change games are the largest failure mode (home QB changed LL = 0.7687). Market delta (pre/post injury report) may disambiguate injuries from strategic benchings. | High |
| 2 | **Opening-line ingestion** | Current market benchmark uses closing lines (near-kickoff). Opening lines would give a fairer pregame market comparison and may reveal where Elo can win on information advantage. | High |
| 3 | **QB availability / injury timeline ingestion** | If a clean source exists for pregame QB availability (not just binary OUT flag), this could improve qb_changed precision. Should only proceed if leakage-free. | Medium |
| 4 | **Uncertainty intervals / prediction confidence audit** | The incumbent outputs point probabilities. Adding conformal prediction or prediction intervals would improve portfolio-readiness. | Medium |
| 5 | **Calibration by era / week** | Residual diagnostics showed early-season (W1–4) has higher error. If validated cleanly, era-specific Platt scaling could help. Risk of overfit on small splits. | Low |
| 6 | **Public dashboard improvements** | Add interactive visualizations, season-over-season calibration plots, confidence calibration per bucket. | Low |
| 7 | **Optional DVOA / manual external benchmark** | Only if licensing is clean and data pipeline is reproducible. Not a priority until QB-change and market-delta paths are exhausted. | Future |

---

*This roadmap is updated after each major experiment. Current incumbent: v2.0.0 (holdout LL 0.6262).*
