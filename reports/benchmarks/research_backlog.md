# Research Backlog — 2026 Season Roadmap

*Based on external assessment and all prior experiments (48+ completed).*

---

## Executive Strategy

The project has completed enough controlled experiments to show the main limitation is **sample size and information availability**, not model complexity. The frozen champion (v3.0.0 Frozen QB Overlay, holdout LL 0.6200) is near the practical ceiling of the current 2021–2025 dataset.

**2026 strategy:**
1. Keep v3.0.0 frozen for Week 1
2. Run only low-risk bounded experiments before the season
3. Shadow-test challengers during the season
4. Use live 2026 failures to determine which research direction deserves reopening

---

## Before Week 1 (Preseason)

### 1. 🥇 Elo Parameter Ensemble

**Risk:** Low | **Cost:** Low | **Production suitability:** High

Average predictions from several neighboring, fold-stable Elo configurations (e.g., K 32/36/40, HFA 35/40/45, reg 0.05/0.10/0.15) rather than treating one selected config as exact.

**Why it could work:** Parameter selection uncertainty is meaningful with only a few seasons. Averaging neighboring high-performing configurations may reduce tuning variance, stabilize early-season predictions, and improve calibration.

**Guardrail:** Use only parameter combinations that performed reasonably across every validation fold. Do not average the entire search grid.

**Promotion standard:** Must beat v3.0.0 on BOTH rolling val and 2025 holdout with Δ ≥ 0.001.

### 2. 🥈 Kalman-Filter Elo (Shadow Model)

**Risk:** Medium | **Cost:** Medium | **Production suitability:** Shadow only

Represent each team with estimated strength + rating variance + process noise + observation noise. Kalman update allows ratings to change faster when uncertainty is high (new coach/QB) and slower when a team is stable.

**Why it differs from Glicko:** Glicko-1's g(RD) damping fundamentally limits prediction sharpness (proven — bug-fixed Glicko still can't beat Elo). Kalman uncertainty mechanics are more flexible.

**Guardrail:** Build retrospective 2021–2025 predictions. Do not delay the production workflow. If it wins on both val and holdout retrospectively, promote it as v4.0.0 shadow for live 2026.

---

## During 2026 Season (Shadow / Research)

### 3. 🥉 Dynamic Bayesian Elo (Research)

**Framework:** PyMC or NumPyro

Represent each team's latent strength as a probability distribution evolving over time via a state-space model. Includes team-level drift, season-to-season variance, QB-change variance, and posterior uncertainty around adjustments.

**Potential advantages:** Explicit uncertainty in early-season ratings, natural shrinkage for volatile teams, credible intervals per game, more principled offseason regression.

**Guardrail:** Keep model very small — do not add dozens of covariates. Test 4 variants: (1) dynamic strength only, (2) + fixed HFA, (3) + binary QB-change, (4) + gated QB-starts.

### 4. Score-Margin Distribution Model (Shadow)

Rather than predicting home win directly, predict the distribution of final scoring margin. Derive win probability from P(M > 0).

**Model options:** Gaussian regression, Student-t, ordered buckets, bivariate Poisson, Skellam score-difference.

**Why it adds value:** Can learn expected margin, game-level uncertainty, tail behavior, and differences between a 1-point and 10-point favorite. Also provides a path to future spread prediction.

**Initial version:** Elo difference + HFA + rolling MOV diff + QB-change indicators → predicted margin + heteroscedastic variance. Derive win probability, calibrate with fold-safe Platt.

### 5. Hierarchical Bradley–Terry (Research Validation)

Jointly estimate team strength, HFA, season effect, and QB-change variance with partial pooling across teams and seasons.

**Main risk:** May produce nearly the same ranking as Elo with much more computation. That is still a useful result — it would validate the current architecture from a different statistical framework.

### 6. Pi-Ratings (Bounded Experiment)

Coupled home/away strengths with nonlinear score-error updates. Not merely two independent Elo ratings.

**Guardrail:** One bounded experiment. Do not open a large parameter hunt.

---

## Operational Improvements (Before Week 1)

### Prediction Vintages

Save at least three snapshots per game:
- Early week
- After final injury reports
- Final locked prediction

Each vintage preserves: expected starting QB, QB source, timestamp, home-win prob, confidence category, whether QB gate fired, overlay magnitude.

### Live Monitoring Triggers

#### Calibration trigger
Reopen only after 100+ graded games with: calibration slope < 0.80 or > 1.20, or |intercept| > 0.15, or live ECE > 0.08.

#### QB overlay trigger
Reopen when 20+ gated games graded AND gated-game LL worse than base model by ≥ 0.02.

#### Early-season trigger
Reopen if Weeks 1–4 again produce LL above ~0.67 and materially trail market or late-season performance.

---

## After the 2026 Regular Season

Add complete 2026 season as new out-of-sample evidence. Then revisit:
- Turnover differential (to_net_3, watchlist)
- Team-specific HFA (watchlist)
- Short-window MOV (watchlist)
- Kalman Elo (if shadow outperformed)
- Dynamic Bayesian strength
- Score-margin distribution model

One additional season = 272 games. Many complex models remain underpowered.

---

## Not Worth Testing (Unless Triggers Met)

| Lane | Reason |
|------|--------|
| Tree/ensemble models | Tested 4x, consistently overfit (<5000 training rows) |
| Neural networks | Structural mismatch: dataset too small, paired observations, temporal structure |
| More calibration | Platt beat isotonic, temperature, shrinkage — all retired |
| Injury features | 10+ variants tested, all rejected, retested on modern spine (val +0.0190, hold +0.0203) |
| Weather features | Retested on modern spine (val +0.0329, hold +0.0098). Never helps. |
| Scheduling features | Retested on modern spine (val +0.0208, hold +0.0020). Never helps. |
| QB identity OHE | Holdout LL 14.51 — catastrophic overfit |
| Market features | Diagnostic only; not pregame; timing mismatch |
| Glicko | Bug fixed, still can't beat Elo (g(RD) damping limits sharpness) |
| Expanded seasons (pre-2021) | Blocked by project governance |

---

## Bottom Line

The biggest opportunity is not a deeper model. It is making the weekly system exceptionally good at **knowing which quarterback is starting, when that information changed, and exactly what the model believed at each point before kickoff**.

Proposed versioning:
- v3.0.1 — operational fixes without probability changes
- v3.1.0 — calibrated or overlay adjustment
- v4.0.0 — new rating architecture
