# ruff: noqa: E501
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
REPORTS = BASE / "reports"
BENCHMARKS = REPORTS / "benchmarks"
PREDICTIONS = REPORTS / "predictions"
EXPERIMENTS = REPORTS / "experiments"
DOCS = BASE / "docs"
BACKTESTS = REPORTS / "backtests"

LEADERBOARD_PATH = BENCHMARKS / "leaderboard.csv"
INCUMBENT_PATH = BENCHMARKS / "nfl_research_incumbent.md"
CARD_PATH = BENCHMARKS / "incumbent_model_card.md"
HOLDOUT_PREDS_PATH = PREDICTIONS / "incumbent_predictions_2025_holdout.csv"
FULL_PREDS_PATH = PREDICTIONS / "incumbent_predictions.csv"
WEEKLY_REPORT_PATH = PREDICTIONS / "weekly_report.md"
PREDICTION_CARDS_PATH = PREDICTIONS / "incumbent_prediction_cards.md"
BACKTEST_REPORT_PATH = BACKTESTS / "2025_backtest_report.md"

INCUMBENT_HOLDOUT_LL = "0.6262"
INCUMBENT_VAL_LL = "0.6334"
INCUMBENT_VERSION = "2.0.0"
INCUMBENT_FEATURES = "qb_changed + rolling_mov_3"
INCUMBENT_NAME = "Standard Elo + qb_changed + rolling_mov_3 + Platt"


# Relative paths for dashboard links
def _rel(p):
    return str(p.relative_to(BASE)) if p else ""


def _gh(p):
    return f"https://github.com/SecuritahGuy/sports-ml-lab/blob/main/{_rel(p)}" if p else ""


R_FULL = _gh(FULL_PREDS_PATH)
R_HOLDOUT = _gh(HOLDOUT_PREDS_PATH)
R_WEEKLY = _gh(WEEKLY_REPORT_PATH)
R_CARDS = _gh(PREDICTION_CARDS_PATH)
R_LB = _gh(LEADERBOARD_PATH)
R_EXPERIMENTS = _gh(EXPERIMENTS)
R_CARD = _gh(CARD_PATH)
R_HISTORY = _gh(BENCHMARKS / "benchmark_history.md")
R_BACKTEST = _gh(BACKTEST_REPORT_PATH)


def _read_leaderboard():
    with open(LEADERBOARD_PATH) as f:
        return list(csv.DictReader(f))


def _fmt(val, decimals=4):
    if val is None or val == "" or val == "nan":
        return "\u2014"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _current_status_block():
    return f"""## Current Status

| Attribute | Value |
|-----------|-------|
| **True incumbent** | {INCUMBENT_NAME} |
| **Version** | v{INCUMBENT_VERSION} |
| **Average validation log loss** | {INCUMBENT_VAL_LL} |
| **2025 holdout log loss** | **{INCUMBENT_HOLDOUT_LL}** |
| **2025 holdout Brier score** | 0.2191 |
| **2025 holdout ROC AUC** | 0.7024 |
| **Active feature set** | {INCUMBENT_FEATURES} |
| **Validation method** | Rolling-origin 3-fold walk-forward |
| **Seasons used** | 2021\u2013current only |
| **Market data** | Diagnostic only (closing moneyline: holdout 0.6090) |
| **Last updated** | 2026-06-24 |
"""


def build_index():
    rows = _read_leaderboard()
    total = len(rows)
    rejected = sum(1 for r in rows if r["decision"] == "rejected")
    promoted = sum(1 for r in rows if r["decision"] in ("promoted", "superseded"))
    diagnostic = sum(1 for r in rows if "diagnostic" in (r["decision"] or ""))

    return f"""# SportsLab / StatSpace NFL Research

**A reproducible ML research lab for pregame NFL win prediction.**
This project demonstrates disciplined probability modeling, walk-forward
validation, systematic feature governance, and strict leakage prevention.

> **This is not a betting bot.** Every model is evaluated by log loss,
> calibration, and leakage prevention \u2014 not ROI. Market data is used
> for diagnostic comparison only, never for training or selection.

---

{_current_status_block()}

---

## What This Project Demonstrates

### Rigorous validation
Every model is tested via **rolling-origin 3-fold walk-forward validation**
(2021\u2192val 2022, 2021\u20132022\u2192val 2023, 2021\u20132023\u2192val 2024).
Selection uses average validation log loss. The **2025 season is a one-shot
holdout**, never accessed during model selection.

### Feature governance
{len([r for r in rows if r['decision'] == 'rejected'])} feature families have been
tested and rejected. Each experiment is documented in a reproducible report.
No feature enters the model without proving itself on both validation and holdout.

### Probability focus
Optimized for **log loss, Brier score, and calibration** \u2014 not accuracy or ROI.
The project treats NFL prediction as a probabilistic modeling problem first.

### Reproducibility
Full prediction artifacts, benchmark registry, and model card are checked into
the repo. Run `make test` (645+ tests) to validate the entire pipeline.

---

## Current Research Incumbent

| Attribute | Value |
|-----------|-------|
| **Model** | {INCUMBENT_NAME} |
| **Version** | v{INCUMBENT_VERSION} |
| **Feature set** | {INCUMBENT_FEATURES} |
| **Selection method** | Rolling-origin 3-fold validation + forward selection |
| **Average validation log loss** | {INCUMBENT_VAL_LL} |
| **2025 holdout log loss** | **{INCUMBENT_HOLDOUT_LL}** |
| **Calibration** | Platt scaling (logistic on Elo prob + features) |

### Rejected feature families ({len([r for r in rows if r['decision'] == 'rejected'])} total)

Weather, scheduling/rest, EPA team efficiency, comprehensive efficiency
(58 columns), injury reports, QB identity OHE (catastrophic overfit),
tree models (overfit), AutoGluon AutoML, Glicko (all 432 configs worse),
team-specific HFA, home/away separate Elo, coach tenure, confidence
calibration tweaks, rolling MOV windows \u2260 3. See [Feature Usage Map](feature-usage-map)
and [Experiments](experiments) for full details.

---

## Registry Summary

| Metric | Count |
|--------|-------|
| Total experiments | {total} |
| Promoted / superseded | {promoted} |
| Rejected | {rejected} |
| Diagnostic | {diagnostic} |

## Pages

- [Benchmarks & Leaderboard](benchmarks) \u2014 incumbent, promotion rules, leaderboard
- [Predictions](predictions) \u2014 prediction artifacts, holdout file, weekly report
- [Model Card](model-card) \u2014 full model documentation
- [Experiments](experiments) \u2014 experiment reports grouped by outcome
- [Backtest Reports](backtests) \u2014 season-by-season analysis (2022\u20132025)
- [Research Roadmap](research-roadmap) \u2014 what worked, what failed, next steps
- [Feature Usage Map](feature-usage-map) \u2014 complete feature governance reference

## Research Philosophy

1. Predict probabilities, not vibes.
2. Optimize first for log loss, Brier score, calibration, and leakage prevention.
3. Accuracy is secondary.
4. ROI is not a primary model-promotion metric.
5. Do not use future data in features.
6. Every feature must be explainable and pregame-safe.
7. Every experiment report must include leakage risk.

## Quick Links

- Full predictions CSV: [`{R_FULL}`]({R_FULL})
- Holdout predictions: [`{R_HOLDOUT}`]({R_HOLDOUT})
- Weekly report: [`{R_WEEKLY}`]({R_WEEKLY})
- Benchmark history: [`{R_HISTORY}`]({R_HISTORY})
- Leaderboard CSV: [`{R_LB}`]({R_LB})
- Prediction cards: [`{R_CARDS}`]({R_CARDS})
"""


def _leaderboard_table(rows_subset):
    cols = ["Experiment", "Decision", "Val LL", "Holdout LL", "Holdout AUC", "Report"]
    out = f"| {' | '.join(cols)} |\n"
    out += f"| {' | '.join('---' for _ in cols)} |\n"
    for r in rows_subset:
        report = r.get("report_path", "")
        rlink = (
            f"[{Path(report).name}](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/{report})"
            if report and report != "nan"
            else "\u2014"
        )
        vals = [
            r["experiment"],
            r["decision"],
            _fmt(r.get("val_ll")),
            _fmt(r.get("holdout_ll")),
            _fmt(r.get("holdout_auc"), 4),
            rlink,
        ]
        out += f"| {' | '.join(vals)} |\n"
    return out


def build_benchmarks():
    rows = _read_leaderboard()
    promoted_rows = [r for r in rows if r["decision"] in ("promoted", "superseded")]
    rejected_rows = [r for r in rows if r["decision"] == "rejected"]
    diagnostic_rows = [r for r in rows if r["decision"] == "diagnostic"]
    market_rows = [r for r in rows if "market" in (r["decision"] or "")]

    return f"""# Benchmarks & Leaderboard

## Current Football-Only Incumbent

**{INCUMBENT_NAME}**

- Version: v{INCUMBENT_VERSION}
- Holdout log loss: **{INCUMBENT_HOLDOUT_LL}**
- Average validation log loss: {INCUMBENT_VAL_LL}
- Feature set: {INCUMBENT_FEATURES}
- Full details: [Model Card](model-card)

### Promotion Rules

1. A challenger must beat **{INCUMBENT_HOLDOUT_LL}** holdout log loss
   to become the new football-only incumbent.
2. The challenger must also have **better average rolling validation
   log loss** than the incumbent.
3. Selection must use average rolling validation log loss only.
4. 2025 holdout is for final evaluation only, never for model selection.
5. Every feature must be pregame-safe, explainable, and leakage-safe.
6. Do not promote based on AUC or ROI alone.

---

### Promoted / Superseded Models

These models were promoted as the research incumbent at some point:

{_leaderboard_table(promoted_rows)}

### Rejected Challengers

These models failed to beat the incumbent:

{_leaderboard_table(rejected_rows)}

### Diagnostics

These experiments produced diagnostic insights but were not promoted:

{_leaderboard_table(diagnostic_rows)}

### Market-Aware Diagnostics

Market-relative benchmarks. Not football-only:

{_leaderboard_table(market_rows)}

### Note on Market Benchmark

Market (no-vig closing moneyline) achieves holdout log loss 0.6090,
significantly better than the football-only incumbent ({INCUMBENT_HOLDOUT_LL}).
The market is the true performance ceiling for pregame NFL prediction.
The incumbent is a purely pregame, market-free benchmark.

---

*Source: [`{R_LB}`]({R_LB}) and [`{R_HISTORY}`]({R_HISTORY})*
"""


def build_predictions():
    full_ok = FULL_PREDS_PATH.exists()
    holdout_ok = HOLDOUT_PREDS_PATH.exists()
    weekly_ok = WEEKLY_REPORT_PATH.exists()
    cards_ok = PREDICTION_CARDS_PATH.exists()

    def _ok(x):
        return "OK" if x else "MISSING"

    return f"""# Predictions

## Latest Prediction Artifacts

| Artifact | Description | Status |
|----------|-------------|--------|
| [`incumbent_predictions.csv`]({R_FULL}) | All eligible games | {_ok(full_ok)} |
| [`incumbent_predictions_2025_holdout.csv`]({R_HOLDOUT}) | 2025 holdout | {_ok(holdout_ok)} |
| [`weekly_report.md`]({R_WEEKLY}) | Weekly report | {_ok(weekly_ok)} |
| [`incumbent_prediction_cards.md`]({R_CARDS}) | Game cards | {_ok(cards_ok)} |

## Prediction Schema

Each prediction CSV contains:

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `season`, `week`, `gameday` | Game timing |
| `away_team`, `home_team` | Teams |
| `home_win_actual` | Actual result (1 = home win) |
| `incumbent_home_win_prob` | Predicted home win probability |
| `predicted_winner` | Home or away team based on prob |
| `confidence_bucket` | Probability range bucket |
| `model_version` | Incumbent version at prediction time |
| `feature_set` | Features used |
| `calibration_method` | Calibration type |
| `caution_qb_change` | QB changed from prior game |
| `caution_neutral` | Near-50% prediction |
| `caution_early_season` | Weeks 1\u20134 |
| `caution_missing_features` | Some features imputed |
| `caution_model_market_disagreement` | Model vs market gap > 0.15 |
| `market_prob_diagnostic` | Market-implied prob (diagnostic) |
| `market_minus_model_diagnostic` | Market prob minus model prob |

## Confidence Buckets

| Bucket | Range | Description |
|--------|-------|-------------|
| 50-55 | 0.50\u20130.55 | Near coin flip |
| 55-60 | 0.55\u20130.60 | Slight favorite |
| 60-65 | 0.60\u20130.65 | Moderate favorite |
| 65-70 | 0.65\u20130.70 | Solid favorite |
| 70-80 | 0.70\u20130.80 | Strong favorite |
| 80+ | 0.80+ | Heavy favorite (rare in NFL) |

## Caution Flags

| Flag | Meaning |
|------|---------|
| QB change | QB did not start prior game |
| Neutral | Near 50% probability |
| Early season | Weeks 1\u20134 |
| Missing features | Imputed input data |
| Model-market disagreement | Gap > 0.15 in probability |

## Market Fields (Diagnostic Only)

The `market_prob_diagnostic` and `market_minus_model_diagnostic`
columns are for research comparison only. Market data comes from
closing moneyline odds and is NOT used in model training.

## Holdout Performance

The 2025 holdout contains **276 games** (regular season and playoffs).
The current incumbent achieves log loss **{INCUMBENT_HOLDOUT_LL}** on this set.

*Holdout file: [`{R_HOLDOUT}`]({R_HOLDOUT})*

## 2025 Backtest Report

A comprehensive 2025 backtest analysis is available, including week-by-week,
team-level, calibration bucket, subgroup diagnostics, and extreme-game analysis.

- [Full Backtest Report](backtest-2025)
- [Weekly Summary CSV]({_gh(BACKTESTS / '2025_weekly_summary.csv')})
- [Team Summary CSV]({_gh(BACKTESTS / '2025_team_summary.csv')})
- [Calibration Buckets CSV]({_gh(BACKTESTS / '2025_calibration_buckets.csv')})
- [Extreme Games CSV]({_gh(BACKTESTS / '2025_extreme_games.csv')})
- [Subgroup Summary CSV]({_gh(BACKTESTS / '2025_subgroup_summary.csv')})
"""


def build_model_card():
    card_text = CARD_PATH.read_text() if CARD_PATH.exists() else "Model card not found."
    parts = card_text.split("\n", 1)
    content = parts[1] if len(parts) > 1 else card_text
    return f"""# Model Card

*This page is generated from [`{R_CARD}`]({R_CARD}).*

{content}
"""


def build_experiments():
    rows = _read_leaderboard()
    promoted_rows = [r for r in rows if r["decision"] in ("promoted", "superseded")]
    rejected_rows = [r for r in rows if r["decision"] == "rejected"]
    diagnostic_rows = [r for r in rows if r["decision"] == "diagnostic"]
    market_rows = [r for r in rows if "market" in (r["decision"] or "")]

    lines = [f"# Experiments\n\nAll {len(rows)} experiments grouped by outcome.\n\n---\n"]

    for label, group in [
        ("Promoted / Superseded", promoted_rows),
        ("Rejected", rejected_rows),
        ("Diagnostic", diagnostic_rows),
        ("Market-Aware", market_rows),
    ]:
        lines.append(f"### {label}\n")
        for r in group:
            report = r.get("report_path", "")
            rlink = (
                f"[{Path(report).name}](https://github.com/SecuritahGuy/sports-ml-lab/blob/main/{report})"
                if report and report != "nan"
                else "\u2014"
            )
            holdout = _fmt(r.get("holdout_ll"))
            val = _fmt(r.get("val_ll"))
            lines.append(
                f"- **{r['experiment']}** ({r['decision']}) \u2014 "
                f"val {val}, holdout {holdout} \u2014 {rlink}"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("### Artifact / Productization\n")
    lines.append(
        "These experiments were about registry, prediction artifacts, and productization:\n"
    )
    lines.append(
        "- **benchmark_registry** \u2014 Created `nfl_research_incumbent.md`, "
        "`benchmark_history.md`, `leaderboard.csv`"
    )
    lines.append(
        "- **predict_incumbent** \u2014 Created `incumbent_predictions.csv`, "
        "holdout file, prediction cards"
    )
    lines.append("- **weekly_report** \u2014 Weekly game report generation")
    lines.append("- **build_dashboard** \u2014 This GitHub Pages dashboard")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Source: [`{R_LB}`]({R_LB}) and [`reports/experiments/`]({R_EXPERIMENTS})*")
    lines.append("")

    return "\n".join(lines)


def build_roadmap():
    # fmt: off
    gh = "https://github.com/SecuritahGuy/sports-ml-lab"
    return f"""# Research Roadmap

A forward-looking research agenda based on 32 completed experiments and current NFL analytics best practices.

*Source: [`sports-ml-lab`]({gh})*

---

## 1. What Has Worked

These techniques and design decisions survived strict rolling-origin validation and are core to the research incumbent:

| Technique | Evidence |
|-----------|----------|
| **Elo/rating spine** | Tuned point-differential Elo (K=36, HFA=40, capped_linear MOV) forms the core signal. Outperformed Glicko and naive logistic regression. |
| **Season regression / decay** | Preseason regression toward mean (reg=0.1) + exponential decay (half-life=32 games) + QB-change bonus regression (0.2). Each improved validation log loss. |
| **qb_changed binary feature** | Largest single-feature gain: \u22120.0072 validation LL. Captures injury/benching shocks that Elo undershoots. |
| **rolling_mov_3** | 3-game rolling average margin of victory. Best window size after sensitivity testing (mov_1 won val but lost holdout). |
| **Platt calibration** | Logistic regression on [elo_prob, qb_changed, rolling_mov_3] beats raw Elo or isotonic calibration. |
| **Rolling-origin validation** | 3-fold walk-forward (train 2021 \u2192 val 2022, train 2021\u20132022 \u2192 val 2023, etc.). Prevents leakage and mimics real forecasting. |
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

*This roadmap is updated after each major experiment. Current incumbent: v{INCUMBENT_VERSION} (holdout LL {INCUMBENT_HOLDOUT_LL}).*
"""
    # fmt: on


def build_feature_usage_map():
    gh = "https://github.com/SecuritahGuy/sports-ml-lab"
    return f"""# Feature Usage Map

*A comprehensive reference for how features are used, tested, and rejected in this project.*

*Source: [`sports-ml-lab`]({gh})*

---

## A. Active Incumbent Features

The current research incumbent (**Standard Elo + qb_changed + rolling_mov_3 + Platt**) uses exactly 5 features in its Platt calibration layer:

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `elo_prob` | Continuous (0\u20131) | `compute_elo_features()` | Elo-implied home win probability from point-differential Elo |
| `home_qb_changed` | Binary | `compute_qb_features()` | 1 if home QB did not start team's prior game |
| `away_qb_changed` | Binary | `compute_qb_features()` | 1 if away QB did not start team's prior game |
| `home_rolling_mov_3` | Continuous | `compute_situational_features()` | Avg margin of victory over home team's last 3 games |
| `away_rolling_mov_3` | Continuous | `compute_situational_features()` | Avg margin of victory over away team's last 3 games |

The Elo rating engine itself has tunable parameters that act as implicit features:

| Parameter | Value | Role |
|-----------|-------|------|
| K-factor | 36 | Learning rate for rating updates |
| Home field advantage | 40 Elo points | Added to home team rating |
| Base preseason regression | 0.1 | Shrink toward league mean each offseason |
| QB-change bonus regression | 0.2 | Additional regression for teams with new starting QB |
| Decay half-life | 32 games | Exponential decay toward prior rating |
| MOV type | capped_linear (scale=0.05, cap=2.0) | Caps blowout influence on rating updates |

---

## B. Promising but Disputed Features

These features showed some signal but did not earn promotion under the strict validation+holdout rule:

| Feature | Best Validation LL | Best Holdout LL | Why Not Promoted |
|---------|-------------------|-----------------|------------------|
| `rolling_mov_3` alone (no qb_changed) | 0.6406 | **0.6255** | Wins holdout but not validation. The 0.6255 holdout beats the incumbent 0.6262, but validation is 0.6406 vs incumbent 0.6334. |
| `rolling_mov_1` (1-game window) | **0.6338** | 0.6302 | Wins validation but loses holdout \u2014 classic overfit. Not promoted. |
| Coach+QB season regression | 0.6309 | 0.6286 | Tiny validation win (0.0006) erased on holdout (-0.0001). |
| O/D Elo (ko52_kd20) + Platt | 0.6376 | 0.6258 | Better holdout but worse validation. Demoted to holdout-informed diagnostic. |
| QB injury flag (single binary) | 0.6464 | 0.6255 | Noise-level holdout improvement; validation 0.0088 worse. |

---

## C. Rejected Feature Families

All 20 rejected experiments. Each was tested via rolling-origin 3-fold validation
and the 2025 one-shot holdout.

### Weather (4 columns)

- **Features**: temperature, wind speed, precipitation flag, dome flag, cold/windy/bad-weather thresholds
- **Validation**: 0.6445 (vs incumbent 0.6363)
- **Holdout**: 0.6439 (vs incumbent 0.6373)
- **Decision**: Rejected \u2014 both worse
- **Report**: `reports/experiments/weather_features.md`

### Scheduling / Rest (6+ columns)

- **Features**: short week, off bye, Thursday/Monday flags, consecutive road games, international
- **Validation**: 0.6599 (vs incumbent 0.6363)
- **Holdout**: 0.6401 (vs incumbent 0.6373)
- **Decision**: Rejected \u2014 both worse
- **Report**: `reports/experiments/schedule_rest_features.md`

### QB Identity OHE

- **Features**: 93 one-hot encoded QB names
- **Holdout**: 14.51 log loss (catastrophic overfit)
- **Decision**: Rejected \u2014 93 classes for 376 training rows
- **Report**: `reports/experiments/qb_features.md`

### Coach Tenure / Win Percentage

- **Features**: coach tenure (games/years), career wins, career win%
- **Decision**: Rejected \u2014 all variants worse on both val and holdout
- **Report**: `reports/experiments/combined_features.md`

### EPA / Team Efficiency (18 columns)

- **Features**: rolling 3/5 avg of passing/rushing/receiving/total EPA per play
- **Validation**: 0.6654 (vs incumbent 0.6363)
- **Holdout**: 0.6495 (vs incumbent 0.6373)
- **Decision**: Rejected \u2014 both worse; made QB-change failure mode worse
- **Report**: `reports/experiments/epa_features.md`

### Comprehensive Efficiency (58 columns, 3 sources)

- **Sources**: Team Stats Total EPA, PFR Advanced Stats (pass/rush/rec/def), Snap Counts
- **Validation**: 0.6597 (vs incumbent 0.6368)
- **Holdout**: 0.6788 (vs incumbent 0.6313)
- **Decision**: Rejected \u2014 58 features added noise at this sample size
- **Report**: `reports/experiments/comprehensive_efficiency.md`

### Injury Report Features (20 columns)

- **Features**: QB OUT flags, position-group injury counts (RB/WR/TE/OL/DL/LB/DB), injury-driven QB change, net differentials
- **Validation**: 0.6486 (vs incumbent 0.6406)
- **Holdout**: 0.6514 (vs incumbent 0.6315/0.6285)
- **Decision**: Rejected \u2014 all worse
- **Report**: `reports/experiments/injury_features.md`

### Team Stats (yards / fantasy / sacks)

- **Features**: rolling aggregates from nflreadpy load_player_stats
- **Validation**: 0.6541 (vs incumbent 0.6368)
- **Holdout**: 0.6415 (vs incumbent 0.6285)
- **Decision**: Rejected \u2014 all variants worse
- **Report**: `reports/experiments/team_stats.md`

### Tree-Based Models (HGB / GB / RF)

- **Best validation**: RandomForest at 0.6329 on curated 27 features
- **Holdout**: 0.6456 (RF) \u2014 classic overfit pattern
- **Decision**: Rejected \u2014 won validation but lost holdout
- **Report**: `reports/experiments/expressive_models.md`

### AutoGluon AutoML

- **Features**: sklearn-only ensemble (RF, ExtraTrees) on 47 pregame features
- **Validation**: 0.6956 (vs Platt 0.6376)
- **Holdout**: 0.6404 (vs Platt 0.6362)
- **Decision**: Rejected \u2014 both worse
- **Report**: `reports/experiments/autogluon.md`

### Glicko Rating System

- **Configurations**: 432 (4 HFA \u00d7 6 init_RD \u00d7 6 sys_c \u00d7 3 QB bonus)
- **Best validation**: 0.6513 (worse than Elo)
- **Best holdout**: 0.7013 (far worse than Elo)
- **Decision**: Rejected \u2014 all 432 configs worse
- **Report**: `reports/experiments/glicko_rating.md`

### Team-Specific HFA

- **Validation**: 0.6355 (worse than global HFA 0.6321)
- **Holdout**: 0.6263 (better than incumbent, but val rules)
- **Decision**: Rejected \u2014 worse validation despite better holdout
- **Report**: `reports/experiments/team_hfa.md`

### Home/Away Separate Elo

- **Validation**: 0.6622 (vs standard 0.6410)
- **Holdout**: 0.6634
- **Decision**: Rejected \u2014 noisier per-split ratings
- **Report**: `reports/experiments/home_away_elo.md`

### Rolling MOV Windows \u2260 3

- **mov_1**: Won validation (0.6338) but lost holdout (0.6302 vs 0.6262)
- **mov_2+**: All worse on validation
- **Decision**: Rejected \u2014 mov_3 confirmed optimal
- **Report**: `reports/experiments/rolling_mov_sensitivity.md`

### Confidence Calibration (temperature, isotonic, shrinkage)

- **Best**: Temperature T=1.50 \u2014 tied incumbent on val (0.6374) and holdout (0.6373)
- **Decision**: Rejected \u2014 no method beat Platt on both val and holdout
- **Report**: `reports/experiments/confidence_calibration.md`

### Residual Blending

- **Approach**: Logistic blend on elo_prob + week/rest/early-season features
- **Decision**: Rejected \u2014 all blends worse
- **Report**: `reports/experiments/residual_blending.md`

### Coach+QB Season Regression

- **Validation**: 0.6309 (wins by 0.0006)
- **Holdout**: 0.6286 (loses by 0.0001)
- **Decision**: Rejected \u2014 signal too weak
- **Report**: `reports/experiments/coach_season_regression.md`

### QB Injury Flag (single binary)

- **Holdout**: 0.6255 (noise-level 0.0003 improvement)
- **Validation**: 0.6464 (0.0088 worse)
- **Decision**: Rejected \u2014 noise-level improvement
- **Report**: `reports/experiments/qb_injury_flag.md`

---

## D. Diagnostic-Only Features (Market Data)

These are NOT part of the football-only model. They are used for benchmarking and diagnostic comparison only:

| Feature | Source | Description |
|---------|--------|-------------|
| `market_prob_diagnostic` | Closing moneyline (no-vig) | Market-implied home win probability |
| `market_minus_model_diagnostic` | Derived | Model error relative to market |
| Caution flag: model-market disagreement | Derived | Gap > 0.15 triggers caution flag |

**Market holdout log loss: 0.6090** \u2014 significantly better than the football-only incumbent (0.6262). The market is the true performance ceiling. Our Elo residuals correlate with market residuals at r=0.9768.

---

## E. Key Lesson

This project has tested **14+ feature families** (58 columns in the largest). Nearly all were rejected because:

1. **Small sample**: ~1,000 training games (2021\u20132024). Broad features (EPA, efficiency, weather) add noise, not signal.
2. **Elo already captures most signal**: Elo probability correlates with market at r=0.9768. Adding weak features on top is difficult.
3. **Discrete signals > continuous noise**: The two features that earned promotion (qb_changed, rolling_mov_3) are discrete or simple rolling aggregates. Broad continuous feature groups consistently overfit.
4. **Validation before holdout**: Multiple features (mov_1, team HFA, O/D Elo) looked good on holdout but failed on validation. The strict validation-first rule prevents false promotions.

Every feature must earn its way in through chronological validation. Features are not ignored \u2014 they are tested systematically and rejected honestly.
"""


def _season_backtest_section(season: int) -> str:
    report_path = BACKTESTS / f"{season}_backtest_report.md"
    weekly_csv = BACKTESTS / f"{season}_weekly_summary.csv"
    team_csv = BACKTESTS / f"{season}_team_summary.csv"
    cal_csv = BACKTESTS / f"{season}_calibration_buckets.csv"
    ext_csv = BACKTESTS / f"{season}_extreme_games.csv"
    sub_csv = BACKTESTS / f"{season}_subgroup_summary.csv"

    def _ok(x):
        return "OK" if x else "MISSING"

    lines = [f"### {season} Season\n"]
    lines.append("| Artifact | Description | Status |")
    lines.append("|----------|-------------|--------|")
    lines.append(f"| [Full Report]({_rel(report_path)}) | Comprehensive Markdown report | {_ok(report_path.exists())} |")
    lines.append(f"| [Weekly Summary]({_rel(weekly_csv)}) | Week-by-week breakdown | {_ok(weekly_csv.exists())} |")
    lines.append(f"| [Team Summary]({_rel(team_csv)}) | Per-team diagnostics | {_ok(team_csv.exists())} |")
    lines.append(f"| [Calibration Buckets]({_rel(cal_csv)}) | Confidence bucket analysis | {_ok(cal_csv.exists())} |")
    lines.append(f"| [Extreme Games]({_rel(ext_csv)}) | Best/worst predictions | {_ok(ext_csv.exists())} |")
    lines.append(f"| [Subgroup Summary]({_rel(sub_csv)}) | Game-context breakdown | {_ok(sub_csv.exists())} |")
    lines.append("")
    return "\n".join(lines)


def build_backtest():
    return f"""# Backtest Reports

*Generated by `sportslab backtest <season>`.*

The backtest evaluates the incumbent model
(**{INCUMBENT_NAME}**) across each NFL season.
Seasons 2022\u20132024 are in-training diagnostics (part of 2021\u20132024 training data).
Season 2025 is a locked holdout.

## Season-by-Season Analysis

{_season_backtest_section(2022)}

{_season_backtest_section(2023)}

{_season_backtest_section(2024)}

{_season_backtest_section(2025)}

## Key Metrics (2025 Holdout)

| Metric | Value |
|--------|-------|
| Games | 276 |
| Log loss | **{INCUMBENT_HOLDOUT_LL}** |
| Accuracy | 0.6630 |
| Brier score | 0.2180 |
| ROC AUC | 0.7050 |

## Research Caveats

- The model was trained on 2021\u20132024 data only. 2025 was a locked holdout.
- Seasons 2022\u20132024 were part of the training window and are diagnostic only.
- Market data is excluded from the model. Market diagnostic only (closing moneyline: 0.6090).
- This is a probabilistic prediction benchmark, not a gambling product.
"""


def _gh(p):
    return f"https://github.com/SecuritahGuy/sports-ml-lab/blob/main/{_rel(p)}" if p else ""


def build_all():
    DOCS.mkdir(parents=True, exist_ok=True)

    pages = {
        "index": ("index.md", build_index),
        "benchmarks": ("benchmarks.md", build_benchmarks),
        "predictions": ("predictions.md", build_predictions),
        "model-card": ("model-card.md", build_model_card),
        "experiments": ("experiments.md", build_experiments),
        "research-roadmap": ("research-roadmap.md", build_roadmap),
        "feature-usage-map": ("feature-usage-map.md", build_feature_usage_map),
        "backtests": ("backtests.md", build_backtest),
    }

    for name, (filename, builder) in pages.items():
        content = builder()
        (DOCS / filename).write_text(content)
        print(f"  ✓ docs/{filename}")

    print(f"\nDashboard built at docs/ \u2014 {len(pages)} pages generated.")
