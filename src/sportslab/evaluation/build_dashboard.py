import csv
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
REPORTS = BASE / "reports"
BENCHMARKS = REPORTS / "benchmarks"
PREDICTIONS = REPORTS / "predictions"
EXPERIMENTS = REPORTS / "experiments"
DOCS = BASE / "docs"

LEADERBOARD_PATH = BENCHMARKS / "leaderboard.csv"
INCUMBENT_PATH = BENCHMARKS / "nfl_research_incumbent.md"
CARD_PATH = BENCHMARKS / "incumbent_model_card.md"
HOLDOUT_PREDS_PATH = PREDICTIONS / "incumbent_predictions_2025_holdout.csv"
FULL_PREDS_PATH = PREDICTIONS / "incumbent_predictions.csv"
WEEKLY_REPORT_PATH = PREDICTIONS / "weekly_report.md"
PREDICTION_CARDS_PATH = PREDICTIONS / "incumbent_prediction_cards.md"

INCUMBENT_HOLDOUT_LL = "0.6262"
INCUMBENT_VAL_LL = "0.6334"
INCUMBENT_VERSION = "v2.0.0"
INCUMBENT_FEATURES = "qb_changed + rolling_mov_3"
INCUMBENT_NAME = "Standard Elo + qb_changed + rolling_mov_3 + Platt"


# Relative paths for dashboard links
def _rel(p):
    return str(p.relative_to(BASE)) if p else ""


def _gh(p):
    return f"https://github.com/timdev/sports-ml-lab/blob/main/{_rel(p)}" if p else ""


R_FULL = _gh(FULL_PREDS_PATH)
R_HOLDOUT = _gh(HOLDOUT_PREDS_PATH)
R_WEEKLY = _gh(WEEKLY_REPORT_PATH)
R_CARDS = _gh(PREDICTION_CARDS_PATH)
R_LB = _gh(LEADERBOARD_PATH)
R_EXPERIMENTS = _gh(EXPERIMENTS)
R_CARD = _gh(CARD_PATH)
R_HISTORY = _gh(BENCHMARKS / "benchmark_history.md")


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


def build_index():
    rows = _read_leaderboard()
    total = len(rows)
    rejected = sum(1 for r in rows if r["decision"] == "rejected")
    promoted = sum(1 for r in rows if r["decision"] in ("promoted", "superseded"))
    diagnostic = sum(1 for r in rows if "diagnostic" in (r["decision"] or ""))

    return f"""# SportsLab / StatSpace NFL Research

A reproducible ML research lab for pregame NFL win prediction.
**This is not a betting bot.** Every model is evaluated by log loss,
calibration, and leakage prevention \u2014 not ROI.

---

## Current Research Incumbent

| Attribute | Value |
|-----------|-------|
| **Model** | {INCUMBENT_NAME} |
| **Version** | {INCUMBENT_VERSION} |
| **Feature set** | {INCUMBENT_FEATURES} |
| **Selection method** | Rolling-origin 3-fold validation + forward selection |
| **Average validation log loss** | {INCUMBENT_VAL_LL} |
| **2025 holdout log loss** | **{INCUMBENT_HOLDOUT_LL}** |
| **Calibration** | Platt scaling (logistic on Elo prob + features) |

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
            f"[{Path(report).name}](https://github.com/timdev/sports-ml-lab/blob/main/{report})"
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

- Version: {INCUMBENT_VERSION}
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
                f"[{Path(report).name}](https://github.com/timdev/sports-ml-lab/blob/main/{report})"
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


def build_all():
    DOCS.mkdir(parents=True, exist_ok=True)

    pages = {
        "index": ("index.md", build_index),
        "benchmarks": ("benchmarks.md", build_benchmarks),
        "predictions": ("predictions.md", build_predictions),
        "model-card": ("model-card.md", build_model_card),
        "experiments": ("experiments.md", build_experiments),
    }

    for name, (filename, builder) in pages.items():
        content = builder()
        (DOCS / filename).write_text(content)
        print(f"  ✓ docs/{filename}")

    print(f"\nDashboard built at docs/ \u2014 {len(pages)} pages generated.")
