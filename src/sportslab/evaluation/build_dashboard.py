# ruff: noqa: E501
"""Build all GitHub Pages docs from live data.

Usage:
    sportslab build-dashboard

Generates docs/*.md from actual prediction artifacts, leaderboard,
and 2026 schedule snapshot. Run before each prediction week.
"""

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
REPORTS = BASE / "reports"
BENCHMARKS = REPORTS / "benchmarks"
PREDICTIONS = REPORTS / "predictions"
EXPERIMENTS = REPORTS / "experiments"
DOCS = BASE / "docs"
SNAPSHOTS = PREDICTIONS / "snapshots"

LEADERBOARD_PATH = BENCHMARKS / "leaderboard.csv"
HISTORY_PATH = BENCHMARKS / "benchmark_history.md"
MANIFEST_PATH = PREDICTIONS / "snapshot_manifest.json"

# v3.0.0 incumbent constants
INCUMBENT_HOLDOUT_LL = "0.6200"
INCUMBENT_VAL_LL = "0.6305"
INCUMBENT_VERSION = "3.0.0"
INCUMBENT_FEATURES = "qb_changed + rolling_mov_3 + frozen QB overlay"
INCUMBENT_NAME = "Standard Elo + qb_changed + rolling_mov_3 + Platt + frozen QB overlay"


def _fmt(val, decimals=4):
    if val is None or val == "" or val == "nan":
        return "\u2014"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _gh(rel_path: str) -> str:
    return f"https://github.com/SecuritahGuy/sports-ml-lab/blob/main/{rel_path}"


# ── Data readers ──


def _read_leaderboard():
    with open(LEADERBOARD_PATH) as f:
        return list(csv.DictReader(f))


def _latest_prediction_snapshot(season: int = 2026) -> Path | None:
    """Find the latest prediction snapshot for a given season."""
    if not MANIFEST_PATH.exists() or not SNAPSHOTS.exists():
        return None
    import json
    manifest = json.loads(MANIFEST_PATH.read_text())
    snaps = [s for s in manifest.get("snapshots", [])
             if s.get("season") == season]
    if not snaps:
        return None
    latest = max(snaps, key=lambda s: s.get("created_at", ""))
    sp = SNAPSHOTS / Path(latest["path"]).name
    return sp if sp.exists() else Path(latest["path"]) if Path(latest["path"]).exists() else None


def _load_schedule_snapshot(season: int = 2026, week: int | None = None) -> pd.DataFrame | None:
    sp = _latest_prediction_snapshot(season)
    if sp is None:
        return None
    df = pd.read_csv(sp)
    if week is not None:
        df = df[df["week"] == week].copy()
    return df


# ── Page builders ──


def _navbar(current: str) -> str:
    pages = [
        ("index", "Home"),
        ("2026-schedule", "2026 Schedule"),
        ("benchmarks", "Benchmarks"),
        ("predictions", "Predictions"),
        ("model-card", "Model Card"),
        ("experiments", "Experiments"),
        ("backtests", "Backtests"),
    ]
    links = []
    for slug, label in pages:
        if slug == current:
            links.append(f"**{label}**")
        else:
            links.append(f"[{label}]({slug})")
    return " | ".join(links) + "\n\n---\n"


def build_index():
    rows = _read_leaderboard()
    total = len(rows)
    rejected = sum(1 for r in rows if r["decision"] == "rejected")
    promoted = sum(1 for r in rows if r["decision"] in ("promoted", "superseded"))
    diagnostic = sum(1 for r in rows if "diagnostic" in (r["decision"] or ""))
    snap = _latest_prediction_snapshot(2026)
    n_future = len(pd.read_csv(snap)) if snap is not None else 0

    return f"""# SportsLab / StatSpace NFL Research

{_navbar("index")}

**A reproducible ML research lab for pregame NFL win prediction.**
This project demonstrates disciplined probability modeling, walk-forward
validation, systematic feature governance, and strict leakage prevention.

> **This is not a betting bot.** Every model is evaluated by log loss,
> calibration, and leakage prevention \u2014 not ROI. Market data is used
> for diagnostic comparison only, never for training or selection.

---

## Current Status

| Attribute | Value |
|-----------|-------|
| **Incumbent** | {INCUMBENT_NAME} |
| **Version** | v{INCUMBENT_VERSION} |
| **Avg validation log loss** | {INCUMBENT_VAL_LL} |
| **2025 holdout log loss** | **{INCUMBENT_HOLDOUT_LL}** |
| **2025 holdout Brier** | 0.2157 |
| **2025 holdout AUC** | 0.7098 |
| **Feature set** | {INCUMBENT_FEATURES} |
| **Validation** | Rolling-origin 3-fold walk-forward |
| **Seasons** | 2021\u2013current only |
| **Market benchmark** | Closing moneyline holdout 0.6090 (diagnostic) |
| **2026 predictions** | {n_future} games generated |
| **Last updated** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |

---

## What This Project Demonstrates

### Rigorous validation
Every model is tested via **rolling-origin 3-fold walk-forward validation**
(2021\u2192val 2022, 2021\u20132022\u2192val 2023, 2021\u20132023\u2192val 2024).
Selection uses average validation log loss. The **2025 season is a one-shot
holdout**, never accessed during model selection.

### Feature governance
{rejected} feature families have been tested and rejected. Each experiment is
documented in a reproducible report. No feature enters the model without
proving itself on both validation and holdout.

### Probability focus
Optimized for **log loss, Brier score, and calibration** \u2014 not accuracy or ROI.
The project treats NFL prediction as a probabilistic modeling problem first.

### Reproducibility
Full prediction artifacts, benchmark registry, and model card are checked into
the repo. Run `make test` (650+ tests) to validate the entire pipeline.

---

## Research Philosophy

1. Predict probabilities, not vibes.
2. Optimize first for log loss, Brier score, calibration, and leakage prevention.
3. Accuracy is secondary.
4. ROI is not a primary model-promotion metric.
5. Do not use future data in features.
6. Every feature must be explainable and pregame-safe.
7. Every experiment report must include leakage risk.

---

## Registry Summary

| Metric | Count |
|--------|-------|
| Total experiments | {total} |
| Promoted / superseded | {promoted} |
| Rejected | {rejected} |
| Diagnostic | {diagnostic} |
"""


def build_schedule_2026():
    snap = _latest_prediction_snapshot(2026)
    if snap is None:
        return f"""# 2026 Schedule & Predictions

{_navbar("2026-schedule")}

No 2026 prediction snapshot found. Run `sportslab predict-week --season 2026 --week 1 --auto-qb` first.
"""

    df = pd.read_csv(snap)
    weeks = sorted(df["week"].unique())

    out = f"""# 2026 Schedule & Predictions

{_navbar("2026-schedule")}

**Snapshot:** `{snap.name}`
**Games:** {len(df)} across weeks {weeks[0]}\u2013{weeks[-1]}

All predictions generated by the v{INCUMBENT_VERSION} incumbent model.
QB starters auto-sourced from nflreadpy depth charts (`--auto-qb`).

---

"""
    for wk in weeks:
        wk_df = df[df["week"] == wk].sort_values("gameday")
        out += f"## Week {wk}\n\n"
        out += "| Date | Away | Home | Predicted | Home Win Prob | Bucket | Caution |\n"
        out += "|------|------|------|-----------|---------------|--------|--------|\n"
        for _, r in wk_df.iterrows():
            prob = r["incumbent_home_win_prob"]
            pred = r["predicted_winner"]
            bucket = r.get("confidence_bucket", "")
            cautions = []
            if r.get("caution_qb_change", 0):
                cautions.append("QB change")
            if r.get("caution_early_season", 0):
                cautions.append("Early season")
            caution_str = ", ".join(cautions) if cautions else "\u2014"
            out += f"| {r['gameday']} | {r['away_team']} | {r['home_team']} | {pred} | {prob:.3f} | {bucket} | {caution_str} |\n"
        out += "\n"
    return out


def build_benchmarks():
    rows = _read_leaderboard()
    promoted_rows = [r for r in rows if r["decision"] in ("promoted", "superseded")]
    rejected_rows = [r for r in rows if r["decision"] == "rejected"]
    diagnostic_rows = [r for r in rows if r["decision"] == "diagnostic"]
    market_rows = [r for r in rows if "market" in (r["decision"] or "")]

    def _leaderboard_table(rows_subset):
        cols = ["Experiment", "Decision", "Val LL", "Holdout LL", "Holdout AUC", "Report"]
        t = f"| {' | '.join(cols)} |\n"
        t += f"| {' | '.join('---' for _ in cols)} |\n"
        for r in rows_subset:
            report = r.get("report_path", "")
            rlink = (
                f"[{Path(report).name}]({_gh(report)})"
                if report and report != "nan"
                else "\u2014"
            )
            t += f"| {r['experiment']} | {r['decision']} | {_fmt(r.get('val_ll'))} | {_fmt(r.get('holdout_ll'))} | {_fmt(r.get('holdout_auc'), 4)} | {rlink} |\n"
        return t

    return f"""# Benchmarks & Leaderboard

{_navbar("benchmarks")}

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

{_leaderboard_table(promoted_rows)}

### Rejected Challengers

{_leaderboard_table(rejected_rows)}

### Diagnostics

{_leaderboard_table(diagnostic_rows)}

### Market-Aware Diagnostics

{_leaderboard_table(market_rows)}

### Note on Market Benchmark

Market (no-vig closing moneyline) achieves holdout log loss 0.6090,
significantly better than the football-only incumbent ({INCUMBENT_HOLDOUT_LL}).
The market is the true performance ceiling for pregame NFL prediction.
The incumbent is a purely pregame, market-free benchmark.

---

*Source: [`leaderboard.csv`]({_gh(str(LEADERBOARD_PATH))}) and [`benchmark_history.md`]({_gh(str(HISTORY_PATH))})*
"""


def build_predictions():
    return f"""# Predictions & Artifacts

{_navbar("predictions")}

## Prediction Schema

Each prediction CSV contains:

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `season`, `week`, `gameday` | Game timing |
| `away_team`, `home_team` | Teams |
| `incumbent_home_win_prob` | Predicted home win probability |
| `predicted_winner` | Home or away team based on prob |
| `confidence_bucket` | Probability range bucket |
| `model_version` | Incumbent version at prediction time |
| `feature_set` | Features used |
| `calibration_method` | Calibration type |
| `qb_source` | QB data source (oracle / live_pregame / auto_qb) |
| `caution_qb_change` | QB changed from prior game |
| `caution_early_season` | Weeks 1\u20134 |

## Confidence Buckets

| Bucket | Range | Description |
|--------|-------|-------------|
| 50-55 | 0.50\u20130.55 | Near coin flip |
| 55-60 | 0.55\u20130.60 | Slight favorite |
| 60-65 | 0.60\u20130.65 | Moderate favorite |
| 65-70 | 0.65\u20130.70 | Solid favorite |
| 70-80 | 0.70\u20130.80 | Strong favorite |
| 80+ | 0.80+ | Heavy favorite (rare in NFL) |

## 2025 Holdout Performance

| Metric | Value |
|--------|-------|
| Games | 276 |
| Log loss | **0.6200** |
| Brier | 0.2157 |
| AUC | 0.7098 |
| Accuracy | 0.6630 |

## 2026 Predictions

See [2026 Schedule & Predictions](2026-schedule) for the current season's
predictions. Updated each week via:

```
sportslab predict-week --season 2026 --week <N> --auto-qb
```

---

*Full registry: [`leaderboard.csv`]({_gh(str(LEADERBOARD_PATH))})*
"""


def build_model_card():
    return f"""# Model Card: v{INCUMBENT_VERSION}

{_navbar("model-card")}

## Model Identity

| Attribute | Value |
|-----------|-------|
| **Name** | {INCUMBENT_NAME} |
| **Version** | v{INCUMBENT_VERSION} |
| **Date** | 2026-06-29 |
| **Type** | Elo ratings + Platt logistic + frozen QB overlay |

## Architecture (Two-Layer)

```
Layer 0: Elo ratings (K=36, HFA=40, reg=0.1, decay=32, qb_bonus=0.2)
         \u2192 elo_prob (raw Elo home win probability)

Layer 1: Platt logistic on [elo_prob, qb_changed, rolling_mov_3]
         \u2192 base probability

Layer 2: Frozen QB overlay (logit-space additive, gated)
         Gate: qb_changed OR career_starts < 17 (either side)
         Adjustment: gamma=1.0 * clip(qb_adj, \u00b140) * ln(10)/400
         Applied only when gate is active
         \u2192 final probability
```

## Performance

| Metric | Value |
|--------|-------|
| Avg validation log loss | {INCUMBENT_VAL_LL} |
| 2025 holdout log loss | **{INCUMBENT_HOLDOUT_LL}** |
| 2025 holdout Brier | 0.2157 |
| 2025 holdout AUC | 0.7098 |
| 2025 holdout Accuracy | 0.6630 |

## Feature Set

Exactly 5 features enter the Platt logistic:
1. `elo_prob` \u2014 raw Elo home win probability
2. `home_qb_changed` \u2014 home QB changed from prior game
3. `away_qb_changed` \u2014 away QB changed from prior game
4. `home_rolling_mov_3` \u2014 home margin of victory, last 3 games
5. `away_rolling_mov_3` \u2014 away margin of victory, last 3 games

The frozen QB overlay uses additional computed signals:
- `home_qb_adj`, `away_qb_adj` \u2014 Bayesian-shrunken per-QB Elo ratings
- `home_qb_team_starts_pre`, `away_qb_team_starts_pre` \u2014 career starts
- `home_qb_changed`, `away_qb_changed` \u2014 QB change flags

## Leakage Prevention

- Elo computed chronologically: emit probability before updating rating
- Rolling MOV excludes current game from its window
- QB change detected from previous game only (no future knowledge)
- Preseason regression resets at season boundaries
- Decay halflife = 32 games, applied chronologically
- Ties excluded from Platt training, treated as 0.5 in Elo updates
- 2025 holdout never accessed during model selection

## Training Data

- Seasons: 2021\u20132024 (1,356 eligible games)
- Rolling-origin 3-fold validation
- 2025 season: locked holdout (276 games)

## Known Limitations

1. **Cold start**: Elo decayed after offseason; rolling_mov_3 = 0 in week 1
2. **QB change oracle**: Uses final actual starter (not pregame-announced)
3. **Market gap**: Market benchmark 0.6090 vs incumbent 0.6200
4. **Small sample**: ~1,000 training games limits feature complexity
5. **No roster features**: Injuries, depth chart changes only via Elo decay
"""


def build_experiments():
    rows = _read_leaderboard()
    promoted_rows = [r for r in rows if r["decision"] in ("promoted", "superseded")]
    rejected_rows = [r for r in rows if r["decision"] == "rejected"]
    diagnostic_rows = [r for r in rows if r["decision"] == "diagnostic"]

    def _table(rows_subset, label):
        if not rows_subset:
            return f"\n### {label}\n\nNone.\n\n"
        t = f"\n### {label}\n\n| Experiment | Decision | Val LL | Holdout LL |\n|-----------|----------|--------|------------|\n"
        for r in rows_subset:
            t += f"| {r['experiment']} | {r['decision']} | {_fmt(r.get('val_ll'))} | {_fmt(r.get('holdout_ll'))} |\n"
        t += "\n"
        return t

    out = f"""# Experiments

{_navbar("experiments")}

All {len(rows)} experiments conducted during feature research (2026-06).

"""
    out += _table(promoted_rows, "Promoted / Superseded")
    out += _table(rejected_rows, "Rejected")
    out += _table(diagnostic_rows, "Diagnostic Only")

    out += "\n---\n*Full details: [`leaderboard.csv`]({})*".format(_gh(str(LEADERBOARD_PATH)))
    return out


def build_backtests():
    return f"""# Backtest Reports

{_navbar("backtests")}

The backtest evaluates the incumbent model
(**{INCUMBENT_NAME}**) across each NFL season.
Seasons 2022\u20132024 are in-training diagnostics (part of 2021\u20132024 training data).
Season 2025 is a locked holdout.

## Key Metrics (2025 Holdout)

| Metric | Value |
|--------|-------|
| Games | 276 |
| Log loss | **0.6200** |
| Brier | 0.2157 |
| AUC | 0.7098 |
| Accuracy | 0.6630 |

## Season-by-Season

### 2025 Season (Locked Holdout)

| Artifact | Description |
|----------|-------------|
| [Full Report]({_gh('reports/backtests/2025_backtest_report.md')}) | Comprehensive Markdown report |
| [Weekly Summary]({_gh('reports/backtests/2025_weekly_summary.csv')}) | Week-by-week breakdown |
| [Team Summary]({_gh('reports/backtests/2025_team_summary.csv')}) | Per-team diagnostics |
| [Calibration Buckets]({_gh('reports/backtests/2025_calibration_buckets.csv')}) | Confidence bucket analysis |
| [Extreme Games]({_gh('reports/backtests/2025_extreme_games.csv')}) | Best/worst predictions |
| [Subgroup Summary]({_gh('reports/backtests/2025_subgroup_summary.csv')}) | Game-context breakdown |

### 2024 Season

| Artifact | Description |
|----------|-------------|
| [Full Report]({_gh('reports/backtests/2024_backtest_report.md')}) | Comprehensive Markdown report |
| [Weekly Summary]({_gh('reports/backtests/2024_weekly_summary.csv')}) | Week-by-week breakdown |
| [Team Summary]({_gh('reports/backtests/2024_team_summary.csv')}) | Per-team diagnostics |
| [Calibration Buckets]({_gh('reports/backtests/2024_calibration_buckets.csv')}) | Confidence bucket analysis |
| [Extreme Games]({_gh('reports/backtests/2024_extreme_games.csv')}) | Best/worst predictions |
| [Subgroup Summary]({_gh('reports/backtests/2024_subgroup_summary.csv')}) | Game-context breakdown |

### 2023 Season

| Artifact | Description |
|----------|-------------|
| [Full Report]({_gh('reports/backtests/2023_backtest_report.md')}) | Comprehensive Markdown report |
| [Weekly Summary]({_gh('reports/backtests/2023_weekly_summary.csv')}) | Week-by-week breakdown |
| [Team Summary]({_gh('reports/backtests/2023_team_summary.csv')}) | Per-team diagnostics |
| [Calibration Buckets]({_gh('reports/backtests/2023_calibration_buckets.csv')}) | Confidence bucket analysis |
| [Extreme Games]({_gh('reports/backtests/2023_extreme_games.csv')}) | Best/worst predictions |
| [Subgroup Summary]({_gh('reports/backtests/2023_subgroup_summary.csv')}) | Game-context breakdown |

### 2022 Season

| Artifact | Description |
|----------|-------------|
| [Full Report]({_gh('reports/backtests/2022_backtest_report.md')}) | Comprehensive Markdown report |
| [Weekly Summary]({_gh('reports/backtests/2022_weekly_summary.csv')}) | Week-by-week breakdown |
| [Team Summary]({_gh('reports/backtests/2022_team_summary.csv')}) | Per-team diagnostics |
| [Calibration Buckets]({_gh('reports/backtests/2022_calibration_buckets.csv')}) | Confidence bucket analysis |
| [Extreme Games]({_gh('reports/backtests/2022_extreme_games.csv')}) | Best/worst predictions |
| [Subgroup Summary]({_gh('reports/backtests/2022_subgroup_summary.csv')}) | Game-context breakdown |

## Research Caveats

- The model was trained on 2021\u20132024 data only. 2025 was a locked holdout.
- Seasons 2022\u20132024 were part of the training window and are diagnostic only.
- Market data is excluded from the model. Market benchmark: 0.6090 holdout LL.
- This is a probabilistic prediction benchmark, not a gambling product.
"""


# ── Page registry ──

PAGES = {
    "index.md": build_index,
    "2026-schedule.md": build_schedule_2026,
    "benchmarks.md": build_benchmarks,
    "predictions.md": build_predictions,
    "model-card.md": build_model_card,
    "experiments.md": build_experiments,
    "backtests.md": build_backtests,
}


def build_all_docs(docs_dir: str | Path = DOCS) -> dict[str, str]:
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for filename, builder_fn in PAGES.items():
        content = builder_fn()
        path = docs_dir / filename
        path.write_text(content)
        results[filename] = str(path)
        print(f"  Generated: {path}")
    return results
