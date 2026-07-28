.PHONY: install test lint format check clean ingest build-features \
        predict-incumbent predict-future weekly-report simulate \
        backtest-2025 audit dashboard no-qb-baseline qb-continuity \
qb-gated-experience qb-depth-experiment turnover-experiment situational-micro \
predict-week grade-week season-report prediction-audit rehearsal-season prediction-index publish-predictions \
data-audit preseason-fire-drill live-preflight \
build-qb-adjustments qb-adjusted-elo roster-strength regularized-logistic \
qb-lift model-trust ralph6 team-site team-site-serve monitoring-report score-margin \
list-vintages compare-vintages pi-ratings pi-ratings-compare pi-statspace

# ── Install ──
install:
	pip install -e ".[dev]"

# ── Quality ──
test:
	python -m pytest --tb=short -x -q tests/

test-all:
	python -m pytest --tb=short -q tests/

test-v:
	python -m pytest --tb=short -v tests/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

check: lint test

# ── Data ──
ingest:
	sportslab ingest-nfl

build-features:
	sportslab build-features

# ── Predictions ──
predict-incumbent:
	sportslab predict-incumbent

predict-future:
	sportslab predict-future

weekly-report:
	sportslab weekly-report

no-qb-baseline:
	sportslab no-qb-baseline

qb-ablation:
	sportslab qb-ablation

qb-continuity:
	sportslab qb-continuity

qb-gated-experience:
	sportslab qb-gated-experience

qb-depth-experiment:
	sportslab qb-depth-experiment

turnover-experiment:
	sportslab turnover-experiment

situational-micro:
	sportslab situational-micro

simulate-oracle:
	sportslab simulate-2025

simulate-live:
	sportslab simulate-2025 --qb-input qb_input_2025.csv

simulate-compare:
	sportslab simulate-2025
	sportslab simulate-2025 --qb-input qb_input_2025.csv --output reports/simulations/simulate_2025_results_live.csv --report reports/simulations/simulate_2025_live_report.md
	python -c "
	import pandas as pd
	o = pd.read_csv('reports/simulations/simulate_2025_results.csv')
	l = pd.read_csv('reports/simulations/simulate_2025_results_live.csv')
	from sklearn.metrics import log_loss
	print(f'Oracle log loss: {log_loss(o.home_win_actual.astype(int), o.incumbent_home_win_prob):.4f}')
	print(f'Live-safe log loss: {log_loss(l.home_win_actual.astype(int), l.incumbent_home_win_prob):.4f}')
	"

# ── Validation ──
backtest-2025:
	sportslab backtest-2025

audit:
	sportslab audit-artifacts

dashboard:
	sportslab build-dashboard

# ── Clean ──
clean:
	rm -rf .pytest_cache/ __pycache__/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Weekly Operations ──
predict-week:
	sportslab predict-week --season $(SEASON) --week $(WEEK) --mode $(MODE)

predict-week-oracle:
	sportslab predict-week --season $(SEASON) --week $(WEEK) --mode dry_run

grade-week:
	sportslab grade-week --season $(SEASON) --week $(WEEK) --mode $(MODE)

season-report:
	sportslab season-report --season $(SEASON)

prediction-audit:
	sportslab prediction-audit --season $(SEASON)

rehearsal-season:
	sportslab rehearsal-season --season $(SEASON)

rehearsal-2025:
	sportslab rehearsal-season --season 2025

weekly-qb-audit:
	sportslab weekly-qb-audit --season $(SEASON) --week $(WEEK) --output reports/predictions/weekly_qb_audit_$(SEASON)_w$(WEEK).csv

prediction-index:
	sportslab build-prediction-index

publish-predictions:
	sportslab publish-predictions

publish-predictions-dry-run:
	sportslab publish-predictions --dry-run

# ── Data Audit ──
data-audit:
	sportslab data-audit

data-audit-seasons:
	sportslab data-audit --seasons $(SEASONS)

# ── Live Preflight ──
live-preflight:
	sportslab live-preflight

live-preflight-qb:
	sportslab live-preflight --qb-input $(QB_INPUT) --seasons $(SEASONS)

# ── Preseason Fire Drill ──
preseason-fire-drill: build-features data-audit predict-week-oracle prediction-audit
	@echo ""
	@echo "=== Preseason Fire Drill Complete ==="
	@echo "  Ingest verified, features built, data healthy."
	@echo "  Dry-run predictions created, audit generated."
	@echo "  Ready for live season."
	@echo "  Next: make predict-week SEASON=2026 WEEK=1 MODE=live QB_INPUT=qb.csv"

# ── QB-Adjusted Elo / Roster Strength ──
build-qb-adjustments:
	sportslab build-qb-adjustments

qb-adjusted-elo:
	sportslab qb-adjusted-elo

gated-qb-elo:
	sportslab gated-qb-elo

frozen-qb-overlay:
	sportslab frozen-qb-overlay

frozen-qb-overlay-foldsafe:
	sportslab frozen-qb-overlay-foldsafe

learned-overlay:
	sportslab learned-overlay

gradient-boosting:
	sportslab gradient-boosting

gam-logistic:
	sportslab gam-logistic

regularized-logistic:
	sportslab regularized-logistic

regularized-logistic-meta:
	sportslab regularized-logistic-meta

calibration-audit:
	sportslab calibration-audit

calibration-remediation:
	sportslab calibration-remediation

roster-overlay:
	sportslab roster-overlay

qb-roster-interaction:
	sportslab qb-roster-interaction

expanded-elo-spine:
	sportslab expanded-elo-spine

elo-ensemble:
	sportslab elo-ensemble

expanded-seasons:
	sportslab expanded-seasons

statspace-fdr:
	sportslab statspace-fdr

statspace-doba:
	sportslab statspace-doba

statspace-chaos:
	sportslab statspace-chaos

statspace-coward-tax:
	sportslab statspace-coward-tax

statspace-qb-lift:
	sportslab statspace-qb-lift

team-profiles:
	sportslab team-profiles

statspace-plots:
	sportslab statspace-plots

statspace-backtest:
	sportslab statspace-backtest

roster-strength:
	sportslab roster-strength

qb-lift:
	sportslab qb-lift

dynamic-elo:
	sportslab dynamic-elo

kalman-elo:
	sportslab kalman-elo

elo-ensemble:
	sportslab elo-ensemble

retest-rejected-features:
	sportslab retest-rejected-features

model-trust:
	sportslab model-trust

monitoring-report:
	sportslab monitoring-report --season $(or $(SEASON),2026) --week $(or $(WEEK),1)

ralph6:
	sportslab ralph6

preseason-elo-prior:
	sportslab preseason-elo-prior

score-margin:
	sportslab score-margin-experiment

team-site:
	sportslab build-team-site

team-site-serve: team-site

# ── Vintages ──
list-vintages:
	sportslab list-vintages $(SEASON_ARG) $(WEEK_ARG)

compare-vintages:
	sportslab compare-vintages $(SEASON_ARG) $(WEEK_ARG) $(if $(OUTPUT),--output $(OUTPUT),)

pi-ratings-compare:
	sportslab pi-ratings-compare

pi-statspace:
	sportslab pi-statspace

# Refresh: ingest scores → rebuild features → repredict → rebuild site
# Usage: make refresh WEEK=1  (grade week 1, predict weeks 2-18)
#        make refresh          (full refresh, all weeks)
refresh:
	sportslab refresh-week $(if $(WEEK),--week $(WEEK),)

# ── Development ──
.PHONY: dev
dev: install check
