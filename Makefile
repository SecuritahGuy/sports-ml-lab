.PHONY: install test lint format check clean ingest build-features \
        predict-incumbent predict-future weekly-report simulate \
        backtest-2025 audit dashboard no-qb-baseline qb-continuity \
qb-gated-experience qb-depth-experiment turnover-experiment situational-micro \
predict-week grade-week season-report prediction-audit rehearsal-season prediction-index publish-predictions \
data-audit preseason-fire-drill live-preflight

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

prediction-index:
	sportslab build-prediction-index

publish-predictions:
	sportslab build-prediction-index
	@echo "  Audit reports are published by 'sportslab prediction-audit --season <YEAR>' (live mode)."
	@echo "  Run: make prediction-audit SEASON=<YEAR> for each tracked season."

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

# ── Development ──
.PHONY: dev
dev: install check
