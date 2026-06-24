VENV = .venv
PYTHON = $(VENV)/bin/python3
SPORTSLAB = $(VENV)/bin/sportslab
RUFF = $(VENV)/bin/ruff
PIP = $(VENV)/bin/pip

.PHONY: install install-dev test lint format clean mlflow ingest-nfl build-features train-baseline train-baseline-team-strength elo-tuning rolling-origin-elo schedule-features margin-aware-elo qb-features weather-features expressive-models market-baseline residual-diagnostics epa-features confidence-calibration market-benchmark decayed-elo team-hfa season-regression residual-blending coach-season-regression autogluon injury-features optuna-elo-search qb-injury-flag glicko qb-market-delta feature-selection combined-features optuna-feature-selection predict-incumbent rolling-mov-sensitivity weekly-report comprehensive-efficiency audit-artifacts venv

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -e .

install-dev: venv
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

mlflow:
	mlflow ui --port 5000

# This project only supports seasons 2021–current.
# Add the current season (e.g. 2026) once week 1 data is available.
ingest-nfl:
	$(SPORTSLAB) ingest-nfl 2021 2022 2023 2024 2025

build-features:
	$(SPORTSLAB) build-features

train-baseline:
	$(SPORTSLAB) train-baseline --feature-set baseline

train-baseline-team-strength:
	$(SPORTSLAB) train-baseline --feature-set team_strength

elo-tuning:
	$(SPORTSLAB) elo-tuning

rolling-origin-elo:
	$(SPORTSLAB) rolling-origin

schedule-features:
	$(SPORTSLAB) schedule-features

margin-aware-elo:
	$(SPORTSLAB) margin-aware-elo

qb-features:
	$(SPORTSLAB) qb-features

weather-features:
	$(SPORTSLAB) weather-features

expressive-models:
	$(SPORTSLAB) expressive-models

market-baseline:
	$(SPORTSLAB) market-baseline

residual-diagnostics:
	$(SPORTSLAB) residual-diagnostics

epa-features:
	$(SPORTSLAB) epa-features

confidence-calibration:
	$(SPORTSLAB) confidence-calibration

market-benchmark:
	$(SPORTSLAB) market-benchmark

decayed-elo:
	$(SPORTSLAB) decayed-elo

team-hfa:
	$(SPORTSLAB) team-hfa

season-regression:
	$(SPORTSLAB) season-regression

residual-blending:
	$(SPORTSLAB) residual-blending

coach-season-regression:
	$(SPORTSLAB) coach-season-regression

autogluon:
	$(SPORTSLAB) autogluon

injury-features:
	$(SPORTSLAB) injury-features

optuna-elo-search:
	$(SPORTSLAB) optuna-search

qb-injury-flag:
	$(SPORTSLAB) qb-injury

glicko:
	$(SPORTSLAB) glicko

qb-market-delta:
	$(SPORTSLAB) qb-market-delta

feature-selection:
	$(SPORTSLAB) feature-selection

combined-features:
	$(SPORTSLAB) combined-features

optuna-feature-selection:
	$(SPORTSLAB) optuna-feature-selection

predict-incumbent:
	$(SPORTSLAB) predict-incumbent

rolling-mov-sensitivity:
	$(SPORTSLAB) rolling-mov-sensitivity

weekly-report:
	$(SPORTSLAB) weekly-report

comprehensive-efficiency:
	$(SPORTSLAB) comprehensive-efficiency

audit-artifacts:
	$(SPORTSLAB) audit-artifacts

build-dashboard:
	$(SPORTSLAB) build-dashboard
