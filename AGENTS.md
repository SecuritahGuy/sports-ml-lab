# Sports ML Lab — Agent Rules

## Project Overview

This repository sets up the foundational structure for an NFL prediction research lab. The project follows reproducible ML research practices with a focus on explainability, proper data usage, and preventing leakage.

## Operating Context

- OpenCode runs on the MacBook.
- Local LLM inference is remote on the System76 through internal LAN Ollama.
- Do not modify Ollama, OpenCode provider config, model config, network config, SSH config, firewall rules, systemd services, or remote model services unless explicitly asked.
- This project is NFL-first. Do not add NBA, odds, betting automation, scraping, or web UI unless explicitly asked.

## NFL Data Scope

- **Allowed seasons: 2021–current only.**
- Do not ingest, train, test, backtest, benchmark, or tune on seasons earlier than 2021.
- If a command, config, or test tries to use a season before 2021, fail clearly.
- Future walk-forward tests must start with 2021 as the earliest training season.

## Environment Configuration

- Local MacBook/OpenCode environment
- Remote Ollama models hosted on System76 laptop via LAN
- All configurations and network settings preserved as-is

## Autonomy Rules

You are allowed and expected to run safe inspection commands before coding.

Do not ask the user questions that can be answered by:
- reading files in the repo
- running command help
- running tests
- inspecting installed Python packages
- inspecting function signatures
- reading local errors
- checking git status

If the prompt asks you to inspect a Python package, you must inspect the installed package locally with Python introspection before writing code that depends on that package.

Do not write implementation code based on guessed APIs.

Always follow this order:
1. Inspect local repo/package state.
2. Summarize facts discovered.
3. Implement the smallest scoped change.
4. Run formatting/lint/tests.
5. Fix failures.
6. Summarize files changed, commands run, what works, what remains TODO.

Do not stop after creating one file if the prompt contains a task list.

## Safe Commands Allowed Without Asking

You may run these without asking:
- pwd
- ls
- find
- cat for non-secret project files
- head / tail for display
- grep / rg
- wc
- git status
- git diff
- git diff --stat
- python -c introspection commands
- python heredoc introspection commands
- python -m pip show / pip freeze
- sportslab --help
- pytest / python -m pytest
- ruff check
- make test / make lint / make format
- python -m compileall src

## Code Edits Allowed Without Asking

You may create or edit files inside:
- src/
- tests/
- configs/
- scripts/
- reports/experiments/
- README.md
- AGENTS.md
- Makefile
- pyproject.toml
- .gitignore
- .env.example
- docs/opencode-commands/ (plain-text command references)

## Ask Before These Actions

Ask before:
- git commit / git push / git pull / git checkout / git merge
- installing new packages not already listed in pyproject.toml
- running commands that download large datasets
- make ingest-nfl (requires network)
- long backtests / Optuna sweeps / AutoGluon runs
- remote SSH scripts
- modifying generated data files
- modifying files outside this repo

## Denied Actions

Never do these unless I explicitly override:
- read .env or secret files
- print environment variables that may contain secrets
- modify SSH keys or SSH config
- modify Ollama, OpenCode provider, network, firewall, systemd, or model service configuration
- use sudo
- run rm -rf (destructive)
- run curl | sh / wget | sh
- destructive git reset / git clean
- delete raw data
- manually edit files under data/raw/
- expose services on 0.0.0.0
- scrape random websites
- use WebFetch when local package inspection is the correct source of truth

## Research Philosophy

The project follows strict principles to ensure research validity:
1. Every feature must be explainable and pregame-safe
2. No future data in features
3. No modification of raw historical data
4. Experiments must report log loss, Brier score, accuracy, calibration notes, and leakage risk
5. Never promote models based on ROI alone

## Sports ML Rules

- Predict probabilities, not vibes.
- Optimize first for log loss, Brier score, calibration, and leakage prevention.
- Accuracy is secondary.
- ROI is not a primary model-promotion metric.
- Do not use future data in features.
- Do not include final score, result, winner, or target columns in model features.
- Every feature must be explainable and pregame-safe.
- Every experiment report must include leakage risk.

## Project Structure

```
sports-ml-lab/
├── AGENTS.md                       # Research agent rules and guidelines
├── README.md                       # Project overview
├── Makefile                        # Build automation commands
├── pyproject.toml                  # Project dependencies and configuration
├── .gitignore                      # Git ignore rules
├── .env.example                    # Environment variable template
├── configs/
│   └── nfl/                        # NFL-specific configurations
├── data/
│   ├── raw/                        # Raw data files
│   ├── nfl/                        # NFL-specific data
│   ├── interim/                    # Intermediate processed data
│   ├── processed/                  # Final processed data
│   └── features/                   # Feature data
├── db/                             # Database files (empty, .gitkeep)
├── src/
│   └── sportslab/                  # Main source code
├── scripts/                        # Utility scripts
├── tests/                          # Unit and integration tests
├── reports/
│   ├── experiments/                # Experiment results
│   └── daily/                      # Daily logs
└── docs/
    └── opencode-commands/          # Command reference docs
```

## Getting Started

1.  Use `make install` to set up the environment
2.  Use `make test` to run tests
3.  Use `make lint` to check code style
4.  Use `make format` to auto-format code
5.  Use `make mlflow` to start MLflow tracking
6.  Use `make ingest-nfl` to download NFL schedule data (requires internet)
7.  Use `sportslab ingest-nfl --seasons 2021 2022 2023 2024 2025` for custom seasons
