# Sports ML Lab - Project Foundation

This repository sets up the foundational structure for an NFL prediction research lab. The project follows reproducible ML research practices with a focus on explainability, proper data usage, and preventing leakage.

## Environment Configuration

- Local MacBook/OpenCode environment
- Remote Ollama models hosted on System76 laptop via LAN
- All configurations and network settings preserved as-is

## Research Philosophy

The project follows strict principles to ensure research validity:
1. Every feature must be explainable and pregame-safe
2. No future data in features
3. No modification of raw historical data
4. Experiments must report log loss, Brier score, accuracy, calibration notes, and leakage risk
5. Never promote models based on ROI alone

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
└── Makefile                        # Build automation commands
```

## Getting Started

1. Use `make install` to set up the environment
2. Use `make test` to run tests
3. Use `make lint` to check code style
4. Use `make format` to auto-format code
5. Use `make mlflow` to start MLflow tracking